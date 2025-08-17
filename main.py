#!/usr/bin/env python3
"""
Main Bootstrap Orchestrator for Alexa (Local Voice Assistant)

Ultra-minimal entrypoint that:
1. Initializes environment and config
2. (Optionally) starts Logger first (kept minimal here)
3. Spawns Loader (process manager) which handles the rest
4. Forwards signals and requests graceful shutdown on Ctrl+C

State flow: INIT → STARTING_LOADER → RUN → SHUTDOWN

Update (graceful Ctrl+C):
- On SIGINT/SIGTERM, we first try an HTTP "graceful shutdown" request to the Loader
  so it can stop all child services cleanly. If that fails or times out, we fall
  back to terminating the Loader process.
"""

import sys
import os
import signal
import subprocess
import time
import json
import configparser
from pathlib import Path
from typing import Optional

import requests


class MainOrchestrator:
    def __init__(self):
        self.config: Optional[configparser.ConfigParser] = None
        self.logger_process: Optional[subprocess.Popen] = None
        self.loader_process: Optional[subprocess.Popen] = None
        self.logger_url = "http://127.0.0.1:5000"
        self.shutdown_requested = False

    # -----------------------------
    # Config helpers
    # -----------------------------
    def load_config(self) -> bool:
        """Load configuration from config/config.ini"""
        try:
            config_path = Path("config/config.ini")
            if not config_path.exists():
                print("ERROR: config/config.ini not found")
                return False

            self.config = configparser.ConfigParser()
            self.config.read(config_path)

            # Verify [main] section exists
            if 'main' not in self.config:
                print("ERROR: [main] section missing in config.ini")
                return False

            return True
        except Exception as e:
            print(f"ERROR: Failed to load config: {e}")
            return False

    def get_main_config(self, key: str, default: str = "") -> str:
        """Get value from [main] section of config"""
        return self.config.get('main', key, fallback=default)

    def get_loader_base_url(self) -> str:
        """
        Determine Loader's base URL.

        Priority:
        1) [loader] base_url
        2) http://127.0.0.1:<port> where port = [loader] port (fallback 5002)
        """
        base_url = None
        if self.config and self.config.has_section('loader'):
            base_url = self.config.get('loader', 'base_url', fallback=None)
            if base_url:
                return base_url.rstrip('/')
            port = self.config.get('loader', 'port', fallback='5002')
            try:
                port_int = int(port)
            except ValueError:
                port_int = 5002
            return f"http://127.0.0.1:{port_int}"
        # No [loader] section → sensible default:
        return "http://127.0.0.1:5002"

    # -----------------------------
    # Signal handling
    # -----------------------------
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.shutdown_requested = True
            self.shutdown()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    # -----------------------------
    # Logging (to logger_service)
    # -----------------------------
    def log_message(self, level: str, message: str, event: Optional[str] = None):
        """Send log message to logger service; fallback to stdout if unavailable"""
        try:
            payload = {"svc": "MAIN", "level": level, "message": message}
            if event:
                payload["event"] = event
            requests.post(f"{self.logger_url}/log", json=payload, timeout=1.5)
        except Exception:
            print(f"MAIN      {level.upper():<6}= {message}")

    # -----------------------------
    # Process spawns
    # -----------------------------
    def start_loader(self) -> bool:
        """Start loader service (process manager)"""
        try:
            self.log_message("info", "Starting Loader service")
            self.loader_process = subprocess.Popen(
                [sys.executable, "-m", "services.loader_service"],
            )
            return True
        except Exception as e:
            self.log_message("error", f"Failed to start loader: {e}")
            return False

    # -----------------------------
    # Graceful shutdown helpers
    # -----------------------------
    def request_loader_graceful_shutdown(self, timeout_sec: float = 20.0) -> bool:
        """
        Ask Loader (via HTTP) to stop all managed services cleanly.
        Tries several common endpoints for compatibility.

        Returns True if an endpoint responded with 200 OK.
        """
        base = self.get_loader_base_url()
        candidate_paths = [
            "/shutdown",           # preferred
            "/stop_all",           # alternate
            "/control/shutdown",   # legacy
            "/admin/shutdown",     # fallback
        ]
        payload = {"reason": "main_received_signal", "source": "MAIN"}
        deadline = time.time() + timeout_sec

        for path in candidate_paths:
            url = f"{base}{path}"
            try:
                resp = requests.post(url, json=payload, timeout=3.0)
                if resp.status_code == 200:
                    self.log_message("info", f"Requested graceful shutdown: {url}")
                    break
            except requests.RequestException:
                continue
        else:
            # None of the endpoints worked
            self.log_message("warning", "No graceful shutdown endpoint responded")
            return False

        # If request accepted, wait until Loader exits or timeout occurs
        if not self.loader_process:
            return True

        while time.time() < deadline:
            if self.loader_process.poll() is not None:
                return True
            time.sleep(0.2)

        self.log_message("warning", "Loader didn't exit before timeout")
        return False

    # -----------------------------
    # Wait lifecycle
    # -----------------------------
    def wait_for_loader(self) -> int:
        """Wait for loader process; break early if shutdown was requested"""
        if not self.loader_process:
            return 1

        try:
            while self.loader_process.poll() is None:
                if self.shutdown_requested:
                    break
                time.sleep(0.1)
            return self.loader_process.returncode or 0
        except Exception as e:
            self.log_message("error", f"Error waiting for loader: {e}")
            return 1

    # -----------------------------
    # Shutdown sequence
    # -----------------------------
    def shutdown(self):
        """
        Graceful shutdown:
        1) Try HTTP graceful shutdown so Loader stops its child services first.
        2) If that fails, send SIGINT/SIGTERM to Loader.
        3) As a last resort, kill the Loader.
        """
        try:
            # 1) Preferred: tell Loader to stop all services
            ok = False
            if self.loader_process and self.loader_process.poll() is None:
                ok = self.request_loader_graceful_shutdown(timeout_sec=25.0)

            # 2) Fallback: signal the Loader process directly
            if not ok and self.loader_process and self.loader_process.poll() is None:
                self.log_message("info", "Sending SIGINT to Loader (fallback)")
                try:
                    self.loader_process.send_signal(signal.SIGINT)
                    self.loader_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.log_message("warning", "SIGINT timeout, sending SIGTERM")
                    try:
                        self.loader_process.terminate()
                        self.loader_process.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        self.log_message("warning", "SIGTERM timeout, killing Loader")
                        self.loader_process.kill()

            # (Optional) You can add logger shutdown here if logger runs as a service
            # For now we just log the final message; logger may be stopped by Loader.
            self.log_message("info", "System shutdown complete")

        except Exception as e:
            print(f"ERROR during shutdown: {e}")

    # -----------------------------
    # Run
    # -----------------------------
    def run(self) -> int:
        """Main execution flow"""
        try:
            # INIT: Load configuration
            if not self.load_config():
                return 1

            # Setup signal handlers
            self.setup_signal_handlers()

            # STARTING_LOADER: Start loader (process manager)
            if not self.start_loader():
                self.shutdown()
                return 1

            # RUN: Wait for loader and forward signals
            exit_code = self.wait_for_loader()

            # SHUTDOWN: Clean shutdown
            self.shutdown()
            return exit_code

        except KeyboardInterrupt:
            # Redundant because of signal handler, but kept for completeness
            self.log_message("info", "Interrupted by user")
            self.shutdown()
            return 0
        except Exception as e:
            print(f"FATAL ERROR: {e}")
            self.shutdown()
            return 1


def main():
    """Entry point"""
    orchestrator = MainOrchestrator()
    exit_code = orchestrator.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
