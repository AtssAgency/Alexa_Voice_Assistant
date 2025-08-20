#!/usr/bin/env python3
"""
Loader Service - Process Manager for Voice Assistant Services

CLI/daemon process that:
- Spawns all microservices in health-gated order (LOGGER → KWD → STT → LLM → TTS)
- Health-gates each service before proceeding to the next
- Restarts crashed services with exponential backoff
- Reports startup times, VRAM snapshots, and system metrics to Logger
- Triggers KWD greeting after all services are ready

State: INIT → STARTING_LOGGER → START_SEQUENCE → HEALTH_GATE → SYSTEM_READY → RUN
"""

import os
import sys
import signal
import subprocess
import time
import psutil
import configparser
import asyncio
import random
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum

# Import config_loader when running as a service module
try:
    from .config_loader import app_config
    HAVE_CONFIG_LOADER = True
except ImportError:
    # When running standalone, config_loader is not available
    HAVE_CONFIG_LOADER = False
    app_config = None


class LoaderState(Enum):
    INIT = "INIT"
    STARTING_LOGGER = "STARTING_LOGGER"
    START_SEQUENCE = "START_SEQUENCE"
    HEALTH_GATE = "HEALTH_GATE"
    RECOVER = "RECOVER"
    SYSTEM_READY = "SYSTEM_READY"
    RUN = "RUN"
    STOPPING = "STOPPING"
    FAILED = "FAILED"


@dataclass
class ServiceConfig:
    name: str
    cmd: str
    port: int
    health_url: Optional[str] = None
    warmup_url: Optional[str] = None


@dataclass
class ServiceProcess:
    config: ServiceConfig
    process: Optional[subprocess.Popen] = None
    pid: Optional[int] = None
    start_time: Optional[float] = None
    ready_time: Optional[float] = None
    restart_count: int = 0
    last_restart_time: Optional[float] = None


class LoaderService:
    def __init__(self):
        self.state = LoaderState.INIT
        
        # Try to use config_loader if available, otherwise use defaults
        if HAVE_CONFIG_LOADER and app_config:
            # --- CORRECTED ACCESS ---
            # Access nested models correctly via app_config.loader
            loader_config = app_config.loader
            paths_config = app_config.loader.paths
            
            # Construct logger_url from the logger's own config
            self.logger_url = f"http://127.0.0.1:{app_config.logger.port}"
            
            # Configuration from config_loader
            self.startup_order = [s.strip() for s in loader_config.startup_order.split(',')]
            self.health_timeout_s = loader_config.health_timeout_s
            self.health_interval_ms = loader_config.health_interval_ms
            self.restart_backoff_ms = [int(x.strip()) for x in loader_config.restart_backoff_ms.split(',')]
            self.crash_loop_threshold = loader_config.crash_loop_threshold
            self.port_hygiene = loader_config.port_hygiene
            self.vram_probe_cmd = loader_config.vram_probe_cmd
            self.warmup_llm = loader_config.warmup_llm
            self.greeting_on_ready = loader_config.greeting_on_ready
            self.python = paths_config.python if paths_config.python else sys.executable
        else:
            # Fallback defaults when config_loader is not available
            self.logger_url = "http://127.0.0.1:5000"
            self.startup_order = []
            self.health_timeout_s = 30
            self.health_interval_ms = 200
            self.restart_backoff_ms = [250, 500, 1000, 2000]
            self.crash_loop_threshold = 3
            self.port_hygiene = True
            self.vram_probe_cmd = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
            self.warmup_llm = True
            self.greeting_on_ready = True
            self.python = sys.executable
        
        # Service management
        self.services: Dict[str, ServiceProcess] = {}
        self.service_configs: Dict[str, ServiceConfig] = {}
        self.current_service_index = 0
        self.shutdown_requested = False
        
        # System timing
        self.system_start_time = None
        self.system_ready_time = None
        
        # Selective service options
        self.execution_plan: List[str] = []
        self.skip_warmup = False
        self.skip_greeting = False
        self.dry_run = False
    
    def load_config(self) -> bool:
        """Load configuration from config/config.ini"""
        try:
            config_path = Path("config/config.ini")
            if not config_path.exists():
                print("ERROR: config/config.ini not found")
                return False
                
            self.config = configparser.ConfigParser()
            self.config.read(config_path)
            
            # Load loader section if not already loaded by Pydantic
            if not HAVE_CONFIG_LOADER:
                if 'loader' in self.config:
                    loader_section = self.config['loader']
                    self.startup_order = [s.strip() for s in loader_section.get('startup_order', 'logger,kwd,stt,ollama,llm,tts').split(',')]
                    self.health_timeout_s = loader_section.getint('health_timeout_s', 30)
                    self.health_interval_ms = loader_section.getint('health_interval_ms', 200)
                    
                    backoff_str = loader_section.get('restart_backoff_ms', '250,500,1000,2000')
                    self.restart_backoff_ms = [int(x.strip()) for x in backoff_str.split(',')]
                    
                    self.crash_loop_threshold = loader_section.getint('crash_loop_threshold', 3)
                    self.port_hygiene = loader_section.getboolean('port_hygiene', True)
                    self.vram_probe_cmd = loader_section.get('vram_probe_cmd', self.vram_probe_cmd)
                    self.warmup_llm = loader_section.getboolean('warmup_llm', True)
                    self.greeting_on_ready = loader_section.getboolean('greeting_on_ready', True)
                
                # Load paths section
                if 'paths' in self.config:
                    paths_section = self.config['paths']
                    python_override = paths_section.get('python', None)
                    if python_override:
                        self.python = python_override
            
            # Load service configurations (always needed)
            for section_name in self.config.sections():
                if section_name.startswith('svc.'):
                    service_name = section_name[4:]  # Remove 'svc.' prefix
                    section = self.config[section_name]
                    
                    config = ServiceConfig(
                        name=section.get('name', service_name.upper()),
                        cmd=section.get('cmd', '').replace('%(python)s', self.python),
                        port=section.getint('port', 0),
                        health_url=section.get('health'),
                        warmup_url=section.get('warmup')
                    )
                    self.service_configs[service_name] = config
            
            return True
        except Exception as e:
            print(f"ERROR: Failed to load config: {e}")
            return False
    
    def parse_cli_args(self):
        """Parse command line arguments for selective service bring-up"""
        parser = argparse.ArgumentParser(description="Alexa FAST Loader Service")
        
        parser.add_argument("--only", type=str, help="Start only these services (comma-separated)")
        parser.add_argument("--skip", type=str, help="Skip these services (comma-separated)")
        parser.add_argument("--since", type=str, help="Start from this service onwards (inclusive)")
        parser.add_argument("--until", type=str, help="Stop after this service (inclusive)")
        parser.add_argument("--no-warmup", action="store_true", help="Skip LLM warmup")
        parser.add_argument("--no-greeting", action="store_true", help="Skip KWD greeting")
        parser.add_argument("--dry-run", action="store_true", help="Print plan and exit")
        
        args = parser.parse_args()
        
        # Check environment variables as fallbacks
        only = args.only or os.getenv("LOADER_ONLY")
        skip = args.skip or os.getenv("LOADER_SKIP")
        since = args.since or os.getenv("LOADER_SINCE")
        until = args.until or os.getenv("LOADER_UNTIL")
        no_warmup = args.no_warmup or bool(os.getenv("LOADER_NO_WARMUP"))
        no_greeting = args.no_greeting or bool(os.getenv("LOADER_NO_GREETING"))
        
        return {
            "only": only,
            "skip": skip,
            "since": since,
            "until": until,
            "no_warmup": no_warmup,
            "no_greeting": no_greeting,
            "dry_run": args.dry_run
        }
    
    def resolve_execution_plan(self, cli_options: Dict[str, Any]) -> List[str]:
        """Resolve execution plan based on CLI options"""
        # Start with full startup order
        full_order = self.startup_order.copy()
        
        # Logger is always first and included
        if "logger" not in full_order:
            full_order.insert(0, "logger")
        elif full_order[0] != "logger":
            full_order.remove("logger")
            full_order.insert(0, "logger")
        
        # Build candidate list (excluding logger which is fixed)
        candidates = full_order[1:]
        
        # Apply --only filter first
        if cli_options["only"]:
            only_set = {s.strip().lower() for s in cli_options["only"].split(",")}
            candidates = [svc for svc in candidates if svc.lower() in only_set]
        
        # Apply --since filter
        if cli_options["since"]:
            since_svc = cli_options["since"].strip().lower()
            try:
                since_index = next(i for i, svc in enumerate(candidates) if svc.lower() == since_svc)
                candidates = candidates[since_index:]
            except StopIteration:
                pass  # Service not found, keep all
        
        # Apply --until filter
        if cli_options["until"]:
            until_svc = cli_options["until"].strip().lower()
            try:
                until_index = next(i for i, svc in enumerate(candidates) if svc.lower() == until_svc)
                candidates = candidates[:until_index + 1]
            except StopIteration:
                pass  # Service not found, keep all
        
        # Apply --skip filter
        if cli_options["skip"]:
            skip_set = {s.strip().lower() for s in cli_options["skip"].split(",")}
            candidates = [svc for svc in candidates if svc.lower() not in skip_set]
        
        # Final plan: logger + selected candidates
        plan = ["logger"] + candidates
        
        # Set execution options
        self.skip_warmup = cli_options["no_warmup"]
        self.skip_greeting = cli_options["no_greeting"]
        self.dry_run = cli_options["dry_run"]
        
        return plan
    
    def print_execution_plan(self, plan: List[str], cli_options: Dict[str, Any]):
        """Print the execution plan"""
        # Determine what was skipped
        all_services = set(self.startup_order)
        planned_services = set(plan)
        skipped_services = all_services - planned_services
        
        # Build plan description
        plan_str = " → ".join(plan)
        
        # Build options description
        options = []
        if skipped_services:
            options.append(f"skip: {','.join(sorted(skipped_services))}")
        if self.skip_warmup:
            options.append("no_warmup:true")
        else:
            options.append("no_warmup:false")
        if self.skip_greeting:
            options.append("no_greeting:true")
        else:
            options.append("no_greeting:false")
        
        options_str = "; ".join(options) if options else "default"
        
        plan_message = f"Plan: {plan_str} ({options_str})"
        
        if self.dry_run:
            print(f"LOADER    INFO  = {plan_message}")
        else:
            self.log_to_logger("info", plan_message)
    
    def log_to_logger(self, level: str, message: str, event: str = None, extra: Dict[str, Any] = None):
        """Send log message to Logger service"""
        try:
            payload = {
                "svc": "LOADER",
                "level": level,
                "message": message
            }
            if event:
                payload["event"] = event
            if extra:
                payload["extra"] = extra
                
            import requests
            requests.post(f"{self.logger_url}/log", json=payload, timeout=2.0)
        except Exception:
            # Fallback to stdout if logger unavailable
            print(f"LOADER    {level.upper():<6}= {message}")
    
    def send_metric(self, metric: str, value: float, dialog_id: str = None):
        """Send metric to Logger service"""
        try:
            payload = {
                "svc": "LOADER",
                "metric": metric,
                "value": value
            }
            if dialog_id:
                payload["dialog_id"] = dialog_id
                
            import requests
            requests.post(f"{self.logger_url}/metrics", json=payload, timeout=2.0)
        except Exception:
            pass
    
    def send_vram_snapshot(self, svc: str, note: str = ""):
        """Capture and send VRAM usage to Logger"""
        try:
            result = subprocess.run(
                self.vram_probe_cmd.split(),
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                vram_mb = int(result.stdout.strip())
                payload = {
                    "svc": svc,
                    "vram_mb": vram_mb,
                    "note": note
                }
                import requests
                requests.post(f"{self.logger_url}/vram", json=payload, timeout=2.0)
        except Exception:
            # VRAM probing is optional
            pass
    
    def kill_stale_processes_on_port(self, port: int) -> bool:
        """Kill any processes using the specified port"""
        if not self.port_hygiene or port == 0:
            return True
            
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    connections = proc.net_connections()
                    for conn in connections:
                        if conn.laddr.port == port:
                            proc.terminate()
                            proc.wait(timeout=3)
                            self.log_to_logger("info", f"Killed stale process {proc.pid} on port {port}")
                            return True
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                    continue
            return True
        except Exception:
            return False
    
    def spawn_service(self, service_name: str) -> bool:
        """Spawn a service process"""
        if service_name not in self.service_configs:
            self.log_to_logger("error", f"Unknown service: {service_name}")
            return False
            
        config = self.service_configs[service_name]
        
        # Port hygiene
        if config.port > 0:
            self.kill_stale_processes_on_port(config.port)
        
        try:
            # Spawn process without individual stdout/stderr logging
            # Services should log through the centralized Logger service
            start_time = time.time()
            process = subprocess.Popen(
                config.cmd.split(),
                stdout=subprocess.DEVNULL,  # Discard stdout - services use Logger service
                stderr=subprocess.DEVNULL,  # Discard stderr - services use Logger service  
                preexec_fn=os.setsid  # Create new process group
            )
            
            # Create service process record
            service_proc = ServiceProcess(
                config=config,
                process=process,
                pid=process.pid,
                start_time=start_time
            )
            self.services[service_name] = service_proc
            
            self.log_to_logger("info", f"Starting {service_name} service", "service_start",
                             {"pid": process.pid, "port": config.port})
            
            return True
            
        except Exception as e:
            self.log_to_logger("error", f"Failed to spawn {service_name}: {e}", "service_error")
            return False
    
    def health_check_service(self, service_name: str) -> bool:
        """Check if service is healthy via /health endpoint."""
        if service_name not in self.services:
            return False
            
        service = self.services[service_name]
        
        # Logger is special - we assume it's ready if we can send logs to it
        if service_name == "logger":
            try:
                # Still check the actual health endpoint for consistency
                import requests
                response = requests.get(f"{self.logger_url}/health", timeout=1.0)
                if response.status_code == 200:
                    data = response.json()
                    return data.get("state") == "READY"
                return False
            except Exception:
                return False
        
        # Regular health check for other services
        if service.config.health_url:
            try:
                import requests
                response = requests.get(service.config.health_url, timeout=1.0)
                if response.status_code == 200:
                    data = response.json()
                    # A service is healthy if it's READY or already ACTIVE
                    return data.get("state") in ["READY", "ACTIVE"]
                return False
            except Exception:
                return False
        
        # No health URL - just check if process is alive (fallback)
        return service.process and service.process.poll() is None
    
    def wait_for_service_health(self, service_name: str) -> bool:
        """Wait for service to become healthy with timeout"""
        start_time = time.time()
        interval_sec = self.health_interval_ms / 1000.0
        
        while time.time() - start_time < self.health_timeout_s:
            if service_name not in self.services:
                return False
                
            service = self.services[service_name]
            
            # Check if process died
            if service.process and service.process.poll() is not None:
                self.log_to_logger("error", f"{service_name} process exited during health check", "service_error")
                return False
            
            # Health check
            if self.health_check_service(service_name):
                # Mark as ready
                service.ready_time = time.time()
                duration_ms = (service.ready_time - service.start_time) * 1000
                
                self.log_to_logger("info", f"Service '{service_name}' ready (PID={service.pid}, port={service.config.port})", 
                                 "service_ready", {"duration_ms": duration_ms})
                
                # Send metrics
                self.send_metric("svc_start_ms", service.start_time * 1000)
                self.send_metric("svc_ready_ms", service.ready_time * 1000)
                self.send_metric("latency_ms", duration_ms)
                
                # VRAM snapshot
                self.send_vram_snapshot(service.config.name, f"after {service_name} load")
                
                return True
            
            time.sleep(interval_sec)
        
        self.log_to_logger("error", f"{service_name} failed health check timeout", "service_error")
        return False
    
    def _perform_warmup_llm(self) -> bool:
        """Warm up LLM service if configured"""
        # Check if warmup is disabled via CLI or not configured
        if self.skip_warmup or not self.warmup_llm or "llm" not in self.services:
            return True
        
        # Check if LLM is in execution plan
        if "llm" not in self.execution_plan:
            return True
            
        llm_config = self.service_configs.get("llm")
        if not llm_config or not llm_config.warmup_url:
            return True
            
        try:
            self.log_to_logger("info", "Warming up LLM service")
            import requests
            response = requests.post(llm_config.warmup_url, timeout=30.0)
            
            if response.status_code == 200:
                self.log_to_logger("info", "LLM warmup completed")
                self.send_vram_snapshot("LLM", "after warmup")
                return True
            else:
                self.log_to_logger("warning", f"LLM warmup failed: {response.status_code}")
                return False
        except Exception as e:
            self.log_to_logger("warning", f"LLM warmup error: {e}")
            return False
    
    def trigger_kwd_greeting(self) -> bool:
        """Trigger KWD greeting after all services ready"""
        # Check if greeting is disabled via CLI or not configured
        if self.skip_greeting or not self.greeting_on_ready or "kwd" not in self.services:
            return True
        
        # Check if KWD is in execution plan
        if "kwd" not in self.execution_plan:
            return True
            
        try:
            kwd_url = "http://127.0.0.1:5002/on-system-ready"
            import requests
            response = requests.post(kwd_url, timeout=5.0)
            
            if response.status_code == 200:
                self.log_to_logger("info", "KWD greeting triggered", "system_ready")
                return True
            else:
                self.log_to_logger("warning", f"KWD greeting failed: {response.status_code}")
                return False
        except Exception as e:
            self.log_to_logger("warning", f"KWD greeting error: {e}")
            return False
    
    def get_restart_backoff(self, restart_count: int) -> float:
        """Get restart backoff with jitter"""
        if restart_count >= len(self.restart_backoff_ms):
            backoff_ms = self.restart_backoff_ms[-1]
        else:
            backoff_ms = self.restart_backoff_ms[restart_count]
        
        # Add ±20% jitter
        jitter = random.uniform(0.8, 1.2)
        return (backoff_ms * jitter) / 1000.0
    
    def is_crash_loop(self, service_name: str) -> bool:
        """Check if service is in crash loop"""
        if service_name not in self.services:
            return False
            
        service = self.services[service_name]
        if service.restart_count < self.crash_loop_threshold:
            return False
            
        # Check if restarts happened within 60s
        if service.last_restart_time:
            time_since_restart = time.time() - service.last_restart_time
            return time_since_restart < 60.0
            
        return False
    
    def restart_service(self, service_name: str) -> bool:
        """Restart a failed service"""
        if service_name not in self.services:
            return False
            
        service = self.services[service_name]
        
        # Check crash loop
        if self.is_crash_loop(service_name):
            self.log_to_logger("error", f"{service_name} in crash loop, giving up", "service_error")
            return False
        
        # Kill existing process
        if service.process:
            try:
                service.process.terminate()
                service.process.wait(timeout=5)
            except Exception:
                pass
        
        # Backoff
        backoff = self.get_restart_backoff(service.restart_count)
        self.log_to_logger("info", f"Restarting {service_name} after {backoff:.1f}s backoff")
        self.send_metric("restart_backoff_ms", backoff * 1000)
        time.sleep(backoff)
        
        # Update restart tracking
        service.restart_count += 1
        service.last_restart_time = time.time()
        
        # Restart
        return self.spawn_service(service_name) and self.wait_for_service_health(service_name)
    
    def supervise_services(self):
        """Monitor running services and restart if needed"""
        for service_name, service in self.services.items():
            if service.process and service.process.poll() is not None:
                # Process died
                exit_code = service.process.returncode
                self.log_to_logger("error", f"{service_name} crashed (exit code {exit_code})", "service_error")
                
                if not self.restart_service(service_name):
                    self.log_to_logger("error", f"Failed to restart {service_name}", "service_error")
                    self.state = LoaderState.FAILED
                    return False
        
        return True
    
    def shutdown_services(self):
        """Gracefully shutdown all services in reverse order"""
        self.log_to_logger("info", "Shutdown initiated", "shutdown_initiated")
        
        # Reverse execution plan order (not full startup order)
        shutdown_order = list(reversed(self.execution_plan)) if self.execution_plan else list(reversed(self.startup_order))
        
        for service_name in shutdown_order:
            if service_name in self.services:
                service = self.services[service_name]
                if service.process and service.process.poll() is None:
                    try:
                        self.log_to_logger("info", f"Stopping {service_name}")
                        service.process.terminate()
                        service.process.wait(timeout=10)
                        self.log_to_logger("info", f"Service '{service_name}' stopped", "service_stop")
                    except subprocess.TimeoutExpired:
                        self.log_to_logger("warning", f"Force killing {service_name}")
                        service.process.kill()
                        service.process.wait()
        
        self.log_to_logger("info", "Shutdown complete", "shutdown_complete")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.shutdown_requested = True
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def run(self) -> int:
        """Main execution loop"""
        try:
            self.system_start_time = time.time()
            self.setup_signal_handlers()
            
            # Parse CLI arguments for selective bring-up
            cli_options = self.parse_cli_args()
            
            # INIT: Load configuration
            if not self.load_config():
                return 1
            
            # Resolve execution plan
            self.execution_plan = self.resolve_execution_plan(cli_options)
            
            # Print/log execution plan
            self.print_execution_plan(self.execution_plan, cli_options)
            
            # Dry run mode - exit after printing plan
            if self.dry_run:
                return 0
            
            # Log startup (only after confirming not dry run)
            self.log_to_logger("info", "Loader service started", "service_start")
            
            # START_SEQUENCE: Start services in execution plan order
            self.state = LoaderState.START_SEQUENCE
            
            for service_name in self.execution_plan:
                if self.shutdown_requested:
                    break
                    
                self.current_service_index = self.execution_plan.index(service_name)
                
                # HEALTH_GATE: Spawn and wait for health
                self.state = LoaderState.HEALTH_GATE
                
                if not self.spawn_service(service_name):
                    self.state = LoaderState.FAILED
                    return 1
                
                if not self.wait_for_service_health(service_name):
                    # RECOVER: Try to restart
                    self.state = LoaderState.RECOVER
                    if not self.restart_service(service_name):
                        self.state = LoaderState.FAILED
                        return 1
                
                self.state = LoaderState.START_SEQUENCE
            
            # All services ready
            if not self.shutdown_requested:
                self.state = LoaderState.SYSTEM_READY
                self.system_ready_time = time.time()
                
                system_ready_duration = (self.system_ready_time - self.system_start_time) * 1000
                self.send_metric("system_ready_ms", system_ready_duration)
                
                # Log that selected services are ready
                if len(self.execution_plan) == len(self.startup_order):
                    self.log_to_logger("info", "All services ready", "all_services_ready")
                else:
                    self.log_to_logger("info", "Selected services ready", "selected_services_ready")
                
                # Warmup LLM
                self._perform_warmup_llm()
                
                # Trigger KWD greeting
                self.trigger_kwd_greeting()
                
                # RUN: Supervision loop
                self.state = LoaderState.RUN
                while not self.shutdown_requested and self.state == LoaderState.RUN:
                    if not self.supervise_services():
                        break
                    time.sleep(1.0)
            
            # STOPPING: Graceful shutdown
            self.state = LoaderState.STOPPING
            self.shutdown_services()
            
            return 0 if self.state != LoaderState.FAILED else 1
            
        except KeyboardInterrupt:
            self.log_to_logger("info", "Interrupted by user")
            self.shutdown_services()
            return 0
        except Exception as e:
            self.log_to_logger("error", f"Loader error: {e}", "service_error")
            self.shutdown_services()
            return 1


def main():
    """Entry point"""
    loader = LoaderService()
    exit_code = loader.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()