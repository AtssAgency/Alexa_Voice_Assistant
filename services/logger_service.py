#!/usr/bin/env python3
"""
Logger Service - Central logging, metrics aggregation, and dialog transcripts

FastAPI service on port 5000 that:
- Accepts events from all services via POST /log
- Aggregates metrics (TTFT, tokens/sec, VRAM) via POST /metrics
- Writes dialog transcripts via POST /dialog/*
- Emits console storyline with configurable verbosity
- Persists structured logs to files (app.log, memory.log, dialog_*.log)

State: INIT → READY → ERROR (with auto-recovery)
"""

import os
import sys
import json
import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import configparser
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn


# Request/Response Models
class LogRequest(BaseModel):
    svc: str
    level: str
    event: Optional[str] = None
    message: str = ""
    dialog_id: Optional[str] = None
    ts_ms: Optional[int] = None
    extra: Optional[Dict[str, Any]] = None


class MetricRequest(BaseModel):
    svc: str
    dialog_id: Optional[str] = None
    metric: str
    value: float
    ts_ms: Optional[int] = None


class VRAMRequest(BaseModel):
    svc: str
    vram_mb: int
    ts_ms: Optional[int] = None
    note: Optional[str] = None


class DialogStartRequest(BaseModel):
    dialog_id: str
    ts_ms: Optional[int] = None


class DialogLineRequest(BaseModel):
    dialog_id: str
    role: str
    text: str


class DialogEndRequest(BaseModel):
    dialog_id: str
    ts_ms: Optional[int] = None


class HealthResponse(BaseModel):
    status: str
    state: str


class LoggerService:
    def __init__(self):
        self.state = "INIT"
        self.config: Optional[configparser.ConfigParser] = None
        
        # File handles
        self.app_log_file = None
        self.memory_log_file = None
        self.memory_csv_writer = None
        self.dialog_files: Dict[str, Any] = {}
        
        # Configuration
        self.log_level = "info"
        self.dialog_prefix = "dialog_"
        self.append_timestamps = True
        self.port = 5000
        
        # Valid service names
        self.valid_services = {"MAIN", "LOADER", "KWD", "STT", "LLM", "TTS", "RMS", "LOGGER"}
        
        # Echo policy - events that show up in console at info level
        self.echo_events = {
            "service_start", "service_ready", "service_stop", "service_error",
            "all_services_ready", "system_ready",
            "kwd_started", "wake_detected", "kwd_stopped", 
            "stt_started", "stt_final_text",
            "llm_stream_end"
        }
    
    def load_config(self) -> bool:
        """Load configuration from config/config.ini"""
        try:
            config_path = Path("config/config.ini")
            if not config_path.exists():
                print("WARNING: config/config.ini not found, using defaults")
                return True
                
            self.config = configparser.ConfigParser()
            self.config.read(config_path)
            
            # Load logger section
            if 'logger' in self.config:
                logger_section = self.config['logger']
                self.log_level = logger_section.get('log_level', 'info').lower()
                self.dialog_prefix = logger_section.get('dialog_prefix', 'dialog_')
                self.append_timestamps = logger_section.getboolean('append_timestamps', True)
                self.port = logger_section.getint('port', 5000)
            
            return True
        except Exception as e:
            print(f"ERROR: Failed to load config: {e}")
            return False
    
    def setup_log_files(self) -> bool:
        """Create logs directory and open log files"""
        try:
            # Create logs directory
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
            
            # Open app.log (JSON lines)
            app_log_path = logs_dir / "app.log"
            self.app_log_file = open(app_log_path, 'w')
            
            # Open memory.log (CSV format)
            memory_log_path = logs_dir / "memory.log"
            self.memory_log_file = open(memory_log_path, 'w', newline='')
            self.memory_csv_writer = csv.writer(self.memory_log_file)
            
            # Write CSV header
            self.memory_csv_writer.writerow(["ts", "svc", "metric", "value", "note"])
            self.memory_log_file.flush()
            
            return True
        except Exception as e:
            print(f"ERROR: Failed to setup log files: {e}")
            return False
    
    def get_timestamp(self) -> str:
        """Get ISO timestamp"""
        return datetime.utcnow().isoformat() + 'Z'
    
    def format_console_message(self, svc: str, level: str, message: str) -> str:
        """Format message for console output using the specified template"""
        return f"{svc:<10}{level.upper():<6}= {message}"
    
    def should_echo_to_console(self, level: str, event: Optional[str]) -> bool:
        """Determine if message should be echoed to console based on log level and echo policy"""
        if self.log_level == "debug":
            return True
        elif self.log_level == "info":
            # Only echo storyline events
            return event in self.echo_events if event else False
        return False
    
    def write_app_log(self, log_data: Dict[str, Any]):
        """Write structured log entry to app.log"""
        try:
            if self.app_log_file:
                json.dump(log_data, self.app_log_file)
                self.app_log_file.write('\n')
                self.app_log_file.flush()
        except Exception as e:
            print(f"ERROR: Failed to write to app.log: {e}")
    
    def write_memory_log(self, svc: str, metric: str, value: float, note: str = ""):
        """Write metric entry to memory.log (CSV format)"""
        try:
            if self.memory_csv_writer:
                timestamp = self.get_timestamp()
                self.memory_csv_writer.writerow([timestamp, svc, metric, value, note])
                self.memory_log_file.flush()
        except Exception as e:
            print(f"ERROR: Failed to write to memory.log: {e}")
    
    def get_dialog_filename(self, dialog_id: str) -> str:
        """Generate dialog log filename with timestamp"""
        timestamp = datetime.now().strftime("%d-%m_%H-%M-%S")
        return f"{self.dialog_prefix}{timestamp}.log"
    
    def start_dialog(self, dialog_id: str):
        """Start a new dialog transcript file"""
        try:
            if dialog_id not in self.dialog_files:
                filename = self.get_dialog_filename(dialog_id)
                filepath = Path("logs") / filename
                self.dialog_files[dialog_id] = open(filepath, 'w')
        except Exception as e:
            print(f"ERROR: Failed to start dialog {dialog_id}: {e}")
    
    def write_dialog_line(self, dialog_id: str, role: str, text: str):
        """Write a line to dialog transcript (USER/ALEXA only)"""
        try:
            if dialog_id in self.dialog_files and role in ["USER", "ALEXA"]:
                file_handle = self.dialog_files[dialog_id]
                file_handle.write(f"{role}: {text}\n")
                file_handle.flush()
        except Exception as e:
            print(f"ERROR: Failed to write dialog line: {e}")
    
    def end_dialog(self, dialog_id: str):
        """Close dialog transcript file"""
        try:
            if dialog_id in self.dialog_files:
                self.dialog_files[dialog_id].close()
                del self.dialog_files[dialog_id]
        except Exception as e:
            print(f"ERROR: Failed to end dialog {dialog_id}: {e}")
    
    def cleanup(self):
        """Close all file handles"""
        try:
            if self.app_log_file:
                self.app_log_file.close()
            if self.memory_log_file:
                self.memory_log_file.close()
            for dialog_file in self.dialog_files.values():
                dialog_file.close()
            self.dialog_files.clear()
        except Exception:
            pass
    
    async def startup(self):
        """Service startup logic"""
        try:
            # Load configuration
            if not self.load_config():
                self.state = "ERROR"
                return
                
            # Setup log files
            if not self.setup_log_files():
                self.state = "ERROR"
                return
                
            self.state = "READY"
            
            # Log our own startup
            startup_data = {
                "ts": self.get_timestamp(),
                "svc": "LOGGER",
                "level": "INFO",
                "event": "service_start",
                "message": f"Logger service started on port {self.port}"
            }
            self.write_app_log(startup_data)
            
            # Echo to console
            console_msg = self.format_console_message("LOGGER", "INFO", f"Logger service started on port {self.port}")
            print(console_msg)
            
        except Exception as e:
            print(f"FATAL: Logger startup failed: {e}")
            self.state = "ERROR"


# Global logger instance
logger_service = LoggerService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    await logger_service.startup()
    yield
    # Shutdown
    logger_service.cleanup()


# FastAPI app
app = FastAPI(
    title="Logger Service",
    description="Central logging, metrics aggregation, and dialog transcripts",
    version="1.0",
    lifespan=lifespan
)


@app.post("/log")
async def log_event(request: LogRequest):
    """Accept log events from all services"""
    try:
        # Validate service name
        if request.svc not in logger_service.valid_services:
            raise HTTPException(status_code=400, detail=f"Invalid service name: {request.svc}")
        
        # Use current timestamp if not provided
        timestamp = logger_service.get_timestamp()
        if request.ts_ms:
            timestamp = datetime.fromtimestamp(request.ts_ms / 1000).isoformat() + 'Z'
        
        # Build log entry
        log_data = {
            "ts": timestamp if logger_service.append_timestamps else None,
            "svc": request.svc,
            "level": request.level.upper(),
            "message": request.message
        }
        
        if request.event:
            log_data["event"] = request.event
        if request.dialog_id:
            log_data["dialog_id"] = request.dialog_id
        if request.extra:
            log_data.update(request.extra)
        
        # Remove None values
        log_data = {k: v for k, v in log_data.items() if v is not None}
        
        # Write to app.log
        logger_service.write_app_log(log_data)
        
        # Console echo based on policy
        if logger_service.should_echo_to_console(request.level, request.event):
            console_msg = logger_service.format_console_message(
                request.svc, request.level, request.message
            )
            print(console_msg)
        
        return {"status": "ok"}
        
    except Exception as e:
        print(f"ERROR in /log endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/metrics")
async def log_metric(request: MetricRequest):
    """Accept metrics from services (TTFT, tokens/sec, etc.)"""
    try:
        # Validate service name
        if request.svc not in logger_service.valid_services:
            raise HTTPException(status_code=400, detail=f"Invalid service name: {request.svc}")
        
        # Create note from dialog_id if provided
        note = f"dialog={request.dialog_id}" if request.dialog_id else ""
        
        # Write to memory.log
        logger_service.write_memory_log(request.svc, request.metric, request.value, note)
        
        return {"status": "ok"}
        
    except Exception as e:
        print(f"ERROR in /metrics endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/vram")
async def log_vram(request: VRAMRequest):
    """Accept VRAM usage snapshots from services"""
    try:
        # Validate service name
        if request.svc not in logger_service.valid_services:
            raise HTTPException(status_code=400, detail=f"Invalid service name: {request.svc}")
        
        # Write to memory.log
        note = request.note or ""
        logger_service.write_memory_log(request.svc, "vram_mb", float(request.vram_mb), note)
        
        return {"status": "ok"}
        
    except Exception as e:
        print(f"ERROR in /vram endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/dialog/start")
async def start_dialog(request: DialogStartRequest):
    """Start a new dialog transcript"""
    try:
        logger_service.start_dialog(request.dialog_id)
        return {"status": "ok"}
        
    except Exception as e:
        print(f"ERROR in /dialog/start endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/dialog/line")
async def add_dialog_line(request: DialogLineRequest):
    """Add a line to dialog transcript (USER/ALEXA only)"""
    try:
        # Only accept USER and ALEXA roles (no system prompts)
        if request.role not in ["USER", "ALEXA"]:
            return {"status": "ignored", "reason": "Only USER/ALEXA roles accepted"}
        
        logger_service.write_dialog_line(request.dialog_id, request.role, request.text)
        return {"status": "ok"}
        
    except Exception as e:
        print(f"ERROR in /dialog/line endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/dialog/end")
async def end_dialog(request: DialogEndRequest):
    """End dialog transcript"""
    try:
        logger_service.end_dialog(request.dialog_id)
        return {"status": "ok"}
        
    except Exception as e:
        print(f"ERROR in /dialog/end endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return HealthResponse(status="ok", state=logger_service.state)


@app.get("/state")
async def get_state():
    """Get current service state"""
    return {"state": logger_service.state, "log_level": logger_service.log_level}


@app.post("/rotate")
async def rotate_logs():
    """Manually rotate log files"""
    try:
        logger_service.cleanup()
        if logger_service.setup_log_files():
            logger_service.state = "READY"
            return {"status": "ok", "message": "Logs rotated successfully"}
        else:
            logger_service.state = "ERROR"
            return {"status": "error", "message": "Failed to rotate logs"}
    except Exception as e:
        logger_service.state = "ERROR"
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Entry point when run as module"""
    # Run FastAPI server
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=logger_service.port,
        log_level="error",  # Suppress uvicorn logs to keep console clean
        access_log=False
    )


if __name__ == "__main__":
    main()
