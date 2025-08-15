"""Utility functions for AI Voice Assistant services"""

import os
import psutil
import logging
import time
from typing import Optional


def check_vram_availability(min_vram_mb: int = 8000) -> bool:
    """Check if system has sufficient VRAM for AI models"""
    try:
        import torch
        if torch.cuda.is_available():
            # Get total VRAM in MB
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            logging.info(f"CUDA available. Total VRAM: {total_vram:.0f}MB")
            return total_vram >= min_vram_mb
        else:
            logging.warning("CUDA not available")
            return False
    except ImportError:
        logging.error("PyTorch not installed, cannot check VRAM")
        return False


def ensure_directory_exists(path: str) -> bool:
    """Ensure a directory exists, create if it doesn't"""
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except Exception as e:
        logging.error(f"Failed to create directory {path}: {e}")
        return False


def get_process_info(pid: int) -> Optional[dict]:
    """Get process information by PID"""
    try:
        process = psutil.Process(pid)
        return {
            'pid': pid,
            'name': process.name(),
            'status': process.status(),
            'cpu_percent': process.cpu_percent(),
            'memory_mb': process.memory_info().rss / (1024 * 1024),
            'create_time': process.create_time()
        }
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def wait_for_port(host: str, port: int, timeout: float = 10.0) -> bool:
    """Wait for a port to become available"""
    import socket
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                return True
        except Exception:
            pass
        
        time.sleep(0.1)
    
    return False


def generate_dialog_id() -> str:
    """Generate a unique dialog ID with timestamp"""
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return f"dialog_{timestamp}"


def setup_logging(level: str = "INFO", service_name: str = "main") -> logging.Logger:
    """Setup logging configuration for a service"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=f'%(asctime)s - {service_name} - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S'
    )
    return logging.getLogger(service_name)