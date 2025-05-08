import subprocess
import sys
import os
import time
import logging
from typing import Dict, Optional
from datetime import datetime

from .message_bus import MessageBus

class ProcessManager:
    """Manages subprocesses for different tabs and handles process monitoring."""
    
    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.heartbeats: Dict[str, float] = {}
        self.message_bus = MessageBus()
        self.setup_message_handlers()
        
    def setup_message_handlers(self):
        """Set up message handlers for process monitoring."""
        self.message_bus.subscribe("heartbeat", self.handle_message)
        
    def handle_message(self, sender: str, message_type: str, data: dict):
        """Handle incoming messages."""
        try:
            if message_type == "heartbeat":
                tab_name = data.get("tab_name")
                if tab_name:
                    self.heartbeats[tab_name] = time.time()
                    logging.debug(f"Received heartbeat from {tab_name}")
                    
            elif message_type == "process_status":
                tab_name = data.get("tab_name")
                status = data.get("status")
                if tab_name and status:
                    logging.info(f"Process {tab_name} status: {status}")
                    
            elif message_type == "process_error":
                tab_name = data.get("tab_name")
                error = data.get("error")
                if tab_name and error:
                    logging.error(f"Process {tab_name} error: {error}")
                    
            elif message_type == "process_log":
                tab_name = data.get("tab_name")
                log = data.get("log")
                if tab_name and log:
                    logging.info(f"Process {tab_name} log: {log}")
                    
        except Exception as e:
            logging.error(f"Error handling message from {sender}: {str(e)}")
        
    def start_tab(self, tab_name: str, script_path: str) -> bool:
        """Start a new tab process."""
        try:
            if tab_name in self.processes:
                logging.warning(f"Tab {tab_name} is already running")
                return False
                
            process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.processes[tab_name] = process
            self.heartbeats[tab_name] = time.time()
            logging.info(f"Started tab {tab_name} with PID {process.pid}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to start tab {tab_name}: {str(e)}")
            return False
            
    def stop_tab(self, tab_name: str) -> bool:
        """Stop a running tab process."""
        try:
            if tab_name not in self.processes:
                logging.warning(f"Tab {tab_name} is not running")
                return False
                
            process = self.processes[tab_name]
            process.terminate()
            process.wait(timeout=5)
            
            del self.processes[tab_name]
            if tab_name in self.heartbeats:
                del self.heartbeats[tab_name]
                
            logging.info(f"Stopped tab {tab_name}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to stop tab {tab_name}: {str(e)}")
            return False
            
    def restart_tab(self, tab_name: str) -> bool:
        """Restart a tab process."""
        if self.stop_tab(tab_name):
            script_path = f"modules/tabs/{tab_name.lower()}_tab.py"
            return self.start_tab(tab_name, script_path)
        return False
        
    def check_status(self, tab_name: str) -> str:
        """Check the status of a tab process."""
        if tab_name not in self.processes:
            return "Not Running"
            
        process = self.processes[tab_name]
        if process.poll() is not None:
            return "Stopped"
            
        last_heartbeat = self.heartbeats.get(tab_name, 0)
        if time.time() - last_heartbeat > 30:  # 30 seconds timeout
            return "Not Responding"
            
        return "Running"
        
    def get_recent_logs(self, tab_name: str) -> str:
        """Get recent logs from a tab process."""
        if tab_name not in self.processes:
            return "Process not running"
            
        process = self.processes[tab_name]
        try:
            stdout, stderr = process.communicate(timeout=1)
            return f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
        except:
            return "Failed to get logs"
            
    def cleanup(self):
        """Clean up all processes."""
        for tab_name in list(self.processes.keys()):
            self.stop_tab(tab_name)
        self.message_bus.cleanup() 