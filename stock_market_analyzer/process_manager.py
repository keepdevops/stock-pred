import logging
import os
import time
import subprocess
import signal
from typing import Dict, Optional
from datetime import datetime

class ProcessManager:
    """Manager class for handling process management and monitoring."""
    
    def __init__(self):
        """Initialize the process manager."""
        self.logger = logging.getLogger(__name__)
        self.processes: Dict[str, subprocess.Popen] = {}
        self.heartbeats: Dict[str, float] = {}
        from .modules.message_bus import MessageBus
        self.message_bus = MessageBus()
        self.setup_message_handlers()
        
    def setup_message_handlers(self):
        """Set up message handlers for process monitoring."""
        self.message_bus.subscribe("ProcessMonitor", self.handle_message)
        
    def handle_message(self, message_type: str, data: Dict):
        """
        Handle incoming messages.
        
        Args:
            message_type: Type of message
            data: Message data
        """
        if message_type == "heartbeat":
            self.handle_heartbeat(data["tab"], data["pid"], data["timestamp"])
            
    def handle_heartbeat(self, tab_name: str, pid: int, timestamp: float):
        """
        Handle a heartbeat message from a process.
        
        Args:
            tab_name: Name of the tab sending the heartbeat
            pid: Process ID
            timestamp: Timestamp of the heartbeat
        """
        self.heartbeats[tab_name] = timestamp
        self.logger.debug(f"Heartbeat received from {tab_name} (PID: {pid})")
        
    def start_tab(self, tab_name: str, script_path: str) -> bool:
        """
        Start a new tab process.
        
        Args:
            tab_name: Name of the tab
            script_path: Path to the script to run
            
        Returns:
            True if process started successfully, False otherwise
        """
        try:
            if tab_name in self.processes:
                self.logger.warning(f"Process for {tab_name} already running")
                return False
                
            # Start the process
            process = subprocess.Popen(
                ["python", script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes[tab_name] = process
            self.heartbeats[tab_name] = time.time()
            
            self.logger.info(f"Started {tab_name} process with PID {process.pid}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting {tab_name} process: {e}")
            return False
            
    def stop_tab(self, tab_name: str) -> bool:
        """
        Stop a tab process.
        
        Args:
            tab_name: Name of the tab to stop
            
        Returns:
            True if process stopped successfully, False otherwise
        """
        try:
            if tab_name not in self.processes:
                self.logger.warning(f"No process found for {tab_name}")
                return False
                
            process = self.processes[tab_name]
            
            # Try to terminate gracefully
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if process doesn't terminate
                process.kill()
                self.logger.warning(f"Process {tab_name} did not terminate gracefully, forcing kill")
                
            del self.processes[tab_name]
            if tab_name in self.heartbeats:
                del self.heartbeats[tab_name]
                
            self.logger.info(f"Stopped {tab_name} process")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping {tab_name} process: {e}")
            return False
            
    def restart_tab(self, tab_name: str) -> bool:
        """
        Restart a tab process.
        
        Args:
            tab_name: Name of the tab to restart
            
        Returns:
            True if process restarted successfully, False otherwise
        """
        try:
            if tab_name not in self.processes:
                self.logger.warning(f"No process found for {tab_name}")
                return False
                
            script_path = self.processes[tab_name].args[1]
            
            # Stop the current process
            if not self.stop_tab(tab_name):
                return False
                
            # Start a new process
            return self.start_tab(tab_name, script_path)
            
        except Exception as e:
            self.logger.error(f"Error restarting {tab_name} process: {e}")
            return False
            
    def check_status(self, tab_name: str) -> str:
        """
        Check the status of a tab process.
        
        Args:
            tab_name: Name of the tab
            
        Returns:
            Status string ("Running", "Stopped", or "Unknown")
        """
        if tab_name not in self.processes:
            return "Stopped"
            
        process = self.processes[tab_name]
        if process.poll() is None:
            # Check if process is responding
            if time.time() - self.heartbeats.get(tab_name, 0) > 30:
                return "Unknown"
            return "Running"
        return "Stopped"
        
    def get_recent_logs(self, tab_name: str) -> str:
        """
        Get recent logs from a tab process.
        
        Args:
            tab_name: Name of the tab
            
        Returns:
            Recent log output
        """
        if tab_name not in self.processes:
            return "No logs available"
            
        process = self.processes[tab_name]
        try:
            stdout, stderr = process.communicate(timeout=1)
            return stdout or stderr or "No logs available"
        except:
            return "No logs available"
            
    def cleanup(self):
        """Clean up all processes."""
        for tab_name in list(self.processes.keys()):
            self.stop_tab(tab_name) 