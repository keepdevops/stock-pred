import os
import time
import logging
import subprocess
from typing import Dict, Optional
from datetime import datetime

class ProcessManager:
    """Manages the lifecycle of tab processes and their monitoring."""
    
    def __init__(self, tab_scripts: Dict[str, str]):
        """
        Initialize the ProcessManager with tab scripts.
        
        Args:
            tab_scripts (Dict[str, str]): Dictionary mapping tab names to their script paths
        """
        self.tab_scripts = tab_scripts
        self.processes: Dict[str, subprocess.Popen] = {}
        self.heartbeats: Dict[str, float] = {}
        self.logger = logging.getLogger(__name__)
        
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
    def launch_tab(self, tab_name: str) -> bool:
        """
        Launch a tab process.
        
        Args:
            tab_name (str): Name of the tab to launch
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if tab_name not in self.tab_scripts:
                self.logger.error(f"No script found for tab: {tab_name}")
                return False
                
            script_path = self.tab_scripts[tab_name]
            if not os.path.exists(script_path):
                self.logger.error(f"Script not found: {script_path}")
                return False
                
            # Launch the process
            process = subprocess.Popen(
                ["python", script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes[tab_name] = process
            self.heartbeats[tab_name] = time.time()
            self.logger.info(f"Launched {tab_name} tab with PID: {process.pid}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error launching {tab_name} tab: {str(e)}")
            return False
            
    def terminate_tab(self, tab_name: str) -> bool:
        """
        Terminate a tab process.
        
        Args:
            tab_name (str): Name of the tab to terminate
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if tab_name not in self.processes:
                self.logger.warning(f"No process found for tab: {tab_name}")
                return False
                
            process = self.processes[tab_name]
            process.terminate()
            process.wait(timeout=5)
            
            del self.processes[tab_name]
            if tab_name in self.heartbeats:
                del self.heartbeats[tab_name]
                
            self.logger.info(f"Terminated {tab_name} tab")
            return True
            
        except Exception as e:
            self.logger.error(f"Error terminating {tab_name} tab: {str(e)}")
            return False
            
    def restart_tab(self, tab_name: str) -> bool:
        """
        Restart a tab process.
        
        Args:
            tab_name (str): Name of the tab to restart
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.terminate_tab(tab_name):
                time.sleep(1)  # Give the process time to fully terminate
                return self.launch_tab(tab_name)
            return False
            
        except Exception as e:
            self.logger.error(f"Error restarting {tab_name} tab: {str(e)}")
            return False
            
    def handle_heartbeat(self, tab_name: str, pid: int, timestamp: float) -> None:
        """
        Handle a heartbeat from a tab process.
        
        Args:
            tab_name (str): Name of the tab sending the heartbeat
            pid (int): Process ID of the tab
            timestamp (float): Timestamp of the heartbeat
        """
        try:
            if tab_name in self.processes and self.processes[tab_name].pid == pid:
                self.heartbeats[tab_name] = timestamp
                self.logger.debug(f"Heartbeat received from {tab_name} tab")
            else:
                self.logger.warning(f"Received heartbeat from unknown process: {tab_name} (PID: {pid})")
                
        except Exception as e:
            self.logger.error(f"Error handling heartbeat: {str(e)}")
            
    def check_status(self, tab_name: str) -> str:
        """
        Check the status of a tab process.
        
        Args:
            tab_name (str): Name of the tab to check
            
        Returns:
            str: Status of the tab ("Running", "Crashed", "Not Found")
        """
        try:
            if tab_name not in self.processes:
                return "Not Found"
                
            process = self.processes[tab_name]
            if process.poll() is not None:
                return "Crashed"
                
            # Check if we've received a heartbeat recently (within last 30 seconds)
            if tab_name in self.heartbeats:
                last_heartbeat = self.heartbeats[tab_name]
                if time.time() - last_heartbeat > 30:
                    return "Crashed"
                    
            return "Running"
            
        except Exception as e:
            self.logger.error(f"Error checking status of {tab_name} tab: {str(e)}")
            return "Crashed"
            
    def get_recent_logs(self, tab_name: str, lines: int = 50) -> Optional[str]:
        """
        Get recent logs from a tab process.
        
        Args:
            tab_name (str): Name of the tab
            lines (int): Number of lines to retrieve
            
        Returns:
            Optional[str]: Recent logs or None if not available
        """
        try:
            log_file = f"logs/{tab_name.lower()}_tab.log"
            if not os.path.exists(log_file):
                return None
                
            with open(log_file, 'r') as f:
                log_lines = f.readlines()
                return ''.join(log_lines[-lines:])
                
        except Exception as e:
            self.logger.error(f"Error getting logs for {tab_name} tab: {str(e)}")
            return None
            
    def cleanup(self) -> None:
        """Clean up all processes and resources."""
        try:
            for tab_name in list(self.processes.keys()):
                self.terminate_tab(tab_name)
                
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}") 