import sys
import os
import logging
from PyQt6.QtCore import QObject, QProcess, pyqtSignal, QProcessEnvironment
from PyQt6.QtWidgets import QApplication, QWidget
from typing import Any, Optional
from .message_bus import MessageBus

class TabProcess(QWidget):
    """Base class for tab processes with inter-tab communication."""
    
    def __init__(self, tab_name: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.tab_name = tab_name
        self.logger = logging.getLogger(__name__)
        self.message_bus = MessageBus()
        self.process = QProcess()
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)
        self.process.finished.connect(self.handle_finished)
        
    def start_process(self):
        """Start the tab process."""
        try:
            tab_name_lower = self.tab_name.lower()
            # Define the module name to execute
            module_name = f"stock_market_analyzer.modules.tabs.{tab_name_lower}_tab"
            
            # Get the project root directory (Corrected: go up two levels)
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

            # Check if the tab module file exists (optional but good practice)
            # Construct path relative to project root
            tab_file_path = os.path.join(project_root, "stock_market_analyzer", "modules", "tabs", f"{tab_name_lower}_tab.py")
            # *Alternative using module name split (should also work now)*:
            # tab_file_path = os.path.join(project_root, *module_name.split('.')) + ".py" 
            
            if not os.path.exists(tab_file_path):
                 # Log the corrected path for verification
                 self.logger.error(f"Tab module file not found: {tab_file_path}")
                 return

            # Set up environment for the child process
            env = QProcessEnvironment.systemEnvironment()
            # Add project root to PYTHONPATH for the child process
            env.insert("PYTHONPATH", project_root + os.pathsep + env.value("PYTHONPATH", ""))
            self.process.setProcessEnvironment(env)

            # Start the process using python -m module.name
            self.process.start(sys.executable, ['-m', module_name])
            
            if not self.process.waitForStarted():
                self.logger.error(f"Failed to start {self.tab_name} tab process using 'python -m'")
                return
                
            # Subscribe to message bus after process is started
            self.message_bus.subscribe(self.tab_name, self.handle_message)
            self.logger.info(f"{self.tab_name} tab process started (PID: {self.process.processId()}) via 'python -m'")
            
        except Exception as e:
            self.logger.error(f"Error starting {self.tab_name} tab process: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        
    def handle_stdout(self):
        """Handle standard output from the process."""
        data = self.process.readAllStandardOutput().data().decode()
        self.logger.info(f"{self.tab_name} tab output: {data}")
        
    def handle_stderr(self):
        """Handle standard error from the process."""
        data = self.process.readAllStandardError().data().decode()
        self.logger.error(f"{self.tab_name} tab error: {data}")
        
    def handle_finished(self, exit_code: int, exit_status: QProcess.ExitStatus):
        """Handle process completion."""
        self.logger.info(f"{self.tab_name} tab process finished with exit code: {exit_code}")
        
    def handle_message(self, sender: str, message_type: str, data: Any):
        """Handle incoming messages from other tabs."""
        try:
            self.logger.info(f"{self.tab_name} tab received message from {sender}: {message_type}")
            
            # Process message based on type
            if message_type == "data_updated":
                self.on_data_updated(sender, data)
            elif message_type == "analysis_requested":
                self.on_analysis_requested(sender, data)
            elif message_type == "analysis_completed":
                self.on_analysis_completed(sender, data[0], data[1])
            elif message_type == "chart_update":
                self.on_chart_update(sender, data[0], data[1])
            elif message_type == "trading_signal":
                self.on_trading_signal(sender, data[0], data[1])
            elif message_type == "error":
                self.on_error(sender, data)
                
        except Exception as e:
            self.logger.error(f"Error handling message in {self.tab_name} tab: {str(e)}")
            
    def publish_message(self, message_type: str, data: Any):
        """Publish a message to the message bus."""
        self.message_bus.publish(self.tab_name, message_type, data)
        
    # Message handlers - to be overridden by subclasses
    def on_data_updated(self, sender: str, data: Any):
        """Handle data update messages."""
        pass
        
    def on_analysis_requested(self, sender: str, analysis_type: str):
        """Handle analysis request messages."""
        pass
        
    def on_analysis_completed(self, sender: str, analysis_type: str, results: Any):
        """Handle analysis completion messages."""
        pass
        
    def on_chart_update(self, sender: str, chart_type: str, data: Any):
        """Handle chart update messages."""
        pass
        
    def on_trading_signal(self, sender: str, signal_type: str, data: Any):
        """Handle trading signal messages."""
        pass
        
    def on_error(self, sender: str, error_message: str):
        """Handle error messages."""
        self.logger.error(f"Error from {sender}: {error_message}")
        
    def stop_process(self):
        """Stop the tab process."""
        if self.process.state() == QProcess.ProcessState.Running:
            self.process.terminate()
            self.process.waitForFinished()
            self.logger.info(f"{self.tab_name} tab process stopped") 