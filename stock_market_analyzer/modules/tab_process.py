import sys
import os
import logging
import traceback
from typing import Any, Optional
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import QProcess, QSharedMemory, QTimer, Qt, QProcessEnvironment
from PyQt6.QtGui import QFont

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from modules.message_bus import MessageBus

class TabProcess(QWidget):
    """Class to manage a tab process."""
    
    def __init__(self, tab_name: str):
        super().__init__()
        self.tab_name = tab_name
        self.process: Optional[QProcess] = None
        self.message_bus = MessageBus()
        self.logger = logging.getLogger(__name__)
        self.cleanup_timer = QTimer()
        self.cleanup_timer.timeout.connect(self.check_process)
        self.cleanup_timer.start(1000)  # Check process every second
        self._is_stopping = False  # Flag to prevent multiple stop attempts
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the UI for the tab process."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Loading label
        self.loading_label = QLabel(f"Loading {self.tab_name}...")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_label.setFont(QFont("Arial", 12))
        layout.addWidget(self.loading_label)
        
    def start_process(self):
        """Start the tab process."""
        try:
            if self.process is not None:
                self.logger.warning(f"Process for {self.tab_name} is already running")
                return
                
            self.process = QProcess()
            self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
            
            # Set up environment
            env = QProcessEnvironment.systemEnvironment()
            env.insert("PYTHONPATH", os.pathsep.join(sys.path))
            self.process.setProcessEnvironment(env)
            
            # Determine the bootstrap script path
            bootstrap_script = os.path.join(
                os.path.dirname(__file__),
                "tabs",
                f"{self.tab_name.lower()}_bootstrap.py"
            )
            
            if not os.path.exists(bootstrap_script):
                raise FileNotFoundError(f"Bootstrap script not found: {bootstrap_script}")
                
            # Start the process
            self.process.start(sys.executable, [bootstrap_script])
            
            if not self.process.waitForStarted(5000):
                raise RuntimeError(f"Failed to start {self.tab_name} process")
                
            self.logger.info(f"Started {self.tab_name} process with PID {self.process.processId()}")
            
            # Subscribe to message bus
            self.message_bus.subscribe(self.tab_name, self.handle_message)
            
            # Connect process signals
            self.process.finished.connect(self.handle_process_finished)
            self.process.errorOccurred.connect(self.handle_process_error)
            self.process.readyReadStandardOutput.connect(self.handle_process_output)
            
        except Exception as e:
            self.logger.error(f"Error starting {self.tab_name} process: {str(e)}")
            self.logger.error(traceback.format_exc())
            if self.process:
                self.process.kill()
                self.process = None
            raise
            
    def stop_process(self):
        """Stop the process."""
        try:
            # Prevent multiple stop attempts
            if self._is_stopping:
                self.logger.debug(f"Process {self.tab_name} is already being stopped")
                return
                
            self._is_stopping = True
            
            # Store a local reference to the process
            process = self.process
            if process is None:
                self.logger.warning(f"Process {self.tab_name} is already stopped or not initialized")
                self._is_stopping = False
                return
                
            self.logger.info(f"Stopping {self.tab_name} process...")
            
            # First try to send shutdown message
            self.message_bus.publish(self.tab_name, "shutdown", {})
            
            # Wait a bit for graceful shutdown
            if not process.waitForFinished(2000):
                # If graceful shutdown fails, try terminate
                process.terminate()
                if not process.waitForFinished(2000):
                    # If terminate fails, force kill
                    self.logger.warning(f"Process {self.tab_name} did not terminate gracefully, forcing kill")
                    process.kill()
                    process.waitForFinished(1000)
                
            # Set process to None after successful shutdown
            self.process = None
            self.logger.info(f"{self.tab_name} process stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping {self.tab_name} process: {str(e)}")
            self.logger.error(traceback.format_exc())
            # Ensure process is set to None even if there's an error
            self.process = None
        finally:
            self._is_stopping = False
            
    def check_process(self):
        """Check if the process is still running and handle any issues."""
        if self.process is None or self._is_stopping:
            return
            
        if self.process.state() == QProcess.ProcessState.NotRunning:
            self.logger.warning(f"Process for {self.tab_name} is not running")
            self.process = None
            
    def handle_process_finished(self, exit_code: int, exit_status: QProcess.ExitStatus):
        """Handle process finished signal."""
        try:
            self.logger.info(f"Process for {self.tab_name} finished with exit code {exit_code}")
            # Only stop if we're not already stopping
            if not self._is_stopping:
                self.stop_process()
        except Exception as e:
            self.logger.error(f"Error handling process finished: {str(e)}")
            self.logger.error(traceback.format_exc())
            
    def handle_process_error(self, error: QProcess.ProcessError):
        """Handle process error signal."""
        try:
            error_msg = f"Process error for {self.tab_name}: {error}"
            self.logger.error(error_msg)
            # Only publish to message bus if it's not already an error message
            if not isinstance(error, str):
                self.message_bus.publish(self.tab_name, "error", error_msg)
            # Only stop if we're not already stopping
            if not self._is_stopping:
                self.stop_process()
        except Exception as e:
            self.logger.error(f"Error handling process error: {str(e)}")
            self.logger.error(traceback.format_exc())
            
    def handle_process_output(self):
        """Handle process output."""
        if self.process:
            try:
                output = self.process.readAllStandardOutput().data().decode()
                if output.strip():
                    print(f"{self.tab_name} process output: {output}")
            except Exception as e:
                print(f"Error handling process output: {str(e)}")
            
    def handle_message(self, sender: str, message_type: str, data: Any):
        """Handle incoming messages."""
        try:
            if message_type == "error":
                self.logger.error(f"Received error from {sender}: {data}")
            elif message_type == "heartbeat":
                self.loading_label.setText(f"{self.tab_name} is running...")
            elif message_type == "shutdown":
                self.logger.info(f"Received shutdown request from {sender}")
                self.stop_process()
            else:
                self.logger.debug(f"Received message from {sender}: {message_type} - {data}")
                
        except Exception as e:
            self.logger.error(f"Error handling message: {str(e)}")
            self.logger.error(traceback.format_exc())
            
    def publish_message(self, message_type: str, data: Any):
        """Publish a message to the message bus."""
        try:
            self.message_bus.publish(self.tab_name, message_type, data)
        except Exception as e:
            self.logger.error(f"Error publishing message: {str(e)}")
            
    def closeEvent(self, event):
        """Handle the close event."""
        try:
            # Stop cleanup timer
            self.cleanup_timer.stop()
            
            # Stop the process
            self.stop_process()
            
            # Unsubscribe from message bus
            self.message_bus.unsubscribe(self.tab_name, self.handle_message)
            
            super().closeEvent(event)
            
        except Exception as e:
            self.logger.error(f"Error in close event: {str(e)}")
            self.logger.error(traceback.format_exc()) 