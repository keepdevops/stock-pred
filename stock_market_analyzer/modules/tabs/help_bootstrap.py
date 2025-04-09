import sys
import os
import logging
import traceback
from typing import Any
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import QProcess, QProcessEnvironment, QTimer

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from stock_market_analyzer.modules.message_bus import MessageBus
from stock_market_analyzer.modules.tabs.help_tab import HelpTab

class HelpProcess(QWidget):
    """Process class for the Help tab."""
    
    def __init__(self):
        super().__init__()
        self.message_bus = MessageBus()
        self.logger = logging.getLogger(__name__)
        self.process = None
        self.heartbeat_timer = QTimer()
        self.heartbeat_timer.timeout.connect(self.send_heartbeat)
        self.heartbeat_timer.start(5000)  # Send heartbeat every 5 seconds
        
        # Set up the widget layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        try:
            # Initialize the help tab
            self.help_tab = HelpTab()
            self.layout.addWidget(self.help_tab)
            
            # Subscribe to message bus
            self.message_bus.subscribe("Help", self.handle_message)
            
            self.logger.info("Help tab initialized successfully")
            
        except Exception as e:
            error_msg = f"Error initializing Help tab: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            
            # Show error message in the UI
            error_label = QLabel(error_msg)
            self.layout.addWidget(error_label)
            
    def send_heartbeat(self):
        """Send heartbeat message to indicate process is alive."""
        try:
            self.message_bus.publish("Help", "heartbeat", {"status": "alive"})
        except Exception as e:
            self.logger.error(f"Error sending heartbeat: {str(e)}")
            
    def start_process(self):
        """Start the help process."""
        try:
            if self.process is None:
                self.process = QProcess()
                self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
                
                # Set up environment variables
                env = QProcessEnvironment.systemEnvironment()
                env.insert("PYTHONPATH", project_root)
                self.process.setProcessEnvironment(env)
                
                # Set up process signals
                self.process.readyReadStandardOutput.connect(self.handle_process_output)
                self.process.finished.connect(self.handle_process_finished)
                self.process.errorOccurred.connect(self.handle_process_error)
                
                # Start the process
                bootstrap_script = os.path.join(project_root, "stock_market_analyzer", "modules", "tabs", "help_bootstrap.py")
                self.process.start(sys.executable, [bootstrap_script])
                
                self.logger.info("Help process started")
                
        except Exception as e:
            error_msg = f"Error starting help process: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            
    def stop_process(self):
        """Stop the help process."""
        try:
            if self.process and self.process.state() == QProcess.ProcessState.Running:
                # Stop heartbeat timer
                self.heartbeat_timer.stop()
                
                # Send shutdown message
                self.message_bus.publish("Help", "shutdown", {"reason": "normal"})
                
                # Terminate process
                self.process.terminate()
                self.process.waitForFinished(5000)  # Wait up to 5 seconds
                if self.process.state() == QProcess.ProcessState.Running:
                    self.process.kill()
                self.process = None
                self.logger.info("Help process stopped")
                
        except Exception as e:
            error_msg = f"Error stopping help process: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            
    def handle_process_output(self):
        """Handle process output."""
        try:
            output = self.process.readAllStandardOutput().data().decode()
            if output:
                self.logger.info(f"Help process output: {output}")
                
        except Exception as e:
            self.logger.error(f"Error handling process output: {str(e)}")
            
    def handle_process_error(self, error):
        """Handle process errors."""
        try:
            error_msg = f"Help process error: {error}"
            self.logger.error(error_msg)
            self.message_bus.publish("Help", "error", error_msg)
            
        except Exception as e:
            self.logger.error(f"Error handling process error: {str(e)}")
            
    def handle_process_finished(self, exit_code, exit_status):
        """Handle process finished signal."""
        try:
            if exit_status == QProcess.ExitStatus.CrashExit:
                self.logger.error("Help process crashed")
                self.message_bus.publish("Help", "error", "Process crashed")
            else:
                self.logger.info(f"Help process finished with exit code {exit_code}")
                
        except Exception as e:
            self.logger.error(f"Error handling process finished: {str(e)}")
            
    def handle_message(self, sender: str, message_type: str, data: Any):
        """Handle incoming messages."""
        try:
            if message_type == "error":
                error_msg = f"Received error from {sender}: {data}"
                self.logger.error(error_msg)
            elif message_type == "shutdown":
                self.logger.info(f"Received shutdown request from {sender}")
                self.stop_process()
                
        except Exception as e:
            error_log = f"Error handling message in Help tab: {str(e)}"
            self.logger.error(error_log)
            
    def closeEvent(self, event):
        """Handle the close event."""
        try:
            # Stop the process
            self.stop_process()
            
            # Unsubscribe from message bus
            self.message_bus.unsubscribe("Help", self.handle_message)
            
            super().closeEvent(event)
            
        except Exception as e:
            self.logger.error(f"Error in close event: {str(e)}")
            self.logger.error(traceback.format_exc())

def main():
    """Main function to start the Help tab process."""
    try:
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create the application
        app = QApplication(sys.argv)
        
        # Create and show the help process
        process = HelpProcess()
        process.show()
        
        # Start the process
        process.start_process()
        
        # Start the event loop
        sys.exit(app.exec())
        
    except Exception as e:
        logging.error(f"Error in Help tab process: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 