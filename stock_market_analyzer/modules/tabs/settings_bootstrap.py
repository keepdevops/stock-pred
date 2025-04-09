import sys
import os
import logging
import traceback
from typing import Any
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import QProcess, QProcessEnvironment

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from stock_market_analyzer.modules.message_bus import MessageBus
from stock_market_analyzer.modules.tabs.settings_tab import SettingsTab

class SettingsProcess(QWidget):
    """Process class for the Settings tab."""
    
    def __init__(self):
        super().__init__()
        self.message_bus = MessageBus()
        self.logger = logging.getLogger(__name__)
        self.process = None
        
        # Set up the widget layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        try:
            # Initialize the settings tab
            self.settings_tab = SettingsTab()
            self.layout.addWidget(self.settings_tab)
            
            # Subscribe to message bus
            self.message_bus.subscribe("Settings", self.handle_message)
            
            self.logger.info("Settings tab initialized successfully")
            
        except Exception as e:
            error_msg = f"Error initializing Settings tab: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            
            # Show error message in the UI
            error_label = QLabel(error_msg)
            self.layout.addWidget(error_label)
            
    def start_process(self):
        """Start the settings process."""
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
                
                # Start the process
                bootstrap_script = os.path.join(project_root, "stock_market_analyzer", "modules", "tabs", "settings_bootstrap.py")
                self.process.start(sys.executable, [bootstrap_script])
                
                self.logger.info("Settings process started")
                
        except Exception as e:
            error_msg = f"Error starting settings process: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            
    def stop_process(self):
        """Stop the settings process."""
        try:
            if self.process and self.process.state() == QProcess.ProcessState.Running:
                self.process.terminate()
                self.process.waitForFinished(5000)  # Wait up to 5 seconds
                if self.process.state() == QProcess.ProcessState.Running:
                    self.process.kill()
                self.process = None
                self.logger.info("Settings process stopped")
                
        except Exception as e:
            error_msg = f"Error stopping settings process: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            
    def handle_process_output(self):
        """Handle process output."""
        try:
            output = self.process.readAllStandardOutput().data().decode()
            if output:
                self.logger.info(f"Settings process output: {output}")
                
        except Exception as e:
            self.logger.error(f"Error handling process output: {str(e)}")
            
    def handle_process_finished(self, exit_code, exit_status):
        """Handle process finished signal."""
        try:
            if exit_status == QProcess.ExitStatus.CrashExit:
                self.logger.error("Settings process crashed")
            else:
                self.logger.info(f"Settings process finished with exit code {exit_code}")
                
        except Exception as e:
            self.logger.error(f"Error handling process finished: {str(e)}")
            
    def handle_message(self, sender: str, message_type: str, data: Any):
        """Handle incoming messages."""
        try:
            if message_type == "error":
                error_msg = f"Received error from {sender}: {data}"
                self.logger.error(error_msg)
        except Exception as e:
            error_log = f"Error handling message in Settings tab: {str(e)}"
            self.logger.error(error_log)
            
    def closeEvent(self, event):
        """Handle the close event."""
        try:
            # Stop the process
            self.stop_process()
            
            # Unsubscribe from message bus
            self.message_bus.unsubscribe("Settings", self.handle_message)
            
            super().closeEvent(event)
            
        except Exception as e:
            self.logger.error(f"Error in close event: {str(e)}")
            self.logger.error(traceback.format_exc())

def main():
    """Main function to start the Settings tab process."""
    try:
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create the application
        app = QApplication(sys.argv)
        
        # Create and show the settings process
        process = SettingsProcess()
        process.show()
        
        # Start the process
        process.start_process()
        
        # Start the event loop
        sys.exit(app.exec())
        
    except Exception as e:
        logging.error(f"Error in Settings tab process: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 