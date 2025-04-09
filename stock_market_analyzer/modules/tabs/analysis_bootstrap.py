import sys
import os
import logging
import traceback
from typing import Any
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import QObject, pyqtSignal, Qt, QTimer

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from stock_market_analyzer.modules.message_bus import MessageBus
from stock_market_analyzer.modules.tabs.analysis_tab import AnalysisTab

class AnalysisProcess(QWidget):
    """Class to manage the Analysis tab process."""
    
    def __init__(self):
        super().__init__()
        self.message_bus = MessageBus()
        self.logger = logging.getLogger(__name__)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the UI for the Analysis tab."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Loading label
        self.loading_label = QLabel("Loading Analysis tab...")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.loading_label)
        
        # Initialize the Analysis tab
        try:
            self.analysis_tab = AnalysisTab()
            layout.addWidget(self.analysis_tab)
            self.loading_label.hide()
            
            # Subscribe to message bus
            self.message_bus.subscribe("Analysis", self.handle_message)
            
            # Start heartbeat timer
            self.heartbeat_timer = QTimer()
            self.heartbeat_timer.timeout.connect(self.send_heartbeat)
            self.heartbeat_timer.start(5000)  # Send heartbeat every 5 seconds
            
        except Exception as e:
            self.logger.error(f"Error initializing Analysis tab: {e}")
            self.logger.error(traceback.format_exc())
            
    def handle_message(self, sender: str, message_type: str, data: Any):
        """Handle incoming messages."""
        try:
            if message_type == "error":
                self.logger.error(f"Received error from {sender}: {data}")
            elif message_type == "shutdown":
                self.logger.info(f"Received shutdown request from {sender}")
                self.close()
                
        except Exception as e:
            self.logger.error(f"Error handling message: {str(e)}")
            
    def send_heartbeat(self):
        """Send heartbeat message to indicate the process is alive."""
        try:
            self.message_bus.publish("Analysis", "heartbeat", {})
        except Exception as e:
            self.logger.error(f"Error sending heartbeat: {str(e)}")
            
    def closeEvent(self, event):
        """Handle the close event."""
        try:
            # Stop heartbeat timer
            self.heartbeat_timer.stop()
            
            # Unsubscribe from message bus
            self.message_bus.unsubscribe("Analysis", self.handle_message)
            
            super().closeEvent(event)
            
        except Exception as e:
            self.logger.error(f"Error in close event: {str(e)}")
            self.logger.error(traceback.format_exc())

def main():
    """Main function for the Analysis tab process."""
    try:
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create application
        app = QApplication(sys.argv)
        
        # Create and show the Analysis process
        process = AnalysisProcess()
        process.show()
        
        # Start the event loop
        sys.exit(app.exec())
        
    except Exception as e:
        logging.error(f"Error in Analysis process: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 