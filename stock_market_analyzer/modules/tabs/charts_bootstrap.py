import sys
import os
import logging
import traceback
from typing import Any
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import QObject, pyqtSignal

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from stock_market_analyzer.modules.message_bus import MessageBus
from stock_market_analyzer.modules.tabs.charts_tab import ChartsTab

class ChartsProcess(QWidget):
    """Process class for the Charts tab."""
    
    def __init__(self):
        super().__init__()
        self.message_bus = MessageBus()
        self.logger = logging.getLogger(__name__)
        
        # Set up the widget layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        try:
            # Initialize the charts tab
            self.charts_tab = ChartsTab()
            self.layout.addWidget(self.charts_tab)
            
            # Subscribe to message bus
            self.message_bus.subscribe("Charts", self.handle_message)
            
            self.logger.info("Charts tab initialized successfully")
            
        except Exception as e:
            error_msg = f"Error initializing Charts tab: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            
            # Show error message in the UI
            error_label = QLabel(error_msg)
            self.layout.addWidget(error_label)
            
    def handle_message(self, sender: str, message_type: str, data: Any):
        """Handle incoming messages."""
        try:
            if message_type == "error":
                error_msg = f"Received error from {sender}: {data}"
                self.logger.error(error_msg)
            elif message_type == "shutdown":
                self.logger.info(f"Received shutdown request from {sender}")
                self.close()
            elif message_type == "data_request":
                self.handle_data_request(sender, data)
            elif message_type == "analysis_request":
                self.handle_analysis_request(sender, data)
            elif message_type == "heartbeat":
                self.handle_heartbeat(sender, data)
                
        except Exception as e:
            error_log = f"Error handling message in Charts tab: {str(e)}"
            self.logger.error(error_log)
            
    def handle_data_request(self, sender: str, data: Any):
        """Handle data request from other tabs."""
        try:
            request_id = data.get('request_id')
            ticker = data.get('ticker')
            
            if not all([request_id, ticker]):
                self.logger.error("Invalid data request")
                return
                
            # Forward the request to the charts tab
            if hasattr(self, 'charts_tab'):
                self.charts_tab.get_ticker_data(ticker)
                
        except Exception as e:
            self.logger.error(f"Error handling data request: {str(e)}")
            
    def handle_analysis_request(self, sender: str, data: Any):
        """Handle analysis request from other tabs."""
        try:
            request_id = data.get('request_id')
            ticker = data.get('ticker')
            analysis_type = data.get('analysis_type')
            
            if not all([request_id, ticker, analysis_type]):
                self.logger.error("Invalid analysis request")
                return
                
            # Forward the request to the charts tab
            if hasattr(self, 'charts_tab'):
                self.charts_tab.update_chart()
                
        except Exception as e:
            self.logger.error(f"Error handling analysis request: {str(e)}")
            
    def handle_heartbeat(self, sender: str, data: Any):
        """Handle heartbeat message."""
        self.logger.debug(f"Heartbeat from {sender}: {data}")
            
    def closeEvent(self, event):
        """Handle the close event."""
        try:
            # Unsubscribe from message bus
            self.message_bus.unsubscribe("Charts", self.handle_message)
            super().closeEvent(event)
        except Exception as e:
            self.logger.error(f"Error in close event: {str(e)}")
            self.logger.error(traceback.format_exc())

def main():
    """Main function to start the Charts tab process."""
    try:
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create the application
        app = QApplication(sys.argv)
        
        # Create and show the charts process
        process = ChartsProcess()
        process.show()
        
        # Start the event loop
        sys.exit(app.exec())
        
    except Exception as e:
        logging.error(f"Error in Charts tab process: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 