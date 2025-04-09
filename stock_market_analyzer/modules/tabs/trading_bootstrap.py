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
from stock_market_analyzer.modules.tabs.trading_tab import TradingTab

class TradingProcess(QWidget):
    """Process class for the Trading tab."""
    
    def __init__(self):
        super().__init__()
        self.message_bus = MessageBus()
        self.logger = logging.getLogger(__name__)
        
        # Set up the widget layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        try:
            # Initialize the trading tab
            self.trading_tab = TradingTab()
            self.layout.addWidget(self.trading_tab)
            
            # Subscribe to message bus
            self.message_bus.subscribe("Trading", self.handle_message)
            
            self.logger.info("Trading tab initialized successfully")
            
        except Exception as e:
            error_msg = f"Error initializing Trading tab: {str(e)}"
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
        except Exception as e:
            error_log = f"Error handling message in Trading tab: {str(e)}"
            self.logger.error(error_log)
            
    def closeEvent(self, event):
        """Handle the close event."""
        try:
            # Unsubscribe from message bus
            self.message_bus.unsubscribe("Trading", self.handle_message)
            super().closeEvent(event)
        except Exception as e:
            self.logger.error(f"Error in close event: {str(e)}")
            self.logger.error(traceback.format_exc())

def main():
    """Main function to start the Trading tab process."""
    try:
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create the application
        app = QApplication(sys.argv)
        
        # Create and show the trading process
        process = TradingProcess()
        process.show()
        
        # Start the event loop
        sys.exit(app.exec())
        
    except Exception as e:
        logging.error(f"Error in Trading tab process: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 