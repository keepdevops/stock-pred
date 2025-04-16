import sys
import os
import logging
import traceback
from PyQt6.QtCore import QProcess, QSharedMemory
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import Qt
from typing import Any

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from stock_market_analyzer.modules.tabs.import_tab import ImportTab
from stock_market_analyzer.modules.message_bus import MessageBus

class ImportProcess(QWidget):
    """Process wrapper for the Import tab."""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.message_bus = MessageBus()
        self.import_tab = None
        
        # Set up the widget layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # Initialize the import tab
        self.init_import_tab()
        
    def init_import_tab(self):
        """Initialize the import tab."""
        try:
            self.import_tab = ImportTab()
            self.layout.addWidget(self.import_tab)
            
            # Subscribe to message bus
            self.message_bus.subscribe("Import", self.handle_message)
            
        except Exception as e:
            error_msg = f"Failed to initialize Import tab: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            
            # Show error message
            error_label = QLabel(error_msg)
            self.layout.addWidget(error_label)
            
    def handle_message(self, sender: str, message_type: str, data: Any):
        """Handle incoming messages."""
        try:
            if message_type == "error":
                error_msg = f"Received error from {sender}: {data}"
                self.logger.error(error_msg)
        except Exception as e:
            error_log = f"Error handling message in Import tab: {str(e)}"
            self.logger.error(error_log)

def main():
    """Main function for the import tab process."""
    app = QApplication.instance() 
    if not app: 
        app = QApplication(sys.argv)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("Starting import tab process")
    
    try:
        window = ImportProcess()
        window.setWindowTitle("Import Tab")
        window.show()
    except Exception as e:
         logger.error(f"Failed to create or show ImportProcess window: {e}")
         logger.error(traceback.format_exc())
         sys.exit(1)

    if __name__ == "__main__":
        sys.exit(app.exec())

if __name__ == "__main__":
    main() 