import sys
import os
import logging
import traceback
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTabWidget, QSplitter,
    QTableWidget, QTableWidgetItem, QTextEdit,
    QFileDialog, QMessageBox, QComboBox, QSpinBox,
    QDoubleSpinBox, QCheckBox, QLineEdit, QGroupBox,
    QFormLayout, QScrollArea, QFrame, QSizePolicy,
    QHeaderView, QApplication
)
from PyQt6.QtCore import Qt, QTimer, QDateTime, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QColor, QIntValidator, QIcon

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from modules.tab_process import TabProcess
from modules.message_bus import MessageBus
from modules.database import DatabaseConnector
from modules.data_service import DataService
from modules.stock_ai_agent import StockAIAgent
from modules.trading.real_trading_agent import RealTradingAgent

class StockGUI(QMainWindow):
    """Main GUI window for the stock market analyzer."""
    
    def __init__(self, db_connector: DatabaseConnector, data_service: DataService, 
                 ai_agent: StockAIAgent, trading_agent: RealTradingAgent, 
                 message_bus: MessageBus):
        super().__init__()
        self.db_connector = db_connector
        self.data_service = data_service
        self.ai_agent = ai_agent
        self.trading_agent = trading_agent
        self.message_bus = message_bus
        self.logger = logging.getLogger(__name__)
        self.tabs = {}
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the GUI components."""
        try:
            # Set window properties
            self.setWindowTitle("Stock Market Analyzer")
            self.setGeometry(100, 100, 1200, 800)
            
            # Create central widget and layout
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            layout = QVBoxLayout(central_widget)
            
            # Create tab widget
            self.tab_widget = QTabWidget()
            layout.addWidget(self.tab_widget)
            
            # Define tabs
            self.tabs = {
                "Data": TabProcess("Data"),
                "Analysis": TabProcess("Analysis"),
                "Models": TabProcess("Models"),
                "Predictions": TabProcess("Predictions"),
                "Charts": TabProcess("Charts"),
                "Trading": TabProcess("Trading"),
                "Import": TabProcess("Import"),
                "Settings": TabProcess("Settings"),
                "Help": TabProcess("Help")
            }
            
            # Add tabs to tab widget
            for name, tab in self.tabs.items():
                self.tab_widget.addTab(tab, name)
                tab.start_process()
                
            # Setup status bar
            self.statusBar().showMessage("Ready")
            
            # Connect tab change signal
            self.tab_widget.currentChanged.connect(self.handle_tab_change)
            
            self.logger.info("GUI setup completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error setting up GUI: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
            
    def handle_tab_change(self, index: int):
        """Handle tab change events."""
        try:
            tab_name = self.tab_widget.tabText(index)
            self.statusBar().showMessage(f"Switched to {tab_name} tab")
            
        except Exception as e:
            self.logger.error(f"Error handling tab change: {str(e)}")
            
    def show_status_message(self, message: str, timeout: int = 3000):
        """Show a message in the status bar."""
        try:
            self.statusBar().showMessage(message, timeout)
        except Exception as e:
            self.logger.error(f"Error showing status message: {str(e)}")
            
    def closeEvent(self, event):
        """Handle the close event."""
        try:
            # Stop all tab processes
            for tab in self.tabs.values():
                tab.stop_process()
                
            # Close database connection
            self.db_connector.close()
            
            super().closeEvent(event)
            
        except Exception as e:
            self.logger.error(f"Error in close event: {str(e)}")
            self.logger.error(traceback.format_exc()) 