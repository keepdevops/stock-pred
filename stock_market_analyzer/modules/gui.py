import os
import sys
import logging
import traceback
from typing import Dict, Any, Optional
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

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import modules using relative imports
from .database import DatabaseConnector
from .data_service import DataService
from .stock_ai_agent import StockAIAgent
from .trading.real_trading_agent import RealTradingAgent
from .message_bus import MessageBus
from .tab_process import TabProcess
from .tabs.data_tab import DataTab
from .tabs.analysis_tab import AnalysisTab
from .tabs.charts_tab import ChartsTab
from .tabs.models_tab import ModelsTab
from .tabs.predictions_tab import PredictionsTab
from .tabs.import_tab import ImportTab
from .tabs.settings_tab import SettingsTab
from .tabs.help_tab import HelpTab
from .tabs.trading_tab import TradingTab

class StockGUI(QMainWindow):
    """Main GUI window for the Stock Market Analyzer."""
    
    def __init__(
        self,
        db_connector: DatabaseConnector,
        data_service: DataService,
        ai_agent: StockAIAgent,
        trading_agent: RealTradingAgent,
        message_bus: MessageBus,
        parent=None
    ):
        """Initialize the GUI.
        
        Args:
            db_connector: Database connection manager.
            data_service: Data service for market data.
            ai_agent: AI agent for predictions.
            trading_agent: Trading agent for real trading.
            message_bus: Message bus for communication.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.db_connector = db_connector
        self.data_service = data_service
        self.ai_agent = ai_agent
        self.trading_agent = trading_agent
        self.message_bus = message_bus
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize UI
        self.setWindowTitle("Stock Market Analyzer")
        self.setMinimumSize(1200, 800)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Initialize tabs
        self.tabs = {}
        self.setup_tabs()
        
        # Setup message bus subscriptions
        self.setup_message_bus()
        
        # Setup status bar
        self.setup_status_bar()
        
    def setup_tabs(self):
        """Set up all tabs in the application."""
        try:
            # Create tab instances
            self.tabs = {
                "Data": TabProcess(DataTab, self.message_bus),
                "Analysis": TabProcess(AnalysisTab, self.message_bus),
                "Charts": TabProcess(ChartsTab, self.message_bus),
                "Models": TabProcess(ModelsTab, self.message_bus),
                "Predictions": TabProcess(PredictionsTab, self.message_bus),
                "Import": TabProcess(ImportTab, self.message_bus),
                "Settings": TabProcess(SettingsTab, self.message_bus),
                "Help": TabProcess(HelpTab, self.message_bus),
                "Trading": TabProcess(TradingTab, self.message_bus)
            }
            
            # Add tabs to tab widget
            for name, tab in self.tabs.items():
                self.tab_widget.addTab(tab, name)
                
            # Start all tab processes
            for tab in self.tabs.values():
                tab.start_process()
                
        except Exception as e:
            self.logger.error(f"Error setting up tabs: {str(e)}")
            self.logger.error(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Failed to initialize tabs: {str(e)}")
            
    def setup_message_bus(self):
        """Set up message bus subscriptions."""
        try:
            self.message_bus.subscribe("all", self.handle_message)
        except Exception as e:
            self.logger.error(f"Error setting up message bus: {str(e)}")
            
    def setup_status_bar(self):
        """Set up the status bar."""
        self.statusBar().showMessage("Ready")
        
    def handle_message(self, sender: str, message_type: str, data: Dict[str, Any]):
        """Handle incoming messages.
        
        Args:
            sender: The sender of the message.
            message_type: The type of message.
            data: The message data.
        """
        try:
            if message_type == "error":
                self.statusBar().showMessage(f"Error from {sender}: {data.get('error', 'Unknown error')}")
            elif message_type == "status":
                self.statusBar().showMessage(f"{sender}: {data.get('status', '')}")
        except Exception as e:
            self.logger.error(f"Error handling message: {str(e)}")
            
    def closeEvent(self, event):
        """Handle window close event."""
        try:
            # Clean up all tabs
            for tab in self.tabs.values():
                tab.cleanup()
                
            # Clean up other resources
            if self.db_connector:
                self.db_connector.close()
                
            event.accept()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            event.accept()  # Still accept the close event 