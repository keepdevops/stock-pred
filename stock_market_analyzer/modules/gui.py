from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QListWidget, QCheckBox, QMessageBox, QTabWidget
)
from PyQt6.QtCore import Qt, QDateTime
from .tab_process import TabProcess
import pandas as pd
import logging
import traceback

class StockGUI(QMainWindow):
    def __init__(self, db_connector, data_service, ai_agent, trading_agent):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Store references to services
        self.db_connector = db_connector
        self.data_service = data_service
        self.ai_agent = ai_agent
        self.trading_agent = trading_agent
        
        # Initialize GUI
        self.setup_gui()
        
    def setup_gui(self):
        """Set up the main GUI components."""
        try:
            self.logger.info("Setting up GUI components")
            
            # Set up the main window
            self.setWindowTitle("Stock Market Analyzer")
            self.setGeometry(100, 100, 1200, 800)
            
            # Create central widget and main layout
            self.central_widget = QWidget()
            self.setCentralWidget(self.central_widget)
            self.main_layout = QVBoxLayout(self.central_widget)
            
            # Create tab widget
            self.tab_widget = QTabWidget()
            self.main_layout.addWidget(self.tab_widget)
            
            # Create and add tab processes
            self.tabs = {
                "Data": TabProcess("Data", self),
                "Analysis": TabProcess("Analysis", self),
                "Models": TabProcess("Models", self),
                "Predictions": TabProcess("Predictions", self),
                "Charts": TabProcess("Charts", self),
                "Trading": TabProcess("Trading", self),
                "Settings": TabProcess("Settings", self),
                "Help": TabProcess("Help", self)
            }
            
            # Add tabs to the tab widget
            for name, tab in self.tabs.items():
                self.tab_widget.addTab(tab, name)
                tab.start_process()
            
            self.logger.info("GUI setup completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error setting up GUI: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
            
    def closeEvent(self, event):
        """Handle window close event."""
        try:
            # Stop all tab processes
            for tab in self.tabs.values():
                tab.stop_process()
            event.accept()
        except Exception as e:
            self.logger.error(f"Error during close: {str(e)}")
            event.accept() 