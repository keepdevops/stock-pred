#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stock Market Analyzer Main File
"""
import sys
import os
import argparse
import logging
import signal
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QThreadPool, QTimer
import traceback

# Configure matplotlib font manager logging level to INFO to reduce verbosity
logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)

# Add project root to Python path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Debug prints
print("Python path:", sys.path)
print("Current directory:", os.getcwd())
print("Project root:", project_root)
print("Modules directory:", os.path.join(project_root, "modules"))
print("Database module path:", os.path.join(project_root, "modules", "database.py"))

# Import modules using relative imports
from modules.database import DatabaseConnector
from modules.data_loader import DataLoader
from modules.data_service import DataService
from modules.stock_ai_agent import StockAIAgent
from modules.trading.real_trading_agent import RealTradingAgent
from modules.gui import StockGUI
from modules.message_bus import MessageBus
from modules.tabs.data_tab import DataTab
from modules.tabs.analysis_tab import AnalysisTab
from modules.tabs.charts_tab import ChartsTab
from modules.tabs.models_tab import ModelsTab
from modules.tabs.predictions_tab import PredictionsTab
from modules.tabs.import_tab import ImportTab
from modules.settings import Settings

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

# Initialize logger
logger = setup_logging()

# Global references for cleanup
db_connector = None
gui_instance = None

def signal_handler(sig, frame):
    """Handle termination signals for clean shutdown."""
    try:
        logger.info(f"Received signal {sig}, performing graceful shutdown")
        
        # Clean up GUI if it exists
        global gui_instance
        if gui_instance is not None:
            try:
                gui_instance.close()
            except Exception as e:
                logger.error(f"Error closing GUI: {e}")
        
        # Clean up database if it exists
        global db_connector
        if db_connector is not None:
            try:
                db_connector.close()
            except Exception as e:
                logger.error(f"Error closing database: {e}")
        
        logger.info("Cleanup completed, exiting")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

class StockMarketAnalyzer:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.message_bus = MessageBus()
        self.database = DatabaseConnector()
        self.data_loader = DataLoader(self.database)
        self.data_service = DataService(self.data_loader)
        self.ai_agent = StockAIAgent(self.data_loader)
        self.trading_agent = RealTradingAgent(self.data_loader)
        self.settings = Settings()
        
        # Create and show main window
        self.gui = StockGUI(
            db_connector=self.database,
            data_service=self.data_service,
            ai_agent=self.ai_agent,
            trading_agent=self.trading_agent,
            message_bus=self.message_bus
        )
        self.gui.show()
        
    def run(self):
        """Run the application."""
        try:
            logger.info("Starting application...")
            
            # Start the application event loop
            logger.info("Starting event loop...")
            sys.exit(self.app.exec())
            
        except Exception as e:
            logger.error(f"Error running application: {e}")
            self.cleanup()
            sys.exit(1)

def main():
    """Main entry point."""
    try:
        app = StockMarketAnalyzer()
        app.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 