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
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Debug prints
print("Python path:", sys.path)
print("Current directory:", os.getcwd())
print("Project root:", project_root)
print("Modules directory:", os.path.join(project_root, "modules"))
print("Database module path:", os.path.join(project_root, "modules", "database.py"))

# Import modules using absolute imports
from modules.database import DatabaseConnector
from modules.data_loader import DataLoader
from modules.data_service import DataService
from modules.stock_ai_agent import StockAIAgent
from modules.trading.real_trading_agent import RealTradingAgent
from modules.gui import StockGUI
from modules.message_bus import MessageBus

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    return logger

# Global references for cleanup
db_connector = None
gui_instance = None

def signal_handler(sig, frame):
    """Handle termination signals for clean shutdown."""
    try:
        logger = setup_logging()
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

def main():
    """Main application entry point."""
    try:
        # Set up logging
        logger = setup_logging()
        
        # Print environment info
        logger.info(f"Python path: {sys.path}")
        logger.info(f"Current directory: {os.getcwd()}")
        logger.info(f"Project root: {project_root}")
        logger.info(f"Modules directory: {os.path.join(project_root, 'modules')}")
        logger.info(f"Database module path: {os.path.join(project_root, 'modules', 'database.py')}")
        
        # Initialize components in correct order
        app = QApplication(sys.argv)
        
        # First initialize database
        global db_connector
        db_connector = DatabaseConnector()
        logger.info("Database connector initialized")
        
        # Then initialize data components
        data_loader = DataLoader(db_connector)
        logger.info("Data loader initialized")
        
        data_service = DataService(data_loader)
        logger.info("Data service initialized")
        
        # Initialize AI and trading components with data loader
        ai_agent = StockAIAgent(data_loader)
        logger.info("AI agent initialized")
        
        trading_agent = RealTradingAgent(data_loader)
        logger.info("Trading agent initialized")
        
        # Initialize message bus
        message_bus = MessageBus()
        logger.info("Message bus initialized")
        
        # Create and show GUI
        global gui_instance
        gui_instance = StockGUI(db_connector, data_service, ai_agent, trading_agent, message_bus)
        gui_instance.show()
        logger.info("GUI initialized and shown")
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start application event loop
        sys.exit(app.exec())
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main() 