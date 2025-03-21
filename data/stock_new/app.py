import sys
import logging
from PyQt5.QtWidgets import QApplication
from src.gui.stock_gui import StockGUI
import duckdb
import os
from datetime import datetime
import json
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QPushButton, QLabel, QProgressBar, QTableWidget, 
                           QGroupBox, QDialog, QCalendarWidget, QDialogButtonBox,
                           QMessageBox, QTableWidgetItem, QTextEdit, QListWidget,
                           QComboBox, QCheckBox)
from PyQt5.QtCore import Qt, QDate, QThread, pyqtSignal
from PyQt5.QtGui import QColor

def setup_logging():
    """Set up logging configuration."""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create log filename with timestamp
    log_filename = f'logs/stock_market_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_filename)
        ]
    )

def setup_database():
    """Initialize DuckDB database and required tables."""
    try:
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        os.makedirs('data/db', exist_ok=True)
        
        # Connect to DuckDB
        conn = duckdb.connect('data/db/stock_market.db')
        
        # Create tables if they don't exist
        conn.execute("""
            CREATE TABLE IF NOT EXISTS stock_data (
                ticker VARCHAR,
                date DATE,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume BIGINT,
                adj_close DOUBLE,
                PRIMARY KEY (ticker, date)
            );
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ticker_metadata (
                ticker VARCHAR PRIMARY KEY,
                name VARCHAR,
                sector VARCHAR,
                industry VARCHAR,
                last_updated TIMESTAMP
            );
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS technical_indicators (
                ticker VARCHAR,
                date DATE,
                sma_20 DOUBLE,
                sma_50 DOUBLE,
                sma_200 DOUBLE,
                rsi_14 DOUBLE,
                macd DOUBLE,
                macd_signal DOUBLE,
                macd_hist DOUBLE,
                PRIMARY KEY (ticker, date)
            );
        """)
        
        conn.close()
        logging.info("Database setup completed successfully")
        
    except Exception as e:
        logging.error(f"Error setting up database: {str(e)}")
        raise

def main():
    """Main application entry point."""
    try:
        # Set up logging
        setup_logging()
        logging.info("Starting Stock Market Application")
        
        # Set up database
        setup_database()
        
        # Create Qt application
        app = QApplication(sys.argv)
        
        # Set application style
        app.setStyle('Fusion')
        
        # Create and show the main window
        gui = StockGUI()
        gui.show()
        
        # Start the event loop
        sys.exit(app.exec_())
        
    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 