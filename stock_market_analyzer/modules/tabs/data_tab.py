import sys
import os
import logging
import traceback
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QCheckBox, QMessageBox,
    QFileDialog, QComboBox, QProgressBar, QListWidget,
    QListWidgetItem, QSplitter, QDateEdit
)
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QTimer, QDate
from PyQt6.QtGui import QFont
import pandas as pd
import yfinance as yf
import json
import time

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ..message_bus import MessageBus
from ..data_stock import DataStock
from ..database import DatabaseConnector
from .base_tab import BaseTab

class DataTab(BaseTab):
    """Data tab for managing stock data."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.message_bus = MessageBus()
        self.logger = logging.getLogger(__name__)
        self.data_stock = DataStock()
        self.db_connector = DatabaseConnector()
        self.setup_ui()
        self.ticker_list = []
        self.data_cache = {}
        self.pending_requests = {}
        self.current_color_scheme = "default"  # Add default color scheme
        self.setup_theme()  # Initialize theme
        
    def setup_ui(self):
        """Setup the data tab UI."""
        # Market type selection
        market_layout = QHBoxLayout()
        market_layout.addWidget(QLabel("Market Type:"))
        self.market_combo = QComboBox()
        self.market_combo.addItems(["US", "HK", "CN"])
        market_layout.addWidget(self.market_combo)
        self.layout.addLayout(market_layout)
        
        # Date range selection
        date_layout = QHBoxLayout()
        date_layout.addWidget(QLabel("Date Range:"))
        
        # Start date
        date_layout.addWidget(QLabel("From:"))
        self.start_date = QDateEdit()
        self.start_date.setCalendarPopup(True)
        self.start_date.setDate(QDate.currentDate().addYears(-1))  # Default to 1 year ago
        date_layout.addWidget(self.start_date)
        
        # End date
        date_layout.addWidget(QLabel("To:"))
        self.end_date = QDateEdit()
        self.end_date.setCalendarPopup(True)
        self.end_date.setDate(QDate.currentDate())  # Default to today
        date_layout.addWidget(self.end_date)
        
        self.layout.addLayout(date_layout)
        
        # Ticker input
        ticker_layout = QHBoxLayout()
        ticker_layout.addWidget(QLabel("Ticker:"))
        self.ticker_edit = QLineEdit()
        ticker_layout.addWidget(self.ticker_edit)
        
        add_button = QPushButton("Add")
        add_button.clicked.connect(self.add_ticker)
        ticker_layout.addWidget(add_button)
        
        self.layout.addLayout(ticker_layout)
        
        # Data table
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "Ticker", "Market", "Open", "High", "Low", "Close", "Volume"
        ])
        self.layout.addWidget(self.table)
        
    def handle_message(self, sender: str, message_type: str, data: dict):
        """Handle incoming messages from the message bus.
        
        Args:
            sender: The sender of the message
            message_type: The type of message
            data: The message data
        """
        try:
            if message_type == "heartbeat":
                self.status_label.setText("Connected")
            elif message_type == "data_request":
                self.handle_data_request(sender, data)
        except Exception as e:
            logging.error(f"Error handling message in DataTab: {str(e)}")
            self.status_label.setText(f"Error: {str(e)}")
            
    def handle_data_request(self, sender: str, data: Any):
        """Handle data request from another tab."""
        try:
            ticker = data.get("ticker")
            if ticker in self.data_cache:
                self.message_bus.publish(
                    self.__class__.__name__,
                    "data_response",
                    {
                        "ticker": ticker,
                        "data": self.data_cache[ticker]
                    }
                )
            else:
                self.message_bus.publish(
                    self.__class__.__name__,
                    "error",
                    f"Data not found for ticker: {ticker}"
                )
        except Exception as e:
            self.logger.error(f"Error handling data request: {e}")
            
    def handle_analysis_request(self, sender: str, data: Any):
        """Handle analysis request from another tab."""
        try:
            ticker = data.get("ticker")
            analysis_type = data.get("analysis_type")
            
            if ticker in self.data_cache:
                # Perform analysis based on type
                if analysis_type == "technical":
                    result = self.perform_technical_analysis(ticker)
                elif analysis_type == "fundamental":
                    result = self.perform_fundamental_analysis(ticker)
                else:
                    result = None
                    
                if result:
                    self.message_bus.publish(
                        self.__class__.__name__,
                        "analysis_response",
                        {
                            "ticker": ticker,
                            "analysis_type": analysis_type,
                            "result": result
                        }
                    )
            else:
                self.message_bus.publish(
                    self.__class__.__name__,
                    "error",
                    f"Data not found for ticker: {ticker}"
                )
        except Exception as e:
            self.logger.error(f"Error handling analysis request: {e}")
            
    def add_ticker(self):
        """Add a new ticker and fetch its data."""
        try:
            ticker = self.ticker_edit.text().strip().upper()
            market = self.market_combo.currentText()
            
            if not ticker:
                QMessageBox.warning(self, "Error", "Please enter a ticker symbol")
                return
                
            # Get selected date range
            start_date = self.start_date.date().toPyDate()
            end_date = self.end_date.date().toPyDate()
            
            # Validate date range
            if start_date > end_date:
                QMessageBox.warning(self, "Error", "Start date cannot be after end date")
                return
                
            # Fetch data using yfinance with date range
            data = self.fetch_ticker_data(ticker, start_date, end_date)
            if data is None or data.empty:
                QMessageBox.warning(self, "Error", f"Could not fetch data for {ticker}")
                return
                
            # Cache the data
            self.data_cache[ticker] = {
                "market": market,
                "data": data,
                "start_date": start_date,
                "end_date": end_date
            }
            
            # Update table
            self.update_table()
            
            # Notify other tabs
            self.message_bus.publish(
                self.__class__.__name__,
                "data_updated",
                {
                    "ticker": ticker,
                    "market": market,
                    "data": data,
                    "start_date": start_date,
                    "end_date": end_date
                }
            )
            
            self.status_label.setText(f"Added {ticker} ({market})")
            
        except Exception as e:
            self.logger.error(f"Error adding ticker: {e}")
            QMessageBox.critical(self, "Error", str(e))
            
    def update_table(self):
        """Update the data table."""
        self.table.setRowCount(0)
        for ticker, cache in self.data_cache.items():
            data = cache["data"]
            market = cache["market"]
            
            row = self.table.rowCount()
            self.table.insertRow(row)
            
            self.table.setItem(row, 0, QTableWidgetItem(ticker))
            self.table.setItem(row, 1, QTableWidgetItem(market))
            self.table.setItem(row, 2, QTableWidgetItem(f"{data['Open'].iloc[-1]:.2f}"))
            self.table.setItem(row, 3, QTableWidgetItem(f"{data['High'].iloc[-1]:.2f}"))
            self.table.setItem(row, 4, QTableWidgetItem(f"{data['Low'].iloc[-1]:.2f}"))
            self.table.setItem(row, 5, QTableWidgetItem(f"{data['Close'].iloc[-1]:.2f}"))
            self.table.setItem(row, 6, QTableWidgetItem(f"{data['Volume'].iloc[-1]:,.0f}"))
            
            # Add date range information
            start_date = cache.get("start_date", "")
            end_date = cache.get("end_date", "")
            if start_date and end_date:
                self.table.setItem(row, 7, QTableWidgetItem(f"{start_date} to {end_date}"))
            
    def perform_technical_analysis(self, ticker: str) -> dict:
        """Perform technical analysis on the given ticker."""
        data = self.data_cache[ticker]["data"]
        
        # Calculate technical indicators
        sma_20 = data['Close'].rolling(window=20).mean()
        sma_50 = data['Close'].rolling(window=50).mean()
        rsi = self.calculate_rsi(data['Close'])
        
        return {
            "sma_20": sma_20.iloc[-1],
            "sma_50": sma_50.iloc[-1],
            "rsi": rsi.iloc[-1]
        }
        
    def perform_fundamental_analysis(self, ticker: str) -> dict:
        """Perform fundamental analysis on the given ticker."""
        # Implement fundamental analysis here
        return {}
        
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def cleanup(self):
        """Cleanup resources."""
        super().cleanup()
        self.data_cache.clear()

    def fetch_ticker_data(self, ticker: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """Fetch ticker data with retry logic."""
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                # Set default dates if not provided
                if start_date is None:
                    start_date = datetime.now() - timedelta(days=365)
                if end_date is None:
                    end_date = datetime.now()
                    
                # Fetch data
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False
                )
                
                if data.empty:
                    self.logger.warning(f"No data found for {ticker}")
                    return None
                    
                return data
                
            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1} failed for {ticker}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                self.logger.error(f"Failed to fetch data for {ticker} after {max_retries} attempts")
                return None

    def update_live_data(self):
        """Update live data for all tickers with error handling."""
        try:
            for ticker in self.ticker_list:
                try:
                    # Get live data
                    ticker_obj = yf.Ticker(ticker)
                    info = ticker_obj.info
                    
                    if not info:
                        self.logger.error(f"No live data available for {ticker}")
                        continue
                        
                    # Update table
                    self.update_ticker_table(ticker, info)
                    
                except Exception as e:
                    self.logger.error(f"Error updating live data for {ticker}: {str(e)}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error in update_live_data: {str(e)}")
            
        # Schedule next update
        QTimer.singleShot(10000, self.update_live_data)  # 10 seconds

def main():
    """Main function for the data tab process."""
    app = QApplication(sys.argv)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting data tab process")
    
    # Create and show the data tab
    window = DataTab()
    window.setWindowTitle("Data Tab")
    window.show()
    
    # Start the application event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 