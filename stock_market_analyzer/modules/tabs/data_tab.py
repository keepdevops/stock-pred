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
    QListWidgetItem, QSplitter, QDateEdit, QGroupBox, QSpinBox
)
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QTimer, QDate
from PyQt6.QtGui import QFont
import pandas as pd
import yfinance as yf
import json
import time
import uuid

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ..message_bus import MessageBus
from ..data_stock import DataStock
from ..database import DatabaseConnector
from .base_tab import BaseTab

class DataTab(BaseTab):
    """Data tab for viewing and managing stock data."""
    
    def __init__(self, parent=None):
        """Initialize the Data tab."""
        # Initialize attributes before parent __init__
        self.data_cache = {}
        self.pending_requests = {}
        self._ui_setup_done = False
        self.main_layout = None
        self.market_combo = None
        self.start_date = None
        self.end_date = None
        self.ticker_edit = None
        self.data_table = None
        self.status_label = None
        self.connection_label = None
        self.heartbeat_timer = None
        self.live_data_timer = None
        
        super().__init__(parent)
        
        # Setup UI after parent initialization
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the UI components."""
        try:
            # Clear the base layout
            while self.main_layout.count():
                item = self.main_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
                    
            self.main_layout.setSpacing(10)
            self.main_layout.setContentsMargins(10, 10, 10, 10)
            
            # Create controls group
            controls_group = QGroupBox("Data Controls")
            controls_layout = QVBoxLayout()
            
            # Top controls
            top_controls = QHBoxLayout()
            
            # Market selection
            market_layout = QHBoxLayout()
            market_layout.addWidget(QLabel("Market:"))
            self.market_combo = QComboBox()
            self.market_combo.addItems(["NYSE", "NASDAQ", "AMEX"])
            self.market_combo.currentTextChanged.connect(self.on_market_changed)
            market_layout.addWidget(self.market_combo)
            top_controls.addLayout(market_layout)
            
            # Date range selection
            date_layout = QHBoxLayout()
            date_layout.addWidget(QLabel("Date Range:"))
            self.start_date = QDateEdit()
            self.start_date.setCalendarPopup(True)
            self.start_date.setDate(QDate.currentDate().addDays(-30))
            self.end_date = QDateEdit()
            self.end_date.setCalendarPopup(True)
            self.end_date.setDate(QDate.currentDate())
            date_layout.addWidget(self.start_date)
            date_layout.addWidget(QLabel("to"))
            date_layout.addWidget(self.end_date)
            top_controls.addLayout(date_layout)
            
            controls_layout.addLayout(top_controls)
            
            # Ticker input
            ticker_layout = QHBoxLayout()
            ticker_layout.addWidget(QLabel("Ticker:"))
            self.ticker_edit = QLineEdit()
            self.ticker_edit.setPlaceholderText("Enter ticker symbol")
            self.ticker_edit.returnPressed.connect(self.on_ticker_entered)
            ticker_layout.addWidget(self.ticker_edit)
            controls_layout.addLayout(ticker_layout)
            
            controls_group.setLayout(controls_layout)
            self.main_layout.addWidget(controls_group)
            
            # Create data table
            table_group = QGroupBox("Stock Data")
            table_layout = QVBoxLayout()
            table_group.setLayout(table_layout)
            
            self.data_table = QTableWidget()
            self.data_table.setColumnCount(7)
            self.data_table.setHorizontalHeaderLabels([
                "Date", "Open", "High", "Low", "Close", "Volume", "Adj Close"
            ])
            self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            table_layout.addWidget(self.data_table)
            
            self.main_layout.addWidget(table_group)
            
            # Create status bar
            status_layout = QHBoxLayout()
            
            self.status_label = QLabel("Ready")
            self.status_label.setStyleSheet("color: green")
            status_layout.addWidget(self.status_label)
            
            self.connection_label = QLabel("Disconnected")
            self.connection_label.setStyleSheet("color: red")
            status_layout.addStretch()
            status_layout.addWidget(self.connection_label)
            
            self.main_layout.addLayout(status_layout)
            
            self._ui_setup_done = True
            
        except Exception as e:
            error_msg = f"Error setting up UI: {str(e)}"
            self.logger.error(error_msg)
            if self.status_label:
                self.status_label.setText(error_msg)
                
    def _setup_message_bus_impl(self):
        """Setup message bus subscriptions."""
        super()._setup_message_bus_impl()
        self.message_bus.subscribe("Data", self.handle_message)
        
    def on_market_changed(self, market: str):
        """Handle market selection change."""
        try:
            self.status_label.setText(f"Selected market: {market}")
            self.ticker_edit.clear()
            self.data_table.setRowCount(0)
            
        except Exception as e:
            error_msg = f"Error handling market change: {str(e)}"
            self.logger.error(error_msg)
            self.status_label.setText(error_msg)
            
    def on_ticker_entered(self):
        """Handle ticker input."""
        try:
            ticker = self.ticker_edit.text().strip().upper()
            if not ticker:
                self.status_label.setText("Please enter a ticker symbol")
                return
                
            market = self.market_combo.currentText()
            start_date = self.start_date.date().toString("yyyy-MM-dd")
            end_date = self.end_date.date().toString("yyyy-MM-dd")
            
            # Request data
            request_id = str(uuid.uuid4())
            self.pending_requests[request_id] = {
                'type': 'data_request',
                'ticker': ticker,
                'market': market,
                'start_date': start_date,
                'end_date': end_date,
                'timestamp': datetime.now()
            }
            
            self.message_bus.publish(
                "Data",
                "data_request",
                {
                    'request_id': request_id,
                    'ticker': ticker,
                    'market': market,
                    'start_date': start_date,
                    'end_date': end_date
                }
            )
            
            self.status_label.setText(f"Fetching data for {ticker}")
            
        except Exception as e:
            error_msg = f"Error handling ticker input: {str(e)}"
            self.logger.error(error_msg)
            self.status_label.setText(error_msg)
            
    def handle_message(self, sender: str, message_type: str, data: Any):
        """Handle incoming messages."""
        try:
            if message_type == "data_response":
                self.handle_data_response(sender, data)
            elif message_type == "error":
                self.status_label.setText(f"Error: {data.get('error', 'Unknown error')}")
            elif message_type == "heartbeat":
                self.connection_label.setText("Connected")
                self.connection_label.setStyleSheet("color: green")
                
        except Exception as e:
            error_msg = f"Error handling message: {str(e)}"
            self.logger.error(error_msg)
            self.status_label.setText(error_msg)
            
    def handle_data_response(self, sender: str, data: Any):
        """Handle data response."""
        try:
            request_id = data.get('request_id')
            if request_id in self.pending_requests:
                ticker = self.pending_requests[request_id]['ticker']
                market_data = data.get('market_data', [])
                
                if market_data:
                    # Clear previous data
                    self.data_table.setRowCount(0)
                    
                    # Add new data
                    for row, data_point in enumerate(market_data):
                        self.data_table.insertRow(row)
                        self.data_table.setItem(row, 0, QTableWidgetItem(data_point['date']))
                        self.data_table.setItem(row, 1, QTableWidgetItem(str(data_point['open'])))
                        self.data_table.setItem(row, 2, QTableWidgetItem(str(data_point['high'])))
                        self.data_table.setItem(row, 3, QTableWidgetItem(str(data_point['low'])))
                        self.data_table.setItem(row, 4, QTableWidgetItem(str(data_point['close'])))
                        self.data_table.setItem(row, 5, QTableWidgetItem(str(data_point['volume'])))
                        self.data_table.setItem(row, 6, QTableWidgetItem(str(data_point['adj_close'])))
                    
                    self.status_label.setText(f"Data loaded for {ticker}")
                else:
                    self.status_label.setText(f"No data available for {ticker}")
                    
                del self.pending_requests[request_id]
                
        except Exception as e:
            error_msg = f"Error handling data response: {str(e)}"
            self.logger.error(error_msg)
            self.status_label.setText(error_msg)
            
    def cleanup(self):
        """Cleanup resources."""
        try:
            if self.heartbeat_timer:
                self.heartbeat_timer.stop()
            if self.live_data_timer:
                self.live_data_timer.stop()
            super().cleanup()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            
    def closeEvent(self, event):
        """Handle the close event."""
        self.cleanup()
        super().closeEvent(event)

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