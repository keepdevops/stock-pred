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
        """Initialize the Data tab."""
        super().__init__(parent)
        self.message_bus = MessageBus()
        self.logger = logging.getLogger(__name__)
        self.data_stock = DataStock()
        self.db_connector = DatabaseConnector()
        self.current_color_scheme = "default"
        self.setup_theme()
        self.setup_ui()
        self.ticker_list = []
        self.data_cache = {}
        self.pending_requests = {}
        self.setup_heartbeat()
        
    def setup_theme(self):
        """Setup the theme for the data tab."""
        try:
            # Load theme settings from config
            theme_config = self.load_theme_config()
            self.current_color_scheme = theme_config.get("color_scheme", "default")
            
            # Apply theme colors
            self.apply_theme_colors()
            
        except Exception as e:
            self.logger.error(f"Error setting up theme: {e}")
            self.current_color_scheme = "default"
            
    def load_theme_config(self):
        """Load theme configuration from settings."""
        try:
            # TODO: Load from actual config file
            return {
                "color_scheme": "default",
                "font_size": 12,
                "font_family": "Arial"
            }
        except Exception as e:
            self.logger.error(f"Error loading theme config: {e}")
            return {}
            
    def apply_theme_colors(self):
        """Apply theme colors to UI elements."""
        try:
            if self.current_color_scheme == "dark":
                # Dark theme colors
                self.setStyleSheet("""
                    QWidget {
                        background-color: #2b2b2b;
                        color: #ffffff;
                    }
                    QTableWidget {
                        background-color: #3c3f41;
                        color: #ffffff;
                    }
                """)
            else:
                # Default theme colors
                self.setStyleSheet("""
                    QWidget {
                        background-color: #ffffff;
                        color: #000000;
                    }
                    QTableWidget {
                        background-color: #ffffff;
                        color: #000000;
                    }
                """)
        except Exception as e:
            self.logger.error(f"Error applying theme colors: {e}")
            
    def setup_ui(self):
        """Setup the data tab UI."""
        # Create main layout
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        
        # Create controls group
        controls_group = QGroupBox("Data Controls")
        controls_layout = QVBoxLayout()
        
        # Market and date range selection
        top_controls = QHBoxLayout()
        
        # Market selection
        market_layout = QHBoxLayout()
        market_layout.addWidget(QLabel("Market:"))
        self.market_combo = QComboBox()
        self.market_combo.addItems(["US", "HK", "CN", "Crypto", "Forex"])
        self.market_combo.currentTextChanged.connect(self.on_market_changed)
        market_layout.addWidget(self.market_combo)
        top_controls.addLayout(market_layout)
        
        # Date range selection
        date_layout = QHBoxLayout()
        date_layout.addWidget(QLabel("From:"))
        self.start_date = QDateEdit()
        self.start_date.setCalendarPopup(True)
        self.start_date.setDate(QDate.currentDate().addYears(-1))
        date_layout.addWidget(self.start_date)
        
        date_layout.addWidget(QLabel("To:"))
        self.end_date = QDateEdit()
        self.end_date.setCalendarPopup(True)
        self.end_date.setDate(QDate.currentDate())
        date_layout.addWidget(self.end_date)
        top_controls.addLayout(date_layout)
        
        controls_layout.addLayout(top_controls)
        
        # Ticker input and buttons
        ticker_layout = QHBoxLayout()
        ticker_layout.addWidget(QLabel("Ticker:"))
        self.ticker_edit = QLineEdit()
        self.ticker_edit.setPlaceholderText("Enter ticker symbol (e.g., AAPL)")
        self.ticker_edit.returnPressed.connect(self.add_ticker)
        ticker_layout.addWidget(self.ticker_edit)
        
        add_button = QPushButton("Add")
        add_button.clicked.connect(self.add_ticker)
        ticker_layout.addWidget(add_button)
        
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh_data)
        ticker_layout.addWidget(refresh_button)
        
        controls_layout.addLayout(ticker_layout)
        
        # Live data controls
        live_controls = QHBoxLayout()
        self.live_data_checkbox = QCheckBox("Live Updates")
        self.live_data_checkbox.stateChanged.connect(self.toggle_live_data)
        live_controls.addWidget(self.live_data_checkbox)
        
        self.update_interval = QSpinBox()
        self.update_interval.setRange(1, 60)
        self.update_interval.setValue(5)
        self.update_interval.setSuffix(" sec")
        live_controls.addWidget(QLabel("Update Interval:"))
        live_controls.addWidget(self.update_interval)
        
        controls_layout.addLayout(live_controls)
        
        controls_group.setLayout(controls_layout)
        self.main_layout.addWidget(controls_group)
        
        # Create splitter for ticker list and data table
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Ticker list
        ticker_list_widget = QWidget()
        ticker_list_layout = QVBoxLayout()
        ticker_list_layout.addWidget(QLabel("Active Tickers:"))
        
        self.ticker_list_widget = QListWidget()
        self.ticker_list_widget.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        ticker_list_layout.addWidget(self.ticker_list_widget)
        
        # Ticker list controls
        list_controls = QHBoxLayout()
        remove_button = QPushButton("Remove Selected")
        remove_button.clicked.connect(self.remove_selected_tickers)
        clear_button = QPushButton("Clear All")
        clear_button.clicked.connect(self.clear_tickers)
        list_controls.addWidget(remove_button)
        list_controls.addWidget(clear_button)
        ticker_list_layout.addLayout(list_controls)
        
        ticker_list_widget.setLayout(ticker_list_layout)
        splitter.addWidget(ticker_list_widget)
        
        # Data table
        table_widget = QWidget()
        table_layout = QVBoxLayout()
        
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(8)
        self.data_table.setHorizontalHeaderLabels([
            "Date", "Open", "High", "Low", "Close", "Volume", "Adj Close", "Change %"
        ])
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table_layout.addWidget(self.data_table)
        
        # Export controls
        export_layout = QHBoxLayout()
        export_csv = QPushButton("Export CSV")
        export_csv.clicked.connect(self.export_to_csv)
        export_json = QPushButton("Export JSON")
        export_json.clicked.connect(self.export_to_json)
        export_layout.addWidget(export_csv)
        export_layout.addWidget(export_json)
        table_layout.addLayout(export_layout)
        
        table_widget.setLayout(table_layout)
        splitter.addWidget(table_widget)
        
        self.main_layout.addWidget(splitter)
        
        # Status bar
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.connection_label = QLabel("⚪ Offline")
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.connection_label)
        self.main_layout.addLayout(status_layout)
        
        # Subscribe to message bus
        self.setup_message_bus()
        
    def setup_message_bus(self):
        """Set up message bus subscriptions."""
        self.message_bus.subscribe("Data", self.handle_message)
        self.message_bus.subscribe("Import", self.handle_import_message)
        self.message_bus.subscribe("Analysis", self.handle_analysis_message)
        self.message_bus.subscribe("Charts", self.handle_charts_message)
        
    def handle_message(self, sender: str, message_type: str, data: Any):
        """Handle incoming messages."""
        try:
            if message_type == "data_request":
                self.handle_data_request(data)
            elif message_type == "refresh":
                self.refresh_data()
            elif message_type == "add_ticker":
                ticker = data.get("ticker")
                if ticker:
                    self.add_ticker(ticker)
            elif message_type == "error":
                self.show_error(f"Error from {sender}: {data}")
            elif message_type == "heartbeat":
                self.update_connection_status(True)
        except Exception as e:
            self.logger.error(f"Error handling message: {str(e)}")
            self.show_error(str(e))
            
    def handle_import_message(self, sender: str, message_type: str, data: Any):
        """Handle messages from Import tab."""
        try:
            if message_type == "data_imported":
                self.process_imported_data(data)
        except Exception as e:
            self.logger.error(f"Error handling import message: {str(e)}")
            self.show_error(str(e))
            
    def handle_analysis_message(self, sender: str, message_type: str, data: Any):
        """Handle messages from Analysis tab."""
        try:
            if message_type == "analysis_complete":
                self.update_analysis_results(data)
        except Exception as e:
            self.logger.error(f"Error handling analysis message: {str(e)}")
            self.show_error(str(e))
            
    def handle_charts_message(self, sender: str, message_type: str, data: Any):
        """Handle messages from Charts tab."""
        try:
            if message_type == "chart_request":
                self.handle_chart_request(data)
        except Exception as e:
            self.logger.error(f"Error handling charts message: {str(e)}")
            self.show_error(str(e))
            
    def process_imported_data(self, data: Dict[str, Any]):
        """Process data imported from Import tab."""
        try:
            request_id = data.get("request_id")
            df = pd.DataFrame.from_records(data.get("data", []))
            metadata = data.get("metadata", {})
            
            # Cache the data
            self.data_cache[request_id] = {
                "data": df,
                "metadata": metadata
            }
            
            # Update the UI
            self.update_table_with_data(df)
            self.status_label.setText(f"Imported data: {metadata.get('shape', (0, 0))} rows")
            
            # Notify other tabs
            self.message_bus.publish("Data", "data_available", {
                "request_id": request_id,
                "shape": metadata.get("shape"),
                "columns": metadata.get("columns")
            })
            
        except Exception as e:
            self.logger.error(f"Error processing imported data: {str(e)}")
            self.show_error(str(e))
            
    def handle_data_request(self, data: Dict[str, Any]):
        """Handle data request from other tabs."""
        try:
            request_id = data.get("request_id")
            if request_id in self.data_cache:
                cached_data = self.data_cache[request_id]
                self.message_bus.publish("Data", "data_response", {
                    "request_id": request_id,
                    "data": cached_data["data"].to_dict("records"),
                    "metadata": cached_data["metadata"]
                })
            else:
                self.message_bus.publish("Data", "error", f"Data not found for request {request_id}")
        except Exception as e:
            self.logger.error(f"Error handling data request: {str(e)}")
            self.show_error(str(e))
            
    def handle_chart_request(self, data: Dict[str, Any]):
        """Handle chart data request from Charts tab."""
        try:
            request_id = data.get("request_id")
            ticker = data.get("ticker")
            
            if ticker in self.ticker_list:
                # Get the data for the requested ticker
                ticker_data = self.get_ticker_data(ticker)
                
                # Send the data to the Charts tab
                self.message_bus.publish("Data", "chart_data_response", {
                    "request_id": request_id,
                    "ticker": ticker,
                    "data": ticker_data.to_dict("records")
                })
            else:
                self.message_bus.publish("Data", "error", f"No data available for ticker {ticker}")
        except Exception as e:
            self.logger.error(f"Error handling chart request: {str(e)}")
            self.show_error(str(e))
            
    def update_analysis_results(self, data: Dict[str, Any]):
        """Update UI with analysis results."""
        try:
            request_id = data.get("request_id")
            results = data.get("results", {})
            
            # Update the data table with analysis results if applicable
            if request_id in self.data_cache:
                self.update_table_with_analysis(request_id, results)
                
        except Exception as e:
            self.logger.error(f"Error updating analysis results: {str(e)}")
            self.show_error(str(e))
            
    def show_error(self, message: str):
        """Show error message in status bar and log."""
        self.status_label.setText(f"Error: {message}")
        self.logger.error(message)
        
    def update_connection_status(self, connected: bool):
        """Update connection status indicator."""
        if connected:
            self.connection_label.setText("🟢 Online")
        else:
            self.connection_label.setText("⚪ Offline")
            
    def setup_heartbeat(self):
        """Setup heartbeat timer."""
        self.heartbeat_timer = QTimer()
        self.heartbeat_timer.timeout.connect(self.send_heartbeat)
        self.heartbeat_timer.start(5000)  # 5 seconds
        
    def send_heartbeat(self):
        """Send heartbeat message."""
        try:
            self.message_bus.publish("Data", "heartbeat", {
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            self.logger.error(f"Error sending heartbeat: {str(e)}")
            self.update_connection_status(False)

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
        self.data_table.setRowCount(0)
        for ticker, cache in self.data_cache.items():
            data = cache["data"]
            market = cache["market"]
            
            row = self.data_table.rowCount()
            self.data_table.insertRow(row)
            
            self.data_table.setItem(row, 0, QTableWidgetItem(f"{data['Date'].iloc[-1]:.2f}"))
            self.data_table.setItem(row, 1, QTableWidgetItem(f"{data['Open'].iloc[-1]:.2f}"))
            self.data_table.setItem(row, 2, QTableWidgetItem(f"{data['High'].iloc[-1]:.2f}"))
            self.data_table.setItem(row, 3, QTableWidgetItem(f"{data['Low'].iloc[-1]:.2f}"))
            self.data_table.setItem(row, 4, QTableWidgetItem(f"{data['Close'].iloc[-1]:.2f}"))
            self.data_table.setItem(row, 5, QTableWidgetItem(f"{data['Volume'].iloc[-1]:,.0f}"))
            self.data_table.setItem(row, 6, QTableWidgetItem(f"{data['Adj Close'].iloc[-1]:.2f}"))
            
            # Add date range information
            start_date = cache.get("start_date", "")
            end_date = cache.get("end_date", "")
            if start_date and end_date:
                self.data_table.setItem(row, 7, QTableWidgetItem(f"{start_date} to {end_date}"))
            
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

    def closeEvent(self, event):
        """Handle the close event."""
        try:
            self.message_bus.publish(
                "Data",
                "shutdown",
                {"source": "Data", "message": "Shutting down"}
            )
            super().closeEvent(event)
        except Exception as e:
            self.logger.error(f"Error in closeEvent: {e}")
            event.ignore()

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