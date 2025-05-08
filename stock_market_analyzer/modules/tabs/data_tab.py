import sys
import os
import logging
import traceback
import time
from typing import Dict, Any, List, Optional
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QCheckBox, QMessageBox,
    QFileDialog, QComboBox, QProgressBar, QListWidget,
    QListWidgetItem, QSplitter, QDateEdit, QGroupBox, QSpinBox,
    QTextEdit, QInputDialog, QTabWidget, QScrollArea, QFrame, QFormLayout
)
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QTimer, QDate
from PyQt6.QtGui import QFont, QPalette, QColor
import pandas as pd
import yfinance as yf
import json
import uuid

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ..message_bus import MessageBus
from ..data_stock import DataStock
from ..database import DatabaseConnector
from .base_tab import BaseTab
from ..settings import Settings
from ..connection_dashboard import ConnectionDashboard
from ..ui.theme import DarkTheme
from ..data_manager import DataManager

class DataTab(BaseTab):
    """Data tab for managing and viewing market data."""
    
    def __init__(self, message_bus: MessageBus, parent=None):
        """Initialize the Data tab.
        
        Args:
            message_bus: The message bus for communication.
            parent: The parent widget.
        """
        # Initialize attributes before parent __init__
        self.logger = logging.getLogger(__name__)
        self.settings = Settings()
        self.current_color_scheme = self.settings.get_color_scheme()
        self._ui_setup_done = False
        self.market_combo = None
        self.start_date_edit = None
        self.end_date_edit = None
        self.ticker_input = None
        self.add_ticker_button = None
        self.data_table = None
        self.status_label = None
        self.data_cache = {}
        self.pending_requests = {}
        self.data_viewer = None
        self.import_button = None
        self.file_type_combo = None
        self.imported_data_table = None  # New table for imported data
        self.connection_status = {}
        self.connection_start_times = {}
        self.messages_received = 0
        self.messages_sent = 0
        self.errors = 0
        self.message_latencies = []
        self.dashboard = ConnectionDashboard()
        self.data_manager = DataManager()
        self.data_manager.register_listener("DataTab", self._on_data_update)
        
        # Call parent __init__ with message bus
        super().__init__(message_bus, parent)
        
        # Setup UI after parent initialization
        self.setup_ui()
        self.setup_message_bus()
        
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
            
            # Market selection
            market_layout = QHBoxLayout()
            market_layout.addWidget(QLabel("Market:"))
            self.market_combo = QComboBox()
            self.market_combo.addItems(["US", "EU", "ASIA"])
            self.market_combo.currentTextChanged.connect(self.on_market_changed)
            market_layout.addWidget(self.market_combo)
            controls_layout.addLayout(market_layout)
            
            # Date range selection
            date_layout = QHBoxLayout()
            
            # Start date
            start_layout = QHBoxLayout()
            start_layout.addWidget(QLabel("Start Date:"))
            self.start_date_edit = QDateEdit()
            self.start_date_edit.setCalendarPopup(True)
            self.start_date_edit.setDate(QDate.currentDate().addDays(-30))
            start_layout.addWidget(self.start_date_edit)
            date_layout.addLayout(start_layout)
            
            # End date
            end_layout = QHBoxLayout()
            end_layout.addWidget(QLabel("End Date:"))
            self.end_date_edit = QDateEdit()
            self.end_date_edit.setCalendarPopup(True)
            self.end_date_edit.setDate(QDate.currentDate())
            end_layout.addWidget(self.end_date_edit)
            date_layout.addLayout(end_layout)
            
            controls_layout.addLayout(date_layout)
            
            # Ticker input
            ticker_layout = QHBoxLayout()
            ticker_layout.addWidget(QLabel("Ticker:"))
            self.ticker_input = QLineEdit()
            self.ticker_input.setPlaceholderText("Enter ticker (e.g., AAPL)")
            ticker_layout.addWidget(self.ticker_input)
            
            self.add_ticker_button = QPushButton("Add Ticker")
            self.add_ticker_button.clicked.connect(self.add_ticker)
            ticker_layout.addWidget(self.add_ticker_button)
            
            controls_layout.addLayout(ticker_layout)
            
            controls_group.setLayout(controls_layout)
            self.main_layout.addWidget(controls_group)
            
            # Create import controls
            import_group = QGroupBox("Import Data")
            import_layout = QHBoxLayout()
            
            # File type selection
            self.file_type_combo = QComboBox()
            self.file_type_combo.addItems(["CSV", "JSON", "Database"])
            import_layout.addWidget(self.file_type_combo)
            
            # Import button
            self.import_button = QPushButton("Import File")
            self.import_button.clicked.connect(self.import_data)
            import_layout.addWidget(self.import_button)
            
            import_group.setLayout(import_layout)
            self.main_layout.addWidget(import_group)
            
            # Create data viewer
            data_viewer_group = QGroupBox("Data Viewer")
            data_viewer_layout = QVBoxLayout()
            
            # Create text edit for data display
            self.data_viewer = QTextEdit()
            self.data_viewer.setReadOnly(True)
            self.data_viewer.setMinimumHeight(200)
            data_viewer_layout.addWidget(self.data_viewer)
            
            data_viewer_group.setLayout(data_viewer_layout)
            self.main_layout.addWidget(data_viewer_group)
            
            # Create imported data table
            imported_data_group = QGroupBox("Imported Data")
            imported_data_layout = QVBoxLayout()
            
            self.imported_data_table = QTableWidget()
            self.imported_data_table.setColumnCount(0)  # Will be set based on data
            imported_data_layout.addWidget(self.imported_data_table)
            
            imported_data_group.setLayout(imported_data_layout)
            self.main_layout.addWidget(imported_data_group)
            
            # Create data table
            data_table_group = QGroupBox("Data Table")
            data_table_layout = QVBoxLayout()
            
            self.data_table = QTableWidget()
            self.data_table.setColumnCount(7)
            self.data_table.setHorizontalHeaderLabels([
                "Date", "Open", "High", "Low", "Close", "Volume", "Adj Close"
            ])
            data_table_layout.addWidget(self.data_table)
            
            data_table_group.setLayout(data_table_layout)
            self.main_layout.addWidget(data_table_group)
            
            # Status bar
            status_layout = QHBoxLayout()
            self.status_label = QLabel("Ready")
            self.status_label.setStyleSheet("color: green")
            status_layout.addWidget(self.status_label)
            self.main_layout.addLayout(status_layout)
            
            # Add color scheme selection
            color_scheme_group = QGroupBox("Color Scheme")
            color_scheme_layout = QHBoxLayout()
            
            color_scheme_combo = QComboBox()
            color_scheme_combo.addItems(["Light", "Dark", "System"])
            color_scheme_combo.setCurrentText(self.current_color_scheme)
            color_scheme_combo.currentTextChanged.connect(self.on_color_scheme_changed)
            color_scheme_layout.addWidget(color_scheme_combo)
            
            color_scheme_group.setLayout(color_scheme_layout)
            self.main_layout.addWidget(color_scheme_group)
            
            # Apply current color scheme
            self.apply_color_scheme()
            
            # Mark UI setup as done
            self._ui_setup_done = True
            
        except Exception as e:
            self.logger.error(f"Error setting up UI: {str(e)}")
            self.logger.error(traceback.format_exc())
            
    def on_market_changed(self, market: str):
        """Handle market selection change.
        
        Args:
            market: The selected market.
        """
        try:
            print(f"Market changed to: {market}")
            self.status_label.setText(f"Selected market: {market}")
            
            # Clear existing data
            self.data_cache.clear()
            self.data_table.setRowCount(0)
            self.data_viewer.clear()
            
            # Request market data
            self.request_market_data(market)
            
        except Exception as e:
            error_msg = f"Error handling market change: {str(e)}"
            self.logger.error(error_msg)
            self.status_label.setText(error_msg)
            
    def add_ticker(self):
        """Add a ticker to the data table."""
        try:
            ticker = self.ticker_input.text().strip().upper()
            if not ticker:
                self.status_label.setText("Please enter a ticker")
                return
                
            print(f"Adding ticker: {ticker}")
            
            # Get date range
            start_date = self.start_date_edit.date().toPyDate()
            end_date = self.end_date_edit.date().toPyDate()
            
            # Request data
            self.request_ticker_data(ticker, start_date, end_date)
            
            # Clear input
            self.ticker_input.clear()
            
        except Exception as e:
            error_msg = f"Error adding ticker: {str(e)}"
            self.logger.error(error_msg)
            self.status_label.setText(error_msg)
            
    def request_market_data(self, market: str):
        """Request market data."""
        try:
            print(f"Requesting market data for: {market}")
            self.message_bus.publish("Data", "request_market_data", {
                "market": market,
                "timestamp": time.time()
            })
            self.status_label.setText(f"Requesting market data for {market}...")
            
        except Exception as e:
            error_msg = f"Error requesting market data: {str(e)}"
            self.logger.error(error_msg)
            self.status_label.setText(error_msg)
            
    def request_ticker_data(self, ticker: str, start_date: QDate, end_date: QDate):
        """Request ticker data."""
        try:
            print(f"Requesting data for ticker: {ticker}")
            self.message_bus.publish("Data", "request_ticker_data", {
                "ticker": ticker,
                "start_date": start_date.toString("yyyy-MM-dd"),
                "end_date": end_date.toString("yyyy-MM-dd"),
                "timestamp": time.time()
            })
            self.status_label.setText(f"Requesting data for {ticker}...")
            
        except Exception as e:
            error_msg = f"Error requesting ticker data: {str(e)}"
            self.logger.error(error_msg)
            self.status_label.setText(error_msg)
            
    def handle_message(self, sender: str, message_type: str, data: Dict[str, Any]):
        """Handle incoming messages.
        
        Args:
            sender: The sender of the message.
            message_type: The type of message.
            data: The message data.
        """
        try:
            if message_type == "MarketData":
                self.handle_market_data(data)
            elif message_type == "DataUpdate":
                self.handle_data_update(sender, data)
            elif message_type == "DataError":
                self.handle_data_error(sender, data)
            elif message_type == "ConnectionStatus":
                self.handle_connection_status(sender, message_type, data)
            elif message_type == "Heartbeat":
                self.handle_heartbeat(sender, message_type, data)
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
            self.logger.error(traceback.format_exc())
            
    def handle_market_data(self, data: Dict[str, Any]):
        """Handle market data messages.
        
        Args:
            data: The message data containing market information.
        """
        try:
            ticker = data.get("ticker")
            if ticker:
                # Update data cache
                self.data_cache[ticker] = data
                
                # Update UI components
                self.update_data_table(ticker, data)
                
                # Update data viewer with latest info
                self.data_viewer.append(f"Received market data for {ticker} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                self.data_viewer.append(f"Data: {json.dumps(data, indent=2)}\n")
                self.data_viewer.verticalScrollBar().setValue(
                    self.data_viewer.verticalScrollBar().maximum()
                )
                
                # Update status
                self.status_label.setText(f"Received market data for {ticker}")
                self.status_label.setStyleSheet("color: green")
                
                # Update dashboard metrics
                self.messages_received += 1
                self.update_dashboard()
                
        except Exception as e:
            self.logger.error(f"Error handling market data: {e}")
            self.logger.error(traceback.format_exc())
            self.status_label.setText(f"Error handling market data: {str(e)}")
            self.status_label.setStyleSheet("color: red")
            self.errors += 1
            self.update_dashboard()
            
    def handle_data_update(self, sender: str, data: Dict[str, Any]):
        """Handle data update messages."""
        try:
            ticker = data.get("ticker")
            market_data = data.get("data")
            if not ticker or not market_data:
                return
                
            # Update data cache
            self.data_cache[ticker] = market_data
            
            # Update table
            self.update_data_table(ticker, market_data)
            
        except Exception as e:
            error_msg = f"Error handling data update: {str(e)}"
            self.logger.error(error_msg)
            self.status_label.setText(error_msg)
            
    def handle_data_error(self, sender: str, data: Dict[str, Any]):
        """Handle data error messages."""
        try:
            error = data.get("error", "Unknown error")
            self.status_label.setText(f"Error: {error}")
            
        except Exception as e:
            error_msg = f"Error handling data error: {str(e)}"
            self.logger.error(error_msg)
            self.status_label.setText(error_msg)
            
    def update_data_table(self, ticker: str, data: Dict[str, Any]):
        """Update the data table with new data.
        
        Args:
            ticker: The ticker symbol.
            data: The market data.
        """
        try:
            # Clear existing rows
            self.data_table.setRowCount(0)
            
            # Add new rows
            for date, values in data.items():
                row = self.data_table.rowCount()
                self.data_table.insertRow(row)
                
                # Set date
                self.data_table.setItem(row, 0, QTableWidgetItem(date))
                
                # Set values
                for col, value in enumerate(values, start=1):
                    self.data_table.setItem(row, col, QTableWidgetItem(str(value)))
                    
            self.status_label.setText(f"Updated data for {ticker}")
            
        except Exception as e:
            error_msg = f"Error updating data table: {str(e)}"
            self.logger.error(error_msg)
            self.status_label.setText(error_msg)
            
    def import_data(self):
        """Import data from file or database."""
        try:
            file_type = self.file_type_combo.currentText()
            print(f"Importing {file_type} data")
            
            if file_type == "CSV":
                self.import_csv()
            elif file_type == "JSON":
                self.import_json()
            elif file_type == "Database":
                self.import_database()
                
        except Exception as e:
            error_msg = f"Error importing data: {str(e)}"
            self.logger.error(error_msg)
            self.status_label.setText(error_msg)
            
    def import_csv(self):
        """Import data from CSV file."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Open CSV File",
                "",
                "CSV Files (*.csv)"
            )
            
            if not file_path:
                return
                
            print(f"Importing CSV file: {file_path}")
            self.status_label.setText(f"Importing CSV file: {file_path}")
            
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Update data viewer
            self.data_viewer.append(f"\nCSV File: {file_path}")
            self.data_viewer.append(f"Number of rows: {len(df)}")
            self.data_viewer.append(f"Number of columns: {len(df.columns)}")
            self.data_viewer.append("\nColumns:")
            for col in df.columns:
                self.data_viewer.append(f"- {col}")
                
            # Update imported data table
            self.update_imported_data_table(df)
            
            self.status_label.setText(f"Imported CSV file: {file_path}")
            
        except Exception as e:
            error_msg = f"Error importing CSV: {str(e)}"
            self.logger.error(error_msg)
            self.status_label.setText(error_msg)
            
    def import_json(self):
        """Import data from JSON file."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Open JSON File",
                "",
                "JSON Files (*.json)"
            )
            
            if not file_path:
                return
                
            print(f"Importing JSON file: {file_path}")
            self.status_label.setText(f"Importing JSON file: {file_path}")
            
            # Read JSON file
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Convert to DataFrame if possible
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                raise ValueError("Unsupported JSON format")
                
            # Update data viewer
            self.data_viewer.append(f"\nJSON File: {file_path}")
            self.data_viewer.append(f"Number of rows: {len(df)}")
            self.data_viewer.append(f"Number of columns: {len(df.columns)}")
            self.data_viewer.append("\nColumns:")
            for col in df.columns:
                self.data_viewer.append(f"- {col}")
                
            # Update imported data table
            self.update_imported_data_table(df)
            
            self.status_label.setText(f"Imported JSON file: {file_path}")
            
        except Exception as e:
            error_msg = f"Error importing JSON: {str(e)}"
            self.logger.error(error_msg)
            self.status_label.setText(error_msg)
            
    def import_database(self):
        """Import data from database."""
        try:
            # Get database connection
            db = DatabaseConnector()
            
            # Get available tables
            tables = db.get_tables()
            
            if not tables:
                self.status_label.setText("No tables found in database")
                return
                
            # Show table selection dialog
            table, ok = QInputDialog.getItem(
                self,
                "Select Table",
                "Choose a table to import:",
                tables,
                0,
                False
            )
            
            if not ok or not table:
                return
                
            print(f"Importing table: {table}")
            self.status_label.setText(f"Importing table: {table}")
            
            # Get table data
            df = db.get_table_data(table)
            
            # Update data viewer
            self.data_viewer.append(f"\nDatabase Table: {table}")
            self.data_viewer.append(f"Number of rows: {len(df)}")
            self.data_viewer.append(f"Number of columns: {len(df.columns)}")
            self.data_viewer.append("\nColumns:")
            for col in df.columns:
                self.data_viewer.append(f"- {col}")
                
            # Update imported data table
            self.update_imported_data_table(df)
            
            self.status_label.setText(f"Imported table: {table}")
            
        except Exception as e:
            error_msg = f"Error importing database: {str(e)}"
            self.logger.error(error_msg)
            self.status_label.setText(error_msg)
            
    def update_imported_data_table(self, df: pd.DataFrame):
        """Update the imported data table with new data.
        
        Args:
            df: The pandas DataFrame containing the imported data.
        """
        try:
            # Clear existing table
            self.imported_data_table.setRowCount(0)
            self.imported_data_table.setColumnCount(0)
            
            if df.empty:
                self.status_label.setText("No data to display")
                return
                
            # Set up table dimensions
            self.imported_data_table.setRowCount(len(df))
            self.imported_data_table.setColumnCount(len(df.columns))
            
            # Set column headers
            self.imported_data_table.setHorizontalHeaderLabels(df.columns)
            
            # Populate table
            for i, row in df.iterrows():
                for j, value in enumerate(row):
                    item = QTableWidgetItem(str(value))
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.imported_data_table.setItem(i, j, item)
                    
            # Resize columns to fit content
            self.imported_data_table.resizeColumnsToContents()
            
            # Update status
            self.status_label.setText(f"Imported {len(df)} rows of data")
            
            # Publish data to other tabs
            self.publish_data_to_tabs(df)
            
        except Exception as e:
            error_msg = f"Error updating imported data table: {str(e)}"
            self.logger.error(error_msg)
            self.status_label.setText(error_msg)
            
    def publish_data_to_tabs(self, df: pd.DataFrame):
        """Publish data to other tabs via DataManager."""
        try:
            if df.empty:
                self.logger.warning("Attempting to publish empty dataframe")
                return
                
            # Create metadata
            metadata = {
                "timestamp": time.time(),
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
                "source": "DataTab",
                "data_type": "market_data"
            }
            
            # Add data to DataManager
            self.data_manager.add_data("DataTab", df, metadata)
            
            # Update status
            self.status_label.setText(f"Published {len(df)} rows of data")
            
        except Exception as e:
            self.handle_error("Error publishing data", e)
            
    def setup_message_bus(self):
        """Set up message bus subscriptions."""
        try:
            print("\n[Data] Setting up message bus...")
            print("[Data] Initializing connection status tracking...")
            
            # Subscribe to data responses
            print("[Data] Subscribing to data channels...")
            self.message_bus.subscribe("market_data", self.handle_market_data)
            self.message_bus.subscribe("ticker_data", self.handle_ticker_data)
            
            # Subscribe to connection status updates
            print("[Data] Subscribing to connection status channels...")
            self.message_bus.subscribe("connection", self.handle_connection_status)
            self.message_bus.subscribe("heartbeat", self.handle_heartbeat)
            self.message_bus.subscribe("shutdown", self.handle_shutdown)
            
            # Initialize connection status
            print("[Data] Initializing connection status for all tabs...")
            self.connection_status = {
                "Analysis": False,
                "Charts": False,
                "Models": False,
                "Predictions": False
            }
            
            # Send initial connection status
            print("[Data] Broadcasting initial connection status...")
            self.message_bus.publish("connection", "status_update", {
                "tab": "Data",
                "status": "connected",
                "tabs": list(self.connection_status.keys())
            })
            
            # Update dashboard
            self.update_dashboard()
            
            print("[Data] Message bus setup completed successfully")
            print("[Data] Active subscriptions:")
            print("  - market_data")
            print("  - ticker_data")
            print("  - connection")
            print("  - heartbeat")
            print("  - shutdown")
            print("[Data] Connection status initialized for tabs:")
            for tab, status in self.connection_status.items():
                print(f"  - {tab}: {'Connected' if status else 'Disconnected'}")
            print()
            
        except Exception as e:
            error_msg = f"Error setting up message bus: {str(e)}"
            print(f"[Data] Error: {error_msg}")
            self.logger.error(error_msg)
            self.status_label.setText(error_msg)
            self.errors += 1
            self.update_dashboard()

    def handle_connection_status(self, sender: str, message_type: str, data: Any):
        """Handle connection status updates."""
        try:
            if isinstance(data, dict):
                status = data.get('status')
                target_tab = data.get('target_tab')
                if target_tab == "Data":
                    print(f"\n[Data] Connection status update received:")
                    print(f"  - Sender: {sender}")
                    print(f"  - Message type: {message_type}")
                    print(f"  - Status: {status}")
                    print(f"  - Target tab: {target_tab}")
                    print(f"  - Timestamp: {time.time()}")
                    
                    self.connection_status = status
                    self.update_connection_status()
                    self.logger.debug(f"Connection status updated: {status}")
                    
                    # Show current connection status for all tabs
                    print("\n[Data] Current connection status for all tabs:")
                    for tab, connected in self.connection_status.items():
                        status_color = "green" if connected else "red"
                        print(f"  - {tab}: {'Connected' if connected else 'Not Connected'} ({status_color})")
                    print()
        except Exception as e:
            error_msg = f"Error handling connection status: {str(e)}"
            print(f"[Data] Error: {error_msg}")
            self.logger.error(error_msg)

    def handle_heartbeat(self, sender: str, message_type: str, data: Any):
        """Handle heartbeat messages."""
        try:
            tab = data.get('tab')
            if tab in self.connection_status:
                print(f"\n[Data] Heartbeat received:")
                print(f"  - From: {tab}")
                print(f"  - Timestamp: {time.time()}")
                print(f"  - Previous status: {'Connected' if self.connection_status[tab] else 'Disconnected'}")
                
                self.connection_status[tab] = True
                self.update_connection_status(tab, True)
                self.logger.debug(f"Heartbeat received from {tab}")
                
                print(f"  - New status: Connected")
                print(f"  - Connection duration: {self.get_connection_duration(tab)}")
                print()
        except Exception as e:
            error_msg = f"Error handling heartbeat: {str(e)}"
            print(f"[Data] Error: {error_msg}")
            self.logger.error(error_msg)

    def get_connection_duration(self, tab: str) -> str:
        """Get the duration of connection for a tab."""
        try:
            if tab in self.connection_status and self.connection_status[tab]:
                # Calculate connection duration
                connection_time = time.time()
                duration = connection_time - self.connection_start_times.get(tab, connection_time)
                return str(duration).split('.')[0]  # Remove microseconds
            return "Not connected"
        except Exception as e:
            return f"Error: {str(e)}"

    def update_connection_status(self, tab: str = None, status: bool = None):
        """Update connection status for a specific tab or all tabs."""
        try:
            if tab and status is not None:
                print(f"\n[Data] Updating connection status for {tab}:")
                print(f"  - Previous status: {'Connected' if self.connection_status.get(tab, False) else 'Disconnected'}")
                print(f"  - New status: {'Connected' if status else 'Disconnected'}")
                
                self.connection_status[tab] = status
                if status:
                    self.connection_start_times[tab] = time.time()
                else:
                    self.connection_start_times.pop(tab, None)
                
                print(f"  - Updated timestamp: {time.time()}")
                print()
            else:
                print("\n[Data] Updating connection status for all tabs:")
                for tab_name, tab_status in self.connection_status.items():
                    print(f"  - {tab_name}: {'Connected' if tab_status else 'Disconnected'}")
                print()
        except Exception as e:
            error_msg = f"Error updating connection status: {str(e)}"
            print(f"[Data] Error: {error_msg}")
            self.logger.error(error_msg)

    def cleanup(self):
        """Clean up resources."""
        try:
            # Unsubscribe from message bus
            self.message_bus.unsubscribe("market_data", self.handle_market_data)
            self.message_bus.unsubscribe("ticker_data", self.handle_ticker_data)
            self.message_bus.unsubscribe("connection", self.handle_connection_status)
            self.message_bus.unsubscribe("heartbeat", self.handle_heartbeat)
            
            # Clear data cache
            self.data_cache.clear()
            
            # Unregister from DataManager
            self.data_manager.unregister_listener("DataTab")
            
            self.logger.info("Cleanup completed")
            
        except Exception as e:
            self.handle_error("Error during cleanup", e)

    def on_color_scheme_changed(self, scheme: str):
        """Handle color scheme changes."""
        try:
            print(f"\n[Data] Changing color scheme to {scheme}")
            self.current_color_scheme = scheme
            self.settings.set_color_scheme(scheme)
            self.apply_color_scheme()
        except Exception as e:
            error_msg = f"Error changing color scheme: {str(e)}"
            print(f"[Data] Error: {error_msg}")
            self.logger.error(error_msg)

    def apply_color_scheme(self):
        """Apply the current color scheme to the UI."""
        try:
            # Use the shared dark theme
            DarkTheme.apply_theme(self)
            print(f"[Data] Applied {self.current_color_scheme} color scheme")
            
        except Exception as e:
            error_msg = f"Error applying color scheme: {str(e)}"
            print(f"[Data] Error: {error_msg}")
            self.logger.error(error_msg)

    def update_dashboard(self):
        """Update the connection dashboard with current status."""
        try:
            self.dashboard.update_connection_data("Data", {
                'status': True,
                'start_time': min(self.connection_start_times.values()) if self.connection_start_times else None,
                'last_heartbeat': time.time(),
                'messages_received': self.messages_received,
                'messages_sent': self.messages_sent,
                'errors': self.errors,
                'latencies': self.message_latencies
            })
        except Exception as e:
            error_msg = f"Error updating dashboard: {str(e)}"
            print(f"[Data] Error: {error_msg}")
            self.logger.error(error_msg)

    def handle_ticker_data(self, data: Dict[str, Any]):
        """Handle ticker data messages.
        
        Args:
            data: The message data containing ticker information.
        """
        try:
            ticker = data.get("ticker")
            if ticker:
                # Update data cache
                self.data_cache[ticker] = data
                
                # Update UI components
                self.update_data_table(ticker, data)
                
                # Update data viewer with latest info
                self.data_viewer.append(f"Received ticker data for {ticker} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                self.data_viewer.append(f"Data: {json.dumps(data, indent=2)}\n")
                self.data_viewer.verticalScrollBar().setValue(
                    self.data_viewer.verticalScrollBar().maximum()
                )
                
                # Update status
                self.status_label.setText(f"Received ticker data for {ticker}")
                self.status_label.setStyleSheet("color: green")
                
                # Update dashboard metrics
                self.messages_received += 1
                self.update_dashboard()
                
        except Exception as e:
            self.logger.error(f"Error handling ticker data: {e}")
            self.logger.error(traceback.format_exc())
            self.status_label.setText(f"Error handling ticker data: {str(e)}")
            self.status_label.setStyleSheet("color: red")
            self.errors += 1
            self.update_dashboard()

    def _on_data_update(self, source: str, data: pd.DataFrame, metadata: Dict[str, Any]):
        """Handle data updates from the DataManager."""
        try:
            # Update the data table
            self.update_data_table(data)
            
            # Extract metadata
            symbols = metadata.get("symbols", [])
            row_count = metadata.get("row_count", 0)
            
            # Update ticker input if needed
            for symbol in symbols:
                if symbol not in self.ticker_input.text().split(','):
                    current_text = self.ticker_input.text()
                    if current_text:
                        self.ticker_input.setText(f"{current_text},{symbol}")
                    else:
                        self.ticker_input.setText(symbol)
            
            # Update status
            self.status_label.setText(f"Received {row_count} rows of data")
            self.status_label.setStyleSheet("color: green")
            
        except Exception as e:
            self.handle_error("Error handling data update", e)

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