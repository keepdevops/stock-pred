import sys
import os
import signal
import logging
import traceback
import yfinance as yf
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QComboBox, QPushButton, QLabel, QFileDialog, QMessageBox, QListWidget,
    QListWidgetItem, QLineEdit, QSplitter, QApplication, QSpinBox, QCheckBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from modules.database import DatabaseConnector
from modules.message_bus import MessageBus

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class DataTab(QWidget):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.db = DatabaseConnector()
        self.message_bus = MessageBus()
        self.live_data_timer = None
        self.setup_ui()
        self.setup_message_bus()
        self.setup_heartbeat()
        
    def setup_ui(self):
        """Set up the user interface."""
        main_layout = QVBoxLayout(self)
        
        # Top controls
        controls_layout = QHBoxLayout()
        
        # Market type selection
        self.market_type_combo = QComboBox()
        self.market_type_combo.addItems(["Stocks", "Forex", "Cryptocurrency"])
        self.market_type_combo.currentTextChanged.connect(self.update_ticker_table)
        controls_layout.addWidget(QLabel("Market Type:"))
        controls_layout.addWidget(self.market_type_combo)
        
        # Ticker input
        ticker_layout = QHBoxLayout()
        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("Enter ticker symbol")
        self.ticker_input.returnPressed.connect(self.add_ticker)
        ticker_layout.addWidget(QLabel("Ticker:"))
        ticker_layout.addWidget(self.ticker_input)
        
        add_button = QPushButton("Add")
        add_button.clicked.connect(self.add_ticker)
        ticker_layout.addWidget(add_button)
        
        controls_layout.addLayout(ticker_layout)
        
        # Live data controls
        live_data_layout = QHBoxLayout()
        self.live_data_checkbox = QCheckBox("Live Data")
        self.live_data_checkbox.stateChanged.connect(self.toggle_live_data)
        live_data_layout.addWidget(self.live_data_checkbox)
        
        self.refresh_interval = QSpinBox()
        self.refresh_interval.setRange(1, 60)
        self.refresh_interval.setValue(5)
        self.refresh_interval.setSuffix(" sec")
        live_data_layout.addWidget(QLabel("Refresh:"))
        live_data_layout.addWidget(self.refresh_interval)
        
        # Color scheme selection
        self.color_scheme_combo = QComboBox()
        self.color_scheme_combo.addItems([
            "Default (Red/Green)",
            "Protanopia (Blue/Yellow)",
            "Deuteranopia (Blue/Orange)",
            "Tritanopia (Red/Blue)",
            "Monochrome (Black/White)"
        ])
        self.color_scheme_combo.currentTextChanged.connect(self.update_color_scheme)
        live_data_layout.addWidget(QLabel("Color Scheme:"))
        live_data_layout.addWidget(self.color_scheme_combo)
        
        controls_layout.addLayout(live_data_layout)
        
        # Refresh button
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.update_ticker_table)
        controls_layout.addWidget(refresh_button)
        
        main_layout.addLayout(controls_layout)
        
        # Splitter for ticker list and data table
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Ticker list
        ticker_list_widget = QWidget()
        ticker_list_layout = QVBoxLayout(ticker_list_widget)
        
        self.ticker_list = QListWidget()
        self.ticker_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.ticker_list.itemSelectionChanged.connect(self.update_selected_tickers)
        ticker_list_layout.addWidget(QLabel("Selected Tickers:"))
        ticker_list_layout.addWidget(self.ticker_list)
        
        # Ticker list controls
        list_controls = QHBoxLayout()
        remove_button = QPushButton("Remove Selected")
        remove_button.clicked.connect(self.remove_selected_tickers)
        clear_button = QPushButton("Clear All")
        clear_button.clicked.connect(self.clear_tickers)
        list_controls.addWidget(remove_button)
        list_controls.addWidget(clear_button)
        ticker_list_layout.addLayout(list_controls)
        
        splitter.addWidget(ticker_list_widget)
        
        # Data table
        table_widget = QWidget()
        table_layout = QVBoxLayout(table_widget)
        
        self.table = QTableWidget()
        self.table.setColumnCount(9)  # Added columns for live data
        self.table.setHorizontalHeaderLabels([
            "Symbol", "Name", "Market", "Last Price", "Volume",
            "Open", "High", "Low", "Change %"
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        table_layout.addWidget(self.table)
        
        # Export buttons
        export_layout = QHBoxLayout()
        export_csv = QPushButton("Export to CSV")
        export_csv.clicked.connect(self.export_to_csv)
        export_json = QPushButton("Export to JSON")
        export_json.clicked.connect(self.export_to_json)
        export_layout.addWidget(export_csv)
        export_layout.addWidget(export_json)
        table_layout.addLayout(export_layout)
        
        splitter.addWidget(table_widget)
        
        # Set initial sizes
        splitter.setSizes([200, 600])
        
        main_layout.addWidget(splitter)
        
        # Status label
        self.status_label = QLabel("Ready")
        main_layout.addWidget(self.status_label)
        
        # Initialize with some data
        self.update_ticker_table()
        
    def toggle_live_data(self, state):
        """Toggle live data updates."""
        if state == Qt.CheckState.Checked:
            self.start_live_data()
        else:
            self.stop_live_data()
            
    def start_live_data(self):
        """Start live data updates."""
        if self.live_data_timer is None:
            interval = self.refresh_interval.value() * 1000  # Convert to milliseconds
            self.live_data_timer = QTimer()
            self.live_data_timer.timeout.connect(self.update_live_data)
            self.live_data_timer.start(interval)
            self.status_label.setText("Live data updates started")
            
    def stop_live_data(self):
        """Stop live data updates."""
        if self.live_data_timer is not None:
            self.live_data_timer.stop()
            self.live_data_timer = None
            self.status_label.setText("Live data updates stopped")
            
    def update_color_scheme(self, scheme_name: str):
        """Update the color scheme for price changes."""
        self.current_color_scheme = scheme_name
        self.update_live_data()  # Refresh the display with new colors
        
    def get_price_change_style(self, change_pct: float) -> tuple:
        """Get the appropriate color and symbol for price changes based on the selected color scheme."""
        if change_pct >= 0:
            symbol = "▲"  # Up triangle
        else:
            symbol = "▼"  # Down triangle
            
        if self.current_color_scheme == "Default (Red/Green)":
            color = Qt.GlobalColor.darkGreen if change_pct >= 0 else Qt.GlobalColor.darkRed
        elif self.current_color_scheme == "Protanopia (Blue/Yellow)":
            color = Qt.GlobalColor.darkBlue if change_pct >= 0 else Qt.GlobalColor.darkYellow
        elif self.current_color_scheme == "Deuteranopia (Blue/Orange)":
            color = Qt.GlobalColor.darkBlue if change_pct >= 0 else Qt.GlobalColor.darkCyan
        elif self.current_color_scheme == "Tritanopia (Red/Blue)":
            color = Qt.GlobalColor.darkRed if change_pct >= 0 else Qt.GlobalColor.darkBlue
        else:  # Monochrome
            color = Qt.GlobalColor.black if change_pct >= 0 else Qt.GlobalColor.darkGray
            
        return color, symbol
        
    def update_live_data(self):
        """Update live data for selected tickers."""
        try:
            selected_tickers = [self.ticker_list.item(i).text() for i in range(self.ticker_list.count())]
            if not selected_tickers:
                return
                
            # Get live data from yfinance
            for ticker in selected_tickers:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    hist = stock.history(period="1d")
                    
                    if not hist.empty:
                        # Find the row with this ticker
                        for row in range(self.table.rowCount()):
                            if self.table.item(row, 0).text() == ticker:
                                # Update live data columns
                                self.table.setItem(row, 3, QTableWidgetItem(f"{hist['Close'].iloc[-1]:.2f}"))
                                self.table.setItem(row, 4, QTableWidgetItem(f"{hist['Volume'].iloc[-1]:,.0f}"))
                                self.table.setItem(row, 5, QTableWidgetItem(f"{hist['Open'].iloc[-1]:.2f}"))
                                self.table.setItem(row, 6, QTableWidgetItem(f"{hist['High'].iloc[-1]:.2f}"))
                                self.table.setItem(row, 7, QTableWidgetItem(f"{hist['Low'].iloc[-1]:.2f}"))
                                
                                # Calculate and display change percentage with accessibility features
                                change_pct = ((hist['Close'].iloc[-1] - hist['Open'].iloc[-1]) / hist['Open'].iloc[-1]) * 100
                                color, symbol = self.get_price_change_style(change_pct)
                                
                                change_text = f"{symbol} {change_pct:+.2f}%"  # + sign for positive values
                                change_item = QTableWidgetItem(change_text)
                                change_item.setForeground(color)
                                self.table.setItem(row, 8, change_item)
                                break
                                
                except Exception as e:
                    self.logger.error(f"Error updating live data for {ticker}: {e}")
                    
            self.status_label.setText(f"Live data updated at {datetime.now().strftime('%H:%M:%S')}")
            
        except Exception as e:
            self.logger.error(f"Error in live data update: {e}")
            self.status_label.setText(f"Error updating live data: {str(e)}")
        
    def add_ticker(self):
        """Add a new ticker to the list."""
        try:
            ticker = self.ticker_input.text().strip().upper()
            if not ticker:
                self.status_label.setText("Please enter a ticker symbol")
                return
                
            # Check if ticker already exists
            for i in range(self.ticker_list.count()):
                if self.ticker_list.item(i).text() == ticker:
                    self.status_label.setText(f"Ticker {ticker} already exists")
                    return
                    
            # Add ticker to list
            self.ticker_list.addItem(ticker)
            self.ticker_input.clear()
            
            # Update table with new ticker
            self.update_ticker_table()
            
            self.status_label.setText(f"Added ticker {ticker}")
            
        except Exception as e:
            self.logger.error(f"Error adding ticker: {e}")
            self.status_label.setText(f"Error: {str(e)}")
        
    def remove_selected_tickers(self):
        """Remove selected tickers from the list."""
        selected_items = self.ticker_list.selectedItems()
        for item in selected_items:
            self.ticker_list.takeItem(self.ticker_list.row(item))
        self.update_ticker_table()
        
    def clear_tickers(self):
        """Clear all tickers from the list."""
        self.ticker_list.clear()
        self.update_ticker_table()
        
    def update_selected_tickers(self):
        """Update the table with selected tickers."""
        self.update_ticker_table()
        
    def update_ticker_table(self):
        """Update the ticker table with data from the database and live data."""
        try:
            market_type = self.market_type_combo.currentText()
            selected_tickers = [self.ticker_list.item(i).text() for i in range(self.ticker_list.count())]
            
            if not selected_tickers:
                self.table.setRowCount(0)
                self.status_label.setText("No tickers selected")
                return
                
            # Get data from database
            db_data = self.db.get_stock_data(market_type=market_type)
            
            # Create a dictionary of existing data
            existing_data = {item['symbol']: item for item in db_data}
            
            # Update table with all selected tickers
            self.table.setRowCount(len(selected_tickers))
            
            for row, ticker in enumerate(selected_tickers):
                # Get data from database if available
                db_item = existing_data.get(ticker, {})
                
                # Set basic information
                self.table.setItem(row, 0, QTableWidgetItem(ticker))
                self.table.setItem(row, 1, QTableWidgetItem(db_item.get('name', 'N/A')))
                self.table.setItem(row, 2, QTableWidgetItem(market_type))  # Always use current market type
                
                # Try to get live data
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    hist = stock.history(period="1d")
                    
                    if not hist.empty:
                        # Update with live data
                        self.table.setItem(row, 3, QTableWidgetItem(f"{hist['Close'].iloc[-1]:.2f}"))
                        self.table.setItem(row, 4, QTableWidgetItem(f"{hist['Volume'].iloc[-1]:,.0f}"))
                        self.table.setItem(row, 5, QTableWidgetItem(f"{hist['Open'].iloc[-1]:.2f}"))
                        self.table.setItem(row, 6, QTableWidgetItem(f"{hist['High'].iloc[-1]:.2f}"))
                        self.table.setItem(row, 7, QTableWidgetItem(f"{hist['Low'].iloc[-1]:.2f}"))
                        
                        # Calculate and display change percentage
                        change_pct = ((hist['Close'].iloc[-1] - hist['Open'].iloc[-1]) / hist['Open'].iloc[-1]) * 100
                        color, symbol = self.get_price_change_style(change_pct)
                        change_text = f"{symbol} {change_pct:+.2f}%"
                        change_item = QTableWidgetItem(change_text)
                        change_item.setForeground(color)
                        self.table.setItem(row, 8, change_item)
                        
                        # Update name if available from yfinance
                        if 'longName' in info:
                            self.table.setItem(row, 1, QTableWidgetItem(info['longName']))
                            
                    else:
                        # Use database data if live data is not available
                        self.table.setItem(row, 3, QTableWidgetItem(str(db_item.get('last_price', 'N/A'))))
                        self.table.setItem(row, 4, QTableWidgetItem(str(db_item.get('volume', 'N/A'))))
                        self.table.setItem(row, 5, QTableWidgetItem('N/A'))
                        self.table.setItem(row, 6, QTableWidgetItem('N/A'))
                        self.table.setItem(row, 7, QTableWidgetItem('N/A'))
                        self.table.setItem(row, 8, QTableWidgetItem('N/A'))
                        
                except Exception as e:
                    self.logger.error(f"Error getting live data for {ticker}: {e}")
                    # Use database data if live data fails
                    self.table.setItem(row, 3, QTableWidgetItem(str(db_item.get('last_price', 'N/A'))))
                    self.table.setItem(row, 4, QTableWidgetItem(str(db_item.get('volume', 'N/A'))))
                    self.table.setItem(row, 5, QTableWidgetItem('N/A'))
                    self.table.setItem(row, 6, QTableWidgetItem('N/A'))
                    self.table.setItem(row, 7, QTableWidgetItem('N/A'))
                    self.table.setItem(row, 8, QTableWidgetItem('N/A'))
            
            self.status_label.setText(f"Loaded {len(selected_tickers)} tickers for {market_type}")
            
        except Exception as e:
            self.logger.error(f"Error updating ticker table: {e}")
            self.status_label.setText(f"Error: {str(e)}")
            
    def export_to_csv(self):
        """Export the current table data to CSV."""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save CSV", "", "CSV Files (*.csv)"
            )
            if file_path:
                with open(file_path, 'w') as f:
                    # Write header
                    header = [self.table.horizontalHeaderItem(i).text() 
                            for i in range(self.table.columnCount())]
                    f.write(','.join(header) + '\n')
                    
                    # Write data
                    for row in range(self.table.rowCount()):
                        row_data = []
                        for col in range(self.table.columnCount()):
                            item = self.table.item(row, col)
                            row_data.append(item.text() if item else '')
                        f.write(','.join(row_data) + '\n')
                
                self.status_label.setText(f"Exported to {file_path}")
                
        except Exception as e:
            self.logger.error(f"Error exporting to CSV: {e}")
            self.status_label.setText(f"Error exporting to CSV: {str(e)}")
            
    def export_to_json(self):
        """Export the current table data to JSON."""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save JSON", "", "JSON Files (*.json)"
            )
            if file_path:
                import json
                data = []
                for row in range(self.table.rowCount()):
                    row_data = {}
                    for col in range(self.table.columnCount()):
                        header = self.table.horizontalHeaderItem(col).text()
                        item = self.table.item(row, col)
                        row_data[header] = item.text() if item else ''
                    data.append(row_data)
                
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=4)
                
                self.status_label.setText(f"Exported to {file_path}")
                
        except Exception as e:
            self.logger.error(f"Error exporting to JSON: {e}")
            self.status_label.setText(f"Error exporting to JSON: {str(e)}")
            
    def setup_message_bus(self):
        """Set up message bus subscriptions."""
        self.message_bus.subscribe("data_tab", self.handle_message)
        
    def handle_message(self, sender: str, message_type: str, data: Any):
        """Handle messages from the message bus."""
        try:
            if message_type == "refresh":
                self.update_ticker_table()
            elif message_type == "add_ticker":
                ticker = data.get("ticker")
                if ticker:
                    self.ticker_input.setText(ticker)
                    self.add_ticker()
            elif message_type == "heartbeat":
                self.logger.debug(f"Received heartbeat from {sender}")
            elif message_type == "shutdown":
                self.logger.info(f"Received shutdown request from {sender}")
                self.close()
                    
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
            self.logger.error(traceback.format_exc())
            
    def setup_heartbeat(self):
        """Set up heartbeat timer."""
        self.heartbeat_timer = QTimer()
        self.heartbeat_timer.timeout.connect(self.send_heartbeat)
        self.heartbeat_timer.start(5000)  # 5 seconds
        
    def send_heartbeat(self):
        """Send a heartbeat message to the message bus."""
        try:
            self.message_bus.publish("Data", "heartbeat", {})
        except Exception as e:
            self.logger.error(f"Error sending heartbeat: {str(e)}")
            self.logger.error(traceback.format_exc())
        
    def closeEvent(self, event):
        """Handle window close event."""
        self.heartbeat_timer.stop()
        self.message_bus.publish({
            "type": "shutdown",
            "source": "data_tab"
        })
        event.accept()

def handle_shutdown(signum, frame):
    """Handle shutdown signal."""
    logger = logging.getLogger(__name__)
    logger.info("Received shutdown signal, performing graceful shutdown")
    QApplication.quit()

def main():
    """Main function to start the data tab process."""
    try:
        # Set up signal handlers
        signal.signal(signal.SIGTERM, handle_shutdown)
        signal.signal(signal.SIGINT, handle_shutdown)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(__name__)
        
        # Create application
        app = QApplication(sys.argv)
        
        # Create tab
        tab = DataTab()
        tab.show()  # Make sure to show the tab
        
        # Set up heartbeat timer
        heartbeat_timer = QTimer()
        heartbeat_timer.timeout.connect(lambda: tab.message_bus.publish("Data", "heartbeat", {}))
        heartbeat_timer.start(1000)  # Send heartbeat every second
        
        logger.info("Data tab process started successfully")
        
        # Start the application
        sys.exit(app.exec())
        
    except Exception as e:
        logger.error(f"Error in data tab process: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 