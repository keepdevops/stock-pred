import sys
import os
import logging
import traceback
from typing import Optional, Dict, Any
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QCheckBox, QMessageBox,
    QFileDialog, QComboBox, QProgressBar
)
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QTimer
import pandas as pd

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from stock_market_analyzer.modules.message_bus import MessageBus
from stock_market_analyzer.modules.data_stock import DataStock
from stock_market_analyzer.modules.database import DatabaseConnector

class DataTab(QWidget):
    """Data tab for displaying and managing stock data."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.message_bus = MessageBus()
        self.logger = logging.getLogger(__name__)
        self.data_stock = DataStock()
        self.db_connector = DatabaseConnector()
        
        # Set up the UI
        self.setup_ui()
        
        # Subscribe to message bus
        self.message_bus.subscribe("Data", self.handle_message)
        
        self.logger.info("Data tab initialized")
        
    def setup_ui(self):
        """Set up the user interface."""
        try:
            # Create main layout
            main_layout = QVBoxLayout()
            self.setLayout(main_layout)
            
            # Market type selection
            market_layout = QHBoxLayout()
            market_label = QLabel("Market Type:")
            self.market_type_combo = QComboBox()
            self.market_type_combo.addItems(["Stocks", "Forex", "Cryptocurrency"])
            self.market_type_combo.currentTextChanged.connect(self.update_ticker_table)
            market_layout.addWidget(market_label)
            market_layout.addWidget(self.market_type_combo)
            main_layout.addLayout(market_layout)
            
            # Ticker selection
            ticker_layout = QHBoxLayout()
            ticker_label = QLabel("Ticker:")
            self.ticker_combo = QComboBox()
            self.ticker_combo.currentTextChanged.connect(self.update_ticker_data)
            ticker_layout.addWidget(ticker_label)
            ticker_layout.addWidget(self.ticker_combo)
            main_layout.addLayout(ticker_layout)
            
            # Progress bar for loading
            self.progress_bar = QProgressBar()
            self.progress_bar.setVisible(False)
            main_layout.addWidget(self.progress_bar)
            
            # Ticker table
            self.ticker_table = QTableWidget()
            self.ticker_table.setColumnCount(5)
            self.ticker_table.setHorizontalHeaderLabels(["Symbol", "Open", "High", "Low", "Close"])
            self.ticker_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            main_layout.addWidget(self.ticker_table)
            
            # Buttons layout
            buttons_layout = QHBoxLayout()
            
            # Export buttons
            self.export_csv_btn = QPushButton("Export to CSV")
            self.export_csv_btn.clicked.connect(self.export_to_csv)
            buttons_layout.addWidget(self.export_csv_btn)
            
            self.export_json_btn = QPushButton("Export to JSON")
            self.export_json_btn.clicked.connect(self.export_to_json)
            buttons_layout.addWidget(self.export_json_btn)
            
            # Share button
            self.share_btn = QPushButton("Share Data")
            self.share_btn.clicked.connect(self.share_data)
            buttons_layout.addWidget(self.share_btn)
            
            main_layout.addLayout(buttons_layout)
            
            # Realtime updates checkbox
            self.realtime_checkbox = QCheckBox("Enable real-time updates")
            self.realtime_checkbox.stateChanged.connect(self.toggle_realtime)
            main_layout.addWidget(self.realtime_checkbox)
            
            # Status label
            self.status_label = QLabel("Ready")
            main_layout.addWidget(self.status_label)
            
        except Exception as e:
            self.logger.error(f"Error setting up Data tab UI: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
            
    def load_data(self):
        """Load data for the entered ticker."""
        try:
            ticker = self.ticker_input.text().strip().upper()
            if not ticker:
                self.status_label.setText("Please enter a ticker symbol")
                return
                
            # Add ticker to list if not already present
            if ticker not in [self.ticker_list.item(row, 0).text() 
                            for row in range(self.ticker_list.rowCount())]:
                row = self.ticker_list.rowCount()
                self.ticker_list.insertRow(row)
                self.ticker_list.setItem(row, 0, QTableWidgetItem(ticker))
                
            # Load historical data
            data = self.data_stock.get_historical_data(
                symbol=ticker,
                start_date="2020-01-01",
                end_date="2023-12-31",
                interval="1d"
            )
            
            # Update data table
            self.update_data(data)
            
            # Publish data update message
            self.message_bus.publish("Data", "data_updated", (ticker, data))
            
            self.status_label.setText(f"Loaded data for {ticker}")
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            self.message_bus.publish("Data", "error", str(e))
            
    def update_data(self, data):
        """Update the data table with new data."""
        try:
            if not data or not isinstance(data, pd.DataFrame):
                return
                
            # Clear existing data
            self.data_table.clear()
            
            # Set up table dimensions
            self.data_table.setRowCount(min(100, len(data)))
            self.data_table.setColumnCount(len(data.columns))
            
            # Set headers
            self.data_table.setHorizontalHeaderLabels(data.columns)
            
            # Populate table
            for i in range(min(100, len(data))):
                for j, col in enumerate(data.columns):
                    value = data.iloc[i, j]
                    item = QTableWidgetItem(str(value))
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    self.data_table.setItem(i, j, item)
                    
            self.status_label.setText(f"Displaying {len(data)} rows of data")
            
        except Exception as e:
            self.logger.error(f"Error updating data table: {str(e)}")
            self.logger.error(traceback.format_exc())
            
    def toggle_realtime(self, state: int):
        """Toggle real-time updates."""
        try:
            ticker = self.ticker_combo.currentText()
            if not ticker:
                self.status_label.setText("Please enter a ticker symbol")
                self.realtime_checkbox.setChecked(False)
                return
                
            if state == Qt.CheckState.Checked.value:
                # Start real-time updates
                self.data_stock.get_realtime_data(ticker)
                self.status_label.setText(f"Real-time updates enabled for {ticker}")
            else:
                # Stop real-time updates
                self.data_stock.stop_realtime_updates()
                self.status_label.setText(f"Real-time updates disabled for {ticker}")
                
        except Exception as e:
            self.logger.error(f"Error toggling real-time updates: {str(e)}")
            self.message_bus.publish("Data", "error", str(e))
            
    def handle_message(self, sender: str, message_type: str, data: Any):
        """Handle incoming messages from the message bus."""
        try:
            if message_type == "error":
                error_msg = f"Received error from {sender}: {data}"
                self.logger.error(error_msg)
                self.status_label.setText(f"Error: {data}")
                
            elif message_type == "heartbeat":
                self.logger.debug(f"Received heartbeat from {sender}")
                
            elif message_type == "data_updated":
                # Handle data updates from other tabs
                ticker, new_data = data
                self.update_data(new_data)
                self.status_label.setText(f"Received updated data for {ticker}")
                
        except Exception as e:
            error_log = f"Error handling message in Data tab: {str(e)}"
            self.logger.error(error_log)
            self.logger.error(traceback.format_exc())
            
    def show_error(self, error_msg):
        """Show an error message."""
        try:
            QMessageBox.critical(self, "Error", str(error_msg))
            self.status_label.setText(f"Error: {error_msg}")
        except Exception as e:
            self.logger.error(f"Error showing error message: {str(e)}")
            self.logger.error(traceback.format_exc())
            
    def closeEvent(self, event):
        """Handle window close event."""
        try:
            self.message_bus.unsubscribe("Data", self.handle_message)
            event.accept()
        except Exception as e:
            self.logger.error(f"Error in Data tab close event: {str(e)}")
            self.logger.error(traceback.format_exc())
            event.accept()

    def update_ticker_table(self):
        """Update the ticker table based on selected market type."""
        try:
            market_type = self.market_type_combo.currentText()
            self.status_label.setText(f"Loading {market_type} data...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # Get data from database
            data = self.db_connector.get_stock_data(market_type=market_type)
            
            # Update ticker combo box
            self.ticker_combo.clear()
            if not data.empty:
                unique_symbols = data['symbol'].unique()
                self.ticker_combo.addItems(unique_symbols)
                self.progress_bar.setValue(50)
                
                # Clear existing table
                self.ticker_table.setRowCount(0)
                
                # Get unique symbols and their latest data
                latest_data = data.sort_values('date').groupby('symbol').last().reset_index()
                
                # Populate table
                self.ticker_table.setRowCount(len(latest_data))
                for row, (_, row_data) in enumerate(latest_data.iterrows()):
                    self.ticker_table.setItem(row, 0, QTableWidgetItem(row_data['symbol']))
                    self.ticker_table.setItem(row, 1, QTableWidgetItem(str(row_data['open'])))
                    self.ticker_table.setItem(row, 2, QTableWidgetItem(str(row_data['high'])))
                    self.ticker_table.setItem(row, 3, QTableWidgetItem(str(row_data['low'])))
                    self.ticker_table.setItem(row, 4, QTableWidgetItem(str(row_data['close'])))
                
                self.status_label.setText(f"Loaded {len(latest_data)} {market_type} symbols")
            else:
                self.status_label.setText(f"No data found for {market_type}")
                
            self.progress_bar.setValue(100)
            QTimer.singleShot(1000, lambda: self.progress_bar.setVisible(False))
            
        except Exception as e:
            self.status_label.setText(f"Error loading data: {str(e)}")
            self.logger.error(f"Error updating ticker table: {e}")
            self.progress_bar.setVisible(False)
            
    def update_ticker_data(self):
        """Update the table to show data for the selected ticker."""
        try:
            ticker = self.ticker_combo.currentText()
            if not ticker:
                return
                
            self.status_label.setText(f"Loading data for {ticker}...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # Get data from database
            data = self.db_connector.get_stock_data()
            ticker_data = data[data['symbol'] == ticker]
            
            if not ticker_data.empty:
                # Sort by date
                ticker_data = ticker_data.sort_values('date', ascending=False)
                
                # Clear existing table
                self.ticker_table.setRowCount(0)
                
                # Populate table with historical data
                self.ticker_table.setRowCount(len(ticker_data))
                for row, (_, row_data) in enumerate(ticker_data.iterrows()):
                    self.ticker_table.setItem(row, 0, QTableWidgetItem(row_data['symbol']))
                    self.ticker_table.setItem(row, 1, QTableWidgetItem(str(row_data['open'])))
                    self.ticker_table.setItem(row, 2, QTableWidgetItem(str(row_data['high'])))
                    self.ticker_table.setItem(row, 3, QTableWidgetItem(str(row_data['low'])))
                    self.ticker_table.setItem(row, 4, QTableWidgetItem(str(row_data['close'])))
                
                self.status_label.setText(f"Loaded {len(ticker_data)} records for {ticker}")
            else:
                self.status_label.setText(f"No data found for {ticker}")
                
            self.progress_bar.setValue(100)
            QTimer.singleShot(1000, lambda: self.progress_bar.setVisible(False))
            
        except Exception as e:
            self.status_label.setText(f"Error loading ticker data: {str(e)}")
            self.logger.error(f"Error updating ticker data: {e}")
            self.progress_bar.setVisible(False)
            
    def export_to_csv(self):
        """Export current table data to CSV."""
        try:
            market_type = self.market_type_combo.currentText()
            ticker = self.ticker_combo.currentText()
            
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save CSV File", "", "CSV Files (*.csv)"
            )
            
            if file_path:
                data = self.db_connector.get_stock_data(market_type=market_type)
                if ticker:
                    data = data[data['symbol'] == ticker]
                    
                if not data.empty:
                    data.to_csv(file_path, index=False)
                    self.status_label.setText(f"Data exported to {file_path}")
                else:
                    self.status_label.setText("No data to export")
                    
        except Exception as e:
            self.status_label.setText(f"Error exporting to CSV: {str(e)}")
            self.logger.error(f"Error exporting to CSV: {e}")
            
    def export_to_json(self):
        """Export current table data to JSON."""
        try:
            market_type = self.market_type_combo.currentText()
            ticker = self.ticker_combo.currentText()
            
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save JSON File", "", "JSON Files (*.json)"
            )
            
            if file_path:
                data = self.db_connector.get_stock_data(market_type=market_type)
                if ticker:
                    data = data[data['symbol'] == ticker]
                    
                if not data.empty:
                    data.to_json(file_path, orient='records')
                    self.status_label.setText(f"Data exported to {file_path}")
                else:
                    self.status_label.setText("No data to export")
                    
        except Exception as e:
            self.status_label.setText(f"Error exporting to JSON: {str(e)}")
            self.logger.error(f"Error exporting to JSON: {e}")
            
    def share_data(self):
        """Share current data with other tabs."""
        try:
            market_type = self.market_type_combo.currentText()
            ticker = self.ticker_combo.currentText()
            
            data = self.db_connector.get_stock_data(market_type=market_type)
            if ticker:
                data = data[data['symbol'] == ticker]
                
            if not data.empty:
                # Convert data to dictionary for message bus
                data_dict = {
                    'market_type': market_type,
                    'ticker': ticker,
                    'data': data.to_dict(orient='records')
                }
                
                # Publish data to message bus
                self.message_bus.publish('data_update', data_dict)
                self.status_label.setText(f"Shared {len(data)} records with other tabs")
            else:
                self.status_label.setText("No data to share")
                
        except Exception as e:
            self.status_label.setText(f"Error sharing data: {str(e)}")
            self.logger.error(f"Error sharing data: {e}")

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