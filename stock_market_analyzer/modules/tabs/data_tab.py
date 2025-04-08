import sys
import os
import logging
from typing import Any
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QCheckBox
)
from PyQt6.QtCore import Qt, QObject, pyqtSignal
import pandas as pd

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(project_root)

from stock_market_analyzer.modules.message_bus import MessageBus
from stock_market_analyzer.modules.data_stock import DataStock

class DataTab(QWidget):
    """Data tab implementation with inter-tab communication."""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.message_bus = MessageBus()
        self.data_stock = DataStock()
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the data tab UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Input controls
        controls_layout = QHBoxLayout()
        
        # Ticker input
        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("Enter ticker symbol")
        controls_layout.addWidget(QLabel("Ticker:"))
        controls_layout.addWidget(self.ticker_input)
        
        # Load button
        self.load_button = QPushButton("Load Data")
        self.load_button.clicked.connect(self.load_data)
        controls_layout.addWidget(self.load_button)
        
        layout.addLayout(controls_layout)
        
        # Ticker list
        self.ticker_list = QTableWidget()
        self.ticker_list.setColumnCount(1)
        self.ticker_list.setHorizontalHeaderLabels(["Tickers"])
        self.ticker_list.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.ticker_list)
        
        # Data table
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(7)
        self.data_table.setHorizontalHeaderLabels([
            "Date", "Open", "High", "Low", "Close", "Volume", "Adj Close"
        ])
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.data_table)
        
        # Realtime updates checkbox
        self.realtime_checkbox = QCheckBox("Enable real-time updates")
        self.realtime_checkbox.stateChanged.connect(self.toggle_realtime)
        layout.addWidget(self.realtime_checkbox)
        
        # Status label
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        
        # Subscribe to message bus
        self.message_bus.subscribe("Data", self.handle_message)
        
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
                ticker,
                start_date="2020-01-01",
                end_date="2023-12-31",
                interval="1d"
            )
            
            # Update data table
            self.update_data_table(data)
            
            # Publish data update message
            self.message_bus.publish("Data", "data_updated", (ticker, data))
            
            self.status_label.setText(f"Loaded data for {ticker}")
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            self.message_bus.publish("Data", "error", str(e))
            
    def update_data_table(self, data: pd.DataFrame):
        """Update the data table with new data."""
        self.data_table.setRowCount(len(data))
        for row, (index, values) in enumerate(data.iterrows()):
            self.data_table.setItem(row, 0, QTableWidgetItem(str(index.date())))
            self.data_table.setItem(row, 1, QTableWidgetItem(f"{values['Open']:.2f}"))
            self.data_table.setItem(row, 2, QTableWidgetItem(f"{values['High']:.2f}"))
            self.data_table.setItem(row, 3, QTableWidgetItem(f"{values['Low']:.2f}"))
            self.data_table.setItem(row, 4, QTableWidgetItem(f"{values['Close']:.2f}"))
            self.data_table.setItem(row, 5, QTableWidgetItem(f"{values['Volume']:,.0f}"))
            self.data_table.setItem(row, 6, QTableWidgetItem(f"{values['Adj Close']:.2f}"))
            
    def toggle_realtime(self, state: int):
        """Toggle real-time updates."""
        try:
            ticker = self.ticker_input.text().strip().upper()
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
        """Handle incoming messages."""
        try:
            if message_type == "analysis_request":
                ticker = data
                self.load_data_for_ticker(ticker)
            elif message_type == "error":
                self.status_label.setText(f"Error: {data}")
        except Exception as e:
            self.logger.error(f"Error handling message: {str(e)}")
            self.message_bus.publish("Data", "error", str(e))

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