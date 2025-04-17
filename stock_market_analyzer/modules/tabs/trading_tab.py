import sys
import os
import logging
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QComboBox, QPushButton, QLabel, QSplitter, QApplication, QSpinBox,
    QDoubleSpinBox, QGroupBox, QCheckBox, QHeaderView, QMessageBox, QDateEdit,
    QLineEdit, QTabWidget, QScrollArea, QFrame, QTextEdit
)
from PyQt6.QtCore import Qt, QTimer, QDate
from PyQt6.QtGui import QFont, QTextCursor, QIntValidator
from modules.tabs.base_tab import BaseTab
import uuid

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ..message_bus import MessageBus

class TradingTab(BaseTab):
    """Trading tab for managing trading strategies and positions."""
    
    def __init__(self, parent=None):
        """Initialize the Trading tab."""
        # Initialize attributes before parent __init__
        self.message_bus = MessageBus()
        self.logger = logging.getLogger(__name__)
        self.current_color_scheme = "default"
        self._ui_setup_done = False
        self.main_layout = None
        self.ticker_combo = None
        self.strategy_combo = None
        self.entry_price_edit = None
        self.stop_loss_edit = None
        self.take_profit_edit = None
        self.position_size_edit = None
        self.enter_position_button = None
        self.exit_position_button = None
        self.positions_table = None
        self.status_label = None
        self.position_cache = {}
        self.pending_requests = {}
        
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
            controls_group = QGroupBox("Trading Controls")
            controls_layout = QVBoxLayout()
            
            # Top controls
            top_controls = QHBoxLayout()
            
            # Ticker selection
            ticker_layout = QHBoxLayout()
            ticker_layout.addWidget(QLabel("Ticker:"))
            self.ticker_combo = QComboBox()
            self.ticker_combo.setEditable(True)
            self.ticker_combo.setInsertPolicy(QComboBox.InsertPolicy.InsertAlphabetically)
            ticker_layout.addWidget(self.ticker_combo)
            top_controls.addLayout(ticker_layout)
            
            # Strategy selection
            strategy_layout = QHBoxLayout()
            strategy_layout.addWidget(QLabel("Strategy:"))
            self.strategy_combo = QComboBox()
            self.strategy_combo.addItems([
                "Moving Average Crossover",
                "RSI Strategy",
                "MACD Strategy",
                "Bollinger Bands"
            ])
            strategy_layout.addWidget(self.strategy_combo)
            top_controls.addLayout(strategy_layout)
            
            controls_layout.addLayout(top_controls)
            
            # Trading parameters
            params_layout = QHBoxLayout()
            
            # Position size
            size_layout = QHBoxLayout()
            size_layout.addWidget(QLabel("Position Size:"))
            self.position_size_edit = QDoubleSpinBox()
            self.position_size_edit.setRange(0.01, 1000000.00)
            self.position_size_edit.setValue(100.00)
            self.position_size_edit.setDecimals(2)
            size_layout.addWidget(self.position_size_edit)
            params_layout.addLayout(size_layout)
            
            # Entry price
            entry_layout = QHBoxLayout()
            entry_layout.addWidget(QLabel("Entry Price:"))
            self.entry_price_edit = QDoubleSpinBox()
            self.entry_price_edit.setRange(0.01, 1000000.00)
            self.entry_price_edit.setValue(0.00)
            self.entry_price_edit.setDecimals(2)
            entry_layout.addWidget(self.entry_price_edit)
            params_layout.addLayout(entry_layout)
            
            controls_layout.addLayout(params_layout)
            
            # Risk management
            risk_layout = QHBoxLayout()
            
            # Stop loss
            stop_layout = QHBoxLayout()
            stop_layout.addWidget(QLabel("Stop Loss:"))
            self.stop_loss_edit = QDoubleSpinBox()
            self.stop_loss_edit.setRange(0.01, 1000000.00)
            self.stop_loss_edit.setValue(0.00)
            self.stop_loss_edit.setDecimals(2)
            stop_layout.addWidget(self.stop_loss_edit)
            risk_layout.addLayout(stop_layout)
            
            # Take profit
            profit_layout = QHBoxLayout()
            profit_layout.addWidget(QLabel("Take Profit:"))
            self.take_profit_edit = QDoubleSpinBox()
            self.take_profit_edit.setRange(0.01, 1000000.00)
            self.take_profit_edit.setValue(0.00)
            self.take_profit_edit.setDecimals(2)
            profit_layout.addWidget(self.take_profit_edit)
            risk_layout.addLayout(profit_layout)
            
            controls_layout.addLayout(risk_layout)
            
            # Action buttons
            button_layout = QHBoxLayout()
            
            self.enter_position_button = QPushButton("Enter Position")
            self.enter_position_button.clicked.connect(self.enter_position)
            button_layout.addWidget(self.enter_position_button)
            
            self.exit_position_button = QPushButton("Exit Position")
            self.exit_position_button.clicked.connect(self.exit_position)
            button_layout.addWidget(self.exit_position_button)
            
            controls_layout.addLayout(button_layout)
            controls_group.setLayout(controls_layout)
            self.main_layout.addWidget(controls_group)
            
            # Create positions table
            positions_group = QGroupBox("Current Positions")
            positions_layout = QVBoxLayout()
            
            self.positions_table = QTableWidget()
            self.positions_table.setColumnCount(6)
            self.positions_table.setHorizontalHeaderLabels([
                "Ticker", "Size", "Entry Price", "Current Price", "P/L", "Status"
            ])
            positions_layout.addWidget(self.positions_table)
            
            positions_group.setLayout(positions_layout)
            self.main_layout.addWidget(positions_group)
            
            # Create order book
            orders_group = QGroupBox("Order Book")
            orders_layout = QVBoxLayout()
            
            self.order_book = QTableWidget()
            self.order_book.setColumnCount(7)
            self.order_book.setHorizontalHeaderLabels([
                "Time", "Ticker", "Type", "Side", "Size", "Price", "Status"
            ])
            orders_layout.addWidget(self.order_book)
            
            orders_group.setLayout(orders_layout)
            self.main_layout.addWidget(orders_group)
            
            # Status bar
            status_layout = QHBoxLayout()
            self.status_label = QLabel("Ready")
            self.status_label.setStyleSheet("color: green")
            status_layout.addWidget(self.status_label)
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
        self.message_bus.subscribe("Trading", self.handle_message)
        
    def handle_message(self, sender: str, message_type: str, data: Any):
        """Handle incoming messages."""
        try:
            if message_type == "trading_signal":
                self.handle_trading_signal(data)
            elif message_type == "data_updated":
                ticker, data = data
                if ticker in self.position_cache:
                    self.update_position(ticker, data['Close'].iloc[-1])
            elif message_type == "error":
                self.status_label.setText(f"Error: {data.get('error', 'Unknown error')}")
            elif message_type == "heartbeat":
                self.status_label.setText("Connected")
                
        except Exception as e:
            error_msg = f"Error handling message: {str(e)}"
            self.logger.error(error_msg)
            self.status_label.setText(error_msg)
            
    def handle_trading_signal(self, data: Any):
        """Handle trading signals."""
        try:
            signal_type = data[0]
            signal_data = data[1]
            
            self.status_label.setText(f"New {signal_type} signal received")
            
            # Update ticker combo if needed
            if signal_data['ticker'] not in [self.ticker_combo.itemText(i) for i in range(self.ticker_combo.count())]:
                self.ticker_combo.addItem(signal_data['ticker'])
                
            # Auto-fill order details for buy signals
            if signal_type == "buy":
                self.ticker_combo.setCurrentText(signal_data['ticker'])
                self.entry_price_edit.setValue(signal_data['current_price'])
                
        except Exception as e:
            error_msg = f"Error handling trading signal: {str(e)}"
            self.logger.error(error_msg)
            self.status_label.setText(error_msg)
            
    def enter_position(self):
        """Enter a new position."""
        try:
            ticker = self.ticker_combo.currentText()
            if not ticker:
                self.status_label.setText("Please select a ticker")
                return
                
            size = self.position_size_edit.value()
            if size <= 0:
                self.status_label.setText("Invalid position size")
                return
                
            entry_price = self.entry_price_edit.value()
            if entry_price <= 0:
                self.status_label.setText("Invalid entry price")
                return
                
            stop_loss = self.stop_loss_edit.value()
            take_profit = self.take_profit_edit.value()
            
            # Create position
            position = {
                'ticker': ticker,
                'size': size,
                'entry_price': entry_price,
                'current_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'status': 'Open'
            }
            
            # Add to position cache
            self.position_cache[ticker] = position
            
            # Update positions table
            self.update_positions_table()
            
            # Create order
            order = {
                'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'ticker': ticker,
                'type': 'Market',
                'side': 'Buy',
                'size': size,
                'price': entry_price,
                'status': 'Filled'
            }
            
            # Add to order book
            self.add_to_order_book(order)
            
            self.status_label.setText(f"Entered position: {ticker} {size} @ {entry_price}")
            
        except Exception as e:
            error_msg = f"Error entering position: {str(e)}"
            self.logger.error(error_msg)
            self.status_label.setText(error_msg)
            
    def exit_position(self):
        """Exit an existing position."""
        try:
            ticker = self.ticker_combo.currentText()
            if not ticker:
                self.status_label.setText("Please select a ticker")
                return
                
            if ticker not in self.position_cache:
                self.status_label.setText(f"No position found for {ticker}")
                return
                
            position = self.position_cache[ticker]
            size = position['size']
            current_price = position['current_price']
            
            # Create order
            order = {
                'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'ticker': ticker,
                'type': 'Market',
                'side': 'Sell',
                'size': size,
                'price': current_price,
                'status': 'Filled'
            }
            
            # Add to order book
            self.add_to_order_book(order)
            
            # Remove from position cache
            del self.position_cache[ticker]
            
            # Update positions table
            self.update_positions_table()
            
            self.status_label.setText(f"Exited position: {ticker} {size} @ {current_price}")
            
        except Exception as e:
            error_msg = f"Error exiting position: {str(e)}"
            self.logger.error(error_msg)
            self.status_label.setText(error_msg)
            
    def add_to_order_book(self, order: dict):
        """Add order to the order book table."""
        try:
            row = self.order_book.rowCount()
            self.order_book.insertRow(row)
            
            self.order_book.setItem(row, 0, QTableWidgetItem(order["time"]))
            self.order_book.setItem(row, 1, QTableWidgetItem(order["ticker"]))
            self.order_book.setItem(row, 2, QTableWidgetItem(order["type"]))
            self.order_book.setItem(row, 3, QTableWidgetItem(order["side"]))
            self.order_book.setItem(row, 4, QTableWidgetItem(str(order["size"])))
            self.order_book.setItem(row, 5, QTableWidgetItem(str(order["price"])))
            self.order_book.setItem(row, 6, QTableWidgetItem(order["status"]))
            
        except Exception as e:
            error_msg = f"Error adding to order book: {str(e)}"
            self.logger.error(error_msg)
            self.status_label.setText(error_msg)
            
    def update_position(self, ticker: str, current_price: float):
        """Update position with current price."""
        try:
            if ticker in self.position_cache:
                self.position_cache[ticker]['current_price'] = current_price
                self.update_positions_table()
                
        except Exception as e:
            error_msg = f"Error updating position: {str(e)}"
            self.logger.error(error_msg)
            self.status_label.setText(error_msg)
            
    def update_positions_table(self):
        """Update the positions table with current data."""
        try:
            self.positions_table.setRowCount(len(self.position_cache))
            for row, (ticker, position) in enumerate(self.position_cache.items()):
                self.positions_table.setItem(row, 0, QTableWidgetItem(ticker))
                self.positions_table.setItem(row, 1, QTableWidgetItem(str(position['size'])))
                self.positions_table.setItem(row, 2, QTableWidgetItem(f"{position['entry_price']:.2f}"))
                self.positions_table.setItem(row, 3, QTableWidgetItem(f"{position['current_price']:.2f}"))
                
                # Calculate P/L
                pl = (position['current_price'] - position['entry_price']) * position['size']
                pl_item = QTableWidgetItem(f"{pl:.2f}")
                pl_item.setForeground(Qt.GlobalColor.green if pl >= 0 else Qt.GlobalColor.red)
                self.positions_table.setItem(row, 4, pl_item)
                
                self.positions_table.setItem(row, 5, QTableWidgetItem(position['status']))
                
        except Exception as e:
            error_msg = f"Error updating positions table: {str(e)}"
            self.logger.error(error_msg)
            self.status_label.setText(error_msg)
            
    def cleanup(self):
        """Clean up resources."""
        try:
            super().cleanup()
            self.position_cache.clear()
            self.pending_requests.clear()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

def main():
    """Main function for the trading tab process."""
    app = QApplication(sys.argv)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting trading tab process")
    
    # Create and show the trading tab
    window = TradingTab()
    window.setWindowTitle("Trading Tab")
    window.show()
    
    # Start the application event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 