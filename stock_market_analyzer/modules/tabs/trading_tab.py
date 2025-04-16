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
    QLineEdit, QTabWidget, QScrollArea, QFrame
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
    """Tab for managing trading strategies and executing trades."""
    
    def __init__(self, parent=None):
        """Initialize the Trading tab."""
        super().__init__(parent)
        self.message_bus = MessageBus()
        self.logger = logging.getLogger(__name__)
        self.trading_cache = {}
        self.pending_requests = {}  # Track pending trading requests
        self.positions = {}
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the trading tab UI."""
        # Create main layout
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        
        # Trading Strategies tab
        strategies_tab = QWidget()
        strategies_layout = QVBoxLayout()
        
        # Add trading strategies UI elements
        strategy_group = QGroupBox("Trading Strategy")
        strategy_layout = QVBoxLayout()
        
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems([
            "Moving Average Crossover",
            "RSI Strategy",
            "MACD Strategy",
            "Bollinger Bands"
        ])
        strategy_layout.addWidget(self.strategy_combo)
        strategy_group.setLayout(strategy_layout)
        strategies_layout.addWidget(strategy_group)
        
        # Trading Controls group
        controls_group = QGroupBox("Trading Controls")
        controls_layout = QVBoxLayout()
        
        # Symbol input
        symbol_layout = QHBoxLayout()
        symbol_label = QLabel("Symbol:")
        self.symbol_input = QLineEdit()
        self.symbol_input.setPlaceholderText("Enter symbol (e.g., AAPL)")
        symbol_layout.addWidget(symbol_label)
        symbol_layout.addWidget(self.symbol_input)
        controls_layout.addLayout(symbol_layout)
        
        # Order type
        order_layout = QHBoxLayout()
        order_label = QLabel("Order Type:")
        self.order_type = QComboBox()
        self.order_type.addItems(["Market", "Limit", "Stop", "Stop Limit"])
        order_layout.addWidget(order_label)
        order_layout.addWidget(self.order_type)
        controls_layout.addLayout(order_layout)
        
        # Quantity input
        quantity_layout = QHBoxLayout()
        quantity_label = QLabel("Quantity:")
        self.quantity_input = QSpinBox()
        self.quantity_input.setRange(1, 10000)
        self.quantity_input.setValue(100)
        quantity_layout.addWidget(quantity_label)
        quantity_layout.addWidget(self.quantity_input)
        controls_layout.addLayout(quantity_layout)
        
        # Price input
        price_layout = QHBoxLayout()
        price_label = QLabel("Price:")
        self.price_input = QDoubleSpinBox()
        self.price_input.setRange(0.01, 1000000.00)
        self.price_input.setDecimals(2)
        self.price_input.setValue(0.00)
        price_layout.addWidget(price_label)
        price_layout.addWidget(self.price_input)
        controls_layout.addLayout(price_layout)
        
        # Buy/Sell buttons
        button_layout = QHBoxLayout()
        buy_button = QPushButton("Buy")
        buy_button.clicked.connect(lambda: self.place_order("buy"))
        sell_button = QPushButton("Sell")
        sell_button.clicked.connect(lambda: self.place_order("sell"))
        button_layout.addWidget(buy_button)
        button_layout.addWidget(sell_button)
        controls_layout.addLayout(button_layout)
        
        controls_group.setLayout(controls_layout)
        strategies_layout.addWidget(controls_group)
        
        # Order Book
        order_book_group = QGroupBox("Order Book")
        order_book_layout = QVBoxLayout()
        self.order_book = QTableWidget()
        self.order_book.setColumnCount(7)
        self.order_book.setHorizontalHeaderLabels([
            "Time", "Symbol", "Type", "Side", "Quantity", "Price", "Status"
        ])
        self.order_book.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        order_book_layout.addWidget(self.order_book)
        order_book_group.setLayout(order_book_layout)
        strategies_layout.addWidget(order_book_group)
        
        # Positions
        positions_group = QGroupBox("Positions")
        positions_layout = QVBoxLayout()
        self.position_table = QTableWidget()
        self.position_table.setColumnCount(5)
        self.position_table.setHorizontalHeaderLabels([
            "Symbol", "Quantity", "Avg Price", "Current Price", "P/L"
        ])
        self.position_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        positions_layout.addWidget(self.position_table)
        positions_group.setLayout(positions_layout)
        strategies_layout.addWidget(positions_group)
        
        # Trade History
        history_group = QGroupBox("Trade History")
        history_layout = QVBoxLayout()
        self.trade_history = QTableWidget()
        self.trade_history.setColumnCount(6)
        self.trade_history.setHorizontalHeaderLabels([
            "Time", "Symbol", "Side", "Quantity", "Price", "Total"
        ])
        self.trade_history.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        history_layout.addWidget(self.trade_history)
        history_group.setLayout(history_layout)
        strategies_layout.addWidget(history_group)
        
        strategies_tab.setLayout(strategies_layout)
        
        # Add status label
        self.status_label = QLabel("Ready")
        strategies_layout.addWidget(self.status_label)
        
        # Add tabs to tab widget
        self.tab_widget.addTab(strategies_tab, "Trading")
        
        # Add tab widget to main layout
        self.main_layout.addWidget(self.tab_widget)
        
        # Subscribe to message bus
        self.message_bus.subscribe("Trading", self.handle_message)
        
        self.logger.info("Trading tab initialized")
        
    def handle_message(self, sender: str, message_type: str, data: Any):
        """Handle incoming messages."""
        try:
            if message_type == "trading_signal":
                self.handle_trading_signal(data)
            elif message_type == "data_updated":
                ticker, data = data
                if ticker in self.positions:
                    self.update_position(ticker, data['Close'].iloc[-1])
            elif message_type == "error":
                error_msg = f"Received error from {sender}: {data}"
                self.logger.error(error_msg)
                
        except Exception as e:
            error_log = f"Error handling message in Trading tab: {str(e)}"
            self.logger.error(error_log)
            
    def handle_trading_signal(self, data: Any):
        """Handle trading signals."""
        try:
            signal_type = data[0]
            signal_data = data[1]
            
            self.trade_log.append(f"Received {signal_type} signal: {signal_data}")
            self.status_label.setText(f"New {signal_type} signal received")
            
            # Update ticker combo if needed
            if signal_data['ticker'] not in [self.ticker_combo.itemText(i) for i in range(self.ticker_combo.count())]:
                self.ticker_combo.addItem(signal_data['ticker'])
                
            # Auto-fill order details for buy signals
            if signal_type == "buy":
                self.ticker_combo.setCurrentText(signal_data['ticker'])
                self.order_type.setCurrentText("Buy")
                self.price.setValue(signal_data['current_price'])
                
        except Exception as e:
            self.logger.error(f"Error handling trading signal: {str(e)}")
            self.message_bus.publish("Trading", "error", str(e))
            
    def add_ticker(self):
        """Add a new ticker to the combo box."""
        ticker = self.ticker_edit.text().strip().upper()
        if not ticker:
            QMessageBox.warning(self, "Error", "Please enter a ticker symbol")
            return
        
        if ticker in [self.ticker_combo.itemText(i) for i in range(self.ticker_combo.count())]:
            QMessageBox.warning(self, "Error", f"Ticker {ticker} already exists")
            return
        
        # Add to combo box
        self.ticker_combo.addItem(ticker)
        self.ticker_edit.clear()
        self.status_label.setText(f"Added ticker: {ticker}")

    def remove_ticker(self):
        """Remove the selected ticker from the combo box."""
        current_index = self.ticker_combo.currentIndex()
        if current_index >= 0:
            ticker = self.ticker_combo.currentText()
            self.ticker_combo.removeItem(current_index)
            self.status_label.setText(f"Removed ticker: {ticker}")

    def place_order(self, side: str):
        """Place a new order."""
        try:
            ticker = self.ticker_combo.currentText()
            if not ticker:
                QMessageBox.warning(self, "Error", "Please select a ticker")
                return
            
            quantity = int(self.quantity.text())
            if quantity <= 0:
                QMessageBox.warning(self, "Error", "Invalid quantity")
                return
            
            order_type = self.order_type.currentText()
            price = float(self.price.text()) if order_type in ["Limit", "Stop"] else None
            
            # Create order
            order = {
                "ticker": ticker,
                "side": side,
                "type": order_type,
                "quantity": quantity,
                "price": price,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Add to order book
            self.add_to_order_book(order)
            
            # Update positions
            self.update_positions(order)
            
            self.status_label.setText(f"Placed {side} order for {ticker}")
            
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            QMessageBox.critical(self, "Error", str(e))

    def add_to_order_book(self, order: dict):
        """Add order to the order book table."""
        row = self.order_book.rowCount()
        self.order_book.insertRow(row)
        
        self.order_book.setItem(row, 0, QTableWidgetItem(order["timestamp"]))
        self.order_book.setItem(row, 1, QTableWidgetItem(order["ticker"]))
        self.order_book.setItem(row, 2, QTableWidgetItem(order["type"]))
        self.order_book.setItem(row, 3, QTableWidgetItem(order["side"]))
        self.order_book.setItem(row, 4, QTableWidgetItem(str(order["quantity"])))
        self.order_book.setItem(row, 5, QTableWidgetItem(str(order["price"]) if order["price"] else "Market"))

    def update_position(self, ticker: str, current_price: float):
        """Update position with current price."""
        if ticker in self.positions:
            self.positions[ticker]['current_price'] = current_price
            self.update_positions_table()
            
    def update_positions_table(self):
        """Update the positions table with current data."""
        self.position_table.setRowCount(len(self.positions))
        for row, (ticker, position) in enumerate(self.positions.items()):
            self.position_table.setItem(row, 0, QTableWidgetItem(ticker))
            self.position_table.setItem(row, 1, QTableWidgetItem(str(position['quantity'])))
            self.position_table.setItem(row, 2, QTableWidgetItem(f"{position['avg_price']:.2f}"))
            self.position_table.setItem(row, 3, QTableWidgetItem(f"{position['current_price']:.2f}"))
            
            # Calculate P/L
            pl = (position['current_price'] - position['avg_price']) * position['quantity']
            pl_item = QTableWidgetItem(f"{pl:.2f}")
            pl_item.setForeground(Qt.GlobalColor.green if pl >= 0 else Qt.GlobalColor.red)
            self.position_table.setItem(row, 4, pl_item)

    def publish_message(self, message_type: str, data: Any):
        """Publish a message to the message bus."""
        try:
            self.message_bus.publish("Trading", message_type, data)
        except Exception as e:
            error_log = f"Error publishing message from Trading tab: {str(e)}"
            self.logger.error(error_log)

    def execute_trade(self):
        """Execute a trade based on the current settings."""
        try:
            # Get trade parameters
            symbol = self.symbol_input.text().strip()
            quantity = self.quantity_input.value()
            order_type = self.order_type_combo.currentText()
            price = self.price_input.value()
            
            if not symbol:
                self.logger.warning("No symbol specified")
                return
                
            # Validate trade parameters
            if quantity <= 0:
                self.logger.warning("Invalid quantity")
                return
                
            if order_type == "Limit" and price <= 0:
                self.logger.warning("Invalid price for limit order")
                return
                
            # Create trade order
            order = {
                "symbol": symbol,
                "quantity": quantity,
                "order_type": order_type,
                "price": price,
                "timestamp": datetime.now().isoformat()
            }
            
            # Publish trade order
            self.message_bus.publish("Trading", "trade_order", order)
            self.logger.info(f"Trade order submitted: {order}")
            
            # Update status
            self.status_label.setText(f"Trade order submitted: {symbol} {quantity} @ {price}")
            
        except Exception as e:
            error_msg = f"Error executing trade: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            self.status_label.setText(error_msg)
            
    def handle_trade_response(self, data):
        """Handle trade execution response."""
        try:
            if "error" in data:
                self.logger.error(f"Trade execution error: {data['error']}")
                self.status_label.setText(f"Trade error: {data['error']}")
                return
                
            # Update trade history
            self.trade_history.append(data)
            self.update_trade_history()
            
            # Update status
            self.status_label.setText(f"Trade executed: {data['symbol']} {data['quantity']} @ {data['price']}")
            
        except Exception as e:
            error_msg = f"Error handling trade response: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            self.status_label.setText(error_msg)

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