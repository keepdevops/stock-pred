import sys
import os
import logging
import traceback
import time
import uuid
from typing import Dict, Any, List, Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QComboBox, QPushButton, QLabel, QSplitter, QApplication, QSpinBox,
    QDoubleSpinBox, QGroupBox, QCheckBox, QHeaderView, QMessageBox, QDateEdit,
    QLineEdit, QTabWidget, QScrollArea, QFrame, QTextEdit
)
from PyQt6.QtCore import Qt, QTimer, QDate
from PyQt6.QtGui import QFont, QTextCursor, QIntValidator
from .base_tab import BaseTab
from ..message_bus import MessageBus
from ..settings import Settings

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class TradingTab(BaseTab):
    """Trading tab for managing trading strategies and positions."""
    
    def __init__(self, message_bus: MessageBus, parent=None):
        """Initialize the Trading tab.
        
        Args:
            message_bus: The message bus for communication.
            parent: The parent widget.
        """
        # Initialize attributes before parent __init__
        self.settings = Settings()
        self.current_color_scheme = self.settings.get_color_scheme()
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
        self.connection_status = {}
        self.connection_start_times = {}
        self.messages_received = 0
        self.messages_sent = 0
        self.errors = 0
        self.message_latencies = []
        self.metrics = {"messages_received": 0, "messages_sent": 0, "errors": 0, "last_message_time": 0}
        self.metrics_labels = {}
        
        super().__init__(message_bus, parent)
        
        # Setup UI after parent initialization
        self.setup_ui()
        
    def _setup_message_bus_impl(self):
        """Setup message bus subscriptions."""
        try:
            # Subscribe to all relevant topics
            self.message_bus.subscribe("Trading", self.handle_trading_message)
            self.message_bus.subscribe("Data", self.handle_data_message)
            self.message_bus.subscribe("ConnectionStatus", self.handle_connection_status)
            self.message_bus.subscribe("Heartbeat", self.handle_heartbeat)
            self.message_bus.subscribe("Shutdown", self.handle_shutdown)
            
            # Send initial connection status
            self.message_bus.publish("ConnectionStatus", "status_update", {
                "tab": self.__class__.__name__,
                "status": "connected"
            })
            
            self.logger.debug("Message bus setup completed for Trading tab")
            
        except Exception as e:
            self.handle_error("Error setting up message bus subscriptions", e)
            
    def setup_ui(self):
        """Setup the UI components."""
        if self._ui_setup_done:
            return
            
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
            self.logger.info("Trading tab initialized")
            
        except Exception as e:
            self.handle_error("Error setting up UI", e)
            
    def handle_trading_message(self, sender: str, message_type: str, data: Dict[str, Any]):
        """Handle trading-related messages."""
        try:
            if not isinstance(data, dict):
                self.logger.error(f"Invalid trading message data: {data}")
                return
                
            # Update metrics
            self.metrics["messages_received"] += 1
            self.metrics["last_message_time"] = time.time()
            
            # Handle different message types
            if message_type == "position_update":
                self.handle_position_update(data)
            elif message_type == "order_update":
                self.handle_order_update(data)
            elif message_type == "error":
                self.handle_trading_error(data)
            else:
                self.logger.warning(f"Unknown trading message type: {message_type}")
                
            # Update metrics display
            self.update_metrics()
            
        except Exception as e:
            self.handle_error("Error handling trading message", e)
            
    def handle_data_message(self, sender: str, message_type: str, data: Dict[str, Any]):
        """Handle data-related messages."""
        try:
            if message_type == "ticker_data":
                self.handle_ticker_data(sender, message_type, data)
            elif message_type == "market_data":
                self.handle_market_data(sender, message_type, data)
            elif message_type == "data_error":
                error_msg = data.get("error", "Unknown error")
                self.handle_error("Data error", Exception(error_msg))
                
        except Exception as e:
            self.handle_error("Error handling data message", e)
            
    def handle_position_update(self, data: Dict[str, Any]):
        """Handle position update messages."""
        try:
            if not all(k in data for k in ["ticker", "position"]):
                self.logger.error("Missing required fields in position update")
                return
                
            ticker = data["ticker"]
            position = data["position"]
            
            # Update positions cache
            self.position_cache[position.get("id")] = position
            
            # Update positions table
            self.update_positions_table()
            
            self.logger.debug(f"Position updated for {ticker}")
            
        except Exception as e:
            self.handle_error("Error handling position update", e)
            
    def handle_order_update(self, data: Dict[str, Any]):
        """Handle order update messages."""
        try:
            if not all(k in data for k in ["ticker", "order"]):
                self.logger.error("Missing required fields in order update")
                return
                
            ticker = data["ticker"]
            order = data["order"]
            
            # Update order book
            self.add_to_order_book(order)
            
            self.logger.debug(f"Order updated for {ticker}")
            
        except Exception as e:
            self.handle_error("Error handling order update", e)
            
    def handle_trading_error(self, data: Dict[str, Any]):
        """Handle trading error messages."""
        try:
            error_msg = data.get("error", "Unknown error")
            self.status_label.setText(f"Error: {error_msg}")
            self.status_label.setStyleSheet("color: red")
            
            self.logger.error(f"Trading error: {error_msg}")
            
        except Exception as e:
            self.handle_error("Error handling trading error", e)
            
    def handle_ticker_data(self, sender: str, message_type: str, data: Dict[str, Any]):
        """Handle ticker data message."""
        try:
            ticker = data.get("ticker")
            if ticker and ticker not in [self.ticker_combo.itemText(i) for i in range(self.ticker_combo.count())]:
                self.ticker_combo.addItem(ticker)
                
        except Exception as e:
            self.handle_error("Error handling ticker data", e)
            
    def handle_market_data(self, sender: str, message_type: str, data: Dict[str, Any]):
        """Handle market data message."""
        try:
            # Update positions with current market prices
            for position in self.position_cache.values():
                if position.get("ticker") in data:
                    position["current_price"] = data[position["ticker"]]
                    self.update_positions_table()
                    
        except Exception as e:
            self.handle_error("Error handling market data", e)
            
    def enter_position(self):
        """Enter a new position."""
        try:
            ticker = self.ticker_combo.currentText()
            strategy = self.strategy_combo.currentText()
            size = self.position_size_edit.value()
            entry_price = self.entry_price_edit.value()
            stop_loss = self.stop_loss_edit.value()
            take_profit = self.take_profit_edit.value()
            
            if not ticker:
                self.status_label.setText("Please select a ticker")
                return
                
            self.status_label.setText(f"Opening position for {ticker}...")
            
            # Publish position request
            self.message_bus.publish(
                "Trading",
                "open_position",
                {
                    "ticker": ticker,
                    "strategy": strategy,
                    "size": size,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "timestamp": time.time()
                }
            )
            
        except Exception as e:
            self.handle_error("Error entering position", e)
            
    def exit_position(self):
        """Exit a position."""
        try:
            selected_items = self.positions_table.selectedItems()
            if not selected_items:
                self.status_label.setText("Please select a position to exit")
                return
                
            position_id = selected_items[0].data(Qt.ItemDataRole.UserRole)
            if position_id in self.position_cache:
                self.status_label.setText(f"Closing position {position_id}...")
                
                # Publish position close request
                self.message_bus.publish(
                    "Trading",
                    "close_position",
                    {
                        "position_id": position_id,
                        "timestamp": time.time()
                    }
                )
                
        except Exception as e:
            self.handle_error("Error exiting position", e)
            
    def add_to_order_book(self, order: Dict[str, Any]):
        """Add an order to the order book."""
        try:
            row = self.order_book.rowCount()
            self.order_book.insertRow(row)
            
            self.order_book.setItem(row, 0, QTableWidgetItem(order.get("time", "")))
            self.order_book.setItem(row, 1, QTableWidgetItem(order.get("ticker", "")))
            self.order_book.setItem(row, 2, QTableWidgetItem(order.get("type", "")))
            self.order_book.setItem(row, 3, QTableWidgetItem(order.get("side", "")))
            self.order_book.setItem(row, 4, QTableWidgetItem(str(order.get("size", 0))))
            self.order_book.setItem(row, 5, QTableWidgetItem(str(order.get("price", 0))))
            self.order_book.setItem(row, 6, QTableWidgetItem(order.get("status", "")))
            
        except Exception as e:
            self.handle_error("Error adding to order book", e)
            
    def update_positions_table(self):
        """Update the positions table."""
        try:
            self.positions_table.setRowCount(len(self.position_cache))
            
            for row, (position_id, position) in enumerate(self.position_cache.items()):
                # Ticker
                ticker_item = QTableWidgetItem(position.get("ticker", ""))
                ticker_item.setData(Qt.ItemDataRole.UserRole, position_id)
                self.positions_table.setItem(row, 0, ticker_item)
                
                # Size
                self.positions_table.setItem(row, 1, QTableWidgetItem(str(position.get("size", 0))))
                
                # Entry Price
                self.positions_table.setItem(row, 2, QTableWidgetItem(str(position.get("entry_price", 0))))
                
                # Current Price
                self.positions_table.setItem(row, 3, QTableWidgetItem(str(position.get("current_price", 0))))
                
                # P/L
                pl = (position.get("current_price", 0) - position.get("entry_price", 0)) * position.get("size", 0)
                pl_item = QTableWidgetItem(f"{pl:.2f}")
                pl_item.setForeground(Qt.GlobalColor.green if pl >= 0 else Qt.GlobalColor.red)
                self.positions_table.setItem(row, 4, pl_item)
                
                # Status
                self.positions_table.setItem(row, 5, QTableWidgetItem(position.get("status", "")))
                
        except Exception as e:
            self.handle_error("Error updating positions table", e)
            
    def handle_connection_status(self, sender: str, message_type: str, data: Dict[str, Any]):
        """Handle connection status updates."""
        try:
            if not isinstance(data, dict):
                self.logger.error(f"Invalid connection status data: {data}")
                return
                
            status = data.get("status")
            if status not in ["connected", "disconnected", "error"]:
                self.logger.error(f"Invalid connection status: {status}")
                return
                
            # Update connection status
            self.connection_status[sender] = status
            
            # Update connection start time if connected
            if status == "connected":
                self.connection_start_times[sender] = time.time()
            elif status == "disconnected":
                self.connection_start_times.pop(sender, None)
                
            # Update status label
            connected = sum(1 for s in self.connection_status.values() if s == "connected")
            total = len(self.connection_status)
            self.status_label.setText(f"Status: Connected: {connected}/{total}")
            self.status_label.setStyleSheet(
                "color: green" if connected > 0 else "color: red"
            )
            
            self.logger.debug(f"Connection status updated for {sender}: {status}")
            
        except Exception as e:
            self.handle_error("Error handling connection status", e)
            
    def handle_heartbeat(self, sender: str, message_type: str, data: Dict[str, Any]):
        """Handle heartbeat messages."""
        try:
            # Handle heartbeat message
            self.logger.debug(f"Heartbeat received from {sender}")
            
        except Exception as e:
            self.handle_error("Error handling heartbeat", e)
            
    def handle_shutdown(self, sender: str, message_type: str, data: Dict[str, Any]):
        """Handle shutdown messages."""
        try:
            # Handle shutdown message
            self.logger.debug(f"Shutdown received from {sender}")
            
        except Exception as e:
            self.handle_error("Error handling shutdown", e)
            
    def update_metrics(self):
        """Update metrics display."""
        try:
            # Calculate average latency
            if self.metrics["messages_received"] > 0:
                avg_latency = sum(self.metrics["latencies"]) / len(self.metrics["latencies"])
            else:
                avg_latency = 0
                
            # Update metrics labels
            self.metrics_labels["messages_received"].setText(
                f"Messages Received: {self.metrics['messages_received']}"
            )
            self.metrics_labels["messages_sent"].setText(
                f"Messages Sent: {self.metrics['messages_sent']}"
            )
            self.metrics_labels["errors"].setText(
                f"Errors: {self.metrics['errors']}"
            )
            self.metrics_labels["avg_latency"].setText(
                f"Average Latency: {avg_latency:.2f}ms"
            )
            
        except Exception as e:
            self.handle_error("Error updating metrics", e)
            
    def cleanup(self):
        """Clean up resources."""
        try:
            # Clear caches
            self.position_cache.clear()
            self.pending_requests.clear()
            self.connection_status.clear()
            self.connection_start_times.clear()
            
            # Reset metrics
            self.messages_received = 0
            self.messages_sent = 0
            self.errors = 0
            self.message_latencies.clear()
            
            # Call parent cleanup
            super().cleanup()
            
        except Exception as e:
            self.handle_error("Error during cleanup", e)

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