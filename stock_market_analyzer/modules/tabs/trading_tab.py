import sys
import os
import logging
from typing import Any
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit
)
from PyQt6.QtCore import QObject, pyqtSignal, Qt
import pandas as pd

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(project_root)

from stock_market_analyzer.modules.message_bus import MessageBus

class TradingTab(QWidget):
    """Trading tab implementation with inter-tab communication."""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.message_bus = MessageBus()
        self.positions = {}
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the trading tab UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Trading controls
        controls_layout = QHBoxLayout()
        
        # Ticker selector
        self.ticker_combo = QComboBox()
        controls_layout.addWidget(QLabel("Ticker:"))
        controls_layout.addWidget(self.ticker_combo)
        
        # Order type selector
        self.order_type_combo = QComboBox()
        self.order_type_combo.addItems(["Buy", "Sell"])
        controls_layout.addWidget(QLabel("Order Type:"))
        controls_layout.addWidget(self.order_type_combo)
        
        # Quantity input
        self.quantity_spin = QSpinBox()
        self.quantity_spin.setRange(1, 1000)
        self.quantity_spin.setValue(1)
        controls_layout.addWidget(QLabel("Quantity:"))
        controls_layout.addWidget(self.quantity_spin)
        
        # Price input
        self.price_spin = QDoubleSpinBox()
        self.price_spin.setRange(0.01, 10000.00)
        self.price_spin.setDecimals(2)
        controls_layout.addWidget(QLabel("Price:"))
        controls_layout.addWidget(self.price_spin)
        
        # Execute button
        self.execute_button = QPushButton("Execute Order")
        self.execute_button.clicked.connect(self.execute_order)
        controls_layout.addWidget(self.execute_button)
        
        layout.addLayout(controls_layout)
        
        # Positions table
        self.positions_table = QTableWidget()
        self.positions_table.setColumnCount(5)
        self.positions_table.setHorizontalHeaderLabels([
            "Ticker", "Quantity", "Avg. Price", "Current Price", "P/L"
        ])
        self.positions_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.positions_table)
        
        # Status label
        self.status_label = QLabel("Ready for trading")
        layout.addWidget(self.status_label)
        
        # Trade log
        self.trade_log = QTextEdit()
        self.trade_log.setReadOnly(True)
        layout.addWidget(self.trade_log)
        
        # Subscribe to message bus
        self.message_bus.subscribe("Trading", self.handle_message)
        
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
                self.status_label.setText(f"Error: {data}")
                
        except Exception as e:
            self.logger.error(f"Error handling message: {str(e)}")
            self.message_bus.publish("Trading", "error", str(e))
            
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
                self.order_type_combo.setCurrentText("Buy")
                self.price_spin.setValue(signal_data['current_price'])
                
        except Exception as e:
            self.logger.error(f"Error handling trading signal: {str(e)}")
            self.message_bus.publish("Trading", "error", str(e))
            
    def execute_order(self):
        """Execute the current order."""
        try:
            self.status_label.setText("Executing order...")
            
            ticker = self.ticker_combo.currentText()
            order_type = self.order_type_combo.currentText()
            quantity = self.quantity_spin.value()
            price = self.price_spin.value()
            
            if not ticker:
                self.status_label.setText("Please select a ticker")
                return
                
            # Update positions
            if order_type == "Buy":
                if ticker in self.positions:
                    current = self.positions[ticker]
                    new_quantity = current['quantity'] + quantity
                    new_avg_price = ((current['quantity'] * current['avg_price']) + 
                                   (quantity * price)) / new_quantity
                    self.positions[ticker] = {
                        'quantity': new_quantity,
                        'avg_price': new_avg_price,
                        'current_price': price
                    }
                else:
                    self.positions[ticker] = {
                        'quantity': quantity,
                        'avg_price': price,
                        'current_price': price
                    }
            else:  # Sell
                if ticker in self.positions:
                    current = self.positions[ticker]
                    if quantity > current['quantity']:
                        self.status_label.setText("Insufficient quantity to sell")
                        return
                    new_quantity = current['quantity'] - quantity
                    if new_quantity == 0:
                        del self.positions[ticker]
                    else:
                        self.positions[ticker] = {
                            'quantity': new_quantity,
                            'avg_price': current['avg_price'],
                            'current_price': price
                        }
                else:
                    self.status_label.setText("No position to sell")
                    return
                    
            # Update positions table
            self.update_positions_table()
            
            # Publish order execution
            self.message_bus.publish("Trading", "order_executed", 
                                   (ticker, order_type, quantity, price))
            
            self.status_label.setText(f"Executed {order_type} order for {ticker}")
            
        except Exception as e:
            self.logger.error(f"Error executing order: {str(e)}")
            self.message_bus.publish("Trading", "error", str(e))
            
    def update_position(self, ticker: str, current_price: float):
        """Update position with current price."""
        if ticker in self.positions:
            self.positions[ticker]['current_price'] = current_price
            self.update_positions_table()
            
    def update_positions_table(self):
        """Update the positions table with current data."""
        self.positions_table.setRowCount(len(self.positions))
        for row, (ticker, position) in enumerate(self.positions.items()):
            self.positions_table.setItem(row, 0, QTableWidgetItem(ticker))
            self.positions_table.setItem(row, 1, QTableWidgetItem(str(position['quantity'])))
            self.positions_table.setItem(row, 2, QTableWidgetItem(f"{position['avg_price']:.2f}"))
            self.positions_table.setItem(row, 3, QTableWidgetItem(f"{position['current_price']:.2f}"))
            
            # Calculate P/L
            pl = (position['current_price'] - position['avg_price']) * position['quantity']
            pl_item = QTableWidgetItem(f"{pl:.2f}")
            pl_item.setForeground(Qt.GlobalColor.green if pl >= 0 else Qt.GlobalColor.red)
            self.positions_table.setItem(row, 4, pl_item)

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