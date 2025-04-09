import sys
import os
import logging
from typing import Any
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton
)
from PyQt6.QtCore import Qt, QObject, pyqtSignal
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from stock_market_analyzer.modules.message_bus import MessageBus

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class ChartsTab(QWidget):
    """Charts tab for the stock market analyzer."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.message_bus = MessageBus()
        self.logger = logging.getLogger(__name__)
        self.current_data = None
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface."""
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # Chart controls
        controls_layout = QHBoxLayout()
        
        # Chart type selector
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems([
            "Price", "Volume", "RSI", "MACD", "Bollinger Bands"
        ])
        self.chart_type_combo.currentTextChanged.connect(self.update_chart)
        controls_layout.addWidget(QLabel("Chart Type:"))
        controls_layout.addWidget(self.chart_type_combo)
        
        # Refresh button
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_chart)
        controls_layout.addWidget(self.refresh_button)
        
        self.layout.addLayout(controls_layout)
        
        # Matplotlib figure
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        
        # Status label
        self.status_label = QLabel("Ready to display charts")
        self.layout.addWidget(self.status_label)
        
        self.logger.info("Charts tab initialized")
        
        # Subscribe to message bus
        self.message_bus.subscribe("Charts", self.handle_message)
        
    def update_chart(self):
        """Update the chart based on current data and selected type."""
        try:
            if self.current_data is None:
                self.status_label.setText("No data available")
                return
                
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            chart_type = self.chart_type_combo.currentText()
            
            if chart_type == "Price":
                self._plot_price(ax)
            elif chart_type == "Volume":
                self._plot_volume(ax)
            elif chart_type == "RSI":
                self._plot_rsi(ax)
            elif chart_type == "MACD":
                self._plot_macd(ax)
            elif chart_type == "Bollinger Bands":
                self._plot_bollinger_bands(ax)
                
            self.canvas.draw()
            self.status_label.setText(f"Displaying {chart_type} chart")
            
        except Exception as e:
            self.logger.error(f"Error updating chart: {str(e)}")
            self.message_bus.publish("Charts", "error", str(e))
            
    def _plot_price(self, ax):
        """Plot price data."""
        ax.plot(self.current_data.index, self.current_data['Close'], label='Close')
        ax.set_title('Price Chart')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True)
        
    def _plot_volume(self, ax):
        """Plot volume data."""
        ax.bar(self.current_data.index, self.current_data['Volume'], label='Volume')
        ax.set_title('Volume Chart')
        ax.set_xlabel('Date')
        ax.set_ylabel('Volume')
        ax.legend()
        ax.grid(True)
        
    def _plot_rsi(self, ax):
        """Plot RSI indicator."""
        # Calculate RSI
        delta = self.current_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        ax.plot(self.current_data.index, rsi, label='RSI')
        ax.axhline(y=70, color='r', linestyle='--', label='Overbought')
        ax.axhline(y=30, color='g', linestyle='--', label='Oversold')
        ax.set_title('RSI Indicator')
        ax.set_xlabel('Date')
        ax.set_ylabel('RSI')
        ax.legend()
        ax.grid(True)
        
    def _plot_macd(self, ax):
        """Plot MACD indicator."""
        # Calculate MACD
        exp1 = self.current_data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = self.current_data['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        
        ax.plot(self.current_data.index, macd, label='MACD')
        ax.plot(self.current_data.index, signal, label='Signal Line')
        ax.set_title('MACD Indicator')
        ax.set_xlabel('Date')
        ax.set_ylabel('MACD')
        ax.legend()
        ax.grid(True)
        
    def _plot_bollinger_bands(self, ax):
        """Plot Bollinger Bands."""
        # Calculate Bollinger Bands
        middle = self.current_data['Close'].rolling(window=20).mean()
        upper = middle + 2 * self.current_data['Close'].rolling(window=20).std()
        lower = middle - 2 * self.current_data['Close'].rolling(window=20).std()
        
        ax.plot(self.current_data.index, self.current_data['Close'], label='Price')
        ax.plot(self.current_data.index, middle, label='Middle Band')
        ax.plot(self.current_data.index, upper, label='Upper Band')
        ax.plot(self.current_data.index, lower, label='Lower Band')
        ax.set_title('Bollinger Bands')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True)
        
    def refresh_chart(self):
        """Refresh the current chart."""
        self.update_chart()
        
    def handle_message(self, sender: str, message_type: str, data: Any):
        """Handle incoming messages."""
        try:
            if message_type == "data_updated":
                ticker, stock_data = data
                self.current_data = stock_data
                self.status_label.setText(f"Received data for {ticker}")
                self.update_chart()
                
            elif message_type == "analysis_completed":
                self.status_label.setText(f"Analysis completed: {data[1]}")
                
            elif message_type == "error":
                error_msg = f"Received error from {sender}: {data}"
                self.logger.error(error_msg)
                
        except Exception as e:
            error_log = f"Error handling message in Charts tab: {str(e)}"
            self.logger.error(error_log)
            
    def publish_message(self, message_type: str, data: Any):
        """Publish a message to the message bus."""
        try:
            self.message_bus.publish("Charts", message_type, data)
        except Exception as e:
            error_log = f"Error publishing message from Charts tab: {str(e)}"
            self.logger.error(error_log)

def main():
    """Main function for the charts tab process."""
    app = QApplication(sys.argv)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting charts tab process")
    
    # Create and show the charts tab
    window = ChartsTab()
    window.setWindowTitle("Charts Tab")
    window.show()
    
    # Start the application event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 