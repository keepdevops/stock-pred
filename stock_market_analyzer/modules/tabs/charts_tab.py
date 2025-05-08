import sys
import os
import logging
import traceback
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton,
    QLabel, QSplitter, QApplication, QCheckBox, QGroupBox,
    QSpinBox, QDoubleSpinBox, QTabWidget, QScrollArea, QLineEdit, QFrame, QGridLayout, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from .base_tab import BaseTab
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
import uuid
from ..message_bus import MessageBus
from ..settings import Settings
from ..data_manager import DataManager

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class ChartsTab(BaseTab):
    """Charts tab for visualizing stock data and predictions."""
    
    def __init__(self, message_bus: MessageBus, parent=None):
        """Initialize the Charts tab.
        
        Args:
            message_bus: The message bus for communication.
            parent: The parent widget.
        """
        # Initialize attributes before parent __init__
        self.settings = Settings()
        self.current_color_scheme = self.settings.get_color_scheme()
        self._ui_setup_done = False
        self.ticker_combo = None
        self.timeframe_combo = None
        self.plot_button = None
        self.figure = None
        self.canvas = None
        self.status_label = None
        self.metrics_labels = {}
        self.data_cache = {}
        self.connection_status = {}
        self.connection_start_times = {}
        self.messages_received = 0
        self.messages_sent = 0
        self.errors = 0
        self.message_latencies = []
        self.data_manager = DataManager()
        self.data_manager.register_listener("ChartsTab", self._on_data_update)
        
        # Call parent __init__ with message bus
        super().__init__(message_bus, parent)
        self.setup_ui()
        
    def _setup_message_bus_impl(self):
        """Setup message bus subscriptions."""
        try:
            # Subscribe to all relevant topics
            self.message_bus.subscribe("Charts", self.handle_message)
            self.message_bus.subscribe("chart_data", self.handle_chart_data)
            self.message_bus.subscribe("chart_error", self.handle_chart_error)
            self.message_bus.subscribe("ConnectionStatus", self.handle_connection_status)
            self.message_bus.subscribe("Heartbeat", self.handle_heartbeat)
            self.message_bus.subscribe("Shutdown", self.handle_shutdown)
            
            # Send initial connection status
            self.message_bus.publish(
                "ConnectionStatus",
                "status_update",
                {
                    "status": "connected",
                    "sender": "ChartsTab",
                    "timestamp": time.time()
                }
            )
            
            self.logger.debug("Message bus setup completed for Charts tab")
            
        except Exception as e:
            self.handle_error("Error setting up message bus subscriptions", e)
            
    def setup_ui(self):
        """Setup the charts tab UI."""
        try:
            # Clear existing layout
            while self.main_layout.count():
                child = self.main_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
                    
            # Create chart controls group
            controls_group = QGroupBox("Chart Controls")
            controls_layout = QVBoxLayout()
            
            # Ticker selection
            ticker_layout = QHBoxLayout()
            ticker_layout.addWidget(QLabel("Ticker:"))
            self.ticker_combo = QComboBox()
            self.ticker_combo.setEditable(True)
            self.ticker_combo.setInsertPolicy(QComboBox.InsertPolicy.InsertAlphabetically)
            self.ticker_combo.currentTextChanged.connect(self.on_ticker_changed)
            ticker_layout.addWidget(self.ticker_combo)
            controls_layout.addLayout(ticker_layout)
            
            # Timeframe selection
            timeframe_layout = QHBoxLayout()
            timeframe_layout.addWidget(QLabel("Timeframe:"))
            self.timeframe_combo = QComboBox()
            self.timeframe_combo.addItems(["1D", "1W", "1M", "3M", "6M", "1Y", "5Y"])
            timeframe_layout.addWidget(self.timeframe_combo)
            controls_layout.addLayout(timeframe_layout)
            
            # Action buttons
            button_layout = QHBoxLayout()
            self.plot_button = QPushButton("Plot Chart")
            self.plot_button.clicked.connect(self.plot_chart)
            button_layout.addWidget(self.plot_button)
            controls_layout.addLayout(button_layout)
            
            controls_group.setLayout(controls_layout)
            self.main_layout.addWidget(controls_group)
            
            # Create chart display area
            self.chart_widget = QWidget()
            self.chart_layout = QVBoxLayout()
            self.chart_widget.setLayout(self.chart_layout)
            
            # Initialize matplotlib figure and canvas
            self.figure = plt.figure()
            self.canvas = FigureCanvas(self.figure)
            self.chart_layout.addWidget(self.canvas)
            
            self.main_layout.addWidget(self.chart_widget)
            
            # Setup metrics
            self.setup_metrics()
            
            self._ui_setup_done = True
            self.logger.info("Charts tab initialized")
            
        except Exception as e:
            self.handle_error("Error setting up UI", e)
            
    def on_ticker_changed(self, ticker: str):
        """Handle ticker selection change."""
        try:
            if ticker:
                self.plot_chart()
        except Exception as e:
            self.handle_error("Error handling ticker change", e)
            
    def plot_chart(self):
        """Plot the selected chart."""
        try:
            ticker = self.ticker_combo.currentText()
            if not ticker:
                self.status_label.setText("Please select a ticker")
                return
                
            # Clear previous plot
            self.figure.clear()
            
            # Get data from cache
            data = self.data_cache.get(ticker)
            if not data:
                self.status_label.setText(f"No data available for {ticker}")
                return
                
            # Create new plot
            ax = self.figure.add_subplot(111)
            ax.plot(data['dates'], data['close'])
            ax.set_title(f"{ticker} Price Chart")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            self.figure.autofmt_xdate()
            
            # Refresh canvas
            self.canvas.draw()
            self.status_label.setText(f"Plotted chart for {ticker}")
            
        except Exception as e:
            self.handle_error("Error plotting chart", e)
            
    def handle_chart_data(self, sender: str, message_type: str, data: Dict[str, Any]):
        """Handle chart data messages."""
        try:
            if not isinstance(data, dict):
                self.logger.error("Invalid chart data format")
                return
                
            ticker = data.get("ticker")
            chart_data = data.get("data")
            
            if not ticker or not chart_data:
                self.logger.error("Missing ticker or chart data")
                return
                
            # Update data cache
            self.data_cache[ticker] = chart_data
            
            # Add ticker to combo box if not already present
            if ticker not in [self.ticker_combo.itemText(i) for i in range(self.ticker_combo.count())]:
                self.ticker_combo.addItem(ticker)
                
            # Update metrics
            self.messages_received += 1
            self.update_metrics()
            
            self.status_label.setText(f"Received chart data for {ticker}")
            
        except Exception as e:
            self.handle_error("Error handling chart data", e)
            
    def handle_chart_error(self, sender: str, message_type: str, data: Dict[str, Any]):
        """Handle chart error messages."""
        try:
            if not isinstance(data, dict):
                self.logger.error("Invalid error message format")
                return
                
            error_msg = data.get("error", "Unknown error")
            self.handle_error("Chart error", Exception(error_msg))
            
            # Update metrics
            self.errors += 1
            self.update_metrics()
            
        except Exception as e:
            self.handle_error("Error handling chart error", e)
            
    def handle_connection_status(self, sender: str, message_type: str, data: Dict[str, Any]):
        """Handle connection status messages."""
        try:
            if not isinstance(data, dict):
                self.logger.error("Invalid connection status data format")
                return
                
            status = data.get("status")
            if status not in ["connected", "disconnected", "error"]:
                self.logger.error(f"Invalid connection status: {status}")
                return
                
            self.connection_status[sender] = status
            
            # Update metrics
            self.update_metrics()
            
        except Exception as e:
            self.handle_error("Error handling connection status", e)
            
    def handle_heartbeat(self, sender: str, message_type: str, data: Dict[str, Any]):
        """Handle heartbeat messages."""
        try:
            self.messages_received += 1
            self.message_latencies.append(time.time() - data.get("timestamp", time.time()))
            self.update_metrics()
        except Exception as e:
            self.handle_error("Error handling heartbeat", e)
            
    def handle_shutdown(self, sender: str, message_type: str, data: Dict[str, Any]):
        """Handle shutdown messages."""
        try:
            self._is_shutting_down = True
            self.cleanup()
        except Exception as e:
            self.handle_error("Error handling shutdown", e)
            
    def update_metrics(self):
        """Update the metrics display."""
        try:
            if not hasattr(self, 'metrics_labels') or not self.metrics_labels:
                self.logger.warning("Metrics labels not initialized")
                return
                
            # Update message counts
            self.metrics_labels["Messages Received"].setText(f"Messages Received: {self.messages_received}")
            self.metrics_labels["Messages Sent"].setText(f"Messages Sent: {self.messages_sent}")
            self.metrics_labels["Errors"].setText(f"Errors: {self.errors}")
            
            # Calculate and update average latency
            if self.message_latencies:
                avg_latency = sum(self.message_latencies) / len(self.message_latencies)
                self.metrics_labels["Average Latency"].setText(f"Average Latency: {avg_latency:.2f}s")
            else:
                self.metrics_labels["Average Latency"].setText("Average Latency: N/A")
                
        except Exception as e:
            self.handle_error("Error updating metrics", e)
            
    def _on_data_update(self, source: str, data: pd.DataFrame, metadata: Dict[str, Any]):
        """Handle data updates from the DataManager."""
        try:
            # Extract metadata
            symbols = metadata.get("symbols", [])
            row_count = metadata.get("row_count", 0)
            columns = metadata.get("columns", [])
            
            # Update ticker combo box
            current_symbols = [self.ticker_combo.itemText(i) for i in range(self.ticker_combo.count())]
            for symbol in symbols:
                if symbol not in current_symbols:
                    self.ticker_combo.addItem(symbol)
                    self.logger.info(f"Added new symbol to charts: {symbol}")
            
            # Update status
            self.status_label.setText(f"Received {row_count} rows of data with {len(symbols)} symbols")
            self.status_label.setStyleSheet("color: green")
            
            # Cache the data for charting
            self.chart_data = data
            
            # Update chart if a symbol is selected
            if self.ticker_combo.currentText():
                self.update_chart()
            
        except Exception as e:
            self.handle_error("Error handling data update", e)
            
    def publish_chart_data(self, chart_data: Dict[str, Any]):
        """Publish chart data via DataManager."""
        try:
            # Create metadata
            metadata = {
                "timestamp": time.time(),
                "chart_type": chart_data.get("type", "unknown"),
                "symbols": chart_data.get("symbols", []),
                "source": "ChartsTab"
            }
            
            # Convert chart data to DataFrame if needed
            if isinstance(chart_data.get("data"), pd.DataFrame):
                df = chart_data["data"]
            else:
                df = pd.DataFrame(chart_data)
                
            # Add data to DataManager
            self.data_manager.add_data("ChartsTab", df, metadata)
            
            # Update status
            self.status_label.setText(f"Published chart data for {len(metadata['symbols'])} symbols")
            
        except Exception as e:
            self.handle_error("Error publishing chart data", e)
            
    def cleanup(self):
        """Cleanup resources."""
        try:
            # Unregister from DataManager
            if hasattr(self, 'data_manager'):
                self.data_manager.unregister_listener(self._on_data_update)
            
            # Clear matplotlib figure
            if hasattr(self, 'figure'):
                plt.close(self.figure)
                self.figure = None
            
            # Clear canvas
            if hasattr(self, 'canvas'):
                self.canvas.deleteLater()
                self.canvas = None
            
            # Clear layout and widgets
            if hasattr(self, 'main_layout'):
                while self.main_layout.count():
                    child = self.main_layout.takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()
            
            # Clear combo boxes and labels
            if hasattr(self, 'ticker_combo'):
                self.ticker_combo.deleteLater()
                self.ticker_combo = None
                
            if hasattr(self, 'timeframe_combo'):
                self.timeframe_combo.deleteLater()
                self.timeframe_combo = None
                
            if hasattr(self, 'plot_button'):
                self.plot_button.deleteLater()
                self.plot_button = None
                
            if hasattr(self, 'chart_widget'):
                self.chart_widget.deleteLater()
                self.chart_widget = None
                
            if hasattr(self, 'chart_layout'):
                self.chart_layout.deleteLater()
                self.chart_layout = None
                
            if hasattr(self, 'main_layout'):
                self.main_layout.deleteLater()
                self.main_layout = None
                
            self.logger.info("Charts tab cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

    def handle_chart_message(self, sender: str, message_type: str, data: Dict[str, Any]):
        """Handle chart-related messages."""
        try:
            if not isinstance(data, dict):
                self.logger.error("Invalid chart message data")
                return
                
            # Process chart message and update metrics
            self.messages_received += 1
            self.update_metrics()
            
        except Exception as e:
            self.handle_error("Error handling chart message", e)
            
    def _cleanup_impl(self):
        """Clean up charts tab specific resources."""
        try:
            if self.chart_widget:
                self.chart_widget.deleteLater()
                self.chart_widget = None
                
            # Clean up matplotlib resources
            if hasattr(self, 'figure'):
                plt.close(self.figure)
            if hasattr(self, 'canvas'):
                self.canvas.deleteLater()
                
        except Exception as e:
            self.handle_error("Error during cleanup", e)

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