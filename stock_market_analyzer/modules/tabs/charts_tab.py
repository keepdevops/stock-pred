import sys
import os
import logging
import traceback
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton,
    QLabel, QSplitter, QApplication, QCheckBox, QGroupBox,
    QSpinBox, QDoubleSpinBox, QTabWidget, QScrollArea, QLineEdit, QFrame
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from modules.tabs.base_tab import BaseTab
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
import uuid
from modules.message_bus import MessageBus

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class ChartsTab(BaseTab):
    """Charts tab for visualizing stock data and predictions."""
    
    def __init__(self, parent=None):
        """Initialize the Charts tab."""
        # Initialize attributes before parent __init__
        self.chart_cache = {}
        self.pending_requests = {}
        self._ui_setup_done = False
        self.main_layout = None
        self.ticker_combo = None
        self.chart_type_combo = None
        self.timeframe_combo = None
        self.plot_button = None
        self.figure = None
        self.canvas = None
        self.status_label = None
        
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
            controls_group = QGroupBox("Chart Controls")
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
            
            # Chart type selection
            chart_layout = QHBoxLayout()
            chart_layout.addWidget(QLabel("Chart Type:"))
            self.chart_type_combo = QComboBox()
            self.chart_type_combo.addItems([
                "Line Chart",
                "Candlestick",
                "Volume",
                "Technical Indicators"
            ])
            chart_layout.addWidget(self.chart_type_combo)
            top_controls.addLayout(chart_layout)
            
            controls_layout.addLayout(top_controls)
            
            # Timeframe selection
            timeframe_layout = QHBoxLayout()
            timeframe_layout.addWidget(QLabel("Timeframe:"))
            self.timeframe_combo = QComboBox()
            self.timeframe_combo.addItems([
                "1D", "5D", "1M", "3M", "6M", "YTD", "1Y", "5Y", "MAX"
            ])
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
            
            # Create chart area
            chart_group = QGroupBox("Chart")
            chart_layout = QVBoxLayout()
            chart_group.setLayout(chart_layout)
            
            # Create matplotlib figure and canvas
            self.figure = Figure(figsize=(8, 6))
            self.canvas = FigureCanvas(self.figure)
            chart_layout.addWidget(self.canvas)
            
            self.main_layout.addWidget(chart_group)
            
            # Create status bar
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
        self.message_bus.subscribe("Charts", self.handle_message)
        
    def plot_chart(self):
        """Plot the selected chart."""
        try:
            ticker = self.ticker_combo.currentText()
            chart_type = self.chart_type_combo.currentText()
            timeframe = self.timeframe_combo.currentText()
            
            if not ticker:
                self.status_label.setText("Please select a ticker")
                return
                
            # Generate request ID
            request_id = str(uuid.uuid4())
            
            # Create request
            request = {
                'request_id': request_id,
                'ticker': ticker,
                'chart_type': chart_type,
                'timeframe': timeframe,
                'timestamp': datetime.now()
            }
            
            # Add to pending requests
            self.pending_requests[request_id] = request
            
            # Publish request
            self.message_bus.publish(
                "Charts",
                "chart_request",
                request
            )
            
            # Update UI
            self.status_label.setText(f"Plotting {chart_type} for {ticker}")
            self.plot_button.setEnabled(False)
            
        except Exception as e:
            error_msg = f"Error plotting chart: {str(e)}"
            self.logger.error(error_msg)
            self.status_label.setText(error_msg)
            
    def handle_message(self, sender: str, message_type: str, data: Any):
        """Handle incoming messages."""
        try:
            if message_type == "chart_response":
                self.handle_chart_response(sender, data)
            elif message_type == "error":
                self.status_label.setText(f"Error: {data.get('error', 'Unknown error')}")
                
        except Exception as e:
            error_msg = f"Error handling message: {str(e)}"
            self.logger.error(error_msg)
            self.status_label.setText(error_msg)
            
    def handle_chart_response(self, sender: str, data: Any):
        """Handle chart response."""
        try:
            request_id = data.get('request_id')
            if request_id in self.pending_requests:
                ticker = self.pending_requests[request_id]['ticker']
                chart_data = data.get('chart_data', {})
                
                if chart_data:
                    # Clear previous chart
                    self.figure.clear()
                    
                    # Plot new chart
                    ax = self.figure.add_subplot(111)
                    
                    if 'dates' in chart_data and 'prices' in chart_data:
                        ax.plot(chart_data['dates'], chart_data['prices'])
                        ax.set_title(f"{ticker} Price Chart")
                        ax.set_xlabel("Date")
                        ax.set_ylabel("Price")
                        ax.grid(True)
                        
                    self.canvas.draw()
                    self.status_label.setText(f"Chart plotted for {ticker}")
                else:
                    self.status_label.setText(f"No data available for {ticker}")
                    
                del self.pending_requests[request_id]
                self.plot_button.setEnabled(True)
                
        except Exception as e:
            error_msg = f"Error handling chart response: {str(e)}"
            self.logger.error(error_msg)
            self.status_label.setText(error_msg)
            
    def cleanup(self):
        """Cleanup resources."""
        try:
            super().cleanup()
            self.chart_cache.clear()
            self.pending_requests.clear()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            
    def closeEvent(self, event):
        """Handle the close event."""
        self.cleanup()
        super().closeEvent(event)

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