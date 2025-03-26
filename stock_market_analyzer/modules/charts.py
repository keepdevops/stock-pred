from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import Qt, QDateTime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd
from typing import List, Dict, Any
import logging
from datetime import datetime

class StockChart(QWidget):
    """Widget for displaying stock price charts."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.setup_chart()
        
    def setup_chart(self):
        """Set up the chart components."""
        # Create matplotlib figure
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        
        # Set layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        
    def update_data(self, data: pd.DataFrame):
        """Update the chart with new data."""
        try:
            # Clear existing plot
            self.ax.clear()
            
            # Plot price data
            self.ax.plot(data['date'], data['close'], label='Price')
            
            # Customize plot
            self.ax.set_title('Stock Price')
            self.ax.set_xlabel('Date')
            self.ax.set_ylabel('Price')
            self.ax.grid(True)
            self.ax.legend()
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)
            
            # Adjust layout and draw
            self.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            self.logger.error(f"Error updating chart: {e}")
            
    def add_real_time_point(self, timestamp: QDateTime, price: float):
        """Add a real-time data point to the chart."""
        try:
            # Convert QDateTime to Python datetime
            py_datetime = timestamp.toPyDateTime() if isinstance(timestamp, QDateTime) else timestamp
            
            # Add new point to the plot
            self.ax.plot(py_datetime, price, 'ro')
            
            # Update y-axis limits if needed
            current_ylim = self.ax.get_ylim()
            if price < current_ylim[0]:
                self.ax.set_ylim(price * 0.95, current_ylim[1])
            elif price > current_ylim[1]:
                self.ax.set_ylim(current_ylim[0], price * 1.05)
                
            # Draw the updated plot
            self.canvas.draw()
            
        except Exception as e:
            self.logger.error(f"Error adding real-time point: {e}")

class TechnicalIndicatorChart(StockChart):
    """Widget for displaying technical indicators."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_indicators()
        
    def setup_indicators(self):
        """Set up technical indicator series."""
        # Create additional axes for indicators
        self.ax2 = self.ax.twinx()
        
    def update_data(self, data: pd.DataFrame):
        """Update the chart with new data and indicators."""
        try:
            # Clear existing plots
            self.ax.clear()
            self.ax2.clear()
            
            # Plot price data
            self.ax.plot(data['date'], data['close'], label='Price', color='blue')
            
            # Plot indicators if available
            if 'ma5' in data.columns:
                self.ax.plot(data['date'], data['ma5'], label='MA5', color='orange')
            if 'ma20' in data.columns:
                self.ax.plot(data['date'], data['ma20'], label='MA20', color='green')
            if 'rsi' in data.columns:
                self.ax2.plot(data['date'], data['rsi'], label='RSI', color='red')
                
            # Customize plot
            self.ax.set_title('Technical Indicators')
            self.ax.set_xlabel('Date')
            self.ax.set_ylabel('Price', color='blue')
            self.ax2.set_ylabel('RSI', color='red')
            
            # Add legends
            lines1, labels1 = self.ax.get_legend_handles_labels()
            lines2, labels2 = self.ax2.get_legend_handles_labels()
            self.ax.legend(lines1 + lines2, labels1 + labels2)
            
            # Add grid
            self.ax.grid(True)
            
            # Rotate x-axis labels
            plt.xticks(rotation=45)
            
            # Adjust layout and draw
            self.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            self.logger.error(f"Error updating technical indicators: {e}")
            
    def add_real_time_point(
        self,
        timestamp: QDateTime,
        price: float,
        indicators: Dict[str, float]
    ):
        """Add a real-time data point with indicators."""
        try:
            # Convert QDateTime to Python datetime
            py_datetime = timestamp.toPyDateTime() if isinstance(timestamp, QDateTime) else timestamp
            
            # Add price point
            self.ax.plot(py_datetime, price, 'bo', label='Price')
            
            # Add indicator points
            if 'ma5' in indicators:
                self.ax.plot(py_datetime, indicators['ma5'], 'o', color='orange', label='MA5')
            if 'ma20' in indicators:
                self.ax.plot(py_datetime, indicators['ma20'], 'o', color='green', label='MA20')
            if 'rsi' in indicators:
                self.ax2.plot(py_datetime, indicators['rsi'], 'ro', label='RSI')
                
            # Update y-axis limits if needed
            current_ylim = self.ax.get_ylim()
            if price < current_ylim[0]:
                self.ax.set_ylim(price * 0.95, current_ylim[1])
            elif price > current_ylim[1]:
                self.ax.set_ylim(current_ylim[0], price * 1.05)
                
            # Draw the updated plot
            self.canvas.draw()
            
        except Exception as e:
            self.logger.error(f"Error adding real-time indicators: {e}") 