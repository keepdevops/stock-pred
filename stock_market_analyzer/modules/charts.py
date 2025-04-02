from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import Qt, QDateTime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd
from typing import List, Dict, Any
import logging
from datetime import datetime
import numpy as np

class StockChart(QWidget):
    """Widget for displaying stock price charts."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.setup_chart()
        
    def setup_chart(self):
        """Set up the chart with matplotlib."""
        self.figure = plt.figure(figsize=(8, 6))
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title('Stock Price')
        self.ax.set_xlabel('Date')
        self.ax.set_ylabel('Price')
        self.ax.grid(True)
        
        # Initialize data arrays
        self.dates = []
        self.prices = []
        self.predictions = []
        self.prediction_dates = []
        
        # Add canvas to layout
        layout = QVBoxLayout(self)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
    def update_data(self, data: pd.DataFrame):
        """Update the chart with new data."""
        try:
            # Clear existing data
            self.ax.clear()
            
            # Plot historical data
            self.ax.plot(data.index, data['close'], label='Historical', color='blue')
            
            # Plot predictions if available
            if self.predictions:
                self.ax.plot(self.prediction_dates, self.predictions, 
                           label='Predictions', color='red', linestyle='--')
            
            # Set labels and title
            self.ax.set_title('Stock Price')
            self.ax.set_xlabel('Date')
            self.ax.set_ylabel('Price')
            self.ax.grid(True)
            self.ax.legend()
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)
            
            # Adjust layout to prevent label cutoff
            self.figure.tight_layout()
            
            # Update canvas
            self.canvas.draw()
            
        except Exception as e:
            self.logger.error(f"Error updating chart: {e}")
            
    def add_real_time_point(self, timestamp: QDateTime, price: float):
        """Add a real-time data point to the chart."""
        try:
            # Convert QDateTime to Python datetime
            py_datetime = timestamp.toPyDateTime()
            
            # Add new point
            self.dates.append(py_datetime)
            self.prices.append(price)
            
            # Update plot
            self.ax.clear()
            self.ax.plot(self.dates, self.prices, label='Real-time', color='blue')
            
            # Plot predictions if available
            if self.predictions:
                self.ax.plot(self.prediction_dates, self.predictions, 
                           label='Predictions', color='red', linestyle='--')
            
            # Update labels and title
            self.ax.set_title('Stock Price')
            self.ax.set_xlabel('Date')
            self.ax.set_ylabel('Price')
            self.ax.grid(True)
            self.ax.legend()
            
            # Rotate x-axis labels
            plt.xticks(rotation=45)
            
            # Adjust layout
            self.figure.tight_layout()
            
            # Update canvas
            self.canvas.draw()
            
        except Exception as e:
            self.logger.error(f"Error adding real-time point: {e}")
            
    def add_predictions(self, predictions: np.ndarray):
        """Add predictions to the chart."""
        try:
            # Generate prediction dates (next 5 days)
            last_date = self.dates[-1] if self.dates else datetime.now()
            self.prediction_dates = [last_date + pd.Timedelta(days=i) for i in range(1, len(predictions) + 1)]
            
            # Store predictions
            self.predictions = predictions.flatten()
            
            # Update plot
            self.ax.clear()
            
            # Plot historical data
            if self.dates and self.prices:
                self.ax.plot(self.dates, self.prices, label='Historical', color='blue')
            
            # Plot predictions
            self.ax.plot(self.prediction_dates, self.predictions, 
                        label='Predictions', color='red', linestyle='--')
            
            # Update labels and title
            self.ax.set_title('Stock Price with Predictions')
            self.ax.set_xlabel('Date')
            self.ax.set_ylabel('Price')
            self.ax.grid(True)
            self.ax.legend()
            
            # Rotate x-axis labels
            plt.xticks(rotation=45)
            
            # Adjust layout
            self.figure.tight_layout()
            
            # Update canvas
            self.canvas.draw()
            
        except Exception as e:
            self.logger.error(f"Error adding predictions: {e}")

    def plot_data(self, data: pd.DataFrame, prediction_data: pd.DataFrame = None, title: str = 'Stock Price', show_volume: bool = True):
        """Plot historical data and predictions.
        
        Args:
            data: DataFrame with historical data
            prediction_data: DataFrame with prediction data
            title: Chart title
            show_volume: Whether to show volume data
        """
        try:
            # Clear existing data
            self.ax.clear()
            
            # Ensure data has required columns
            if 'date' not in data.columns or 'close' not in data.columns:
                raise ValueError("Historical data must have 'date' and 'close' columns")
                
            # Plot historical data
            self.ax.plot(data['date'], data['close'], label='Historical', color='blue')
            
            # Plot volume if requested
            if show_volume and 'volume' in data.columns:
                # Create twin axis for volume
                ax2 = self.ax.twinx()
                ax2.bar(data['date'], data['volume'], alpha=0.3, color='gray', label='Volume')
                ax2.set_ylabel('Volume')
                
                # Add volume to legend
                lines1, labels1 = self.ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                self.ax.legend(lines1 + lines2, labels1 + labels2)
            
            # Plot predictions if available
            if prediction_data is not None:
                try:
                    # Verify prediction data has required columns
                    if 'date' not in prediction_data.columns or 'close' not in prediction_data.columns:
                        self.logger.error("Prediction data missing required columns: 'date' or 'close'")
                    else:
                        # Convert date column to datetime if needed
                        if not pd.api.types.is_datetime64_any_dtype(prediction_data['date']):
                            prediction_data['date'] = pd.to_datetime(prediction_data['date'])
                            
                        # Convert close column to numeric if needed
                        if not pd.api.types.is_numeric_dtype(prediction_data['close']):
                            prediction_data['close'] = pd.to_numeric(prediction_data['close'], errors='coerce')
                            
                        # Remove any NaN values
                        prediction_data = prediction_data.dropna(subset=['close'])
                        
                        if not prediction_data.empty:
                            self.ax.plot(prediction_data['date'], prediction_data['close'],
                                       label='Predictions', color='red', linestyle='--')
                        else:
                            self.logger.warning("Prediction data is empty after cleaning")
                except Exception as pred_error:
                    self.logger.error(f"Error plotting prediction data: {pred_error}")
            
            # Set labels and title
            self.ax.set_title(title)
            self.ax.set_xlabel('Date')
            self.ax.set_ylabel('Price')
            self.ax.grid(True)
            
            # Add legend if not showing volume
            if not show_volume:
                self.ax.legend()
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)
            
            # Adjust layout to prevent label cutoff
            self.figure.tight_layout()
            
            # Update canvas
            self.canvas.draw()
            
        except Exception as e:
            self.logger.error(f"Error plotting data: {e}")

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