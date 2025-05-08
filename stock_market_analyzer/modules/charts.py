from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import Qt, QDateTime
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd
from typing import List, Dict, Any
import logging
from datetime import datetime
import numpy as np
import mplfinance as mpf
import matplotlib.dates as mdates
from PyQt6.QtCharts import QChart, QChartView, QLineSeries, QCandlestickSeries, QCandlestickSet, QBarSeries, QBarSet, QDateTimeAxis, QValueAxis

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
        
    def update_data(self, data: pd.DataFrame, title: str = None):
        """Update the chart with new data."""
        try:
            if data is None or data.empty:
                return
                
            # Make a copy to avoid modifying the original data
            df = data.copy()
            
            # Clear the axes
            self.ax.clear()
            if hasattr(self, 'volume_ax'):
                self.volume_ax.clear()
                
            # Set title if provided
            if title:
                self.ax.set_title(title)
            else:
                self.ax.set_title('Stock Price')
                
            # Check if we have multiple symbols
            multiple_symbols = 'symbol' in df.columns and len(df['symbol'].unique()) > 1
            
            if multiple_symbols:
                # Plot each symbol separately with different colors
                symbols = df['symbol'].unique()
                
                # Define colors
                colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']
                
                for i, symbol in enumerate(symbols):
                    symbol_data = df[df['symbol'] == symbol].copy()
                    
                    # Sort by date
                    if 'date' in symbol_data.columns:
                        symbol_data = symbol_data.sort_values('date')
                        
                    # Plot line chart for the symbol with appropriate color
                    color = colors[i % len(colors)]
                    self.ax.plot(symbol_data['date'], symbol_data['close'], label=symbol, color=color)
                
                # Add legend
                self.ax.legend()
            else:
                # Single symbol - use candlestick chart
                # Check if we need to set the index to date
                if not isinstance(df.index, pd.DatetimeIndex):
                    date_col = None
                    if 'date' in df.columns:
                        date_col = 'date'
                    elif 'Date' in df.columns:
                        date_col = 'Date'
                        
                    if date_col:
                        df = df.set_index(date_col)
                        df.index = pd.to_datetime(df.index)
                
                # Make sure column names match what mplfinance expects
                df.columns = [col.lower() for col in df.columns]
                
                # Ensure all required columns are present
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    # For missing columns, add dummy data based on 'close'
                    for col in missing_cols:
                        if col == 'volume':
                            df[col] = 0
                        elif 'close' in df.columns:
                            df[col] = df['close']
                
                # Use the last 100 points for better visibility
                if len(df) > 100:
                    df = df.iloc[-100:]
                
                # Set up style and colors
                mc = mpf.make_marketcolors(
                    up='green', down='red',
                    wick={'up': 'green', 'down': 'red'},
                    volume={'up': 'green', 'down': 'red'},
                    edge={'up': 'green', 'down': 'red'},
                    ohlc='inherit'
                )
                
                s = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', y_on_right=False)
                
                # Create plot
                if hasattr(self, 'volume_ax'):
                    mpf.plot(
                        df, type='candle', style=s,
                        ax=self.ax, volume=self.volume_ax,
                        warn_too_much_data=10000  # Silence warnings for large datasets
                    )
                else:
                    mpf.plot(
                        df, type='candle', style=s,
                        ax=self.ax, volume=False,
                        warn_too_much_data=10000  # Silence warnings for large datasets
                    )
            
            # Configure date formatting and labels
            self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            self.ax.set_xlabel('Date')
            self.ax.set_ylabel('Price')
            self.ax.grid(True)
            
            # Rotate x-axis labels
            plt.xticks(rotation=45)
            
            # Adjust layout
            self.figure.tight_layout()
            
            # Redraw
            self.canvas.draw()
            
        except Exception as e:
            self.logger.error(f"Error updating chart: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            
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
        
    def update_data(self, data: pd.DataFrame, title: str = None):
        """Update the chart with new data and indicators."""
        try:
            # Clear existing plots
            self.ax.clear()
            self.ax2.clear()
            
            # Set title if provided
            if title:
                self.ax.set_title(title)
            else:
                self.ax.set_title('Technical Indicators')
            
            # Check if we have multiple symbols
            multiple_symbols = 'symbol' in data.columns and len(data['symbol'].unique()) > 1
            
            if multiple_symbols:
                # Plot each symbol separately with different colors
                symbols = data['symbol'].unique()
                
                # Define colors
                colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']
                
                for i, symbol in enumerate(symbols):
                    symbol_data = data[data['symbol'] == symbol].copy()
                    
                    # Sort by date
                    if 'date' in symbol_data.columns:
                        symbol_data = symbol_data.sort_values('date')
                        
                    # Plot price data for the symbol with appropriate color
                    color = colors[i % len(colors)]
                    self.ax.plot(symbol_data['date'], symbol_data['close'], label=f"{symbol} Price", color=color)
                    
                    # Plot indicators if available for this symbol
                    if 'ma5' in symbol_data.columns:
                        self.ax.plot(symbol_data['date'], symbol_data['ma5'], 
                                   label=f"{symbol} MA5", 
                                   color=color, 
                                   linestyle='--')
                                   
                    if 'ma20' in symbol_data.columns:
                        self.ax.plot(symbol_data['date'], symbol_data['ma20'], 
                                   label=f"{symbol} MA20", 
                                   color=color, 
                                   linestyle=':')
                                   
                    if 'rsi' in symbol_data.columns:
                        rsi_color = colors[(i + 5) % len(colors)]  # Use different color for RSI
                        self.ax2.plot(symbol_data['date'], symbol_data['rsi'], 
                                    label=f"{symbol} RSI", 
                                    color=rsi_color)
            else:
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

class IchimokuCloudChart(QWidget):
    """Chart for Ichimoku Cloud indicators."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.figure = Figure(figsize=(10, 5))
        self.canvas = FigureCanvas(self.figure)
        
        # Set up the layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        
        # Create the axes
        self.ax = self.figure.add_subplot(111)
        self.figure.set_tight_layout(True)
        
    def update_data(self, data: pd.DataFrame):
        """Compatibility method that calls update_chart."""
        return self.update_chart(data)
        
    def update_chart(self, data: pd.DataFrame):
        """Update the Ichimoku Cloud chart with new data."""
        try:
            if data is None or data.empty:
                return
                
            self.logger.info("Calculating Ichimoku Cloud indicators")
            
            # Clear the axes
            self.ax.clear()
            
            # Convert the index to ensure it's datetime
            df = data.copy()
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'date' in df.columns:
                    df = df.set_index('date')
                df.index = pd.to_datetime(df.index)
            
            # Make sure column names match what mplfinance expects (lowercase)
            df.columns = [col.lower() for col in df.columns]
            
            # Calculate Ichimoku Cloud components
            # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
            high_9 = df['high'].rolling(window=9).max()
            low_9 = df['low'].rolling(window=9).min()
            tenkan_sen = (high_9 + low_9) / 2
            
            # Kijun-sen (Base Line): (26-period high + 26-period low)/2
            high_26 = df['high'].rolling(window=26).max()
            low_26 = df['low'].rolling(window=26).min()
            kijun_sen = (high_26 + low_26) / 2
            
            # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
            
            # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
            high_52 = df['high'].rolling(window=52).max()
            low_52 = df['low'].rolling(window=52).min()
            senkou_span_b = ((high_52 + low_52) / 2).shift(26)
            
            # Chikou Span (Lagging Span): Close price shifted -26
            chikou_span = df['close'].shift(-26)
            
            # Make sure we have enough data for the Ichimoku cloud
            # Only use data points where all components are available
            valid_idx = pd.Series(True, index=df.index)
            for series in [tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b]:
                valid_idx = valid_idx & series.notna()
            
            df_valid = df[valid_idx]
            
            # Use the last 100 points for better visibility
            if len(df_valid) > 100:
                df_valid = df_valid.iloc[-100:]
                # We need to also slice our indicators to match the df_valid length
                valid_idx_subset = valid_idx[valid_idx].index[-100:]
                tenkan_sen_subset = tenkan_sen.loc[valid_idx_subset]
                kijun_sen_subset = kijun_sen.loc[valid_idx_subset]
                senkou_span_a_subset = senkou_span_a.loc[valid_idx_subset]
                senkou_span_b_subset = senkou_span_b.loc[valid_idx_subset]
            else:
                tenkan_sen_subset = tenkan_sen[valid_idx]
                kijun_sen_subset = kijun_sen[valid_idx]
                senkou_span_a_subset = senkou_span_a[valid_idx]
                senkou_span_b_subset = senkou_span_b[valid_idx]
            
            # Plot price data - using the same df_valid.index for x-axis in all plots
            self.ax.plot(df_valid.index, df_valid['close'], label='Close', color='black', linewidth=1.5)
            self.ax.plot(df_valid.index, tenkan_sen_subset, label='Tenkan-sen', color='red', linewidth=1)
            self.ax.plot(df_valid.index, kijun_sen_subset, label='Kijun-sen', color='blue', linewidth=1)
            
            # Plot cloud with fill_between - using the same df_valid.index
            self.ax.fill_between(
                df_valid.index, 
                senkou_span_a_subset, 
                senkou_span_b_subset, 
                where=(senkou_span_a_subset >= senkou_span_b_subset), 
                color='lightgreen', 
                alpha=0.3, 
                label='Cloud (Bullish)'
            )
            
            self.ax.fill_between(
                df_valid.index, 
                senkou_span_a_subset, 
                senkou_span_b_subset, 
                where=(senkou_span_a_subset < senkou_span_b_subset), 
                color='lightcoral', 
                alpha=0.3, 
                label='Cloud (Bearish)'
            )
            
            # Set title and labels
            self.ax.set_title('Ichimoku Cloud Chart')
            self.ax.set_ylabel('Price')
            
            # Add legend
            self.ax.legend(loc='best')
            
            # Rotate x-axis labels for better readability
            plt.setp(self.ax.get_xticklabels(), rotation=45, ha='right')
            
            # Adjust layout
            self.figure.tight_layout()
            
            # Draw the chart
            self.canvas.draw()
            
        except Exception as e:
            self.logger.error(f"Error updating Ichimoku Cloud chart: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            # Display error message on chart
            self.ax.clear()
            self.ax.text(0.5, 0.5, f"Error rendering Ichimoku Cloud chart:\n{str(e)}", 
                       horizontalalignment='center', verticalalignment='center',
                       transform=self.ax.transAxes)
            self.canvas.draw()


class VWAPChart(StockChart):
    """Widget for displaying VWAP (Volume Weighted Average Price) chart."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        
    def setup_chart(self):
        """Set up VWAP chart."""
        # Call parent setup_chart to initialize figure and ax
        super().setup_chart()
        self.ax.set_title('VWAP (Volume Weighted Average Price)')
        self.ax2 = self.ax.twinx()  # Twin axis for volume
        
    def update_data(self, data: pd.DataFrame):
        """Update chart with VWAP indicators."""
        try:
            # Clear existing plot
            self.ax.clear()
            self.ax2.clear()
            
            # Check if data contains VWAP indicators
            if 'vwap' not in data.columns:
                self.logger.info("Calculating VWAP indicators")
                from modules.technical_indicators import TechnicalIndicators
                data = TechnicalIndicators.add_vwap(data)
            
            # Plot price
            self.ax.plot(data['date'], data['close'], label='Price', color='black', linewidth=1)
            
            # Plot VWAP
            self.ax.plot(data['date'], data['vwap'], label='VWAP', color='purple', linewidth=1.5)
            
            # Plot VWAP bands if available
            if 'vwap_upper_1' in data.columns:
                self.ax.plot(data['date'], data['vwap_upper_1'], label='VWAP Upper Band (1σ)', 
                            color='red', linestyle='--', linewidth=0.8)
                self.ax.plot(data['date'], data['vwap_lower_1'], label='VWAP Lower Band (1σ)', 
                            color='green', linestyle='--', linewidth=0.8)
            
            if 'vwap_upper_2' in data.columns:
                self.ax.plot(data['date'], data['vwap_upper_2'], label='VWAP Upper Band (2σ)', 
                            color='darkred', linestyle=':', linewidth=0.8)
                self.ax.plot(data['date'], data['vwap_lower_2'], label='VWAP Lower Band (2σ)', 
                            color='darkgreen', linestyle=':', linewidth=0.8)
            
            # Plot volume
            self.ax2.bar(data['date'], data['volume'], label='Volume', color='gray', alpha=0.3)
            
            # Set labels and title
            self.ax.set_title('VWAP (Volume Weighted Average Price)')
            self.ax.set_xlabel('Date')
            self.ax.set_ylabel('Price')
            self.ax2.set_ylabel('Volume')
            
            # Add legend
            lines1, labels1 = self.ax.get_legend_handles_labels()
            lines2, labels2 = self.ax2.get_legend_handles_labels()
            self.ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # Add grid
            self.ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels
            plt.xticks(rotation=45)
            
            # Adjust layout and draw
            self.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            self.logger.error(f"Error updating VWAP chart: {e}")


class PatternRecognitionChart(StockChart):
    """Widget for displaying pattern recognition chart."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        
    def setup_chart(self):
        """Set up pattern recognition chart."""
        # Call parent setup_chart to initialize figure and ax
        super().setup_chart()
        self.ax.set_title('Pattern Recognition')
        
    def update_data(self, data: pd.DataFrame):
        """Update chart with pattern recognition."""
        try:
            # Clear existing plot
            self.ax.clear()
            
            # Check if data contains pattern indicators
            pattern_columns = [
                'head_and_shoulders', 'double_top', 'double_bottom', 
                'ascending_triangle', 'descending_triangle', 'symmetrical_triangle'
            ]
            
            # If no pattern columns are present, calculate them
            if not any(col in data.columns for col in pattern_columns):
                self.logger.info("Calculating pattern recognition indicators")
                from modules.technical_indicators import PatternRecognition
                data = PatternRecognition.find_all_patterns(data)
            
            # Plot price data as candlesticks if OHLC data is available
            if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                import mplfinance as mpf
                
                # Create a copy of the dataframe for plotting
                df_plot = data.copy()
                
                # Make sure date is the index and is datetime
                if 'date' in df_plot.columns:
                    df_plot = df_plot.set_index('date')
                    
                # Ensure column names are lowercase
                df_plot.columns = [col.lower() for col in df_plot.columns]
                
                # Plot candlestick chart
                mpf.plot(df_plot, type='candle', ax=self.ax, show_nontrading=False, warn_too_much_data=10000)
            else:
                # Plot line chart if OHLC data is not available
                self.ax.plot(data['date'], data['close'], label='Price', color='blue')
            
            # Mark patterns on the chart
            y_height = max(data['high']) if 'high' in data.columns else max(data['close'])
            y_low = min(data['low']) if 'low' in data.columns else min(data['close'])
            y_range = y_height - y_low
            
            # Create a list to store pattern markers for legend
            pattern_markers = []
            pattern_labels = []
            
            # Mark Head and Shoulders pattern
            if 'head_shoulders' in data.columns or 'head_and_shoulders' in data.columns:
                pattern_col = 'head_shoulders' if 'head_shoulders' in data.columns else 'head_and_shoulders'
                pattern_indices = data[data[pattern_col] == 1].index
                
                if not pattern_indices.empty:
                    pattern_dates = data['date'].iloc[pattern_indices]
                    y_pos = y_height + (y_range * 0.02)  # Position slightly above the chart
                    hs_marker, = self.ax.plot(pattern_dates, [y_pos] * len(pattern_dates), 'v', 
                                         markersize=10, color='red', label='Head & Shoulders')
                    pattern_markers.append(hs_marker)
                    pattern_labels.append('Head & Shoulders')
                    
                    # Annotate the pattern
                    for date in pattern_dates:
                        self.ax.annotate('H&S', xy=(date, y_pos), xytext=(0, 5), 
                                      textcoords='offset points', ha='center', fontsize=8)
            
            # Mark Double Top pattern
            if 'double_top' in data.columns:
                pattern_indices = data[data['double_top'] == 1].index
                
                if not pattern_indices.empty:
                    pattern_dates = data['date'].iloc[pattern_indices]
                    y_pos = y_height + (y_range * 0.04)  # Position above H&S markers
                    dt_marker, = self.ax.plot(pattern_dates, [y_pos] * len(pattern_dates), '^', 
                                        markersize=10, color='purple', label='Double Top')
                    pattern_markers.append(dt_marker)
                    pattern_labels.append('Double Top')
                    
                    # Annotate the pattern
                    for date in pattern_dates:
                        self.ax.annotate('DT', xy=(date, y_pos), xytext=(0, 5), 
                                      textcoords='offset points', ha='center', fontsize=8)
            
            # Mark Double Bottom pattern
            if 'double_bottom' in data.columns:
                pattern_indices = data[data['double_bottom'] == 1].index
                
                if not pattern_indices.empty:
                    pattern_dates = data['date'].iloc[pattern_indices]
                    y_pos = y_low - (y_range * 0.02)  # Position slightly below the chart
                    db_marker, = self.ax.plot(pattern_dates, [y_pos] * len(pattern_dates), '^', 
                                        markersize=10, color='green', label='Double Bottom')
                    pattern_markers.append(db_marker)
                    pattern_labels.append('Double Bottom')
                    
                    # Annotate the pattern
                    for date in pattern_dates:
                        self.ax.annotate('DB', xy=(date, y_pos), xytext=(0, -15), 
                                      textcoords='offset points', ha='center', fontsize=8)
            
            # Mark Triangle patterns
            triangle_cols = ['ascending_triangle', 'descending_triangle', 'symmetrical_triangle']
            triangle_colors = ['green', 'red', 'blue']
            triangle_labels = ['Ascending Triangle', 'Descending Triangle', 'Symmetrical Triangle']
            
            for i, col in enumerate(triangle_cols):
                if col in data.columns:
                    pattern_indices = data[data[col] == 1].index
                    
                    if not pattern_indices.empty:
                        pattern_dates = data['date'].iloc[pattern_indices]
                        y_pos = y_height + (y_range * 0.06) + (i * y_range * 0.02)  # Stagger positions
                        tri_marker, = self.ax.plot(pattern_dates, [y_pos] * len(pattern_dates), 's', 
                                           markersize=8, color=triangle_colors[i], label=triangle_labels[i])
                        pattern_markers.append(tri_marker)
                        pattern_labels.append(triangle_labels[i])
                        
                        # Annotate the pattern
                        for date in pattern_dates:
                            self.ax.annotate(col[0].upper(), xy=(date, y_pos), xytext=(0, 5), 
                                          textcoords='offset points', ha='center', fontsize=8)
            
            # Set labels and title
            self.ax.set_title('Pattern Recognition')
            self.ax.set_xlabel('Date')
            self.ax.set_ylabel('Price')
            
            # Add pattern legend if patterns were found
            if pattern_markers:
                self.ax.legend(handles=pattern_markers, labels=pattern_labels, loc='upper left')
            
            # Add grid
            self.ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels
            plt.xticks(rotation=45)
            
            # Adjust y-axis limits to accommodate markers and annotations
            self.ax.set_ylim(y_low - (y_range * 0.05), y_height + (y_range * 0.1))
            
            # Adjust layout and draw
            self.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            self.logger.error(f"Error updating pattern recognition chart: {e}")


class SentimentChart(StockChart):
    """Widget for displaying sentiment analysis chart."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        
    def setup_chart(self):
        """Set up sentiment chart."""
        # Call parent setup_chart to initialize figure and ax
        super().setup_chart()
        self.ax.set_title('Sentiment Analysis')
        
    def update_data(self, data: pd.DataFrame, sentiment_data: Dict = None):
        """Update chart with sentiment analysis."""
        try:
            # Clear existing plot
            self.ax.clear()
            
            # Plot price data
            self.ax.plot(data['date'], data['close'], label='Price', color='blue')
            
            # Create twin axis for sentiment metrics
            ax2 = self.ax.twinx()
            
            # Plot sentiment indicators from the data
            if 'market_sentiment' in data.columns:
                ax2.plot(data['date'], data['market_sentiment'], label='Market Sentiment', color='purple')
            
            if 'social_sentiment' in data.columns:
                ax2.plot(data['date'], data['social_sentiment'], label='Social Sentiment', color='green')
                
            if 'news_sentiment' in data.columns:
                ax2.plot(data['date'], data['news_sentiment'], label='News Sentiment', color='orange')
            
            # Plot current sentiment data if provided
            if sentiment_data:
                # Create a bar chart at the right edge of the plot for current sentiment
                last_date = data['date'].iloc[-1]
                
                # Calculate the width of one day
                if len(data['date']) > 1:
                    day_width = (data['date'].iloc[-1] - data['date'].iloc[-2]).total_seconds() / (24*60*60)
                else:
                    day_width = 1
                
                # Add sentiment bars
                if 'overall_sentiment' in sentiment_data:
                    # Scale sentiment from [-1, 1] to [0, 100]
                    sentiment_value = (sentiment_data['overall_sentiment'] + 1) * 50
                    bar_color = 'green' if sentiment_value > 50 else 'red'
                    ax2.bar(last_date + pd.Timedelta(days=day_width), sentiment_value, 
                           width=day_width, color=bar_color, alpha=0.7, label='Current Sentiment')
                    
                    # Add text annotation
                    if 'sentiment_interpretation' in sentiment_data:
                        self.ax.annotate(sentiment_data['sentiment_interpretation'], 
                                      xy=(last_date + pd.Timedelta(days=day_width), data['close'].iloc[-1]),
                                      xytext=(30, 0), textcoords='offset points',
                                      arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
            
            # Set labels and title
            self.ax.set_title('Sentiment Analysis')
            self.ax.set_xlabel('Date')
            self.ax.set_ylabel('Price', color='blue')
            ax2.set_ylabel('Sentiment (0-100)', color='purple')
            
            # Set y-axis limits for sentiment
            ax2.set_ylim(0, 100)
            
            # Add horizontal lines for sentiment zones
            ax2.axhline(y=80, color='green', linestyle='--', alpha=0.3)  # Very Bullish
            ax2.axhline(y=60, color='lightgreen', linestyle='--', alpha=0.3)  # Bullish
            ax2.axhline(y=40, color='lightcoral', linestyle='--', alpha=0.3)  # Bearish
            ax2.axhline(y=20, color='red', linestyle='--', alpha=0.3)  # Very Bearish
            
            # Add sentiment zone labels
            ax2.text(data['date'].iloc[0], 90, 'Very Bullish', color='green', alpha=0.7)
            ax2.text(data['date'].iloc[0], 70, 'Bullish', color='green', alpha=0.7)
            ax2.text(data['date'].iloc[0], 50, 'Neutral', color='gray', alpha=0.7)
            ax2.text(data['date'].iloc[0], 30, 'Bearish', color='red', alpha=0.7)
            ax2.text(data['date'].iloc[0], 10, 'Very Bearish', color='red', alpha=0.7)
            
            # Add legend
            lines1, labels1 = self.ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            self.ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # Add grid
            self.ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels
            plt.xticks(rotation=45)
            
            # Adjust layout and draw
            self.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            self.logger.error(f"Error updating sentiment chart: {e}") 

class CombinedTechnicalChart(QWidget):
    """Widget for displaying a comprehensive view of all technical indicators."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.setup_chart()
        
    def setup_chart(self):
        """Set up the combined technical chart."""
        self.figure = plt.figure(figsize=(10, 12))
        
        # Create a 4x1 grid of subplots
        self.price_ax = self.figure.add_subplot(411)  # Price and cloud
        self.volume_ax = self.figure.add_subplot(412)  # Volume and VWAP
        self.momentum_ax = self.figure.add_subplot(413)  # Momentum indicators
        self.sentiment_ax = self.figure.add_subplot(414)  # Sentiment
        
        # Set titles and grid
        self.price_ax.set_title('Price and Ichimoku Cloud')
        self.volume_ax.set_title('Volume and VWAP')
        self.momentum_ax.set_title('Momentum Indicators')
        self.sentiment_ax.set_title('Sentiment Analysis')
        
        # Add grid to all
        self.price_ax.grid(True, alpha=0.3)
        self.volume_ax.grid(True, alpha=0.3)
        self.momentum_ax.grid(True, alpha=0.3)
        self.sentiment_ax.grid(True, alpha=0.3)
        
        # Add canvas to layout
        layout = QVBoxLayout(self)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
    def update_data(self, data: pd.DataFrame):
        """Update the chart with all technical indicators."""
        try:
            # Clear existing plots
            self.price_ax.clear()
            self.volume_ax.clear()
            self.momentum_ax.clear()
            self.sentiment_ax.clear()
            
            # Make sure we have all the necessary indicators
            from modules.technical_indicators import TechnicalIndicators, PatternRecognition, SentimentAnalysis
            
            # Calculate all indicators if they're not already in the data
            if not all(col in data.columns for col in ['tenkan_sen', 'vwap', 'mfi', 'market_sentiment']):
                data = TechnicalIndicators.add_all_indicators(data)
                data = PatternRecognition.find_all_patterns(data)
            
            # Get symbol name if available
            symbol = data.get('symbol', 'Unknown')
            if hasattr(data, 'name'):
                symbol = data.name
            
            # 1. Price plot with Ichimoku Cloud
            # ------------------------------
            # Plot price
            self.price_ax.plot(data['date'], data['close'], label='Price', color='black', linewidth=1.5)
            
            # Plot Ichimoku Cloud components
            if 'tenkan_sen' in data.columns:
                self.price_ax.plot(data['date'], data['tenkan_sen'], 
                                 label='Tenkan-sen', color='#0496ff', linewidth=0.8)
                self.price_ax.plot(data['date'], data['kijun_sen'], 
                                 label='Kijun-sen', color='#991515', linewidth=0.8)
                
                # Plot Cloud (Kumo)
                # Fill between span A and B to create the cloud
                self.price_ax.fill_between(
                    data['date'],
                    data['senkou_span_a'],
                    data['senkou_span_b'],
                    where=data['senkou_span_a'] >= data['senkou_span_b'],
                    color='#c9daf8',
                    alpha=0.5,
                    label='Bullish Cloud'
                )
                
                self.price_ax.fill_between(
                    data['date'],
                    data['senkou_span_a'],
                    data['senkou_span_b'],
                    where=data['senkou_span_a'] < data['senkou_span_b'],
                    color='#f4cccc',
                    alpha=0.5,
                    label='Bearish Cloud'
                )
            
            # Mark patterns if any
            patterns = ['head_shoulders', 'double_top', 'double_bottom', 
                      'ascending_triangle', 'descending_triangle', 'symmetrical_triangle']
            
            pattern_colors = {
                'head_shoulders': 'red',
                'double_top': 'purple',
                'double_bottom': 'green',
                'ascending_triangle': 'blue',
                'descending_triangle': 'orange',
                'symmetrical_triangle': 'brown'
            }
            
            for pattern in patterns:
                if pattern in data.columns:
                    pattern_dates = data.loc[data[pattern] == 1, 'date']
                    if not pattern_dates.empty:
                        y_pos = data.loc[data['date'].isin(pattern_dates), 'high'].max() * 1.02
                        self.price_ax.scatter(pattern_dates, [y_pos] * len(pattern_dates),
                                           marker='^', color=pattern_colors.get(pattern, 'red'),
                                           s=100, label=pattern.replace('_', ' ').title())
                        
                        # Annotate the pattern
                        for date in pattern_dates:
                            self.price_ax.annotate(pattern[:2].upper(), 
                                                xy=(date, y_pos), 
                                                xytext=(0, 5),
                                                textcoords='offset points',
                                                ha='center',
                                                fontsize=8)
            
            self.price_ax.set_title(f'Price and Patterns - {symbol}')
            self.price_ax.set_ylabel('Price')
            self.price_ax.legend(loc='upper left', fontsize='small')
            
            # 2. Volume and VWAP
            # ------------------------------
            # Plot volume bars
            self.volume_ax.bar(data['date'], data['volume'], color='gray', alpha=0.3, label='Volume')
            
            # Create twin axis for VWAP
            vwap_ax = self.volume_ax.twinx()
            
            # Plot VWAP if available
            if 'vwap' in data.columns:
                vwap_ax.plot(data['date'], data['vwap'], color='purple', linewidth=1.5, label='VWAP')
                
                # Plot VWAP bands if available
                if 'vwap_upper_1' in data.columns:
                    vwap_ax.plot(data['date'], data['vwap_upper_1'], color='red', linestyle='--', 
                               alpha=0.7, linewidth=0.8, label='Upper Band (1σ)')
                    vwap_ax.plot(data['date'], data['vwap_lower_1'], color='green', linestyle='--', 
                               alpha=0.7, linewidth=0.8, label='Lower Band (1σ)')
            
            self.volume_ax.set_ylabel('Volume')
            vwap_ax.set_ylabel('VWAP')
            
            # Combine legends
            lines1, labels1 = self.volume_ax.get_legend_handles_labels()
            lines2, labels2 = vwap_ax.get_legend_handles_labels()
            self.volume_ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize='small')
            
            # 3. Momentum Indicators
            # ------------------------------
            # Create a twin axis for different scales
            momentum_ax2 = self.momentum_ax.twinx()
            
            # RSI (0-100 scale)
            if 'rsi' in data.columns:
                self.momentum_ax.plot(data['date'], data['rsi'], color='blue', linewidth=1, label='RSI')
                # Add overbought/oversold lines
                self.momentum_ax.axhline(y=70, color='red', linestyle='--', alpha=0.3)
                self.momentum_ax.axhline(y=30, color='green', linestyle='--', alpha=0.3)
            
            # MFI (0-100 scale)
            if 'mfi' in data.columns:
                self.momentum_ax.plot(data['date'], data['mfi'], color='purple', linewidth=1, label='MFI')
            
            # Williams %R (0-100 scale, transformed from -100-0)
            if 'williams_r' in data.columns:
                # Transform Williams %R from -100-0 to 0-100 scale for display
                williams_r_transformed = 100 + data['williams_r'] 
                self.momentum_ax.plot(data['date'], williams_r_transformed, 
                                   color='orange', linewidth=1, label='Williams %R')
            
            # CCI (different scale)
            if 'cci' in data.columns:
                momentum_ax2.plot(data['date'], data['cci'], color='green', linewidth=1, label='CCI')
                momentum_ax2.axhline(y=100, color='red', linestyle=':', alpha=0.3)
                momentum_ax2.axhline(y=-100, color='green', linestyle=':', alpha=0.3)
            
            self.momentum_ax.set_ylabel('Oscillators (0-100)')
            self.momentum_ax.set_ylim(0, 100)
            momentum_ax2.set_ylabel('CCI')
            
            # Combine legends
            lines1, labels1 = self.momentum_ax.get_legend_handles_labels()
            lines2, labels2 = momentum_ax2.get_legend_handles_labels()
            self.momentum_ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize='small')
            
            # 4. Sentiment Analysis
            # ------------------------------
            # Plot market sentiment if available
            if 'market_sentiment' in data.columns:
                self.sentiment_ax.plot(data['date'], data['market_sentiment'], 
                                    color='blue', linewidth=1.5, label='Market Sentiment')
            
            # Get real-time sentiment if available
            try:
                sentiment_data = SentimentAnalysis.get_combined_sentiment(symbol)
                
                # Add current sentiment marker
                if sentiment_data:
                    last_date = data['date'].iloc[-1]
                    sentiment_value = (sentiment_data['overall_sentiment'] + 1) * 50  # Scale to 0-100
                    
                    # Create sentiment marker
                    self.sentiment_ax.scatter([last_date], [sentiment_value], 
                                          s=100, color='purple', 
                                          label=f"Current: {sentiment_data['sentiment_interpretation']}")
                    
                    # Add text annotation
                    self.sentiment_ax.annotate(
                        sentiment_data['sentiment_interpretation'],
                        xy=(last_date, sentiment_value),
                        xytext=(30, 0),
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2")
                    )
            except Exception as e:
                self.logger.warning(f"Error getting real-time sentiment: {e}")
            
            # Add sentiment zones
            self.sentiment_ax.axhline(y=80, color='green', linestyle='--', alpha=0.3)
            self.sentiment_ax.axhline(y=60, color='lightgreen', linestyle='--', alpha=0.3)
            self.sentiment_ax.axhline(y=40, color='lightcoral', linestyle='--', alpha=0.3)
            self.sentiment_ax.axhline(y=20, color='red', linestyle='--', alpha=0.3)
            
            # Add sentiment zone labels
            self.sentiment_ax.text(data['date'].iloc[0], 90, 'Very Bullish', color='green', alpha=0.7)
            self.sentiment_ax.text(data['date'].iloc[0], 70, 'Bullish', color='green', alpha=0.7)
            self.sentiment_ax.text(data['date'].iloc[0], 50, 'Neutral', color='gray', alpha=0.7)
            self.sentiment_ax.text(data['date'].iloc[0], 30, 'Bearish', color='red', alpha=0.7)
            self.sentiment_ax.text(data['date'].iloc[0], 10, 'Very Bearish', color='red', alpha=0.7)
            
            self.sentiment_ax.set_ylabel('Sentiment (0-100)')
            self.sentiment_ax.set_ylim(0, 100)
            self.sentiment_ax.legend(loc='upper left', fontsize='small')
            
            # Final formatting
            self.sentiment_ax.set_xlabel('Date')
            
            # Rotate x-axis labels on all subplots
            for ax in [self.price_ax, self.volume_ax, self.momentum_ax, self.sentiment_ax]:
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Adjust layout and draw
            self.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            self.logger.error(f"Error updating combined technical chart: {e}")
            import traceback
            self.logger.error(traceback.format_exc()) 

    def update_chart(self, data, chart_type='candlestick', indicators=None):
        """Update the chart with new data"""
        try:
            if data is None or len(data) == 0:
                self.logger.error("No data provided to update chart")
                return
            
            self.logger.info(f"Updating chart with {len(data)} data points")
            self.figure.clear()
            
            # Normalize column names to lowercase if needed
            column_mapping = {}
            for col in data.columns:
                if col.lower() in ['date', 'open', 'high', 'low', 'close', 'volume']:
                    column_mapping[col] = col.lower()
            
            # Create a copy of the dataframe with normalized column names
            data_normalized = data.copy()
            if column_mapping:
                data_normalized = data_normalized.rename(columns=column_mapping)
            
            # Check if required columns exist
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data_normalized.columns]
            
            if missing_columns:
                self.logger.warning(f"Missing required columns: {missing_columns}")
                # Try to find capitalized versions
                for col in missing_columns:
                    cap_col = col.capitalize()
                    if cap_col in data.columns:
                        data_normalized[col] = data[cap_col]
                        self.logger.info(f"Using {cap_col} for {col}")
            
            # Sort data by date
            if 'date' in data_normalized.columns:
                # Convert date to datetime format if it's not already
                if not pd.api.types.is_datetime64_any_dtype(data_normalized['date']):
                    data_normalized['date'] = pd.to_datetime(data_normalized['date'])
                data_normalized = data_normalized.sort_values('date')
            
            # Create main axis and volume axis
            self.ax1 = self.figure.add_subplot(111)
            self.ax2 = self.ax1.twinx()
            
            # Plot based on chart type
            if chart_type == 'candlestick':
                self._plot_candlestick(data_normalized, self.ax1)
            elif chart_type == 'line':
                self._plot_line(data_normalized, self.ax1)
            elif chart_type == 'ohlc':
                self._plot_ohlc(data_normalized, self.ax1)
            
            # Plot volume
            if 'volume' in data_normalized.columns:
                self._plot_volume(data_normalized, self.ax2)
            
            # Add indicators if requested
            if indicators:
                self.update_technical_indicators(data_normalized, indicators)
            
            self.ax1.set_xlabel('Date')
            self.ax1.set_ylabel('Price')
            self.ax2.set_ylabel('Volume')
            
            # Format date axis
            if len(data_normalized) > 0:
                self.ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                self.ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
            
            # Rotate date labels
            plt.setp(self.ax1.xaxis.get_majorticklabels(), rotation=45)
            
            # Adjust layout
            self.figure.tight_layout()
            
            # Draw the canvas
            self.canvas.draw()
            self.logger.info("Chart updated successfully")
            
        except Exception as e:
            self.logger.error(f"Error updating chart: {str(e)}")
            self.logger.error(traceback.format_exc())

    def _plot_candlestick(self, data, ax):
        """Plot candlestick chart"""
        try:
            # Convert data to the format required by mplfinance
            if 'date' not in data.columns:
                self.logger.error("Date column is required for candlestick chart")
                return
            
            # Create candlestick data
            ohlc_data = []
            for index, row in data.iterrows():
                # Convert date to mdates number
                date_num = mdates.date2num(row['date'])
                # Append data point
                ohlc_data.append([date_num, row['open'], row['high'], row['low'], row['close']])
            
            # Create candlestick
            candlestick_ohlc(ax, ohlc_data, width=0.6, colorup='green', colordown='red')
            
        except Exception as e:
            self.logger.error(f"Error plotting candlestick: {str(e)}")
            self.logger.error(traceback.format_exc())

    def _plot_ohlc(self, data, ax):
        """Plot OHLC chart"""
        try:
            # Convert data to the format required by mplfinance
            if 'date' not in data.columns:
                self.logger.error("Date column is required for OHLC chart")
                return
            
            # Create OHLC data
            ohlc_data = []
            for index, row in data.iterrows():
                # Convert date to mdates number
                date_num = mdates.date2num(row['date'])
                # Append data point
                ohlc_data.append([date_num, row['open'], row['high'], row['low'], row['close']])
            
            # Create OHLC chart
            candlestick_ohlc(ax, ohlc_data, width=0.4, colorup='green', colordown='red')
            
        except Exception as e:
            self.logger.error(f"Error plotting OHLC: {str(e)}")
            self.logger.error(traceback.format_exc())

    def _plot_line(self, data, ax):
        """Plot line chart"""
        try:
            if 'date' not in data.columns or 'close' not in data.columns:
                self.logger.error("Date and Close columns are required for line chart")
                return
            
            # Plot line
            ax.plot(data['date'], data['close'], 'b-', label='Close')
            
        except Exception as e:
            self.logger.error(f"Error plotting line: {str(e)}")
            self.logger.error(traceback.format_exc())

    def _plot_volume(self, data, ax):
        """Plot volume"""
        try:
            if 'date' not in data.columns or 'volume' not in data.columns:
                self.logger.error("Date and Volume columns are required for volume chart")
                return
            
            # Plot volume bars
            ax.bar(data['date'], data['volume'], color='#E0E0E0', alpha=0.5)
            
        except Exception as e:
            self.logger.error(f"Error plotting volume: {str(e)}")
            self.logger.error(traceback.format_exc())

    def update_technical_indicators(self, data, indicators):
        """Add technical indicators to the chart"""
        try:
            self.logger.info(f"Updating technical indicators: {indicators}")
            
            # Check if required columns exist
            if 'close' not in data.columns or 'date' not in data.columns:
                self.logger.error(f"Required columns missing for technical indicators: {'close' if 'close' not in data.columns else 'date'}")
                return
            
            # Plot selected indicators
            if 'SMA' in indicators:
                self._plot_sma(data, periods=[20, 50, 200])
            
            if 'EMA' in indicators:
                self._plot_ema(data, periods=[12, 26])
            
            if 'MACD' in indicators:
                self._plot_macd(data)
            
            if 'RSI' in indicators:
                self._plot_rsi(data)
            
            if 'Bollinger' in indicators:
                self._plot_bollinger_bands(data)
            
            if 'Ichimoku' in indicators:
                self._plot_ichimoku(data)
            
            # Add legend
            self.ax1.legend(loc='best')
            
        except Exception as e:
            self.logger.error(f"Error updating technical indicators: {str(e)}")
            self.logger.error(traceback.format_exc()) 