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
    QSpinBox, QDoubleSpinBox, QTabWidget
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

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class ChartsTab(BaseTab):
    """Charts tab for visualizing stock data, predictions, and analysis results."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.chart_cache = {}
        self.pending_requests = {}
        
    def setup_ui(self):
        """Setup the charts tab UI."""
        main_layout = QVBoxLayout(self)
        
        # Top controls
        controls_layout = QHBoxLayout()
        
        # Chart type selection
        controls_layout.addWidget(QLabel("Chart Type:"))
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems([
            "Price Chart",
            "Technical Indicators",
            "Model Predictions",
            "Analysis Results",
            "Correlation Matrix"
        ])
        self.chart_type_combo.currentIndexChanged.connect(self.on_chart_type_changed)
        controls_layout.addWidget(self.chart_type_combo)
        
        # Ticker selection
        controls_layout.addWidget(QLabel("Ticker:"))
        self.ticker_combo = QComboBox()
        controls_layout.addWidget(self.ticker_combo)
        
        # Model selection (for predictions)
        self.model_label = QLabel("Model:")
        self.model_combo = QComboBox()
        controls_layout.addWidget(self.model_label)
        controls_layout.addWidget(self.model_combo)
        
        # Time period selection
        controls_layout.addWidget(QLabel("Period:"))
        self.period_combo = QComboBox()
        self.period_combo.addItems(["1D", "1W", "1M", "3M", "6M", "1Y", "5Y", "MAX"])
        controls_layout.addWidget(self.period_combo)
        
        # Update button
        update_button = QPushButton("Update Chart")
        update_button.clicked.connect(self.update_chart)
        controls_layout.addWidget(update_button)
        
        main_layout.addLayout(controls_layout)
        
        # Chart options
        options_layout = QHBoxLayout()
        
        # Technical indicators
        self.indicators_group = QGroupBox("Technical Indicators")
        indicators_layout = QVBoxLayout()
        self.ma_check = QCheckBox("Moving Averages")
        self.rsi_check = QCheckBox("RSI")
        self.macd_check = QCheckBox("MACD")
        self.bb_check = QCheckBox("Bollinger Bands")
        indicators_layout.addWidget(self.ma_check)
        indicators_layout.addWidget(self.rsi_check)
        indicators_layout.addWidget(self.macd_check)
        indicators_layout.addWidget(self.bb_check)
        self.indicators_group.setLayout(indicators_layout)
        options_layout.addWidget(self.indicators_group)
        
        # Analysis options
        self.analysis_group = QGroupBox("Analysis Options")
        analysis_layout = QVBoxLayout()
        self.trend_check = QCheckBox("Trend Analysis")
        self.seasonality_check = QCheckBox("Seasonality")
        self.volatility_check = QCheckBox("Volatility")
        analysis_layout.addWidget(self.trend_check)
        analysis_layout.addWidget(self.seasonality_check)
        analysis_layout.addWidget(self.volatility_check)
        self.analysis_group.setLayout(analysis_layout)
        options_layout.addWidget(self.analysis_group)
        
        main_layout.addLayout(options_layout)
        
        # Chart area
        self.figure = plt.figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas)
        
        # Status label
        self.status_label = QLabel("Ready")
        main_layout.addWidget(self.status_label)
        
        # Initialize visibility
        self.update_controls_visibility()
        
    def on_chart_type_changed(self):
        """Handle chart type change."""
        self.update_controls_visibility()
        
    def update_controls_visibility(self):
        """Update visibility of controls based on chart type."""
        chart_type = self.chart_type_combo.currentText()
        
        # Show/hide model selection
        show_model = chart_type == "Model Predictions"
        self.model_label.setVisible(show_model)
        self.model_combo.setVisible(show_model)
        
        # Show/hide technical indicators
        show_indicators = chart_type in ["Price Chart", "Technical Indicators"]
        self.indicators_group.setVisible(show_indicators)
        
        # Show/hide analysis options
        show_analysis = chart_type in ["Analysis Results", "Technical Indicators"]
        self.analysis_group.setVisible(show_analysis)
        
    def update_chart(self):
        """Update the chart based on current selections."""
        try:
            chart_type = self.chart_type_combo.currentText()
            ticker = self.ticker_combo.currentText()
            
            if not ticker:
                self.status_label.setText("Please select a ticker")
                return
                
            # Clear previous chart
            self.figure.clear()
            
            if chart_type == "Price Chart":
                self.plot_price_chart(ticker)
            elif chart_type == "Technical Indicators":
                self.plot_technical_indicators(ticker)
            elif chart_type == "Model Predictions":
                self.plot_model_predictions(ticker)
            elif chart_type == "Analysis Results":
                self.plot_analysis_results(ticker)
            elif chart_type == "Correlation Matrix":
                self.plot_correlation_matrix(ticker)
                
            self.canvas.draw()
            self.status_label.setText(f"Updated {chart_type} for {ticker}")
            
        except Exception as e:
            self.logger.error(f"Error updating chart: {e}")
            self.status_label.setText(f"Error: {str(e)}")
            
    def plot_price_chart(self, ticker: str):
        """Plot price chart with selected indicators."""
        try:
            # Get data from cache or request it
            data = self.get_ticker_data(ticker)
            if data is None:
                return
                
            ax = self.figure.add_subplot(111)
            
            # Plot price
            ax.plot(data.index, data['Close'], label='Close Price')
            
            # Plot selected indicators
            if self.ma_check.isChecked():
                self.plot_moving_averages(ax, data)
            if self.bb_check.isChecked():
                self.plot_bollinger_bands(ax, data)
                
            ax.set_title(f"{ticker} Price Chart")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            ax.grid(True)
            
        except Exception as e:
            self.logger.error(f"Error plotting price chart: {e}")
            raise
            
    def plot_technical_indicators(self, ticker: str):
        """Plot technical indicators."""
        try:
            data = self.get_ticker_data(ticker)
            if data is None:
                return
                
            # Create subplots
            gs = self.figure.add_gridspec(3, 1, height_ratios=[2, 1, 1])
            ax1 = self.figure.add_subplot(gs[0])
            ax2 = self.figure.add_subplot(gs[1])
            ax3 = self.figure.add_subplot(gs[2])
            
            # Plot price
            ax1.plot(data.index, data['Close'], label='Close Price')
            
            # Plot selected indicators
            if self.ma_check.isChecked():
                self.plot_moving_averages(ax1, data)
            if self.rsi_check.isChecked():
                self.plot_rsi(ax2, data)
            if self.macd_check.isChecked():
                self.plot_macd(ax3, data)
                
            ax1.set_title(f"{ticker} Technical Analysis")
            ax1.grid(True)
            ax2.grid(True)
            ax3.grid(True)
            
        except Exception as e:
            self.logger.error(f"Error plotting technical indicators: {e}")
            raise
            
    def plot_model_predictions(self, ticker: str):
        """Plot model predictions."""
        try:
            model = self.model_combo.currentText()
            if not model:
                self.status_label.setText("Please select a model")
                return
                
            # Request predictions
            request_id = str(uuid.uuid4())
            self.pending_requests[request_id] = {
                'type': 'predictions',
                'ticker': ticker,
                'model': model,
                'timestamp': datetime.now()
            }
            
            self.message_bus.publish(
                "Charts",
                "prediction_request",
                {
                    'request_id': request_id,
                    'ticker': ticker,
                    'model': model
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error plotting model predictions: {e}")
            raise
            
    def plot_analysis_results(self, ticker: str):
        """Plot analysis results."""
        try:
            data = self.get_ticker_data(ticker)
            if data is None:
                return
                
            # Create subplots based on selected options
            num_plots = sum([
                self.trend_check.isChecked(),
                self.seasonality_check.isChecked(),
                self.volatility_check.isChecked()
            ])
            
            if num_plots == 0:
                self.status_label.setText("Please select at least one analysis option")
                return
                
            gs = self.figure.add_gridspec(num_plots, 1)
            plot_idx = 0
            
            if self.trend_check.isChecked():
                ax = self.figure.add_subplot(gs[plot_idx])
                self.plot_trend_analysis(ax, data)
                plot_idx += 1
                
            if self.seasonality_check.isChecked():
                ax = self.figure.add_subplot(gs[plot_idx])
                self.plot_seasonality(ax, data)
                plot_idx += 1
                
            if self.volatility_check.isChecked():
                ax = self.figure.add_subplot(gs[plot_idx])
                self.plot_volatility(ax, data)
                
            self.figure.suptitle(f"{ticker} Analysis Results")
            
        except Exception as e:
            self.logger.error(f"Error plotting analysis results: {e}")
            raise
            
    def plot_correlation_matrix(self, ticker: str):
        """Plot correlation matrix."""
        try:
            data = self.get_ticker_data(ticker)
            if data is None:
                return
                
            # Calculate correlations
            corr = data[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
            
            # Plot heatmap
            ax = self.figure.add_subplot(111)
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            ax.set_title(f"{ticker} Correlation Matrix")
            
        except Exception as e:
            self.logger.error(f"Error plotting correlation matrix: {e}")
            raise
            
    def process_message(self, sender: str, message_type: str, data: Any):
        """Process incoming messages."""
        try:
            if message_type == "data_response":
                self.handle_data_response(sender, data)
            elif message_type == "prediction_response":
                self.handle_prediction_response(sender, data)
            elif message_type == "analysis_response":
                self.handle_analysis_response(sender, data)
            elif message_type == "model_list":
                self.handle_model_list(sender, data)
            elif message_type == "error":
                self.handle_error(sender, data)
                
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            self.status_label.setText(f"Error: {str(e)}")
            
    def handle_prediction_response(self, sender: str, data: Any):
        """Handle prediction response from Model tab."""
        try:
            request_id = data.get('request_id')
            if request_id in self.pending_requests:
                results = data.get('results', {})
                predictions = results.get('predictions', [])
                
                # Plot predictions
                ax = self.figure.add_subplot(111)
                
                # Plot historical data
                ticker = self.pending_requests[request_id]['ticker']
                hist_data = self.get_ticker_data(ticker)
                if hist_data is not None:
                    ax.plot(hist_data.index, hist_data['Close'], label='Historical')
                
                # Plot predictions
                pred_dates = [pred['date'] for pred in predictions]
                pred_prices = [pred['price'] for pred in predictions]
                ax.plot(pred_dates, pred_prices, 'r--', label='Predictions')
                
                ax.set_title(f"{ticker} Price Predictions")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price")
                ax.legend()
                ax.grid(True)
                
                self.canvas.draw()
                
        except Exception as e:
            self.logger.error(f"Error handling prediction response: {e}")
            
    def handle_analysis_response(self, sender: str, data: Any):
        """Handle analysis response from Analysis tab."""
        try:
            results = data.get('results', {})
            ticker = data.get('ticker')
            
            # Update chart based on analysis type
            chart_type = self.chart_type_combo.currentText()
            if chart_type == "Analysis Results":
                self.plot_analysis_results(ticker)
                
        except Exception as e:
            self.logger.error(f"Error handling analysis response: {e}")
            
    def handle_model_list(self, sender: str, data: Any):
        """Handle model list update from Model tab."""
        try:
            models = data.get('models', [])
            self.model_combo.clear()
            self.model_combo.addItems(models)
            
        except Exception as e:
            self.logger.error(f"Error handling model list: {e}")
            
    def handle_data_response(self, sender: str, data: Any):
        """Handle data response from Data tab."""
        try:
            request_id = data.get('request_id')
            if request_id in self.pending_requests:
                ticker = self.pending_requests[request_id]['ticker']
                df = pd.DataFrame(data.get('data', []))
                
                if not df.empty:
                    # Convert date column to datetime and set as index
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                    
                    # Cache the data
                    self.chart_cache[ticker] = df
                    
                    # Update the chart
                    self.update_chart()
                    
        except Exception as e:
            self.logger.error(f"Error handling data response: {e}")
            
    def get_ticker_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get ticker data from cache or request it."""
        try:
            if ticker in self.chart_cache:
                return self.chart_cache[ticker]
                
            # Request data
            request_id = str(uuid.uuid4())
            self.pending_requests[request_id] = {
                'type': 'data',
                'ticker': ticker,
                'timestamp': datetime.now()
            }
            
            self.message_bus.publish(
                "Charts",
                "data_request",
                {
                    'request_id': request_id,
                    'ticker': ticker
                }
            )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting ticker data: {e}")
            return None
            
    def cleanup(self):
        """Cleanup resources."""
        super().cleanup()
        self.chart_cache.clear()
        self.pending_requests.clear()
        plt.close('all')

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