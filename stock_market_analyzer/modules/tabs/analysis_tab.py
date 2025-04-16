import sys
import os
import logging
import traceback
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QComboBox, QPushButton, QLabel, QFileDialog, QMessageBox, QListWidget,
    QListWidgetItem, QSplitter, QApplication, QSpinBox, QCheckBox,
    QGroupBox, QRadioButton, QTabWidget, QScrollArea, QLineEdit, QTextEdit, QFrame
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from modules.tabs.base_tab import BaseTab
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import uuid
from modules.message_bus import MessageBus

class AnalysisTab(BaseTab):
    """Analysis tab for processing and analyzing stock data."""
    
    def __init__(self, parent=None):
        """Initialize the Analysis tab."""
        super().__init__(parent)
        self.message_bus = MessageBus()
        self.logger = logging.getLogger(__name__)
        self.setup_ui()
        self.analysis_cache = {}
        self.pending_requests = {}
        self.data_cache = {}  # Cache for processed data
        
    def setup_ui(self):
        """Setup the analysis tab UI."""
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create scroll area for each tab
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        
        # Analysis tab
        analysis_tab = QWidget()
        analysis_layout = QVBoxLayout()
        
        # Add analysis UI elements
        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("Enter ticker symbol")
        analysis_layout.addWidget(self.ticker_input)
        
        self.analysis_type_combo = QComboBox()
        self.analysis_type_combo.addItems(["Technical", "Fundamental", "Sentiment"])
        analysis_layout.addWidget(self.analysis_type_combo)
        
        analyze_button = QPushButton("Analyze")
        analyze_button.clicked.connect(self.analyze)
        analysis_layout.addWidget(analyze_button)
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        analysis_layout.addWidget(self.result_text)
        
        analysis_tab.setLayout(analysis_layout)
        
        # Add tabs to tab widget
        self.tab_widget.addTab(analysis_tab, "Analysis")
        
        # Add tab widget to main layout
        self.main_layout.addWidget(self.tab_widget)
        
        # Subscribe to message bus
        self.message_bus.subscribe("Analysis", self.handle_message)
        
        self.logger.info("Analysis tab initialized")
        
    def hide_all_options(self):
        """Hide all analysis option widgets."""
        self.technical_options.hide()
        self.statistical_options.hide()
        self.timeseries_options.hide()
        self.correlation_options.hide()
        
    def update_analysis_options(self):
        """Update visible analysis options based on selected type."""
        self.hide_all_options()
        
        analysis_type = self.analysis_type_combo.currentText()
        if analysis_type == "Technical Analysis":
            self.technical_options.show()
        elif analysis_type == "Fundamental Analysis":
            self.statistical_options.show()
        elif analysis_type == "Correlation Analysis":
            self.correlation_options.show()
        elif analysis_type == "Volatility Analysis":
            self.timeseries_options.show()
            
    def run_analysis(self):
        """Run the selected analysis on the data."""
        try:
            # Get data from message bus
            data = self.get_data_from_bus()
            if data is None:
                self.status_label.setText("No data available for analysis")
                return
                
            analysis_type = self.analysis_type_combo.currentText()
            results = {}
            
            if analysis_type == "Technical Analysis":
                results = self.run_technical_analysis(data)
            elif analysis_type == "Fundamental Analysis":
                results = self.run_statistical_analysis(data)
            elif analysis_type == "Correlation Analysis":
                results = self.run_correlation_analysis(data)
            elif analysis_type == "Volatility Analysis":
                results = self.run_timeseries_analysis(data)
                
            # Update results table
            self.update_results_table(results)
            
            # Generate summary
            summary = self.generate_summary(results)
            self.summary_text.setText(summary)
            
            # Cache results
            self.data_cache[analysis_type] = results
            
            self.status_label.setText("Analysis completed")
            
        except Exception as e:
            self.logger.error(f"Error running analysis: {e}")
            self.status_label.setText(f"Error: {str(e)}")
            
    def run_technical_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run technical analysis on the data."""
        results = {}
        
        if self.ma_check.isChecked():
            # Calculate moving averages
            data['MA20'] = data['Close'].rolling(window=20).mean()
            data['MA50'] = data['Close'].rolling(window=50).mean()
            data['MA200'] = data['Close'].rolling(window=200).mean()
            results['Moving Averages'] = {
                'MA20': data['MA20'].iloc[-1],
                'MA50': data['MA50'].iloc[-1],
                'MA200': data['MA200'].iloc[-1]
            }
            
        if self.rsi_check.isChecked():
            # Calculate RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            results['RSI'] = {
                'Current': data['RSI'].iloc[-1],
                'Overbought': data['RSI'] > 70,
                'Oversold': data['RSI'] < 30
            }
            
        if self.macd_check.isChecked():
            # Calculate MACD
            exp1 = data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = exp1 - exp2
            data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
            results['MACD'] = {
                'MACD': data['MACD'].iloc[-1],
                'Signal': data['Signal'].iloc[-1],
                'Histogram': data['MACD'].iloc[-1] - data['Signal'].iloc[-1]
            }
            
        if self.bollinger_check.isChecked():
            # Calculate Bollinger Bands
            data['MA20'] = data['Close'].rolling(window=20).mean()
            data['STD20'] = data['Close'].rolling(window=20).std()
            data['Upper'] = data['MA20'] + (data['STD20'] * 2)
            data['Lower'] = data['MA20'] - (data['STD20'] * 2)
            results['Bollinger Bands'] = {
                'Upper': data['Upper'].iloc[-1],
                'Middle': data['MA20'].iloc[-1],
                'Lower': data['Lower'].iloc[-1]
            }
            
        return results
        
    def run_statistical_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run statistical analysis on the data."""
        results = {}
        
        if self.descriptive_check.isChecked():
            # Calculate descriptive statistics
            results['Descriptive Statistics'] = {
                'Mean': data['Close'].mean(),
                'Median': data['Close'].median(),
                'Std Dev': data['Close'].std(),
                'Skewness': data['Close'].skew(),
                'Kurtosis': data['Close'].kurtosis()
            }
            
        if self.normality_check.isChecked():
            # Test for normality
            stat, p = stats.normaltest(data['Close'])
            results['Normality Test'] = {
                'Statistic': stat,
                'p-value': p,
                'Normal': p > 0.05
            }
            
        if self.stationarity_check.isChecked():
            # Test for stationarity
            result = adfuller(data['Close'])
            results['Stationarity Test'] = {
                'ADF Statistic': result[0],
                'p-value': result[1],
                'Stationary': result[1] < 0.05
            }
            
        return results
        
    def run_timeseries_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run time series analysis on the data."""
        results = {}
        
        if self.trend_check.isChecked():
            # Analyze trend
            trend = np.polyfit(range(len(data)), data['Close'], 1)
            results['Trend Analysis'] = {
                'Slope': trend[0],
                'Intercept': trend[1],
                'Direction': 'Upward' if trend[0] > 0 else 'Downward'
            }
            
        if self.seasonality_check.isChecked():
            # Analyze seasonality
            decomposition = seasonal_decompose(data['Close'], period=12)
            results['Seasonality'] = {
                'Seasonal Strength': decomposition.seasonal.std() / data['Close'].std(),
                'Trend Strength': decomposition.trend.std() / data['Close'].std()
            }
            
        if self.decomposition_check.isChecked():
            # Decompose time series
            decomposition = seasonal_decompose(data['Close'], period=12)
            results['Decomposition'] = {
                'Trend': decomposition.trend.dropna().tolist(),
                'Seasonal': decomposition.seasonal.dropna().tolist(),
                'Residual': decomposition.resid.dropna().tolist()
            }
            
        return results
        
    def run_correlation_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run correlation analysis on the data."""
        results = {}
        
        if self.pearson_check.isChecked():
            # Calculate Pearson correlation
            corr_matrix = data[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
            results['Pearson Correlation'] = corr_matrix.to_dict()
            
        if self.spearman_check.isChecked():
            # Calculate Spearman correlation
            corr_matrix = data[['Open', 'High', 'Low', 'Close', 'Volume']].corr(method='spearman')
            results['Spearman Correlation'] = corr_matrix.to_dict()
            
        return results
        
    def update_results_table(self, results: Dict[str, Any]):
        """Update the results table with analysis results."""
        try:
            self.results_table.setRowCount(0)
            
            row = 0
            for category, metrics in results.items():
                for metric, value in metrics.items():
                    if isinstance(value, dict):
                        for sub_metric, sub_value in value.items():
                            self.results_table.insertRow(row)
                            self.results_table.setItem(row, 0, QTableWidgetItem(f"{category} - {metric} - {sub_metric}"))
                            self.results_table.setItem(row, 1, QTableWidgetItem(str(sub_value)))
                            self.results_table.setItem(row, 2, QTableWidgetItem(self.get_metric_description(sub_metric)))
                            row += 1
                    else:
                        self.results_table.insertRow(row)
                        self.results_table.setItem(row, 0, QTableWidgetItem(f"{category} - {metric}"))
                        self.results_table.setItem(row, 1, QTableWidgetItem(str(value)))
                        self.results_table.setItem(row, 2, QTableWidgetItem(self.get_metric_description(metric)))
                        row += 1
                        
        except Exception as e:
            self.logger.error(f"Error updating results table: {e}")
            
    def get_metric_description(self, metric: str) -> str:
        """Get description for a metric."""
        descriptions = {
            'MA20': '20-day Moving Average',
            'MA50': '50-day Moving Average',
            'MA200': '200-day Moving Average',
            'RSI': 'Relative Strength Index',
            'MACD': 'Moving Average Convergence Divergence',
            'Upper': 'Upper Bollinger Band',
            'Lower': 'Lower Bollinger Band',
            'Mean': 'Arithmetic Mean',
            'Median': 'Median Value',
            'Std Dev': 'Standard Deviation',
            'Skewness': 'Measure of Asymmetry',
            'Kurtosis': 'Measure of Tailedness',
            'Slope': 'Trend Slope',
            'Direction': 'Trend Direction',
            'Seasonal Strength': 'Strength of Seasonal Component',
            'Trend Strength': 'Strength of Trend Component'
        }
        return descriptions.get(metric, '')
        
    def generate_summary(self, results: Dict[str, Any]) -> str:
        """Generate a summary of the analysis results."""
        summary = []
        
        for category, metrics in results.items():
            summary.append(f"\n{category}:")
            for metric, value in metrics.items():
                if isinstance(value, dict):
                    for sub_metric, sub_value in value.items():
                        summary.append(f"  {sub_metric}: {sub_value}")
                else:
                    summary.append(f"  {metric}: {value}")
                    
        return "\n".join(summary)
        
    def send_to_charts(self):
        """Send analysis results to the Charts tab."""
        try:
            analysis_type = self.analysis_type_combo.currentText()
            if analysis_type not in self.data_cache:
                self.status_label.setText("No analysis results to send")
                return
                
            # Publish results to message bus
            self.message_bus.publish(
                "Analysis",
                "analysis_results",
                {
                    'type': analysis_type,
                    'results': self.data_cache[analysis_type]
                }
            )
            
            self.status_label.setText("Results sent to Charts tab")
            
        except Exception as e:
            self.logger.error(f"Error sending to charts: {e}")
            self.status_label.setText(f"Error: {str(e)}")
            
    def get_data_from_bus(self) -> Optional[pd.DataFrame]:
        """Get data from the message bus."""
        try:
            # Subscribe to data updates
            self.message_bus.subscribe("Data", self.handle_data_update)
            return None  # Data will be received asynchronously
            
        except Exception as e:
            self.logger.error(f"Error getting data from bus: {e}")
            return None
            
    def handle_data_update(self, sender: str, message_type: str, data: Any):
        """Handle data updates from the message bus."""
        try:
            if message_type == "data_updated":
                # Process the received data
                self.process_received_data(data)
                
        except Exception as e:
            self.logger.error(f"Error handling data update: {e}")
            
    def process_received_data(self, data: Dict[str, Any]):
        """Process received data from the message bus."""
        try:
            # Convert data to DataFrame
            df = pd.DataFrame(data['data'])
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            # Store processed data
            self.data_cache['raw_data'] = df
            
            self.status_label.setText("Data received and processed")
            
        except Exception as e:
            self.logger.error(f"Error processing received data: {e}")
            
    def process_message(self, sender: str, message_type: str, data: Any):
        """Process incoming messages."""
        try:
            if message_type == "data_updated":
                self.handle_data_update(sender, data)
            elif message_type == "data_response":
                self.handle_data_response(sender, data)
            elif message_type == "analysis_request":
                self.handle_analysis_request(sender, data)
            elif message_type == "analysis_response":
                self.handle_analysis_response(sender, data)
            elif message_type == "error":
                self.handle_error(sender, data)
                
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            self.status_label.setText(f"Error: {str(e)}")
            
    def handle_data_response(self, sender: str, data: Any):
        """Handle data response from Data tab."""
        try:
            ticker = data.get("ticker")
            if ticker:
                self.analysis_cache[ticker] = data.get("data")
                self.run_analysis()  # Automatically run analysis when data is received
        except Exception as e:
            self.logger.error(f"Error handling data response: {e}")
            
    def request_data(self):
        """Request data from Data tab."""
        try:
            ticker = self.ticker_combo.currentText()
            if not ticker:
                self.status_label.setText("Please select a ticker")
                return
                
            # Generate unique request ID
            request_id = str(uuid.uuid4())
            self.pending_requests[request_id] = {
                'ticker': ticker,
                'timestamp': datetime.now(),
                'status': 'pending'
            }
            
            # Request data from Data tab
            self.message_bus.publish(
                "Analysis",
                "data_request",
                {
                    'request_id': request_id,
                    'ticker': ticker,
                    'market': self.market_combo.currentText(),
                    'start_date': self.start_date.date().toString("yyyy-MM-dd"),
                    'end_date': self.end_date.date().toString("yyyy-MM-dd")
                }
            )
            
            self.status_label.setText(f"Requested data for {ticker}")
            
        except Exception as e:
            self.logger.error(f"Error requesting data: {e}")
            self.status_label.setText(f"Error: {str(e)}")
            
    def refresh_analysis(self):
        """Refresh the current analysis."""
        try:
            ticker = self.ticker_combo.currentText()
            if not ticker or ticker not in self.analysis_cache:
                self.status_label.setText("No analysis to refresh")
                return
                
            # Request fresh data
            self.request_data()
            
            # Run analysis when data is received
            self.run_analysis()
            
        except Exception as e:
            self.logger.error(f"Error refreshing analysis: {e}")
            self.status_label.setText(f"Error: {str(e)}")
            
    def handle_analysis_request(self, sender: str, data: Any):
        """Handle analysis request from other tabs."""
        try:
            request_id = data.get('request_id')
            ticker = data.get('ticker')
            analysis_type = data.get('analysis_type')
            
            if not all([request_id, ticker, analysis_type]):
                self.logger.error("Invalid analysis request")
                return
                
            # Check if we have the requested data
            if ticker not in self.analysis_cache:
                self.message_bus.publish(
                    "Analysis",
                    "error",
                    {
                        'request_id': request_id,
                        'error': f"No data available for {ticker}"
                    }
                )
                return
                
            # Run the requested analysis
            results = self.run_analysis_by_type(
                self.analysis_cache[ticker],
                analysis_type
            )
            
            # Send results back
            self.message_bus.publish(
                "Analysis",
                "analysis_response",
                {
                    'request_id': request_id,
                    'ticker': ticker,
                    'analysis_type': analysis_type,
                    'results': results
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error handling analysis request: {e}")
            self.message_bus.publish(
                "Analysis",
                "error",
                {
                    'request_id': data.get('request_id'),
                    'error': str(e)
                }
            )
            
    def handle_analysis_response(self, sender: str, data: Any):
        """Handle analysis response from other tabs."""
        try:
            request_id = data.get('request_id')
            if request_id in self.pending_requests:
                # Update pending request
                self.pending_requests[request_id]['status'] = 'completed'
                self.pending_requests[request_id]['results'] = data.get('results')
                
                # Update UI with results
                self.update_results_table(data.get('results'))
                
        except Exception as e:
            self.logger.error(f"Error handling analysis response: {e}")
            
    def handle_error(self, sender: str, data: Any):
        """Handle error messages."""
        try:
            request_id = data.get('request_id')
            error_message = data.get('error')
            
            if request_id in self.pending_requests:
                self.pending_requests[request_id]['status'] = 'error'
                self.pending_requests[request_id]['error'] = error_message
                
            self.status_label.setText(f"Error: {error_message}")
            
        except Exception as e:
            self.logger.error(f"Error handling error message: {e}")
            
    def run_analysis_by_type(self, data: pd.DataFrame, analysis_type: str) -> Dict[str, Any]:
        """Run analysis based on type."""
        if analysis_type == "Technical Analysis":
            return self.run_technical_analysis(data)
        elif analysis_type == "Fundamental Analysis":
            return self.run_statistical_analysis(data)
        elif analysis_type == "Correlation Analysis":
            return self.run_correlation_analysis(data)
        elif analysis_type == "Volatility Analysis":
            return self.run_timeseries_analysis(data)
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
            
    def cleanup(self):
        """Cleanup resources."""
        super().cleanup()
        self.data_cache.clear()
        self.analysis_cache.clear()
        self.pending_requests.clear()

    def analyze(self):
        """Run analysis on the selected data."""
        try:
            ticker = self.ticker_input.text()
            if not ticker:
                self.status_label.setText("Please enter a ticker symbol")
                return
                
            analysis_type = self.analysis_type_combo.currentText()
            self.logger.info(f"Running {analysis_type} analysis for {ticker}")
            
            # Request data from Data tab
            request_id = str(uuid.uuid4())
            self.pending_requests[request_id] = {
                'ticker': ticker,
                'analysis_type': analysis_type,
                'timestamp': datetime.now()
            }
            
            self.message_bus.publish(
                "Analysis",
                "data_request",
                {
                    'request_id': request_id,
                    'ticker': ticker
                }
            )
            
            self.status_label.setText(f"Requesting data for {ticker}")
            
        except Exception as e:
            error_msg = f"Error starting analysis: {str(e)}"
            self.logger.error(error_msg)
            self.status_label.setText(error_msg)

def main():
    """Main function for the analysis tab process."""
    app = QApplication(sys.argv)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting analysis tab process")
    
    # Create and show the analysis tab
    window = AnalysisTab()
    window.setWindowTitle("Analysis Tab")
    window.show()
    
    # Start the application event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 