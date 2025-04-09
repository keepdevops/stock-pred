import sys
import os
import logging
from typing import Any, Dict, Optional
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QComboBox,
    QGroupBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QFileDialog
)
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QTimer
from PyQt6.QtGui import QFont
import pandas as pd
import numpy as np
import talib
from stock_market_analyzer.modules.message_bus import MessageBus

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class AnalysisTab(QWidget):
    """Analysis tab for the stock market analyzer."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.message_bus = MessageBus()
        self.logger = logging.getLogger(__name__)
        self.current_data = None
        self.analysis_results = {}
        self.setup_ui()
        
        # Subscribe to message bus
        self.message_bus.subscribe("Analysis", self.handle_message)
        
        # Set up heartbeat timer
        self.heartbeat_timer = QTimer()
        self.heartbeat_timer.timeout.connect(self.send_heartbeat)
        self.heartbeat_timer.start(5000)  # Send heartbeat every 5 seconds
        
    def setup_ui(self):
        """Setup the UI for the analysis tab."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Analysis type selection
        analysis_type_group = QGroupBox("Analysis Type")
        analysis_type_layout = QVBoxLayout()
        
        self.analysis_type_combo = QComboBox()
        self.analysis_type_combo.addItems([
            "Technical Indicators",
            "Pattern Recognition",
            "Volatility Analysis",
            "Trend Analysis",
            "Volume Analysis"
        ])
        self.analysis_type_combo.currentTextChanged.connect(self.update_analysis_options)
        analysis_type_layout.addWidget(self.analysis_type_combo)
        
        analysis_type_group.setLayout(analysis_type_layout)
        layout.addWidget(analysis_type_group)
        
        # Analysis parameters
        self.parameters_group = QGroupBox("Analysis Parameters")
        self.parameters_layout = QVBoxLayout()
        self.parameters_group.setLayout(self.parameters_layout)
        layout.addWidget(self.parameters_group)
        
        # Results display
        results_group = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout()
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(["Indicator", "Value", "Signal"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        results_layout.addWidget(self.results_table)
        
        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        results_layout.addWidget(self.analysis_text)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        # Buttons
        buttons_layout = QHBoxLayout()
        
        self.analyze_button = QPushButton("Run Analysis")
        self.analyze_button.clicked.connect(self.run_analysis)
        buttons_layout.addWidget(self.analyze_button)
        
        self.export_button = QPushButton("Export Results")
        self.export_button.clicked.connect(self.export_results)
        buttons_layout.addWidget(self.export_button)
        
        layout.addLayout(buttons_layout)
        
        # Status label
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        
        self.logger.info("Analysis tab initialized")
        
    def update_analysis_options(self):
        """Update analysis options based on selected type."""
        # Clear existing parameters
        while self.parameters_layout.count():
            child = self.parameters_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                
        analysis_type = self.analysis_type_combo.currentText()
        
        if analysis_type == "Technical Indicators":
            # Add RSI parameters
            rsi_layout = QHBoxLayout()
            rsi_layout.addWidget(QLabel("RSI Period:"))
            rsi_period = QSpinBox()
            rsi_period.setRange(2, 30)
            rsi_period.setValue(14)
            rsi_layout.addWidget(rsi_period)
            self.parameters_layout.addLayout(rsi_layout)
            
            # Add MACD parameters
            macd_layout = QHBoxLayout()
            macd_layout.addWidget(QLabel("MACD Fast:"))
            macd_fast = QSpinBox()
            macd_fast.setRange(2, 30)
            macd_fast.setValue(12)
            macd_layout.addWidget(macd_fast)
            
            macd_layout.addWidget(QLabel("MACD Slow:"))
            macd_slow = QSpinBox()
            macd_slow.setRange(2, 30)
            macd_slow.setValue(26)
            macd_layout.addWidget(macd_slow)
            self.parameters_layout.addLayout(macd_layout)
            
        elif analysis_type == "Pattern Recognition":
            # Add pattern selection
            pattern_layout = QHBoxLayout()
            pattern_layout.addWidget(QLabel("Pattern:"))
            pattern_combo = QComboBox()
            pattern_combo.addItems([
                "Doji",
                "Hammer",
                "Engulfing",
                "Morning Star",
                "Evening Star"
            ])
            pattern_layout.addWidget(pattern_combo)
            self.parameters_layout.addLayout(pattern_layout)
            
        elif analysis_type == "Volatility Analysis":
            # Add Bollinger Bands parameters
            bb_layout = QHBoxLayout()
            bb_layout.addWidget(QLabel("BB Period:"))
            bb_period = QSpinBox()
            bb_period.setRange(2, 30)
            bb_period.setValue(20)
            bb_layout.addWidget(bb_period)
            
            bb_layout.addWidget(QLabel("BB Std Dev:"))
            bb_std = QDoubleSpinBox()
            bb_std.setRange(1.0, 3.0)
            bb_std.setValue(2.0)
            bb_std.setSingleStep(0.1)
            bb_layout.addWidget(bb_std)
            self.parameters_layout.addLayout(bb_layout)
            
        elif analysis_type == "Trend Analysis":
            # Add moving average parameters
            ma_layout = QHBoxLayout()
            ma_layout.addWidget(QLabel("MA Period:"))
            ma_period = QSpinBox()
            ma_period.setRange(2, 200)
            ma_period.setValue(50)
            ma_layout.addWidget(ma_period)
            self.parameters_layout.addLayout(ma_layout)
            
        elif analysis_type == "Volume Analysis":
            # Add volume parameters
            volume_layout = QHBoxLayout()
            volume_layout.addWidget(QLabel("Volume MA Period:"))
            volume_ma = QSpinBox()
            volume_ma.setRange(2, 30)
            volume_ma.setValue(20)
            volume_layout.addWidget(volume_ma)
            self.parameters_layout.addLayout(volume_layout)
            
    def run_analysis(self):
        """Run the selected analysis."""
        try:
            if self.current_data is None:
                self.status_label.setText("No data available for analysis")
                return
                
            analysis_type = self.analysis_type_combo.currentText()
            self.analysis_results = {}
            
            if analysis_type == "Technical Indicators":
                self.run_technical_indicators()
            elif analysis_type == "Pattern Recognition":
                self.run_pattern_recognition()
            elif analysis_type == "Volatility Analysis":
                self.run_volatility_analysis()
            elif analysis_type == "Trend Analysis":
                self.run_trend_analysis()
            elif analysis_type == "Volume Analysis":
                self.run_volume_analysis()
                
            self.update_results_display()
            self.status_label.setText("Analysis completed")
            
            # Publish results to message bus
            self.message_bus.publish("Analysis", "analysis_completed", {
                "type": analysis_type,
                "results": self.analysis_results
            })
            
        except Exception as e:
            error_msg = f"Error running analysis: {str(e)}"
            self.logger.error(error_msg)
            self.status_label.setText(error_msg)
            self.message_bus.publish("Analysis", "error", error_msg)
            
    def run_technical_indicators(self):
        """Run technical indicators analysis."""
        try:
            # Get RSI
            rsi_period = self.findChild(QSpinBox, "RSI Period").value()
            rsi = talib.RSI(self.current_data['Close'], timeperiod=rsi_period)
            self.analysis_results['RSI'] = {
                'value': rsi.iloc[-1],
                'signal': 'Overbought' if rsi.iloc[-1] > 70 else 'Oversold' if rsi.iloc[-1] < 30 else 'Neutral'
            }
            
            # Get MACD
            macd_fast = self.findChild(QSpinBox, "MACD Fast").value()
            macd_slow = self.findChild(QSpinBox, "MACD Slow").value()
            macd, signal, hist = talib.MACD(
                self.current_data['Close'],
                fastperiod=macd_fast,
                slowperiod=macd_slow,
                signalperiod=9
            )
            self.analysis_results['MACD'] = {
                'value': macd.iloc[-1],
                'signal': 'Bullish' if macd.iloc[-1] > signal.iloc[-1] else 'Bearish'
            }
            
        except Exception as e:
            raise Exception(f"Error in technical indicators: {str(e)}")
            
    def run_pattern_recognition(self):
        """Run pattern recognition analysis."""
        try:
            pattern = self.findChild(QComboBox, "Pattern").currentText()
            # Implement pattern recognition logic
            # This is a placeholder - actual implementation would use TA-Lib or custom pattern recognition
            self.analysis_results['Pattern'] = {
                'value': pattern,
                'signal': 'Detected'  # Placeholder
            }
            
        except Exception as e:
            raise Exception(f"Error in pattern recognition: {str(e)}")
            
    def run_volatility_analysis(self):
        """Run volatility analysis."""
        try:
            bb_period = self.findChild(QSpinBox, "BB Period").value()
            bb_std = self.findChild(QDoubleSpinBox, "BB Std Dev").value()
            
            upper, middle, lower = talib.BBANDS(
                self.current_data['Close'],
                timeperiod=bb_period,
                nbdevup=bb_std,
                nbdevdn=bb_std
            )
            
            current_price = self.current_data['Close'].iloc[-1]
            self.analysis_results['Bollinger Bands'] = {
                'value': f"Upper: {upper.iloc[-1]:.2f}, Middle: {middle.iloc[-1]:.2f}, Lower: {lower.iloc[-1]:.2f}",
                'signal': 'Overbought' if current_price > upper.iloc[-1] else 'Oversold' if current_price < lower.iloc[-1] else 'Neutral'
            }
            
        except Exception as e:
            raise Exception(f"Error in volatility analysis: {str(e)}")
            
    def run_trend_analysis(self):
        """Run trend analysis."""
        try:
            ma_period = self.findChild(QSpinBox, "MA Period").value()
            ma = talib.SMA(self.current_data['Close'], timeperiod=ma_period)
            
            current_price = self.current_data['Close'].iloc[-1]
            self.analysis_results['Trend'] = {
                'value': f"MA: {ma.iloc[-1]:.2f}",
                'signal': 'Bullish' if current_price > ma.iloc[-1] else 'Bearish'
            }
            
        except Exception as e:
            raise Exception(f"Error in trend analysis: {str(e)}")
            
    def run_volume_analysis(self):
        """Run volume analysis."""
        try:
            volume_ma = self.findChild(QSpinBox, "Volume MA Period").value()
            volume_ma = talib.SMA(self.current_data['Volume'], timeperiod=volume_ma)
            
            current_volume = self.current_data['Volume'].iloc[-1]
            self.analysis_results['Volume'] = {
                'value': f"Volume MA: {volume_ma.iloc[-1]:.2f}",
                'signal': 'High' if current_volume > volume_ma.iloc[-1] else 'Low'
            }
            
        except Exception as e:
            raise Exception(f"Error in volume analysis: {str(e)}")
            
    def update_results_display(self):
        """Update the results display with analysis results."""
        try:
            self.results_table.setRowCount(len(self.analysis_results))
            row = 0
            
            for indicator, data in self.analysis_results.items():
                self.results_table.setItem(row, 0, QTableWidgetItem(indicator))
                self.results_table.setItem(row, 1, QTableWidgetItem(str(data['value'])))
                self.results_table.setItem(row, 2, QTableWidgetItem(data['signal']))
                row += 1
                
            # Update analysis text
            analysis_text = "Analysis Results:\n\n"
            for indicator, data in self.analysis_results.items():
                analysis_text += f"{indicator}: {data['value']} ({data['signal']})\n"
            self.analysis_text.setText(analysis_text)
            
        except Exception as e:
            raise Exception(f"Error updating results display: {str(e)}")
            
    def export_results(self):
        """Export analysis results to a file."""
        try:
            if not self.analysis_results:
                self.status_label.setText("No results to export")
                return
                
            # Create a DataFrame from the results
            results_df = pd.DataFrame([
                {
                    'Indicator': indicator,
                    'Value': data['value'],
                    'Signal': data['signal']
                }
                for indicator, data in self.analysis_results.items()
            ])
            
            # Save to CSV
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Analysis Results",
                "",
                "CSV Files (*.csv);;All Files (*)"
            )
            
            if file_path:
                results_df.to_csv(file_path, index=False)
                self.status_label.setText(f"Results exported to {file_path}")
                
        except Exception as e:
            error_msg = f"Error exporting results: {str(e)}"
            self.logger.error(error_msg)
            self.status_label.setText(error_msg)
            
    def handle_message(self, sender: str, message_type: str, data: Any):
        """Handle incoming messages."""
        try:
            if message_type == "data_updated":
                self.current_data = data[1]  # data is (ticker, dataframe)
                self.status_label.setText(f"Received new data for {data[0]}")
                
            elif message_type == "error":
                error_msg = f"Received error from {sender}: {data}"
                self.logger.error(error_msg)
                self.status_label.setText(f"Error: {data}")
                
            elif message_type == "heartbeat":
                self.logger.debug(f"Received heartbeat from {sender}")
                
        except Exception as e:
            error_log = f"Error handling message in Analysis tab: {str(e)}"
            self.logger.error(error_log)
            
    def send_heartbeat(self):
        """Send heartbeat message."""
        try:
            self.message_bus.publish("Analysis", "heartbeat", "Alive")
        except Exception as e:
            self.logger.error(f"Error sending heartbeat: {str(e)}")
            
    def publish_message(self, message_type: str, data: Any):
        """Publish a message to the message bus."""
        try:
            self.message_bus.publish("Analysis", message_type, data)
        except Exception as e:
            error_log = f"Error publishing message from Analysis tab: {str(e)}"
            self.logger.error(error_log)

    def closeEvent(self, event):
        """Handle the close event."""
        try:
            # Stop heartbeat timer
            self.heartbeat_timer.stop()
            
            # Unsubscribe from message bus
            self.message_bus.unsubscribe("Analysis", self.handle_message)
            
            super().closeEvent(event)
            
        except Exception as e:
            self.logger.error(f"Error in close event: {str(e)}")

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