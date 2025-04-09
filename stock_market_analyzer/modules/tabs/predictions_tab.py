import sys
import os
import logging
from typing import Any
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QComboBox,
    QSpinBox, QProgressBar
)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from stock_market_analyzer.modules.message_bus import MessageBus

class PredictionsTab(QWidget):
    """Predictions tab for the stock market analyzer."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.message_bus = MessageBus()
        self.logger = logging.getLogger(__name__)
        self.available_data = {}  # Store available data by ticker
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the predictions tab UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # --- Data Selection ---
        data_layout = QHBoxLayout()
        data_layout.addWidget(QLabel("Select Data:"))
        self.data_combo = QComboBox()
        self.data_combo.currentIndexChanged.connect(self.on_data_selected)
        data_layout.addWidget(self.data_combo)
        layout.addLayout(data_layout)

        # --- Prediction Parameters ---
        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("Days to Predict:"))
        self.days_spinbox = QSpinBox()
        self.days_spinbox.setRange(1, 30)
        self.days_spinbox.setValue(7)
        params_layout.addWidget(self.days_spinbox)

        params_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["LSTM", "Linear Regression", "Random Forest"])
        params_layout.addWidget(self.model_combo)
        layout.addLayout(params_layout)

        # --- Prediction Button ---
        self.predict_button = QPushButton("Generate Predictions")
        self.predict_button.clicked.connect(self.generate_predictions)
        self.predict_button.setEnabled(False)  # Disabled until data is selected
        layout.addWidget(self.predict_button)

        # --- Progress Bar ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # --- Results Display ---
        layout.addWidget(QLabel("Prediction Results:"))
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        layout.addWidget(self.results_text)
        
        # Status label
        self.status_label = QLabel("Select data and parameters to generate predictions.")
        layout.addWidget(self.status_label)
        
        # Subscribe to message bus
        self.message_bus.subscribe("Predictions", self.handle_message)
        self.logger.info("Predictions tab initialized")

    def log_message(self, message: str):
        """Appends a message to the results area."""
        self.logger.info(message)
        self.results_text.append(message)
        QApplication.processEvents()  # Update UI

    def handle_message(self, sender: str, message_type: str, data: Any):
        """Handle incoming messages."""
        try:
            if message_type == "data_updated":
                # Update available data
                ticker, df = data
                self.available_data[ticker] = df
                self.update_data_combo()
                self.log_message(f"Received data update for {ticker}")
                
            elif message_type == "error":
                error_msg = f"Received error from {sender}: {data}"
                self.log_message(error_msg)
                self.status_label.setText("Error occurred. Check log for details.")
                
        except Exception as e:
            error_log = f"Error handling message in Predictions tab: {str(e)}"
            self.logger.error(error_log)
            self.log_message(error_log)

    def update_data_combo(self):
        """Update the data combo box with available data sources."""
        current_text = self.data_combo.currentText()
        self.data_combo.clear()
        self.data_combo.addItems(self.available_data.keys())
        
        # Try to restore previous selection
        index = self.data_combo.findText(current_text)
        if index >= 0:
            self.data_combo.setCurrentIndex(index)
            
        self.predict_button.setEnabled(self.data_combo.count() > 0)

    def on_data_selected(self):
        """Handle data selection change."""
        if self.data_combo.currentText():
            self.predict_button.setEnabled(True)
            self.status_label.setText("Ready to generate predictions.")
        else:
            self.predict_button.setEnabled(False)
            self.status_label.setText("No data selected.")

    def generate_predictions(self):
        """Generate predictions based on selected data and parameters."""
        try:
            selected_ticker = self.data_combo.currentText()
            days_to_predict = self.days_spinbox.value()
            model_type = self.model_combo.currentText()
            
            if not selected_ticker or selected_ticker not in self.available_data:
                self.log_message("Error: No valid data selected.")
                return

            self.log_message(f"=== Generating Predictions ===")
            self.log_message(f"Ticker: {selected_ticker}")
            self.log_message(f"Days to predict: {days_to_predict}")
            self.log_message(f"Model: {model_type}")
            
            # Get the data
            df = self.available_data[selected_ticker]
            
            # Show progress
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            
            # Simulate prediction process
            self.progress_bar.setValue(20)
            self.log_message("Preprocessing data...")
            QApplication.processEvents()
            
            self.progress_bar.setValue(40)
            self.log_message("Training model...")
            QApplication.processEvents()
            
            self.progress_bar.setValue(60)
            self.log_message("Generating predictions...")
            QApplication.processEvents()
            
            # Generate sample predictions
            last_date = df.index[-1]
            predictions = []
            for i in range(1, days_to_predict + 1):
                pred_date = last_date + timedelta(days=i)
                pred_price = df['Close'].iloc[-1] * (1 + np.random.normal(0, 0.01))
                predictions.append((pred_date, pred_price))
            
            self.progress_bar.setValue(80)
            
            # Display predictions
            self.log_message("\nPredicted Prices:")
            for date, price in predictions:
                self.log_message(f"{date.strftime('%Y-%m-%d')}: ${price:.2f}")
            
            self.progress_bar.setValue(100)
            self.status_label.setText("Predictions generated successfully.")
            
            # Publish predictions
            self.message_bus.publish("Predictions", "predictions_generated", 
                                   (selected_ticker, predictions))
            
        except Exception as e:
            error_msg = f"Error generating predictions: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            self.log_message(error_msg)
            self.status_label.setText("Error generating predictions.")
            self.message_bus.publish("Predictions", "error", error_msg)
            
        finally:
            self.progress_bar.setVisible(False)
            self.log_message("=== Prediction Process Complete ===")

    def publish_message(self, message_type: str, data: Any):
        """Publish a message to the message bus."""
        try:
            self.message_bus.publish("Predictions", message_type, data)
        except Exception as e:
            error_log = f"Error publishing message from Predictions tab: {str(e)}"
            self.logger.error(error_log)

def main():
    """Main function for the predictions tab process."""
    # Ensure QApplication instance exists
    app = QApplication.instance() 
    if not app: 
        app = QApplication(sys.argv)

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("Starting predictions tab process")
    
    # Create and show the predictions tab
    try:
        window = PredictionsTab()
        window.setWindowTitle("Predictions Tab")
        window.show()
    except Exception as e:
         logger.error(f"Failed to create or show PredictionsTab window: {e}")
         logger.error(traceback.format_exc())
         sys.exit(1)

    if __name__ == "__main__":
        sys.exit(app.exec())

if __name__ == "__main__":
    import traceback
    main() 