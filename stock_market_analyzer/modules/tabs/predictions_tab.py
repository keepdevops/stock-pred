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
    QComboBox, QPushButton, QLabel, QSplitter, QApplication, QSpinBox,
    QDoubleSpinBox, QGroupBox, QCheckBox, QHeaderView, QMessageBox, QDateEdit,
    QListWidget
)
from PyQt6.QtCore import Qt, QTimer, QDate
from PyQt6.QtGui import QFont
from modules.tabs.base_tab import BaseTab
import uuid

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from modules.message_bus import MessageBus

class PredictionsTab(BaseTab):
    """Tab for making and viewing stock predictions."""
    
    def __init__(self, parent=None):
        """Initialize the Predictions tab."""
        super().__init__(parent)
        self.message_bus = MessageBus()
        self.logger = logging.getLogger(__name__)
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        self.setup_ui()
        self.prediction_cache = {}
        self.pending_requests = {}  # Track pending prediction requests
        
    def setup_ui(self):
        """Setup the predictions tab UI."""
        # Create predictions list
        self.predictions_list = QListWidget()
        self.main_layout.addWidget(self.predictions_list)
        
        # Add status label
        self.status_label = QLabel("Ready")
        self.main_layout.addWidget(self.status_label)
        
        # Subscribe to message bus
        self.message_bus.subscribe("Predictions", self.handle_message)
        
        self.logger.info("Predictions tab initialized")
        
        # Top controls
        controls_layout = QHBoxLayout()
        
        # Ticker selection
        controls_layout.addWidget(QLabel("Ticker:"))
        self.ticker_combo = QComboBox()
        controls_layout.addWidget(self.ticker_combo)
        
        # Model selection
        controls_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        controls_layout.addWidget(self.model_combo)
        
        # Prediction horizon
        controls_layout.addWidget(QLabel("Horizon (days):"))
        self.horizon_spin = QSpinBox()
        self.horizon_spin.setRange(1, 365)
        self.horizon_spin.setValue(30)
        controls_layout.addWidget(self.horizon_spin)
        
        # Make prediction button
        predict_button = QPushButton("Make Prediction")
        predict_button.clicked.connect(self.make_prediction)
        controls_layout.addWidget(predict_button)
        
        # Refresh button
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh_prediction)
        controls_layout.addWidget(refresh_button)
        
        self.main_layout.addLayout(controls_layout)
        
        # Splitter for results
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Predictions table
        predictions_widget = QWidget()
        predictions_layout = QVBoxLayout(predictions_widget)
        
        self.predictions_table = QTableWidget()
        self.predictions_table.setColumnCount(3)
        self.predictions_table.setHorizontalHeaderLabels([
            "Date", "Predicted Price", "Confidence"
        ])
        predictions_layout.addWidget(self.predictions_table)
        
        splitter.addWidget(predictions_widget)
        
        # Prediction details
        details_widget = QWidget()
        details_layout = QVBoxLayout(details_widget)
        
        self.details_text = QLabel("Prediction details will appear here")
        self.details_text.setWordWrap(True)
        details_layout.addWidget(self.details_text)
        
        splitter.addWidget(details_widget)
        
        # Set initial sizes
        splitter.setSizes([400, 400])
        
        self.main_layout.addWidget(splitter)
        
    def make_prediction(self):
        """Make a prediction using the selected model."""
        try:
            ticker = self.ticker_combo.currentText()
            model = self.model_combo.currentText()
            horizon = self.horizon_spin.value()
            
            if not all([ticker, model]):
                self.status_label.setText("Please select a ticker and model")
                return
                
            # Generate unique request ID
            request_id = str(uuid.uuid4())
            self.pending_requests[request_id] = {
                'ticker': ticker,
                'model': model,
                'horizon': horizon,
                'timestamp': datetime.now(),
                'status': 'pending'
            }
            
            # Request prediction from Model tab
            self.message_bus.publish(
                "Predictions",
                "prediction_request",
                {
                    'request_id': request_id,
                    'ticker': ticker,
                    'model': model,
                    'horizon': horizon
                }
            )
            
            self.status_label.setText(f"Requested prediction for {ticker}")
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {e}")
            self.status_label.setText(f"Error: {str(e)}")
            
    def refresh_prediction(self):
        """Refresh the current prediction."""
        try:
            ticker = self.ticker_combo.currentText()
            if not ticker or ticker not in self.prediction_cache:
                self.status_label.setText("No prediction to refresh")
                return
                
            # Make new prediction
            self.make_prediction()
            
        except Exception as e:
            self.logger.error(f"Error refreshing prediction: {e}")
            self.status_label.setText(f"Error: {str(e)}")
            
    def process_message(self, sender: str, message_type: str, data: Any):
        """Process incoming messages."""
        try:
            if message_type == "prediction_response":
                self.handle_prediction_response(sender, data)
            elif message_type == "model_list":
                self.handle_model_list(sender, data)
            elif message_type == "data_updated":
                self.handle_data_update(sender, data)
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
                # Update pending request
                self.pending_requests[request_id]['status'] = 'completed'
                self.pending_requests[request_id]['results'] = data.get('results')
                
                # Update predictions table
                self.update_predictions_table(data.get('results'))
                
                # Update details
                self.update_prediction_details(data.get('details'))
                
        except Exception as e:
            self.logger.error(f"Error handling prediction response: {e}")
            
    def handle_model_list(self, sender: str, data: Any):
        """Handle model list update from Model tab."""
        try:
            models = data.get('models', [])
            self.model_combo.clear()
            self.model_combo.addItems(models)
            
        except Exception as e:
            self.logger.error(f"Error handling model list: {e}")
            
    def handle_data_update(self, sender: str, data: Any):
        """Handle data updates from Data tab."""
        try:
            ticker = data.get("ticker")
            if ticker and ticker not in [self.ticker_combo.itemText(i) for i in range(self.ticker_combo.count())]:
                self.ticker_combo.addItem(ticker)
                
        except Exception as e:
            self.logger.error(f"Error handling data update: {e}")
            
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
            
    def update_predictions_table(self, results: Dict[str, Any]):
        """Update the predictions table with results."""
        try:
            self.predictions_table.setRowCount(0)
            
            predictions = results.get('predictions', [])
            for i, pred in enumerate(predictions):
                self.predictions_table.insertRow(i)
                self.predictions_table.setItem(i, 0, QTableWidgetItem(str(pred['date'])))
                self.predictions_table.setItem(i, 1, QTableWidgetItem(f"{pred['price']:.2f}"))
                self.predictions_table.setItem(i, 2, QTableWidgetItem(f"{pred['confidence']:.2%}"))
                
        except Exception as e:
            self.logger.error(f"Error updating predictions table: {e}")
            
    def update_prediction_details(self, details: Dict[str, Any]):
        """Update the prediction details text."""
        try:
            text = []
            text.append(f"Model: {details.get('model', 'N/A')}")
            text.append(f"Horizon: {details.get('horizon', 'N/A')} days")
            text.append(f"Accuracy: {details.get('accuracy', 'N/A')}")
            text.append(f"Last Training: {details.get('last_training', 'N/A')}")
            text.append(f"Features Used: {', '.join(details.get('features', []))}")
            
            self.details_text.setText("\n".join(text))
            
        except Exception as e:
            self.logger.error(f"Error updating prediction details: {e}")
            
    def cleanup(self):
        """Cleanup resources."""
        super().cleanup()
        self.prediction_cache.clear()
        self.pending_requests.clear()

def main():
    """Main function for the predictions tab process."""
    app = QApplication(sys.argv)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting predictions tab process")
    
    # Create and show the predictions tab
    window = PredictionsTab()
    window.setWindowTitle("Predictions Tab")
    window.show()
    
    # Start the application event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 