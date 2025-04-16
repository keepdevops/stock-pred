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
        self.logger = logging.getLogger(__name__)
        
        # Initialize instance variables
        self.prediction_cache = {}
        self.pending_requests = {}
        
        # Initialize UI components
        self.main_layout = None
        self.ticker_combo = None
        self.model_combo = None
        self.horizon_spin = None
        self.confidence_spin = None
        self.predict_button = None
        self.refresh_button = None
        self.predictions_table = None
        self.metrics_table = None
        self.model_details = None
        self.status_label = None
        self.progress_label = None
        
        # Setup UI and message bus
        self.setup_ui()
        self.setup_message_bus()
        
    def _setup_ui_impl(self):
        """Setup the UI components."""
        # Create main layout
        self.main_layout = QVBoxLayout()
        
        # Create controls group
        controls_group = QGroupBox("Prediction Controls")
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
        
        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        model_layout.addWidget(self.model_combo)
        top_controls.addLayout(model_layout)
        
        controls_layout.addLayout(top_controls)
        
        # Prediction parameters
        params_layout = QHBoxLayout()
        
        # Prediction horizon
        horizon_layout = QHBoxLayout()
        horizon_layout.addWidget(QLabel("Horizon:"))
        self.horizon_spin = QSpinBox()
        self.horizon_spin.setRange(1, 365)
        self.horizon_spin.setValue(30)
        self.horizon_spin.setSuffix(" days")
        horizon_layout.addWidget(self.horizon_spin)
        params_layout.addLayout(horizon_layout)
        
        # Confidence threshold
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("Min. Confidence:"))
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.1, 1.0)
        self.confidence_spin.setValue(0.7)
        self.confidence_spin.setSingleStep(0.1)
        self.confidence_spin.setDecimals(2)
        confidence_layout.addWidget(self.confidence_spin)
        params_layout.addLayout(confidence_layout)
        
        controls_layout.addLayout(params_layout)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.predict_button = QPushButton("Make Prediction")
        self.predict_button.clicked.connect(self.make_prediction)
        button_layout.addWidget(self.predict_button)
        
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_prediction)
        button_layout.addWidget(self.refresh_button)
        
        controls_layout.addLayout(button_layout)
        controls_group.setLayout(controls_layout)
        self.main_layout.addWidget(controls_group)
        
        # Create predictions table
        self.predictions_table = QTableWidget()
        self.predictions_table.setColumnCount(4)
        self.predictions_table.setHorizontalHeaderLabels([
            "Date", "Predicted Price", "Confidence", "Direction"
        ])
        self.main_layout.addWidget(self.predictions_table)
        
        # Create metrics table
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.main_layout.addWidget(self.metrics_table)
        
        # Create model details label
        self.model_details = QLabel()
        self.main_layout.addWidget(self.model_details)
        
        # Status bar
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.progress_label = QLabel()
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.progress_label)
        self.main_layout.addLayout(status_layout)
        
        # Set the main layout
        self.setLayout(self.main_layout)
        
    def _setup_message_bus_impl(self):
        """Setup message bus subscriptions."""
        self.message_bus.subscribe("Models", self.handle_model_message)
        self.message_bus.subscribe("Predictions", self.handle_prediction_response)
        self.logger.debug("Subscribed to Models and Predictions topics")
        
    def on_ticker_changed(self, ticker: str):
        """Handle ticker selection change."""
        try:
            if ticker:
                # Request data availability
                self.message_bus.publish(
                    "Predictions",
                    "data_request",
                    {
                        'ticker': ticker,
                        'purpose': 'check_availability'
                    }
                )
        except Exception as e:
            self.logger.error(f"Error handling ticker change: {e}")
            
    def on_model_changed(self, model: str):
        """Handle model selection change."""
        try:
            if model:
                # Request model details
                self.message_bus.publish(
                    "Predictions",
                    "model_info_request",
                    {
                        'model': model
                    }
                )
        except Exception as e:
            self.logger.error(f"Error handling model change: {e}")
            
    def make_prediction(self, ticker=None, model=None, days=None):
        """Make a prediction request."""
        try:
            # Use provided values or get from UI
            ticker = ticker or self.ticker_combo.currentText()
            model = model or self.model_combo.currentText()
            days = days or self.horizon_spin.value()
            
            if not ticker:
                self.status_label.setText("Please select a ticker")
                return False
            
            if not model:
                # If no model is selected but we have models available, select the first one
                if self.model_combo.count() > 0:
                    model = self.model_combo.itemText(0)
                    self.model_combo.setCurrentText(model)
                else:
                    self.status_label.setText("No models available")
                    return False
            
            # Generate request ID
            request_id = str(uuid.uuid4())
            
            # Create request
            request = {
                'request_id': request_id,
                'ticker': ticker,
                'model': model,
                'days': days,
                'confidence_threshold': self.confidence_spin.value(),
                'timestamp': datetime.now()
            }
            
            # Add to pending requests
            self.pending_requests[request_id] = request
            
            # Publish request
            self.message_bus.publish(
                "Predictions",
                "prediction_request",
                request
            )
            
            # Update UI
            self.status_label.setText(f"Requesting prediction for {ticker}")
            self.progress_label.setText("⟳")
            self.predict_button.setEnabled(False)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            self.status_label.setText(f"Error: {str(e)}")
            return False
            
    def handle_message(self, sender: str, message_type: str, data: Any):
        """Handle incoming messages."""
        try:
            if message_type == "prediction_response":
                self.handle_prediction_response(data)
            elif message_type == "model_info_response":
                self.handle_model_info_response(data)
            elif message_type == "error":
                self.handle_error(data)
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
            
    def handle_data_message(self, sender: str, message_type: str, data: Any):
        """Handle messages from Data tab."""
        try:
            if message_type == "data_available":
                ticker = data.get('ticker')
                if ticker and ticker not in [self.ticker_combo.itemText(i) for i in range(self.ticker_combo.count())]:
                    self.ticker_combo.addItem(ticker)
        except Exception as e:
            self.logger.error(f"Error handling data message: {e}")
            
    def handle_model_message(self, topic: str, message_type: str, data: dict):
        """Handle messages from Models tab."""
        try:
            self.logger.debug(f"Received model message: {message_type} with data: {data}")
            if message_type == "model_list":
                models = data.get("models", [])
                self.logger.debug(f"Updating model combo with models: {models}")
                
                current = self.model_combo.currentText()
                self.model_combo.clear()
                
                if models:
                    self.model_combo.addItems(models)
                    index = self.model_combo.findText(current)
                    if index >= 0:
                        self.model_combo.setCurrentIndex(index)
                    elif self.model_combo.count() > 0:
                        self.model_combo.setCurrentIndex(0)
                
            elif message_type == "model_added":
                model_name = data.get("model_name")
                if model_name:
                    current_items = [self.model_combo.itemText(i) for i in range(self.model_combo.count())]
                    if model_name not in current_items:
                        self.model_combo.addItem(model_name)
                        self.logger.debug(f"Added model {model_name} to combo box")
        except Exception as e:
            self.logger.error(f"Error handling model message: {e}", exc_info=True)
            
    def handle_prediction_response(self, topic: str, message_type: str, data: dict):
        """Handle prediction response messages."""
        try:
            self.logger.debug(f"Received prediction message: {message_type}")
            if message_type == "prediction_response":
                request_id = data.get("request_id")
                predictions = data.get("predictions", [])
                metrics = data.get("metrics", {})
                
                # Update predictions table
                self.update_predictions_table(predictions)
                
                # Update metrics table
                self.update_metrics_table(metrics)
                
                # Update status
                self.status_label.setText("Prediction completed")
                self.progress_label.setText("")
                
            elif message_type == "prediction_error":
                error_message = data.get("error", "Unknown error")
                self.status_label.setText(f"Error: {error_message}")
                self.progress_label.setText("")
                
        except Exception as e:
            self.logger.error(f"Error handling prediction response: {e}")
            self.status_label.setText("Error processing prediction response")
            
    def update_predictions_table(self, predictions: List[Dict[str, Any]]):
        """Update the predictions table."""
        try:
            self.predictions_table.setRowCount(0)
            for i, pred in enumerate(predictions):
                self.predictions_table.insertRow(i)
                
                # Date
                date_item = QTableWidgetItem(pred['date'].strftime("%Y-%m-%d"))
                self.predictions_table.setItem(i, 0, date_item)
                
                # Price
                price_item = QTableWidgetItem(f"{pred['price']:.2f}")
                self.predictions_table.setItem(i, 1, price_item)
                
                # Confidence
                conf_item = QTableWidgetItem(f"{pred['confidence']:.1%}")
                self.predictions_table.setItem(i, 2, conf_item)
                
                # Direction
                direction = "↑" if pred.get('direction', 0) > 0 else "↓"
                dir_item = QTableWidgetItem(direction)
                dir_item.setForeground(
                    Qt.GlobalColor.green if direction == "↑" else Qt.GlobalColor.red
                )
                self.predictions_table.setItem(i, 3, dir_item)
                
        except Exception as e:
            self.logger.error(f"Error updating predictions table: {e}")
            
    def update_model_details(self, details: Dict[str, Any]):
        """Update model details display."""
        try:
            text = []
            text.append(f"Model: {details.get('model', 'N/A')}")
            text.append(f"Type: {details.get('type', 'N/A')}")
            text.append(f"Features: {', '.join(details.get('features', []))}")
            text.append(f"Last Training: {details.get('last_training', 'N/A')}")
            
            self.model_details.setText("\n".join(text))
            
        except Exception as e:
            self.logger.error(f"Error updating model details: {e}")
            
    def update_metrics_table(self, metrics: Dict[str, Any]):
        """Update performance metrics table."""
        try:
            self.metrics_table.setRowCount(0)
            for i, (metric, value) in enumerate(metrics.items()):
                self.metrics_table.insertRow(i)
                self.metrics_table.setItem(i, 0, QTableWidgetItem(metric))
                self.metrics_table.setItem(i, 1, QTableWidgetItem(f"{value:.4f}"))
                
        except Exception as e:
            self.logger.error(f"Error updating metrics table: {e}")
            
    def handle_error(self, data: Dict[str, Any]):
        """Handle error messages."""
        try:
            error_msg = data.get('error', 'Unknown error')
            request_id = data.get('request_id')
            
            if request_id in self.pending_requests:
                self.pending_requests[request_id]['status'] = 'error'
                self.pending_requests[request_id]['error'] = error_msg
                
            self.status_label.setText(f"Error: {error_msg}")
            self.progress_label.setText("❌")
            self.predict_button.setEnabled(True)
            
        except Exception as e:
            self.logger.error(f"Error handling error message: {e}")
            
    def cleanup(self):
        """Cleanup resources."""
        try:
            super().cleanup()
            self.prediction_cache.clear()
            self.pending_requests.clear()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def test_input_fields(self):
        """Test input field functionality."""
        # Test ticker input
        self.ticker_combo.setText("AAPL")
        assert self.ticker_combo.text() == "AAPL"
        
        # Test model combo box
        self.model_combo.setCurrentText("Technical")
        assert self.model_combo.currentText() == "Technical"
        
        # Test result text
        test_text = "Test analysis results"
        self.model_details.setText(test_text)
        assert self.model_details.text() == test_text

    def test_request_tracking(self):
        """Test request tracking functionality."""
        
        # Test adding pending request
        request_id = str(uuid.uuid4())
        self.pending_requests[request_id] = {
            'type': 'data',
            'ticker': 'AAPL',
            'timestamp': datetime.now()
        }
        assert request_id in self.pending_requests
        
        # Test request completion
        data_response = {
            'request_id': request_id,
            'ticker': 'AAPL',
            'data': []
        }
        self.handle_data_response("Data", data_response)
        
        # Test request cleanup
        self.cleanup()
        assert len(self.pending_requests) == 0

    def run_tests(self):
        """Run all tests for the Charts tab."""
        try:
            self.test_input_fields()
            self.test_pub_sub()
            self.test_message_handling()
            self.test_chart_types()
            self.test_data_cache()
            self.test_request_tracking()
            self.test_message_flow()
            self.status_label.setText("All tests passed successfully")
            self.logger.info("All Charts tab tests passed")
        except Exception as e:
            error_msg = f"Test failed: {str(e)}"
            self.status_label.setText(error_msg)
            self.logger.error(error_msg)

    def refresh_prediction(self):
        """Refresh the current prediction."""
        try:
            ticker = self.ticker_combo.currentText()
            if not ticker:
                self.status_label.setText("No ticker selected")
                return
            
            # Clear current display
            self.predictions_table.setRowCount(0)
            self.metrics_table.setRowCount(0)
            self.model_details.clear()
            
            # If we have a cached prediction for this ticker, update the display
            if ticker in self.prediction_cache:
                cached_data = self.prediction_cache[ticker]
                self.update_predictions_table(cached_data.get('predictions', []))
                self.update_model_details(cached_data.get('details', {}))
                self.update_metrics_table(cached_data.get('metrics', {}))
                self.status_label.setText(f"Refreshed predictions for {ticker}")
            else:
                # If no cached prediction exists, trigger a new prediction
                model = self.model_combo.currentText()
                days = self.horizon_spin.value()
                
                if not self.make_prediction(ticker=ticker, model=model, days=days):
                    raise Exception("Failed to create prediction request")
            
        except Exception as e:
            self.logger.error(f"Error refreshing prediction: {str(e)}")
            self.status_label.setText(f"Error refreshing prediction: {str(e)}")
            traceback.print_exc()

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