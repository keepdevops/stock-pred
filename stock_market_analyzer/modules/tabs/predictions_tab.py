import sys
import os
import logging
import traceback
import time
import uuid
import pandas as pd
from typing import Dict, Any, List, Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QComboBox, QPushButton, QLabel, QSplitter, QApplication, QSpinBox,
    QDoubleSpinBox, QGroupBox, QCheckBox, QHeaderView, QMessageBox, QDateEdit,
    QListWidget
)
from PyQt6.QtCore import Qt, QTimer, QDate
from PyQt6.QtGui import QFont
from modules.tabs.base_tab import BaseTab
from modules.message_bus import MessageBus
from modules.settings import Settings
from ..data_manager import DataManager

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class PredictionsTab(BaseTab):
    """Predictions tab for making and viewing stock predictions."""
    
    def __init__(self, message_bus: MessageBus, parent=None):
        """Initialize the Predictions tab.
        
        Args:
            message_bus: The message bus for communication.
            parent: The parent widget.
        """
        # Initialize attributes before parent __init__
        self.settings = Settings()
        self.current_color_scheme = self.settings.get_color_scheme()
        self._ui_setup_done = False
        self.ticker_combo = None
        self.model_combo = None
        self.horizon_spin = None
        self.confidence_spin = None
        self.predict_button = None
        self.refresh_button = None
        self.predictions_table = None
        self.metrics_table = None
        self.model_details = None
        self.progress_label = None
        self.prediction_cache = {}
        self.pending_requests = {}
        self.data_manager = DataManager()
        self.data_manager.register_listener("PredictionsTab", self._on_data_update)
        
        # Call parent __init__ with message bus
        super().__init__(message_bus, parent)
        
    def _setup_message_bus_impl(self):
        """Setup message bus subscriptions."""
        try:
            self.message_bus.subscribe("Models", self.handle_model_message)
            self.message_bus.subscribe("Predictions", self.handle_prediction_response)
            self.message_bus.subscribe("Data", self.handle_data_message)
            self.message_bus.subscribe("Import", self.handle_imported_data)
            self.logger.debug("Subscribed to Models, Predictions, Data, and Import topics")
        except Exception as e:
            self.handle_error("Error setting up message bus subscriptions", e)
            
    def _setup_ui_impl(self):
        """Setup the UI components."""
        try:
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
            self.ticker_combo.currentTextChanged.connect(self.on_ticker_changed)
            ticker_layout.addWidget(self.ticker_combo)
            top_controls.addLayout(ticker_layout)
            
            # Model selection
            model_layout = QHBoxLayout()
            model_layout.addWidget(QLabel("Model:"))
            self.model_combo = QComboBox()
            self.model_combo.currentTextChanged.connect(self.on_model_changed)
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
            
            # Progress label
            self.progress_label = QLabel()
            self.main_layout.addWidget(self.progress_label)
            
        except Exception as e:
            self.handle_error("Error setting up UI", e)
            
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
            self.handle_error("Error handling ticker change", e)
            
    def on_model_changed(self, model: str):
        """Handle model selection change."""
        try:
            if model:
                # Request model details
                self.message_bus.publish(
                    "Predictions",
                    "model_details_request",
                    {
                        'model': model
                    }
                )
        except Exception as e:
            self.handle_error("Error handling model change", e)
            
    def make_prediction(self):
        """Make a prediction for the selected ticker and model."""
        try:
            ticker = self.ticker_combo.currentText()
            model = self.model_combo.currentText()
            horizon = self.horizon_spin.value()
            confidence = self.confidence_spin.value()
            
            if not ticker or not model:
                self.status_label.setText("Please select a ticker and model")
                return
                
            # Generate request ID
            request_id = str(uuid.uuid4())
            self.pending_requests[request_id] = {
                'ticker': ticker,
                'model': model,
                'horizon': horizon,
                'confidence': confidence
            }
            
            # Send prediction request
            self.message_bus.publish(
                "Predictions",
                "prediction_request",
                {
                    'request_id': request_id,
                    'ticker': ticker,
                    'model': model,
                    'horizon': horizon,
                    'confidence': confidence
                }
            )
            
            self.status_label.setText(f"Making prediction for {ticker} using {model}...")
            
        except Exception as e:
            self.handle_error("Error making prediction", e)
            
    def handle_model_message(self, sender: str, message_type: str, data: Any):
        """Handle model-related messages."""
        try:
            if message_type == "models_list":
                self.model_combo.clear()
                self.model_combo.addItems(data.get('models', []))
            elif message_type == "model_details":
                self.update_model_details(data)
        except Exception as e:
            self.handle_error("Error handling model message", e)
            
    def handle_prediction_response(self, sender: str, message_type: str, data: Any):
        """Handle prediction response messages."""
        try:
            if message_type == "prediction_result":
                request_id = data.get('request_id')
                if request_id in self.pending_requests:
                    del self.pending_requests[request_id]
                    self.update_predictions_table(data.get('predictions', []))
                    self.update_metrics_table(data.get('metrics', {}))
                    self.status_label.setText("Prediction completed")
            elif message_type == "prediction_error":
                self.handle_error("Prediction error", Exception(data.get('error', 'Unknown error')))
        except Exception as e:
            self.handle_error("Error handling prediction response", e)
            
    def handle_data_message(self, sender: str, message_type: str, data: Any):
        """Handle data-related messages."""
        try:
            if message_type == "data_available":
                ticker = data.get('ticker')
                if ticker:
                    self.status_label.setText(f"Data available for {ticker}")
            elif message_type == "data_error":
                self.handle_error("Data error", Exception(data.get('error', 'Unknown error')))
        except Exception as e:
            self.handle_error("Error handling data message", e)
            
    def handle_imported_data(self, sender: str, message_type: str, data: Any):
        """Handle imported data messages."""
        try:
            if message_type == "import_complete":
                ticker = data.get('ticker')
                if ticker and ticker not in [self.ticker_combo.itemText(i) for i in range(self.ticker_combo.count())]:
                    self.ticker_combo.addItem(ticker)
                    self.status_label.setText(f"Added {ticker} to ticker list")
        except Exception as e:
            self.handle_error("Error handling imported data", e)
            
    def update_predictions_table(self, predictions: List[Dict[str, Any]]):
        """Update the predictions table with new data."""
        try:
            self.predictions_table.setRowCount(len(predictions))
            for i, pred in enumerate(predictions):
                self.predictions_table.setItem(i, 0, QTableWidgetItem(pred.get('date', '')))
                self.predictions_table.setItem(i, 1, QTableWidgetItem(str(pred.get('price', ''))))
                self.predictions_table.setItem(i, 2, QTableWidgetItem(str(pred.get('confidence', ''))))
                self.predictions_table.setItem(i, 3, QTableWidgetItem(pred.get('direction', '')))
        except Exception as e:
            self.handle_error("Error updating predictions table", e)
            
    def update_model_details(self, details: Dict[str, Any]):
        """Update the model details display."""
        try:
            text = f"Model: {details.get('name', 'Unknown')}\n"
            text += f"Type: {details.get('type', 'Unknown')}\n"
            text += f"Parameters: {details.get('parameters', 'None')}\n"
            text += f"Last Updated: {details.get('last_updated', 'Never')}"
            self.model_details.setText(text)
        except Exception as e:
            self.handle_error("Error updating model details", e)
            
    def update_metrics_table(self, metrics: Dict[str, Any]):
        """Update the metrics table with new data."""
        try:
            self.metrics_table.setRowCount(len(metrics))
            for i, (metric, value) in enumerate(metrics.items()):
                self.metrics_table.setItem(i, 0, QTableWidgetItem(metric))
                self.metrics_table.setItem(i, 1, QTableWidgetItem(str(value)))
        except Exception as e:
            self.handle_error("Error updating metrics table", e)
            
    def refresh_prediction(self):
        """Refresh the current prediction."""
        try:
            ticker = self.ticker_combo.currentText()
            if not ticker:
                self.status_label.setText("Please select a ticker")
                return
                
            # Request latest data
            self.message_bus.publish(
                "Predictions",
                "refresh_request",
                {
                    'ticker': ticker
                }
            )
            
            self.status_label.setText(f"Refreshing data for {ticker}...")
            
        except Exception as e:
            self.handle_error("Error refreshing prediction", e)
            
    def _on_data_update(self, source: str, data: pd.DataFrame, metadata: Dict[str, Any]):
        """Handle data updates from the DataManager."""
        try:
            # Extract metadata
            symbols = metadata.get("symbols", [])
            row_count = metadata.get("row_count", 0)
            columns = metadata.get("columns", [])
            
            # Update ticker combo box
            current_symbols = [self.ticker_combo.itemText(i) for i in range(self.ticker_combo.count())]
            for symbol in symbols:
                if symbol not in current_symbols:
                    self.ticker_combo.addItem(symbol)
                    self.logger.info(f"Added new symbol to predictions: {symbol}")
            
            # Update status
            self.status_label.setText(f"Received {row_count} rows of data with {len(symbols)} symbols")
            self.status_label.setStyleSheet("color: green")
            
            # Cache the data for predictions
            self.prediction_cache = data
            
        except Exception as e:
            self.handle_error("Error handling data update", e)
            
    def publish_prediction_results(self, results: Dict[str, Any]):
        """Publish prediction results via DataManager."""
        try:
            # Create metadata
            metadata = {
                "timestamp": time.time(),
                "prediction_type": results.get("type", "unknown"),
                "symbols": results.get("symbols", []),
                "horizon": results.get("horizon", 0),
                "confidence": results.get("confidence", 0.0),
                "source": "PredictionsTab"
            }
            
            # Convert results to DataFrame if needed
            if isinstance(results.get("data"), pd.DataFrame):
                df = results["data"]
            else:
                df = pd.DataFrame(results)
                
            # Add data to DataManager
            self.data_manager.add_data("PredictionsTab", df, metadata)
            
            # Update status
            self.status_label.setText(f"Published prediction results for {len(metadata['symbols'])} symbols")
            
        except Exception as e:
            self.handle_error("Error publishing prediction results", e)
            
    def cleanup(self):
        """Clean up resources."""
        try:
            # Unregister from DataManager
            self.data_manager.unregister_listener("PredictionsTab")
            
            # Clear caches
            self.prediction_cache.clear()
            self.pending_requests.clear()
            
            # Call parent cleanup
            super().cleanup()
            
        except Exception as e:
            self.handle_error("Error during cleanup", e)

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