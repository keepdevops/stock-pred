import sys
import os
import logging
from typing import Any, Dict
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QComboBox, QTextEdit, QFileDialog # Added widgets
)
import pandas as pd

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from stock_market_analyzer.modules.message_bus import MessageBus
# Import the ModelTrainer
try:
    from src.models.model_trainer import ModelTrainer
except ImportError as e:
    logging.error(f"Failed to import ModelTrainer: {e}. Ensure src directory is accessible.")
    # Define a placeholder if import fails to avoid crashing the tab
    class ModelTrainer:
        def __init__(self): logging.warning("Using placeholder ModelTrainer.")
        def prepare_data(self, df): logging.warning("Placeholder prepare_data"); return df
        def train(self, df): logging.warning("Placeholder train")
        def save_model(self, path): logging.warning(f"Placeholder save_model to {path}")

class ModelsTab(QWidget):
    """Models tab for the stock market analyzer."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.message_bus = MessageBus()
        self.logger = logging.getLogger(__name__)
        self.model_trainer = ModelTrainer() # Instantiate trainer
        self.available_data: Dict[str, pd.DataFrame] = {} # To store dataframes by ticker
        
        # Set up the widget layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # Add a placeholder label
        self.placeholder_label = QLabel("Models tab - Coming soon")
        self.layout.addWidget(self.placeholder_label)
        
        self.logger.info("Models tab initialized")
        
    def handle_message(self, sender: str, message_type: str, data: Any):
        """Handle incoming messages."""
        try:
            if message_type == "error":
                error_msg = f"Received error from {sender}: {data}"
                self.logger.error(error_msg)
        except Exception as e:
            error_log = f"Error handling message in Models tab: {str(e)}"
            self.logger.error(error_log)
            
    def publish_message(self, message_type: str, data: Any):
        """Publish a message to the message bus."""
        try:
            self.message_bus.publish("Models", message_type, data)
        except Exception as e:
            error_log = f"Error publishing message from Models tab: {str(e)}"
            self.logger.error(error_log)

    def setup_ui(self):
        """Setup the models tab UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # --- Controls --- 
        controls_layout = QHBoxLayout()

        # Ticker Selector for Training Data
        controls_layout.addWidget(QLabel("Data Ticker:"))
        self.data_ticker_combo = QComboBox()
        controls_layout.addWidget(self.data_ticker_combo)

        # Model Type Selector
        controls_layout.addWidget(QLabel("Model Type:"))
        self.model_type_combo = QComboBox()
        # Add actual model types your trainer supports
        self.model_type_combo.addItems(["LSTM", "XGBoost", "Transformer"]) 
        controls_layout.addWidget(self.model_type_combo)
        
        # Train Button
        self.train_button = QPushButton("Train Model")
        self.train_button.clicked.connect(self.train_model)
        controls_layout.addWidget(self.train_button)

        layout.addLayout(controls_layout)

        # --- Log/Status Display --- 
        layout.addWidget(QLabel("Training Log:"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        
        # Status label (repurposed for general status)
        self.status_label = QLabel("Models tab ready")
        layout.addWidget(self.status_label)
        
        # Subscribe to message bus
        self.message_bus.subscribe("Models", self.handle_message)
        self.log_message("Models tab initialized.")

    def log_message(self, message: str):
        """Appends a message to the log area."""
        self.logger.info(message) # Log to console/file as well
        self.log_text.append(message)
        QApplication.processEvents() # Update UI
        
    def train_model(self):
        """Handles the model training process."""
        selected_ticker = self.data_ticker_combo.currentText()
        selected_model_type = self.model_type_combo.currentText()

        if not selected_ticker:
            self.status_label.setText("Please select a ticker with loaded data.")
            self.log_message("Training aborted: No data ticker selected.")
            return

        if selected_ticker not in self.available_data:
            self.status_label.setText(f"Data for {selected_ticker} not available.")
            self.log_message(f"Training aborted: Data for {selected_ticker} not found.")
            return

        self.status_label.setText(f"Starting {selected_model_type} training for {selected_ticker}...")
        self.log_message(f"=== Starting Training ===")
        self.log_message(f"Ticker: {selected_ticker}")
        self.log_message(f"Model Type: {selected_model_type}")

        try:
            training_data = self.available_data[selected_ticker].copy()
            
            # 1. Prepare Data (Assuming ModelTrainer handles specific model prep)
            self.log_message("Preparing data...")
            prepared_data = self.model_trainer.prepare_data(training_data)
            self.log_message("Data preparation complete.")

            # 2. Train Model (Assuming ModelTrainer handles specific model training)
            self.log_message(f"Training {selected_model_type} model...")
            # Pass model type if trainer needs it, otherwise adapt trainer
            # self.model_trainer.train(prepared_data, model_type=selected_model_type) 
            self.model_trainer.train(prepared_data)
            self.log_message("Model training complete.")

            # 3. Save Model
            self.log_message("Prompting for model save location...")
            # Suggest a filename
            default_filename = f"{selected_ticker}_{selected_model_type}_model.joblib" # Or .pkl, .h5 etc.
            save_path, _ = QFileDialog.getSaveFileName(
                self, 
                "Save Trained Model", 
                default_filename, # Default file path/name
                "Model Files (*.joblib *.pkl *.h5);;All Files (*)" # File filters
            )

            if save_path:
                self.log_message(f"Saving model to: {save_path}")
                self.model_trainer.save_model(save_path)
                self.log_message("Model saved successfully.")
                self.status_label.setText("Training complete. Model saved.")
                # Optionally publish a message
                self.message_bus.publish("Models", "model_trained", {"ticker": selected_ticker, "type": selected_model_type, "path": save_path})
            else:
                self.log_message("Model saving cancelled.")
                self.status_label.setText("Training complete. Model not saved.")

            self.log_message(f"=== Training Finished ===")

        except Exception as e:
            error_msg = f"Error during training: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            self.log_message(error_msg)
            self.status_label.setText("Training failed.")
            self.message_bus.publish("Models", "error", error_msg)

def main():
    """Main function for the models tab process."""
    app = QApplication(sys.argv)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting models tab process")
    
    # Create and show the models tab
    window = ModelsTab()
    window.setWindowTitle("Models Tab")
    window.show()
    
    # Start the application event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 