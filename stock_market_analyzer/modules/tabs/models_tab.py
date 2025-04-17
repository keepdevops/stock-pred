import sys
import os
import logging
import traceback
from typing import List, Dict, Any
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QComboBox, QPushButton, QLabel, QFileDialog, QMessageBox, QListWidget,
    QListWidgetItem, QLineEdit, QSplitter, QApplication, QSpinBox, QCheckBox,
    QTabWidget, QScrollArea, QHeaderView, QFrame, QFormLayout, QGroupBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from .base_tab import BaseTab
from ..message_bus import MessageBus
import uuid
from ..models.model_manager import ModelManager
from datetime import datetime

class ModelsTab(BaseTab):
    """Models tab for managing and training prediction models."""
    
    def __init__(self, parent=None):
        """Initialize the Models tab."""
        # Initialize attributes before parent __init__
        self.model_cache = {}
        self.pending_requests = {}
        self._ui_setup_done = False
        self.main_layout = None
        self.model_list = None
        self.model_type_combo = None
        self.train_button = None
        self.status_label = None
        
        super().__init__(parent)
        
        # Setup UI after parent initialization
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the UI components."""
        try:
            # Clear the base layout
            while self.main_layout.count():
                item = self.main_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
                    
            self.main_layout.setSpacing(10)
            self.main_layout.setContentsMargins(10, 10, 10, 10)
            
            # Create models group
            models_group = QGroupBox("Model Management")
            models_layout = QVBoxLayout()
            
            # Model list
            self.model_list = QListWidget()
            self.model_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
            models_layout.addWidget(self.model_list)
            
            # Model type selection
            type_layout = QHBoxLayout()
            type_layout.addWidget(QLabel("Model Type:"))
            self.model_type_combo = QComboBox()
            self.model_type_combo.addItems([
                "LSTM",
                "GRU",
                "Transformer",
                "Random Forest",
                "XGBoost"
            ])
            type_layout.addWidget(self.model_type_combo)
            models_layout.addLayout(type_layout)
            
            # Action buttons
            button_layout = QHBoxLayout()
            
            self.train_button = QPushButton("Train New Model")
            self.train_button.clicked.connect(self.train_model)
            button_layout.addWidget(self.train_button)
            
            models_layout.addLayout(button_layout)
            models_group.setLayout(models_layout)
            self.main_layout.addWidget(models_group)
            
            # Create status bar
            status_layout = QHBoxLayout()
            
            self.status_label = QLabel("Ready")
            self.status_label.setStyleSheet("color: green")
            status_layout.addWidget(self.status_label)
            
            self.main_layout.addLayout(status_layout)
            
            self._ui_setup_done = True
            
        except Exception as e:
            error_msg = f"Error setting up UI: {str(e)}"
            self.logger.error(error_msg)
            if self.status_label:
                self.status_label.setText(error_msg)
                
    def _setup_message_bus_impl(self):
        """Setup message bus subscriptions."""
        super()._setup_message_bus_impl()
        self.message_bus.subscribe("Models", self.handle_message)
        
    def train_model(self):
        """Train a new model."""
        try:
            model_type = self.model_type_combo.currentText()
            
            # Generate request ID
            request_id = str(uuid.uuid4())
            
            # Create request
            request = {
                'request_id': request_id,
                'model_type': model_type,
                'timestamp': datetime.now()
            }
            
            # Add to pending requests
            self.pending_requests[request_id] = request
            
            # Publish request
            self.message_bus.publish(
                "Models",
                "train_model_request",
                request
            )
            
            # Update UI
            self.status_label.setText(f"Training {model_type} model...")
            self.train_button.setEnabled(False)
            
        except Exception as e:
            error_msg = f"Error training model: {str(e)}"
            self.logger.error(error_msg)
            self.status_label.setText(error_msg)
            
    def handle_message(self, sender: str, message_type: str, data: Any):
        """Handle incoming messages."""
        try:
            if message_type == "model_trained":
                self.handle_model_trained(sender, data)
            elif message_type == "error":
                self.status_label.setText(f"Error: {data.get('error', 'Unknown error')}")
                self.train_button.setEnabled(True)
                
        except Exception as e:
            error_msg = f"Error handling message: {str(e)}"
            self.logger.error(error_msg)
            self.status_label.setText(error_msg)
            
    def handle_model_trained(self, sender: str, data: Any):
        """Handle model trained response."""
        try:
            request_id = data.get('request_id')
            if request_id in self.pending_requests:
                request = self.pending_requests[request_id]
                model_name = data.get('model_name')
                
                if model_name:
                    # Add model to list
                    self.model_list.addItem(model_name)
                    self.model_cache[model_name] = data
                    
                self.status_label.setText(f"Model {model_name} trained successfully")
                self.train_button.setEnabled(True)
                del self.pending_requests[request_id]
                
        except Exception as e:
            error_msg = f"Error handling model trained response: {str(e)}"
            self.logger.error(error_msg)
            self.status_label.setText(error_msg)
            
    def cleanup(self):
        """Cleanup resources."""
        try:
            super().cleanup()
            self.model_cache.clear()
            self.pending_requests.clear()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            
    def closeEvent(self, event):
        """Handle the close event."""
        self.cleanup()
        super().closeEvent(event)

def main():
    """Main function for running the Models tab."""
    app = QApplication(sys.argv)
    window = ModelsTab()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()