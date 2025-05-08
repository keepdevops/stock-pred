import sys
import os
import logging
import traceback
import time
import uuid
from typing import Dict, Any, List, Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QTableWidget, QTableWidgetItem, QGroupBox,
    QHeaderView, QMessageBox, QFileDialog, QSpinBox, QCheckBox,
    QTabWidget, QScrollArea, QFrame, QFormLayout, QListWidget,
    QListWidgetItem, QSplitter, QApplication, QDoubleSpinBox, QLineEdit
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from .base_tab import BaseTab
from ..message_bus import MessageBus
from ..settings import Settings
from ..connection_dashboard import ConnectionDashboard
from ..models.model_manager import ModelManager
import numpy as np

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class ModelsTab(BaseTab):
    """Models tab for managing and training prediction models."""
    
    def __init__(self, message_bus: MessageBus, parent=None):
        """Initialize the Models tab.
        
        Args:
            message_bus: The message bus for communication.
            parent: The parent widget.
        """
        # Initialize attributes before parent __init__
        self.settings = Settings()
        self.current_color_scheme = self.settings.get_color_scheme()
        self._ui_setup_done = False
        self.model_list = None
        self.model_type_combo = None
        self.train_button = None
        self.model_cache = {}
        self.pending_requests = {}
        self.connection_status = {}
        self.connection_start_times = {}
        self.messages_received = 0
        self.messages_sent = 0
        self.errors = 0
        self.message_latencies = []
        self.dashboard = ConnectionDashboard()
        self.model_manager = ModelManager()
        
        # Call parent __init__ with message bus
        super().__init__(message_bus, parent)
        
    def _setup_message_bus_impl(self):
        """Setup message bus subscriptions."""
        try:
            self.message_bus.subscribe("Models", self.handle_model_message)
            self.message_bus.subscribe("Data", self.handle_data_message)
            self.message_bus.subscribe("ConnectionStatus", self.handle_connection_status)
            self.logger.debug("Subscribed to Models, Data, and ConnectionStatus topics")
        except Exception as e:
            self.handle_error("Error setting up message bus subscriptions", e)
            
    def setup_ui(self):
        """Set up the UI components."""
        if self._ui_setup_done:
            return
            
        try:
            # Clear the base layout
            while self.main_layout.count():
                item = self.main_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
                    
            self.main_layout.setSpacing(10)
            self.main_layout.setContentsMargins(10, 10, 10, 10)
            
            # Create model type selection
            model_type_group = QGroupBox("Model Type")
            model_type_layout = QHBoxLayout()
            model_type_group.setLayout(model_type_layout)
            
            self.model_type_combo = QComboBox()
            self.model_type_combo.addItem("LSTM")
            self.model_type_combo.addItem("Transformer")
            
            # Only add XGBoost if it's available
            try:
                import xgboost
                self.model_type_combo.addItem("XGBoost")
            except ImportError:
                pass
                
            model_type_layout.addWidget(QLabel("Model Type:"))
            model_type_layout.addWidget(self.model_type_combo)
            
            # Create train button
            self.train_button = QPushButton("Train Model")
            self.train_button.clicked.connect(self.train_model)
            model_type_layout.addWidget(self.train_button)
            
            self.main_layout.addWidget(model_type_group)
            
            # Create model list
            model_list_group = QGroupBox("Models")
            model_list_layout = QVBoxLayout()
            model_list_group.setLayout(model_list_layout)
            
            self.model_list = QListWidget()
            self.model_list.itemSelectionChanged.connect(self.on_model_selected)
            model_list_layout.addWidget(self.model_list)
            
            self.main_layout.addWidget(model_list_group)
            
            # Update model list
            self.update_model_list()
            
            self._ui_setup_done = True
            self.logger.info("Models tab initialized")
            
        except Exception as e:
            self.handle_error("Error setting up UI", e)
            
    def train_model(self):
        """Train a new model."""
        try:
            model_type = self.model_type_combo.currentText()
            
            if not model_type:
                self.status_label.setText("Please select a model type")
                return
                
            self.status_label.setText(f"Training {model_type} model...")
            
            # Publish training request
            self.message_bus.publish(
                "Models",
                "train_model",
                {
                    "model_type": model_type,
                    "timestamp": time.time()
                }
            )
            
        except Exception as e:
            self.handle_error("Error training model", e)
            
    def on_model_selected(self):
        """Handle model selection."""
        try:
            selected_items = self.model_list.selectedItems()
            if not selected_items:
                return
                
            model_name = selected_items[0].text()
            self.status_label.setText(f"Selected model: {model_name}")
            
            # Publish model selection
            self.message_bus.publish(
                "Models",
                "model_selected",
                {
                    "model_name": model_name,
                    "timestamp": time.time()
                }
            )
            
        except Exception as e:
            self.handle_error("Error handling model selection", e)
            
    def update_model_list(self):
        """Update the model list."""
        try:
            if not self.model_manager:
                return
                
            # Clear current items
            self.model_list.clear()
            
            # Add models to the list
            for model in self.model_manager.get_trained_models():
                self.model_list.addItem(model.name)
                
        except Exception as e:
            self.handle_error("Error updating model list", e)
            
    def handle_model_message(self, sender: str, message_type: str, data: Dict[str, Any]):
        """Handle model-related messages."""
        try:
            if message_type == "model_trained":
                self.update_model_list()
                self.status_label.setText("Model training completed")
            elif message_type == "model_error":
                error_msg = data.get("error", "Unknown error")
                self.handle_error("Model error", Exception(error_msg))
                
        except Exception as e:
            self.handle_error("Error handling model message", e)
            
    def handle_data_message(self, sender: str, message_type: str, data: Dict[str, Any]):
        """Handle data-related messages."""
        try:
            if message_type == "data_ready":
                self.status_label.setText("Data ready for training")
            elif message_type == "data_error":
                error_msg = data.get("error", "Unknown error")
                self.handle_error("Data error", Exception(error_msg))
                
        except Exception as e:
            self.handle_error("Error handling data message", e)
            
    def handle_connection_status(self, sender: str, message_type: str, data: Dict[str, Any]):
        """Handle connection status messages."""
        try:
            status = data.get("status")
            if status:
                self.connection_status[sender] = status
                self.update_connection_dashboard()
                
        except Exception as e:
            self.handle_error("Error handling connection status", e)
            
    def update_connection_dashboard(self):
        """Update the connection dashboard."""
        try:
            # Update connection data for each tab
            for tab, status in self.connection_status.items():
                self.dashboard.update_connection_data(tab, {
                    'status': status == 'connected',
                    'start_time': self.connection_start_times.get(tab),
                    'last_heartbeat': time.time(),
                    'messages_received': self.messages_received,
                    'messages_sent': self.messages_sent,
                    'errors': self.errors,
                    'latencies': self.message_latencies
                })
            
        except Exception as e:
            self.handle_error("Error updating connection dashboard", e)
            
    def cleanup(self):
        """Clean up resources."""
        try:
            # Clear caches
            self.model_cache.clear()
            self.pending_requests.clear()
            self.connection_status.clear()
            self.connection_start_times.clear()
            
            # Reset metrics
            self.messages_received = 0
            self.messages_sent = 0
            self.errors = 0
            self.message_latencies.clear()
            
            # Clean up model manager
            if self.model_manager:
                self.model_manager.cleanup()
                
            # Call parent cleanup
            super().cleanup()
            
        except Exception as e:
            self.handle_error("Error during cleanup", e)

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