import sys
import os
import logging
import traceback
from typing import List, Dict, Any
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QComboBox, QPushButton, QLabel, QFileDialog, QMessageBox, QListWidget,
    QListWidgetItem, QLineEdit, QSplitter, QApplication, QSpinBox, QCheckBox,
    QTabWidget, QScrollArea, QHeaderView, QFrame
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from .base_tab import BaseTab
from ..message_bus import MessageBus
import uuid

class ModelsTab(BaseTab):
    """Tab for managing and training machine learning models."""
    
    def __init__(self, parent=None):
        """Initialize the Models tab."""
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.model_manager = None
        self.status_label = None
        self.models_list = None
        self.setup_ui()  # Call setup_ui after initializing instance variables
        self.setup_message_bus()
        
    def setup_message_bus(self):
        """Setup message bus subscriptions."""
        self.message_bus.subscribe("Models", self.handle_message)
        self.logger.debug("Subscribed to Models topic")
        
    def setup_ui(self):
        """Setup the models tab UI."""
        # Create main layout
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        
        # Initialize model manager
        try:
            from ..model_manager import ModelManager
            self.model_manager = ModelManager()
            self.logger.info("Model manager initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize model manager: {e}")
            self.model_manager = None
            
        # Create models list
        self.models_list = QListWidget()
        self.main_layout.addWidget(self.models_list)
        
        # Add status label
        self.status_label = QLabel("Ready")
        self.main_layout.addWidget(self.status_label)
        
        self.logger.info("Models tab initialized")
        
        # Initial model list update
        self.update_model_list()
        
    def update_model_list(self):
        """Update the list of models."""
        try:
            if self.model_manager:
                models = self.model_manager.get_trained_models()
                
                # Clear and update the list widget
                self.models_list.clear()
                model_names = []
                for model in models:
                    if isinstance(model, dict):
                        model_name = model.get('name', 'Unknown Model')
                    else:
                        model_name = str(model)
                    self.models_list.addItem(model_name)
                    model_names.append(model_name)
                
                # Publish model list update
                self.logger.debug(f"Publishing model list update with models: {model_names}")
                self.message_bus.publish("Models", "model_list", {
                    "models": model_names,
                    "source": "ModelsTab"
                })
                
                # Also publish individual model added events
                for model_name in model_names:
                    self.message_bus.publish("Models", "model_added", {
                        "model_name": model_name,
                        "model_type": "LSTM"
                    })
        except Exception as e:
            self.logger.error(f"Error updating model list: {e}")
            
    def handle_message(self, topic: str, message_type: str, data: dict):
        """Handle incoming messages."""
        try:
            if message_type == "error":
                error_message = data.get('error', 'Unknown error')
                self.status_label.setText(f"Error: {error_message}")
                self.logger.error(f"Received error: {error_message}")
            elif message_type in ["model_updated", "model_added", "model_deleted"]:
                self.update_model_list()
                self.status_label.setText("Model list updated")
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
            self.status_label.setText(f"Error handling message: {str(e)}")

    def cleanup(self):
        """Clean up resources before destruction."""
        if hasattr(self, 'message_bus'):
            self.message_bus.unsubscribe("Models", self.handle_message)
        super().cleanup()

    def closeEvent(self, event):
        """Handle the widget close event."""
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