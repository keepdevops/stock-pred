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

class ModelsTab(BaseTab):
    """Tab for managing and training machine learning models."""
    
    def __init__(self, parent=None):
        """Initialize the Models tab."""
        super().__init__(parent)
        self.message_bus = MessageBus()
        self.logger = logging.getLogger(__name__)
        self.model_manager = None  # Will be initialized in setup_ui
        self.status_label = QLabel("Ready")
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the models tab UI."""
        # Create main layout if it doesn't exist
        if not hasattr(self, 'main_layout'):
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
        self.main_layout.addWidget(self.status_label)
        
        # Subscribe to message bus
        self.message_bus.subscribe("Models", self.handle_message)
        
        self.logger.info("Models tab initialized")
        
        # Initial model list update
        self.update_model_list()
        
    def update_model_list(self):
        """Update the list of trained models."""
        try:
            if self.model_manager is None:
                self.logger.warning("Model manager not initialized")
                return
                
            models = self.model_manager.get_trained_models()
            self.models_list.clear()
            self.models_list.addItems([model.get('name', '') for model in models])
            
        except Exception as e:
            self.logger.error(f"Error updating model list: {e}")
            self.status_label.setText(f"Error: {str(e)}")
            
    def handle_message(self, sender: str, message_type: str, data: Any):
        """Handle incoming messages."""
        try:
            if message_type == "error":
                error_msg = f"Received error from {sender}: {data}"
                self.logger.error(error_msg)
                self.status_label.setText(error_msg)
            elif message_type == "model_updated":
                self.update_model_list()
        except Exception as e:
            self.logger.error(f"Error handling message in Models tab: {str(e)}")

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