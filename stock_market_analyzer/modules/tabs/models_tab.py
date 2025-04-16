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

class ModelsTab(BaseTab):
    """Tab for managing and training machine learning models."""
    
    def __init__(self, parent=None):
        """Initialize the Models tab."""
        super().__init__(parent)
        self.model_manager = None  # Will be initialized in setup_ui
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the models tab UI."""
        # Create main layout
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create scroll area for each tab
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        
        # Models List tab
        models_list_tab = QWidget()
        models_list_layout = QVBoxLayout()
        
        # Add models list UI elements
        self.models_table = QTableWidget()
        self.models_table.setColumnCount(4)
        self.models_table.setHorizontalHeaderLabels([
            "Model Name", "Type", "Status", "Last Updated"
        ])
        self.models_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        models_list_layout.addWidget(self.models_table)
        
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.update_model_list)
        models_list_layout.addWidget(refresh_button)
        
        models_list_tab.setLayout(models_list_layout)
        
        # Add tabs to tab widget
        self.tab_widget.addTab(models_list_tab, "Models List")
        
        # Add tab widget to main layout
        self.main_layout.addWidget(self.tab_widget)
        
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
            self.models_table.setRowCount(len(models))
            
            for row, model in enumerate(models):
                self.models_table.setItem(row, 0, QTableWidgetItem(model.get('name', '')))
                self.models_table.setItem(row, 1, QTableWidgetItem(model.get('type', '')))
                self.models_table.setItem(row, 2, QTableWidgetItem(model.get('status', '')))
                self.models_table.setItem(row, 3, QTableWidgetItem(model.get('last_updated', '')))
                
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