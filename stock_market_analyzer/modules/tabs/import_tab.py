import sys
import os
import logging
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import psycopg2
import mysql.connector
from sqlalchemy import create_engine
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QComboBox, QPushButton, QLabel, QSplitter, QApplication, QSpinBox,
    QDoubleSpinBox, QGroupBox, QCheckBox, QHeaderView, QMessageBox, QDateEdit,
    QTabWidget, QScrollArea, QFrame, QFileDialog, QTextEdit, QFormLayout, QLineEdit,
    QGridLayout
)
from PyQt6.QtCore import Qt, QTimer, QDate
from PyQt6.QtGui import QFont, QTextCursor
from modules.tabs.base_tab import BaseTab
from modules.database import DatabaseConnector
import uuid
from modules.message_bus import MessageBus

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class ImportTab(BaseTab):
    """Import tab for importing stock data from files or databases."""
    
    def __init__(self, parent=None):
        """Initialize the Import tab."""
        # Initialize attributes before parent __init__
        self.import_progress = 0
        self.import_status = "Ready"
        self._ui_setup_done = False
        self.main_layout = None
        self.file_path_edit = None
        self.file_type_combo = None
        self.import_button = None
        self.host_edit = None
        self.port_edit = None
        self.user_edit = None
        self.password_edit = None
        self.database_edit = None
        self.db_import_button = None
        self.status_label = None
        self.selected_file_path = None
        self.selected_db_path = None
        self.import_cache = {}
        self.pending_requests = {}
        
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
            
            # Create file import group
            file_group = QGroupBox("File Import")
            file_layout = QVBoxLayout()
            
            # File selection
            file_select_layout = QHBoxLayout()
            self.file_path_edit = QLineEdit()
            self.file_path_edit.setPlaceholderText("Select file...")
            self.file_path_edit.setReadOnly(True)
            file_select_layout.addWidget(self.file_path_edit)
            
            browse_button = QPushButton("Browse")
            browse_button.clicked.connect(self.browse_file)
            file_select_layout.addWidget(browse_button)
            
            file_layout.addLayout(file_select_layout)
            
            # File type selection
            type_layout = QHBoxLayout()
            type_layout.addWidget(QLabel("File Type:"))
            self.file_type_combo = QComboBox()
            self.file_type_combo.addItems([
                "CSV",
                "Excel",
                "JSON",
                "Parquet"
            ])
            type_layout.addWidget(self.file_type_combo)
            file_layout.addLayout(type_layout)
            
            # Import button
            self.import_button = QPushButton("Import File")
            self.import_button.clicked.connect(self.import_file)
            file_layout.addWidget(self.import_button)
            
            file_group.setLayout(file_layout)
            self.main_layout.addWidget(file_group)
            
            # Create database import group
            db_group = QGroupBox("Database Import")
            db_layout = QVBoxLayout()
            
            # Connection settings
            settings_layout = QGridLayout()
            
            # Host
            settings_layout.addWidget(QLabel("Host:"), 0, 0)
            self.host_edit = QLineEdit()
            self.host_edit.setPlaceholderText("localhost")
            settings_layout.addWidget(self.host_edit, 0, 1)
            
            # Port
            settings_layout.addWidget(QLabel("Port:"), 0, 2)
            self.port_edit = QLineEdit()
            self.port_edit.setPlaceholderText("5432")
            settings_layout.addWidget(self.port_edit, 0, 3)
            
            # User
            settings_layout.addWidget(QLabel("User:"), 1, 0)
            self.user_edit = QLineEdit()
            self.user_edit.setPlaceholderText("username")
            settings_layout.addWidget(self.user_edit, 1, 1)
            
            # Password
            settings_layout.addWidget(QLabel("Password:"), 1, 2)
            self.password_edit = QLineEdit()
            self.password_edit.setEchoMode(QLineEdit.EchoMode.Password)
            settings_layout.addWidget(self.password_edit, 1, 3)
            
            # Database
            settings_layout.addWidget(QLabel("Database:"), 2, 0)
            self.database_edit = QLineEdit()
            self.database_edit.setPlaceholderText("database_name")
            settings_layout.addWidget(self.database_edit, 2, 1, 1, 3)
            
            db_layout.addLayout(settings_layout)
            
            # Import button
            self.db_import_button = QPushButton("Import from Database")
            self.db_import_button.clicked.connect(self.import_database)
            db_layout.addWidget(self.db_import_button)
            
            db_group.setLayout(db_layout)
            self.main_layout.addWidget(db_group)
            
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
        self.message_bus.subscribe("Import", self.handle_message)
        
    def browse_file(self):
        """Open file dialog to select a file."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select File",
                "",
                "All Files (*);;CSV Files (*.csv);;Excel Files (*.xlsx);;JSON Files (*.json);;Parquet Files (*.parquet)"
            )
            
            if file_path:
                self.file_path_edit.setText(file_path)
                self.selected_file_path = file_path
                
        except Exception as e:
            error_msg = f"Error browsing file: {str(e)}"
            self.logger.error(error_msg)
            self.status_label.setText(error_msg)
            
    def import_file(self):
        """Import data from selected file."""
        try:
            file_path = self.file_path_edit.text()
            file_type = self.file_type_combo.currentText()
            
            if not file_path:
                self.status_label.setText("Please select a file")
                return
                
            # Generate request ID
            request_id = str(uuid.uuid4())
            
            # Create request
            request = {
                'request_id': request_id,
                'file_path': file_path,
                'file_type': file_type,
                'timestamp': datetime.now()
            }
            
            # Add to pending requests
            self.pending_requests[request_id] = request
            
            # Publish request
            self.message_bus.publish(
                "Import",
                "file_import_request",
                request
            )
            
            # Update UI
            self.status_label.setText(f"Importing {file_type} file: {file_path}")
            self.import_button.setEnabled(False)
            
        except Exception as e:
            error_msg = f"Error importing file: {str(e)}"
            self.logger.error(error_msg)
            self.status_label.setText(error_msg)
            
    def import_database(self):
        """Import data from database."""
        try:
            host = self.host_edit.text()
            port = self.port_edit.text()
            user = self.user_edit.text()
            password = self.password_edit.text()
            database = self.database_edit.text()
            
            if not all([host, port, user, database]):
                self.status_label.setText("Please fill in all database connection details")
                return
                
            # Generate request ID
            request_id = str(uuid.uuid4())
            
            # Create request
            request = {
                'request_id': request_id,
                'host': host,
                'port': port,
                'user': user,
                'password': password,
                'database': database,
                'timestamp': datetime.now()
            }
            
            # Add to pending requests
            self.pending_requests[request_id] = request
            
            # Publish request
            self.message_bus.publish(
                "Import",
                "db_import_request",
                request
            )
            
            # Update UI
            self.status_label.setText(f"Importing from database: {database}")
            self.db_import_button.setEnabled(False)
            
        except Exception as e:
            error_msg = f"Error importing from database: {str(e)}"
            self.logger.error(error_msg)
            self.status_label.setText(error_msg)
            
    def handle_message(self, sender: str, message_type: str, data: Any):
        """Handle incoming messages."""
        try:
            if message_type == "import_response":
                self.handle_import_response(sender, data)
            elif message_type == "error":
                self.status_label.setText(f"Error: {data.get('error', 'Unknown error')}")
                
        except Exception as e:
            error_msg = f"Error handling message: {str(e)}"
            self.logger.error(error_msg)
            self.status_label.setText(error_msg)
            
    def handle_import_response(self, sender: str, data: Any):
        """Handle import response."""
        try:
            request_id = data.get('request_id')
            if request_id in self.pending_requests:
                request = self.pending_requests[request_id]
                
                if 'file_path' in request:
                    self.status_label.setText(f"Successfully imported file: {request['file_path']}")
                    self.import_button.setEnabled(True)
                else:
                    self.status_label.setText(f"Successfully imported from database: {request['database']}")
                    self.db_import_button.setEnabled(True)
                    
                del self.pending_requests[request_id]
                
        except Exception as e:
            error_msg = f"Error handling import response: {str(e)}"
            self.logger.error(error_msg)
            self.status_label.setText(error_msg)
            
    def cleanup(self):
        """Cleanup resources."""
        try:
            super().cleanup()
            self.import_cache.clear()
            self.pending_requests.clear()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            
    def closeEvent(self, event):
        """Handle the close event."""
        self.cleanup()
        super().closeEvent(event)

def main():
    """Main function for the import tab process."""
    app = QApplication.instance() 
    if not app: 
        app = QApplication(sys.argv)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("Starting import tab process")
    
    try:
        window = ImportTab()
        window.setWindowTitle("Import Tab")
        window.show()
    except Exception as e:
         logger.error(f"Failed to create or show ImportTab window: {e}")
         logger.error(traceback.format_exc())
         sys.exit(1)

    if __name__ == "__main__":
        sys.exit(app.exec())

if __name__ == "__main__":
    import traceback 
    main() 