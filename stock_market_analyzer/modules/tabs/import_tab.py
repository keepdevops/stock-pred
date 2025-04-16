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
    QTabWidget, QScrollArea, QFrame, QFileDialog, QTextEdit
)
from PyQt6.QtCore import Qt, QTimer, QDate
from PyQt6.QtGui import QFont, QTextCursor
from modules.tabs.base_tab import BaseTab
import uuid

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ..message_bus import MessageBus

class ImportTab(BaseTab):
    """Tab for importing data from various sources."""
    
    def __init__(self, parent=None):
        """Initialize the Import tab."""
        super().__init__(parent)
        self.message_bus = MessageBus()
        self.logger = logging.getLogger(__name__)
        self.import_cache = {}
        self.pending_requests = {}  # Track pending import requests
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the import tab UI."""
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create scroll area for each tab
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        
        # File Import tab
        file_import_tab = QWidget()
        file_import_layout = QVBoxLayout()
        
        # Add file import UI elements
        self.file_path_label = QLabel("No file selected")
        file_import_layout.addWidget(self.file_path_label)
        
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_file)
        file_import_layout.addWidget(browse_button)
        
        self.file_type_combo = QComboBox()
        self.file_type_combo.addItems(["CSV", "Excel", "JSON", "Parquet", "DuckDB"])
        file_import_layout.addWidget(self.file_type_combo)
        
        self.import_button = QPushButton("Import")
        self.import_button.clicked.connect(self.import_file)
        self.import_button.setEnabled(False)  # Disable until file is selected
        file_import_layout.addWidget(self.import_button)
        
        file_import_tab.setLayout(file_import_layout)
        
        # Database Import tab
        db_import_tab = QWidget()
        db_import_layout = QVBoxLayout()
        
        # Add database import UI elements
        self.db_type_combo = QComboBox()
        self.db_type_combo.addItems(["SQLite", "PostgreSQL", "MySQL", "DuckDB"])
        db_import_layout.addWidget(self.db_type_combo)
        
        self.db_path_label = QLabel("No database selected")
        db_import_layout.addWidget(self.db_path_label)
        
        db_browse_button = QPushButton("Browse")
        db_browse_button.clicked.connect(self.browse_database)
        db_import_layout.addWidget(db_browse_button)
        
        self.db_import_button = QPushButton("Import")
        self.db_import_button.clicked.connect(self.import_database)
        self.db_import_button.setEnabled(False)  # Disable until database is selected
        db_import_layout.addWidget(self.db_import_button)
        
        db_import_tab.setLayout(db_import_layout)
        
        # Add tabs to tab widget
        self.tab_widget.addTab(file_import_tab, "File Import")
        self.tab_widget.addTab(db_import_tab, "Database Import")
        
        # Add tab widget to main layout
        self.main_layout.addWidget(self.tab_widget)
        
        # Subscribe to message bus
        self.message_bus.subscribe("Import", self.handle_message)
        
        self.logger.info("Import tab initialized")

    def browse_file(self):
        """Opens a file dialog to select a data file."""
        file_type = self.file_type_combo.currentText().lower()
        file_filters = {
            "csv": "CSV Files (*.csv)",
            "excel": "Excel Files (*.xlsx *.xls)",
            "json": "JSON Files (*.json)",
            "parquet": "Parquet Files (*.parquet)",
            "hdf5": "HDF5 Files (*.h5 *.hdf5)",
            "sqlite": "SQLite Files (*.db *.sqlite)",
            "feather": "Feather Files (*.feather)",
            "pickle": "Pickle Files (*.pkl *.pickle)",
            "stata": "Stata Files (*.dta)",
            "sas": "SAS Files (*.sas7bdat)",
            "spss": "SPSS Files (*.sav)",
            "duckdb": "DuckDB Files (*.duckdb)"
        }
        
        file_filter = file_filters.get(file_type, "All Files (*)")
        path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Data File", 
            "", 
            file_filter
        )
        
        if path:
            self.selected_file_path = path
            self.file_path_label.setText(path)
            self.import_button.setEnabled(True)
            self.status_label.setText("File selected. Ready to import.")
            self.log_message(f"Selected file: {path}")
        else:
            self.selected_file_path = None
            self.file_path_label.clear()
            self.import_button.setEnabled(False)
            self.status_label.setText("File selection cancelled.")

    def import_file(self):
        """Import data from a file."""
        try:
            file_path = self.file_path_label.text()
            if not file_path:
                self.logger.warning("No file selected")
                return
                
            file_type = self.file_type_combo.currentText().lower()
            self.logger.info(f"Importing {file_type} file: {file_path}")
            
            # Import data based on file type
            if file_type == "csv":
                df = pd.read_csv(file_path)
            elif file_type == "excel":
                df = pd.read_excel(file_path)
            elif file_type == "json":
                df = pd.read_json(file_path)
            elif file_type == "parquet":
                df = pd.read_parquet(file_path)
            elif file_type == "duckdb":
                import duckdb
                conn = duckdb.connect(file_path)
                df = conn.execute("SELECT * FROM data").fetchdf()
                conn.close()
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
            # Process and publish the data
            self.process_and_publish_data(df)
            
        except Exception as e:
            error_msg = f"Error importing file: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            self.status_label.setText(error_msg)
            
    def import_database(self):
        """Import data from a database."""
        try:
            if not self.selected_db_path:
                self.logger.warning("No database selected")
                return
                
            db_type = self.db_type_combo.currentText().lower()
            self.logger.info(f"Importing from {db_type} database: {self.selected_db_path}")
            
            # Import data based on database type
            if db_type == "sqlite":
                import sqlite3
                conn = sqlite3.connect(self.selected_db_path)
                df = pd.read_sql("SELECT * FROM data", conn)
                conn.close()
            elif db_type == "duckdb":
                import duckdb
                conn = duckdb.connect(self.selected_db_path)
                df = conn.execute("SELECT * FROM data").fetchdf()
                conn.close()
            elif db_type == "postgresql":
                # PostgreSQL uses connection strings
                conn = psycopg2.connect(self.selected_db_path)
                df = pd.read_sql("SELECT * FROM data", conn)
                conn.close()
            elif db_type == "mysql":
                # MySQL uses connection strings
                conn = mysql.connector.connect(self.selected_db_path)
                df = pd.read_sql("SELECT * FROM data", conn)
                conn.close()
            else:
                raise ValueError(f"Unsupported database type: {db_type}")
                
            # Process and publish the data
            self.process_and_publish_data(df)
            
        except Exception as e:
            error_msg = f"Error importing from database: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            self.status_label.setText(error_msg)
            
    def import_data(self):
        """Import data from the selected source."""
        try:
            if self.tab_widget.currentIndex() == 0:  # File Import
                self.import_file()
            else:  # Database Import
                self.import_database()
        except Exception as e:
            error_msg = f"Error during import: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            self.status_label.setText(error_msg)

    def import_from_file(self):
        """Import data from a file."""
        if not self.selected_file_path:
            self.log_message("No file selected.")
            return

        file_ext = os.path.splitext(self.selected_file_path)[1].lower()
        self.log_message(f"=== Starting Import: {os.path.basename(self.selected_file_path)} ===")

        # Read file based on type
        file_type = self.file_type_combo.currentText().lower()
        if file_type == "csv":
            # Try both pandas and polars
            try:
                df = pd.read_csv(self.selected_file_path)
            except Exception as e:
                self.log_message(f"Pandas CSV import failed, trying Polars: {str(e)}")
                df = pl.read_csv(self.selected_file_path).to_pandas()
        elif file_type == "excel":
            df = pd.read_excel(self.selected_file_path)
        elif file_type == "json":
            # Try both pandas and polars
            try:
                df = pd.read_json(self.selected_file_path)
            except Exception as e:
                self.log_message(f"Pandas JSON import failed, trying Polars: {str(e)}")
                df = pl.read_json(self.selected_file_path).to_pandas()
        elif file_type == "parquet":
            # Try both pandas and polars
            try:
                df = pd.read_parquet(self.selected_file_path)
            except Exception as e:
                self.log_message(f"Pandas Parquet import failed, trying Polars: {str(e)}")
                df = pl.read_parquet(self.selected_file_path).to_pandas()
        elif file_type == "hdf5":
            df = pd.read_hdf(self.selected_file_path)
        elif file_type == "sqlite":
            conn = sqlite3.connect(self.selected_file_path)
            df = pd.read_sql("SELECT * FROM data", conn)
            conn.close()
        elif file_type == "feather":
            df = pd.read_feather(self.selected_file_path)
        elif file_type == "pickle":
            df = pd.read_pickle(self.selected_file_path)
        elif file_type == "stata":
            df = pd.read_stata(self.selected_file_path)
        elif file_type == "sas":
            df = pd.read_sas(self.selected_file_path)
        elif file_type == "spss":
            df = pd.read_spss(self.selected_file_path)
        elif file_type == "duckdb":
            conn = duckdb.connect(self.selected_file_path)
            df = conn.execute("SELECT * FROM data").fetchdf()
            conn.close()
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        self.process_and_publish_data(df)

    def import_from_database(self):
        """Import data from a database."""
        db_type = self.db_type_combo.currentText()
        query = self.query_edit.toPlainText()
        
        if not query:
            raise ValueError("Please enter a SQL query")
            
        self.log_message(f"=== Starting Database Import: {db_type} ===")
        
        # Get connection parameters from UI
        params = {}
        for i in range(self.connection_layout.rowCount()):
            label = self.connection_layout.itemAt(i, QFormLayout.LabelRole).widget().text()
            value = self.connection_layout.itemAt(i, QFormLayout.FieldRole).widget().text()
            params[label.lower().replace(":", "")] = value
            
        # Create database connection
        if db_type == "SQLite":
            conn = sqlite3.connect(params["database file"])
            df = pd.read_sql(query, conn)
            conn.close()
        elif db_type == "DuckDB":
            conn = duckdb.connect(params["database file"])
            df = conn.execute(query).fetchdf()
            conn.close()
        elif db_type == "MySQL":
            conn = mysql.connector.connect(
                host=params["host"],
                port=int(params["port"]),
                database=params["database"],
                user=params["username"],
                password=params["password"]
            )
            df = pd.read_sql(query, conn)
            conn.close()
        elif db_type == "PostgreSQL":
            conn = psycopg2.connect(
                host=params["host"],
                port=params["port"],
                database=params["database"],
                user=params["username"],
                password=params["password"]
            )
            df = pd.read_sql(query, conn)
            conn.close()
        elif db_type == "SQL Server":
            conn = pyodbc.connect(
                f"DRIVER={{SQL Server}};SERVER={params['host']};DATABASE={params['database']};"
                f"UID={params['username']};PWD={params['password']}"
            )
            df = pd.read_sql(query, conn)
            conn.close()
        elif db_type == "Oracle":
            conn = create_engine(
                f"oracle+cx_oracle://{params['username']}:{params['password']}@"
                f"{params['host']}:{params['port']}/{params['database']}"
            ).connect()
            df = pd.read_sql(query, conn)
            conn.close()
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
        
        self.process_and_publish_data(df)

    def update_preview_table(self, df):
        """Update the preview table with the imported data."""
        # Clear existing data
        self.preview_table.setRowCount(0)
        self.preview_table.setColumnCount(0)
        
        if df is None or df.empty:
            self.row_count_label.setText("Rows: 0")
            return
            
        # Set column headers
        self.preview_table.setColumnCount(len(df.columns))
        self.preview_table.setHorizontalHeaderLabels(df.columns)
        
        # Set row count
        self.preview_table.setRowCount(min(100, len(df)))  # Show first 100 rows
        self.row_count_label.setText(f"Rows: {len(df)} (showing first 100)")
        
        # Fill table with data
        for i in range(min(100, len(df))):
            for j, col in enumerate(df.columns):
                value = df.iloc[i, j]
                if pd.isna(value):
                    item = QTableWidgetItem("")
                else:
                    item = QTableWidgetItem(str(value))
                self.preview_table.setItem(i, j, item)
        
        # Resize columns to fit content
        self.preview_table.resizeColumnsToContents()

    def process_and_publish_data(self, df):
        """Process the DataFrame and publish it to the message bus."""
        # Convert date strings to datetime if present
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for col in date_columns:
            df[col] = pd.to_datetime(df[col])

        # Validate required columns
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

        # Standardize column names
        df = df.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close'
        })

        # Set date as index if it's not already
        if 'date' in df.columns:
            df.set_index('date', inplace=True)

        # Sort by date
        df.sort_index(inplace=True)

        # Update preview table
        self.update_preview_table(df)

        # Publish the imported data
        source = "file" if self.tab_widget.currentIndex() == 0 else "database"
        self.message_bus.publish("Import", "data_imported", (source, df))
        self.log_message(f"Successfully imported data from {source}")
        self.log_message(f"Data shape: {df.shape}")
        self.log_message(f"Date range: {df.index.min()} to {df.index.max()}")

    def log_message(self, message: str):
        """Appends a message to the log area."""
        self.logger.info(message)
        self.log_text.append(message)
        QApplication.processEvents()

    def handle_message(self, sender: str, message_type: str, data: Any):
        """Handle incoming messages."""
        try:
            if message_type == "error":
                error_msg = f"Received error from {sender}: {data}"
                self.log_message(error_msg)
        except Exception as e:
            error_log = f"Error handling message in ImportTab: {str(e)}"
            self.logger.error(error_log)
            self.log_message(error_log)

    def browse_database(self):
        """Opens a file dialog to select a database file."""
        db_type = self.db_type_combo.currentText().lower()
        file_filters = {
            "sqlite": "SQLite Files (*.db *.sqlite)",
            "duckdb": "DuckDB Files (*.duckdb)",
            "postgresql": "All Files (*)",  # PostgreSQL uses connection strings
            "mysql": "All Files (*)"  # MySQL uses connection strings
        }
        
        file_filter = file_filters.get(db_type, "All Files (*)")
        path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Database File", 
            "", 
            file_filter
        )
        
        if path:
            self.selected_db_path = path
            self.db_path_label.setText(path)
            self.db_import_button.setEnabled(True)
            self.status_label.setText("Database selected. Ready to import.")
            self.log_message(f"Selected database: {path}")
        else:
            self.selected_db_path = None
            self.db_path_label.clear()
            self.db_import_button.setEnabled(False)
            self.status_label.setText("Database selection cancelled.")

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