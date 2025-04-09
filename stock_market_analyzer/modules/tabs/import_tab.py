import sys
import os
import logging
from typing import Any
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QFileDialog,
    QComboBox, QGroupBox, QFormLayout, QTabWidget, QTableWidget,
    QTableWidgetItem, QHeaderView
)
import pandas as pd
import sqlite3
import pyodbc
import psycopg2
import mysql.connector
from sqlalchemy import create_engine
import duckdb
import polars as pl
import json

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from stock_market_analyzer.modules.message_bus import MessageBus

class ImportTab(QWidget):
    """Tab for importing data from files and databases."""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.message_bus = MessageBus()
        self.selected_file_path = None
        self.db_connection = None
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the import tab UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create tab widget for different import sources
        self.tab_widget = QTabWidget()
        
        # File Import Tab
        file_tab = QWidget()
        file_layout = QVBoxLayout()
        
        # File Selection
        file_select_group = QGroupBox("File Selection")
        file_select_layout = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("Select a file to import...")
        self.file_path_edit.setReadOnly(True)
        file_select_layout.addWidget(self.file_path_edit)

        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_file)
        file_select_layout.addWidget(self.browse_button)
        file_select_group.setLayout(file_select_layout)
        file_layout.addWidget(file_select_group)
        
        # File Type Selection
        file_type_group = QGroupBox("File Type")
        file_type_layout = QFormLayout()
        self.file_type_combo = QComboBox()
        self.file_type_combo.addItems([
            "CSV", "Excel", "JSON", "Parquet", "HDF5", "SQLite",
            "Feather", "Pickle", "Stata", "SAS", "SPSS", "DuckDB"
        ])
        file_type_layout.addRow("Select File Type:", self.file_type_combo)
        file_type_group.setLayout(file_type_layout)
        file_layout.addWidget(file_type_group)
        
        file_tab.setLayout(file_layout)
        self.tab_widget.addTab(file_tab, "File Import")
        
        # Database Import Tab
        db_tab = QWidget()
        db_layout = QVBoxLayout()
        
        # Database Type Selection
        db_type_group = QGroupBox("Database Type")
        db_type_layout = QFormLayout()
        self.db_type_combo = QComboBox()
        self.db_type_combo.addItems([
            "SQLite", "MySQL", "PostgreSQL", "SQL Server", "Oracle", "DuckDB"
        ])
        self.db_type_combo.currentTextChanged.connect(self.update_db_connection_fields)
        db_type_layout.addRow("Database Type:", self.db_type_combo)
        db_type_group.setLayout(db_type_layout)
        db_layout.addWidget(db_type_group)
        
        # Connection Settings
        self.connection_group = QGroupBox("Connection Settings")
        self.connection_layout = QFormLayout()
        self.connection_group.setLayout(self.connection_layout)
        db_layout.addWidget(self.connection_group)
        
        # Query Settings
        query_group = QGroupBox("Query Settings")
        query_layout = QFormLayout()
        self.query_edit = QTextEdit()
        self.query_edit.setPlaceholderText("Enter SQL query...")
        query_layout.addRow("SQL Query:", self.query_edit)
        query_group.setLayout(query_layout)
        db_layout.addWidget(query_group)
        
        db_tab.setLayout(db_layout)
        self.tab_widget.addTab(db_tab, "Database Import")
        
        layout.addWidget(self.tab_widget)
        
        # Import Button
        self.import_button = QPushButton("Import Data")
        self.import_button.clicked.connect(self.import_data)
        self.import_button.setEnabled(False)
        layout.addWidget(self.import_button)

        # Data Preview Table
        preview_group = QGroupBox("Data Preview")
        preview_layout = QVBoxLayout()
        
        # Create table widget
        self.preview_table = QTableWidget()
        self.preview_table.setColumnCount(0)
        self.preview_table.setRowCount(0)
        self.preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.preview_table.setAlternatingRowColors(True)
        preview_layout.addWidget(self.preview_table)
        
        # Add row count label
        self.row_count_label = QLabel("Rows: 0")
        preview_layout.addWidget(self.row_count_label)
        
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        # Log/Status Display
        layout.addWidget(QLabel("Import Log:"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        
        # Status label
        self.status_label = QLabel("Import tab ready. Select a data source.")
        layout.addWidget(self.status_label)
        
        # Subscribe to message bus
        self.message_bus.subscribe("Import", self.handle_message)
        self.log_message("Import tab initialized.")
        
        # Initialize database connection fields
        self.update_db_connection_fields()

    def update_db_connection_fields(self):
        """Update database connection fields based on selected database type."""
        # Clear existing fields
        while self.connection_layout.rowCount() > 0:
            self.connection_layout.removeRow(0)
            
        db_type = self.db_type_combo.currentText()
        
        if db_type in ["SQLite", "DuckDB"]:
            self.connection_layout.addRow("Database File:", QLineEdit())
        else:
            self.connection_layout.addRow("Host:", QLineEdit())
            self.connection_layout.addRow("Port:", QLineEdit())
            self.connection_layout.addRow("Database:", QLineEdit())
            self.connection_layout.addRow("Username:", QLineEdit())
            self.connection_layout.addRow("Password:", QLineEdit())

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
            self.file_path_edit.setText(path)
            self.import_button.setEnabled(True)
            self.status_label.setText("File selected. Ready to import.")
            self.log_message(f"Selected file: {path}")
        else:
            self.selected_file_path = None
            self.file_path_edit.clear()
            self.import_button.setEnabled(False)
            self.status_label.setText("File selection cancelled.")

    def import_data(self):
        """Import data from the selected source."""
        try:
            if self.tab_widget.currentIndex() == 0:  # File Import
                self.import_from_file()
            else:  # Database Import
                self.import_from_database()
        except Exception as e:
            error_msg = f"Error during import: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            self.log_message(error_msg)
            self.message_bus.publish("Import", "error", error_msg)
        finally:
            self.log_message("=== Import Finished ===")

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