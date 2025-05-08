import sys
import os
import logging
import traceback
import time
import uuid
import pandas as pd
import json
from typing import Dict, Any, List, Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QComboBox, QPushButton, QLabel, QFileDialog, QMessageBox, QListWidget,
    QListWidgetItem, QSplitter, QApplication, QSpinBox, QCheckBox,
    QGroupBox, QLineEdit, QFormLayout, QFrame, QHeaderView, QTabWidget, QScrollArea,
    QGridLayout
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QPalette, QColor
from .base_tab import BaseTab
from ..message_bus import MessageBus
from ..database import DatabaseConnector
from ..settings import Settings
from ..data_manager import DataManager

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class ImportTab(BaseTab):
    """Import tab for loading data from files or databases."""
    
    def __init__(self, message_bus: MessageBus, parent=None):
        """Initialize the Import tab.
        
        Args:
            message_bus: The message bus for communication.
            parent: The parent widget.
        """
        # Initialize attributes before parent __init__
        self.settings = Settings()
        self.current_color_scheme = self.settings.get_color_scheme()
        self._ui_setup_done = False
        self.source_combo = None
        self.file_type_combo = None
        self.import_button = None
        self.data_table = None
        self.database_fields = {}
        self.database_checkboxes = {}
        self.database_form = None
        self.use_database_check = None
        self.ticker_listbox = None
        self.select_all_button = None
        self.deselect_all_button = None
        self.cached_data = {}
        self.pending_requests = {}
        self.connection_status = {}
        self.connection_labels = {}
        self.messages_received = 0
        self.messages_sent = 0
        self.errors = 0
        self.message_latencies = []
        self.metrics_labels = {}
        self.database_connector = DatabaseConnector()
        self.data_manager = DataManager()
        
        # Call parent __init__ with message bus
        super().__init__(message_bus, parent)
        
        # Set dark theme
        self.set_dark_theme()
        
    def set_dark_theme(self):
        """Set dark theme for the tab."""
        dark_theme = """
        QWidget {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        QGroupBox {
            border: 1px solid #3c3c3c;
            border-radius: 5px;
            margin-top: 1ex;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px 0 3px;
            color: #ffffff;
        }
        QComboBox {
            background-color: #3c3c3c;
            border: 1px solid #3c3c3c;
            border-radius: 3px;
            padding: 5px;
            min-width: 6em;
        }
        QComboBox:hover {
            border: 1px solid #4c4c4c;
        }
        QComboBox::drop-down {
            border: none;
        }
        QComboBox::down-arrow {
            image: url(down_arrow.png);
            width: 12px;
            height: 12px;
        }
        QPushButton {
            background-color: #3c3c3c;
            border: 1px solid #3c3c3c;
            border-radius: 3px;
            padding: 5px 10px;
            color: #ffffff;
        }
        QPushButton:hover {
            background-color: #4c4c4c;
            border: 1px solid #4c4c4c;
        }
        QPushButton:pressed {
            background-color: #2c2c2c;
        }
        QLineEdit {
            background-color: #3c3c3c;
            border: 1px solid #3c3c3c;
            border-radius: 3px;
            padding: 5px;
            color: #ffffff;
        }
        QLineEdit:focus {
            border: 1px solid #4c4c4c;
        }
        QListWidget {
            background-color: #3c3c3c;
            border: 1px solid #3c3c3c;
            border-radius: 3px;
            color: #ffffff;
        }
        QListWidget::item {
            padding: 5px;
        }
        QListWidget::item:selected {
            background-color: #4c4c4c;
        }
        QTableWidget {
            background-color: #3c3c3c;
            border: 1px solid #3c3c3c;
            border-radius: 3px;
            color: #ffffff;
            gridline-color: #4c4c4c;
        }
        QTableWidget::item {
            padding: 5px;
        }
        QTableWidget::item:selected {
            background-color: #4c4c4c;
        }
        QHeaderView::section {
            background-color: #2b2b2b;
            color: #ffffff;
            padding: 5px;
            border: 1px solid #3c3c3c;
        }
        QCheckBox {
            color: #ffffff;
        }
        QCheckBox::indicator {
            width: 15px;
            height: 15px;
        }
        QCheckBox::indicator:unchecked {
            background-color: #3c3c3c;
            border: 1px solid #4c4c4c;
        }
        QCheckBox::indicator:checked {
            background-color: #4c4c4c;
            border: 1px solid #4c4c4c;
        }
        QLabel {
            color: #ffffff;
        }
        QLabel[status="error"] {
            color: #ff4444;
        }
        QLabel[status="success"] {
            color: #44ff44;
        }
        """
        self.setStyleSheet(dark_theme)
        
        # Set dark palette
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(43, 43, 43))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Base, QColor(60, 60, 60))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(43, 43, 43))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(43, 43, 43))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Button, QColor(60, 60, 60))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
        self.setPalette(palette)
        
    def _setup_message_bus_impl(self):
        """Setup message bus subscriptions."""
        try:
            self.message_bus.subscribe("Import", self.handle_import_message)
            self.message_bus.subscribe("ConnectionStatus", self.handle_connection_status)
            self.message_bus.subscribe("Heartbeat", self.handle_heartbeat)
            self.message_bus.subscribe("Shutdown", self.handle_shutdown)
            
            # Publish initial connection status
            self.message_bus.publish(
                "ConnectionStatus",
                "status_update",
                {
                    "status": "connected",
                    "sender": "ImportTab",
                    "timestamp": time.time()
                }
            )
            
            self.logger.debug("Subscribed to Import, ConnectionStatus, Heartbeat, and Shutdown topics")
        except Exception as e:
            self.handle_error("Error setting up message bus subscriptions", e)
            
    def setup_ui(self):
        """Setup the UI components."""
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
            
            # Create source selection group
            source_group = QGroupBox("Data Source")
            source_layout = QHBoxLayout()
            
            # Source type selection
            source_layout.addWidget(QLabel("Source:"))
            self.source_combo = QComboBox()
            self.source_combo.addItems(["File", "Database"])
            self.source_combo.currentTextChanged.connect(self.on_source_changed)
            source_layout.addWidget(self.source_combo)
            
            # File type selection (only visible for file source)
            self.file_type_combo = QComboBox()
            self.file_type_combo.addItems(["CSV", "JSON"])
            source_layout.addWidget(self.file_type_combo)
            
            source_group.setLayout(source_layout)
            self.main_layout.addWidget(source_group)
            
            # Create database connection form
            self.database_form = QGroupBox("Database Connection")
            db_layout = QFormLayout()
            
            # Database fields
            self.database_fields = {
                "host": QLineEdit(),
                "port": QLineEdit(),
                "database": QLineEdit(),
                "username": QLineEdit(),
                "password": QLineEdit(),
                "table": QLineEdit()
            }
            
            # Create checkboxes for each field
            for field_name, field_widget in self.database_fields.items():
                field_layout = QHBoxLayout()
                checkbox = QCheckBox()
                checkbox.setChecked(True)
                checkbox.stateChanged.connect(lambda state, w=field_widget: self.toggle_database_field(w, state))
                self.database_checkboxes[field_name] = checkbox
                field_layout.addWidget(checkbox)
                field_layout.addWidget(field_widget)
                db_layout.addRow(field_name.capitalize() + ":", field_layout)
            
            self.database_form.setLayout(db_layout)
            self.main_layout.addWidget(self.database_form)
            
            # Create import button
            button_layout = QHBoxLayout()
            self.import_button = QPushButton("Import Data")
            self.import_button.clicked.connect(self.import_data)
            button_layout.addWidget(self.import_button)
            self.main_layout.addLayout(button_layout)
            
            # Create metrics group
            metrics_group = QGroupBox("Metrics")
            metrics_layout = QGridLayout()
            
            # Create metrics labels
            metrics = ["Messages Received", "Messages Sent", "Errors", "Average Latency"]
            for i, metric in enumerate(metrics):
                label = QLabel(f"{metric}: 0")
                self.metrics_labels[metric] = label
                metrics_layout.addWidget(label, i // 2, i % 2)
                
            metrics_group.setLayout(metrics_layout)
            self.main_layout.addWidget(metrics_group)
            
            # Create connection status group
            connection_group = QGroupBox("Connection Status")
            connection_layout = QVBoxLayout()
            
            # Create connection status labels
            tabs = ["Analysis", "Charts", "Models", "Predictions"]
            for tab in tabs:
                label = QLabel(f"{tab}: Not Connected")
                label.setStyleSheet("color: red")
                self.connection_labels[tab] = label
                connection_layout.addWidget(label)
                
            connection_group.setLayout(connection_layout)
            self.main_layout.addWidget(connection_group)
            
            # Create ticker selection group
            ticker_group = QGroupBox("Ticker Selection")
            ticker_layout = QVBoxLayout()
            
            # Ticker listbox with custom items
            self.ticker_listbox = QListWidget()
            self.ticker_listbox.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
            self.ticker_listbox.itemSelectionChanged.connect(self.on_ticker_selection_changed)
            ticker_layout.addWidget(self.ticker_listbox)
            
            # Selection buttons
            button_layout = QHBoxLayout()
            self.select_all_button = QPushButton("Select All")
            self.select_all_button.clicked.connect(self.select_all_tickers)
            self.deselect_all_button = QPushButton("Deselect All")
            self.deselect_all_button.clicked.connect(self.deselect_all_tickers)
            button_layout.addWidget(self.select_all_button)
            button_layout.addWidget(self.deselect_all_button)
            ticker_layout.addLayout(button_layout)
            
            ticker_group.setLayout(ticker_layout)
            self.main_layout.addWidget(ticker_group)
            
            # Create data preview table
            table_group = QGroupBox("Data Preview")
            table_layout = QVBoxLayout()
            self.data_table = QTableWidget()
            self.data_table.setColumnCount(0)
            table_layout.addWidget(self.data_table)
            table_group.setLayout(table_layout)
            self.main_layout.addWidget(table_group)
            
            # Add status bar
            self.status_label = QLabel("Ready")
            self.status_label.setStyleSheet("color: green")
            self.main_layout.addWidget(self.status_label)
            
            self._ui_setup_done = True
            self.logger.info("Import tab initialized")
            
        except Exception as e:
            self.handle_error("Error setting up UI", e)
            
    def on_source_changed(self, source: str):
        """Handle source type change."""
        try:
            self.database_form.setVisible(source == "Database")
            self.file_type_combo.setVisible(source == "File")
        except Exception as e:
            self.handle_error("Error handling source change", e)
            
    def toggle_database_field(self, widget: QLineEdit, state: int):
        """Toggle database field visibility."""
        try:
            widget.setEnabled(state == Qt.CheckState.Checked.value)
        except Exception as e:
            self.handle_error("Error toggling database field", e)
            
    def import_data(self):
        """Import data from selected source."""
        try:
            source = self.source_combo.currentText()
            if source == "File":
                self.import_file()
            elif source == "Database":
                self.import_database()
        except Exception as e:
            self.handle_error("Error importing data", e)
            
    def select_all_tickers(self):
        """Select all tickers in the list."""
        try:
            self.ticker_listbox.selectAll()
        except Exception as e:
            self.handle_error("Error selecting all tickers", e)
            
    def deselect_all_tickers(self):
        """Deselect all tickers in the list."""
        try:
            self.ticker_listbox.clearSelection()
        except Exception as e:
            self.handle_error("Error deselecting all tickers", e)
            
    def update_ticker_list(self, tickers: List[str]):
        """Update the ticker list."""
        try:
            self.ticker_listbox.clear()
            self.ticker_listbox.addItems(tickers)
        except Exception as e:
            self.handle_error("Error updating ticker list", e)
            
    def get_selected_tickers(self) -> List[str]:
        """Get list of selected tickers."""
        try:
            return [item.text() for item in self.ticker_listbox.selectedItems()]
        except Exception as e:
            self.handle_error("Error getting selected tickers", e)
            return []
            
    def cache_data(self, symbol: str, data_dict: Dict[str, Any]):
        """Cache imported data."""
        try:
            self.cached_data[symbol] = data_dict
        except Exception as e:
            self.handle_error("Error caching data", e)
            
    def update_connection_status(self, tab: str, connected: bool):
        """Update connection status for a tab."""
        try:
            if tab in self.connection_labels:
                label = self.connection_labels[tab]
                if connected:
                    label.setText(f"{tab}: Connected")
                    label.setStyleSheet("color: green")
                else:
                    label.setText(f"{tab}: Not Connected")
                    label.setStyleSheet("color: red")
        except Exception as e:
            self.handle_error("Error updating connection status", e)
            
    def handle_import_message(self, sender: str, message_type: str, data: Dict[str, Any]):
        """Handle import-related messages."""
        try:
            if not isinstance(data, dict):
                self.logger.error("Invalid import message data format")
                return
                
            self.messages_received += 1
            
            if message_type == "import_complete":
                df_data = data.get("data")
                if df_data is None:
                    self.logger.error("Missing data in import complete message")
                    return
                    
                try:
                    df = pd.DataFrame(df_data)
                    self.update_preview_table(df)
                    self.status_label.setText("Import completed successfully")
                    self.status_label.setStyleSheet("color: green")
                except Exception as e:
                    self.handle_error("Error processing imported data", e)
                    
            elif message_type == "import_error":
                error_msg = data.get("error", "Unknown error")
                self.handle_error("Import error", Exception(error_msg))
                
            # Update metrics after processing message
            self.update_metrics()
            
        except Exception as e:
            self.handle_error("Error handling import message", e)
            
    def handle_connection_status(self, sender: str, message_type: str, data: Dict[str, Any]):
        """Handle connection status messages."""
        try:
            if not isinstance(data, dict):
                self.logger.error("Invalid connection status data format")
                return
                
            status = data.get("status")
            if status not in ["connected", "disconnected", "error"]:
                self.logger.error(f"Invalid connection status: {status}")
                return
                
            self.connection_status[sender] = status
            self.update_connection_status(sender, status == "connected")
            
            # Update metrics
            self.update_metrics()
            
        except Exception as e:
            self.handle_error("Error handling connection status", e)
            
    def handle_heartbeat(self, sender: str, message_type: str, data: Dict[str, Any]):
        """Handle heartbeat messages."""
        try:
            self.messages_received += 1
            self.message_latencies.append(time.time() - data.get("timestamp", time.time()))
            self.update_metrics()
        except Exception as e:
            self.handle_error("Error handling heartbeat", e)
            
    def handle_shutdown(self, sender: str, message_type: str, data: Dict[str, Any]):
        """Handle shutdown messages."""
        try:
            self._is_shutting_down = True
            self.cleanup()
        except Exception as e:
            self.handle_error("Error handling shutdown", e)
            
    def update_metrics(self):
        """Update the metrics display."""
        try:
            if not hasattr(self, 'metrics_labels'):
                return
                
            # Update message counts
            self.metrics_labels["Messages Received"].setText(f"Messages Received: {self.messages_received}")
            self.metrics_labels["Messages Sent"].setText(f"Messages Sent: {self.messages_sent}")
            self.metrics_labels["Errors"].setText(f"Errors: {self.errors}")
            
            # Calculate and update average latency
            if self.message_latencies:
                avg_latency = sum(self.message_latencies) / len(self.message_latencies)
                self.metrics_labels["Average Latency"].setText(f"Average Latency: {avg_latency:.2f}s")
            else:
                self.metrics_labels["Average Latency"].setText("Average Latency: N/A")
                
        except Exception as e:
            self.handle_error("Error updating metrics", e)
            
    def import_file(self):
        """Import data from file."""
        try:
            file_type = self.file_type_combo.currentText()
            
            # Create file dialog with proper macOS handling
            dialog = QFileDialog(self)
            dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
            dialog.setNameFilter(f"{file_type} Files (*.{file_type.lower()})")
            dialog.setWindowTitle(f"Select {file_type} File")
            
            # Set options to avoid macOS-specific issues
            dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
            
            if dialog.exec():
                file_path = dialog.selectedFiles()[0]
                
                if file_path:
                    try:
                        if file_type == "CSV":
                            df = pd.read_csv(file_path)
                        else:  # JSON
                            df = pd.read_json(file_path)
                            
                        self.update_preview_table(df)
                        self.publish_data(df)
                        
                    except pd.errors.EmptyDataError:
                        QMessageBox.warning(self, "Error", "The selected file is empty.")
                    except pd.errors.ParserError:
                        QMessageBox.warning(self, "Error", "Failed to parse the selected file. Please check the file format.")
                    except Exception as e:
                        QMessageBox.warning(self, "Error", f"Error reading file: {str(e)}")
                        
        except Exception as e:
            self.handle_error("Error importing file", e)
            
    def import_database(self):
        """Import data from database."""
        try:
            # Get database connection parameters
            params = {
                field: widget.text()
                for field, widget in self.database_fields.items()
                if self.database_checkboxes[field].isChecked()
            }
            
            if not params:
                self.status_label.setText("Please select at least one database parameter")
                return
                
            # Connect to database
            self.database_connector.connect(**params)
            
            # Get data
            df = self.database_connector.get_data()
            
            # Update UI
            self.update_preview_table(df)
            self.publish_data(df)
            
        except Exception as e:
            self.handle_error("Error importing from database", e)
            
    def update_preview_table(self, df: pd.DataFrame):
        """Update the preview table with imported data."""
        try:
            if df.empty:
                self.logger.warning("Empty dataframe received for preview")
                return
                
            # Clear existing table
            self.data_table.setRowCount(0)
            self.data_table.setColumnCount(0)
            
            # Set up table dimensions
            self.data_table.setRowCount(len(df))
            self.data_table.setColumnCount(len(df.columns))
            
            # Set column headers
            self.data_table.setHorizontalHeaderLabels(df.columns)
            
            # Populate table
            for i, row in df.iterrows():
                for j, value in enumerate(row):
                    item = QTableWidgetItem(str(value))
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.data_table.setItem(i, j, item)
                    
            # Resize columns to fit content
            self.data_table.resizeColumnsToContents()
            
            # Update status
            self.status_label.setText(f"Preview updated with {len(df)} rows")
            self.status_label.setStyleSheet("color: green")
            
        except Exception as e:
            self.handle_error("Error updating preview table", e)
            
    def on_ticker_selection_changed(self):
        """Handle ticker selection change."""
        try:
            selected_tickers = self.get_selected_tickers()
            self.status_label.setText(f"Selected {len(selected_tickers)} tickers")
        except Exception as e:
            self.handle_error("Error handling ticker selection", e)
            
    def publish_data(self, df: pd.DataFrame):
        """Publish imported data."""
        try:
            if df.empty:
                self.logger.warning("Attempting to publish empty dataframe")
                return
                
            # Extract symbols from the dataframe
            symbol_column = None
            for col in df.columns:
                if col.lower() in ['symbol', 'ticker', 'stock_symbol']:
                    symbol_column = col
                    break
                    
            if symbol_column:
                # Get unique symbols from the dataframe
                symbols = df[symbol_column].unique().tolist()
                # Update the ticker list with the symbols
                self.update_ticker_list(symbols)
                self.logger.info(f"Updated ticker list with {len(symbols)} symbols")
            else:
                self.logger.warning("No symbol column found in imported data")
                
            # Create metadata
            metadata = {
                "timestamp": time.time(),
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
                "symbols": symbols if symbol_column else [],
                "source": "ImportTab",
                "data_type": "market_data"
            }
            
            # Add data to DataManager
            self.data_manager.add_data("ImportTab", df, metadata)
            
            # Update status with more detailed information
            status_text = f"Published {len(df)} rows of data"
            if symbols:
                status_text += f" for {len(symbols)} symbols"
            self.status_label.setText(status_text)
            self.status_label.setStyleSheet("color: green")
            
        except Exception as e:
            self.handle_error("Error publishing data", e)
            
    def cleanup(self):
        """Clean up resources."""
        try:
            # Unsubscribe from message bus topics
            self.message_bus.unsubscribe("Import", self.handle_import_message)
            self.message_bus.unsubscribe("ConnectionStatus", self.handle_connection_status)
            self.message_bus.unsubscribe("Heartbeat", self.handle_heartbeat)
            self.message_bus.unsubscribe("Shutdown", self.handle_shutdown)
            
            # Clear caches
            self.cached_data.clear()
            self.pending_requests.clear()
            self.connection_status.clear()
            
            # Reset metrics
            self.messages_received = 0
            self.messages_sent = 0
            self.errors = 0
            self.message_latencies.clear()
            
            # Clear UI components
            if self.data_table:
                self.data_table.setRowCount(0)
                self.data_table.setColumnCount(0)
                
            if self.ticker_listbox:
                self.ticker_listbox.clear()
                
            # Reset status
            self.status_label.setText("Ready")
            self.status_label.setStyleSheet("color: green")
            
            # Update metrics display
            self.update_metrics()
            
            # Clean up database connection
            if self.database_connector:
                self.database_connector.disconnect()
                
            # Call parent cleanup
            super().cleanup()
            
            self.logger.info("Import tab cleanup completed")
            
        except Exception as e:
            self.handle_error("Error during cleanup", e)
            
    def handle_error(self, message: str, error: Exception):
        """Handle errors with proper logging and UI updates."""
        try:
            # Log error
            self.logger.error(f"{message}: {str(error)}")
            self.logger.debug(traceback.format_exc())
            
            # Update error count
            self.errors += 1
            
            # Update status
            self.status_label.setText(f"Error: {message}")
            self.status_label.setStyleSheet("color: red")
            
            # Update metrics
            self.update_metrics()
            
            # Show error message to user
            QMessageBox.critical(self, "Error", f"{message}\n{str(error)}")
            
        except Exception as e:
            # If error handling fails, log it and continue
            self.logger.error(f"Error in handle_error: {str(e)}")
            self.logger.debug(traceback.format_exc())

def main():
    """Main function for the import tab process."""
    app = QApplication(sys.argv)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting import tab process")
    
    # Create and show the import tab
    window = ImportTab()
    window.setWindowTitle("Import Tab")
    window.show()
    
    # Start the application event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 