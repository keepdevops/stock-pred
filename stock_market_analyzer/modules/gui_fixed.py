from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QListWidget, QCheckBox, QMessageBox
)
from PyQt6.QtCore import Qt, QDateTime
import pandas as pd
import logging
import traceback

class StockGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.setup_gui()
        
    def setup_gui(self):
        """Set up the main GUI components."""
        try:
            self.logger.info("Setting up GUI components")
            
            # Set up the main window
            self.setWindowTitle("Stock Market Analyzer")
            self.setGeometry(100, 100, 1200, 800)
            
            # Create central widget and main layout
            self.central_widget = QWidget()
            self.setCentralWidget(self.central_widget)
            self.main_layout = QVBoxLayout(self.central_widget)
            
            # Create tab widget
            self.tab_widget = QTabWidget()
            self.main_layout.addWidget(self.tab_widget)
            
            # Create and add tabs
            self.tab_widget.addTab(self.create_data_tab(), "Data")
            
            self.logger.info("GUI setup completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error setting up GUI: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
            
    def create_data_tab(self):
        """Create the data tab with input controls and data display."""
        try:
            self.logger.info("Creating data tab")
            
            # Create data tab widget
            self.data_tab = QWidget()
            data_layout = QVBoxLayout()
            self.data_tab.setLayout(data_layout)
            
            # Create input controls
            input_layout = QHBoxLayout()
            
            # Ticker input
            self.ticker_input = QLineEdit()
            self.ticker_input.setPlaceholderText("Enter ticker symbol")
            input_layout.addWidget(self.ticker_input)
            
            # Add ticker button
            add_ticker_btn = QPushButton("Add Ticker")
            add_ticker_btn.clicked.connect(self.add_ticker)
            input_layout.addWidget(add_ticker_btn)
            
            # Remove ticker button
            remove_ticker_btn = QPushButton("Remove Ticker")
            remove_ticker_btn.clicked.connect(self.remove_ticker)
            input_layout.addWidget(remove_ticker_btn)
            
            data_layout.addLayout(input_layout)
            
            # Create ticker list
            self.ticker_list = QListWidget()
            self.ticker_list.itemSelectionChanged.connect(self.on_ticker_selected)
            data_layout.addWidget(self.ticker_list)
            
            # Create data table
            self.data_table = QTableWidget()
            self.data_table.setColumnCount(7)
            self.data_table.setHorizontalHeaderLabels([
                "Date", "Open", "High", "Low", "Close", "Volume", "Adj Close"
            ])
            self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            self.data_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            data_layout.addWidget(self.data_table)
            
            # Add realtime checkbox
            self.realtime_checkbox = QCheckBox("Enable Realtime Updates")
            self.realtime_checkbox.stateChanged.connect(self.toggle_realtime)
            data_layout.addWidget(self.realtime_checkbox)
            
            self.logger.info("Data tab created successfully")
            return self.data_tab
            
        except Exception as e:
            self.logger.error(f"Error creating data tab: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
            
    def add_ticker(self):
        """Add a new ticker to the list."""
        try:
            ticker = self.ticker_input.text().strip().upper()
            if not ticker:
                self.show_status_message("Please enter a ticker symbol", "warning")
                return
            
            # Check if ticker already exists
            items = self.ticker_list.findItems(ticker, Qt.MatchFlag.MatchExactly)
            if items:
                self.show_status_message(f"Ticker {ticker} already exists", "warning")
                return
                
            # Add ticker to list
            self.ticker_list.addItem(ticker)
            self.ticker_input.clear()
            self.show_status_message(f"Added ticker: {ticker}", "info")
            
        except Exception as e:
            self.logger.error(f"Error adding ticker: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            self.show_status_message(f"Error adding ticker: {str(e)}", "error")
            
    def remove_ticker(self):
        """Remove selected ticker from the list."""
        try:
            selected_items = self.ticker_list.selectedItems()
            if not selected_items:
                QMessageBox.warning(self, "Selection Error", "Please select a ticker to remove")
                return
                
            for item in selected_items:
                ticker = item.text()
                self.ticker_list.takeItem(self.ticker_list.row(item))
                self.logger.info(f"Removed ticker: {ticker}")
                
            # Clear data table if no tickers left
            if self.ticker_list.count() == 0:
                self.data_table.setRowCount(0)
                
        except Exception as e:
            self.logger.error(f"Error removing ticker: {e}")
            QMessageBox.critical(self, "Error", f"Failed to remove ticker: {str(e)}")
            
    def on_ticker_selected(self):
        """Handle ticker selection."""
        try:
            selected_items = self.ticker_list.selectedItems()
            if not selected_items:
                return
                
            ticker = selected_items[0].text()
            self.logger.info(f"Selected ticker: {ticker}")
            
        except Exception as e:
            self.logger.error(f"Error handling ticker selection: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            self.show_status_message(f"Error handling ticker selection: {str(e)}", "error")
            
    def toggle_realtime(self, state):
        """Toggle realtime updates."""
        try:
            if state == Qt.CheckState.Checked.value:
                self.logger.info("Realtime updates enabled")
            else:
                self.logger.info("Realtime updates disabled")
                
        except Exception as e:
            self.logger.error(f"Error toggling realtime updates: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
    def show_status_message(self, message, message_type="info"):
        """Show a status message to the user."""
        try:
            if message_type == "error":
                QMessageBox.critical(self, "Error", message)
            elif message_type == "warning":
                QMessageBox.warning(self, "Warning", message)
            else:
                QMessageBox.information(self, "Information", message)
                
        except Exception as e:
            self.logger.error(f"Error showing status message: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}") 