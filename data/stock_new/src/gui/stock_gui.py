import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QProgressBar, 
                            QTableWidget, QTableWidgetItem, QTabWidget, 
                            QComboBox, QDateEdit, QMessageBox, QFileDialog, 
                            QCalendarWidget, QGroupBox, QRadioButton, 
                           QButtonGroup, QGridLayout, QDialog, QCheckBox, 
                           QDialogButtonBox, QTextEdit, QListWidget, QFrame,
                           QSplitter, QLineEdit, QListWidgetItem, QScrollArea,
                           QMenu, QHeaderView)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QDate, QSize, QTimer
from PyQt5.QtGui import QFont, QColor
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import polars as pl
import json
import yfinance as yf
import requests
import time
import shutil
from pathlib import Path
from typing import List, Dict, Set
from PyQt5.QtCore import QSettings
import gc

# Import your TickerManager
from src.data.ticker_manager import TickerManager
from src.utils.color_scheme import ColorScheme
from .date_dialog import DateRangeDialog
from .ticker_detail_dialog import TickerDetailDialog
from .color_scheme_dialog import ColorSchemeDialog
from .color_schemes import ColorSchemeManager

# Add this new class for detailed progress tracking
class TickerProgressWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Overall progress
        progress_group = QGroupBox("Overall Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.total_progress = QProgressBar()
        self.total_progress.setTextVisible(True)
        progress_layout.addWidget(self.total_progress)
        
        self.status_label = QLabel("Ready")
        progress_layout.addWidget(self.status_label)
        
        layout.addWidget(progress_group)
        
        # Current ticker progress
        current_group = QGroupBox("Current Ticker")
        current_layout = QVBoxLayout(current_group)
        
        self.current_ticker_label = QLabel("No ticker processing")
        current_layout.addWidget(self.current_ticker_label)
        
        self.current_progress = QProgressBar()
        self.current_progress.setTextVisible(True)
        current_layout.addWidget(self.current_progress)
        
        layout.addWidget(current_group)
        
        # Processing log
        log_group = QGroupBox("Processing Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_widget = QTableWidget()
        self.log_widget.setColumnCount(4)
        self.log_widget.setHorizontalHeaderLabels(["Ticker", "Status", "Time", "Message"])
        self.log_widget.horizontalHeader().setStretchLastSection(True)
        log_layout.addWidget(self.log_widget)
        
        layout.addWidget(log_group)

    def update_total_progress(self, value: int, message: str):
        self.total_progress.setValue(value)
        self.status_label.setText(message)

    def update_current_progress(self, ticker: str, value: int, message: str):
        self.current_ticker_label.setText(f"Processing: {ticker}")
        self.current_progress.setValue(value)

    def add_log_entry(self, ticker: str, status: str, message: str):
        row = self.log_widget.rowCount()
        self.log_widget.insertRow(row)
        
        time_str = datetime.now().strftime("%H:%M:%S")
        
        self.log_widget.setItem(row, 0, QTableWidgetItem(ticker))
        self.log_widget.setItem(row, 1, QTableWidgetItem(status))
        self.log_widget.setItem(row, 2, QTableWidgetItem(time_str))
        self.log_widget.setItem(row, 3, QTableWidgetItem(message))
        
        # Scroll to bottom
        self.log_widget.scrollToBottom()

    def clear_log(self):
        self.log_widget.setRowCount(0)
        self.current_ticker_label.setText("No ticker processing")
        self.current_progress.setValue(0)
        self.total_progress.setValue(0)
        self.status_label.setText("Ready")

# Update the DataFetchWorker to provide more detailed progress
class DataFetchWorker(QThread):
    progress = pyqtSignal(int, str)
    result = pyqtSignal(dict)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, ticker_manager, ticker, start_date, end_date):
        super().__init__()
        self.ticker_manager = ticker_manager
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.is_running = True

    def run(self):
        try:
            self.progress.emit(25, f"Fetching {self.ticker}")
            
            # Get data
            data = self.ticker_manager.get_single_ticker_data(
                self.ticker, 
                self.start_date, 
                self.end_date
            )
            
            if data is not None:
                self.progress.emit(75, "Processing data")
                
                # Calculate percentage change
                start_price = data['start_price']
                last_price = data['last_price']
                pct_change = ((last_price - start_price) / start_price) * 100
                
                # Add to data
                data['change_pct'] = pct_change
                
                self.result.emit(data)
                self.progress.emit(100, "Complete")
            else:
                self.error.emit(f"No data available for {self.ticker}")
            
        except Exception as e:
            self.error.emit(f"Error: {str(e)}")
            logging.error(f"Worker error: {e}")
        finally:
            self.finished.emit()

    def stop(self):
        self.is_running = False

class MultiTickerSelector(QWidget):
    selection_changed = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_tickers: Set[str] = set()
        self.setup_ui()
        self.apply_styles()

    def apply_styles(self):
        """Apply custom styles to the widget."""
        self.setStyleSheet("""
            QListWidget {
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 2px;
            }
            QListWidget::item {
                padding: 4px;
                border-bottom: 1px solid #eee;
            }
            QListWidget::item:selected {
                background-color: #e6f3ff;
                color: black;
            }
            QLineEdit {
                padding: 6px;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
            QComboBox {
                padding: 6px;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
            QPushButton {
                padding: 6px 12px;
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: #f8f9fa;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
            QLabel {
                color: #666;
            }
        """)

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # Search and filter
        filter_layout = QHBoxLayout()
        
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Search tickers...")
        self.filter_input.textChanged.connect(self.filter_tickers)
        filter_layout.addWidget(self.filter_input)

        self.category_combo = QComboBox()
        self.category_combo.setMinimumWidth(120)
        self.category_combo.currentTextChanged.connect(self.on_category_changed)
        filter_layout.addWidget(self.category_combo)

        layout.addLayout(filter_layout)

        # Ticker list with checkboxes
        self.ticker_list = QListWidget()
        self.ticker_list.setSelectionMode(QListWidget.MultiSelection)
        self.ticker_list.itemChanged.connect(self.on_item_changed)
        self.ticker_list.setMinimumHeight(400)
        layout.addWidget(self.ticker_list)

        # Selected tickers count
        self.count_label = QLabel("Selected: 0 tickers")
        self.count_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.count_label)

        # Quick selection buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all)
        clear_all_btn = QPushButton("Clear All")
        clear_all_btn.clicked.connect(self.clear_all)
        
        button_layout.addWidget(select_all_btn)
        button_layout.addWidget(clear_all_btn)
        layout.addLayout(button_layout)

    def populate_categories(self, categories: Dict[str, List[str]]):
        """Populate category dropdown and ticker lists."""
        self.categories = categories
        self.category_combo.clear()
        self.category_combo.addItems(categories.keys())

    def on_category_changed(self, category: str):
        """Handle category selection change."""
        self.ticker_list.clear()
        if category in self.categories:
            for ticker in self.categories[category]:
                item = QListWidgetItem(ticker)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked if ticker in self.selected_tickers else Qt.Unchecked)
                self.ticker_list.addItem(item)

    def filter_tickers(self, text: str):
        """Filter ticker list based on search text."""
        category = self.category_combo.currentText()
        self.ticker_list.clear()
        if category in self.categories:
            for ticker in self.categories[category]:
                if text.lower() in ticker.lower():
                    item = QListWidgetItem(ticker)
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(Qt.Checked if ticker in self.selected_tickers else Qt.Unchecked)
                    self.ticker_list.addItem(item)

    def on_item_changed(self, item: QListWidgetItem):
        """Handle ticker selection change."""
        ticker = item.text()
        if item.checkState() == Qt.Checked:
            self.selected_tickers.add(ticker)
        else:
            self.selected_tickers.discard(ticker)
        
        self.count_label.setText(f"Selected: {len(self.selected_tickers)} tickers")
        self.selection_changed.emit(list(self.selected_tickers))

    def select_all(self):
        """Select all visible tickers."""
        for i in range(self.ticker_list.count()):
            item = self.ticker_list.item(i)
            item.setCheckState(Qt.Checked)

    def clear_all(self):
        """Clear all ticker selections."""
        self.selected_tickers.clear()
        for i in range(self.ticker_list.count()):
            item = self.ticker_list.item(i)
            item.setCheckState(Qt.Unchecked)
        self.count_label.setText("Selected: 0 tickers")
        self.selection_changed.emit([])

class StockGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ticker_manager = TickerManager()
        self.selected_tickers = set()
        self.init_date_ranges()  # Initialize date ranges first
        self.setup_ui()  # Then setup UI

    def init_date_ranges(self):
        """Initialize date range selectors."""
        # Create calendar widgets
        self.start_calendar = QCalendarWidget()
        self.end_calendar = QCalendarWidget()
        
        # Set maximum date to today to prevent future date selection
        today = QDate.currentDate()
        self.start_calendar.setMaximumDate(today)
        self.end_calendar.setMaximumDate(today)
        
        # Set default date range (last 30 days)
        self.set_default_dates()
        
        # Connect date change signals
        self.start_calendar.selectionChanged.connect(
            lambda: self.on_start_date_changed(self.start_calendar.selectedDate())
        )
        self.end_calendar.selectionChanged.connect(
            lambda: self.on_end_date_changed(self.end_calendar.selectedDate())
        )

    def setup_ui(self):
        # Main widget and layout
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Left panel for ticker selection
        left_panel = QFrame()
        left_layout = QVBoxLayout(left_panel)
        
        # Category selector
        category_label = QLabel("Category:")
        self.category_combo = QComboBox()
        self.category_combo.addItems(self.ticker_manager.get_category_names())
        self.category_combo.currentTextChanged.connect(self.on_category_changed)
        
        left_layout.addWidget(category_label)
        left_layout.addWidget(self.category_combo)

        # Search box
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search tickers...")
        self.search_box.textChanged.connect(self.filter_tickers)
        left_layout.addWidget(self.search_box)

        # Ticker list
        self.ticker_list = QListWidget()
        self.ticker_list.itemChanged.connect(self.on_ticker_selection_changed)
        left_layout.addWidget(self.ticker_list)

        # Selection buttons
        button_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        clear_all_btn = QPushButton("Clear All")
        select_all_btn.clicked.connect(self.select_all_tickers)
        clear_all_btn.clicked.connect(self.clear_all_tickers)
        button_layout.addWidget(select_all_btn)
        button_layout.addWidget(clear_all_btn)
        left_layout.addLayout(button_layout)

        # Selected count
        self.ticker_count_label = QLabel("Selected: 0")
        left_layout.addWidget(self.ticker_count_label)

        main_layout.addWidget(left_panel)

        # Right panel
        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)

        # Date Selection Group
        date_group = QGroupBox("Date Selection")
        date_layout = QVBoxLayout(date_group)

        # Add date display labels
        date_display_layout = QHBoxLayout()
        self.start_date_label = QLabel("Start: ")
        self.end_date_label = QLabel("End: ")
        date_display_layout.addWidget(self.start_date_label)
        date_display_layout.addWidget(self.end_date_label)
        date_layout.addLayout(date_display_layout)

        # Quick Period Selection
        period_layout = QHBoxLayout()
        period_label = QLabel("Quick Select:")
        self.period_combo = QComboBox()
        self.period_combo.addItems(self.time_periods.keys())
        self.period_combo.currentTextChanged.connect(self.on_period_changed)
        period_layout.addWidget(period_label)
        period_layout.addWidget(self.period_combo)
        date_layout.addLayout(period_layout)

        # Calendar Layout
        calendar_layout = QHBoxLayout()
        
        # Start Date
        start_layout = QVBoxLayout()
        start_label = QLabel("Start Date:")
        self.start_calendar = QCalendarWidget()
        self.start_calendar.clicked.connect(self.on_start_date_changed)
        start_layout.addWidget(start_label)
        start_layout.addWidget(self.start_calendar)
        
        # End Date
        end_layout = QVBoxLayout()
        end_label = QLabel("End Date:")
        self.end_calendar = QCalendarWidget()
        self.end_calendar.setMaximumDate(QDate.currentDate())
        self.end_calendar.clicked.connect(self.on_end_date_changed)
        end_layout.addWidget(end_label)
        end_layout.addWidget(self.end_calendar)

        calendar_layout.addLayout(start_layout)
        calendar_layout.addLayout(end_layout)
        date_layout.addLayout(calendar_layout)
        
        # Add date group to right panel
        right_layout.addWidget(date_group)

        # Periodicity Selection
        periodicity_group = QGroupBox("Data Periodicity")
        periodicity_layout = QVBoxLayout(periodicity_group)
        
        # Create button group for periodicities
        self.periodicity_group = QButtonGroup()
        
        # Create radio buttons
        for period in self.periodicities:
            radio = QRadioButton(period)
            if period == '1d':  # Default selection
                radio.setChecked(True)
            self.periodicity_group.addButton(radio)
            periodicity_layout.addWidget(radio)

        right_layout.addWidget(periodicity_group)

        # Create tab widget for different views
        self.tab_widget = QTabWidget()
        
        # Summary tab
        self.summary_tab = QWidget()
        summary_layout = QVBoxLayout(self.summary_tab)
        self.quote_container = QWidget()
        self.quote_container_layout = QVBoxLayout(self.quote_container)
        summary_layout.addWidget(self.quote_container)
        self.tab_widget.addTab(self.summary_tab, "Summary")
        
        # Historical data tab
        self.historical_tab = QWidget()
        historical_layout = QVBoxLayout(self.historical_tab)
        self.historical_table = QTableWidget()
        self.historical_table.setEditTriggers(QTableWidget.NoEditTriggers)
        historical_layout.addWidget(self.historical_table)
        self.tab_widget.addTab(self.historical_tab, "Historical Data")
        
        # Technical indicators tab
        self.technical_tab = QWidget()
        technical_layout = QVBoxLayout(self.technical_tab)
        self.technical_table = QTableWidget()
        self.technical_table.setEditTriggers(QTableWidget.NoEditTriggers)
        technical_layout.addWidget(self.technical_table)
        self.tab_widget.addTab(self.technical_tab, "Technical Indicators")
        
        right_layout.addWidget(self.tab_widget)

        # Status and Progress
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        right_layout.addWidget(self.progress_bar)

        # Buttons
        button_layout = QHBoxLayout()
        
        self.fetch_button = QPushButton("Fetch Data")
        self.fetch_button.clicked.connect(self.get_quotes)
        button_layout.addWidget(self.fetch_button)

        self.export_button = QPushButton("Export Data")
        self.export_button.setEnabled(False)
        self.export_menu = QMenu(self)
        self.export_menu.addAction("CSV", lambda: self.export_data("csv"))
        self.export_menu.addAction("JSON", lambda: self.export_data("json"))
        self.export_menu.addAction("SQLite", lambda: self.export_data("sqlite"))
        self.export_menu.addAction("DuckDB", lambda: self.export_data("duckdb"))
        self.export_menu.addAction("Convert to Polars", lambda: self.export_data("polars"))
        self.export_button.setMenu(self.export_menu)
        button_layout.addWidget(self.export_button)

        right_layout.addLayout(button_layout)

        main_layout.addWidget(right_panel)

        # Set default dates
        self.set_default_dates()

        # Initialize first category
        self.on_category_changed(self.category_combo.currentText())

        # Window setup
        self.setWindowTitle("Stock Quote Viewer")
        self.setGeometry(100, 100, 1200, 800)
        self.apply_styles()

    def apply_styles(self):
        """Apply dark mode styling."""
        dark_style = """
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QFrame {
                background-color: #2d2d2d;
                border-radius: 5px;
                padding: 10px;
                margin: 5px;
                border: 1px solid #3d3d3d;
            }
            QPushButton {
                padding: 8px;
                background-color: #0d47a1;
                color: white;
                border-radius: 4px;
                margin: 5px;
                font-weight: bold;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
            QPushButton:pressed {
                background-color: #0a3d91;
            }
            QPushButton:disabled {
                background-color: #424242;
                color: #6e6e6e;
            }
            QComboBox {
                padding: 5px;
                margin: 5px;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
                background-color: #2d2d2d;
                color: white;
                min-height: 20px;
            }
            QComboBox::drop-down {
                border: none;
                background-color: #0d47a1;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-width: 0px;
            }
            QComboBox QAbstractItemView {
                background-color: #2d2d2d;
                color: white;
                selection-background-color: #0d47a1;
                selection-color: white;
            }
            QLabel {
                padding: 5px;
                color: white;
            }
            QListWidget {
                border: 1px solid #3d3d3d;
                border-radius: 3px;
                padding: 5px;
                background-color: #2d2d2d;
                color: white;
            }
            QListWidget::item {
                padding: 5px;
                color: white;
            }
            QListWidget::item:selected {
                background-color: #0d47a1;
                color: white;
            }
            QListWidget::item:hover {
                background-color: #1565c0;
            }
            QLineEdit {
                padding: 5px;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
                background-color: #2d2d2d;
                color: white;
            }
            QGroupBox {
                border: 1px solid #3d3d3d;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
                color: white;
            }
            QCalendarWidget {
                background-color: #2d2d2d;
                color: white;
            }
            QCalendarWidget QToolButton {
                color: white;
                background-color: #2d2d2d;
                padding: 5px;
                border: none;
            }
            QCalendarWidget QMenu {
                background-color: #2d2d2d;
                color: white;
            }
            QCalendarWidget QSpinBox {
                background-color: #2d2d2d;
                color: white;
                selection-background-color: #0d47a1;
                selection-color: white;
            }
            QCalendarWidget QAbstractItemView:enabled {
                background-color: #2d2d2d;
                color: white;
                selection-background-color: #0d47a1;
                selection-color: white;
            }
            QCalendarWidget QWidget#qt_calendar_navigationbar {
                background-color: #2d2d2d;
            }
            QCalendarWidget QAbstractItemView:disabled {
                color: #666666;
            }
            QProgressBar {
                border: 1px solid #3d3d3d;
                border-radius: 3px;
                background-color: #2d2d2d;
                color: white;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #0d47a1;
            }
            QRadioButton {
                color: white;
                spacing: 5px;
            }
            QRadioButton::indicator {
                width: 13px;
                height: 13px;
            }
            QRadioButton::indicator:checked {
                background-color: #0d47a1;
                border: 2px solid white;
                border-radius: 7px;
            }
            QRadioButton::indicator:unchecked {
                background-color: #2d2d2d;
                border: 2px solid white;
                border-radius: 7px;
            }
            QMenu {
                background-color: #2d2d2d;
                color: white;
                border: 1px solid #3d3d3d;
            }
            QMenu::item {
                padding: 5px 20px;
            }
            QMenu::item:selected {
                background-color: #0d47a1;
            }
            QScrollBar:vertical {
                border: none;
                background-color: #2d2d2d;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background-color: #3d3d3d;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                border: none;
                background-color: #2d2d2d;
                height: 10px;
                margin: 0px;
            }
            QScrollBar::handle:horizontal {
                background-color: #3d3d3d;
                min-width: 20px;
                border-radius: 5px;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            QLabel#date_label {
                color: #00ff00;
                font-weight: bold;
                padding: 5px;
                background-color: #2d2d2d;
                border-radius: 3px;
            }
        """
        self.setStyleSheet(dark_style)
        
        # Set object names for date labels to apply specific styles
        self.start_date_label.setObjectName("date_label")
        self.end_date_label.setObjectName("date_label")

    def on_category_changed(self, category: str):
        """Update ticker list when category changes."""
        self.ticker_list.clear()
        for ticker in self.ticker_manager.get_tickers_in_category(category):
            item = QListWidgetItem(ticker)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if ticker in self.selected_tickers else Qt.Unchecked)
            self.ticker_list.addItem(item)

    def on_ticker_selection_changed(self, item: QListWidgetItem):
        """Handle ticker selection changes."""
        ticker = item.text()
        if item.checkState() == Qt.Checked:
            self.selected_tickers.add(ticker)
        else:
            self.selected_tickers.discard(ticker)
        
        # Update count label
        self.ticker_count_label.setText(f"Selected: {len(self.selected_tickers)}")

    def select_all_tickers(self):
        """Select all tickers in current category."""
        for i in range(self.ticker_list.count()):
            item = self.ticker_list.item(i)
            item.setCheckState(Qt.Checked)

    def clear_all_tickers(self):
        """Clear all ticker selections."""
        for i in range(self.ticker_list.count()):
            item = self.ticker_list.item(i)
            item.setCheckState(Qt.Unchecked)

    def clear_quotes(self):
        """Clear all quote displays."""
        # Remove all widgets from the quote container
        while self.quote_container_layout.count():
            child = self.quote_container_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def set_default_dates(self):
        """Set default date range to last 30 days."""
        try:
            # Set end date to today
            today = QDate.currentDate()
            self.end_calendar.setSelectedDate(today)
            
            # Set start date to 30 days ago
            start_date = today.addDays(-30)
            self.start_calendar.setSelectedDate(start_date)
            
            # Update UI elements
            self.update_date_labels(start_date, today)
            self.update_period_combo(start_date, today)
            
        except Exception as e:
            logging.error(f"Error setting default dates: {e}")

    def update_date_labels(self, start_date: QDate, end_date: QDate):
        """Update the date display labels."""
        try:
            self.start_date_label.setText(
                f"Start: {start_date.toString('yyyy-MM-dd')}"
            )
            self.end_date_label.setText(
                f"End: {end_date.toString('yyyy-MM-dd')}"
            )
            
            # Update status with date range
            days_between = start_date.daysTo(end_date)
            self.status_label.setText(
                f"Date Range: {days_between} days selected"
            )
            
        except Exception as e:
            logging.error(f"Error in update_date_labels: {e}")

    def on_period_changed(self, period: str):
        """Handle quick period selection."""
        try:
            if period.startswith('Custom'):
                return
                
            end_date = QDate.currentDate()
            delta = self.time_periods[period]
            start_date = end_date.addDays(-delta.days)
            
            self.start_calendar.setSelectedDate(start_date)
            self.end_calendar.setSelectedDate(end_date)
            
            self.update_date_labels(start_date, end_date)
            
        except Exception as e:
            logging.error(f"Error in on_period_changed: {e}")

    def on_start_date_changed(self, qdate: QDate):
        """Handle start date changes."""
        try:
            # Ensure start date is not after end date
            end_date = self.end_calendar.selectedDate()
            if qdate > end_date:
                self.start_calendar.setSelectedDate(end_date)
                return
                
            # Update period combo and labels
            self.update_period_combo(qdate, end_date)
            self.update_date_labels(qdate, end_date)
            
        except Exception as e:
            logging.error(f"Error handling start date change: {e}")

    def on_end_date_changed(self, qdate: QDate):
        """Handle end date changes."""
        try:
            # Ensure end date is not before start date and not in future
            start_date = self.start_calendar.selectedDate()
            today = QDate.currentDate()
            
            if qdate < start_date:
                self.end_calendar.setSelectedDate(start_date)
                return
                
            if qdate > today:
                self.end_calendar.setSelectedDate(today)
                return
                
            # Update period combo and labels
            self.update_period_combo(start_date, qdate)
            self.update_date_labels(start_date, qdate)
            
        except Exception as e:
            logging.error(f"Error handling end date change: {e}")

    def update_period_combo(self, start_date: QDate, end_date: QDate):
        """Update period combo box based on selected dates."""
        try:
            days_diff = start_date.daysTo(end_date)
            
            # Find matching period or set to custom
            period_found = False
            for period, delta in self.time_periods.items():
                if abs(delta.days - days_diff) <= 1:  # Allow 1 day difference
                    self.period_combo.blockSignals(True)  # Prevent recursive calls
                    self.period_combo.setCurrentText(period)
                    self.period_combo.blockSignals(False)
                    period_found = True
                    break
            
            if not period_found:
                # Remove any existing custom period
                for i in range(self.period_combo.count()):
                    if self.period_combo.itemText(i).startswith('Custom'):
                        self.period_combo.removeItem(i)
                        break
                
                # Add new custom period
                custom_text = f"Custom ({days_diff} days)"
                self.period_combo.blockSignals(True)
                self.period_combo.addItem(custom_text)
                self.period_combo.setCurrentText(custom_text)
                self.period_combo.blockSignals(False)
                self.custom_period = timedelta(days=days_diff)
            
        except Exception as e:
            logging.error(f"Error in update_period_combo: {e}")

    def get_selected_periodicity(self) -> str:
        """Get the selected periodicity."""
        selected_button = self.periodicity_group.checkedButton()
        return selected_button.text() if selected_button else '1d'

    def get_quotes(self):
        """Fetch quotes and historical data for selected tickers."""
        if not self.selected_tickers:
            self.status_label.setText("Please select at least one ticker")
            return

        try:
            start_date = self.start_calendar.selectedDate().toPyDate()
            end_date = self.end_calendar.selectedDate().toPyDate()
            
            # Validate date range
            if start_date > end_date:
                self.status_label.setText("Invalid date range")
                return
            
            # Ensure end date includes the full day
            end_date = datetime.combine(end_date, datetime.max.time())
            
            interval = self.get_selected_periodicity()

            self.fetch_button.setEnabled(False)
            self.export_button.setEnabled(False)
            self.status_label.setText(
                f"Fetching {interval} data from {start_date.strftime('%Y-%m-%d')} "
                f"to {end_date.strftime('%Y-%m-%d')}..."
            )
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            self.clear_quotes()

            # Get historical data with technical indicators
            self.current_data = self.ticker_manager.get_quotes_df(
                list(self.selected_tickers),
                start_date,
                end_date,
                interval
            )
            
            if not self.current_data.empty:
                self.update_all_views()
                self.status_label.setText("Data updated successfully")
                self.export_button.setEnabled(True)
            else:
                self.status_label.setText("No data received")

        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            logging.error(f"Error fetching data: {e}")

        finally:
            QTimer.singleShot(2000, lambda: self.fetch_button.setEnabled(True))
            self.progress_bar.setVisible(False)

    def update_all_views(self):
        """Update all data views."""
        try:
            # Update summary tab
            self.update_summary_view()
            
            # Update historical data tab
            self.update_historical_table()
            
            # Update technical indicators tab
            self.update_technical_table()
            
        except Exception as e:
            logging.error(f"Error updating views: {e}")

    def update_summary_view(self):
        """Update the summary view with latest data."""
        try:
            latest_data = self.current_data.groupby('Symbol').last()
            for symbol in latest_data.index:
                row = latest_data.loc[symbol]
                quote_label = QLabel()
                quote_text = (
                    f"{symbol}: ${row['Close']:.2f} "
                    f"({row['Daily_Return']*100:+.2f}%)"
                )
                quote_label.setText(quote_text)
                quote_label.setStyleSheet(
                    "color: #00ff00;" if row['Daily_Return'] >= 0 else "color: #ff4444;"
                )
                self.quote_container_layout.addWidget(quote_label)
                
        except Exception as e:
            logging.error(f"Error updating summary view: {e}")

    def update_historical_table(self):
        """Update the historical data table."""
        try:
            df = self.current_data
            self.historical_table.clear()
            self.historical_table.setRowCount(len(df))
            self.historical_table.setColumnCount(7)
            
            headers = ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']
            self.historical_table.setHorizontalHeaderLabels(headers)
            
            for i, (_, row) in enumerate(df.iterrows()):
                for j, col in enumerate(headers):
                    item = QTableWidgetItem(str(row[col]))
                    self.historical_table.setItem(i, j, item)
            
            # Adjust column widths
            self.historical_table.horizontalHeader().setSectionResizeMode(
                QHeaderView.ResizeToContents
            )

        except Exception as e:
            logging.error(f"Error updating historical table: {e}")

    def update_technical_table(self):
        """Update the technical indicators table."""
        try:
            df = self.current_data
            self.technical_table.clear()
            self.technical_table.setRowCount(len(df))
            self.technical_table.setColumnCount(8)
            
            headers = ['Date', 'Symbol', 'RSI', 'MACD', 'SMA_20', 'BB_Upper', 
                      'BB_Lower', 'Volatility']
            self.technical_table.setHorizontalHeaderLabels(headers)
            
            for i, (_, row) in enumerate(df.iterrows()):
                for j, col in enumerate(headers):
                    value = row[col]
                    if isinstance(value, float):
                        value = f"{value:.2f}"
                    item = QTableWidgetItem(str(value))
                    self.technical_table.setItem(i, j, item)
            
            # Adjust column widths
            self.technical_table.horizontalHeader().setSectionResizeMode(
                QHeaderView.ResizeToContents
            )

        except Exception as e:
            logging.error(f"Error updating technical table: {e}")

    def export_data(self, format_type: str):
        """Export data in the specified format."""
        if not hasattr(self, 'current_data') or self.current_data.empty:
            QMessageBox.warning(self, "Export Error", "No data available to export")
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stock_data_{timestamp}"
            
            if format_type == "csv":
                path = self.ticker_manager.export_to_csv(self.current_data, filename)
                msg = f"Data exported to CSV: {path}"
            
            elif format_type == "json":
                path = self.ticker_manager.export_to_json(self.current_data, filename)
                msg = f"Data exported to JSON: {path}"
            
            elif format_type == "sqlite":
                path = self.ticker_manager.export_to_sqlite(
                    self.current_data, filename, "stock_quotes"
                )
                msg = f"Data exported to SQLite: {path}"
            
            elif format_type == "duckdb":
                path = self.ticker_manager.export_to_duckdb(
                    self.current_data, filename, "stock_quotes"
                )
                msg = f"Data exported to DuckDB: {path}"
            
            elif format_type == "polars":
                polars_df = self.ticker_manager.convert_to_polars(self.current_data)
                msg = f"Data converted to Polars DataFrame with {len(polars_df)} rows"
            
            QMessageBox.information(self, "Export Success", msg)
            self.status_label.setText(f"Export successful: {format_type}")

        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))
            self.status_label.setText(f"Export failed: {str(e)}")
            logging.error(f"Export error: {e}")

    def closeEvent(self, event):
        """Clean up on close."""
        try:
            # Clear any cached data
            if hasattr(self, 'current_data'):
                del self.current_data
            
            # Clear the ticker manager
            if hasattr(self, 'ticker_manager'):
                del self.ticker_manager
                
            # Force garbage collection
            gc.collect()
            
            event.accept()
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")
            event.accept()

    def filter_tickers(self, search_text: str):
        """Filter tickers based on search text."""
        category = self.category_combo.currentText()
        tickers = self.ticker_manager.get_tickers_in_category(category)
        
        self.ticker_list.clear()
        search_text = search_text.upper()
        
        for ticker in tickers:
            if search_text in ticker:
                item = QListWidgetItem(ticker)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked if ticker in self.selected_tickers else Qt.Unchecked)
                self.ticker_list.addItem(item)

def main():
    app = QApplication(sys.argv)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create and show the GUI
    gui = StockGUI()
    gui.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 