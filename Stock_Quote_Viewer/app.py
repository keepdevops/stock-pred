import sys
import logging
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QComboBox, QListWidget, 
                            QPushButton, QLineEdit, QMessageBox, QProgressBar,
                            QListWidgetItem, QFileDialog, QGroupBox, QDateEdit,
                            QCheckBox, QRadioButton, QSpinBox, QTextEdit, QDockWidget,
                            QTableWidget, QTableWidgetItem)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QDate
from src.data.ticker_manager import TickerManager
from datetime import datetime
import os
import functools
import pandas_datareader.data as web

# Set up logging for the root logger - will be captured by GUI log handler
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Console output for debugging
    ]
)
logger = logging.getLogger(__name__)

def signal_safe(func):
    """Decorator to make signal handlers safe from parameter mismatches"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TypeError as e:
            if "too many values to unpack" in str(e):
                logger.warning(f"Signal parameter mismatch in {func.__name__}: {e}")
                # Try to call with just the first argument if multiple were provided
                if len(args) > 1:
                    return func(args[0])
            else:
                logger.error(f"TypeError in {func.__name__}: {e}")
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
    return wrapper

class QTextEditLogger(logging.Handler):
    """Custom logging handler that redirects log messages to a QTextEdit widget"""
    def __init__(self, text_edit):
        super().__init__()
        self.text_edit = text_edit
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                          datefmt='%Y-%m-%d %H:%M:%S')

    def emit(self, record):
        try:
            msg = self.formatter.format(record)
            self.text_edit.append(msg)
            # Auto-scroll to the bottom
            self.text_edit.verticalScrollBar().setValue(
                self.text_edit.verticalScrollBar().maximum()
            )
        except Exception:
            self.handleError(record)

class DataDownloadWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, str)
    ticker_data = pyqtSignal(str, object)

    def __init__(self, ticker_manager, tickers, start_date, end_date, download_mode='sequential', 
                 cleaning_options=None, interval='1d', processing_mode='sequential', batch_size=None, data_source='yfinance', pdr_sub_source=None):
        super().__init__()
        self.ticker_manager = ticker_manager
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.download_mode = download_mode
        self.cleaning_options = cleaning_options
        self.interval = interval
        self.processing_mode = processing_mode
        self.batch_size = batch_size
        self.data_source = data_source
        self.pdr_sub_source = pdr_sub_source

    def run(self):
        try:
            data = {}
            total_tickers = len(self.tickers)
            processed = 0

            if self.processing_mode == 'sequential':
                # Process one ticker at a time
                for ticker in self.tickers:
                    self.progress.emit(processed, f"Processing {ticker} ({processed+1}/{total_tickers})")
                    ticker_data = self.ticker_manager.get_historical_data(
                        [ticker],
                        self.start_date,
                        self.end_date,
                        download_mode=self.download_mode,
                        clean_data=True,
                        interval=self.interval,
                        data_source=self.data_source,
                        pdr_sub_source=self.pdr_sub_source
                    )
                    if ticker_data:
                        data.update(ticker_data)
                        # Emit per-ticker data for live display
                        if ticker in ticker_data and ticker_data[ticker] is not None:
                            self.ticker_data.emit(ticker, ticker_data[ticker])
                    processed += 1
                    self.progress.emit(processed, f"Completed {processed}/{total_tickers}")

            elif self.processing_mode == 'parallel':
                # Process all tickers in parallel
                self.progress.emit(0, f"Processing all {total_tickers} tickers in parallel")
                data = self.ticker_manager.get_historical_data(
                    self.tickers,
                    self.start_date,
                    self.end_date,
                    download_mode='parallel',
                    clean_data=True,
                    interval=self.interval,
                    data_source=self.data_source,
                    pdr_sub_source=self.pdr_sub_source
                )
                # Emit all data at once (optional: could emit per ticker if needed)
                for ticker, df in data.items():
                    if df is not None:
                        self.ticker_data.emit(ticker, df)
                self.progress.emit(total_tickers, f"Completed all {total_tickers} tickers")

            elif self.processing_mode == 'async':
                # Process all tickers asynchronously
                self.progress.emit(0, f"Processing all {total_tickers} tickers asynchronously")
                data = self.ticker_manager.get_historical_data(
                    self.tickers,
                    self.start_date,
                    self.end_date,
                    download_mode='async',
                    clean_data=True,
                    interval=self.interval,
                    data_source=self.data_source,
                    pdr_sub_source=self.pdr_sub_source
                )
                for ticker, df in data.items():
                    if df is not None:
                        self.ticker_data.emit(ticker, df)
                self.progress.emit(total_tickers, f"Completed all {total_tickers} tickers")

            else:  # batch processing
                # Process tickers in batches
                for i in range(0, total_tickers, self.batch_size):
                    batch = self.tickers[i:i + self.batch_size]
                    batch_size = len(batch)
                    self.progress.emit(processed, f"Processing batch {i//self.batch_size + 1} ({batch_size} tickers)")
                    batch_data = self.ticker_manager.get_historical_data(
                        batch,
                        self.start_date,
                        self.end_date,
                        download_mode=self.download_mode,
                        clean_data=True,
                        interval=self.interval,
                        data_source=self.data_source,
                        pdr_sub_source=self.pdr_sub_source
                    )
                    if batch_data:
                        data.update(batch_data)
                        for ticker, df in batch_data.items():
                            if df is not None:
                                self.ticker_data.emit(ticker, df)
                    processed += len(batch)
                    self.progress.emit(processed, f"Completed {processed}/{total_tickers}")

            if data:
                # Apply cleaning options if provided
                if self.cleaning_options:
                    for ticker, df in data.items():
                        if df is not None:
                            data[ticker] = self.ticker_manager.clean_data(df, self.cleaning_options)
                
                self.finished.emit(data)
            else:
                self.error.emit("No data was downloaded")
                
        except Exception as e:
            logger.error(f"Error in download worker: {e}")
            self.error.emit(str(e))

class StockGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        # Add attributes for db name and dir
        self.db_name = ''
        self.db_dir = ''
        self.ticker_manager = None
        self.selected_tickers = set()
        self.current_data = {}
        self.data_source = 'yfinance'
        self.init_ui()
        self.setup_log_display()
        self.resize(1200, 800)           # Initial size
        self.setMinimumSize(400, 300)    # Allow shrinking, but not below 400x300 pixels
        # self.showMaximized()           # (Optional) Start maximized

    def init_ui(self):
        self.setWindowTitle('Stock Quote Viewer')
        self.setGeometry(100, 100, 1000, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # --- DB Name/Dir Controls ---
        db_layout = QHBoxLayout()
        db_name_label = QLabel('Database Name:')
        self.db_name_edit = QLineEdit()
        self.db_name_edit.setPlaceholderText('Enter database name (e.g. mydata.duckdb)')
        db_layout.addWidget(db_name_label)
        db_layout.addWidget(self.db_name_edit)
        db_dir_label = QLabel('Save Directory:')
        self.db_dir_edit = QLineEdit()
        self.db_dir_edit.setPlaceholderText('Choose directory')
        db_layout.addWidget(db_dir_label)
        db_layout.addWidget(self.db_dir_edit)
        db_dir_btn = QPushButton('Browse')
        db_dir_btn.clicked.connect(self.browse_db_dir)
        db_layout.addWidget(db_dir_btn)
        apply_db_btn = QPushButton('Apply DB Settings')
        apply_db_btn.clicked.connect(self.apply_db_settings)
        db_layout.addWidget(apply_db_btn)
        layout.addLayout(db_layout)

        # --- Data Source Selection ---
        data_source_group = QGroupBox('Data Source')
        data_source_layout = QHBoxLayout()
        self.yfinance_radio = QRadioButton('yfinance')
        self.pdr_radio = QRadioButton('pandas_datareader')
        self.yfinance_radio.setChecked(True)
        data_source_layout.addWidget(self.yfinance_radio)
        data_source_layout.addWidget(self.pdr_radio)
        self.pdr_source_combo = QComboBox()
        self.pdr_source_combo.addItems(['stooq', 'av', 'fred'])  # Add more as needed
        self.pdr_source_combo.setVisible(False)
        data_source_layout.addWidget(self.pdr_source_combo)
        data_source_group.setLayout(data_source_layout)
        layout.addWidget(data_source_group)
        self.yfinance_radio.toggled.connect(lambda: self.set_data_source('yfinance'))
        self.pdr_radio.toggled.connect(lambda: self.set_data_source('pandas_datareader'))

        # Show/hide the sub-source combo based on selection
        self.pdr_radio.toggled.connect(lambda checked: self.pdr_source_combo.setVisible(checked))

        # --- Top Controls (Category/Search) ---
        top_controls = QHBoxLayout()
        category_label = QLabel('Category:')
        self.category_combo = QComboBox()
        # TickerManager will be created after DB settings applied
        self.category_combo.currentTextChanged.connect(self.update_ticker_list)
        top_controls.addWidget(category_label)
        top_controls.addWidget(self.category_combo)
        search_label = QLabel('Search:')
        self.search_box = QLineEdit()
        self.search_box.textChanged.connect(self.filter_tickers)
        top_controls.addWidget(search_label)
        top_controls.addWidget(self.search_box)
        layout.addLayout(top_controls)

        # --- Main Content ---
        content_layout = QHBoxLayout()
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        self.ticker_list = QListWidget()
        self.ticker_list.setSelectionMode(QListWidget.MultiSelection)
        self.ticker_list.itemChanged.connect(self.on_ticker_selection_changed)
        left_layout.addWidget(self.ticker_list)
        button_layout = QHBoxLayout()
        self.select_all_btn = QPushButton('Select All')
        self.clear_btn = QPushButton('Clear Selection')
        self.select_all_btn.clicked.connect(self.select_all_tickers)
        self.clear_btn.clicked.connect(self.clear_selection)
        button_layout.addWidget(self.select_all_btn)
        button_layout.addWidget(self.clear_btn)
        left_layout.addLayout(button_layout)
        content_layout.addWidget(left_panel)

        # --- Right Panel ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Date Range Selection
        date_group = QGroupBox("Date Range")
        date_layout = QVBoxLayout()
        
        # Start Date
        start_date_layout = QHBoxLayout()
        start_date_layout.addWidget(QLabel("Start:"))
        self.start_date = QDateEdit()
        self.start_date.setCalendarPopup(True)
        self.start_date.setDate(QDate.currentDate().addYears(-1))
        start_date_layout.addWidget(self.start_date)
        date_layout.addLayout(start_date_layout)
        
        # End Date
        end_date_layout = QHBoxLayout()
        end_date_layout.addWidget(QLabel("End:"))
        self.end_date = QDateEdit()
        self.end_date.setCalendarPopup(True)
        self.end_date.setDate(QDate.currentDate())
        end_date_layout.addWidget(self.end_date)
        date_layout.addLayout(end_date_layout)
        
        # Interval Selection
        interval_layout = QHBoxLayout()
        interval_layout.addWidget(QLabel("Interval:"))
        self.interval_combo = QComboBox()
        self.interval_combo.addItems(['1d', '1w', '1mo', '3mo', '1y', '5y', 'max'])
        interval_layout.addWidget(self.interval_combo)
        date_layout.addLayout(interval_layout)
        
        date_group.setLayout(date_layout)
        right_layout.addWidget(date_group)

        # Download mode selector
        download_mode_layout = QHBoxLayout()
        download_mode_label = QLabel('Download Mode:')
        self.download_mode_combo = QComboBox()
        self.download_mode_combo.addItems(['sequential', 'parallel', 'async'])
        download_mode_layout.addWidget(download_mode_label)
        download_mode_layout.addWidget(self.download_mode_combo)
        right_layout.addLayout(download_mode_layout)

        # Ticker Processing Mode
        processing_group = QGroupBox("Ticker Processing Mode")
        processing_layout = QVBoxLayout()
        
        # Create radio buttons for processing modes
        self.sequential_radio = QRadioButton("Sequential (One by One)")
        self.parallel_radio = QRadioButton("Parallel (All at Once)")
        self.async_radio = QRadioButton("Asynchronous (Non-blocking)")
        self.batch_radio = QRadioButton("Batch Processing (Groups)")
        
        # Set sequential as default
        self.sequential_radio.setChecked(True)
        
        # Add radio buttons to layout
        processing_layout.addWidget(self.sequential_radio)
        processing_layout.addWidget(self.parallel_radio)
        processing_layout.addWidget(self.async_radio)
        processing_layout.addWidget(self.batch_radio)
        
        # Add batch size selector
        batch_size_layout = QHBoxLayout()
        batch_size_label = QLabel("Batch Size:")
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 10)
        self.batch_size_spin.setValue(3)
        self.batch_size_spin.setEnabled(False)  # Disabled by default
        batch_size_layout.addWidget(batch_size_label)
        batch_size_layout.addWidget(self.batch_size_spin)
        processing_layout.addLayout(batch_size_layout)
        
        processing_group.setLayout(processing_layout)
        right_layout.addWidget(processing_group)

        # Data Cleaning Options
        cleaning_group = QGroupBox("Data Cleaning Options")
        cleaning_layout = QVBoxLayout()
        
        # Create checkboxes for cleaning options
        self.clean_missing = QCheckBox("Handle Missing Values")
        self.clean_outliers = QCheckBox("Remove Outliers")
        self.clean_dates = QCheckBox("Standardize Dates")
        self.clean_gaps = QCheckBox("Fill Trading Gaps")
        self.clean_volume = QCheckBox("Normalize Volume")
        self.clean_duplicates = QCheckBox("Remove Duplicates")
        
        # Set all checkboxes to checked by default
        for checkbox in [self.clean_missing, self.clean_outliers, self.clean_dates,
                        self.clean_gaps, self.clean_volume, self.clean_duplicates]:
            checkbox.setChecked(True)
            cleaning_layout.addWidget(checkbox)
        
        cleaning_group.setLayout(cleaning_layout)
        right_layout.addWidget(cleaning_group)

        # Download button
        self.download_btn = QPushButton('Download Data')
        self.download_btn.clicked.connect(self.download_data)
        right_layout.addWidget(self.download_btn)

        # Export controls
        export_layout = QHBoxLayout()
        export_label = QLabel('Export Format:')
        self.export_combo = QComboBox()
        # REMOVE or comment out this line:
        # self.export_combo.addItems(self.ticker_manager.get_export_formats())
        export_layout.addWidget(export_label)
        export_layout.addWidget(self.export_combo)
        right_layout.addLayout(export_layout)

        # Export button
        self.export_btn = QPushButton('Export Data')
        self.export_btn.clicked.connect(self.export_data)
        self.export_btn.setEnabled(False)
        right_layout.addWidget(self.export_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        right_layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel('')
        right_layout.addWidget(self.status_label)

        # Add QTableWidget for live data display
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(0)
        self.data_table.setRowCount(0)
        self.data_table.setMinimumHeight(200)
        right_layout.addWidget(QLabel('Live Data Preview:'))
        right_layout.addWidget(self.data_table)

        content_layout.addWidget(right_panel)
        layout.addLayout(content_layout)

        # Initialize ticker list
        # self.update_ticker_list()

        # Connect pdr_source_combo to set pdr_sub_source
        self.pdr_source_combo.currentTextChanged.connect(
            lambda text: setattr(self, 'pdr_sub_source', text)
        )

    def setup_log_display(self):
        """Set up log display as a dock widget"""
        self.log_dock = QDockWidget("Log Messages", self)
        self.log_dock.setAllowedAreas(Qt.BottomDockWidgetArea)
        
        # Create log text widget
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        # Add clear button
        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(self.clear_log)
        log_layout.addWidget(clear_btn)
        
        # Set as dock widget content
        self.log_dock.setWidget(log_widget)
        
        # Add dock to main window
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)
        
        # Create and install log handler
        log_handler = QTextEditLogger(self.log_text)
        log_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        log_handler.setFormatter(formatter)
        
        # Add handler to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(log_handler)
        
        # Initial log message
        logger.info("Log display initialized in GUI")
        
    def clear_log(self):
        """Clear the log display"""
        self.log_text.clear()
        logger.info("Log cleared")

    def browse_db_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, 'Select Directory')
        if dir_path:
            self.db_dir_edit.setText(dir_path)

    def apply_db_settings(self):
        self.db_name = self.db_name_edit.text().strip()
        self.db_dir = self.db_dir_edit.text().strip()
        if not self.db_name or not self.db_dir:
            QMessageBox.warning(self, 'Warning', 'Please specify both database name and directory.')
            return
        self.ticker_manager = TickerManager(db_name=self.db_name, db_dir=self.db_dir)
        self.category_combo.blockSignals(True)
        self.category_combo.clear()
        categories = self.ticker_manager.get_categories()
        self.category_combo.addItems(categories)
        self.category_combo.blockSignals(False)
        if categories:
            self.category_combo.setCurrentIndex(0)
        self.update_ticker_list()
        logger.info(f"Database settings applied: {self.db_name} @ {self.db_dir}")
        self.export_combo.clear()
        self.export_combo.addItems(self.ticker_manager.get_export_formats())

    def update_ticker_list(self, category='All'):
        if not self.ticker_manager:
            return
        self.ticker_list.clear()
        tickers = self.ticker_manager.get_tickers_by_category(category)
        for ticker in tickers:
            item = QListWidgetItem(ticker)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if ticker in self.selected_tickers else Qt.Unchecked)
            self.ticker_list.addItem(item)
        logger.info(f"Updated ticker list for category {category} with {len(tickers)} tickers")

    def filter_tickers(self, text):
        if not self.ticker_manager:
            return
        if not text:
            self.update_ticker_list(self.category_combo.currentText())
            return
        filtered_tickers = self.ticker_manager.search_tickers(text)
        self.ticker_list.clear()
        for ticker in filtered_tickers:
            item = QListWidgetItem(ticker)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if ticker in self.selected_tickers else Qt.Unchecked)
            self.ticker_list.addItem(item)
        logger.info(f"Filtered ticker list with search text '{text}'")

    def on_ticker_selection_changed(self, item):
        """Handle ticker selection changes"""
        ticker = item.text()
        if item.checkState() == Qt.Checked:
            self.selected_tickers.add(ticker)
        else:
            self.selected_tickers.discard(ticker)
        logger.info(f"Ticker selection updated: {len(self.selected_tickers)} selected")

    def select_all_tickers(self):
        """Select all visible tickers"""
        for i in range(self.ticker_list.count()):
            item = self.ticker_list.item(i)
            item.setCheckState(Qt.Checked)
            self.selected_tickers.add(item.text())
        logger.info("Selected all visible tickers")

    def clear_selection(self):
        """Clear all ticker selections"""
        for i in range(self.ticker_list.count()):
            item = self.ticker_list.item(i)
            item.setCheckState(Qt.Unchecked)
        self.selected_tickers.clear()
        logger.info("Cleared all ticker selections")

    def set_data_source(self, source):
        if source == 'yfinance' and self.yfinance_radio.isChecked():
            self.data_source = 'yfinance'
            self.pdr_sub_source = None
        elif source == 'pandas_datareader' and self.pdr_radio.isChecked():
            self.data_source = 'pandas_datareader'
            self.pdr_sub_source = self.pdr_source_combo.currentText()

    def download_data(self):
        if not self.selected_tickers:
            QMessageBox.warning(self, "Warning", "Please select at least one ticker")
            return
        if not self.ticker_manager:
            QMessageBox.warning(self, "Warning", "Please apply database settings first.")
            return

        try:
            # Get date range from date pickers
            start_date = self.start_date.date().toString('yyyy-MM-dd')
            end_date = self.end_date.date().toString('yyyy-MM-dd')

            # Get download mode and interval
            download_mode = self.download_mode_combo.currentText()
            interval = self.interval_combo.currentText()

            # Get processing mode
            if self.sequential_radio.isChecked():
                processing_mode = 'sequential'
            elif self.parallel_radio.isChecked():
                processing_mode = 'parallel'
            elif self.async_radio.isChecked():
                processing_mode = 'async'
            else:  # batch
                processing_mode = 'batch'
                batch_size = self.batch_size_spin.value()

            # Get cleaning options
            cleaning_options = {
                'handle_missing': self.clean_missing.isChecked(),
                'remove_outliers': self.clean_outliers.isChecked(),
                'standardize_dates': self.clean_dates.isChecked(),
                'fill_gaps': self.clean_gaps.isChecked(),
                'normalize_volume': self.clean_volume.isChecked(),
                'remove_duplicates': self.clean_duplicates.isChecked()
            }

            # Show progress bar
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, len(self.selected_tickers))
            status_text = f"Downloading data using {download_mode} mode with {interval} interval"
            if processing_mode == 'batch':
                status_text += f" (Batch size: {batch_size})"
            self.status_label.setText(status_text)
            self.download_btn.setEnabled(False)

            # Create and start worker thread
            self.worker = DataDownloadWorker(
                self.ticker_manager,
                list(self.selected_tickers),
                start_date,
                end_date,
                download_mode,
                cleaning_options,
                interval,
                processing_mode,
                batch_size if processing_mode == 'batch' else None,
                self.data_source,
                getattr(self, 'pdr_sub_source', None)
            )
            self.worker.finished.connect(self.on_download_finished)
            self.worker.error.connect(self.on_download_error)
            self.worker.progress.connect(self.update_progress)
            self.worker.ticker_data.connect(self.on_ticker_data)
            self.data_table.setRowCount(0)
            self.data_table.setColumnCount(0)
            self.worker.start()

        except Exception as e:
            logger.error(f"Error starting download: {e}")
            QMessageBox.critical(self, "Error", f"Failed to start download: {str(e)}")

    def on_download_finished(self, data):
        """Handle successful data download"""
        self.current_data = data
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Downloaded data for {len(data)} tickers")
        self.download_btn.setEnabled(True)
        self.export_btn.setEnabled(True)  # Enable export button after successful download
        logger.info(f"Successfully downloaded data for {len(data)} tickers")

    def on_download_error(self, error_msg):
        """Handle download error"""
        self.progress_bar.setVisible(False)
        self.status_label.setText("Download failed")
        self.download_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Failed to download data: {error_msg}")
        logger.error(f"Data download failed: {error_msg}")

    def export_data(self):
        """Export downloaded data to selected format"""
        if not self.current_data:
            QMessageBox.warning(self, "Warning", "No data to export. Please download data first.")
            return

        try:
            # Get export format
            export_format = self.export_combo.currentText()
            
            # Get filename from user
            default_name = f"stock_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save Data",
                os.path.join(self.ticker_manager.get_export_dir(), default_name),
                "All Files (*.*)"
            )
            
            if not filename:
                return
            
            # Remove extension if present
            filename = os.path.splitext(filename)[0]
            
            # Export data
            export_path = self.ticker_manager.export_data(
                self.current_data,
                format=export_format,
                filename=os.path.basename(filename)
            )
            
            # Show success message
            QMessageBox.information(
                self,
                "Export Successful",
                f"Data exported successfully to:\n{export_path}"
            )
            self.status_label.setText(f"Exported data to {export_path}")
            logger.info(f"Successfully exported data to {export_path}")
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            QMessageBox.critical(self, "Error", f"Failed to export data: {str(e)}")
            self.status_label.setText("Export failed")

    def quick_select(self, period):
        """Handle quick selection of date ranges"""
        end_date = QDate.currentDate()
        start_date = QDate(end_date)
        
        if period == "1d":
            start_date = end_date.addDays(-1)
        elif period == "1w":
            start_date = end_date.addDays(-7)
        elif period == "1mo":
            start_date = end_date.addMonths(-1)
        elif period == "1y":
            start_date = end_date.addYears(-1)
        elif period == "5y":
            start_date = end_date.addYears(-5)
        elif period == "max":
            start_date = end_date.addYears(-20)  # Maximum reasonable range
        
        self.start_date.setDate(start_date)
        self.end_date.setDate(end_date)

    @signal_safe
    def update_progress(self, value, message=""):
        """Update progress bar with current value and optional message"""
        try:
            self.progress_bar.setValue(value)
            if message:
                self.status_label.setText(message)
            logger.info(f"Progress update: {value}% - {message}")
        except Exception as e:
            logger.error(f"Error in update_progress: {str(e)}")
            # Try to show the error in the GUI
            try:
                self.status_label.setText(f"Error updating progress: {str(e)}")
            except:
                pass

    def on_ticker_data(self, ticker, df):
        # Show the latest ticker's data in the table (append as new row)
        if df is not None and not df.empty:
            if self.data_table.columnCount() == 0:
                self.data_table.setColumnCount(len(df.columns) + 1)
                self.data_table.setHorizontalHeaderLabels(['Ticker'] + list(df.columns))
            row = self.data_table.rowCount()
            self.data_table.insertRow(row)
            self.data_table.setItem(row, 0, QTableWidgetItem(ticker))
            for col, col_name in enumerate(df.columns):
                val = str(df.iloc[0][col_name])
                self.data_table.setItem(row, col + 1, QTableWidgetItem(val))

def main():
    """Main application entry point"""
    try:
        app = QApplication(sys.argv)
        gui = StockGUI()
        gui.show()
        sys.exit(app.exec_())
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 