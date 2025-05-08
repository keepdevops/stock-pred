import sys
import time
import logging
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QComboBox, QListWidget, 
                            QPushButton, QLineEdit, QMessageBox, QProgressBar,
                            QListWidgetItem, QFileDialog, QGroupBox, QDateEdit,
                            QCheckBox, QRadioButton, QSpinBox, QTextEdit, QDockWidget,
                            QTableWidget, QTableWidgetItem, QScrollArea, QInputDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QDate, QTimer
from src.data.ticker_manager import TickerManager
from datetime import datetime
import os
import functools
import pandas_datareader.data as web
import time
import queue
import requests
import pandas as pd
from src.data.twelvedata_downloader import TwelveDataDownloader
from bs4 import BeautifulSoup

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

class BatchDownloadWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(int, str)
    error = pyqtSignal(str)

    def __init__(self, api_key, category, ticker_manager, interval, start_date, end_date, cache_dir=None):
        super().__init__()
        self.api_key = api_key
        self.category = category
        self.ticker_manager = ticker_manager
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date
        self.cache_dir = cache_dir

    def run(self):
        try:
            downloader = TwelveDataDownloader(self.api_key)
            symbols = self.ticker_manager.tickers.get(self.category, [])
            total = len(symbols)
            if not symbols:
                self.error.emit(f"No symbols found for category '{self.category}'")
                return
            # Warn if 1min/5min/15min/30min/45min and >7 days
            if self.interval in ["1min", "5min", "15min", "30min", "45min"]:
                dt1 = datetime.strptime(self.start_date, "%Y-%m-%d")
                dt2 = datetime.strptime(self.end_date, "%Y-%m-%d")
                days = (dt2 - dt1).days
                if days > 7:
                    self.error.emit(f"Twelve Data free tier only allows a few days of 1-minute data. You requested {days} days. Reduce the date range or use a lower granularity.")
                    return
            # Run batch download, emit progress
            BATCH_SIZE = 8
            all_data = {}
            for i in range(0, total, BATCH_SIZE):
                batch = symbols[i:i+BATCH_SIZE]
                self.progress.emit(i, f"Downloading batch {i//BATCH_SIZE+1} of {(total+BATCH_SIZE-1)//BATCH_SIZE} ({len(batch)} symbols)")
                result = downloader.batch_download_category_to_cache(
                    category=self.category,
                    ticker_manager=self.ticker_manager,
                    interval=self.interval,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    cache_dir=self.cache_dir,
                    verbose=False
                )
                all_data.update(result)
            self.finished.emit(len(all_data), f"Downloaded and cached data for {len(all_data)} symbols in category '{self.category}'.")
        except Exception as e:
            self.error.emit(str(e))

class ScheduledBatchDownloadWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(int, str)
    error = pyqtSignal(str)
    quota_update = pyqtSignal(str)

    def __init__(self, provider, api_key, category, ticker_manager, interval, start_date, end_date, batch_size, total_hours, start_time=None):
        super().__init__()
        self.provider = provider
        self.api_key = api_key
        self.category = category
        self.ticker_manager = ticker_manager
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date
        self.batch_size = batch_size
        self.total_hours = total_hours
        self.start_time = start_time

    def run(self):
        try:
            symbols = self.ticker_manager.tickers.get(self.category, [])
            total = len(symbols)
            if not symbols:
                self.error.emit(f"No symbols found for category '{self.category}'")
                return
            num_batches = (total + self.batch_size - 1) // self.batch_size
            interval_seconds = self.total_hours * 3600 / num_batches
            if self.start_time is None:
                start_time = datetime.datetime.utcnow()
            else:
                start_time = self.start_time
            for i in range(num_batches):
                now = datetime.datetime.utcnow()
                scheduled_time = start_time + datetime.timedelta(seconds=i * interval_seconds)
                wait_seconds = (scheduled_time - now).total_seconds()
                if wait_seconds > 0:
                    self.progress.emit(i, f"Waiting {wait_seconds/60:.2f} minutes for next batch...")
                    time.sleep(wait_seconds)
                batch = symbols[i*self.batch_size:(i+1)*self.batch_size]
                self.progress.emit(i, f"Downloading batch {i+1}/{num_batches} ({len(batch)} symbols) at {datetime.datetime.utcnow().strftime('%H:%M UTC')}")
                try:
                    if self.provider == 'twelvedata':
                        from src.data.twelvedata_downloader import TwelveDataDownloader
                        downloader = TwelveDataDownloader(self.api_key)
                        temp_category = '__temp_batch__'
                        self.ticker_manager.tickers[temp_category] = batch
                        downloader.batch_download_category_to_cache(
                            category=temp_category,
                            ticker_manager=self.ticker_manager,
                            interval=self.interval,
                            start_date=self.start_date,
                            end_date=self.end_date,
                            cache_dir=None,
                            batch_size=self.batch_size,
                            verbose=False
                        )
                        del self.ticker_manager.tickers[temp_category]
                    elif self.provider == 'yfinance':
                        # Use TickerManager's get_historical_data for yfinance
                        data = self.ticker_manager.get_historical_data(
                            batch,
                            self.start_date,
                            self.end_date,
                            interval=self.interval,
                            data_source='yfinance',
                            download_mode='sequential',
                            clean_data=True
                        )
                        # Save each DataFrame to cache
                        cache_dir = getattr(self.ticker_manager, 'cache_dir', None)
                        if cache_dir:
                            for ticker, df in data.items():
                                if df is not None and not df.empty:
                                    cache_file = cache_dir / f"{ticker}_{self.start_date}_{self.end_date}_{self.interval}_yfinance.csv"
                                    df.to_csv(cache_file)
                    elif self.provider == 'tiingo':
                        self.progress.emit(i, "Tiingo batch download not yet implemented.")
                    elif self.provider == 'alphavantage':
                        self.progress.emit(i, "Alpha Vantage batch download not yet implemented.")
                    else:
                        self.progress.emit(i, f"Batch download not supported for provider: {self.provider}")
                    self.quota_update.emit("update")
                except Exception as e:
                    self.error.emit(str(e))
                    return
            self.finished.emit(total, f"Scheduled batch download complete for {total} symbols in category '{self.category}'.")
        except Exception as e:
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
        self.log_queue = queue.Queue()
        # Set up a timer to poll the queue and update the log
        self.log_timer = QTimer()
        self.log_timer.timeout.connect(self.process_log_queue)
        self.log_timer.start(100)  # Check every 100 ms
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
        self.twelvedata_radio = QRadioButton('Twelve Data')
        self.alphavantage_radio = QRadioButton('Alpha Vantage')
        self.stooq_radio = QRadioButton('Stooq')
        self.yfinance_radio.setChecked(True)
        data_source_layout.addWidget(self.yfinance_radio)
        data_source_layout.addWidget(self.pdr_radio)
        data_source_layout.addWidget(self.twelvedata_radio)
        data_source_layout.addWidget(self.alphavantage_radio)
        data_source_layout.addWidget(self.stooq_radio)

        # Sub-source combo for pandas_datareader
        self.pdr_source_combo = QComboBox()
        self.pdr_source_combo.addItems(['stooq', 'av', 'fred', 'tiingo'])
        self.pdr_source_combo.setVisible(False)
        data_source_layout.addWidget(self.pdr_source_combo)

        # Stooq API Key input (for consistency/future use)
        self.stooq_api_label = QLabel('Stooq API Key:')
        self.stooq_api_input = QLineEdit()
        self.stooq_api_input.setPlaceholderText('Enter your Stooq API Key (not required)')
        self.stooq_api_label.setVisible(False)
        self.stooq_api_input.setVisible(False)
        data_source_layout.addWidget(self.stooq_api_label)
        data_source_layout.addWidget(self.stooq_api_input)

        # Tiingo API Key input
        self.tiingo_api_label = QLabel('Tiingo API Key:')
        self.tiingo_api_input = QLineEdit()
        self.tiingo_api_input.setPlaceholderText('Enter your Tiingo API Key')
        self.tiingo_api_label.setVisible(False)
        self.tiingo_api_input.setVisible(False)
        data_source_layout.addWidget(self.tiingo_api_label)
        data_source_layout.addWidget(self.tiingo_api_input)

        # Twelve Data API Key input
        self.twelvedata_api_label = QLabel('Twelve Data API Key:')
        self.twelvedata_api_input = QLineEdit()
        self.twelvedata_api_input.setPlaceholderText('Enter your Twelve Data API Key')
        self.twelvedata_api_label.setVisible(False)
        self.twelvedata_api_input.setVisible(False)
        data_source_layout.addWidget(self.twelvedata_api_label)
        data_source_layout.addWidget(self.twelvedata_api_input)

        # Alpha Vantage API Key input
        self.alphavantage_api_label = QLabel('Alpha Vantage API Key:')
        self.alphavantage_api_input = QLineEdit()
        self.alphavantage_api_input.setPlaceholderText('Enter your Alpha Vantage API Key')
        self.alphavantage_api_label.setVisible(False)
        self.alphavantage_api_input.setVisible(False)
        data_source_layout.addWidget(self.alphavantage_api_label)
        data_source_layout.addWidget(self.alphavantage_api_input)

        data_source_group.setLayout(data_source_layout)
        # Make scrollable
        data_source_scroll = QScrollArea()
        data_source_scroll.setWidgetResizable(True)
        data_source_scroll.setWidget(data_source_group)
        data_source_scroll.setStyleSheet('background-color: #222; color: #fff;')
        data_source_group.setStyleSheet('color: #fff; background-color: #222;')
        layout.addWidget(data_source_scroll)
        self.yfinance_radio.toggled.connect(lambda: self.set_data_source('yfinance'))
        self.pdr_radio.toggled.connect(lambda: self.set_data_source('pandas_datareader'))
        self.twelvedata_radio.toggled.connect(lambda: self.set_data_source('twelvedata'))
        self.alphavantage_radio.toggled.connect(lambda: self.set_data_source('alphavantage'))
        self.stooq_radio.toggled.connect(lambda: self.set_data_source('stooq'))

        # Show/hide the sub-source combo based on selection
        self.pdr_radio.toggled.connect(lambda checked: self.pdr_source_combo.setVisible(checked))

        # Show/hide the Tiingo, Twelve Data, and Alpha Vantage API key input based on selection
        def handle_pdr_sub_source_change(text):
            if text == 'tiingo':
                self.tiingo_api_label.setVisible(True)
                self.tiingo_api_input.setVisible(True)
                self.twelvedata_api_label.setVisible(False)
                self.twelvedata_api_input.setVisible(False)
                self.alphavantage_api_label.setVisible(False)
                self.alphavantage_api_input.setVisible(False)
                self.stooq_api_label.setVisible(False)
                self.stooq_api_input.setVisible(False)
                self.download_mode_combo.setCurrentText('sequential')
                self.download_mode_combo.setEnabled(False)
            elif text == 'twelvedata':
                self.tiingo_api_label.setVisible(False)
                self.tiingo_api_input.setVisible(False)
                self.twelvedata_api_label.setVisible(True)
                self.twelvedata_api_input.setVisible(True)
                self.alphavantage_api_label.setVisible(False)
                self.alphavantage_api_input.setVisible(False)
                self.stooq_api_label.setVisible(False)
                self.stooq_api_input.setVisible(False)
                self.download_mode_combo.setCurrentText('sequential')
                self.download_mode_combo.setEnabled(False)
            elif text == 'stooq':
                self.tiingo_api_label.setVisible(False)
                self.tiingo_api_input.setVisible(False)
                self.twelvedata_api_label.setVisible(False)
                self.twelvedata_api_input.setVisible(False)
                self.alphavantage_api_label.setVisible(False)
                self.alphavantage_api_input.setVisible(False)
                self.stooq_api_label.setVisible(True)
                self.stooq_api_input.setVisible(True)
                self.download_mode_combo.setEnabled(True)
            else:
                self.tiingo_api_label.setVisible(False)
                self.tiingo_api_input.setVisible(False)
                self.twelvedata_api_label.setVisible(False)
                self.twelvedata_api_input.setVisible(False)
                self.alphavantage_api_label.setVisible(False)
                self.alphavantage_api_input.setVisible(False)
                self.stooq_api_label.setVisible(False)
                self.stooq_api_input.setVisible(False)
                self.download_mode_combo.setEnabled(True)
        self.pdr_source_combo.currentTextChanged.connect(handle_pdr_sub_source_change)

        # Show/hide API key fields for top-level selections
        self.twelvedata_radio.toggled.connect(
            lambda checked: (
                self.twelvedata_api_label.setVisible(checked),
                self.twelvedata_api_input.setVisible(checked),
                self.pdr_source_combo.setVisible(False),
                self.tiingo_api_label.setVisible(False),
                self.tiingo_api_input.setVisible(False),
                self.alphavantage_api_label.setVisible(False),
                self.alphavantage_api_input.setVisible(False),
                self.stooq_api_label.setVisible(False),
                self.stooq_api_input.setVisible(False),
                self.download_mode_combo.setCurrentText('sequential'),
                self.download_mode_combo.setEnabled(False)
            )
        )
        self.alphavantage_radio.toggled.connect(
            lambda checked: (
                self.alphavantage_api_label.setVisible(checked),
                self.alphavantage_api_input.setVisible(checked),
                self.pdr_source_combo.setVisible(False),
                self.tiingo_api_label.setVisible(False),
                self.tiingo_api_input.setVisible(False),
                self.twelvedata_api_label.setVisible(False),
                self.twelvedata_api_input.setVisible(False),
                self.stooq_api_label.setVisible(False),
                self.stooq_api_input.setVisible(False),
                self.download_mode_combo.setCurrentText('sequential'),
                self.download_mode_combo.setEnabled(False)
            )
        )

        # --- Top Controls (Category/Search) ---
        top_controls = QHBoxLayout()
        category_label = QLabel('Category:')
        self.category_combo = QComboBox()
        # TickerManager will be created after DB settings applied
        self.category_combo.currentTextChanged.connect(self.update_ticker_list)
        top_controls.addWidget(category_label)
        top_controls.addWidget(self.category_combo)
        
        # Add Instruments button
        self.add_instruments_btn = QPushButton('Add Instruments')
        self.add_instruments_btn.clicked.connect(self.add_instruments_categories)
        top_controls.addWidget(self.add_instruments_btn)

        # Add button to fetch Stooq tickers in top controls
        self.add_stooq_btn = QPushButton('Add Stooq Tickers')
        self.add_stooq_btn.clicked.connect(self.fetch_and_add_stooq_tickers)
        top_controls.addWidget(self.add_stooq_btn)

        # Add button to open and process CSV files from hard drive
        self.open_csv_btn = QPushButton('Open CSV Files')
        self.open_csv_btn.setToolTip('Open and process one or more CSV/TXT files from your hard drive')
        self.open_csv_btn.clicked.connect(self.open_and_process_csv_files)
        top_controls.addWidget(self.open_csv_btn)

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
        # Make scrollable
        date_scroll = QScrollArea()
        date_scroll.setWidgetResizable(True)
        date_scroll.setWidget(date_group)
        date_scroll.setStyleSheet('background-color: #222; color: #fff;')
        date_group.setStyleSheet('color: #fff; background-color: #222;')
        right_layout.addWidget(date_scroll)

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
        
        # Add batch size selector (greyscale, smaller font for two-digit values)
        batch_size_layout = QHBoxLayout()
        batch_size_label = QLabel("Batch Size:")
        batch_size_label.setStyleSheet('''
            QLabel {
                font-size: 10px;
                font-weight: bold;
                padding: 4px 8px;
                margin-right: 8px;
                background-color: #f5f5f5;
                border: 2px solid #757575;
                border-radius: 6px;
                color: #212121;
            }
        ''')
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 10)
        self.batch_size_spin.setValue(3)
        self.batch_size_spin.setEnabled(False)  # Disabled by default
        self.batch_size_spin.setMinimumWidth(40)
        self.batch_size_spin.setMinimumHeight(24)
        self.batch_size_spin.setStyleSheet('''
            QSpinBox {
                font-size: 10px;
                font-weight: bold;
                padding: 4px 8px;
                margin-left: 8px;
                background-color: #fafafa;
                border: 2px solid #757575;
                border-radius: 6px;
                color: #212121;
            }
        ''')
        batch_size_layout.addWidget(batch_size_label)
        batch_size_layout.addWidget(self.batch_size_spin)
        processing_layout.addLayout(batch_size_layout)
        
        processing_group.setLayout(processing_layout)
        # Make scrollable
        processing_scroll = QScrollArea()
        processing_scroll.setWidgetResizable(True)
        processing_scroll.setWidget(processing_group)
        processing_scroll.setStyleSheet('background-color: #222; color: #fff;')
        processing_group.setStyleSheet('color: #fff; background-color: #222;')
        right_layout.addWidget(processing_scroll)

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
        # Make scrollable
        cleaning_scroll = QScrollArea()
        cleaning_scroll.setWidgetResizable(True)
        cleaning_scroll.setWidget(cleaning_group)
        cleaning_scroll.setStyleSheet('background-color: #222; color: #fff;')
        cleaning_group.setStyleSheet('color: #fff; background-color: #222;')
        right_layout.addWidget(cleaning_scroll)

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

        # Batch Download button (smaller width, font fits button)
        self.batch_download_btn = QPushButton('Batch Download Category to Cache')
        self.batch_download_btn.clicked.connect(self.batch_download_category_to_cache_gui)
        self.batch_download_btn.setFixedWidth(220)
        self.batch_download_btn.setStyleSheet('''
            QPushButton {
                font-size: 10px;
                font-weight: bold;
                color: #000000;
                background-color: #e0e0e0;
                border: 2px solid #757575;
                border-radius: 8px;
                padding: 8px 8px;
                margin: 10px 0px;
            }
            QPushButton:hover, QPushButton:focus {
                background-color: #bdbdbd;
                color: #000000;
            }
        ''')
        right_layout.addWidget(self.batch_download_btn)

        self.scheduled_batch_btn = QPushButton('Scheduled Batch Download (23h)')
        self.scheduled_batch_btn.clicked.connect(self.scheduled_batch_download_gui)
        right_layout.addWidget(self.scheduled_batch_btn)

        content_layout.addWidget(right_panel)
        layout.addLayout(content_layout)

        # Initialize ticker list
        # self.update_ticker_list()

        # Connect pdr_source_combo to set pdr_sub_source
        self.pdr_source_combo.currentTextChanged.connect(
            lambda text: setattr(self, 'pdr_sub_source', text)
        )

        # Update download mode combo based on data source
        if self.data_source == 'pandas_datareader' and getattr(self, 'pdr_sub_source', None) == 'tiingo':
            self.download_mode_combo.setCurrentText('sequential')
            self.download_mode_combo.setEnabled(False)
        else:
            self.download_mode_combo.setEnabled(True)

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
        elif source == 'twelvedata' and self.twelvedata_radio.isChecked():
            self.data_source = 'twelvedata'
            self.pdr_sub_source = None
        elif source == 'alphavantage' and self.alphavantage_radio.isChecked():
            self.data_source = 'alphavantage'
            self.pdr_sub_source = None
        elif source == 'stooq' and self.stooq_radio.isChecked():
            self.data_source = 'stooq'
            self.pdr_sub_source = None
        if self.data_source == 'twelvedata':
            api_key = self.twelvedata_api_input.text().strip()
            if api_key:
                os.environ['TWELVEDATA_API_KEY'] = api_key

    def download_data(self):
        if not self.selected_tickers:
            QMessageBox.warning(self, "Warning", "Please select at least one ticker")
            return
        if not self.ticker_manager:
            QMessageBox.warning(self, "Warning", "Please apply database settings first.")
            return

        # --- Set Tiingo API key if needed ---
        if self.data_source == 'pandas_datareader' and getattr(self, 'pdr_sub_source', None) == 'tiingo':
            api_key = self.tiingo_api_input.text().strip()
            if api_key:
                os.environ['TIINGO_API_KEY'] = api_key

        try:
            # Get date range from date pickers
            start_date = self.start_date.date().toString('yyyy-MM-dd')
            end_date = self.end_date.date().toString('yyyy-MM-dd')

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
            status_text = f"Downloading data using {processing_mode} mode with {self.interval_combo.currentText()} interval"
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
                processing_mode,
                cleaning_options,
                self.interval_combo.currentText(),
                processing_mode,
                batch_size if processing_mode == 'batch' else None,
                self.data_source,
                getattr(self, 'pdr_sub_source', None)
            )
            self.worker.error.connect(self.log_message)
            self.worker.finished.connect(self.log_message)
            self.worker.progress.connect(self.update_progress)
            self.worker.ticker_data.connect(self.on_ticker_data)
            self.data_table.setRowCount(0)
            self.data_table.setColumnCount(0)
            self.worker.start()

        except Exception as e:
            logger.error(f"Error starting download: {e}")
            QMessageBox.critical(self, "Error", f"Failed to start download: {str(e)}")

    def log_message(self, msg):
        self.log_text.append(str(msg))

    def on_download_finished(self, data):
        """Handle successful data download"""
        self.current_data = data
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Downloaded data for {len(data)} tickers")
        self.download_btn.setEnabled(True)
        self.export_btn.setEnabled(True)  # Enable export button after successful download
        logger.info(f"Successfully downloaded data for {len(data)} tickers")
        self.update_quota_label()

    def on_download_error(self, error_msg):
        """Handle download error"""
        self.progress_bar.setVisible(False)
        self.status_label.setText("Download failed")
        self.download_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Failed to download data: {error_msg}")
        logger.error(f"Data download failed: {error_msg}")
        self.update_quota_label()

    def export_data(self):
        """Export downloaded data to selected format, or process loaded files if present"""
        if not self.ticker_manager:
            QMessageBox.warning(self, "Warning", "Please apply database settings first.")
            return
        if hasattr(self, 'raw_data') and self.raw_data:
            # Process loaded files before exporting
            processed = self.process_loaded_files()
            if not processed:
                return
        if not self.current_data:
            QMessageBox.warning(self, "Warning", "No data to export. Please download or process data first.")
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

    def process_loaded_files(self):
        """
        Clean and process files loaded into self.raw_data, store in self.current_data, and show cleaning summary.
        """
        if not self.ticker_manager:
            QMessageBox.warning(self, 'Warning', 'Please apply database settings first.')
            return False
        if not hasattr(self, 'raw_data') or not self.raw_data:
            QMessageBox.warning(self, 'No Data', 'No files loaded to process.')
            return False
        loaded = 0
        tickers = []
        cleaning_summaries = []
        failed_tickers = []
        self.current_data = {}
        # Clear the live data table before processing
        self.data_table.setRowCount(0)
        self.data_table.setColumnCount(0)
        for ticker, df in self.raw_data.items():
            try:
                stats = {}
                stats['original_rows'] = len(df)
                stats['original_missing'] = int(df.isnull().sum().sum())
                stats['original_duplicates'] = int(df.duplicated().sum())
                cleaned_df = self.ticker_manager.clean_data(df)
                stats['cleaned_rows'] = len(cleaned_df)
                stats['cleaned_missing'] = int(cleaned_df.isnull().sum().sum())
                stats['cleaned_duplicates'] = int(cleaned_df.duplicated().sum())
                summary = (
                    f"{ticker}: "
                    f"Rows: {stats['original_rows']}→{stats['cleaned_rows']}, "
                    f"Missing: {stats['original_missing']}→{stats['cleaned_missing']}, "
                    f"Duplicates: {stats['original_duplicates']}→{stats['cleaned_duplicates']}"
                )
                cleaning_summaries.append(summary)
                self.current_data[ticker] = cleaned_df
                loaded += 1
                tickers.append(ticker)
                # Update live preview for this ticker (show all rows)
                self.status_label.setText(f"Previewing cleaned data for {ticker}")
                preview_df = cleaned_df
                if not preview_df.empty:
                    # Always show all columns seen so far
                    current_headers = [self.data_table.horizontalHeaderItem(i).text() for i in range(self.data_table.columnCount())] if self.data_table.columnCount() > 0 else []
                    all_columns = set(current_headers[1:] if current_headers else []) | set(preview_df.columns)
                    all_columns = ['Ticker'] + sorted(all_columns)
                    # If new columns are found, update the table structure
                    if self.data_table.columnCount() != len(all_columns) or current_headers != all_columns:
                        self.data_table.setColumnCount(len(all_columns))
                        self.data_table.setHorizontalHeaderLabels(all_columns)
                    # Add all rows for this ticker
                    for i in range(len(preview_df)):
                        row = self.data_table.rowCount()
                        self.data_table.insertRow(row)
                        self.data_table.setItem(row, 0, QTableWidgetItem(ticker))
                        for col_idx, col_name in enumerate(all_columns[1:], start=1):
                            val = str(preview_df.iloc[i][col_name]) if col_name in preview_df.columns else ''
                            self.data_table.setItem(row, col_idx, QTableWidgetItem(val))
            except Exception as e:
                logger.error(f"Failed to clean/process {ticker}: {e}")
                self.log_text.append(f"Failed to clean/process {ticker}: {e}")
                cleaning_summaries.append(f"{ticker}: Cleaning failed ({e})")
                failed_tickers.append(ticker)
        msg = (
            f'Processed {loaded} files. Tickers: {", ".join(tickers)}\n\n'
            f'Failed to process {len(failed_tickers)} files: {", ".join(failed_tickers)}\n\n' if failed_tickers else ''
            f'Cleaning summary:\n' + '\n'.join(cleaning_summaries)
        )
        QMessageBox.information(self, 'Processing Summary', msg)
        return loaded > 0

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
        self.progress_bar.setValue(value)
        if message:
            self.log_text.append(str(message))
        else:
            self.log_text.append(str(value))
        logger.info(f"Progress update: {value}% - {message}")

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

    def process_log_queue(self):
        while not self.log_queue.empty():
            msg = self.log_queue.get()
            self.log_text.append(str(msg))

    def add_instruments_categories(self):
        api_key = self.twelvedata_api_input.text().strip()
        if not api_key:
            QMessageBox.warning(self, "API Key Required", "Please enter your Twelve Data API key in the input field.")
            return
        # Define endpoints and category names
        endpoints = [
            ("Forex", f"https://api.twelvedata.com/forex_pairs?apikey={api_key}"),
            ("Crypto", f"https://api.twelvedata.com/cryptocurrencies?apikey={api_key}"),
            ("Commodities", f"https://api.twelvedata.com/commodities?apikey={api_key}")
        ]
        added = []
        for category_name, url in endpoints:
            try:
                response = requests.get(url)
                data = response.json()
                if 'data' in data:
                    df = pd.DataFrame(data['data'])
                    if 'symbol' in df.columns:
                        symbols = df['symbol'].dropna().astype(str).str.strip().tolist()
                        self.ticker_manager.tickers[category_name] = symbols
                        if category_name not in self.ticker_manager.categories:
                            self.ticker_manager.categories.append(category_name)
                            self.category_combo.addItem(category_name)
                        added.append((category_name, len(symbols)))
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to fetch {category_name} instruments: {e}")
        if added:
            msg = "\n".join([f"Loaded {count} symbols into category '{cat}'." for cat, count in added])
            QMessageBox.information(self, "Success", msg)
        else:
            QMessageBox.warning(self, "No Data", "No instruments were added.")

    def batch_download_category_to_cache_gui(self):
        if not self.ticker_manager:
            QMessageBox.warning(self, "Warning", "Please apply database settings first.")
            return
        category = self.category_combo.currentText()
        interval = self.interval_combo.currentText()
        start_date = self.start_date.date().toString('yyyy-MM-dd')
        end_date = self.end_date.date().toString('yyyy-MM-dd')
        api_key = self.twelvedata_api_input.text().strip()
        if not api_key:
            QMessageBox.warning(self, "API Key Required", "Please enter your Twelve Data API key in the input field.")
            return
        self.status_label.setText("Batch downloading category to cache...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.batch_download_btn.setEnabled(False)
        self.worker = BatchDownloadWorker(api_key, category, self.ticker_manager, interval, start_date, end_date)
        self.worker.progress.connect(self.on_batch_download_progress)
        self.worker.finished.connect(self.on_batch_download_finished)
        self.worker.error.connect(self.on_batch_download_error)
        self.worker.start()

    def on_batch_download_progress(self, value, message):
        self.status_label.setText(message)
        self.log_text.append(message)

    def on_batch_download_finished(self, count, message):
        self.status_label.setText(message)
        self.log_text.append(message)
        self.progress_bar.setVisible(False)
        self.batch_download_btn.setEnabled(True)
        QMessageBox.information(self, "Batch Download Complete", message)
        self.update_quota_label()

    def on_batch_download_error(self, error_msg):
        self.status_label.setText("Batch download failed.")
        self.log_text.append(f"Error: {error_msg}")
        self.progress_bar.setVisible(False)
        self.batch_download_btn.setEnabled(True)
        if 'quota' in error_msg.lower() or 'limit' in error_msg.lower():
            QMessageBox.warning(self, "Twelve Data API Quota Exceeded", "You have reached your Twelve Data API quota. Please wait for your quota to reset or upgrade your plan.")
        else:
            QMessageBox.critical(self, "Error", f"Batch download failed: {error_msg}")
        self.update_quota_label()

    def update_quota_label(self):
        # Implementation of update_quota_label method
        pass

    def scheduled_batch_download_gui(self):
        if not self.ticker_manager:
            QMessageBox.warning(self, "Warning", "Please apply database settings first.")
            return
        category = self.category_combo.currentText()
        interval = self.interval_combo.currentText()
        start_date = self.start_date.date().toString('yyyy-MM-dd')
        end_date = self.end_date.date().toString('yyyy-MM-dd')
        provider = self.data_source
        if provider == 'twelvedata':
            api_key = self.twelvedata_api_input.text().strip()
        elif provider == 'tiingo':
            api_key = self.tiingo_api_input.text().strip()
        elif provider == 'alphavantage':
            api_key = self.alphavantage_api_input.text().strip()
        else:
            api_key = ''
        if not api_key:
            QMessageBox.warning(self, "API Key Required", f"Please enter your {provider.title()} API key in the input field.")
            return
        # Prompt for batch size and total hours
        max_batch = len(self.ticker_manager.tickers.get(category, []))
        batch_size, ok1 = QInputDialog.getInt(self, "Batch Size", f"Enter batch size (max {max_batch}):", min=1, max=max_batch, value=8)
        if not ok1:
            return
        total_hours, ok2 = QInputDialog.getInt(self, "Total Hours", "Enter total period in hours:", min=1, max=24, value=23)
        if not ok2:
            return
        self.status_label.setText("Scheduled batch downloading...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.scheduled_worker = ScheduledBatchDownloadWorker(
            provider, api_key, category, self.ticker_manager, interval, start_date, end_date, batch_size, total_hours
        )
        self.scheduled_worker.progress.connect(self.on_batch_download_progress)
        self.scheduled_worker.finished.connect(self.on_batch_download_finished)
        self.scheduled_worker.error.connect(self.on_batch_download_error)
        self.scheduled_worker.quota_update.connect(lambda _: self.update_quota_label())
        self.scheduled_worker.start()

    def fetch_and_add_stooq_tickers(self):
        """
        Adds predefined Stooq Forex, Commodities, and Stocks categories with example tickers to the category combo box.
        """
        if not self.ticker_manager:
            QMessageBox.warning(self, "Warning", "Please apply database settings first.")
            return
        # Predefined example tickers
        stooq_forex = [
            'EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD',
            'EURJPY', 'GBPJPY', 'EURGBP', 'USDHKD', 'USDSEK', 'USDSGD', 'USDNOK', 'USDZAR', 'USDMXN'
        ]
        stooq_commodities = [
            'GOLD', 'SILVER', 'OIL', 'COPPER', 'PLATINUM', 'PALLADIUM', 'NGAS', 'CORN', 'WHEAT', 'SOYB'
        ]
        stooq_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'DIS'
        ]
        # Add categories and tickers
        if stooq_forex:
            self.ticker_manager.tickers['Stooq Forex'] = stooq_forex
            if 'Stooq Forex' not in self.ticker_manager.categories:
                self.ticker_manager.categories.append('Stooq Forex')
                self.category_combo.addItem('Stooq Forex')
        if stooq_commodities:
            self.ticker_manager.tickers['Stooq Commodities'] = stooq_commodities
            if 'Stooq Commodities' not in self.ticker_manager.categories:
                self.ticker_manager.categories.append('Stooq Commodities')
                self.category_combo.addItem('Stooq Commodities')
        if stooq_stocks:
            self.ticker_manager.tickers['Stooq Stocks'] = stooq_stocks
            if 'Stooq Stocks' not in self.ticker_manager.categories:
                self.ticker_manager.categories.append('Stooq Stocks')
                self.category_combo.addItem('Stooq Stocks')
        msg = (
            f"Added categories and example tickers:\n"
            f"Stooq Forex: {', '.join(stooq_forex)}\n"
            f"Stooq Commodities: {', '.join(stooq_commodities)}\n"
            f"Stooq Stocks: {', '.join(stooq_stocks)}"
        )
        QMessageBox.information(self, "Stooq Tickers", msg)

    def open_and_process_csv_files(self):
        import os
        import pandas as pd
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        files, _ = QFileDialog.getOpenFileNames(self, 'Select CSV or TXT Files', '', 'CSV or TXT Files (*.csv *.txt)')
        if not files:
            return
        loaded = 0
        failed = 0
        tickers = []
        failed_files = []
        self.raw_data = {}
        for file_path in files:
            try:
                # Try both comma and semicolon delimiters for robustness
                try:
                    df = pd.read_csv(file_path, delimiter=',')
                    if df.shape[1] == 1:
                        df = pd.read_csv(file_path, delimiter=';')
                except Exception:
                    df = pd.read_csv(file_path, delimiter=';')
                ticker = os.path.splitext(os.path.basename(file_path))[0]
                self.raw_data[ticker] = df
                loaded += 1
                tickers.append(ticker)
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                self.log_text.append(f"Failed to load {file_path}: {e}")
                failed += 1
                failed_files.append(os.path.basename(file_path))
        msg = (
            f'Loaded {loaded} files. Tickers: {", ".join(tickers)}\n'
            f'Failed to load {failed} files: {", ".join(failed_files)}' if failed else ''
        )
        QMessageBox.information(self, 'File Load Summary', msg)
        if loaded:
            self.export_btn.setEnabled(True)
        else:
            self.export_btn.setEnabled(False)

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