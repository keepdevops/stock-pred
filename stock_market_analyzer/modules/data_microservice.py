import yfinance as yf
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import traceback
from PyQt6.QtCore import QObject, pyqtSignal, QThread

class DataMicroservice(QObject):
    """Microservice for handling stock data operations."""
    
    # Signals
    data_loaded = pyqtSignal(str, pd.DataFrame)  # symbol, data
    data_error = pyqtSignal(str, str)  # symbol, error message
    progress_updated = pyqtSignal(int)  # progress percentage
    
    def __init__(self):
        """Initialize the data microservice."""
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.worker_thread = None
        self.logger.info("Data microservice initialized")
    
    def fetch_stock_data(self, symbol: str, start_date: str = None, end_date: str = None):
        """Fetch stock data for the given symbol."""
        try:
            # Create worker thread
            self.worker_thread = DataWorker(symbol, start_date, end_date)
            self.worker_thread.data_loaded.connect(self._on_data_loaded)
            self.worker_thread.data_error.connect(self._on_data_error)
            self.worker_thread.progress_updated.connect(self._on_progress_updated)
            
            # Start worker thread
            self.worker_thread.start()
            
        except Exception as e:
            self.logger.error(f"Error starting data fetch for {symbol}: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            self.data_error.emit(symbol, str(e))
    
    def _on_data_loaded(self, symbol: str, data: pd.DataFrame):
        """Handle successful data loading."""
        try:
            if data is not None and not data.empty:
                self.logger.info(f"Successfully loaded data for {symbol}")
                self.data_loaded.emit(symbol, data)
            else:
                error_msg = f"No data available for {symbol}"
                self.logger.warning(error_msg)
                self.data_error.emit(symbol, error_msg)
                
        except Exception as e:
            self.logger.error(f"Error processing data for {symbol}: {str(e)}")
            self.data_error.emit(symbol, str(e))
    
    def _on_data_error(self, symbol: str, error_msg: str):
        """Handle data loading errors."""
        self.logger.error(f"Error loading data for {symbol}: {error_msg}")
        self.data_error.emit(symbol, error_msg)
    
    def _on_progress_updated(self, progress: int):
        """Handle progress updates."""
        self.progress_updated.emit(progress)

class DataWorker(QThread):
    """Worker thread for fetching stock data."""
    
    # Signals
    data_loaded = pyqtSignal(str, pd.DataFrame)
    data_error = pyqtSignal(str, str)
    progress_updated = pyqtSignal(int)
    
    def __init__(self, symbol: str, start_date: str = None, end_date: str = None):
        """Initialize the data worker."""
        super().__init__()
        self.symbol = symbol
        self.start_date = start_date or (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.logger = logging.getLogger(__name__)
    
    def run(self):
        """Run the data fetching process."""
        try:
            self.logger.info(f"Fetching data for {self.symbol}")
            self.progress_updated.emit(10)
            
            # Download data from yfinance
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(
                start=self.start_date,
                end=self.end_date,
                interval='1d'
            )
            
            self.progress_updated.emit(50)
            
            if data.empty:
                error_msg = f"No data available for {self.symbol}"
                self.logger.warning(error_msg)
                self.data_error.emit(self.symbol, error_msg)
                return
            
            # Reset index to make date a column
            data = data.reset_index()
            
            # Rename columns to match our schema
            data = data.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Adj Close': 'adj_close'
            })
            
            # Convert date column to datetime
            data['date'] = pd.to_datetime(data['date'])
            
            self.progress_updated.emit(90)
            
            # Emit the loaded data
            self.data_loaded.emit(self.symbol, data)
            self.progress_updated.emit(100)
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {self.symbol}: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            self.data_error.emit(self.symbol, str(e)) 