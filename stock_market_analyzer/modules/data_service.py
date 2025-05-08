from typing import Dict, Optional, Any, List
import pandas as pd
from PyQt6.QtCore import QObject, pyqtSignal
import logging
import traceback
import os
from queue import Queue
from threading import Thread, Event
import json
from datetime import datetime
import queue
import io

class DataServiceRequest:
    """Represents a request to the data service."""
    def __init__(self, action: str, params: Dict[str, Any]):
        self.action = action
        self.params = params
        self.id = f"{action}_{datetime.now().timestamp()}"

class DataServiceResponse:
    """Represents a response from the data service."""
    def __init__(self, request_id: str, success: bool, data: Any = None, error: str = None):
        self.request_id = request_id
        self.success = success
        self.data = data
        self.error = error

class DataService(QObject):
    """Microservice for handling all data-related operations."""
    
    # Define signals for async communication
    data_loaded = pyqtSignal(str, object)  # symbol, data
    data_updated = pyqtSignal(str, object)  # symbol, data
    data_error = pyqtSignal(str)  # error message
    import_complete = pyqtSignal(str, object)  # file_path, data
    import_error = pyqtSignal(str)  # error message
    source_changed = pyqtSignal(str)  # new source
    service_status = pyqtSignal(str)  # service status updates
    
    def __init__(self, data_loader):
        """Initialize the data service."""
        super().__init__()
        self.data_loader = data_loader
        self.logger = logging.getLogger(__name__)
        
        # Service state
        self.cached_data: Dict[str, pd.DataFrame] = {}
        self.current_symbol: Optional[str] = None
        self._source = getattr(self.data_loader, 'source', 'yahoo')
        self._data_dir = getattr(self.data_loader, 'data_dir', os.path.join(os.getcwd(), 'data'))
        
        # Service queues and control
        self.request_queue = Queue()
        self.response_queue = Queue()
        self.stop_event = Event()
        
        # Start service worker thread
        self.worker_thread = Thread(target=self._process_requests, daemon=True)
        self.worker_thread.start()
        
        self.logger.info("Data service initialized and worker thread started")
        
    def _process_requests(self):
        """Process requests in the background thread."""
        while not self.stop_event.is_set():
            try:
                request = self.request_queue.get(timeout=1)
                if request is None:
                    continue
                    
                self.logger.debug(f"Processing request: {request.action}")
                
                if request.action == "load_data":
                    self._handle_load_data(request)
                elif request.action == "import_data":
                    self._handle_import_data(request)
                elif request.action == "update_data":
                    self._handle_update_data(request)
                elif request.action == "set_source":
                    self._handle_set_source(request)
                elif request.action == "set_data_dir":
                    self._handle_set_data_dir(request)
                    
            except queue.Empty:
                # This is expected when there are no requests to process
                continue
            except Exception as e:
                self.logger.error(f"Error processing request: {str(e)}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                
    def _handle_load_data(self, request: DataServiceRequest):
        """Handle data loading requests."""
        try:
            symbol = request.params.get('symbol')
            if not symbol:
                self.data_error.emit("No symbol provided")
                return

            # Load data from the data loader
            data = self.data_loader.load_data(symbol)
            
            if data is None or data.empty:
                self.data_error.emit(f"No data available for {symbol}")
                return
                
            # Ensure data is a DataFrame
            if isinstance(data, str):
                try:
                    data = pd.read_csv(io.StringIO(data))
                except Exception as e:
                    self.data_error.emit(f"Error converting data to DataFrame: {str(e)}")
                    return
            
            # Validate and process the data
            processed_data = self._validate_and_process_data(data)
            if processed_data is None or processed_data.empty:
                self.data_error.emit(f"Invalid data format for {symbol}")
                return
            
            # Cache the data
            self.cached_data[symbol] = processed_data
            self.current_symbol = symbol
            
            # Emit the data as a DataFrame
            self.data_loaded.emit(symbol, processed_data)
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            self.data_error.emit(f"Error loading data: {str(e)}")
            
    def _handle_import_data(self, request: DataServiceRequest):
        """Handle import data request."""
        try:
            file_path = request.params["file_path"]
            format_type = request.params["format_type"]
            
            if format_type.upper() == "CSV":
                data = pd.read_csv(file_path)
            elif format_type.upper() == "JSON":
                data = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported format type: {format_type}")
                
            # Validate and process data
            self._validate_and_process_data(data)
            
            # Emit success signal
            self.import_complete.emit(file_path, data)
            
        except Exception as e:
            self.logger.error(f"Error importing data: {str(e)}")
            self.import_error.emit(str(e))
            
    def _validate_and_process_data(self, data: pd.DataFrame):
        """Validate and process imported data."""
        try:
            # Check if data is empty
            if data is None or data.empty:
                raise ValueError("Empty dataset received")
                
            # Convert column names to lowercase for consistency
            data.columns = data.columns.str.lower()
            
            # Validate data structure
            required_columns = ["date", "open", "high", "low", "close", "volume"]
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
                
            # Convert date column to datetime if needed
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                
            # Sort by date
            data = data.sort_values('date')
            
            # Calculate technical indicators if not present
            if 'rsi' not in data.columns:
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                data['rsi'] = 100 - (100 / (1 + rs))
                
            if 'ma5' not in data.columns:
                data['ma5'] = data['close'].rolling(window=5).mean()
                
            if 'ma20' not in data.columns:
                data['ma20'] = data['close'].rolling(window=20).mean()
                
            return data
            
        except Exception as e:
            self.logger.error(f"Error validating data: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
            
    def _handle_update_data(self, request: DataServiceRequest):
        """Handle update data request."""
        try:
            symbol = request.params["symbol"]
            new_data = request.params["data"]
            
            if symbol in self.cached_data:
                # Merge new data with existing data
                existing_data = self.cached_data[symbol]
                updated_data = pd.concat([existing_data, new_data]).drop_duplicates()
                self.cached_data[symbol] = updated_data
                self.data_updated.emit(symbol, updated_data)
            else:
                # Store new data
                self.cached_data[symbol] = new_data
                self.data_updated.emit(symbol, new_data)
                
        except Exception as e:
            self.logger.error(f"Error updating data: {str(e)}")
            self.data_error.emit(str(e))
            
    def _handle_set_source(self, request: DataServiceRequest):
        """Handle set source request."""
        try:
            source = request.params["source"]
            self._source = source
            if hasattr(self.data_loader, 'source'):
                self.data_loader.source = source
            self.source_changed.emit(source)
        except Exception as e:
            self.logger.error(f"Error setting source: {str(e)}")
            self.data_error.emit(str(e))
            
    def _handle_set_data_dir(self, request: DataServiceRequest):
        """Handle set data directory request."""
        try:
            data_dir = request.params["data_dir"]
            self._data_dir = data_dir
            if hasattr(self.data_loader, 'data_dir'):
                self.data_loader.data_dir = data_dir
        except Exception as e:
            self.logger.error(f"Error setting data directory: {str(e)}")
            self.data_error.emit(str(e))
            
    # Public API methods
    def load_data(self, symbol: str) -> None:
        """Queue a load data request."""
        request = DataServiceRequest("load_data", {"symbol": symbol})
        self.request_queue.put(request)
        
    def import_data(self, file_path: str, format_type: str) -> None:
        """Queue an import data request."""
        request = DataServiceRequest("import_data", {
            "file_path": file_path,
            "format_type": format_type
        })
        self.request_queue.put(request)
        
    def update_data(self, symbol: str, new_data: pd.DataFrame) -> None:
        """Queue an update data request."""
        request = DataServiceRequest("update_data", {
            "symbol": symbol,
            "data": new_data
        })
        self.request_queue.put(request)
        
    @property
    def source(self) -> str:
        """Get the current data source."""
        return self._source
        
    @source.setter
    def source(self, value: str) -> None:
        """Queue a set source request."""
        request = DataServiceRequest("set_source", {"source": value})
        self.request_queue.put(request)
        
    @property
    def data_dir(self) -> str:
        """Get the current data directory."""
        return self._data_dir
        
    @data_dir.setter
    def data_dir(self, value: str) -> None:
        """Queue a set data directory request."""
        request = DataServiceRequest("set_data_dir", {"data_dir": value})
        self.request_queue.put(request)
        
    def get_available_sources(self) -> List[str]:
        """Get list of available data sources."""
        try:
            if hasattr(self.data_loader, 'get_available_sources'):
                return self.data_loader.get_available_sources()
            return ['yahoo', 'alpha_vantage', 'local']
        except Exception as e:
            self.logger.error(f"Error getting available sources: {str(e)}")
            return ['yahoo']
            
    def get_current_data(self) -> Optional[pd.DataFrame]:
        """Get the current data from cache."""
        if self.current_symbol and self.current_symbol in self.cached_data:
            return self.cached_data[self.current_symbol]
        return None
        
    def clear_cache(self) -> None:
        """Clear the data cache."""
        self.cached_data.clear()
        self.current_symbol = None
        
    def get_symbols(self) -> list:
        """Get list of available symbols."""
        return list(self.cached_data.keys())
        
    def shutdown(self):
        """Shutdown the service."""
        self.logger.info("Shutting down data service...")
        self.stop_event.set()
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)
        self.logger.info("Data service shutdown complete") 