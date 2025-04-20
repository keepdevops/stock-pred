import pandas as pd
from typing import Dict, Any, List, Optional
import threading
from queue import Queue
import logging

class DataManager:
    """Manages shared data between tabs."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(DataManager, cls).__new__(cls)
                    cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the data manager."""
        self.logger = logging.getLogger(__name__)
        self._data_cache = {}
        self._data_queue = Queue()
        self._listeners = {}
        self._lock = threading.Lock()
        
    def add_data(self, source: str, data: pd.DataFrame, metadata: Dict[str, Any] = None):
        """Add new data to the manager.
        
        Args:
            source: The source of the data (e.g., 'ImportTab')
            data: The pandas DataFrame containing the data
            metadata: Optional metadata about the data
        """
        with self._lock:
            # Store the data
            self._data_cache[source] = {
                'data': data,
                'metadata': metadata or {},
                'timestamp': pd.Timestamp.now()
            }
            
            # Notify listeners
            self._notify_listeners(source, data, metadata)
            
    def get_data(self, source: str) -> Optional[Dict[str, Any]]:
        """Get data from a specific source.
        
        Args:
            source: The source of the data
            
        Returns:
            Dictionary containing the data and metadata, or None if not found
        """
        return self._data_cache.get(source)
    
    def register_listener(self, listener_id: str, callback):
        """Register a listener for data updates.
        
        Args:
            listener_id: Unique identifier for the listener
            callback: Function to call when data is updated
        """
        with self._lock:
            self._listeners[listener_id] = callback
            
    def unregister_listener(self, listener_id: str):
        """Unregister a listener.
        
        Args:
            listener_id: The listener to unregister
        """
        with self._lock:
            self._listeners.pop(listener_id, None)
            
    def _notify_listeners(self, source: str, data: pd.DataFrame, metadata: Dict[str, Any]):
        """Notify all registered listeners of new data.
        
        Args:
            source: The source of the data
            data: The new data
            metadata: Associated metadata
        """
        for listener_id, callback in self._listeners.items():
            try:
                callback(source, data, metadata)
            except Exception as e:
                self.logger.error(f"Error notifying listener {listener_id}: {str(e)}")
                
    def clear_cache(self):
        """Clear all cached data."""
        with self._lock:
            self._data_cache.clear() 