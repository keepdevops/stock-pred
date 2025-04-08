from PyQt6.QtCore import QObject, pyqtSignal
import logging
from typing import Any, Dict, Optional

class MessageBus(QObject):
    """Central message bus for inter-tab communication."""
    
    # Define signals for different types of messages
    data_updated = pyqtSignal(str, object)  # symbol, data
    analysis_requested = pyqtSignal(str, str)  # symbol, analysis_type
    analysis_completed = pyqtSignal(str, str, object)  # symbol, analysis_type, results
    chart_update_requested = pyqtSignal(str, str, object)  # symbol, chart_type, data
    trading_signal = pyqtSignal(str, str, object)  # symbol, signal_type, data
    error_occurred = pyqtSignal(str)  # error_message
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self._subscribers: Dict[str, list] = {}
        
    def subscribe(self, tab_name: str, callback: callable):
        """Subscribe a tab to receive messages."""
        if tab_name not in self._subscribers:
            self._subscribers[tab_name] = []
        self._subscribers[tab_name].append(callback)
        self.logger.info(f"Tab '{tab_name}' subscribed to message bus")
        
    def unsubscribe(self, tab_name: str, callback: callable):
        """Unsubscribe a tab from receiving messages."""
        if tab_name in self._subscribers:
            self._subscribers[tab_name].remove(callback)
            self.logger.info(f"Tab '{tab_name}' unsubscribed from message bus")
            
    def publish(self, tab_name: str, message_type: str, data: Any):
        """Publish a message to all subscribers."""
        self.logger.info(f"Publishing message from '{tab_name}': {message_type}")
        
        # Emit appropriate signal based on message type
        if message_type == "data_updated":
            self.data_updated.emit(tab_name, data)
        elif message_type == "analysis_requested":
            self.analysis_requested.emit(tab_name, data)
        elif message_type == "analysis_completed":
            self.analysis_completed.emit(tab_name, data[0], data[1])
        elif message_type == "chart_update":
            self.chart_update_requested.emit(tab_name, data[0], data[1])
        elif message_type == "trading_signal":
            self.trading_signal.emit(tab_name, data[0], data[1])
        elif message_type == "error":
            self.error_occurred.emit(data)
            
        # Notify subscribers
        for subscribers in self._subscribers.values():
            for callback in subscribers:
                try:
                    callback(tab_name, message_type, data)
                except Exception as e:
                    self.logger.error(f"Error in subscriber callback: {str(e)}") 