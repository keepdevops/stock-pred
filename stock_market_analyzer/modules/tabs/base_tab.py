import sys
import os
import signal
import logging
import traceback
from typing import Any, Optional
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextEdit
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont
from modules.message_bus import MessageBus

class BaseTab(QWidget):
    """Base class for all tabs in the application."""
    
    _shared_message_bus = None  # Class variable for shared message bus
    
    @classmethod
    def get_message_bus(cls):
        """Get the shared message bus instance."""
        if cls._shared_message_bus is None:
            cls._shared_message_bus = MessageBus()
        return cls._shared_message_bus
    
    @classmethod
    def set_message_bus(cls, message_bus):
        """Set the shared message bus instance."""
        cls._shared_message_bus = message_bus
    
    def __init__(self, parent=None):
        """Initialize the base tab."""
        super().__init__(parent)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.message_bus = self.get_message_bus()
        self._ui_setup_done = False
        self._message_bus_setup_done = False
        self.heartbeat_timer = None
        self.log_text = QTextEdit()  # Initialize log text widget
        self.log_text.setReadOnly(True)
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        self.status_label = QLabel("Ready")
        self.main_layout.addWidget(self.status_label)
        self.main_layout.addWidget(self.log_text)
        self.setup_ui()
        self.setup_message_bus()
        self.setup_heartbeat()
        
    def setup_ui(self):
        """Setup the UI components."""
        if not self._ui_setup_done:
            self._setup_ui_impl()
            self._ui_setup_done = True
        
    def setup_message_bus(self):
        """Setup message bus subscriptions."""
        if not self._message_bus_setup_done:
            self._setup_message_bus_impl()
            self._message_bus_setup_done = True
        
    def _setup_ui_impl(self):
        """Implementation of UI setup. To be overridden by subclasses."""
        pass
    
    def _setup_message_bus_impl(self):
        """Implementation of message bus setup. To be overridden by subclasses."""
        # Subscribe to general messages
        self.message_bus.subscribe("general", lambda sender, msg_type, msg_data: self.handle_message(sender, msg_type, msg_data))
        
        # Subscribe to tab-specific messages
        self.message_bus.subscribe(self.__class__.__name__, lambda sender, msg_type, msg_data: self.handle_message(sender, msg_type, msg_data))
        
    def handle_message(self, sender: str, message_type: str, data: dict):
        """Handle incoming messages from the message bus.
        
        Args:
            sender: The sender of the message
            message_type: The type of message
            data: The message data
        """
        logging.debug(f"Received message from {sender}: {message_type}")
        try:
            if message_type == "heartbeat":
                self.handle_heartbeat(sender, data)
            elif message_type == "shutdown":
                self.handle_shutdown()
            else:
                self.process_message(sender, message_type, data)
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
            
    def process_message(self, sender: str, message_type: str, data: Any):
        """Process specific message types. Override in subclasses."""
        pass
        
    def setup_heartbeat(self):
        """Setup heartbeat timer."""
        self.heartbeat_timer = QTimer(self)
        self.heartbeat_timer.timeout.connect(self.send_heartbeat)
        self.heartbeat_timer.start(5000)  # Send heartbeat every 5 seconds
        
    def send_heartbeat(self):
        """Send heartbeat message."""
        self.message_bus.publish(
            self.__class__.__name__,
            "heartbeat",
            {"status": "alive"}
        )
        
    def handle_heartbeat(self, sender: str, data: Any):
        """Handle heartbeat message."""
        self.logger.debug(f"Heartbeat from {sender}: {data}")
        
    def handle_shutdown(self):
        """Handle shutdown message."""
        self.logger.info("Received shutdown signal")
        self.cleanup()
        
    def cleanup(self):
        """Cleanup resources. Override in subclasses."""
        if self.heartbeat_timer:
            self.heartbeat_timer.stop()
        self.message_bus.publish(self.__class__.__name__, "shutdown", {})
        
    def closeEvent(self, event):
        """Handle tab close event."""
        self.cleanup()
        super().closeEvent(event) 