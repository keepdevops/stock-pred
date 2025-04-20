import sys
import os
import logging
import traceback
import time
from typing import Any, Optional, Dict, List
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QGroupBox, QGridLayout
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from ..ui.theme import DarkTheme

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ..message_bus import MessageBus

class BaseTab(QWidget):
    """Base class for all tabs in the application."""
    
    # Class variables for shared resources
    _shared_message_bus = None
    _connection_status = {}
    _connection_start_times = {}
    
    def __init__(self, message_bus: MessageBus, parent=None):
        """Initialize the base tab.
        
        Args:
            message_bus: The message bus for communication.
            parent: The parent widget.
        """
        super().__init__(parent)
        self.message_bus = message_bus
        self.logger = logging.getLogger(__name__)
        self.connection_status = {}
        self.connection_start_times = {}
        self._ui_setup_done = False
        self._message_bus_setup_done = False
        self._timers = []
        self._error_handling = False
        
        # Initialize basic layout
        self.main_layout = QVBoxLayout()
        self.main_layout.setSpacing(10)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.setLayout(self.main_layout)
        
        # Apply dark theme
        DarkTheme.apply_theme(self)
        
        # Set window flags
        self.setWindowFlags(Qt.WindowType.Window)
        
        # Initialize status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: green")
        self.main_layout.addWidget(self.status_label)
        
        # Initialize metrics
        self.metrics_labels = {}
        self.messages_received = 0
        self.messages_sent = 0
        self.errors = 0
        self.message_latencies = []
        
        # Setup UI first
        self.setup_ui()
        
        # Then setup message bus
        self.setup_message_bus()
        
    def showEvent(self, event):
        """Handle show event to center window."""
        super().showEvent(event)
        DarkTheme.center_window(self)
        
    def setup_message_bus(self):
        """Setup message bus subscriptions."""
        if self._message_bus_setup_done:
            return
            
        try:
            if not self.message_bus:
                self.logger.error("Message bus not provided")
                self.status_label.setText("Error: Message bus not provided")
                return
                
            # Subscribe to system messages
            self.message_bus.subscribe("system", self.handle_message)
            self.message_bus.subscribe("connection", self.handle_connection_status)
            self.message_bus.subscribe("heartbeat", self.handle_heartbeat)
            self.message_bus.subscribe("shutdown", self.handle_shutdown)
            
            # Call child class implementation
            self._setup_message_bus_impl()
            
            # Send initial connection status
            self.message_bus.publish("connection", "status_update", {
                "tab": self.__class__.__name__,
                "status": "connected"
            })
            
            self._message_bus_setup_done = True
            self.logger.debug(f"Message bus setup completed for {self.__class__.__name__}")
            
        except Exception as e:
            self.handle_error("Error setting up message bus", e)
            
    def _setup_message_bus_impl(self):
        """Override this method to implement tab-specific message bus setup."""
        pass
        
    def setup_ui(self):
        """Setup the UI components."""
        if self._ui_setup_done:
            return
            
        try:
            # Clear the base layout except for status label
            while self.main_layout.count() > 1:  # Keep status label
                item = self.main_layout.takeAt(1)  # Start from index 1
                if item.widget():
                    item.widget().deleteLater()
                    
            # Call implementation-specific UI setup
            self._setup_ui_impl()
            
            self._ui_setup_done = True
            self.logger.info(f"{self.__class__.__name__} tab initialized")
            
        except Exception as e:
            self.handle_error("Error setting up UI", e)
            
    def _setup_ui_impl(self):
        """Override this method to implement tab-specific UI setup."""
        pass
        
    def setup_metrics(self):
        """Setup the metrics display."""
        try:
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
            
        except Exception as e:
            self.handle_error("Error setting up metrics", e)
            
    def update_metrics(self):
        """Update the metrics display."""
        try:
            if not hasattr(self, 'metrics_labels') or not self.metrics_labels:
                self.logger.warning("Metrics labels not initialized")
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
            
    def handle_error(self, context: str, error: Exception):
        """Handle errors consistently across the application."""
        if self._error_handling:
            # Prevent recursive error handling
            self.logger.error("Recursive error detected, aborting error handling")
            return
            
        self._error_handling = True
        try:
            error_msg = f"{context}: {str(error)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            
            if hasattr(self, 'status_label') and self.status_label:
                self.status_label.setText(error_msg)
                self.status_label.setStyleSheet("color: red")
                
            self.errors += 1
            self.update_metrics()
            
        except Exception as e:
            self.logger.error(f"Error in handle_error: {str(e)}")
            self.logger.error(traceback.format_exc())
        finally:
            self._error_handling = False
            
    def add_timer(self, timer: QTimer):
        """Add a timer to be cleaned up later.
        
        Args:
            timer: The timer to add.
        """
        self._timers.append(timer)
        
    def handle_message(self, sender: str, message_type: str, data: Dict[str, Any]):
        """Handle incoming messages.
        
        Args:
            sender: The sender of the message.
            message_type: The type of message.
            data: The message data.
        """
        try:
            if message_type == "error":
                self.logger.error(f"Error from {sender}: {data.get('error', 'Unknown error')}")
                self.status_label.setText(f"Error: {data.get('error', 'Unknown error')}")
            elif message_type == "status":
                self.status_label.setText(data.get('status', 'Unknown status'))
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
            self.status_label.setText(f"Error: {str(e)}")
            
    def handle_connection_status(self, sender: str, message_type: str, data: Dict[str, Any]):
        """Handle connection status updates.
        
        Args:
            sender: The sender of the message.
            message_type: The type of message.
            data: The message data.
        """
        try:
            status = data.get('status', 'unknown')
            self.connection_status[sender] = status
            if status == 'connected':
                self.connection_start_times[sender] = time.time()
                self.status_label.setStyleSheet("color: green")
            else:
                self.status_label.setStyleSheet("color: yellow")
            self.update_connection_status()
        except Exception as e:
            self.logger.error(f"Error handling connection status: {e}")
            
    def handle_heartbeat(self, sender: str, data: Dict[str, Any]):
        """Handle heartbeat messages.
        
        Args:
            sender: The sender of the message.
            data: The message data.
        """
        try:
            if sender in self.connection_start_times:
                elapsed = time.time() - self.connection_start_times[sender]
                self.logger.debug(f"Heartbeat from {sender} after {elapsed:.2f} seconds")
        except Exception as e:
            self.logger.error(f"Error handling heartbeat: {e}")
            
    def handle_shutdown(self, sender: str, data: Dict[str, Any]):
        """Handle shutdown messages.
        
        Args:
            sender: The sender of the message.
            data: The message data.
        """
        try:
            self.logger.info(f"Received shutdown signal from {sender}")
            self.cleanup()
            self.close()
        except Exception as e:
            self.logger.error(f"Error handling shutdown: {e}")
            
    def update_connection_status(self):
        """Update the connection status display."""
        try:
            connected = sum(1 for status in self.connection_status.values() if status == 'connected')
            total = len(self.connection_status)
            self.status_label.setText(f"Connected: {connected}/{total}")
        except Exception as e:
            self.logger.error(f"Error updating connection status: {e}")
            
    def cleanup(self):
        """Clean up resources."""
        try:
            # Stop all timers
            for timer in self._timers:
                if timer:
                    timer.stop()
            self._timers.clear()
            
            # Clear connection status
            self.connection_status.clear()
            self.connection_start_times.clear()
            
            # Unsubscribe from message bus
            if self.message_bus:
                self.message_bus.unsubscribe_all(self)
                
            # Reset metrics
            self.messages_received = 0
            self.messages_sent = 0
            self.errors = 0
            self.message_latencies.clear()
            
            # Reset metrics labels
            for label in self.metrics_labels.values():
                if label:
                    label.setText("0")
                    
            # Call implementation-specific cleanup
            self._cleanup_impl()
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            
    def closeEvent(self, event):
        """Handle close event."""
        try:
            self.cleanup()
            event.accept()
        except Exception as e:
            self.logger.error(f"Error during close: {e}")
            event.accept()
            
    def __del__(self):
        """Destructor."""
        try:
            self.cleanup()
        except Exception as e:
            self.logger.error(f"Error in destructor: {e}")
            
    def _cleanup_impl(self):
        """Implementation-specific cleanup."""
        pass 