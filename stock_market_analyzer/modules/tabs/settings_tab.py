import sys
import os
import logging
import traceback
import time
import uuid
from typing import Dict, Any, List, Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton,
    QLabel, QSplitter, QApplication, QCheckBox, QGroupBox,
    QSpinBox, QDoubleSpinBox, QTabWidget, QScrollArea, QLineEdit, QFrame, QGridLayout
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from .base_tab import BaseTab
from ..message_bus import MessageBus
from ..settings import Settings

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class SettingsTab(BaseTab):
    """Settings tab for managing application settings."""
    
    def __init__(self, message_bus: MessageBus, parent=None):
        """Initialize the Settings tab.
        
        Args:
            message_bus: The message bus for communication.
            parent: The parent widget.
        """
        # Initialize attributes before parent __init__
        self.settings = Settings()
        self.settings_cache = {}
        self.pending_requests = {}
        self.connection_status = {}
        self.connection_start_times = {}
        self.messages_received = 0
        self.messages_sent = 0
        self.errors = 0
        self.message_latencies = []
        self._ui_setup_done = False
        self.main_layout = None
        self.theme_combo = None
        self.save_button = None
        self.status_label = None
        self.metrics_labels = {}
        
        # Call parent __init__ with message bus
        super().__init__(message_bus, parent)
        
    def _setup_message_bus_impl(self):
        """Setup message bus subscriptions."""
        try:
            # Subscribe to all relevant topics
            self.message_bus.subscribe("Settings", self.handle_settings_message)
            self.message_bus.subscribe("ConnectionStatus", self.handle_connection_status)
            self.message_bus.subscribe("Heartbeat", self.handle_heartbeat)
            self.message_bus.subscribe("Shutdown", self.handle_shutdown)
            
            # Send initial connection status
            self.message_bus.publish("ConnectionStatus", "status_update", {
                "tab": self.__class__.__name__,
                "status": "connected"
            })
            
            self.logger.debug("Message bus setup completed for Settings tab")
            
        except Exception as e:
            self.handle_error("Error setting up message bus subscriptions", e)
            
    def setup_ui(self):
        """Setup the UI components."""
        if self._ui_setup_done:
            return
            
        try:
            # Clear the base layout
            while self.main_layout.count():
                item = self.main_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
                    
            self.main_layout.setSpacing(10)
            self.main_layout.setContentsMargins(10, 10, 10, 10)
            
            # Create settings group
            settings_group = QGroupBox("Settings")
            settings_layout = QVBoxLayout()
            
            # Theme selection
            theme_layout = QHBoxLayout()
            theme_layout.addWidget(QLabel("Theme:"))
            self.theme_combo = QComboBox()
            self.theme_combo.addItems([
                "Light",
                "Dark",
                "System"
            ])
            theme_layout.addWidget(self.theme_combo)
            settings_layout.addLayout(theme_layout)
            
            # Action buttons
            button_layout = QHBoxLayout()
            
            self.save_button = QPushButton("Save Settings")
            self.save_button.clicked.connect(self.save_settings)
            button_layout.addWidget(self.save_button)
            
            settings_layout.addLayout(button_layout)
            settings_group.setLayout(settings_layout)
            self.main_layout.addWidget(settings_group)
            
            # Create metrics group
            metrics_group = QGroupBox("Metrics")
            metrics_layout = QGridLayout()
            
            # Add metrics labels
            self.metrics_labels["messages_received"] = QLabel("Messages Received: 0")
            self.metrics_labels["messages_sent"] = QLabel("Messages Sent: 0")
            self.metrics_labels["errors"] = QLabel("Errors: 0")
            self.metrics_labels["avg_latency"] = QLabel("Average Latency: 0ms")
            
            metrics_layout.addWidget(self.metrics_labels["messages_received"], 0, 0)
            metrics_layout.addWidget(self.metrics_labels["messages_sent"], 0, 1)
            metrics_layout.addWidget(self.metrics_labels["errors"], 1, 0)
            metrics_layout.addWidget(self.metrics_labels["avg_latency"], 1, 1)
            
            metrics_group.setLayout(metrics_layout)
            self.main_layout.addWidget(metrics_group)
            
            # Create status bar
            status_layout = QHBoxLayout()
            
            self.status_label = QLabel("Ready")
            self.status_label.setStyleSheet("color: green")
            status_layout.addWidget(self.status_label)
            
            self.main_layout.addLayout(status_layout)
            
            self._ui_setup_done = True
            self.logger.info("Settings tab UI setup completed")
            
        except Exception as e:
            self.handle_error("Error setting up UI", e)
            
    def save_settings(self):
        """Save the current settings."""
        try:
            theme = self.theme_combo.currentText()
            
            if not theme:
                self.status_label.setText("Please select a theme")
                return
                
            # Generate request ID
            request_id = str(uuid.uuid4())
            
            # Store request
            self.pending_requests[request_id] = {
                "theme": theme,
                "timestamp": time.time()
            }
            
            # Publish settings save request
            self.message_bus.publish("Settings", "save_settings", {
                "request_id": request_id,
                "theme": theme,
                "timestamp": time.time()
            })
            
            self.status_label.setText("Saving settings...")
            self.messages_sent += 1
            
        except Exception as e:
            self.handle_error("Error saving settings", e)
            
    def handle_settings_message(self, sender: str, message_type: str, data: Dict[str, Any]):
        """Handle settings-related messages."""
        try:
            if not isinstance(data, dict):
                self.logger.error(f"Invalid settings message data: {data}")
                return
                
            if message_type == "settings_saved":
                self.handle_settings_saved(sender, data)
            elif message_type == "settings_error":
                self.handle_settings_error(sender, data)
            elif message_type == "settings_loaded":
                self.handle_settings_loaded(sender, data)
            else:
                self.logger.warning(f"Unknown settings message type: {message_type}")
                
            self.messages_received += 1
            self.message_latencies.append(time.time() - data.get("timestamp", time.time()))
            
            # Update metrics
            self.update_metrics()
            
        except Exception as e:
            self.handle_error("Error handling settings message", e)
            
    def handle_connection_status(self, sender: str, message_type: str, data: Dict[str, Any]):
        """Handle connection status updates."""
        try:
            if not isinstance(data, dict):
                self.logger.error(f"Invalid connection status data: {data}")
                return
                
            status = data.get("status")
            if status not in ["connected", "disconnected", "error"]:
                self.logger.error(f"Invalid connection status: {status}")
                return
                
            # Update connection status
            self.connection_status[sender] = status
            
            # Update connection start time if connected
            if status == "connected":
                self.connection_start_times[sender] = time.time()
            elif status == "disconnected":
                self.connection_start_times.pop(sender, None)
                
            # Update status label
            connected = sum(1 for s in self.connection_status.values() if s == "connected")
            total = len(self.connection_status)
            self.status_label.setText(f"Status: Connected: {connected}/{total}")
            self.status_label.setStyleSheet(
                "color: green" if connected > 0 else "color: red"
            )
            
            self.logger.debug(f"Connection status updated for {sender}: {status}")
            
        except Exception as e:
            self.handle_error("Error handling connection status", e)
            
    def handle_settings_saved(self, sender: str, data: Dict[str, Any]):
        """Handle settings saved response."""
        try:
            request_id = data.get("request_id")
            if not request_id:
                self.logger.error("Missing request ID in settings saved response")
                return
                
            # Remove from pending requests
            if request_id in self.pending_requests:
                del self.pending_requests[request_id]
                
            # Update status
            self.status_label.setText("Settings saved successfully")
            self.status_label.setStyleSheet("color: green")
            
            self.logger.info("Settings saved successfully")
            
        except Exception as e:
            self.handle_error("Error handling settings saved response", e)
            
    def handle_settings_error(self, sender: str, data: Dict[str, Any]):
        """Handle settings error response."""
        try:
            request_id = data.get("request_id")
            error_message = data.get("error", "Unknown error")
            
            if not request_id:
                self.logger.error("Missing request ID in settings error response")
                return
                
            # Remove from pending requests
            if request_id in self.pending_requests:
                del self.pending_requests[request_id]
                
            # Update status
            self.status_label.setText(f"Error: {error_message}")
            self.status_label.setStyleSheet("color: red")
            
            self.logger.error(f"Settings error: {error_message}")
            
        except Exception as e:
            self.handle_error("Error handling settings error response", e)
            
    def handle_settings_loaded(self, sender: str, data: Dict[str, Any]):
        """Handle settings loaded response."""
        try:
            settings = data.get("settings", {})
            if not settings:
                self.logger.warning("No settings received")
                return
                
            # Update settings cache
            self.settings_cache = settings
            
            # Update UI
            theme = settings.get("theme", "Light")
            self.theme_combo.setCurrentText(theme)
            
            self.logger.info("Settings loaded successfully")
            
        except Exception as e:
            self.handle_error("Error handling settings loaded response", e)
            
    def update_metrics(self):
        """Update the metrics display."""
        try:
            # Update message counts
            self.metrics_labels["messages_received"].setText(f"Messages Received: {self.messages_received}")
            self.metrics_labels["messages_sent"].setText(f"Messages Sent: {self.messages_sent}")
            self.metrics_labels["errors"].setText(f"Errors: {self.errors}")
            
            # Calculate and update average latency
            if self.message_latencies:
                avg_latency = sum(self.message_latencies) / len(self.message_latencies)
                self.metrics_labels["avg_latency"].setText(f"Average Latency: {avg_latency*1000:.2f}ms")
            else:
                self.metrics_labels["avg_latency"].setText("Average Latency: 0ms")
                
            self.logger.debug("Metrics updated")
            
        except Exception as e:
            self.handle_error("Error updating metrics", e)
            
    def cleanup(self):
        """Clean up resources."""
        try:
            # Clear caches
            self.settings_cache.clear()
            self.pending_requests.clear()
            self.connection_status.clear()
            self.connection_start_times.clear()
            
            # Reset metrics
            self.messages_received = 0
            self.messages_sent = 0
            self.errors = 0
            self.message_latencies.clear()
            
            # Reset metrics labels
            for label in self.metrics_labels.values():
                label.setText("0")
                
            # Reset status label
            if self.status_label:
                self.status_label.setText("Status: Disconnected")
                self.status_label.setStyleSheet("color: red")
                
            # Call parent cleanup
            super().cleanup()
            
            self.logger.info("Settings tab cleanup completed")
            
        except Exception as e:
            self.handle_error("Error during cleanup", e)

def main():
    """Main function for the settings tab process."""
    app = QApplication(sys.argv)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting settings tab process")
    
    # Create and show the settings tab
    window = SettingsTab()
    window.setWindowTitle("Settings Tab")
    window.show()
    
    # Start the application event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 