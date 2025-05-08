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
    QSpinBox, QDoubleSpinBox, QTabWidget, QScrollArea, QLineEdit, QFrame,
    QTextEdit, QMessageBox, QGridLayout
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

class HelpTab(BaseTab):
    """Help tab for providing documentation and support."""
    
    def __init__(self, message_bus: MessageBus, parent=None):
        """Initialize the Help tab.
        
        Args:
            message_bus: The message bus for communication.
            parent: The parent widget.
        """
        # Initialize attributes before parent __init__
        self.settings = Settings()
        self.current_color_scheme = self.settings.get_color_scheme()
        self._ui_setup_done = False
        self.help_cache = {}
        self.pending_requests = {}
        self.connection_status = {}
        self.connection_start_times = {}
        self.messages_received = 0
        self.messages_sent = 0
        self.errors = 0
        self.message_latencies = []
        self.main_layout = None
        self.topic_combo = None
        self.search_edit = None
        self.search_button = None
        self.content_text = None
        self.status_label = None
        self.metrics_labels = {}
        
        # Call parent __init__ with message bus
        super().__init__(message_bus, parent)
        
    def _setup_message_bus_impl(self):
        """Setup message bus subscriptions."""
        try:
            # Subscribe to help-related topics
            self.message_bus.subscribe("Help", self.handle_help_message)
            self.message_bus.subscribe("ConnectionStatus", self.handle_connection_status)
            self.message_bus.subscribe("Heartbeat", self.handle_heartbeat)
            self.message_bus.subscribe("Shutdown", self.handle_shutdown)
            
            # Send initial connection status
            self.message_bus.publish("ConnectionStatus", "status_update", {
                "tab": self.__class__.__name__,
                "status": "connected"
            })
            
            self.logger.debug("Message bus setup completed for Help tab")
            
        except Exception as e:
            self.handle_error("Error setting up message bus subscriptions", e)
            
    def _setup_ui_impl(self):
        """Setup the UI components."""
        try:
            # Create help group
            help_group = QGroupBox("Help")
            help_layout = QVBoxLayout()
            
            # Topic selection
            topic_layout = QHBoxLayout()
            topic_layout.addWidget(QLabel("Topic:"))
            self.topic_combo = QComboBox()
            self.topic_combo.addItems([
                "Getting Started",
                "Data Import",
                "Analysis",
                "Charts",
                "Models",
                "Predictions",
                "Trading",
                "Settings"
            ])
            topic_layout.addWidget(self.topic_combo)
            help_layout.addLayout(topic_layout)
            
            # Search input
            search_layout = QHBoxLayout()
            search_layout.addWidget(QLabel("Search:"))
            self.search_edit = QLineEdit()
            self.search_edit.setPlaceholderText("Enter search terms")
            search_layout.addWidget(self.search_edit)
            
            self.search_button = QPushButton("Search")
            self.search_button.clicked.connect(self.search_help)
            search_layout.addWidget(self.search_button)
            
            help_layout.addLayout(search_layout)
            
            # Content display
            content_layout = QVBoxLayout()
            self.content_text = QTextEdit()
            self.content_text.setReadOnly(True)
            content_layout.addWidget(self.content_text)
            
            help_layout.addLayout(content_layout)
            help_group.setLayout(help_layout)
            self.main_layout.addWidget(help_group)
            
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
            self.handle_error("Error setting up UI", e)
            
    def search_help(self):
        """Search help content."""
        try:
            topic = self.topic_combo.currentText()
            search_terms = self.search_edit.text().strip()
            
            if not search_terms:
                self.status_label.setText("Please enter search terms")
                return
                
            # Generate request ID
            request_id = str(uuid.uuid4())
            
            # Store request
            self.pending_requests[request_id] = {
                "topic": topic,
                "search_terms": search_terms,
                "timestamp": time.time()
            }
            
            # Publish help request
            self.message_bus.publish("Help", "search_help", {
                "request_id": request_id,
                "topic": topic,
                "search_terms": search_terms,
                "timestamp": time.time()
            })
            
            self.status_label.setText(f"Searching help for: {search_terms}...")
            self.messages_sent += 1
            
        except Exception as e:
            self.handle_error("Error searching help", e)
            
    def handle_help_message(self, sender: str, message_type: str, data: Dict[str, Any]):
        """Handle help-related messages."""
        try:
            if not isinstance(data, dict):
                self.logger.error(f"Invalid help message data: {data}")
                return
                
            if message_type == "help_content":
                self.handle_help_content(sender, data)
            elif message_type == "help_error":
                self.handle_help_error(sender, data)
            elif message_type == "help_topics":
                self.handle_help_topics(sender, data)
            else:
                self.logger.warning(f"Unknown help message type: {message_type}")
                
            self.messages_received += 1
            self.message_latencies.append(time.time() - data.get("timestamp", time.time()))
            
            # Update metrics
            self.update_metrics()
            
        except Exception as e:
            self.handle_error("Error handling help message", e)
            
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
            
    def handle_help_content(self, sender: str, data: Dict[str, Any]):
        """Handle help content response."""
        try:
            request_id = data.get("request_id")
            content = data.get("content")
            
            if not all([request_id, content]):
                self.logger.error("Missing required data in help content response")
                return
                
            # Remove from pending requests
            if request_id in self.pending_requests:
                del self.pending_requests[request_id]
                
            # Update help cache
            topic = data.get("topic")
            if topic:
                self.help_cache[topic] = content
                
            # Update content display
            self.content_text.setHtml(content)
            
            # Update status
            self.status_label.setText("Help content loaded")
            self.status_label.setStyleSheet("color: green")
            
            self.logger.info("Help content loaded successfully")
            
        except Exception as e:
            self.handle_error("Error handling help content response", e)
            
    def handle_help_error(self, sender: str, data: Dict[str, Any]):
        """Handle help error response."""
        try:
            request_id = data.get("request_id")
            error_message = data.get("error", "Unknown error")
            
            if not request_id:
                self.logger.error("Missing request ID in help error response")
                return
                
            # Remove from pending requests
            if request_id in self.pending_requests:
                del self.pending_requests[request_id]
                
            # Update status
            self.status_label.setText(f"Error: {error_message}")
            self.status_label.setStyleSheet("color: red")
            
            self.logger.error(f"Help error: {error_message}")
            
        except Exception as e:
            self.handle_error("Error handling help error response", e)
            
    def handle_help_topics(self, sender: str, data: Dict[str, Any]):
        """Handle help topics response."""
        try:
            topics = data.get("topics", [])
            if not topics:
                self.logger.warning("No help topics received")
                return
                
            # Update topic combo box
            self.topic_combo.clear()
            self.topic_combo.addItems(topics)
            
            self.logger.info("Help topics updated successfully")
            
        except Exception as e:
            self.handle_error("Error handling help topics response", e)
            
    def update_metrics(self):
        """Update the metrics display."""
        try:
            if not self.metrics_labels:
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
            
    def cleanup(self):
        """Clean up resources."""
        try:
            # Clear caches
            self.help_cache.clear()
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
            
            self.logger.info("Help tab cleanup completed")
            
        except Exception as e:
            self.handle_error("Error during cleanup", e)

def main():
    """Main function for the help tab process."""
    app = QApplication(sys.argv)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting help tab process")
    
    # Create and show the help tab
    window = HelpTab()
    window.setWindowTitle("Help Tab")
    window.show()
    
    # Start the application event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 