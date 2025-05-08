import sys
import os
import time
import logging
import traceback
from typing import Dict, Optional, Any
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTableWidget, QTableWidgetItem, QTextEdit, QLabel,
    QSplitter, QHeaderView, QGroupBox, QGridLayout
)
from PyQt6.QtCore import QTimer, Qt
from .base_tab import BaseTab
from ..process_manager import ProcessManager
from ..message_bus import MessageBus
from ..settings import Settings

class MonitorTab(BaseTab):
    """Tab for monitoring the trading agent and other processes."""
    
    def __init__(self, message_bus: MessageBus, parent=None):
        """Initialize the Monitor tab.
        
        Args:
            message_bus: The message bus for communication.
            parent: The parent widget.
        """
        # Initialize attributes before parent __init__
        self.settings = Settings()
        self.current_color_scheme = self.settings.get_color_scheme()
        self._ui_setup_done = False
        self.process_manager = None
        self.process_list = None
        self.status_label = None
        self.log_viewer = None
        self.refresh_button = None
        self.restart_button = None
        self.terminate_button = None
        self.refresh_timer = None
        self.connection_status = {}
        self.connection_start_times = {}
        self.messages_received = 0
        self.messages_sent = 0
        self.errors = 0
        self.message_latencies = []
        self.pending_requests = {}
        
        # Call parent __init__ with message bus
        super().__init__(message_bus, parent)
        
    def _setup_message_bus_impl(self):
        """Setup message bus subscriptions."""
        try:
            # Subscribe to all relevant topics
            self.message_bus.subscribe("Monitor", self.handle_monitor_message)
            self.message_bus.subscribe("Process", self.handle_process_message)
            self.message_bus.subscribe("ConnectionStatus", self.handle_connection_status)
            self.message_bus.subscribe("Heartbeat", self.handle_heartbeat)
            self.message_bus.subscribe("Shutdown", self.handle_shutdown)
            
            # Send initial connection status
            self.message_bus.publish("ConnectionStatus", "status_update", {
                "tab": self.__class__.__name__,
                "status": "connected"
            })
            
            self.logger.debug("Message bus setup completed for Monitor tab")
            
        except Exception as e:
            self.handle_error("Error setting up message bus subscriptions", e)
            
    def setup_ui(self):
        """Set up the user interface."""
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
            
            # Create splitter for process list and details
            splitter = QSplitter(Qt.Orientation.Horizontal)
            
            # Create process list widget
            process_group = QGroupBox("Processes")
            process_layout = QVBoxLayout()
            
            # Add process list controls
            controls_layout = QHBoxLayout()
            self.refresh_button = QPushButton("Refresh")
            self.refresh_button.clicked.connect(self.refresh_processes)
            controls_layout.addWidget(self.refresh_button)
            
            self.restart_button = QPushButton("Restart Selected")
            self.restart_button.clicked.connect(self.restart_selected)
            controls_layout.addWidget(self.restart_button)
            
            self.terminate_button = QPushButton("Terminate Selected")
            self.terminate_button.clicked.connect(self.terminate_selected)
            controls_layout.addWidget(self.terminate_button)
            
            process_layout.addLayout(controls_layout)
            
            # Create process list
            self.process_list = QTableWidget()
            self.process_list.setColumnCount(5)  # Added memory usage column
            self.process_list.setHorizontalHeaderLabels([
                "Process", "Status", "PID", "Last Heartbeat", "Memory Usage"
            ])
            self.process_list.horizontalHeader().setSectionResizeMode(
                QHeaderView.ResizeMode.ResizeToContents
            )
            self.process_list.setSelectionBehavior(
                QTableWidget.SelectionBehavior.SelectRows
            )
            self.process_list.itemSelectionChanged.connect(
                self.on_process_selected
            )
            process_layout.addWidget(self.process_list)
            process_group.setLayout(process_layout)
            
            # Create details widget
            details_group = QGroupBox("Details")
            details_layout = QVBoxLayout()
            
            # Create metrics group
            metrics_group = QGroupBox("Metrics")
            metrics_layout = QGridLayout()
            
            # Add metrics labels
            self.messages_received_label = QLabel("Messages Received: 0")
            self.messages_sent_label = QLabel("Messages Sent: 0")
            self.errors_label = QLabel("Errors: 0")
            self.avg_latency_label = QLabel("Average Latency: 0ms")
            
            metrics_layout.addWidget(self.messages_received_label, 0, 0)
            metrics_layout.addWidget(self.messages_sent_label, 0, 1)
            metrics_layout.addWidget(self.errors_label, 1, 0)
            metrics_layout.addWidget(self.avg_latency_label, 1, 1)
            
            metrics_group.setLayout(metrics_layout)
            details_layout.addWidget(metrics_group)
            
            # Create status label
            self.status_label = QLabel("Status: Not Connected")
            self.status_label.setStyleSheet("color: red")
            details_layout.addWidget(self.status_label)
            
            # Create log viewer
            self.log_viewer = QTextEdit()
            self.log_viewer.setReadOnly(True)
            details_layout.addWidget(self.log_viewer)
            
            details_group.setLayout(details_layout)
            
            # Add widgets to splitter
            splitter.addWidget(process_group)
            splitter.addWidget(details_group)
            
            # Set splitter sizes
            splitter.setSizes([300, 500])
            
            # Add splitter to main layout
            self.main_layout.addWidget(splitter)
            
            self._ui_setup_done = True
            self.logger.info("Monitor tab UI setup completed")
            
        except Exception as e:
            self.handle_error("Error setting up UI", e)
            
    def setup_monitoring(self):
        """Set up process monitoring."""
        try:
            if self.refresh_timer:
                self.refresh_timer.stop()
                self.refresh_timer.deleteLater()
                
            # Create refresh timer
            self.refresh_timer = QTimer(self)
            self.refresh_timer.timeout.connect(self.refresh_processes)
            self.refresh_timer.start(5000)  # Refresh every 5 seconds
            
            self.logger.info("Process monitoring setup completed")
            
        except Exception as e:
            self.handle_error("Error setting up monitoring", e)
            
    def set_process_manager(self, process_manager: ProcessManager):
        """Set the process manager instance."""
        try:
            if not process_manager:
                self.logger.error("Invalid process manager provided")
                return
                
            self.process_manager = process_manager
            self.status_label.setText("Status: Connected")
            self.status_label.setStyleSheet("color: green")
            self.refresh_processes()
            
            self.logger.info("Process manager set successfully")
            
        except Exception as e:
            self.handle_error("Error setting process manager", e)
            
    def refresh_processes(self):
        """Refresh the process list."""
        try:
            if not self.process_manager:
                self.logger.warning("Process manager not available")
                return
                
            # Clear current items
            self.process_list.setRowCount(0)
            
            # Add processes to the list
            for tab_name in self.process_manager.processes:
                process = self.process_manager.processes[tab_name]
                status = self.process_manager.check_status(tab_name)
                pid = process.pid if process else "N/A"
                last_heartbeat = time.strftime(
                    "%Y-%m-%d %H:%M:%S",
                    time.localtime(self.process_manager.heartbeats.get(tab_name, 0))
                )
                
                row = self.process_list.rowCount()
                self.process_list.insertRow(row)
                
                self.process_list.setItem(row, 0, QTableWidgetItem(tab_name))
                self.process_list.setItem(row, 1, QTableWidgetItem(status))
                self.process_list.setItem(row, 2, QTableWidgetItem(str(pid)))
                self.process_list.setItem(row, 3, QTableWidgetItem(last_heartbeat))
                self.process_list.setItem(row, 4, QTableWidgetItem(f"{process.memory_usage if process else 'N/A'} MB"))
                
            self.logger.debug("Process list refreshed")
            
        except Exception as e:
            self.handle_error("Error refreshing processes", e)
            
    def on_process_selected(self):
        """Handle process selection."""
        try:
            if not self.process_manager:
                self.logger.warning("Process manager not available")
                return
                
            selected_items = self.process_list.selectedItems()
            if not selected_items:
                self.log_viewer.clear()
                return
                
            tab_name = selected_items[0].text()
            process = self.process_manager.processes.get(tab_name)
            
            if process:
                self.log_viewer.clear()
                self.log_viewer.append(f"Process: {tab_name}")
                self.log_viewer.append(f"Status: {self.process_manager.check_status(tab_name)}")
                self.log_viewer.append(f"PID: {process.pid}")
                self.log_viewer.append(f"Last Heartbeat: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.process_manager.heartbeats.get(tab_name, 0)))}")
                
            self.logger.debug(f"Process {tab_name} selected")
            
        except Exception as e:
            self.handle_error("Error handling process selection", e)
            
    def restart_selected(self):
        """Restart the selected process."""
        try:
            if not self.process_manager:
                self.logger.warning("Process manager not available")
                return
                
            selected_items = self.process_list.selectedItems()
            if not selected_items:
                self.logger.warning("No process selected")
                return
                
            tab_name = selected_items[0].text()
            self.process_manager.restart_process(tab_name)
            self.refresh_processes()
            
            self.logger.info(f"Process {tab_name} restarted")
            
        except Exception as e:
            self.handle_error("Error restarting process", e)
            
    def terminate_selected(self):
        """Terminate the selected process."""
        try:
            if not self.process_manager:
                self.logger.warning("Process manager not available")
                return
                
            selected_items = self.process_list.selectedItems()
            if not selected_items:
                self.logger.warning("No process selected")
                return
                
            tab_name = selected_items[0].text()
            self.process_manager.terminate_process(tab_name)
            self.refresh_processes()
            
            self.logger.info(f"Process {tab_name} terminated")
            
        except Exception as e:
            self.handle_error("Error terminating process", e)
            
    def update_metrics(self):
        """Update the metrics display."""
        try:
            # Update message counts
            self.messages_received_label.setText(f"Messages Received: {self.messages_received}")
            self.messages_sent_label.setText(f"Messages Sent: {self.messages_sent}")
            self.errors_label.setText(f"Errors: {self.errors}")
            
            # Calculate and update average latency
            if self.message_latencies:
                avg_latency = sum(self.message_latencies) / len(self.message_latencies)
                self.avg_latency_label.setText(f"Average Latency: {avg_latency*1000:.2f}ms")
            else:
                self.avg_latency_label.setText("Average Latency: 0ms")
                
            self.logger.debug("Metrics updated")
            
        except Exception as e:
            self.handle_error("Error updating metrics", e)
            
    def handle_monitor_message(self, sender: str, message_type: str, data: Dict[str, Any]):
        """Handle monitor-related messages."""
        try:
            if not isinstance(data, dict):
                self.logger.error(f"Invalid monitor message data: {data}")
                return
                
            if message_type == "status_update":
                self.handle_status_update(sender, data)
            elif message_type == "log_update":
                self.handle_log_update(sender, data)
            elif message_type == "metrics_update":
                self.handle_metrics_update(sender, data)
            else:
                self.logger.warning(f"Unknown monitor message type: {message_type}")
                
            self.messages_received += 1
            self.message_latencies.append(time.time() - data.get("timestamp", time.time()))
            
            # Update metrics
            self.update_metrics()
            
        except Exception as e:
            self.handle_error("Error handling monitor message", e)
            
    def handle_process_message(self, sender: str, message_type: str, data: Dict[str, Any]):
        """Handle process-related messages."""
        try:
            if not isinstance(data, dict):
                self.logger.error(f"Invalid process message data: {data}")
                return
                
            if message_type == "process_started":
                self.handle_process_started(sender, data)
            elif message_type == "process_stopped":
                self.handle_process_stopped(sender, data)
            else:
                self.logger.warning(f"Unknown process message type: {message_type}")
                
            self.messages_received += 1
            self.message_latencies.append(time.time() - data.get("timestamp", time.time()))
            
            # Update metrics
            self.update_metrics()
            
        except Exception as e:
            self.handle_error("Error handling process message", e)
            
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
            
    def handle_status_update(self, sender: str, data: Dict[str, Any]):
        """Handle status update messages."""
        try:
            status = data.get("status", "unknown")
            if status not in ["connected", "disconnected", "error"]:
                self.logger.error(f"Invalid status: {status}")
                return
                
            self.status_label.setText(f"Status: {status}")
            self.status_label.setStyleSheet(
                "color: green" if status == "connected" else "color: red"
            )
            self.logger.debug(f"Status updated for {sender}: {status}")
            
        except Exception as e:
            self.handle_error("Error handling status update", e)
            
    def handle_log_update(self, sender: str, data: Dict[str, Any]):
        """Handle log update messages."""
        try:
            log_message = data.get("message", "")
            if not log_message:
                self.logger.warning("Empty log message received")
                return
                
            self.log_viewer.append(log_message)
            self.logger.debug(f"Log updated for {sender}: {log_message}")
            
        except Exception as e:
            self.handle_error("Error handling log update", e)
            
    def handle_process_started(self, sender: str, data: Dict[str, Any]):
        """Handle process started messages."""
        try:
            process_name = data.get("process_name", "")
            if not process_name:
                self.logger.error("Missing process name in process_started message")
                return
                
            self.log_viewer.append(f"Process {process_name} started")
            self.refresh_processes()
            self.logger.info(f"Process {process_name} started")
            
        except Exception as e:
            self.handle_error("Error handling process started", e)
            
    def handle_process_stopped(self, sender: str, data: Dict[str, Any]):
        """Handle process stopped messages."""
        try:
            process_name = data.get("process_name", "")
            if not process_name:
                self.logger.error("Missing process name in process_stopped message")
                return
                
            self.log_viewer.append(f"Process {process_name} stopped")
            self.refresh_processes()
            self.logger.info(f"Process {process_name} stopped")
            
        except Exception as e:
            self.handle_error("Error handling process stopped", e)
            
    def handle_metrics_update(self, sender: str, data: Dict[str, Any]):
        """Handle metrics update messages."""
        try:
            if not isinstance(data, dict):
                self.logger.error(f"Invalid metrics data: {data}")
                return
                
            # Update process memory usage if available
            if "memory_usage" in data and sender in self.process_manager.processes:
                process = self.process_manager.processes[sender]
                if process:
                    process.memory_usage = data["memory_usage"]
                    self.refresh_processes()
                    
            self.logger.debug(f"Metrics updated for {sender}")
            
        except Exception as e:
            self.handle_error("Error handling metrics update", e)
            
    def handle_heartbeat(self, sender: str, data: Dict[str, Any]):
        """Handle heartbeat messages."""
        try:
            if not isinstance(data, dict):
                self.logger.error(f"Invalid heartbeat data: {data}")
                return
                
            # Update heartbeat timestamp
            if "timestamp" in data:
                self.process_manager.heartbeats[sender] = data["timestamp"]
                self.refresh_processes()
                
            self.logger.debug(f"Heartbeat received from {sender}")
            
        except Exception as e:
            self.handle_error("Error handling heartbeat", e)
            
    def handle_shutdown(self, sender: str, data: Dict[str, Any]):
        """Handle shutdown messages."""
        try:
            if not isinstance(data, dict):
                self.logger.error(f"Invalid shutdown data: {data}")
                return
                
            # Handle shutdown logic
            self.logger.info(f"Shutdown received from {sender}")
            
        except Exception as e:
            self.handle_error("Error handling shutdown", e)
            
    def cleanup(self):
        """Clean up resources."""
        try:
            # Stop refresh timer
            if self.refresh_timer:
                self.refresh_timer.stop()
                self.refresh_timer.deleteLater()
                self.refresh_timer = None
                
            # Clear caches
            self.connection_status.clear()
            self.connection_start_times.clear()
            self.pending_requests.clear()
            
            # Reset metrics
            self.messages_received = 0
            self.messages_sent = 0
            self.errors = 0
            self.message_latencies.clear()
            
            # Reset metrics labels
            self.messages_received_label.setText("Messages Received: 0")
            self.messages_sent_label.setText("Messages Sent: 0")
            self.errors_label.setText("Errors: 0")
            self.avg_latency_label.setText("Average Latency: 0ms")
            
            # Reset status label
            if self.status_label:
                self.status_label.setText("Status: Disconnected")
                self.status_label.setStyleSheet("color: red")
                
            # Clear log viewer
            if self.log_viewer:
                self.log_viewer.clear()
                
            # Clear process list
            if self.process_list:
                self.process_list.setRowCount(0)
                
            # Call parent cleanup
            super().cleanup()
            
            self.logger.info("Monitor tab cleanup completed")
            
        except Exception as e:
            self.handle_error("Error during cleanup", e)

def main():
    """Main function for the monitor tab process."""
    app = QApplication(sys.argv)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting monitor tab process")
    
    # Create and show the monitor tab
    window = MonitorTab()
    window.setWindowTitle("Monitor Tab")
    window.show()
    
    # Start the application event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 