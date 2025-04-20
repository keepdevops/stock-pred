import sys
import logging
import csv
from datetime import datetime, timedelta
from typing import Dict, Any, List
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QPushButton, QGroupBox, QComboBox, QFileDialog,
    QMessageBox, QToolTip
)
from PyQt6.QtCore import Qt, QTimer, QDateTime
from PyQt6.QtGui import QColor, QFont

class ConnectionDashboard(QWidget):
    """Dashboard for monitoring tab connections and diagnostics."""
    
    def __init__(self, parent=None):
        """Initialize the connection dashboard."""
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.connection_data = {}
        self.message_history = []
        self.setup_ui()
        self.setup_refresh_timer()
        
    def setup_ui(self):
        """Setup the dashboard UI."""
        try:
            # Main layout
            layout = QVBoxLayout()
            layout.setSpacing(10)
            layout.setContentsMargins(10, 10, 10, 10)
            
            # Controls group
            controls_group = QGroupBox("Controls")
            controls_layout = QHBoxLayout()
            
            # Refresh button
            refresh_button = QPushButton("Refresh")
            refresh_button.clicked.connect(self.refresh_data)
            refresh_button.setToolTip("Manually refresh the dashboard data")
            controls_layout.addWidget(refresh_button)
            
            # Export button
            export_button = QPushButton("Export Logs")
            export_button.clicked.connect(self.export_logs)
            export_button.setToolTip("Export connection logs to CSV")
            controls_layout.addWidget(export_button)
            
            # Auto-refresh combo
            auto_refresh_label = QLabel("Auto-refresh:")
            auto_refresh_combo = QComboBox()
            auto_refresh_combo.addItems(["Off", "1s", "5s", "10s", "30s"])
            auto_refresh_combo.currentTextChanged.connect(self.on_auto_refresh_changed)
            auto_refresh_combo.setToolTip("Set automatic refresh interval")
            controls_layout.addWidget(auto_refresh_label)
            controls_layout.addWidget(auto_refresh_combo)
            
            controls_group.setLayout(controls_layout)
            layout.addWidget(controls_group)
            
            # Connection status table
            self.status_table = QTableWidget()
            self.status_table.setColumnCount(8)  # Added message rate column
            self.status_table.setHorizontalHeaderLabels([
                "Tab", "Status", "Duration", "Last Heartbeat",
                "Messages Received", "Messages Sent", "Message Rate", "Errors"
            ])
            self.status_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            self.status_table.setToolTip("Connection status for all tabs")
            layout.addWidget(self.status_table)
            
            # Message statistics group
            stats_group = QGroupBox("Message Statistics")
            stats_layout = QHBoxLayout()
            
            # Total messages
            self.total_messages_label = QLabel("Total Messages: 0")
            self.total_messages_label.setToolTip("Total messages across all tabs")
            stats_layout.addWidget(self.total_messages_label)
            
            # Error count
            self.error_count_label = QLabel("Errors: 0")
            self.error_count_label.setToolTip("Total errors across all tabs")
            stats_layout.addWidget(self.error_count_label)
            
            # Average latency
            self.latency_label = QLabel("Avg Latency: 0ms")
            self.latency_label.setToolTip("Average message processing latency")
            stats_layout.addWidget(self.latency_label)
            
            # Message rate
            self.message_rate_label = QLabel("Message Rate: 0/s")
            self.message_rate_label.setToolTip("Average messages per second")
            stats_layout.addWidget(self.message_rate_label)
            
            stats_group.setLayout(stats_layout)
            layout.addWidget(stats_group)
            
            self.setLayout(layout)
            self.setWindowTitle("Connection Dashboard")
            
        except Exception as e:
            error_msg = f"Error setting up dashboard UI: {str(e)}"
            self.logger.error(error_msg)
            QMessageBox.critical(self, "Error", error_msg)
            
    def setup_refresh_timer(self):
        """Setup the auto-refresh timer."""
        try:
            self.refresh_timer = QTimer()
            self.refresh_timer.timeout.connect(self.refresh_data)
            self.refresh_timer.setInterval(5000)  # Default to 5s
            self.refresh_timer.start()
        except Exception as e:
            error_msg = f"Error setting up refresh timer: {str(e)}"
            self.logger.error(error_msg)
            QMessageBox.critical(self, "Error", error_msg)
            
    def on_auto_refresh_changed(self, interval: str):
        """Handle auto-refresh interval changes."""
        try:
            if interval == "Off":
                self.refresh_timer.stop()
            else:
                interval_ms = int(interval[:-1]) * 1000
                self.refresh_timer.setInterval(interval_ms)
                self.refresh_timer.start()
        except Exception as e:
            error_msg = f"Error changing refresh interval: {str(e)}"
            self.logger.error(error_msg)
            QMessageBox.critical(self, "Error", error_msg)
            
    def update_connection_data(self, tab: str, data: Dict[str, Any]):
        """Update connection data for a tab."""
        try:
            if not isinstance(data, dict):
                raise ValueError("Data must be a dictionary")
                
            if tab not in self.connection_data:
                self.connection_data[tab] = {
                    'status': False,
                    'start_time': None,
                    'last_heartbeat': None,
                    'messages_received': 0,
                    'messages_sent': 0,
                    'errors': 0,
                    'latencies': [],
                    'message_history': []
                }
                
            # Update data
            self.connection_data[tab].update(data)
            
            # Add to message history
            self.message_history.append({
                'timestamp': datetime.now(),
                'tab': tab,
                'data': data
            })
            
            # Keep only last 1000 messages
            if len(self.message_history) > 1000:
                self.message_history = self.message_history[-1000:]
                
            self.refresh_data()
            
        except Exception as e:
            error_msg = f"Error updating connection data: {str(e)}"
            self.logger.error(error_msg)
            QMessageBox.critical(self, "Error", error_msg)
            
    def refresh_data(self):
        """Refresh the dashboard data."""
        try:
            # Update status table
            self.status_table.setRowCount(len(self.connection_data))
            for row, (tab, data) in enumerate(self.connection_data.items()):
                # Tab name
                self.status_table.setItem(row, 0, QTableWidgetItem(tab))
                
                # Status with color coding
                status_item = QTableWidgetItem("Connected" if data['status'] else "Disconnected")
                status_color = QColor(0, 255, 0) if data['status'] else QColor(255, 0, 0)
                status_item.setBackground(status_color)
                status_item.setToolTip(f"Connection status for {tab}")
                self.status_table.setItem(row, 1, status_item)
                
                # Duration
                duration = "N/A"
                if data['status'] and data['start_time']:
                    duration = str(datetime.now() - data['start_time']).split('.')[0]
                duration_item = QTableWidgetItem(duration)
                duration_item.setToolTip(f"Connection duration for {tab}")
                self.status_table.setItem(row, 2, duration_item)
                
                # Last heartbeat
                last_heartbeat = "Never" if not data['last_heartbeat'] else str(data['last_heartbeat'])
                heartbeat_item = QTableWidgetItem(last_heartbeat)
                heartbeat_item.setToolTip(f"Last heartbeat for {tab}")
                self.status_table.setItem(row, 3, heartbeat_item)
                
                # Messages received
                received_item = QTableWidgetItem(str(data['messages_received']))
                received_item.setToolTip(f"Total messages received by {tab}")
                self.status_table.setItem(row, 4, received_item)
                
                # Messages sent
                sent_item = QTableWidgetItem(str(data['messages_sent']))
                sent_item.setToolTip(f"Total messages sent by {tab}")
                self.status_table.setItem(row, 5, sent_item)
                
                # Message rate
                rate = self.calculate_message_rate(tab)
                rate_item = QTableWidgetItem(f"{rate:.2f}/s")
                rate_item.setToolTip(f"Message rate for {tab}")
                self.status_table.setItem(row, 6, rate_item)
                
                # Errors
                error_item = QTableWidgetItem(str(data['errors']))
                error_item.setBackground(QColor(255, 0, 0) if data['errors'] > 0 else QColor(255, 255, 255))
                error_item.setToolTip(f"Total errors for {tab}")
                self.status_table.setItem(row, 7, error_item)
                
            # Update statistics
            total_messages = sum(data['messages_received'] + data['messages_sent'] 
                               for data in self.connection_data.values())
            total_errors = sum(data['errors'] for data in self.connection_data.values())
            avg_latency = 0
            if any(data['latencies'] for data in self.connection_data.values()):
                all_latencies = [lat for data in self.connection_data.values() 
                               for lat in data['latencies']]
                avg_latency = sum(all_latencies) / len(all_latencies)
                
            self.total_messages_label.setText(f"Total Messages: {total_messages}")
            self.error_count_label.setText(f"Errors: {total_errors}")
            self.latency_label.setText(f"Avg Latency: {avg_latency:.2f}ms")
            
            # Calculate overall message rate
            overall_rate = self.calculate_overall_message_rate()
            self.message_rate_label.setText(f"Message Rate: {overall_rate:.2f}/s")
            
        except Exception as e:
            error_msg = f"Error refreshing dashboard data: {str(e)}"
            self.logger.error(error_msg)
            QMessageBox.critical(self, "Error", error_msg)
            
    def calculate_message_rate(self, tab: str) -> float:
        """Calculate message rate for a specific tab."""
        try:
            if tab not in self.connection_data:
                return 0.0
                
            data = self.connection_data[tab]
            if not data['start_time']:
                return 0.0
                
            duration = (datetime.now() - data['start_time']).total_seconds()
            if duration == 0:
                return 0.0
                
            return (data['messages_received'] + data['messages_sent']) / duration
            
        except Exception as e:
            self.logger.error(f"Error calculating message rate: {str(e)}")
            return 0.0
            
    def calculate_overall_message_rate(self) -> float:
        """Calculate overall message rate across all tabs."""
        try:
            total_messages = 0
            total_duration = 0
            
            for data in self.connection_data.values():
                if data['start_time']:
                    duration = (datetime.now() - data['start_time']).total_seconds()
                    total_duration = max(total_duration, duration)
                    total_messages += data['messages_received'] + data['messages_sent']
                    
            if total_duration == 0:
                return 0.0
                
            return total_messages / total_duration
            
        except Exception as e:
            self.logger.error(f"Error calculating overall message rate: {str(e)}")
            return 0.0
            
    def export_logs(self):
        """Export connection logs to CSV file."""
        try:
            file_name, _ = QFileDialog.getSaveFileName(
                self,
                "Export Connection Logs",
                "",
                "CSV Files (*.csv)"
            )
            
            if not file_name:
                return
                
            with open(file_name, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    'Timestamp', 'Tab', 'Status', 'Messages Received',
                    'Messages Sent', 'Errors', 'Latency'
                ])
                
                for entry in self.message_history:
                    data = entry['data']
                    writer.writerow([
                        entry['timestamp'],
                        entry['tab'],
                        'Connected' if data['status'] else 'Disconnected',
                        data['messages_received'],
                        data['messages_sent'],
                        data['errors'],
                        sum(data['latencies']) / len(data['latencies']) if data['latencies'] else 0
                    ])
                    
            QMessageBox.information(self, "Success", "Logs exported successfully")
            
        except Exception as e:
            error_msg = f"Error exporting logs: {str(e)}"
            self.logger.error(error_msg)
            QMessageBox.critical(self, "Error", error_msg)
            
    def show_dashboard(self):
        """Show the dashboard window."""
        self.show()
        self.raise_()
        self.activateWindow() 