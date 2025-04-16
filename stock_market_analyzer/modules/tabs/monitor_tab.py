import sys
import os
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                            QTableWidget, QTableWidgetItem, QTextEdit, QLabel)
from PyQt6.QtCore import QTimer, Qt
import logging
from typing import Dict, Optional
from datetime import datetime

from ..process_manager import ProcessManager
from .base_tab import BaseTab

class MonitorTab(BaseTab):
    """Tab for monitoring the trading agent and other processes."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.process_manager: Optional[ProcessManager] = None
        self.setup_ui()
        self.setup_monitoring()
        
    def setup_ui(self):
        """Set up the user interface."""
        # Create main layout
        main_layout = QVBoxLayout(self)
        
        # Create splitter for process list and details
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Create process list widget
        self.process_list = QTableWidget()
        self.process_list.setColumnCount(4)
        self.process_list.setHorizontalHeaderLabels([
            "Process", "Status", "PID", "Last Heartbeat"
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
        
        # Create details widget
        details_widget = QWidget()
        details_layout = QVBoxLayout(details_widget)
        
        # Create status label
        self.status_label = QLabel("Status: Not Connected")
        details_layout.addWidget(self.status_label)
        
        # Create log viewer
        self.log_viewer = QTextEdit()
        self.log_viewer.setReadOnly(True)
        details_layout.addWidget(self.log_viewer)
        
        # Create control buttons
        control_layout = QHBoxLayout()
        
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_processes)
        control_layout.addWidget(self.refresh_button)
        
        self.restart_button = QPushButton("Restart Selected")
        self.restart_button.clicked.connect(self.restart_selected)
        control_layout.addWidget(self.restart_button)
        
        self.terminate_button = QPushButton("Terminate Selected")
        self.terminate_button.clicked.connect(self.terminate_selected)
        control_layout.addWidget(self.terminate_button)
        
        details_layout.addLayout(control_layout)
        
        # Add widgets to splitter
        splitter.addWidget(self.process_list)
        splitter.addWidget(details_widget)
        
        # Set splitter sizes
        splitter.setSizes([200, 400])
        
        # Add splitter to main layout
        main_layout.addWidget(splitter)
        
    def setup_monitoring(self):
        """Set up process monitoring."""
        # Create refresh timer
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_processes)
        self.refresh_timer.start(5000)  # Refresh every 5 seconds
        
    def set_process_manager(self, process_manager: ProcessManager):
        """Set the process manager instance."""
        self.process_manager = process_manager
        self.status_label.setText("Status: Connected")
        self.refresh_processes()
        
    def refresh_processes(self):
        """Refresh the process list."""
        if not self.process_manager:
            return
            
        # Clear current items
        self.process_list.setRowCount(0)
        
        # Add processes to the list
        for tab_name in self.process_manager.processes:
            process = self.process_manager.processes[tab_name]
            status = self.process_manager.check_status(tab_name)
            pid = process.pid if process else "N/A"
            last_heartbeat = datetime.fromtimestamp(
                self.process_manager.heartbeats.get(tab_name, 0)
            ).strftime("%Y-%m-%d %H:%M:%S")
            
            row = self.process_list.rowCount()
            self.process_list.insertRow(row)
            
            self.process_list.setItem(row, 0, QTableWidgetItem(tab_name))
            self.process_list.setItem(row, 1, QTableWidgetItem(status))
            self.process_list.setItem(row, 2, QTableWidgetItem(str(pid)))
            self.process_list.setItem(row, 3, QTableWidgetItem(last_heartbeat))
            
    def on_process_selected(self):
        """Handle process selection."""
        if not self.process_manager:
            return
            
        selected_items = self.process_list.selectedItems()
        if not selected_items:
            return
            
        tab_name = selected_items[0].text()
        logs = self.process_manager.get_recent_logs(tab_name)
        if logs:
            self.log_viewer.setPlainText(logs)
        else:
            self.log_viewer.setPlainText("No logs available")
            
    def restart_selected(self):
        """Restart the selected process."""
        if not self.process_manager:
            return
            
        selected_items = self.process_list.selectedItems()
        if not selected_items:
            return
            
        tab_name = selected_items[0].text()
        if self.process_manager.restart_tab(tab_name):
            self.refresh_processes()
            
    def terminate_selected(self):
        """Terminate the selected process."""
        if not self.process_manager:
            return
            
        selected_items = self.process_list.selectedItems()
        if not selected_items:
            return
            
        tab_name = selected_items[0].text()
        if self.process_manager.terminate_tab(tab_name):
            self.refresh_processes()
            
    def cleanup(self):
        """Clean up resources."""
        if self.refresh_timer:
            self.refresh_timer.stop()
        if self.process_manager:
            self.process_manager.cleanup() 