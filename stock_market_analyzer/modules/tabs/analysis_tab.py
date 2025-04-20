import sys
import os
import logging
import traceback
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QComboBox, QPushButton, QLabel, QFileDialog, QMessageBox, QListWidget,
    QListWidgetItem, QSplitter, QApplication, QSpinBox, QCheckBox,
    QGroupBox, QRadioButton, QTabWidget, QScrollArea, QLineEdit, QTextEdit, QFrame,
    QHeaderView, QFormLayout, QGridLayout
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from .base_tab import BaseTab
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import uuid
from ..message_bus import MessageBus
from ..connection_dashboard import ConnectionDashboard
from ..settings import Settings
from ..data_manager import DataManager

class AnalysisTab(BaseTab):
    """Analysis tab for analyzing market data."""
    
    def __init__(self, message_bus: MessageBus, parent=None):
        """Initialize the Analysis tab.
        
        Args:
            message_bus: The message bus for communication.
            parent: The parent widget.
        """
        # Initialize attributes before parent __init__
        self.settings = Settings()
        self.current_color_scheme = self.settings.get_color_scheme()
        self._ui_setup_done = False
        self.ticker_combo = None
        self.analysis_type_combo = None
        self.run_button = None
        self.results_table = None
        self.status_label = None
        self.metrics_labels = {}
        self.analysis_cache = {}
        self.pending_requests = {}
        self.connection_status = {}
        self.connection_start_times = {}
        self.messages_received = 0
        self.messages_sent = 0
        self.errors = 0
        self.message_latencies = []
        self.connection_dashboard = None
        self.data_manager = DataManager()
        self.data_manager.register_listener("AnalysisTab", self._on_data_update)
        
        # Call parent __init__ with message bus
        super().__init__(message_bus, parent)
        
        self.setup_ui()
        
    def _setup_message_bus_impl(self):
        """Setup message bus subscriptions."""
        try:
            # Subscribe to all relevant topics
            self.message_bus.subscribe("Analysis", self.handle_analysis_message)
            self.message_bus.subscribe("Data", self.handle_data_message)
            self.message_bus.subscribe("ConnectionStatus", self.handle_connection_status)
            self.message_bus.subscribe("Heartbeat", self.handle_heartbeat)
            self.message_bus.subscribe("Shutdown", self.handle_shutdown)
            
            # Send initial connection status
            self.message_bus.publish(
                "ConnectionStatus",
                "status_update",
                {
                    "status": "connected",
                    "sender": "AnalysisTab",
                    "timestamp": time.time()
                }
            )
            
            self.logger.debug("Message bus setup completed for Analysis tab")
            
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
            
            # Create controls group
            controls_group = QGroupBox("Analysis Controls")
            controls_layout = QVBoxLayout()
            
            # Top controls
            top_controls = QHBoxLayout()
            
            # Ticker selection
            ticker_layout = QHBoxLayout()
            ticker_layout.addWidget(QLabel("Ticker:"))
            self.ticker_combo = QComboBox()
            self.ticker_combo.setEditable(True)
            self.ticker_combo.setInsertPolicy(QComboBox.InsertPolicy.InsertAlphabetically)
            ticker_layout.addWidget(self.ticker_combo)
            top_controls.addLayout(ticker_layout)
            
            # Analysis type selection
            analysis_layout = QHBoxLayout()
            analysis_layout.addWidget(QLabel("Analysis Type:"))
            self.analysis_type_combo = QComboBox()
            self.analysis_type_combo.addItems([
                "Technical Analysis",
                "Fundamental Analysis",
                "Sentiment Analysis",
                "Risk Analysis"
            ])
            analysis_layout.addWidget(self.analysis_type_combo)
            top_controls.addLayout(analysis_layout)
            
            controls_layout.addLayout(top_controls)
            
            # Action buttons
            button_layout = QHBoxLayout()
            self.run_button = QPushButton("Run Analysis")
            self.run_button.clicked.connect(self.run_analysis)
            button_layout.addWidget(self.run_button)
            controls_layout.addLayout(button_layout)
            
            controls_group.setLayout(controls_layout)
            self.main_layout.addWidget(controls_group)
            
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
            
            # Create results table
            results_group = QGroupBox("Analysis Results")
            results_layout = QVBoxLayout()
            
            self.results_table = QTableWidget()
            self.results_table.setColumnCount(4)
            self.results_table.setHorizontalHeaderLabels([
                "Metric", "Value", "Change", "Signal"
            ])
            self.results_table.horizontalHeader().setSectionResizeMode(
                QHeaderView.ResizeMode.Stretch
            )
            results_layout.addWidget(self.results_table)
            
            results_group.setLayout(results_layout)
            self.main_layout.addWidget(results_group)
            
            # Add status bar
            self.status_label = QLabel("Ready")
            self.status_label.setStyleSheet("color: green")
            self.main_layout.addWidget(self.status_label)
            
            # Create connection dashboard
            self.connection_dashboard = QGroupBox("Connection Status")
            dashboard_layout = QVBoxLayout()
            
            # Add connection status label
            self.connection_status_label = QLabel("Status: Disconnected")
            dashboard_layout.addWidget(self.connection_status_label)
            
            # Add metrics to dashboard
            metrics_layout = QGridLayout()
            metrics = ["Messages Received", "Messages Sent", "Errors", "Average Latency"]
            for i, metric in enumerate(metrics):
                label = QLabel(f"{metric}: 0")
                self.metrics_labels[metric] = label
                metrics_layout.addWidget(label, i // 2, i % 2)
                
            dashboard_layout.addLayout(metrics_layout)
            self.connection_dashboard.setLayout(dashboard_layout)
            self.main_layout.addWidget(self.connection_dashboard)
            
            self._ui_setup_done = True
            self.logger.info("Analysis tab initialized")
            
        except Exception as e:
            self.handle_error("Error setting up UI", e)
            
    def run_analysis(self):
        """Run the selected analysis."""
        try:
            ticker = self.ticker_combo.currentText()
            analysis_type = self.analysis_type_combo.currentText()
            
            if not ticker:
                self.status_label.setText("Please select a ticker")
                return
                
            self.status_label.setText(f"Running {analysis_type} for {ticker}...")
            
            # Publish analysis request
            self.message_bus.publish(
                "Analysis",
                "run_analysis",
                {
                    "ticker": ticker,
                    "analysis_type": analysis_type,
                    "timestamp": time.time()
                }
            )
            
        except Exception as e:
            self.handle_error("Error running analysis", e)
            
    def handle_analysis_message(self, sender: str, message_type: str, data: Dict[str, Any]):
        """Handle analysis-related messages."""
        try:
            if not isinstance(data, dict):
                self.logger.error("Invalid analysis message data format")
                return
                
            self.messages_received += 1
            
            if message_type == "analysis_result":
                self.handle_analysis_result(sender, data)
            elif message_type == "analysis_error":
                self.handle_analysis_error(sender, data)
            else:
                self.logger.warning(f"Unknown analysis message type: {message_type}")
                
            # Update metrics
            self.update_metrics()
            
        except Exception as e:
            self.handle_error("Error handling analysis message", e)
            
    def handle_data_message(self, sender: str, message_type: str, data: Dict[str, Any]):
        """Handle data-related messages."""
        try:
            if not isinstance(data, dict):
                self.logger.error("Invalid data message format")
                return
                
            self.messages_received += 1
            
            if message_type == "data_update":
                print("I got it")  # Print message when receiving data
                
                # Extract metadata
                metadata = data.get("metadata", {})
                symbols = metadata.get("symbols", [])
                row_count = metadata.get("row_count", 0)
                columns = metadata.get("columns", [])
                
                # Update ticker combo with new symbols
                for symbol in symbols:
                    if self.ticker_combo.findText(symbol) == -1:
                        self.ticker_combo.addItem(symbol)
                        self.logger.info(f"Added new symbol to analysis: {symbol}")
                
                # Update status with detailed information
                status_text = f"Received {row_count} rows of data"
                if symbols:
                    status_text += f" for {len(symbols)} symbols"
                if columns:
                    status_text += f" with {len(columns)} columns"
                self.status_label.setText(status_text)
                self.status_label.setStyleSheet("color: green")
                
                # Cache the data for later analysis
                self.analysis_cache.update(data.get("data", {}))
                
            elif message_type == "data_error":
                self.handle_data_error(sender, data)
            else:
                self.logger.warning(f"Unknown data message type: {message_type}")
                
            # Update metrics
            self.update_metrics()
            
        except Exception as e:
            self.handle_error("Error handling data message", e)
            
    def handle_connection_status(self, sender: str, message_type: str, data: Dict[str, Any]):
        """Handle connection status messages."""
        try:
            if not isinstance(data, dict):
                self.logger.error("Invalid connection status data format")
                return
                
            status = data.get("status")
            if status not in ["connected", "disconnected", "error"]:
                self.logger.error(f"Invalid connection status: {status}")
                return
                
            self.connection_status = status
            self.update_connection_dashboard()
            
            # Update metrics
            self.update_metrics()
            
        except Exception as e:
            self.handle_error("Error handling connection status", e)
            
    def handle_heartbeat(self, sender: str, message_type: str, data: Dict[str, Any]):
        """Handle heartbeat messages."""
        try:
            self.messages_received += 1
            self.message_latencies.append(time.time() - data.get("timestamp", time.time()))
            self.update_metrics()
        except Exception as e:
            self.handle_error("Error handling heartbeat", e)
            
    def handle_shutdown(self, sender: str, message_type: str, data: Dict[str, Any]):
        """Handle shutdown messages."""
        try:
            self._is_shutting_down = True
            self.cleanup()
        except Exception as e:
            self.handle_error("Error handling shutdown", e)
            
    def handle_analysis_result(self, sender: str, data: Dict[str, Any]):
        """Handle analysis result messages."""
        try:
            results = data.get("results")
            if not isinstance(results, list):
                self.logger.error("Invalid analysis results format")
                return
                
            # Update results table
            self.update_results_table(results)
            
            # Update status
            self.status_label.setText("Analysis completed successfully")
            self.status_label.setStyleSheet("color: green")
            
        except Exception as e:
            self.handle_error("Error handling analysis result", e)
            
    def handle_analysis_error(self, sender: str, data: Dict[str, Any]):
        """Handle analysis error messages."""
        try:
            error_msg = data.get("error", "Unknown error")
            self.handle_error("Analysis error", Exception(error_msg))
        except Exception as e:
            self.handle_error("Error handling analysis error", e)
            
    def handle_data_error(self, sender: str, data: Dict[str, Any]):
        """Handle data error messages."""
        try:
            error_msg = data.get("error", "Unknown error")
            self.handle_error("Data error", Exception(error_msg))
        except Exception as e:
            self.handle_error("Error handling data error", e)
            
    def update_metrics(self):
        """Update the metrics display."""
        try:
            if not hasattr(self, 'metrics_labels'):
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
            
    def update_results_table(self, results: List[Dict[str, Any]]):
        """Update the results table with analysis data."""
        try:
            # Clear existing table
            self.results_table.setRowCount(0)
            
            # Set up table
            self.results_table.setRowCount(len(results))
            
            # Populate table
            for i, result in enumerate(results):
                # Metric
                metric_item = QTableWidgetItem(result.get("metric", ""))
                metric_item.setFlags(metric_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.results_table.setItem(i, 0, metric_item)
                
                # Value
                value_item = QTableWidgetItem(str(result.get("value", "")))
                value_item.setFlags(value_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.results_table.setItem(i, 1, value_item)
                
                # Change
                change = result.get("change", 0)
                change_item = QTableWidgetItem(f"{change:.2f}%")
                change_item.setFlags(change_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.results_table.setItem(i, 2, change_item)
                
                # Signal
                signal = result.get("signal", "")
                signal_item = QTableWidgetItem(signal)
                signal_item.setFlags(signal_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.results_table.setItem(i, 3, signal_item)
                
            # Resize columns
            self.results_table.resizeColumnsToContents()
            
        except Exception as e:
            self.handle_error("Error updating results table", e)
            
    def update_connection_dashboard(self):
        """Update the connection dashboard with current metrics."""
        try:
            if not self.connection_dashboard:
                return
                
            # Update connection status
            status = "Connected" if self.connection_status == "connected" else "Disconnected"
            self.connection_status_label.setText(f"Status: {status}")
            
            # Update metrics
            self.update_metrics()
            
        except Exception as e:
            self.handle_error("Error updating connection dashboard", e)
            
    def _on_data_update(self, source: str, data: pd.DataFrame, metadata: Dict[str, Any]):
        """Handle data updates from the DataManager."""
        try:
            # Extract metadata
            symbols = metadata.get("symbols", [])
            row_count = metadata.get("row_count", 0)
            columns = metadata.get("columns", [])
            
            # Update ticker combo box
            current_symbols = [self.ticker_combo.itemText(i) for i in range(self.ticker_combo.count())]
            for symbol in symbols:
                if symbol not in current_symbols:
                    self.ticker_combo.addItem(symbol)
                    self.logger.info(f"Added new symbol to analysis: {symbol}")
            
            # Update status
            self.status_label.setText(f"Received {row_count} rows of data with {len(symbols)} symbols")
            self.status_label.setStyleSheet("color: green")
            
            # Cache the data for later analysis
            self.data_cache = data
            
        except Exception as e:
            self.handle_error("Error handling data update", e)
            
    def publish_analysis_results(self, results: Dict[str, Any]):
        """Publish analysis results via DataManager."""
        try:
            # Create metadata
            metadata = {
                "timestamp": time.time(),
                "analysis_type": results.get("type", "unknown"),
                "symbols": results.get("symbols", []),
                "source": "AnalysisTab"
            }
            
            # Convert results to DataFrame if needed
            if isinstance(results.get("data"), pd.DataFrame):
                df = results["data"]
            else:
                df = pd.DataFrame(results)
                
            # Add data to DataManager
            self.data_manager.add_data("AnalysisTab", df, metadata)
            
            # Update status
            self.status_label.setText(f"Published analysis results for {len(metadata['symbols'])} symbols")
            
        except Exception as e:
            self.handle_error("Error publishing analysis results", e)
            
    def cleanup(self):
        """Clean up resources."""
        try:
            # Unregister from DataManager
            self.data_manager.unregister_listener("AnalysisTab")
            
            # Clear caches
            self.analysis_cache.clear()
            self.pending_requests.clear()
            self.connection_status.clear()
            self.connection_start_times.clear()
            
            # Reset metrics
            self.messages_received = 0
            self.messages_sent = 0
            self.errors = 0
            self.message_latencies.clear()
            
            # Clear UI components
            if self.results_table:
                self.results_table.setRowCount(0)
                self.results_table.setColumnCount(0)
                
            if self.ticker_combo:
                self.ticker_combo.clear()
                
            # Reset status
            self.status_label.setText("Ready")
            self.status_label.setStyleSheet("color: green")
            
            # Update metrics display
            self.update_metrics()
            
            # Call parent cleanup
            super().cleanup()
            
            self.logger.info("Analysis tab cleanup completed")
            
        except Exception as e:
            self.handle_error("Error during cleanup", e)
            
    def handle_error(self, message: str, error: Exception):
        """Handle errors with proper logging and UI updates."""
        try:
            # Log error
            self.logger.error(f"{message}: {str(error)}")
            self.logger.debug(traceback.format_exc())
            
            # Update error count
            self.errors += 1
            
            # Update status
            self.status_label.setText(f"Error: {message}")
            self.status_label.setStyleSheet("color: red")
            
            # Update metrics
            self.update_metrics()
            
            # Show error message to user
            QMessageBox.critical(self, "Error", f"{message}\n{str(error)}")
            
        except Exception as e:
            # If error handling fails, log it and continue
            self.logger.error(f"Error in handle_error: {str(e)}")
            self.logger.debug(traceback.format_exc())

def main():
    """Main function for the analysis tab process."""
    app = QApplication(sys.argv)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting analysis tab process")
    
    # Create and show the analysis tab
    window = AnalysisTab()
    window.setWindowTitle("Analysis Tab")
    window.show()
    
    # Start the application event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 