import sys
import os
import logging
import traceback
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QComboBox, QPushButton, QLabel, QFileDialog, QMessageBox, QListWidget,
    QListWidgetItem, QSplitter, QApplication, QSpinBox, QCheckBox,
    QGroupBox, QRadioButton, QTabWidget, QScrollArea, QLineEdit, QTextEdit, QFrame,
    QHeaderView
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from modules.tabs.base_tab import BaseTab
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import uuid
from modules.message_bus import MessageBus

class AnalysisTab(BaseTab):
    """Analysis tab for performing stock analysis."""
    
    def __init__(self, parent=None):
        """Initialize the Analysis tab."""
        # Initialize attributes before parent __init__
        self.analysis_cache = {}
        self.pending_requests = {}
        self._ui_setup_done = False
        self.main_layout = None
        self.ticker_combo = None
        self.analysis_type_combo = None
        self.run_button = None
        self.results_table = None
        self.status_label = None
        
        super().__init__(parent)
        
        # Setup UI after parent initialization
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the UI components."""
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
            
            # Create results table
            results_group = QGroupBox("Analysis Results")
            results_layout = QVBoxLayout()
            results_group.setLayout(results_layout)
            
            self.results_table = QTableWidget()
            self.results_table.setColumnCount(3)
            self.results_table.setHorizontalHeaderLabels([
                "Metric", "Value", "Interpretation"
            ])
            self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            results_layout.addWidget(self.results_table)
            
            self.main_layout.addWidget(results_group)
            
            # Create status bar
            status_layout = QHBoxLayout()
            
            self.status_label = QLabel("Ready")
            self.status_label.setStyleSheet("color: green")
            status_layout.addWidget(self.status_label)
            
            self.main_layout.addLayout(status_layout)
            
            self._ui_setup_done = True
            
        except Exception as e:
            error_msg = f"Error setting up UI: {str(e)}"
            self.logger.error(error_msg)
            if self.status_label:
                self.status_label.setText(error_msg)
                
    def _setup_message_bus_impl(self):
        """Setup message bus subscriptions."""
        super()._setup_message_bus_impl()
        self.message_bus.subscribe("Analysis", self.handle_message)
        
    def run_analysis(self):
        """Run the selected analysis."""
        try:
            ticker = self.ticker_combo.currentText()
            analysis_type = self.analysis_type_combo.currentText()
            
            if not ticker:
                self.status_label.setText("Please select a ticker")
                return
                
            # Generate request ID
            request_id = str(uuid.uuid4())
            
            # Create request
            request = {
                'request_id': request_id,
                'ticker': ticker,
                'analysis_type': analysis_type,
                'timestamp': datetime.now()
            }
            
            # Add to pending requests
            self.pending_requests[request_id] = request
            
            # Publish request
            self.message_bus.publish(
                "Analysis",
                "analysis_request",
                request
            )
            
            # Update UI
            self.status_label.setText(f"Running {analysis_type} for {ticker}")
            self.run_button.setEnabled(False)
            
        except Exception as e:
            error_msg = f"Error running analysis: {str(e)}"
            self.logger.error(error_msg)
            self.status_label.setText(error_msg)
            
    def handle_message(self, sender: str, message_type: str, data: Any):
        """Handle incoming messages."""
        try:
            if message_type == "analysis_response":
                self.handle_analysis_response(sender, data)
            elif message_type == "error":
                self.status_label.setText(f"Error: {data.get('error', 'Unknown error')}")
                
        except Exception as e:
            error_msg = f"Error handling message: {str(e)}"
            self.logger.error(error_msg)
            self.status_label.setText(error_msg)
            
    def handle_analysis_response(self, sender: str, data: Any):
        """Handle analysis response."""
        try:
            request_id = data.get('request_id')
            if request_id in self.pending_requests:
                ticker = self.pending_requests[request_id]['ticker']
                results = data.get('results', [])
                
                if results:
                    # Clear previous results
                    self.results_table.setRowCount(0)
                    
                    # Add new results
                    for row, result in enumerate(results):
                        self.results_table.insertRow(row)
                        self.results_table.setItem(row, 0, QTableWidgetItem(result['metric']))
                        self.results_table.setItem(row, 1, QTableWidgetItem(str(result['value'])))
                        self.results_table.setItem(row, 2, QTableWidgetItem(result['interpretation']))
                    
                    self.status_label.setText(f"Analysis completed for {ticker}")
                else:
                    self.status_label.setText(f"No results available for {ticker}")
                    
                del self.pending_requests[request_id]
                self.run_button.setEnabled(True)
                
        except Exception as e:
            error_msg = f"Error handling analysis response: {str(e)}"
            self.logger.error(error_msg)
            self.status_label.setText(error_msg)
            
    def cleanup(self):
        """Cleanup resources."""
        try:
            super().cleanup()
            self.analysis_cache.clear()
            self.pending_requests.clear()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            
    def closeEvent(self, event):
        """Handle the close event."""
        self.cleanup()
        super().closeEvent(event)

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