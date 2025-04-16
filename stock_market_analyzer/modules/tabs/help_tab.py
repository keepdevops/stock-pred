import sys
import os
import logging
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QComboBox,
    QTabWidget, QScrollArea, QFrame, QGroupBox, QMessageBox,
    QTableWidget, QTableWidgetItem, QSplitter, QSpinBox,
    QDoubleSpinBox, QCheckBox, QHeaderView, QDateEdit
)
from PyQt6.QtCore import Qt, QTimer, QDate
from PyQt6.QtGui import QFont, QTextCursor
import uuid

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from ..message_bus import MessageBus
from .base_tab import BaseTab

class HelpTab(BaseTab):
    """Help tab for the stock market analyzer."""
    
    def __init__(self, parent=None):
        """Initialize the Help tab."""
        super().__init__(parent)
        self.message_bus = MessageBus()
        self.logger = logging.getLogger(__name__)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the help tab UI."""
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create scroll area for each tab
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        
        # Help tab
        help_tab = QWidget()
        help_layout = QVBoxLayout()
        
        # Add help UI elements
        self.help_text = QTextEdit()
        self.help_text.setReadOnly(True)
        self.help_text.setHtml("""
            <h1>Stock Market Analyzer Help</h1>
            <p>Welcome to the Stock Market Analyzer application. This tool helps you analyze stock market data and make informed trading decisions.</p>
            <h2>Features</h2>
            <ul>
                <li>Data Import: Import data from various sources including CSV, Excel, JSON, and databases</li>
                <li>Analysis: Perform technical and fundamental analysis on stock data</li>
                <li>Charts: Visualize stock data and analysis results</li>
                <li>Models: Train and use machine learning models for predictions</li>
                <li>Trading: Execute trades based on analysis and predictions</li>
            </ul>
            <h2>Getting Started</h2>
            <ol>
                <li>Import your stock data using the Import tab</li>
                <li>Analyze the data using the Analysis tab</li>
                <li>View the results in the Charts tab</li>
                <li>Use the Models tab to make predictions</li>
                <li>Execute trades using the Trading tab</li>
            </ol>
        """)
        help_layout.addWidget(self.help_text)
        
        help_tab.setLayout(help_layout)
        
        # Add tabs to tab widget
        self.tab_widget.addTab(help_tab, "Help")
        
        # Add tab widget to main layout
        self.main_layout.addWidget(self.tab_widget)
        
        # Subscribe to message bus
        self.message_bus.subscribe("Help", self.handle_message)
        
        self.logger.info("Help tab initialized")
        
    def process_message(self, sender: str, message_type: str, data: Any):
        """Process incoming messages."""
        try:
            if message_type == "help_request":
                self.handle_help_request(sender, data)
            elif message_type == "help_response":
                self.handle_help_response(sender, data)
            elif message_type == "error":
                self.logger.error(f"Error from {sender}: {data.get('error', 'Unknown error')}")
            elif message_type == "heartbeat":
                self.status_label.setText("Connected")
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
            self.logger.error(traceback.format_exc())
            
    def handle_help_request(self, sender: str, data: Any):
        """Handle help request from other tabs."""
        try:
            request_id = data.get('request_id')
            topic = data.get('topic')
            
            if not all([request_id, topic]):
                self.logger.error("Invalid help request")
                return
                
            # Get help content for the topic
            help_content = self.get_help_content(topic)
            
            # Send response
            self.message_bus.publish(
                "Help",
                "help_response",
                {
                    'request_id': request_id,
                    'topic': topic,
                    'content': help_content
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error handling help request: {str(e)}")
            
    def handle_help_response(self, sender: str, data: Any):
        """Handle help response from other tabs."""
        try:
            request_id = data.get('request_id')
            if request_id in self.pending_requests:
                topic = self.pending_requests[request_id]['topic']
                content = data.get('content', '')
                
                # Cache the help content
                self.help_cache[topic] = content
                
                # Update UI with help content
                self.update_help_content(topic, content)
                
        except Exception as e:
            self.logger.error(f"Error handling help response: {str(e)}")
            
    def get_help_content(self, topic: str) -> str:
        """Get help content for a topic."""
        if topic in self.help_cache:
            return self.help_cache[topic]
            
        # Default help content
        return f"No help content available for topic: {topic}"
        
    def update_help_content(self, topic: str, content: str):
        """Update the UI with help content."""
        # This method can be overridden by subclasses to update their specific UI
        pass
        
    def cleanup(self):
        """Cleanup resources."""
        super().cleanup()
        self.help_cache.clear()
        self.pending_requests.clear()

    def create_general_help_tab(self):
        """Create the general help tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        
        text = QTextEdit()
        text.setReadOnly(True)
        text.setHtml("""
            <h1>Stock Market Analyzer</h1>
            <p>Welcome to the Stock Market Analyzer application. This tool helps you analyze stock market data, make predictions, and manage your trading strategies.</p>
            
            <h2>Features</h2>
            <ul>
                <li>Data Import: Import data from various sources including CSV, Excel, and databases</li>
                <li>Data Analysis: View and analyze stock data with interactive charts</li>
                <li>Predictions: Get AI-powered predictions for stock movements</li>
                <li>Trading: Manage your trading strategies and execute trades</li>
                <li>Settings: Configure application settings and preferences</li>
            </ul>
            
            <h2>Getting Started</h2>
            <p>To get started, use the Import tab to load your data, then navigate to the Data tab to view and analyze it. You can make predictions using the Predictions tab and manage your trading strategies in the Trading tab.</p>
        """)
        layout.addWidget(text)
        tab.setLayout(layout)
        return tab
        
    def create_data_help_tab(self):
        """Create the data help tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        
        text = QTextEdit()
        text.setReadOnly(True)
        text.setHtml("""
            <h1>Data Management</h1>
            <p>The Data tab provides tools for managing and analyzing your stock data:</p>
            
            <h2>Data Import</h2>
            <p>You can import data from various sources:</p>
            <ul>
                <li>CSV files</li>
                <li>Excel files (XLSX)</li>
                <li>JSON files</li>
                <li>Parquet files</li>
                <li>DuckDB files</li>
                <li>SQLite databases</li>
                <li>PostgreSQL databases</li>
                <li>MySQL databases</li>
            </ul>
            
            <h2>Data Preview</h2>
            <p>After importing, you can preview the data in the table below the import options. The preview shows the first 100 rows of the imported data.</p>
        """)
        layout.addWidget(text)
        tab.setLayout(layout)
        return tab
        
    def create_analysis_help_tab(self):
        """Create the analysis help tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        
        text = QTextEdit()
        text.setReadOnly(True)
        text.setHtml("""
            <h1>Data Analysis</h1>
            <p>The Analysis tab provides tools for analyzing your stock data:</p>
            
            <h2>Technical Analysis</h2>
            <p>Apply various technical indicators to your data:</p>
            <ul>
                <li>Moving averages</li>
                <li>RSI</li>
                <li>MACD</li>
                <li>Bollinger Bands</li>
            </ul>
            
            <h2>Data Filtering</h2>
            <p>Filter and sort your data by various criteria:</p>
            <ul>
                <li>Date range</li>
                <li>Price range</li>
                <li>Volume</li>
                <li>Technical indicators</li>
            </ul>
        """)
        layout.addWidget(text)
        tab.setLayout(layout)
        return tab
        
    def create_charts_help_tab(self):
        """Create the charts help tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        
        text = QTextEdit()
        text.setReadOnly(True)
        text.setHtml("""
            <h1>Charts</h1>
            <p>The Charts tab provides visualization tools for your stock data:</p>
            
            <h2>Chart Types</h2>
            <p>View your data in various chart types:</p>
            <ul>
                <li>Line charts</li>
                <li>Candlestick charts</li>
                <li>Volume charts</li>
                <li>Technical indicator charts</li>
            </ul>
            
            <h2>Chart Controls</h2>
            <p>Customize your charts with various controls:</p>
            <ul>
                <li>Zoom in/out</li>
                <li>Pan</li>
                <li>Add/remove indicators</li>
                <li>Change time periods</li>
            </ul>
        """)
        layout.addWidget(text)
        tab.setLayout(layout)
        return tab
        
    def create_trading_help_tab(self):
        """Create the trading help tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        
        text = QTextEdit()
        text.setReadOnly(True)
        text.setHtml("""
            <h1>Trading</h1>
            <p>The Trading tab helps you manage your trading strategies:</p>
            
            <h2>Trading Modes</h2>
            <p>Choose between different trading modes:</p>
            <ul>
                <li>Paper Trading: Practice trading without real money</li>
                <li>Live Trading: Execute real trades with your broker</li>
            </ul>
            
            <h2>Risk Management</h2>
            <p>Configure risk management settings:</p>
            <ul>
                <li>Maximum risk per trade</li>
                <li>Stop loss levels</li>
                <li>Take profit targets</li>
            </ul>
            
            <h2>Auto-Trading</h2>
            <p>Enable auto-trading to execute trades automatically based on your strategies and AI predictions.</p>
        """)
        layout.addWidget(text)
        tab.setLayout(layout)
        return tab

def main():
    """Main function for the help tab process."""
    # Ensure QApplication instance exists
    app = QApplication.instance() 
    if not app: 
        app = QApplication(sys.argv)

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("Starting help tab process")
    
    # Create and show the help tab
    try:
        window = HelpTab()
        window.setWindowTitle("Help Tab")
        window.show()
    except Exception as e:
         logger.error(f"Failed to create or show HelpTab window: {e}")
         logger.error(traceback.format_exc())
         sys.exit(1)

    if __name__ == "__main__":
        sys.exit(app.exec())

if __name__ == "__main__":
    main() 