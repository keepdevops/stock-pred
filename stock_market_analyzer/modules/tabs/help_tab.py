import sys
import os
import logging
from typing import Any
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QComboBox,
    QTabWidget, QScrollArea, QFrame, QGroupBox, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QTextCursor

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from stock_market_analyzer.modules.message_bus import MessageBus

class HelpTab(QWidget):
    """Help tab for the stock market analyzer."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.message_bus = MessageBus()
        self.logger = logging.getLogger(__name__)
        self.heartbeat_timer = QTimer()
        self.heartbeat_timer.timeout.connect(self.send_heartbeat)
        self.heartbeat_timer.start(5000)  # Send heartbeat every 5 seconds
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the help tab UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Create tab widget
        tab_widget = QTabWidget()
        
        # Create scroll area for each tab
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        
        # Overview tab
        overview_tab = QWidget()
        overview_layout = QVBoxLayout()
        overview_text = QTextEdit()
        overview_text.setReadOnly(True)
        overview_text.setHtml("""
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
        overview_layout.addWidget(overview_text)
        overview_tab.setLayout(overview_layout)
        
        # Data Import tab
        import_tab = QWidget()
        import_layout = QVBoxLayout()
        import_text = QTextEdit()
        import_text.setReadOnly(True)
        import_text.setHtml("""
            <h1>Data Import</h1>
            <p>The Import tab allows you to import data from various sources:</p>
            
            <h2>File Import</h2>
            <p>You can import data from the following file types:</p>
            <ul>
                <li>CSV files</li>
                <li>Excel files (XLSX)</li>
                <li>JSON files</li>
                <li>Parquet files</li>
                <li>DuckDB files</li>
            </ul>
            
            <h2>Database Import</h2>
            <p>You can also import data from various databases:</p>
            <ul>
                <li>SQLite</li>
                <li>PostgreSQL</li>
                <li>MySQL</li>
                <li>DuckDB</li>
            </ul>
            
            <h2>Data Preview</h2>
            <p>After importing, you can preview the data in the table below the import options. The preview shows the first 100 rows of the imported data.</p>
        """)
        import_layout.addWidget(import_text)
        import_tab.setLayout(import_layout)
        
        # Data Analysis tab
        analysis_tab = QWidget()
        analysis_layout = QVBoxLayout()
        analysis_text = QTextEdit()
        analysis_text.setReadOnly(True)
        analysis_text.setHtml("""
            <h1>Data Analysis</h1>
            <p>The Data tab provides tools for analyzing your stock data:</p>
            
            <h2>Charts</h2>
            <p>View your data in various chart types:</p>
            <ul>
                <li>Line charts</li>
                <li>Candlestick charts</li>
                <li>Volume charts</li>
                <li>Technical indicator charts</li>
            </ul>
            
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
        analysis_layout.addWidget(analysis_text)
        analysis_tab.setLayout(analysis_layout)
        
        # Predictions tab
        predictions_tab = QWidget()
        predictions_layout = QVBoxLayout()
        predictions_text = QTextEdit()
        predictions_text.setReadOnly(True)
        predictions_text.setHtml("""
            <h1>Predictions</h1>
            <p>The Predictions tab uses AI to predict future stock movements:</p>
            
            <h2>AI Models</h2>
            <p>Choose from various AI models for predictions:</p>
            <ul>
                <li>GPT-4</li>
                <li>GPT-3.5</li>
                <li>Claude</li>
            </ul>
            
            <h2>Prediction Settings</h2>
            <p>Configure prediction parameters:</p>
            <ul>
                <li>Time horizon</li>
                <li>Confidence threshold</li>
                <li>Technical indicators</li>
            </ul>
            
            <h2>Results</h2>
            <p>View prediction results including:</p>
            <ul>
                <li>Predicted price movements</li>
                <li>Confidence scores</li>
                <li>Supporting analysis</li>
            </ul>
        """)
        predictions_layout.addWidget(predictions_text)
        predictions_tab.setLayout(predictions_layout)
        
        # Trading tab
        trading_tab = QWidget()
        trading_layout = QVBoxLayout()
        trading_text = QTextEdit()
        trading_text.setReadOnly(True)
        trading_text.setHtml("""
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
        trading_layout.addWidget(trading_text)
        trading_tab.setLayout(trading_layout)
        
        # Settings tab
        settings_tab = QWidget()
        settings_layout = QVBoxLayout()
        settings_text = QTextEdit()
        settings_text.setReadOnly(True)
        settings_text.setHtml("""
            <h1>Settings</h1>
            <p>The Settings tab allows you to configure the application:</p>
            
            <h2>Database Settings</h2>
            <p>Configure your database connection:</p>
            <ul>
                <li>Database type (SQLite, PostgreSQL, MySQL, DuckDB)</li>
                <li>Connection details</li>
                <li>Credentials</li>
            </ul>
            
            <h2>Trading Settings</h2>
            <p>Configure trading preferences:</p>
            <ul>
                <li>Trading mode</li>
                <li>Risk parameters</li>
                <li>Auto-trading options</li>
            </ul>
            
            <h2>AI Settings</h2>
            <p>Configure AI model settings:</p>
            <ul>
                <li>Model selection</li>
                <li>Confidence thresholds</li>
            </ul>
        """)
        settings_layout.addWidget(settings_text)
        settings_tab.setLayout(settings_layout)
        
        # Add tabs to tab widget
        tab_widget.addTab(overview_tab, "Overview")
        tab_widget.addTab(import_tab, "Data Import")
        tab_widget.addTab(analysis_tab, "Data Analysis")
        tab_widget.addTab(predictions_tab, "Predictions")
        tab_widget.addTab(trading_tab, "Trading")
        tab_widget.addTab(settings_tab, "Settings")
        
        layout.addWidget(tab_widget)
        
        # Add status label
        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Subscribe to message bus
        self.message_bus.subscribe("Help", self.handle_message)
        
        self.logger.info("Help tab initialized")
        
    def send_heartbeat(self):
        """Send heartbeat message to indicate tab is alive."""
        try:
            self.message_bus.publish("Help", "heartbeat", {"status": "alive"})
        except Exception as e:
            self.logger.error(f"Error sending heartbeat: {str(e)}")
            
    def handle_message(self, sender: str, message_type: str, data: Any):
        """Handle incoming messages."""
        try:
            if message_type == "error":
                error_msg = f"Received error from {sender}: {data}"
                self.logger.error(error_msg)
                self.status_label.setText(f"Error: {error_msg}")
                QMessageBox.critical(self, "Error", error_msg)
            elif message_type == "heartbeat":
                self.status_label.setText(f"Status: Connected to {sender}")
            elif message_type == "shutdown":
                self.logger.info(f"Received shutdown request from {sender}")
                self.close()
                
        except Exception as e:
            error_log = f"Error handling message in Help tab: {str(e)}"
            self.logger.error(error_log)
            
    def publish_message(self, message_type: str, data: Any):
        """Publish a message to the message bus."""
        try:
            self.message_bus.publish("Help", message_type, data)
        except Exception as e:
            error_log = f"Error publishing message from Help tab: {str(e)}"
            self.logger.error(error_log)
            
    def closeEvent(self, event):
        """Handle the close event."""
        try:
            # Stop heartbeat timer
            self.heartbeat_timer.stop()
            
            # Unsubscribe from message bus
            self.message_bus.unsubscribe("Help", self.handle_message)
            
            super().closeEvent(event)
            
        except Exception as e:
            self.logger.error(f"Error in close event: {str(e)}")

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
    import traceback
    main() 