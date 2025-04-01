from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem,
    QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox,
    QProgressBar, QMessageBox, QFileDialog, QSplitter, QDialog, QCheckBox,
    QInputDialog
)
from PyQt5.QtCore import Qt, QTimer, QDateTime, QThread, pyqtSignal
from PyQt5.QtGui import QFont
import logging
import pandas as pd
from typing import Dict, Any, Optional
import os

from .charts import StockChart, TechnicalIndicatorChart
from .realtime import RealTimeDataManager, AsyncTaskManager

class StockGUI(QMainWindow):
    def __init__(self, db, data_loader, ai_agent, trading_agent, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.db = db
        self.data_loader = data_loader
        self.ai_agent = ai_agent
        self.trading_agent = trading_agent
        
        # Initialize data storage
        self.current_data = None
        
        # Initialize managers
        self.realtime_manager = RealTimeDataManager()
        self.async_manager = AsyncTaskManager()
        
        # Connect signals
        self.realtime_manager.price_update.connect(self.on_price_update)
        self.realtime_manager.indicator_update.connect(self.on_indicator_update)
        self.realtime_manager.error_occurred.connect(self.on_realtime_error)
        self.async_manager.task_started.connect(self.on_task_started)
        self.async_manager.task_completed.connect(self.on_task_completed)
        self.async_manager.task_error.connect(self.on_task_error)
        
        # Set up GUI
        self.setup_gui()
        
    def setup_gui(self):
        """Set up the main GUI window."""
        self.setWindowTitle("Stock Market Analyzer")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create tab widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # Add tabs
        self.tabs.addTab(self.create_data_tab(), "Data")
        self.tabs.addTab(self.create_import_tab(), "Import")
        self.tabs.addTab(self.create_analysis_tab(), "Analysis")
        self.tabs.addTab(self.create_charts_tab(), "Charts")
        self.tabs.addTab(self.create_trading_tab(), "Trading")
        self.tabs.addTab(self.create_models_tab(), "Models")
        self.tabs.addTab(self.create_settings_tab(), "Settings")
        
    def create_import_tab(self):
        """Create the import tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Import controls
        controls_group = QGroupBox("Import Controls")
        controls_layout = QVBoxLayout()
        
        # File selection
        file_layout = QHBoxLayout()
        self.file_path = QLineEdit()
        self.file_path.setReadOnly(True)
        file_layout.addWidget(QLabel("File:"))
        file_layout.addWidget(self.file_path)
        
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_file)
        file_layout.addWidget(browse_button)
        controls_layout.addLayout(file_layout)
        
        # Format selection
        format_layout = QHBoxLayout()
        self.format_combo = QComboBox()
        self.format_combo.addItems([
            "CSV", "JSON", "DuckDB", "Keras", "Pandas DataFrame", "Polars DataFrame"
        ])
        format_layout.addWidget(QLabel("Format:"))
        format_layout.addWidget(self.format_combo)
        controls_layout.addLayout(format_layout)
        
        # Preview button
        preview_button = QPushButton("Preview Data")
        preview_button.clicked.connect(self.preview_import_data)
        controls_layout.addWidget(preview_button)
        
        # Import button
        import_button = QPushButton("Import Data")
        import_button.clicked.connect(self.import_data)
        controls_layout.addWidget(import_button)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Preview table
        preview_group = QGroupBox("Data Preview")
        preview_layout = QVBoxLayout()
        
        self.preview_table = QTableWidget()
        self.preview_table.setColumnCount(6)
        self.preview_table.setHorizontalHeaderLabels([
            "Date", "Open", "High", "Low", "Close", "Volume"
        ])
        preview_layout.addWidget(self.preview_table)
        
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        return tab
        
    def create_data_tab(self):
        """Create the data tab."""
        data_widget = QWidget()
        layout = QVBoxLayout(data_widget)
        
        # Data controls
        controls_layout = QHBoxLayout()
        
        # Symbol entry
        self.symbol_entry = QLineEdit()
        self.symbol_entry.setPlaceholderText("Enter stock symbol")
        controls_layout.addWidget(QLabel("Symbol:"))
        controls_layout.addWidget(self.symbol_entry)
        
        # Load button
        self.load_button = QPushButton("Load Data")
        self.load_button.clicked.connect(self.load_data)
        controls_layout.addWidget(self.load_button)
        
        # Real-time toggle
        self.realtime_button = QCheckBox("Start Real-time")
        self.realtime_button.stateChanged.connect(self.toggle_realtime)
        controls_layout.addWidget(self.realtime_button)
        
        layout.addLayout(controls_layout)
        
        # Create splitter for charts and data table
        splitter = QSplitter(Qt.Vertical)
        
        # Charts widget
        charts_widget = QWidget()
        charts_layout = QVBoxLayout(charts_widget)
        
        self.price_chart = StockChart()
        charts_layout.addWidget(self.price_chart)
        
        self.indicator_chart = TechnicalIndicatorChart()
        charts_layout.addWidget(self.indicator_chart)
        
        splitter.addWidget(charts_widget)
        
        # Data table
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(6)
        self.data_table.setHorizontalHeaderLabels([
            "Date", "Open", "High", "Low", "Close", "Volume"
        ])
        self.data_table.horizontalHeader().setStretchLastSection(True)
        splitter.addWidget(self.data_table)
        
        layout.addWidget(splitter)
        
        return data_widget
        
    def toggle_realtime(self, checked: bool):
        """Toggle real-time updates for the current symbol."""
        symbol = self.symbol_entry.text().strip().upper()
        if not symbol:
            QMessageBox.warning(self, "Warning", "Please enter a stock symbol")
            self.realtime_button.setChecked(False)
            return
            
        # Don't start real-time updates for imported data
        if symbol.startswith("IMPORT_"):
            QMessageBox.warning(self, "Warning", "Real-time updates are not available for imported data")
            self.realtime_button.setChecked(False)
            return
            
        if checked:
            self.realtime_manager.start_updates(symbol)
            self.realtime_button.setText("Stop Real-time")
        else:
            self.realtime_manager.stop_updates(symbol)
            self.realtime_button.setText("Start Real-time")
            
    def on_price_update(self, symbol: str, price: float):
        """Handle real-time price updates."""
        try:
            # Update price chart
            timestamp = QDateTime.currentDateTime()
            self.price_chart.add_real_time_point(timestamp, price)
            
            # Update data table
            if self.data_table.rowCount() > 0:
                latest_row = self.data_table.rowCount() - 1
                self.data_table.setItem(latest_row, 4, QTableWidgetItem(f"{price:.2f}"))
                
        except Exception as e:
            self.logger.error(f"Error handling price update: {e}")
            
    def on_indicator_update(self, symbol: str, indicators: Dict[str, float]):
        """Handle real-time indicator updates."""
        try:
            timestamp = QDateTime.currentDateTime()
            self.indicator_chart.add_real_time_point(timestamp, indicators.get('close', 0), indicators)
            
        except Exception as e:
            self.logger.error(f"Error handling indicator update: {e}")
            
    def on_realtime_error(self, error: str):
        """Handle real-time update errors."""
        self.logger.error(f"Real-time error: {error}")
        QMessageBox.warning(self, "Real-time Error", error)
        
    def on_task_started(self, task_name: str):
        """Handle task start."""
        self.logger.info(f"Task started: {task_name}")
        
    def on_task_completed(self, task_name: str, result: Any):
        """Handle task completion."""
        self.logger.info(f"Task completed: {task_name}")
        
        try:
            # Handle import data task completion
            if task_name.startswith("import_data_"):
                self.logger.info(f"Processing import data task completion for {task_name}")
                if isinstance(result, pd.DataFrame):
                    self.logger.info(f"Received DataFrame with {len(result)} rows")
                    # Store the data
                    self.current_data = result
                    self.logger.info("Successfully stored data in current_data")
                    
                    # Update data table
                    self.logger.info("Updating data table display")
                    self.data_table.setRowCount(len(result))
                    for i, (_, row) in enumerate(result.iterrows()):
                        self.data_table.setItem(i, 0, QTableWidgetItem(row['date'].strftime('%Y-%m-%d')))
                        self.data_table.setItem(i, 1, QTableWidgetItem(f"{row['open']:.2f}"))
                        self.data_table.setItem(i, 2, QTableWidgetItem(f"{row['high']:.2f}"))
                        self.data_table.setItem(i, 3, QTableWidgetItem(f"{row['low']:.2f}"))
                        self.data_table.setItem(i, 4, QTableWidgetItem(f"{row['close']:.2f}"))
                        self.data_table.setItem(i, 5, QTableWidgetItem(f"{row['volume']:,.0f}"))
                    
                    # Update charts
                    self.logger.info("Updating charts")
                    self.price_chart.update_data(result)
                    self.indicator_chart.update_data(result)
                    
                    # Generate a unique symbol for the imported data
                    base_name = os.path.splitext(os.path.basename(self.file_path.text()))[0]
                    symbol = f"IMPORT_{base_name[:8].upper()}"
                    self.symbol_entry.setText(symbol)
                    self.logger.info(f"Set symbol to: {symbol}")
                    
                    # Stop any existing real-time updates
                    if self.realtime_button.isChecked():
                        self.logger.info("Stopping real-time updates")
                        self.realtime_button.setChecked(False)
                        self.realtime_manager.stop_updates(self.symbol_entry.text().strip().upper())
                    
                    QMessageBox.information(self, "Success", "Data imported successfully")
                else:
                    self.logger.error(f"Invalid data format received: {type(result)}")
                    raise ValueError("Invalid data format received")
                    
            # Handle analysis task completion
            elif task_name.startswith("analysis_"):
                self.logger.info(f"Processing analysis task completion for {task_name}")
                if isinstance(result, dict):
                    self.logger.info(f"Received analysis results with keys: {list(result.keys())}")
                    
                    # Format analysis results
                    analysis_text = f"Analysis Results for {self.symbol_entry.text()}\n\n"
                    
                    # Add technical analysis results
                    if 'technical' in result:
                        self.logger.info("Processing technical analysis results")
                        analysis_text += "Technical Analysis:\n"
                        tech_results = result['technical']
                        
                        # Moving Averages
                        analysis_text += "\nMoving Averages:\n"
                        analysis_text += f"  SMA (20): {tech_results.get('sma_20', 'N/A'):.2f}\n"
                        analysis_text += f"  SMA (50): {tech_results.get('sma_50', 'N/A'):.2f}\n"
                        analysis_text += f"  SMA (200): {tech_results.get('sma_200', 'N/A'):.2f}\n"
                        
                        # RSI
                        analysis_text += f"\nRSI (14): {tech_results.get('rsi', 'N/A'):.2f}\n"
                        
                        # MACD
                        analysis_text += "\nMACD:\n"
                        analysis_text += f"  MACD: {tech_results.get('macd', 'N/A'):.2f}\n"
                        analysis_text += f"  Signal: {tech_results.get('macd_signal', 'N/A'):.2f}\n"
                        
                        # Bollinger Bands
                        analysis_text += "\nBollinger Bands:\n"
                        analysis_text += f"  Upper: {tech_results.get('bb_upper', 'N/A'):.2f}\n"
                        analysis_text += f"  Middle: {tech_results.get('bb_middle', 'N/A'):.2f}\n"
                        analysis_text += f"  Lower: {tech_results.get('bb_lower', 'N/A'):.2f}\n"
                        
                        # Volume Indicators
                        analysis_text += "\nVolume Indicators:\n"
                        analysis_text += f"  Volume MA: {tech_results.get('volume_ma', 'N/A'):,.0f}\n"
                        analysis_text += f"  Volume Ratio: {tech_results.get('volume_ratio', 'N/A'):.2f}\n"
                        
                        # Price Momentum
                        analysis_text += "\nPrice Momentum:\n"
                        analysis_text += f"  Daily Return: {tech_results.get('daily_return', 'N/A'):.2%}\n"
                        analysis_text += f"  Volatility: {tech_results.get('volatility', 'N/A'):.2%}\n"
                            
                    # Add fundamental analysis results
                    if 'fundamental' in result:
                        self.logger.info("Processing fundamental analysis results")
                        analysis_text += "\nFundamental Analysis:\n"
                        fund_results = result['fundamental']
                        for metric, value in fund_results.items():
                            if isinstance(value, float):
                                analysis_text += f"  {metric}: {value:.2f}\n"
                            else:
                                analysis_text += f"  {metric}: {value}\n"
                            
                    # Add sentiment analysis results
                    if 'sentiment' in result:
                        self.logger.info("Processing sentiment analysis results")
                        analysis_text += "\nSentiment Analysis:\n"
                        sent_results = result['sentiment']
                        for metric, value in sent_results.items():
                            if isinstance(value, float):
                                analysis_text += f"  {metric}: {value:.2f}\n"
                            else:
                                analysis_text += f"  {metric}: {value}\n"
                            
                    # Update analysis display
                    self.logger.info("Updating analysis text display")
                    self.analysis_text.setText(analysis_text)
                    
                    # Show success message without triggering rerun
                    self.logger.info("Showing success message")
                    msg = QMessageBox(self)
                    msg.setWindowTitle("Success")
                    msg.setText("Analysis completed successfully")
                    msg.setStandardButtons(QMessageBox.Ok)
                    msg.setDefaultButton(QMessageBox.Ok)
                    msg.exec_()
                    
                else:
                    self.logger.error(f"Invalid analysis results format: {type(result)}")
                    raise ValueError("Invalid analysis results format")
                    
        except Exception as e:
            self.logger.error(f"Error handling task completion: {e}")
            self.logger.error(f"Error type: {type(e).__name__}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            QMessageBox.critical(self, "Error", f"Failed to process results: {str(e)}")
            
    def on_task_error(self, task_name: str, error: str):
        """Handle task errors."""
        self.logger.error(f"Task error: {task_name} - {error}")
        
        # Handle import data task errors
        if task_name.startswith("import_data_"):
            QMessageBox.critical(self, "Import Error", error)
        
    def load_data(self):
        """Load stock data for the specified symbol."""
        try:
            symbol = self.symbol_entry.text().strip().upper()
            if not symbol:
                QMessageBox.warning(self, "Warning", "Please enter a stock symbol")
                return
                
            # Show loading message
            self.analysis_text.setText(f"Loading data for {symbol}...")
            
            # Create async task for data loading
            async def load_data_task():
                data = await self.data_loader.load_data_async(symbol)
                if data is None or data.empty:
                    raise ValueError(f"No data available for symbol {symbol}")
                return data
                
            self.async_manager.create_task(f"load_data_{symbol}", load_data_task())
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")
            self.analysis_text.setText(f"Error: {str(e)}")
            
    def update_data_display(self, data: pd.DataFrame):
        """Update the data display with new data."""
        try:
            # Store the current data
            self.current_data = data
            
            # Update data table
            self.data_table.setRowCount(len(data))
            
            for i, (_, row) in enumerate(data.iterrows()):
                self.data_table.setItem(i, 0, QTableWidgetItem(row['date'].strftime('%Y-%m-%d')))
                self.data_table.setItem(i, 1, QTableWidgetItem(f"{row['open']:.2f}"))
                self.data_table.setItem(i, 2, QTableWidgetItem(f"{row['high']:.2f}"))
                self.data_table.setItem(i, 3, QTableWidgetItem(f"{row['low']:.2f}"))
                self.data_table.setItem(i, 4, QTableWidgetItem(f"{row['close']:.2f}"))
                self.data_table.setItem(i, 5, QTableWidgetItem(f"{row['volume']:,.0f}"))
                
            # Update charts
            self.price_chart.update_data(data)
            self.indicator_chart.update_data(data)
            
            # Update symbol entry with filename
            symbol = os.path.splitext(os.path.basename(self.file_path.text()))[0]
            self.symbol_entry.setText(symbol)
            
        except Exception as e:
            self.logger.error(f"Error updating data display: {e}")
            QMessageBox.critical(self, "Error", f"Failed to update display: {str(e)}")
            
    def closeEvent(self, event):
        """Handle application close."""
        try:
            # Stop all real-time updates
            self.realtime_manager.stop_updates(self.symbol_entry.text().strip().upper())
            
            # Clean up async task manager
            self.async_manager.cleanup()
            
            super().closeEvent(event)
            
        except Exception as e:
            self.logger.error(f"Error during close: {e}")
            super().closeEvent(event)

    def create_analysis_tab(self):
        """Create the analysis tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Analysis controls
        controls_layout = QHBoxLayout()
        
        # Analysis type selection
        self.analysis_type = QComboBox()
        self.analysis_type.addItems(['Technical', 'Fundamental', 'Sentiment'])
        controls_layout.addWidget(QLabel("Analysis Type:"))
        controls_layout.addWidget(self.analysis_type)
        
        # Run analysis button
        self.run_analysis_btn = QPushButton("Run Analysis")
        self.run_analysis_btn.clicked.connect(self.run_analysis)
        controls_layout.addWidget(self.run_analysis_btn)
        
        layout.addLayout(controls_layout)
        
        # Analysis results
        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        layout.addWidget(self.analysis_text)
        
        return tab
        
    def create_charts_tab(self):
        """Create the charts tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Chart controls
        controls_layout = QHBoxLayout()
        
        # Chart type selection
        self.chart_type = QComboBox()
        self.chart_type.addItems(['Price', 'Volume', 'Technical Indicators'])
        controls_layout.addWidget(QLabel("Chart Type:"))
        controls_layout.addWidget(self.chart_type)
        
        # Update chart button
        self.update_chart_btn = QPushButton("Update Chart")
        self.update_chart_btn.clicked.connect(self.update_chart)
        controls_layout.addWidget(self.update_chart_btn)
        
        layout.addLayout(controls_layout)
        
        # Create chart widgets
        self.price_chart = StockChart()
        self.technical_chart = TechnicalIndicatorChart()
        
        # Add charts to layout
        layout.addWidget(self.price_chart)
        layout.addWidget(self.technical_chart)
        
        return tab

    def create_trading_tab(self):
        """Set up the trading tab."""
        trading_widget = QWidget()
        layout = QVBoxLayout(trading_widget)
        
        # Trading controls
        controls_layout = QHBoxLayout()
        
        # Position size input
        position_label = QLabel("Position Size:")
        self.position_entry = QLineEdit()
        self.position_entry.setPlaceholderText("Enter position size")
        controls_layout.addWidget(position_label)
        controls_layout.addWidget(self.position_entry)
        
        # Buy/Sell buttons
        buy_button = QPushButton("Buy")
        buy_button.clicked.connect(self.execute_buy)
        sell_button = QPushButton("Sell")
        sell_button.clicked.connect(self.execute_sell)
        
        controls_layout.addWidget(buy_button)
        controls_layout.addWidget(sell_button)
        
        layout.addLayout(controls_layout)
        
        # Trading history table
        self.trading_table = QTableWidget()
        self.trading_table.setColumnCount(5)
        self.trading_table.setHorizontalHeaderLabels([
            "Date", "Type", "Symbol", "Size", "Price"
        ])
        self.trading_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.trading_table)
        
        self.tabs.addTab(trading_widget, "Trading")
        
        return trading_widget
        
    def create_models_tab(self):
        """Set up the model management tab."""
        model_widget = QWidget()
        layout = QVBoxLayout(model_widget)
        
        # Model selection
        selection_layout = QHBoxLayout()
        model_label = QLabel("Active Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(self.ai_agent.get_available_models())
        self.model_combo.currentTextChanged.connect(self.on_model_change)
        selection_layout.addWidget(model_label)
        selection_layout.addWidget(self.model_combo)
        
        # Refresh button
        refresh_button = QPushButton("Refresh Models")
        refresh_button.clicked.connect(self.refresh_models)
        selection_layout.addWidget(refresh_button)
        
        layout.addLayout(selection_layout)
        
        # Model parameters
        param_group = QGroupBox("Model Parameters")
        param_layout = QVBoxLayout()
        self.param_widgets = {}
        self.create_param_widgets(param_layout)
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        # Training controls
        train_group = QGroupBox("Training")
        train_layout = QVBoxLayout()
        
        # Training parameters
        param_layout = QHBoxLayout()
        
        # Epochs
        epochs_label = QLabel("Epochs:")
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(100)
        param_layout.addWidget(epochs_label)
        param_layout.addWidget(self.epochs_spin)
        
        # Batch size
        batch_label = QLabel("Batch Size:")
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 256)
        self.batch_spin.setValue(32)
        param_layout.addWidget(batch_label)
        param_layout.addWidget(self.batch_spin)
        
        # Sequence length (for LSTM)
        seq_label = QLabel("Sequence Length:")
        self.seq_spin = QSpinBox()
        self.seq_spin.setRange(1, 100)
        self.seq_spin.setValue(10)
        param_layout.addWidget(seq_label)
        param_layout.addWidget(self.seq_spin)
        
        train_layout.addLayout(param_layout)
        
        # Training buttons
        button_layout = QHBoxLayout()
        train_button = QPushButton("Train Model")
        train_button.clicked.connect(self.train_model)
        tune_button = QPushButton("Tune Hyperparameters")
        tune_button.clicked.connect(self.tune_model)
        predict_button = QPushButton("Make Prediction")
        predict_button.clicked.connect(self.make_prediction)
        save_button = QPushButton("Save Model")
        save_button.clicked.connect(self.save_model)
        
        button_layout.addWidget(train_button)
        button_layout.addWidget(tune_button)
        button_layout.addWidget(predict_button)
        button_layout.addWidget(save_button)
        train_layout.addLayout(button_layout)
        
        train_group.setLayout(train_layout)
        layout.addWidget(train_group)
        
        # Model status
        status_group = QGroupBox("Model Status")
        status_layout = QVBoxLayout()
        self.model_status = QTextEdit()
        self.model_status.setReadOnly(True)
        status_layout.addWidget(self.model_status)
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        return model_widget
        
    def create_settings_tab(self):
        """Set up the settings tab."""
        settings_widget = QWidget()
        layout = QVBoxLayout(settings_widget)
        
        # API settings
        api_group = QGroupBox("API Settings")
        api_layout = QVBoxLayout()
        
        api_key_label = QLabel("API Key:")
        self.api_key_entry = QLineEdit()
        self.api_key_entry.setEchoMode(QLineEdit.Password)
        api_layout.addWidget(api_key_label)
        api_layout.addWidget(self.api_key_entry)
        
        api_group.setLayout(api_layout)
        layout.addWidget(api_group)
        
        # Save button
        save_button = QPushButton("Save Settings")
        save_button.clicked.connect(self.save_settings)
        layout.addWidget(save_button)
        
        self.tabs.addTab(settings_widget, "Settings")
        
        return settings_widget
        
    def create_param_widgets(self, layout):
        """Create parameter input widgets based on the active model."""
        # Clear existing widgets
        for widget in self.param_widgets.values():
            layout.removeWidget(widget)
        self.param_widgets.clear()
        
        # Get model parameters
        model = self.ai_agent.get_active_model()
        if model is None:
            return
            
        # Create widgets for each parameter
        for param, value in model.__dict__.items():
            if not param.startswith('_'):
                param_layout = QHBoxLayout()
                label = QLabel(f"{param}:")
                
                if isinstance(value, int):
                    widget = QSpinBox()
                    widget.setRange(1, 1000)
                    widget.setValue(value)
                elif isinstance(value, float):
                    widget = QDoubleSpinBox()
                    widget.setRange(0.0, 1000.0)
                    widget.setValue(value)
                    widget.setSingleStep(0.1)
                else:
                    widget = QLineEdit()
                    widget.setText(str(value))
                    
                param_layout.addWidget(label)
                param_layout.addWidget(widget)
                layout.addLayout(param_layout)
                self.param_widgets[param] = widget
                
    def on_model_change(self, model_id):
        """Handle model selection change."""
        try:
            self.ai_agent.set_active_model(model_id)
            self.create_param_widgets(self.param_widgets['layout'])
            self.update_model_status()
        except Exception as e:
            self.logger.error(f"Error changing model: {e}")
            QMessageBox.critical(self, "Error", f"Failed to change model: {str(e)}")
            
    def run_analysis(self):
        """Run analysis on the loaded data."""
        try:
            if self.current_data is None:
                self.logger.warning("No data available for analysis")
                QMessageBox.warning(self, "Warning", "No data available for analysis. Please import or load data first.")
                return
                
            # Validate data format
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in self.current_data.columns]
            if missing_columns:
                self.logger.error(f"Missing required columns for analysis: {missing_columns}")
                QMessageBox.warning(self, "Warning", f"Missing required columns: {', '.join(missing_columns)}")
                return
                
            # Get analysis type
            analysis_type = self.analysis_type.currentText()
            self.logger.info(f"Starting {analysis_type} analysis on {len(self.current_data)} rows of data")
            
            # Check if analysis is already running
            task_name = f"analysis_{analysis_type.lower()}"
            if task_name in self.async_manager.tasks:
                self.logger.warning(f"Analysis task {task_name} is already running")
                QMessageBox.warning(self, "Warning", "Analysis is already running. Please wait for it to complete.")
                return
            
            # Create async task for analysis
            async def analysis_task():
                try:
                    self.logger.info(f"Running {analysis_type} analysis")
                    # Run analysis based on type
                    if analysis_type == 'Technical':
                        self.logger.info("Calculating technical indicators")
                        results = self.ai_agent.analyze_technical(self.current_data)
                        self.logger.info("Technical analysis completed")
                    elif analysis_type == 'Fundamental':
                        self.logger.info("Calculating fundamental metrics")
                        results = self.ai_agent.analyze_fundamental(self.current_data)
                        self.logger.info("Fundamental analysis completed")
                    elif analysis_type == 'Sentiment':
                        self.logger.info("Calculating sentiment metrics")
                        results = self.ai_agent.analyze_sentiment(self.current_data)
                        self.logger.info("Sentiment analysis completed")
                    else:
                        self.logger.error(f"Unknown analysis type: {analysis_type}")
                        raise ValueError(f"Unknown analysis type: {analysis_type}")
                        
                    self.logger.info(f"Analysis completed successfully with {len(results)} results")
                    return results
                    
                except Exception as e:
                    self.logger.error(f"Analysis failed: {str(e)}")
                    raise ValueError(f"Analysis failed: {str(e)}")
                    
            # Create and run the task
            self.logger.info(f"Creating analysis task: {task_name}")
            self.async_manager.create_task(task_name, analysis_task())
            
        except Exception as e:
            self.logger.error(f"Error in run_analysis: {e}")
            QMessageBox.critical(self, "Error", f"Analysis failed: {str(e)}")
            
    def execute_buy(self):
        """Execute a buy order."""
        try:
            # Get the current symbol
            symbol = self.symbol_entry.text().strip().upper()
            if not symbol:
                QMessageBox.warning(self, "Warning", "Please enter a stock symbol")
                return
                
            # Get the current price from the data table
            if self.data_table.rowCount() == 0:
                QMessageBox.warning(self, "Warning", "No data available. Please load data first.")
                return
                
            # Get the most recent price
            latest_row = self.data_table.rowCount() - 1
            latest_values = [self.data_table.item(latest_row, col).text() for col in range(self.data_table.columnCount())]
            current_price = float(latest_values[4])  # Close price
            
            # Get current balance
            current_balance = self.trading_agent.get_balance()
            
            # Calculate maximum position size based on risk management
            max_position_size = self.trading_agent.calculate_position_size(symbol, current_price)
            
            # Get position size from user
            size_str = self.position_entry.text().strip()
            if not size_str:
                # If no size specified, use the calculated maximum
                size = max_position_size
                self.position_entry.clear()
                self.position_entry.insert(0, f"{size:.2f}")
            else:
                try:
                    size = float(size_str)
                    if size <= 0:
                        QMessageBox.warning(self, "Warning", "Position size must be greater than 0")
                        return
                    if size > max_position_size:
                        QMessageBox.warning(
                            self, 
                            "Warning", 
                            f"Position size too large. Maximum allowed: {max_position_size:.2f} shares\n"
                            f"Current balance: ${current_balance:.2f}\n"
                            f"Current price: ${current_price:.2f}"
                        )
                        return
                except ValueError:
                    messagebox.showwarning("Warning", "Please enter a valid number for position size")
                    return
            
            # Place the buy order
            if self.trading_agent.place_buy_order(symbol, size, current_price):
                self.update_trading_display()
                messagebox.showinfo(
                    "Success",
                    f"Buy order placed successfully:\n"
                    f"Symbol: {symbol}\n"
                    f"Size: {size:.2f} shares\n"
                    f"Price: ${current_price:.2f}\n"
                    f"Total: ${(size * current_price):.2f}\n"
                    f"Remaining balance: ${(current_balance - size * current_price):.2f}"
                )
            else:
                messagebox.showerror("Error", "Failed to place buy order")
            
        except Exception as e:
            self.logger.error(f"Error executing buy order: {e}")
            messagebox.showerror("Error", f"Buy order failed: {str(e)}")
            
    def execute_sell(self):
        """Execute a sell order."""
        try:
            # Get the current symbol
            symbol = self.symbol_entry.get().strip().upper()
            if not symbol:
                messagebox.showwarning("Warning", "Please enter a stock symbol")
                return
                
            # Get the current price from the data tree
            if not self.data_tree.get_children():
                messagebox.showwarning("Warning", "No data available. Please load data first.")
                return
                
            # Get the most recent price
            latest_item = self.data_tree.get_children()[-1]
            latest_values = self.data_tree.item(latest_item)['values']
            current_price = float(latest_values[4])  # Close price
            
            # Get position size
            size_str = self.position_entry.get().strip()
            if not size_str:
                messagebox.showwarning("Warning", "Please enter a position size")
                return
                
            try:
                size = float(size_str)
                if size <= 0:
                    messagebox.showwarning("Warning", "Position size must be greater than 0")
                    return
            except ValueError:
                messagebox.showwarning("Warning", "Please enter a valid number for position size")
                return
                
            self.trading_agent.place_sell_order(symbol, size, current_price)
            self.update_trading_display()
            
        except Exception as e:
            self.logger.error(f"Error executing sell order: {e}")
            messagebox.showerror("Error", f"Sell order failed: {str(e)}")
            
    def save_settings(self):
        """Save the current settings."""
        try:
            # Implementation depends on what settings need to be saved
            messagebox.showinfo("Success", "Settings saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving settings: {e}")
            messagebox.showerror("Error", f"Failed to save settings: {str(e)}")
            
    def update_model_status(self):
        """Update the model status display."""
        model = self.ai_agent.get_active_model()
        if model is None:
            self.model_status.setText("No model selected")
            return
            
        status = f"Model Type: {model.__class__.__name__}\n"
        status += f"Parameters:\n"
        
        # Get model parameters
        if hasattr(model, 'get_config'):
            # Keras model
            config = model.get_config()
            for key, value in config.items():
                if not key.startswith('_'):
                    status += f"  {key}: {value}\n"
        else:
            # Scikit-learn model
            for key, value in model.get_params().items():
                status += f"  {key}: {value}\n"
                
        self.model_status.setText(status)
        
    def show_progress_dialog(self, title, message):
        """Show a progress dialog."""
        dialog = QMessageBox(self)
        dialog.setWindowTitle(title)
        dialog.setText(message)
        dialog.setStandardButtons(QMessageBox.NoButton)
        dialog.show()
        return dialog

    def train_model(self):
        """Train the active model."""
        try:
            # Get model parameters
            model = self.ai_agent.get_active_model()
            if model is None:
                QMessageBox.warning(self, "Warning", "No model selected")
                return
            
            # Get training parameters
            epochs = self.epochs_spin.value()
            batch_size = self.batch_spin.value()
            
            # Train the model
            self.ai_agent.train_model(model, epochs, batch_size)
            
            # Update model status
            self.update_model_status()
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            QMessageBox.critical(self, "Error", f"Training failed: {str(e)}")
            
    def tune_model(self):
        """Tune the hyperparameters of the active model."""
        try:
            # Get model parameters
            model = self.ai_agent.get_active_model()
            if model is None:
                QMessageBox.warning(self, "Warning", "No model selected")
                return
            
            # Tune the model
            self.ai_agent.tune_model(model)
            
            # Update model status
            self.update_model_status()
            
        except Exception as e:
            self.logger.error(f"Error tuning model: {e}")
            QMessageBox.critical(self, "Error", f"Tuning failed: {str(e)}")

    def update_chart(self):
        """Update the charts based on the selected chart type."""
        try:
            if self.current_data is None:
                QMessageBox.warning(self, "Warning", "No data available to display")
                return
                
            chart_type = self.chart_type.currentText()
            
            if chart_type == 'Price':
                self.price_chart.update_data(self.current_data)
                self.price_chart.show()
                self.technical_chart.hide()
            elif chart_type == 'Volume':
                # Create a copy of the data with volume as the main metric
                volume_data = self.current_data.copy()
                volume_data['close'] = volume_data['volume']
                self.price_chart.update_data(volume_data)
                self.price_chart.show()
                self.technical_chart.hide()
            elif chart_type == 'Technical Indicators':
                # Calculate technical indicators
                data = self.current_data.copy()
                data['ma5'] = data['close'].rolling(window=5).mean()
                data['ma20'] = data['close'].rolling(window=20).mean()
                data['rsi'] = self.calculate_rsi(data['close'])
                
                self.technical_chart.update_data(data)
                self.price_chart.hide()
                self.technical_chart.show()
                
        except Exception as e:
            self.logger.error(f"Error updating chart: {e}")
            QMessageBox.critical(self, "Error", f"Error updating chart: {str(e)}")
            
    def calculate_rsi(self, prices, period=14):
        """Calculate the Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def make_prediction(self):
        """Make predictions using the active model."""
        try:
            if self.current_data is None:
                QMessageBox.warning(self, "Warning", "No data available for prediction")
                return
                
            # Get predictions
            predictions = self.ai_agent.predict(self.current_data)
            
            if predictions is not None:
                # Update the chart with predictions
                self.price_chart.add_predictions(predictions)
                
                # Show prediction results
                latest_pred = predictions[-1][0]
                latest_actual = self.current_data['close'].iloc[-1]
                prediction_text = f"Latest Prediction: ${latest_pred:.2f}\n"
                prediction_text += f"Latest Actual: ${latest_actual:.2f}\n"
                prediction_text += f"Difference: ${(latest_pred - latest_actual):.2f}"
                
                self.model_status.append(prediction_text)
                
            else:
                QMessageBox.warning(self, "Warning", "Failed to make predictions")
                
        except Exception as e:
            self.logger.error(f"Error making predictions: {e}")
            QMessageBox.critical(self, "Error", f"Prediction failed: {str(e)}")

    def show_import_dialog(self):
        """Show dialog for importing data from various formats."""
        try:
            dialog = QDialog(self)
            dialog.setWindowTitle("Import Data")
            layout = QVBoxLayout(dialog)
            
            # File selection
            file_layout = QHBoxLayout()
            self.file_path = QLineEdit()
            self.file_path.setReadOnly(True)
            file_layout.addWidget(QLabel("File:"))
            file_layout.addWidget(self.file_path)
            
            browse_button = QPushButton("Browse")
            browse_button.clicked.connect(self.browse_file)
            file_layout.addWidget(browse_button)
            layout.addLayout(file_layout)
            
            # Format selection
            format_layout = QHBoxLayout()
            self.format_combo = QComboBox()
            self.format_combo.addItems([
                "CSV", "JSON", "DuckDB", "Keras", "Pandas DataFrame", "Polars DataFrame"
            ])
            format_layout.addWidget(QLabel("Format:"))
            format_layout.addWidget(self.format_combo)
            layout.addLayout(format_layout)
            
            # Import button
            import_button = QPushButton("Import")
            import_button.clicked.connect(lambda: self.import_data(dialog))
            layout.addWidget(import_button)
            
            dialog.exec_()
            
        except Exception as e:
            self.logger.error(f"Error showing import dialog: {e}")
            QMessageBox.critical(self, "Error", f"Failed to show import dialog: {str(e)}")
            
    def browse_file(self):
        """Open file browser dialog."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Data File",
                "",
                "All Files (*.*);;CSV Files (*.csv);;JSON Files (*.json);;DuckDB Files (*.db);;HDF5 Files (*.h5)"
            )
            if file_path:
                self.file_path.setText(file_path)
                
        except Exception as e:
            self.logger.error(f"Error browsing file: {e}")
            QMessageBox.critical(self, "Error", f"Failed to browse file: {str(e)}")
            
    def import_data(self):
        """Import data from the selected file."""
        try:
            file_path = self.file_path.text()
            if not file_path:
                self.logger.warning("No file selected for import")
                QMessageBox.warning(self, "Warning", "Please select a file")
                return
                
            format_type = self.format_combo.currentText()
            self.logger.info(f"Starting data import from {file_path} in {format_type} format")
            
            # Create async task for data import
            async def import_data_task():
                try:
                    self.logger.info(f"Loading data from {file_path}")
                    # Load data based on format
                    if format_type == "CSV":
                        data = pd.read_csv(file_path)
                        self.logger.info(f"Successfully loaded CSV data with {len(data)} rows")
                    elif format_type == "JSON":
                        self.logger.info("Attempting to load JSON data")
                        try:
                            # Read the file content first to check for issues
                            with open(file_path, 'r') as f:
                                content = f.read()
                                self.logger.info(f"File content preview: {content[:500]}")
                            
                            data = pd.read_json(file_path)
                            self.logger.info(f"Successfully loaded JSON data with {len(data)} rows")
                            self.logger.info(f"JSON data columns: {data.columns.tolist()}")
                            
                            # Validate data structure
                            if data.empty:
                                raise ValueError("JSON file contains no data")
                                
                            # Check for required columns
                            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
                            missing_columns = [col for col in required_columns if col not in data.columns]
                            
                            if missing_columns:
                                self.logger.warning(f"Missing required columns: {missing_columns}")
                                self.logger.info(f"Available columns: {data.columns.tolist()}")
                                
                                # Try to find similar column names
                                column_mapping = {}
                                for col in missing_columns:
                                    # Look for similar column names
                                    similar_cols = [c for c in data.columns if col in c.lower()]
                                    if similar_cols:
                                        column_mapping[col] = similar_cols[0]
                                        self.logger.info(f"Mapping column {similar_cols[0]} to {col}")
                                    else:
                                        raise ValueError(f"Missing required column: {col}")
                                
                                # Rename columns according to mapping
                                data = data.rename(columns={v: k for k, v in column_mapping.items()})
                                self.logger.info("Successfully mapped column names")
                            
                            # Ensure date column is datetime
                            self.logger.info("Converting date column to datetime")
                            data['date'] = pd.to_datetime(data['date'])
                            
                            # Sort by date
                            self.logger.info("Sorting data by date")
                            data = data.sort_values('date')
                            
                            # Validate data types
                            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                            for col in numeric_columns:
                                if not pd.api.types.is_numeric_dtype(data[col]):
                                    self.logger.warning(f"Column {col} is not numeric, attempting conversion")
                                    data[col] = pd.to_numeric(data[col], errors='coerce')
                                    if data[col].isna().any():
                                        raise ValueError(f"Column {col} contains invalid numeric values")
                            
                            self.logger.info("Data validation completed successfully")
                            return data
                            
                        except Exception as e:
                            self.logger.error(f"Error loading JSON data: {str(e)}")
                            raise ValueError(f"Failed to load JSON data: {str(e)}")
                            
                    elif format_type == "DuckDB":
                        try:
                            import duckdb
                            self.logger.info("Connecting to DuckDB file")
                            conn = duckdb.connect(file_path)
                            # Get list of tables
                            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                            if not tables:
                                self.logger.error("No tables found in DuckDB file")
                                raise ValueError("No tables found in DuckDB file")
                            
                            # Get the first table's data
                            table_name = tables[0][0]
                            self.logger.info(f"Loading data from table: {table_name}")
                            data = conn.execute(f"SELECT * FROM {table_name}").df()
                            
                            # Close the connection
                            conn.close()
                            self.logger.info(f"Successfully loaded DuckDB data with {len(data)} rows")
                            
                            # Check if we got any data
                            if data.empty:
                                self.logger.error(f"No data found in table '{table_name}'")
                                raise ValueError(f"No data found in table '{table_name}'")
                                
                        except Exception as e:
                            self.logger.error(f"Error reading DuckDB file: {str(e)}")
                            raise ValueError(f"Error reading DuckDB file: {str(e)}")
                            
                    elif format_type == "Keras":
                        import tensorflow as tf
                        self.logger.info("Loading Keras model data")
                        data = pd.DataFrame(tf.keras.models.load_model(file_path).predict())
                        self.logger.info(f"Successfully loaded Keras data with {len(data)} rows")
                    elif format_type == "Pandas DataFrame":
                        data = pd.read_pickle(file_path)
                        self.logger.info(f"Successfully loaded Pandas DataFrame with {len(data)} rows")
                    elif format_type == "Polars DataFrame":
                        import polars as pl
                        self.logger.info("Loading Polars DataFrame")
                        data = pl.read_csv(file_path).to_pandas()
                        self.logger.info(f"Successfully loaded Polars DataFrame with {len(data)} rows")
                    
                    # Final validation
                    if data is None or data.empty:
                        raise ValueError("No data was loaded")
                        
                    self.logger.info("Data import task completed successfully")
                    return data
                    
                except Exception as e:
                    self.logger.error(f"Error in import_data_task: {str(e)}")
                    self.logger.error(f"Error type: {type(e).__name__}")
                    import traceback
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
                    raise ValueError(f"Error importing data: {str(e)}")
                    
            # Create the task
            task_name = f"import_data_{format_type}"
            self.logger.info(f"Creating import task: {task_name}")
            
            # Create and start the task
            self.async_manager.create_task(task_name, import_data_task())
            
        except Exception as e:
            self.logger.error(f"Error in import_data: {e}")
            self.logger.error(f"Error type: {type(e).__name__}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            QMessageBox.critical(self, "Error", f"Failed to import data: {str(e)}")
            
    def preview_import_data(self):
        """Preview the imported data."""
        try:
            file_path = self.file_path.text()
            if not file_path:
                QMessageBox.warning(self, "Warning", "Please select a file")
                return
                
            format_type = self.format_combo.currentText()
            
            # Load data based on format
            if format_type == "CSV":
                data = pd.read_csv(file_path)
            elif format_type == "JSON":
                data = pd.read_json(file_path)
            elif format_type == "DuckDB":
                try:
                    import duckdb
                    conn = duckdb.connect(file_path)
                    # Get list of tables
                    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                    if not tables:
                        raise ValueError("No tables found in DuckDB file")
                    
                    # Get the first table's data
                    table_name = tables[0][0]
                    data = conn.execute(f"SELECT * FROM {table_name}").df()
                    
                    # Close the connection
                    conn.close()
                    
                    # Check if we got any data
                    if data.empty:
                        raise ValueError(f"No data found in table '{table_name}'")
                        
                except Exception as e:
                    raise ValueError(f"Error reading DuckDB file: {str(e)}")
                    
            elif format_type == "Keras":
                import tensorflow as tf
                data = pd.DataFrame(tf.keras.models.load_model(file_path).predict())
            elif format_type == "Pandas DataFrame":
                data = pd.read_pickle(file_path)
            elif format_type == "Polars DataFrame":
                import polars as pl
                data = pl.read_csv(file_path).to_pandas()
            
            # Update preview table
            self.preview_table.setRowCount(len(data))
            for i, (_, row) in enumerate(data.iterrows()):
                for j, value in enumerate(row):
                    self.preview_table.setItem(i, j, QTableWidgetItem(str(value)))
            
            # Update symbol entry with filename
            symbol = os.path.splitext(os.path.basename(file_path))[0]
            self.symbol_entry.setText(symbol)
            
        except Exception as e:
            self.logger.error(f"Error previewing import data: {e}")
            QMessageBox.critical(self, "Error", f"Failed to preview data: {str(e)}")

    def refresh_models(self):
        """Refresh the list of available models."""
        try:
            # Reload available models
            self.ai_agent.load_available_models()
            
            # Update combo box
            current_model = self.model_combo.currentText()
            self.model_combo.clear()
            self.model_combo.addItems(self.ai_agent.get_available_models())
            
            # Restore previous selection if still available
            if current_model in self.ai_agent.get_available_models():
                self.model_combo.setCurrentText(current_model)
                
            self.update_model_status()
            
        except Exception as e:
            self.logger.error(f"Error refreshing models: {e}")
            QMessageBox.critical(self, "Error", f"Failed to refresh models: {str(e)}")
            
    def save_model(self):
        """Save the current model."""
        try:
            model = self.ai_agent.get_active_model()
            if model is None:
                QMessageBox.warning(self, "Warning", "No model selected")
                return
                
            # Get model name from user
            name, ok = QInputDialog.getText(self, "Save Model", "Enter model name:")
            if ok and name:
                self.ai_agent.save_model(model, name)
                self.refresh_models()
                QMessageBox.information(self, "Success", f"Model saved as: {name}")
                
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save model: {str(e)}") 