from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem,
    QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox,
    QProgressBar, QMessageBox, QFileDialog, QSplitter, QDialog, QCheckBox,
    QInputDialog, QGridLayout, QApplication
)
from PyQt5.QtCore import Qt, QTimer, QDateTime, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QFont
import logging
import pandas as pd
from typing import Dict, Any, Optional
import os

from .charts import StockChart, TechnicalIndicatorChart
from .realtime import RealTimeDataManager, AsyncTaskManager

class ModelTrainingWorker(QObject):
    """Worker class for model training in a separate thread."""
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, ai_agent, data, model_type, params):
        super().__init__()
        self.ai_agent = ai_agent
        self.data = data
        self.model_type = model_type
        self.params = params
        
    def run(self):
        """Run the model training process."""
        try:
            self.log.emit(f"Starting {self.model_type} model training")
            
            # Create model
            model = self.ai_agent.create_model(self.model_type, self.params)
            self.log.emit("Model created successfully")
            
            # Prepare data
            X, y = self.ai_agent.prepare_training_data(self.data)
            self.log.emit(f"Training data prepared: X shape {X.shape}, y shape {y.shape}")
            
            # Train model
            history = self.ai_agent.train_model(model, X, y)
            self.log.emit("Model training completed successfully")
            self.progress.emit(100)
            
            # Signal completion
            self.finished.emit()
            
        except Exception as e:
            self.log.emit(f"Error during training: {str(e)}")
            self.error.emit(str(e))

class StockGUI(QMainWindow):
    # Define signals at class level
    training_progress_signal = pyqtSignal(int)
    training_log_signal = pyqtSignal(str)
    training_complete_signal = pyqtSignal()
    training_error_signal = pyqtSignal(str)
    
    def __init__(self, db, data_loader, ai_agent, trading_agent, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.db = db
        self.data_loader = data_loader
        self.ai_agent = ai_agent
        self.trading_agent = trading_agent
        self.current_data = None
        self.current_symbol = None
        self.training_thread = None
        self.is_training = False
        
        # Initialize managers
        self.realtime_manager = RealTimeDataManager()
        self.async_manager = AsyncTaskManager()
        
        # Set up GUI first
        self.setup_gui()
        
        # Connect manager signals
        self.realtime_manager.price_update.connect(self.on_price_update)
        self.realtime_manager.indicator_update.connect(self.on_indicator_update)
        self.realtime_manager.error_occurred.connect(self.on_realtime_error)
        self.async_manager.task_started.connect(self.on_task_started)
        self.async_manager.task_completed.connect(self.on_task_completed)
        self.async_manager.task_error.connect(self.on_task_error)
        
        # Connect training signals to slots
        self.training_progress_signal.connect(self.training_progress.setValue)
        self.training_log_signal.connect(self.training_log.append)
        self.training_complete_signal.connect(self._on_training_complete)
        self.training_error_signal.connect(self._on_training_error)
        
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
        self.tabs.addTab(self.create_prediction_tab(), "Prediction")
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
            # Handle load_data task completion
            if task_name.startswith("load_data_"):
                # Extract symbol from task name
                symbol = task_name.replace("load_data_", "")
                self.current_symbol = symbol
                self.logger.info(f"Set current_symbol to: {symbol}")
                
                # Update data
                self.current_data = result
                
                # Update data table
                self.data_table.setRowCount(len(result))
                for i, (_, row) in enumerate(result.iterrows()):
                    self.data_table.setItem(i, 0, QTableWidgetItem(row['date'].strftime('%Y-%m-%d')))
                    self.data_table.setItem(i, 1, QTableWidgetItem(f"{row['open']:.2f}"))
                    self.data_table.setItem(i, 2, QTableWidgetItem(f"{row['high']:.2f}"))
                    self.data_table.setItem(i, 3, QTableWidgetItem(f"{row['low']:.2f}"))
                    self.data_table.setItem(i, 4, QTableWidgetItem(f"{row['close']:.2f}"))
                    self.data_table.setItem(i, 5, QTableWidgetItem(f"{row['volume']:,.0f}"))
                
                # Update charts
                self.price_chart.update_data(result)
                self.indicator_chart.update_data(result)
                
                # Show success message
                QMessageBox.information(self, "Success", f"Data loaded for {symbol}")
                
            # Handle import data task completion
            elif task_name.startswith("import_data_"):
                self.logger.info(f"Processing import data task completion for {task_name}")
                if isinstance(result, pd.DataFrame):
                    self.logger.info(f"Received DataFrame with {len(result)} rows")
                    self.logger.info(f"DataFrame columns: {result.columns.tolist()}")
                    
                    # Check for required columns and map if necessary
                    required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
                    missing_columns = [col for col in required_columns if col not in result.columns]
                    
                    if missing_columns:
                        self.logger.warning(f"Missing required columns: {missing_columns}")
                        self.logger.info(f"Available columns: {result.columns.tolist()}")
                        
                        # Try to find similar column names
                        column_mapping = {}
                        for col in missing_columns:
                            # Look for similar column names
                            similar_cols = [c for c in result.columns if col in c.lower()]
                            if similar_cols:
                                column_mapping[col] = similar_cols[0]
                                self.logger.info(f"Mapping column {similar_cols[0]} to {col}")
                            else:
                                raise ValueError(f"Missing required column: {col}")
                        
                        # Rename columns according to mapping
                        result = result.rename(columns={v: k for k, v in column_mapping.items()})
                        self.logger.info("Successfully mapped column names")
                    
                    # Ensure date column is datetime
                    self.logger.info("Converting date column to datetime")
                    result['date'] = pd.to_datetime(result['date'])
                    
                    # Sort by date
                    self.logger.info("Sorting data by date")
                    result = result.sort_values('date')
                    
                    # Validate data types
                    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                    for col in numeric_columns:
                        if not pd.api.types.is_numeric_dtype(result[col]):
                            self.logger.warning(f"Column {col} is not numeric, attempting conversion")
                            result[col] = pd.to_numeric(result[col], errors='coerce')
                            if result[col].isna().any():
                                raise ValueError(f"Column {col} contains invalid numeric values")
                    
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
                    self.current_symbol = symbol
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
        """Set up the models tab with model selection and training controls."""
        try:
            self.logger.info("Setting up models tab")
            self.models_tab = QWidget()
            layout = QVBoxLayout()
            
            # Model Selection
            model_selection_layout = QHBoxLayout()
            model_selection_label = QLabel("Select Model:")
            self.model_combo = QComboBox()
            self.model_combo.addItems(self.ai_agent.get_available_models())
            self.model_combo.currentTextChanged.connect(self.on_model_change)
            model_selection_layout.addWidget(model_selection_label)
            model_selection_layout.addWidget(self.model_combo)
            layout.addLayout(model_selection_layout)
            
            # Model Type Selection
            self.logger.info("Added model type selection")
            model_type_layout = QHBoxLayout()
            model_type_label = QLabel("Model Type:")
            self.model_type_combo = QComboBox()
            self.model_type_combo.addItems(['LSTM', 'XGBoost', 'Transformer'])
            model_type_layout.addWidget(model_type_label)
            model_type_layout.addWidget(self.model_type_combo)
            layout.addLayout(model_type_layout)
            
            # Model Description
            self.model_description = QTextEdit()
            self.model_description.setReadOnly(True)
            self.model_description.setMaximumHeight(100)
            self.model_description.setPlainText(
                "LSTM: Long Short-Term Memory neural network for time series prediction\n"
                "XGBoost: Gradient boosting model for regression\n"
                "Transformer: Attention-based model for sequence prediction"
            )
            layout.addWidget(self.model_description)
            
            # Model Status
            self.model_status = QTextEdit()
            self.model_status.setReadOnly(True)
            self.model_status.setMaximumHeight(100)
            layout.addWidget(self.model_status)
            
            # Model Parameters
            self.logger.info("Added LSTM parameters")
            self.lstm_params = QWidget()
            lstm_layout = QGridLayout()
            lstm_layout.addWidget(QLabel("Input Dimension:"), 0, 0)
            self.lstm_input_dim = QSpinBox()
            self.lstm_input_dim.setRange(1, 100)
            self.lstm_input_dim.setValue(5)
            lstm_layout.addWidget(self.lstm_input_dim, 0, 1)
            
            lstm_layout.addWidget(QLabel("Hidden Dimension:"), 1, 0)
            self.lstm_hidden_dim = QSpinBox()
            self.lstm_hidden_dim.setRange(32, 512)
            self.lstm_hidden_dim.setValue(64)
            lstm_layout.addWidget(self.lstm_hidden_dim, 1, 1)
            
            lstm_layout.addWidget(QLabel("Number of Layers:"), 2, 0)
            self.lstm_num_layers = QSpinBox()
            self.lstm_num_layers.setRange(1, 5)
            self.lstm_num_layers.setValue(2)
            lstm_layout.addWidget(self.lstm_num_layers, 2, 1)
            
            lstm_layout.addWidget(QLabel("Dropout:"), 3, 0)
            self.lstm_dropout = QDoubleSpinBox()
            self.lstm_dropout.setRange(0.0, 0.5)
            self.lstm_dropout.setValue(0.2)
            self.lstm_dropout.setSingleStep(0.1)
            lstm_layout.addWidget(self.lstm_dropout, 3, 1)
            
            self.lstm_params.setLayout(lstm_layout)
            layout.addWidget(self.lstm_params)
            
            self.logger.info("Added XGBoost parameters")
            self.xgb_params = QWidget()
            xgb_layout = QGridLayout()
            xgb_layout.addWidget(QLabel("Learning Rate:"), 0, 0)
            self.xgb_learning_rate = QDoubleSpinBox()
            self.xgb_learning_rate.setRange(0.01, 1.0)
            self.xgb_learning_rate.setValue(0.1)
            self.xgb_learning_rate.setSingleStep(0.01)
            xgb_layout.addWidget(self.xgb_learning_rate, 0, 1)
            
            xgb_layout.addWidget(QLabel("Max Depth:"), 1, 0)
            self.xgb_max_depth = QSpinBox()
            self.xgb_max_depth.setRange(3, 10)
            self.xgb_max_depth.setValue(6)
            xgb_layout.addWidget(self.xgb_max_depth, 1, 1)
            
            xgb_layout.addWidget(QLabel("Number of Trees:"), 2, 0)
            self.xgb_n_trees = QSpinBox()
            self.xgb_n_trees.setRange(50, 500)
            self.xgb_n_trees.setValue(100)
            xgb_layout.addWidget(self.xgb_n_trees, 2, 1)
            
            self.xgb_params.setLayout(xgb_layout)
            layout.addWidget(self.xgb_params)
            
            self.logger.info("Added Transformer parameters")
            self.transformer_params = QWidget()
            transformer_layout = QGridLayout()
            transformer_layout.addWidget(QLabel("Input Dimension:"), 0, 0)
            self.transformer_input_dim = QSpinBox()
            self.transformer_input_dim.setRange(1, 100)
            self.transformer_input_dim.setValue(5)
            transformer_layout.addWidget(self.transformer_input_dim, 0, 1)
            
            transformer_layout.addWidget(QLabel("Hidden Dimension:"), 1, 0)
            self.transformer_hidden_dim = QSpinBox()
            self.transformer_hidden_dim.setRange(64, 1024)
            self.transformer_hidden_dim.setValue(256)
            transformer_layout.addWidget(self.transformer_hidden_dim, 1, 1)
            
            transformer_layout.addWidget(QLabel("Number of Heads:"), 2, 0)
            self.transformer_n_heads = QSpinBox()
            self.transformer_n_heads.setRange(1, 8)
            self.transformer_n_heads.setValue(4)
            transformer_layout.addWidget(self.transformer_n_heads, 2, 1)
            
            transformer_layout.addWidget(QLabel("Number of Layers:"), 3, 0)
            self.transformer_n_layers = QSpinBox()
            self.transformer_n_layers.setRange(1, 6)
            self.transformer_n_layers.setValue(2)
            transformer_layout.addWidget(self.transformer_n_layers, 3, 1)
            
            self.transformer_params.setLayout(transformer_layout)
            layout.addWidget(self.transformer_params)
            
            # Training Controls
            self.logger.info("Added training controls")
            training_layout = QHBoxLayout()
            
            self.train_button = QPushButton("Train Model")
            self.train_button.clicked.connect(self.train_model)
            training_layout.addWidget(self.train_button)
            
            self.stop_button = QPushButton("Stop Training")
            self.stop_button.clicked.connect(self.stop_training)
            self.stop_button.setEnabled(False)
            training_layout.addWidget(self.stop_button)
            
            layout.addLayout(training_layout)
            
            # Progress Bar
            self.training_progress = QProgressBar()
            self.training_progress.setRange(0, 100)
            layout.addWidget(self.training_progress)
            
            # Training Log
            self.training_log = QTextEdit()
            self.training_log.setReadOnly(True)
            layout.addWidget(self.training_log)
            
            # Model Management
            model_management_layout = QHBoxLayout()
            
            # Save Model Button
            self.save_model_button = QPushButton("Save Model")
            self.save_model_button.clicked.connect(self.save_model)
            self.save_model_button.setEnabled(False)  # Disabled until model is trained
            model_management_layout.addWidget(self.save_model_button)
            
            # Load Model Button
            self.load_model_button = QPushButton("Load Model")
            self.load_model_button.clicked.connect(self.load_model)
            model_management_layout.addWidget(self.load_model_button)
            
            # Refresh Models Button
            refresh_button = QPushButton("Refresh Models")
            refresh_button.clicked.connect(self.refresh_models)
            model_management_layout.addWidget(refresh_button)
            
            layout.addLayout(model_management_layout)
            
            # Set layout for the models tab
            self.logger.info("Models tab layout set")
            self.models_tab.setLayout(layout)
            
            self.logger.info("Models tab setup completed successfully")
            return self.models_tab
            
        except Exception as e:
            self.logger.error(f"Error setting up models tab: {e}")
            raise

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
                # Construct the model path
                model_path = os.path.join(self.ai_agent.models_dir, f"{name}.keras")
                self.ai_agent.save_model(model_path, model)
                self.refresh_models()
                QMessageBox.information(self, "Success", f"Model saved as: {name}")
                
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save model: {str(e)}")

    def load_model(self):
        """Load a saved model."""
        try:
            available_models = self.ai_agent.get_available_models()
            if not available_models:
                QMessageBox.warning(self, "Warning", "No saved models available!")
                return
                
            # Show model selection dialog
            model_name, ok = QInputDialog.getItem(
                self, "Load Model", "Select model to load:",
                available_models, 0, False
            )
            
            if ok and model_name:
                try:
                    self.ai_agent.set_active_model(model_name)
                    QMessageBox.information(self, "Success", f"Model loaded: {model_name}")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
                    
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            QMessageBox.critical(self, "Error", f"Error loading model: {str(e)}")

    def train_model(self):
        """Start model training in a separate thread."""
        try:
            if self.current_data is None or self.current_data.empty:
                QMessageBox.warning(self, "Warning", "No data available for training!")
                return
                
            self.logger.info("Starting model training")
            
            # Get model type and parameters
            model_type = self.model_type_combo.currentText()
            self.logger.info(f"Selected model type: {model_type}")
            
            # Collect model parameters based on type
            if model_type == 'LSTM':
                params = {
                    'input_dim': self.lstm_input_dim.value(),
                    'hidden_dim': self.lstm_hidden_dim.value(),
                    'num_layers': self.lstm_num_layers.value(),
                    'dropout': self.lstm_dropout.value()
                }
            elif model_type == 'XGBoost':
                params = {
                    'learning_rate': self.xgb_learning_rate.value(),
                    'max_depth': self.xgb_max_depth.value(),
                    'n_trees': self.xgb_n_trees.value()
                }
            else:  # Transformer
                params = {
                    'input_dim': self.transformer_input_dim.value(),
                    'hidden_dim': self.transformer_hidden_dim.value(),
                    'n_heads': self.transformer_n_heads.value(),
                    'n_layers': self.transformer_n_layers.value()
                }
                
            self.logger.info(f"Model parameters: {params}")
            
            # Create and start training thread
            self.training_thread = QThread()
            self.training_worker = ModelTrainingWorker(
                self.ai_agent,
                self.current_data,
                model_type,
                params
            )
            self.training_worker.moveToThread(self.training_thread)
            
            # Connect signals
            self.training_thread.started.connect(self.training_worker.run)
            self.training_worker.finished.connect(self.training_thread.quit)
            self.training_worker.progress.connect(self.update_training_progress)
            self.training_worker.log.connect(self.update_training_log)
            self.training_worker.error.connect(self.handle_training_error)
            
            # Update UI state
            self.train_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.training_progress.setValue(0)
            self.training_log.clear()
            
            # Start training
            self.logger.info("Training thread started")
            self.training_thread.start()
            
        except Exception as e:
            self.logger.error(f"Error starting model training: {e}")
            QMessageBox.critical(self, "Error", f"Error starting model training: {str(e)}")

    def update_training_progress(self, value):
        """Update the training progress bar."""
        self.training_progress.setValue(value)
        
    def update_training_log(self, message):
        """Update the training log display."""
        self.training_log.append(message)
        
    def handle_training_error(self, error):
        """Handle training errors."""
        self.logger.error(f"Error in training thread: {error}")
        QMessageBox.critical(self, "Training Error", str(error))
        self.stop_training()
        
    def stop_training(self):
        """Stop the model training process."""
        try:
            if hasattr(self, 'training_thread') and self.training_thread.isRunning():
                self.training_thread.terminate()
                self.training_thread.wait()
                self.logger.info("Training stopped by user")
                
            # Reset UI state
            self.train_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.training_progress.setValue(0)
            self.save_model_button.setEnabled(True)  # Enable save button after training
            
        except Exception as e:
            self.logger.error(f"Error stopping training: {e}")
            QMessageBox.critical(self, "Error", f"Error stopping training: {str(e)}")
        
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
            self.update_model_status()
        except Exception as e:
            self.logger.error(f"Error changing model: {e}")
            QMessageBox.critical(self, "Error", f"Failed to change model: {str(e)}")
            
    def run_analysis(self):
        """Run the selected analysis on the current data."""
        try:
            if self.current_data is None or self.current_data.empty:
                QMessageBox.warning(self, "Warning", "No data loaded")
                return
                
            # Get the selected analysis type
            analysis_type = self.analysis_type.currentText()
            self.logger.info(f"Running {analysis_type} analysis")
            
            # Create a unique task name
            task_name = f"analysis_{analysis_type}_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
            
            # Define the analysis task
            async def analysis_task():
                try:
                    # Perform the analysis based on the type
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
            # Get order details
            symbol = self.symbol_entry.text().strip().upper()
            size_str = self.position_entry.text().strip()
            
            # Validate inputs
            if not symbol or not size_str:
                QMessageBox.warning(self, "Warning", "Please enter both symbol and position size")
                return
                
            try:
                size = float(size_str)
                if size <= 0:
                    raise ValueError("Position size must be positive")
            except ValueError:
                QMessageBox.warning(self, "Warning", "Please enter a valid number for position size")
                return
            
            # Get current price
            if self.current_data is None or self.current_data.empty:
                QMessageBox.warning(self, "Warning", "No price data available")
                return
                
            current_price = self.current_data['close'].iloc[-1]
            
            # Calculate total value
            total_value = size * current_price
            
            # Create order
            order = {
                'type': 'BUY',
                'symbol': symbol,
                'size': size,
                'price': current_price,
                'total_value': total_value,
                'timestamp': pd.Timestamp.now()
            }
            
            # Add to orders list
            self.orders.append(order)
            
            # Update orders table
            self.update_orders_table()
            
            # Show success message
            QMessageBox.information(self, "Success", f"Buy order executed:\nSymbol: {symbol}\nSize: {size}\nPrice: ${current_price:.2f}\nTotal: ${total_value:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error executing buy order: {str(e)}")
            QMessageBox.critical(self, "Error", f"Buy order failed: {str(e)}")
            
    def execute_sell(self):
        """Execute a sell order."""
        try:
            # Get order details
            symbol = self.symbol_entry.text().strip().upper()
            size_str = self.position_entry.text().strip()
            
            # Validate inputs
            if not symbol or not size_str:
                QMessageBox.warning(self, "Warning", "Please enter both symbol and position size")
                return
                
            try:
                size = float(size_str)
                if size <= 0:
                    raise ValueError("Position size must be positive")
            except ValueError:
                QMessageBox.warning(self, "Warning", "Please enter a valid number for position size")
                return
                
            # Get current price
            if self.current_data is None or self.current_data.empty:
                QMessageBox.warning(self, "Warning", "No price data available")
                return
                
            current_price = self.current_data['close'].iloc[-1]
            
            # Calculate total value
            total_value = size * current_price
            
            # Create order
            order = {
                'type': 'SELL',
                'symbol': symbol,
                'size': size,
                'price': current_price,
                'total_value': total_value,
                'timestamp': pd.Timestamp.now()
            }
            
            # Add to orders list
            self.orders.append(order)
            
            # Update orders table
            self.update_orders_table()
            
            # Show success message
            QMessageBox.information(self, "Success", f"Sell order executed:\nSymbol: {symbol}\nSize: {size}\nPrice: ${current_price:.2f}\nTotal: ${total_value:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error executing sell order: {str(e)}")
            QMessageBox.critical(self, "Error", f"Sell order failed: {str(e)}")
            
    def save_settings(self):
        """Save the current settings."""
        try:
            # Implementation depends on what settings need to be saved
            QMessageBox.information(self, "Success", "Settings saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving settings: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save settings: {str(e)}")

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
            
    def _on_training_complete(self):
        """Handle training completion in the main thread."""
        try:
            self.logger.info("Training completed successfully")
            self.train_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.save_model_button.setEnabled(True)  # Enable save button after training
            QMessageBox.information(self, "Success", "Model training completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error handling training completion: {e}")
            QMessageBox.critical(self, "Error", f"Error handling training completion: {str(e)}")
            
    def _on_training_error(self, error_msg: str):
        """Handle training error in the main thread."""
        try:
            self.logger.error(f"Training error: {error_msg}")
            self.train_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.save_model_button.setEnabled(False)  # Disable save button on error
            QMessageBox.critical(self, "Error", f"Training failed: {error_msg}")
            
        except Exception as e:
            self.logger.error(f"Error handling training error: {e}")
            QMessageBox.critical(self, "Error", f"Error handling training error: {str(e)}")

    def _train_model_thread(self, model_type: str, params: dict):
        """Thread function for model training."""
        try:
            self.logger.info(f"Training {model_type} model in thread")
            # Create model using AI agent
            model = self.ai_agent.create_model(model_type, params)
            self.logger.info("Model created successfully")
            self.training_log_signal.emit("Model created successfully")
            
            # Prepare data for training
            X, y = self.ai_agent.prepare_training_data(self.current_data)
            self.logger.info(f"Training data prepared: X shape {X.shape}, y shape {y.shape}")
            self.training_log_signal.emit(f"Training data prepared: X shape {X.shape}, y shape {y.shape}")
            
            # Train model
            history = self.ai_agent.train_model(model, X, y)
            self.logger.info("Model training completed")
            self.training_log_signal.emit("Training completed successfully")
            self.training_progress_signal.emit(100)
            
            # Emit completion signal
            self.training_complete_signal.emit()
            
        except Exception as e:
            self.logger.error(f"Error in training thread: {e}")
            self.training_log_signal.emit(f"Error during training: {str(e)}")
            self.training_error_signal.emit(str(e))

    def create_prediction_tab(self):
        """Create the prediction tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Prediction controls
        controls_group = QGroupBox("Prediction Controls")
        controls_layout = QGridLayout()
        
        # Model selection
        controls_layout.addWidget(QLabel("Model:"), 0, 0)
        self.prediction_model_combo = QComboBox()
        self.prediction_model_combo.addItems(self.ai_agent.get_available_models())
        self.prediction_model_combo.currentTextChanged.connect(self.on_prediction_model_change)
        controls_layout.addWidget(self.prediction_model_combo, 0, 1)
        
        # Prediction range
        controls_layout.addWidget(QLabel("Prediction Range (days):"), 1, 0)
        self.prediction_range = QSpinBox()
        self.prediction_range.setRange(1, 350)
        self.prediction_range.setValue(30)
        controls_layout.addWidget(self.prediction_range, 1, 1)
        
        # Predict button
        self.predict_button = QPushButton("Generate Prediction")
        self.predict_button.clicked.connect(self.generate_prediction)
        controls_layout.addWidget(self.predict_button, 2, 0, 1, 2)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Prediction chart
        chart_group = QGroupBox("Prediction Chart")
        chart_layout = QVBoxLayout()
        
        self.prediction_chart = StockChart()
        chart_layout.addWidget(self.prediction_chart)
        
        chart_group.setLayout(chart_layout)
        layout.addWidget(chart_group)
        
        # Prediction metrics
        metrics_group = QGroupBox("Prediction Metrics")
        metrics_layout = QGridLayout()
        
        self.prediction_metrics = QTextEdit()
        self.prediction_metrics.setReadOnly(True)
        metrics_layout.addWidget(self.prediction_metrics, 0, 0)
        
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)
        
        return tab
        
    def on_prediction_model_change(self, model_name: str):
        """Handle prediction model selection change."""
        try:
            if not model_name:
                return
                
            self.ai_agent.set_active_model(model_name)
            self.logger.info(f"Selected prediction model: {model_name}")
            
        except Exception as e:
            self.logger.error(f"Error changing prediction model: {e}")
            QMessageBox.critical(self, "Error", f"Failed to change model: {str(e)}")
            
    def generate_prediction(self):
        """Generate predictions using the selected model."""
        try:
            if self.current_data is None:
                QMessageBox.warning(self, "Warning", "No data loaded. Please load data first.")
                return
                
            if self.ai_agent.get_active_model() is None:
                QMessageBox.warning(self, "Warning", "No model selected. Please select a model first.")
                return
            
            # Set current_symbol if not already set
            if not hasattr(self, 'current_symbol') or self.current_symbol is None:
                self.current_symbol = self.symbol_entry.text().strip().upper()
                if not self.current_symbol:
                    self.current_symbol = "UNKNOWN"
                
            # Get prediction range
            days = self.prediction_range.value()
            
            # Show a progress message
            self.prediction_metrics.setText("Generating predictions... Please wait.")
            QApplication.processEvents()  # Update the UI
            
            # Generate predictions
            self.logger.info(f"Generating {days} days of predictions using active model")
            predictions = self.ai_agent.predict_future(self.current_data, days=days)
            
            # Validate predictions
            if predictions is None or len(predictions) == 0:
                self.logger.error("Model returned empty predictions")
                QMessageBox.warning(self, "Warning", "The model generated empty predictions. Please try a different model or data.")
                return
                
            self.logger.info(f"Successfully generated {len(predictions)} predictions")
            
            # Create prediction dates
            last_date = pd.to_datetime(self.current_data['date'].iloc[-1])
            prediction_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(predictions))
            
            # Create prediction DataFrame
            prediction_df = pd.DataFrame({
                'date': prediction_dates,
                'close': predictions
            })
            
            self.logger.info(f"Created prediction DataFrame with shape {prediction_df.shape}")
            
            # Update prediction chart
            try:
                self.prediction_chart.plot_data(
                    self.current_data,
                    prediction_data=prediction_df,
                    title=f"Price Prediction for {self.current_symbol}",
                    show_volume=False
                )
                self.logger.info("Successfully updated prediction chart")
            except Exception as chart_error:
                self.logger.error(f"Error updating prediction chart: {chart_error}")
                QMessageBox.warning(self, "Chart Error", f"Could not display predictions: {str(chart_error)}")
            
            # Calculate and display metrics
            try:
                last_price = self.current_data['close'].iloc[-1]
                predicted_price = predictions[-1]
                price_change = predicted_price - last_price
                price_change_pct = (price_change / last_price) * 100
                
                metrics_text = f"""
                Prediction Metrics:
                -------------------
                Last Price: ${last_price:.2f}
                Predicted Price: ${predicted_price:.2f}
                Price Change: ${price_change:.2f} ({price_change_pct:.2f}%)
                Prediction Range: {days} days
                """
                
                self.prediction_metrics.setText(metrics_text)
                self.logger.info("Successfully updated prediction metrics")
            except Exception as metrics_error:
                self.logger.error(f"Error calculating prediction metrics: {metrics_error}")
                self.prediction_metrics.setText("Error calculating prediction metrics")
            
        except Exception as e:
            self.logger.error(f"Error generating prediction: {e}")
            QMessageBox.critical(self, "Error", f"Failed to generate prediction: {str(e)}") 