from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QComboBox,
                             QTableWidget, QTableWidgetItem, QMessageBox,
                             QFileDialog, QTabWidget, QInputDialog, QProgressBar,
                             QTextEdit, QGroupBox, QGridLayout, QSpinBox,
                             QDialog, QFormLayout, QDialogButtonBox, QListWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import duckdb
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import h5py
import xarray as xr
import geopandas as gpd
import requests
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

from stock_market_analyzer.config.config_manager import ConfigurationManager
from stock_market_analyzer.modules.database import DatabaseConnector
from stock_market_analyzer.modules.data_loader import DataLoader
from stock_market_analyzer.modules.stock_ai_agent import StockAIAgent
from stock_market_analyzer.modules.trading.real_trading_agent import RealTradingAgent
from stock_market_analyzer.modules.trading.strategy_optimizer import StrategyOptimizer, TradingStrategy
from stock_market_analyzer.modules.trading.strategy_trainer import StrategyTrainer
from stock_market_analyzer.utils.visualization import StockVisualizer

class DataLoaderThread(QThread):
    """Thread for loading stock data asynchronously."""
    data_loaded = pyqtSignal(object, str)  # Signal to emit when data is loaded
    progress = pyqtSignal(int)  # Signal for progress updates
    
    def __init__(self, data_loader: DataLoader, symbol: str):
        super().__init__()
        self.data_loader = data_loader
        self.symbol = symbol
        
    def run(self):
        """Load data and emit signal when done."""
        try:
            data = self.data_loader.load_data(self.symbol)
            self.data_loaded.emit(data, "")
        except Exception as e:
            self.data_loaded.emit(None, str(e))

class DatabaseConnectionDialog(QDialog):
    """Dialog for database connection settings."""
    
    def __init__(self, parent=None, db_type="PostgreSQL"):
        super().__init__(parent)
        self.db_type = db_type
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the dialog UI."""
        self.setWindowTitle(f"{self.db_type} Connection")
        layout = QFormLayout()
        
        # Create input fields
        self.host_input = QLineEdit()
        self.port_input = QLineEdit()
        self.database_input = QLineEdit()
        self.username_input = QLineEdit()
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        
        # Set default ports
        if self.db_type == "PostgreSQL":
            self.port_input.setText("5432")
        elif self.db_type == "MySQL":
            self.port_input.setText("3306")
        elif self.db_type == "SQL Server":
            self.port_input.setText("1433")
            
        # Add fields to layout
        layout.addRow("Host:", self.host_input)
        layout.addRow("Port:", self.port_input)
        layout.addRow("Database:", self.database_input)
        layout.addRow("Username:", self.username_input)
        layout.addRow("Password:", self.password_input)
        
        # Add buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)
        
        self.setLayout(layout)
        
    def get_connection_string(self):
        """Get the connection string based on the database type."""
        host = self.host_input.text()
        port = self.port_input.text()
        database = self.database_input.text()
        username = self.username_input.text()
        password = self.password_input.text()
        
        if self.db_type == "PostgreSQL":
            return f"postgresql://{username}:{password}@{host}:{port}/{database}"
        elif self.db_type == "MySQL":
            return f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
        elif self.db_type == "SQL Server":
            return f"mssql+pyodbc://{username}:{password}@{host}:{port}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")

class StockGUI(QMainWindow):
    """Main GUI window for the stock market analyzer."""
    
    def __init__(self, db: DatabaseConnector, data_loader: DataLoader,
                 ai_agent: StockAIAgent, trading_agent: RealTradingAgent,
                 config_manager: ConfigurationManager):
        super().__init__()
        
        self.logger = logging.getLogger(__name__)
        self.db = db
        self.data_loader = data_loader
        self.ai_agent = ai_agent
        self.trading_agent = trading_agent
        self.config_manager = config_manager
        self.visualizer = StockVisualizer()
        
        # Load configuration
        self.config = self.config_manager.get_config()
        
        # Initialize strategy optimizer with configuration
        self.strategy_optimizer = StrategyOptimizer(
            self.config.get('trading', {
                'initial_capital': 10000.0,
                'risk_free_rate': 0.02
            })
        )
        
        # Initialize strategy trainer with configuration
        self.strategy_trainer = StrategyTrainer(
            self.config.get('ai_model', {
                'batch_size': 32,
                'epochs': 100,
                'learning_rate': 0.001
            })
        )
        
        # Add default strategies
        self._add_default_strategies()
        
        self.current_data = None
        self.current_symbol = None
        self.current_model = None
        
        self.init_ui()
        self.logger.info("GUI initialized successfully")
        
    def _add_default_strategies(self):
        """Add default trading strategies to the optimizer."""
        # Moving Average Crossover Strategy
        ma_strategy = TradingStrategy(
            name='Moving Average Crossover',
            description='Uses short and long-term moving averages to generate signals',
            parameters={
                'short_window': 20,
                'long_window': 50
            }
        )
        self.strategy_optimizer.add_strategy(ma_strategy)
        
        # RSI Strategy
        rsi_strategy = TradingStrategy(
            name='RSI Strategy',
            description='Uses Relative Strength Index to identify overbought/oversold conditions',
            parameters={
                'rsi_period': 14,
                'overbought': 70,
                'oversold': 30
            }
        )
        self.strategy_optimizer.add_strategy(rsi_strategy)
        
        # Bollinger Bands Strategy
        bb_strategy = TradingStrategy(
            name='Bollinger Bands',
            description='Uses Bollinger Bands to identify price breakouts',
            parameters={
                'window': 20,
                'num_std': 2
            }
        )
        self.strategy_optimizer.add_strategy(bb_strategy)
        
        # AI-Powered Strategy
        ai_strategy = TradingStrategy(
            name='AI-Powered',
            description='Uses AI model predictions to generate trading signals',
            parameters={
                'ai_model': self.ai_agent
            }
        )
        self.strategy_optimizer.add_strategy(ai_strategy)
        
    def init_ui(self):
        """Initialize the GUI components."""
        # Create main layout
        main_layout = QVBoxLayout()
        
        # Create tab widget
        self.tabs = QTabWidget()
        
        # Create tabs
        data_tab = QWidget()
        analysis_tab = QWidget()
        prediction_tab = QWidget()
        trading_tab = QWidget()
        
        # Add tabs to widget
        self.tabs.addTab(data_tab, "Data")
        self.tabs.addTab(analysis_tab, "Analysis")
        self.tabs.addTab(prediction_tab, "Prediction")
        self.tabs.addTab(trading_tab, "Trading")
        
        # Create layouts for each tab
        data_layout = QVBoxLayout()
        analysis_layout = QVBoxLayout()
        prediction_layout = QVBoxLayout()
        trading_layout = QVBoxLayout()
        
        # Data Tab Components
        data_source_group = QGroupBox("Data Source")
        data_source_layout = QVBoxLayout()
        
        # Data source selection
        data_source_combo_layout = QHBoxLayout()
        self.data_source_combo = QComboBox()
        self.data_source_combo.addItems([
            "CSV", "Excel", "JSON", "HDF5", "NetCDF", "GeoJSON", "Shapefile",
            "SQLite", "DuckDB", "PostgreSQL", "MySQL", "SQL Server", "REST API",
            "Polars", "Keras Model"
        ])
        data_source_combo_layout.addWidget(QLabel("Source Type:"))
        data_source_combo_layout.addWidget(self.data_source_combo)
        
        # File path input
        file_path_layout = QHBoxLayout()
        self.file_path_input = QLineEdit()
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_file)
        file_path_layout.addWidget(QLabel("File Path:"))
        file_path_layout.addWidget(self.file_path_input)
        file_path_layout.addWidget(browse_button)
        
        # Import button
        import_button = QPushButton("Import Data")
        import_button.clicked.connect(self.import_data)
        
        # Add components to data source group
        data_source_layout.addLayout(data_source_combo_layout)
        data_source_layout.addLayout(file_path_layout)
        data_source_layout.addWidget(import_button)
        data_source_group.setLayout(data_source_layout)
        
        # Add data source group to data tab
        data_layout.addWidget(data_source_group)
        data_tab.setLayout(data_layout)
        
        # Analysis Tab Components
        analysis_group = QGroupBox("Technical Analysis")
        analysis_layout_inner = QVBoxLayout()
        
        # Add indicators selection
        indicators_list = QListWidget()
        indicators = [
            "Moving Average (MA)", "Relative Strength Index (RSI)",
            "Moving Average Convergence Divergence (MACD)",
            "Bollinger Bands", "Stochastic Oscillator",
            "Average True Range (ATR)", "On-Balance Volume (OBV)"
        ]
        indicators_list.addItems(indicators)
        indicators_list.setSelectionMode(QListWidget.MultiSelection)
        
        # Add analysis button
        analyze_button = QPushButton("Analyze")
        analyze_button.clicked.connect(self.perform_analysis)
        
        analysis_layout_inner.addWidget(QLabel("Select Indicators:"))
        analysis_layout_inner.addWidget(indicators_list)
        analysis_layout_inner.addWidget(analyze_button)
        analysis_group.setLayout(analysis_layout_inner)
        analysis_layout.addWidget(analysis_group)
        analysis_tab.setLayout(analysis_layout)
        
        # Prediction Tab Components
        prediction_group = QGroupBox("Model Training")
        prediction_layout_inner = QVBoxLayout()
        
        # Training parameters
        params_layout = QGridLayout()
        self.epochs_input = QLineEdit("100")
        self.batch_size_input = QLineEdit("32")
        params_layout.addWidget(QLabel("Epochs:"), 0, 0)
        params_layout.addWidget(self.epochs_input, 0, 1)
        params_layout.addWidget(QLabel("Batch Size:"), 1, 0)
        params_layout.addWidget(self.batch_size_input, 1, 1)
        
        # Training and prediction buttons
        train_button = QPushButton("Train Model")
        train_button.clicked.connect(self.train_model)
        predict_button = QPushButton("Make Prediction")
        predict_button.clicked.connect(self.perform_price_prediction)
        
        prediction_layout_inner.addLayout(params_layout)
        prediction_layout_inner.addWidget(train_button)
        prediction_layout_inner.addWidget(predict_button)
        prediction_group.setLayout(prediction_layout_inner)
        prediction_layout.addWidget(prediction_group)
        prediction_tab.setLayout(prediction_layout)
        
        # Trading Tab Components
        trading_group = QGroupBox("Trading Strategy")
        trading_layout_inner = QVBoxLayout()
        
        # Strategy selection
        strategy_combo = QComboBox()
        strategy_combo.addItems([
            "Moving Average Crossover",
            "RSI Strategy",
            "MACD Strategy",
            "Bollinger Bands Strategy",
            "Custom Strategy"
        ])
        
        # Strategy parameters
        strategy_params = QGroupBox("Strategy Parameters")
        strategy_params_layout = QGridLayout()
        
        # Add some common parameters
        self.initial_capital = QLineEdit("10000")
        self.risk_percentage = QLineEdit("2")
        self.stop_loss = QLineEdit("2")
        self.take_profit = QLineEdit("5")
        
        strategy_params_layout.addWidget(QLabel("Initial Capital:"), 0, 0)
        strategy_params_layout.addWidget(self.initial_capital, 0, 1)
        strategy_params_layout.addWidget(QLabel("Risk %:"), 1, 0)
        strategy_params_layout.addWidget(self.risk_percentage, 1, 1)
        strategy_params_layout.addWidget(QLabel("Stop Loss %:"), 2, 0)
        strategy_params_layout.addWidget(self.stop_loss, 2, 1)
        strategy_params_layout.addWidget(QLabel("Take Profit %:"), 3, 0)
        strategy_params_layout.addWidget(self.take_profit, 3, 1)
        
        strategy_params.setLayout(strategy_params_layout)
        
        # Backtest and live trading buttons
        backtest_button = QPushButton("Run Backtest")
        backtest_button.clicked.connect(self.run_backtest)
        live_trading_button = QPushButton("Start Live Trading")
        live_trading_button.clicked.connect(self.start_live_trading)
        
        trading_layout_inner.addWidget(QLabel("Select Strategy:"))
        trading_layout_inner.addWidget(strategy_combo)
        trading_layout_inner.addWidget(strategy_params)
        trading_layout_inner.addWidget(backtest_button)
        trading_layout_inner.addWidget(live_trading_button)
        
        trading_group.setLayout(trading_layout_inner)
        trading_layout.addWidget(trading_group)
        trading_tab.setLayout(trading_layout)
        
        # Add tabs to main layout
        main_layout.addWidget(self.tabs)
        
        # Create central widget and set layout
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # Set window properties
        self.setWindowTitle('Stock Market Analyzer')
        self.setGeometry(100, 100, 800, 600)
        
        logging.info("GUI initialized successfully")
        
    def import_data(self):
        """Import data from the selected source."""
        try:
            file_path = self.file_path_input.text()
            if not file_path:
                QMessageBox.warning(self, "Warning", "Please select a file first!")
                return

            data_source = self.data_source_combo.currentText()
            
            if data_source == "CSV":
                self.current_data = pd.read_csv(file_path)
            elif data_source == "Excel":
                self.current_data = pd.read_excel(file_path)
            elif data_source == "JSON":
                self.current_data = pd.read_json(file_path)
            elif data_source == "HDF5":
                with h5py.File(file_path, 'r') as f:
                    # Assuming the first dataset in the file
                    dataset_name = list(f.keys())[0]
                    self.current_data = pd.DataFrame(f[dataset_name][()])
            elif data_source == "NetCDF":
                ds = xr.open_dataset(file_path)
                self.current_data = ds.to_dataframe()
            elif data_source == "GeoJSON":
                self.current_data = gpd.read_file(file_path)
            elif data_source == "Shapefile":
                self.current_data = gpd.read_file(file_path)
            elif data_source == "SQLite":
                conn = sqlite3.connect(file_path)
                tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
                table_name, ok = QInputDialog.getItem(self, "Select Table", 
                                                    "Choose a table:", tables['name'].tolist(), 0, False)
                if ok and table_name:
                    self.current_data = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                conn.close()
            elif data_source == "DuckDB":
                conn = duckdb.connect(file_path)
                tables = conn.execute("SHOW TABLES").fetchall()
                table_name, ok = QInputDialog.getItem(self, "Select Table", 
                                                    "Choose a table:", [t[0] for t in tables], 0, False)
                if ok and table_name:
                    self.current_data = conn.execute(f"SELECT * FROM {table_name}").df()
                conn.close()
            elif data_source == "Polars":
                self.current_data = pl.read_parquet(file_path).to_pandas()
            elif data_source == "Keras Model":
                self.current_model = tf.keras.models.load_model(file_path)
                QMessageBox.information(self, "Success", "Model loaded successfully!")
                return
            elif data_source in ["PostgreSQL", "MySQL", "SQL Server"]:
                # Show connection dialog
                dialog = DatabaseConnectionDialog(self, data_source)
                if dialog.exec_():
                    conn_str = dialog.get_connection_string()
                    engine = create_engine(conn_str)
                    tables = pd.read_sql_query("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';", engine)
                    table_name, ok = QInputDialog.getItem(self, "Select Table", 
                                                        "Choose a table:", tables['table_name'].tolist(), 0, False)
                    if ok and table_name:
                        self.current_data = pd.read_sql_table(table_name, engine)
            elif data_source == "REST API":
                url, ok = QInputDialog.getText(self, "API URL", 
                                             "Enter the API endpoint URL:")
                if ok and url:
                    response = requests.get(url)
                    response.raise_for_status()
                    self.current_data = pd.DataFrame(response.json())
                
            # Log original column names
            if self.current_data is not None and not self.current_data.empty:
                logging.info(f"Original columns: {list(self.current_data.columns)}")
                
                # Try to map common column name variations
                column_mapping = {
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume',
                    'Adj Close': 'close',
                    'Adj_Close': 'close',
                    'AdjClose': 'close',
                    'Price': 'close',
                    'Vol': 'volume',
                    'Volume_(BTC)': 'volume',
                    'Volume_(Currency)': 'volume',
                    'Volume_(USD)': 'volume',
                    'Date': 'date',
                    'Timestamp': 'date',
                    'Time': 'date',
                    'price': 'close',
                    'opening_price': 'open',
                    'closing_price': 'close',
                    'highest_price': 'high',
                    'lowest_price': 'low',
                    'trading_volume': 'volume'
                }
                
                # Convert column names to lowercase
                self.current_data.columns = self.current_data.columns.str.lower()
                logging.info(f"Lowercase columns: {list(self.current_data.columns)}")
                
                # Map known column variations
                self.current_data = self.current_data.rename(columns=column_mapping)
                logging.info(f"Mapped columns: {list(self.current_data.columns)}")
                
                # Ensure required columns exist
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                missing_columns = [col for col in required_columns if col not in self.current_data.columns]
                if missing_columns:
                    logging.warning(f"Missing required columns: {missing_columns}")
                    
                    # If we have at least a close price column, we can fill in the rest
                    if 'close' in self.current_data.columns:
                        for col in required_columns:
                            if col not in self.current_data.columns:
                                if col in ['open', 'high', 'low']:
                                    self.current_data[col] = self.current_data['close']
                                    logging.info(f"Created {col} column from close price")
                                elif col == 'volume':
                                    self.current_data[col] = 1000000  # Default volume
                                    logging.info("Created volume column with default value")
                    else:
                        # Try to identify price-related columns
                        price_columns = [col for col in self.current_data.columns if 'price' in col]
                        if price_columns:
                            logging.info(f"Found price-related columns: {price_columns}")
                            # Use the first price column as close price
                            self.current_data['close'] = self.current_data[price_columns[0]]
                            self.current_data['open'] = self.current_data[price_columns[0]]
                            self.current_data['high'] = self.current_data[price_columns[0]]
                            self.current_data['low'] = self.current_data[price_columns[0]]
                            self.current_data['volume'] = 1000000  # Default volume
                            logging.info("Created required columns from available price column")
                
                QMessageBox.information(self, "Success", f"Data imported successfully! Shape: {self.current_data.shape}")
                logging.info(f"Data imported successfully from {data_source}. Shape: {self.current_data.shape}")
                logging.info(f"Final columns: {list(self.current_data.columns)}")
            else:
                QMessageBox.warning(self, "Warning", "No data was imported!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error importing data: {str(e)}")
            logging.error(f"Error importing data: {str(e)}")
            
    def _import_keras_model(self) -> Optional[keras.Model]:
        """Import a Keras model."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Keras Model",
            "",
            "HDF5 files (*.h5);;All files (*.*)"
        )
        
        if not file_path:
            return None
            
        self.progress_bar.setValue(20)
        
        try:
            model = keras.models.load_model(file_path)
            self.progress_bar.setValue(100)
            return model
            
        except Exception as e:
            raise Exception(f"Error loading Keras model: {str(e)}")
            
    def _import_polars_dataframe(self) -> pd.DataFrame:
        """Import a Polars DataFrame."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Polars DataFrame",
            "",
            "Parquet files (*.parquet);;CSV files (*.csv);;All files (*.*)"
        )
        
        if not file_path:
            return None
            
        self.progress_bar.setValue(20)
        
        try:
            # Read with Polars
            df = pl.read_parquet(file_path) if file_path.endswith('.parquet') else pl.read_csv(file_path)
            
            # Convert to pandas for display
            data = df.to_pandas()
            
            self.progress_bar.setValue(100)
            return data
            
        except Exception as e:
            raise Exception(f"Error reading Polars DataFrame: {str(e)}")
            
    def _import_database(self, db_type: str) -> pd.DataFrame:
        """Import data from databases."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {db_type} database",
            "",
            f"{db_type} files (*.db)"
        )
        
        if not file_path:
            return None
            
        self.progress_bar.setValue(20)
        
        try:
            if db_type == 'SQLite':
                conn = sqlite3.connect(file_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
            else:  # DuckDB
                conn = duckdb.connect(file_path)
                tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                
            if not tables:
                raise ValueError('No tables found in database')
                
            self.progress_bar.setValue(40)
            
            table_name, ok = QInputDialog.getItem(
                self,
                'Select Table',
                'Choose a table to import:',
                [table[0] for table in tables],
                0,
                False
            )
            
            if not ok or not table_name:
                return None
                
            self.progress_bar.setValue(60)
            
            if db_type == 'SQLite':
                data = pd.read_sql(f"SELECT * FROM {table_name}", conn)
            else:  # DuckDB
                data = conn.execute(f"SELECT * FROM {table_name}").df()
                
            conn.close()
            self.progress_bar.setValue(100)
            return data
            
        except Exception as e:
            raise Exception(f"Error reading {db_type} database: {str(e)}")
            
    def _import_file(self, file_type: str) -> pd.DataFrame:
        """Import data from file formats."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {file_type} file",
            "",
            f"{file_type} files (*.{file_type.lower()})"
        )
        
        if not file_path:
            return None
            
        self.progress_bar.setValue(20)
        
        try:
            if file_type == 'CSV':
                data = pd.read_csv(file_path)
            elif file_type == 'JSON':
                data = pd.read_json(file_path)
            elif file_type == 'Excel':
                data = pd.read_excel(file_path)
            elif file_type == 'Parquet':
                data = pd.read_parquet(file_path)
                
            self.progress_bar.setValue(100)
            return data
            
        except Exception as e:
            raise Exception(f"Error reading {file_type} file: {str(e)}")
            
    def _get_data_source_name(self, source_type: str) -> str:
        """Get a name for the data source."""
        if source_type in ['CSV', 'JSON', 'Excel', 'Parquet']:
            return f"{source_type.lower()}_data"
        elif source_type in ['SQLite', 'DuckDB']:
            return f"{source_type.lower()}_database"
        elif source_type == 'Polars DataFrame':
            return "polars_dataframe"
        elif source_type == 'Keras Model':
            return "keras_model"
        else:
            return "unknown_source"
            
    def load_data(self):
        """Load data for the specified symbol."""
        symbol = self.symbol_input.text().strip().upper()
        if not symbol:
            QMessageBox.warning(self, 'Error', 'Please enter a symbol')
            return
            
        self.statusBar().showMessage(f'Loading data for {symbol}...')
        self.load_button.setEnabled(False)
        
        # Create and start data loader thread
        self.loader_thread = DataLoaderThread(self.data_loader, symbol)
        self.loader_thread.data_loaded.connect(self.on_data_loaded)
        self.loader_thread.start()
        
    def on_data_loaded(self, data: Optional[pd.DataFrame], error: str):
        """Handle loaded data."""
        self.load_button.setEnabled(True)
        
        if error:
            self.statusBar().showMessage(f'Error: {error}')
            QMessageBox.warning(self, 'Error', f'Failed to load data: {error}')
            return
            
        if data is None or data.empty:
            self.statusBar().showMessage('No data available')
            return
            
        self.current_data = data
        self.current_symbol = self.symbol_input.text().strip().upper()
        
        # Update table
        self.update_data_table()
        
        # Update status
        self.statusBar().showMessage(f'Data loaded for {self.current_symbol}')
        
    def update_data_table(self):
        """Update the data table with current data."""
        if self.current_data is None:
            return
            
        # Get the first 100 rows for display
        display_data = self.current_data.head(100)
        
        # Set up table
        self.data_table.setRowCount(len(display_data))
        self.data_table.setColumnCount(len(display_data.columns))
        self.data_table.setHorizontalHeaderLabels(display_data.columns)
        
        # Fill table with data
        for i, (_, row) in enumerate(display_data.iterrows()):
            for j, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                self.data_table.setItem(i, j, item)
                
        # Adjust column widths
        self.data_table.resizeColumnsToContents()
        
    def train_model(self):
        if self.current_data is None:
            QMessageBox.warning(self, "Warning", "Please load data first!")
            return

        try:
            epochs = int(self.epochs_input.text()) if hasattr(self, 'epochs_input') and self.epochs_input.text() else 10
            batch_size = int(self.batch_size_input.text()) if hasattr(self, 'batch_size_input') and self.batch_size_input.text() else 32
            
            self.ai_agent.train_model(data=self.current_data, epochs=epochs, batch_size=batch_size)
            model_path = os.path.join('models', 'stock_model.h5')
            self.ai_agent.save_model(model_path)
            QMessageBox.information(self, "Success", "Model trained and saved successfully!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error training model: {str(e)}")
            logging.error(f"Error training model: {str(e)}")
            
    def perform_price_prediction(self):
        """
        Perform price prediction using the trained model.
        """
        try:
            if self.current_data is None:
                QMessageBox.warning(self, "Warning", "Please load data first!")
                return
                
            if not hasattr(self.ai_agent, 'current_model') or self.ai_agent.current_model is None:
                # Train the model if not already trained
                epochs = int(self.epochs_input.text() or "100")
                batch_size = int(self.batch_size_input.text() or "32")
                self.train_model()
                
            # Get the last sequence of data for prediction
            prediction = self.ai_agent.predict(self.current_data)
            
            if prediction is not None:
                # Get actual prices for comparison
                actual_prices = self.current_data['close'].values[-30:]  # Last 30 days
                dates = pd.to_datetime(self.current_data['date'].values[-30:])
                
                # Create future dates for prediction
                future_dates = pd.date_range(
                    start=dates[-1],
                    periods=len(prediction) + 1,
                    freq='D'
                )[1:]  # Skip the first date as it's the last actual date
                
                # Plot actual vs predicted prices
                plt.figure(figsize=(12, 6))
                plt.plot(dates, actual_prices, label='Actual', color='blue')
                plt.plot(future_dates, prediction.flatten(), label='Predicted', color='red', linestyle='--')
                
                plt.title('Stock Price Prediction')
                plt.xlabel('Date')
                plt.ylabel('Price')
                plt.legend()
                plt.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Save the plot
                plt.savefig('predictions.png')
                plt.close()
                
                # Calculate prediction metrics
                last_actual = actual_prices[-1]
                next_predicted = prediction[0][0]
                change_pct = ((next_predicted - last_actual) / last_actual) * 100
                
                message = (
                    f"Prediction completed!\n\n"
                    f"Last Actual Price: ${last_actual:.2f}\n"
                    f"Next Predicted Price: ${next_predicted:.2f}\n"
                    f"Predicted Change: {change_pct:+.2f}%\n\n"
                    f"Plot saved as 'predictions.png'"
                )
                
                QMessageBox.information(self, "Success", message)
            else:
                QMessageBox.warning(self, "Warning", "Failed to generate prediction!")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during prediction: {str(e)}")
            logging.error(f"Error during prediction: {str(e)}")
            
    def optimize_strategies(self):
        """Optimize trading strategies using historical data."""
        if self.current_data is None:
            QMessageBox.warning(self, 'Error', 'No data loaded')
            return
            
        try:
            # Update initial capital
            self.strategy_optimizer.config['initial_capital'] = float(self.initial_capital.text())
            
            # Optimize strategies
            optimized_strategies = self.strategy_optimizer.optimize_strategies(self.current_data)
            
            # Update strategy table
            self.strategy_table.setRowCount(len(optimized_strategies))
            for i, strategy in enumerate(optimized_strategies):
                metrics = strategy.performance_metrics
                self.strategy_table.setItem(i, 0, QTableWidgetItem(strategy.name))
                self.strategy_table.setItem(i, 1, QTableWidgetItem(f"{metrics['total_return']:.2%}"))
                self.strategy_table.setItem(i, 2, QTableWidgetItem(f"{metrics['sharpe_ratio']:.2f}"))
                self.strategy_table.setItem(i, 3, QTableWidgetItem(f"{metrics['max_drawdown']:.2%}"))
                self.strategy_table.setItem(i, 4, QTableWidgetItem(f"{metrics['win_rate']:.2%}"))
                self.strategy_table.setItem(i, 5, QTableWidgetItem(f"{metrics['profit_factor']:.2f}"))
                
            # Get and display recommendations
            recommendations = self.strategy_optimizer.get_strategy_recommendations()
            recommendations_text = "Strategy Recommendations:\n\n"
            
            for rec in recommendations:
                recommendations_text += f"Strategy: {rec['strategy']}\n"
                recommendations_text += f"Description: {rec['description']}\n"
                recommendations_text += f"Recommendation: {rec['recommendation']}\n"
                recommendations_text += f"Performance:\n"
                for metric, value in rec['performance'].items():
                    recommendations_text += f"  {metric}: {value:.2%}\n"
                recommendations_text += "\n"
                
            self.recommendations_text.setText(recommendations_text)
            
            # Create visualization of best strategy
            best_strategy = self.strategy_optimizer.best_strategy
            if best_strategy:
                results = self.strategy_optimizer._backtest_strategy(best_strategy, self.current_data)
                fig = self.visualizer.plot_trading_results(
                    results['portfolio_value'],
                    results['returns'],
                    title=f'Best Strategy Performance - {best_strategy.name}'
                )
                if fig:
                    fig.savefig('best_strategy.png')
                    
        except Exception as e:
            self.logger.error(f"Error optimizing strategies: {e}")
            QMessageBox.warning(self, 'Error', f'Strategy optimization failed: {str(e)}')
            
    def train_from_strategy(self):
        """Train a model using data from a selected strategy."""
        if self.current_data is None:
            QMessageBox.warning(self, 'Error', 'No data loaded')
            return
            
        try:
            # Get selected strategy
            strategy_name = self.strategy_selector.currentText()
            strategy = next(s for s in self.strategy_optimizer.strategies if s.name == strategy_name)
            
            # Update trainer configuration
            self.strategy_trainer.config.update({
                'batch_size': self.strategy_batch_size.value(),
                'epochs': self.strategy_epochs.value()
            })
            
            # Run strategy backtest
            results = self.strategy_optimizer._backtest_strategy(strategy, self.current_data)
            
            # Prepare training data
            training_data = self.strategy_trainer.prepare_training_data(results, strategy_name)
            
            # Train model
            model = self.strategy_trainer.train_model(training_data)
            
            # Save model
            self.strategy_trainer.save_models('models')
            
            QMessageBox.information(self, 'Success', f'Model trained successfully for strategy: {strategy_name}')
            
            # Update current model
            self.current_model = model
            
        except Exception as e:
            self.logger.error(f"Error training model from strategy: {e}")
            QMessageBox.warning(self, 'Error', f'Failed to train model: {str(e)}')
            
    def perform_real_trading(self):
        """Perform real trading based on predictions."""
        if self.current_data is None:
            QMessageBox.warning(self, 'Error', 'No data loaded')
            return
            
        try:
            # Get trading parameters
            initial_capital, ok = QInputDialog.getDouble(
                self,
                'Trading Parameters',
                'Enter initial capital:',
                value=10000.0,
                min=1000.0,
                max=1000000.0,
                decimals=2
            )
            
            if not ok:
                return
                
            # Configure trading agent with best strategy
            best_strategy = self.strategy_optimizer.best_strategy
            if best_strategy:
                # Prepare features for prediction
                features = self.strategy_trainer._calculate_features(self.current_data)
                
                # Get predictions from trained model
                predictions = self.strategy_trainer.predict(best_strategy.name, features)
                
                # Update trading agent configuration
                self.trading_agent.config.update({
                    'initial_capital': initial_capital,
                    'symbol': self.current_symbol,
                    'strategy': best_strategy,
                    'predictions': predictions
                })
                
                # Start trading
                self.trading_agent.start_trading(self.current_data)
                
                # Show trading results
                results = self.trading_agent.get_trading_results()
                
                # Create visualization
                fig = self.visualizer.plot_trading_results(
                    results['portfolio_value'],
                    results['trades'],
                    title=f'Trading Results - {self.current_symbol} ({best_strategy.name})'
                )
                
                if fig:
                    # Save and show plot
                    fig.savefig('trading_results.png')
                    # You would typically show this in a separate window or widget
            else:
                QMessageBox.warning(self, 'Error', 'No optimized strategy available')
                
        except Exception as e:
            self.logger.error(f"Error performing real trading: {e}")
            QMessageBox.warning(self, 'Error', f'Trading failed: {str(e)}')
            
    def analyze_data(self):
        """Perform analysis based on selected type."""
        if self.current_data is None:
            QMessageBox.warning(self, 'Error', 'No data loaded')
            return
            
        analysis_type = self.analysis_type.currentText()
        
        try:
            if analysis_type == 'Technical Analysis':
                self.perform_technical_analysis()
            elif analysis_type == 'Price Prediction':
                self.perform_price_prediction()
            else:  # Real Trading
                self.perform_real_trading()
                
        except Exception as e:
            self.logger.error(f"Error performing analysis: {e}")
            QMessageBox.warning(self, 'Error', f'Analysis failed: {str(e)}')
            
    def perform_technical_analysis(self):
        """Perform technical analysis on current data."""
        try:
            if self.current_data is None:
                QMessageBox.warning(self, "Warning", "Please load data first!")
                return
                
            # Get selected indicators
            indicators = [item.text() for item in self.findChild(QListWidget).selectedItems()]
            
            if not indicators:
                QMessageBox.warning(self, "Warning", "Please select at least one indicator!")
                return
                
            # Calculate indicators
            results = {}
            for indicator in indicators:
                if "Moving Average" in indicator:
                    results['SMA_20'] = self.current_data['close'].rolling(window=20).mean()
                    results['SMA_50'] = self.current_data['close'].rolling(window=50).mean()
                    results['SMA_200'] = self.current_data['close'].rolling(window=200).mean()
                    
                elif "RSI" in indicator:
                    delta = self.current_data['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    results['RSI'] = 100 - (100 / (1 + rs))
                    
                elif "MACD" in indicator:
                    exp1 = self.current_data['close'].ewm(span=12, adjust=False).mean()
                    exp2 = self.current_data['close'].ewm(span=26, adjust=False).mean()
                    results['MACD'] = exp1 - exp2
                    results['Signal Line'] = results['MACD'].ewm(span=9, adjust=False).mean()
                    
                elif "Bollinger" in indicator:
                    sma = self.current_data['close'].rolling(window=20).mean()
                    std = self.current_data['close'].rolling(window=20).std()
                    results['BB_Upper'] = sma + (std * 2)
                    results['BB_Middle'] = sma
                    results['BB_Lower'] = sma - (std * 2)
                    
                elif "Stochastic" in indicator:
                    high_14 = self.current_data['high'].rolling(window=14).max()
                    low_14 = self.current_data['low'].rolling(window=14).min()
                    results['%K'] = ((self.current_data['close'] - low_14) / (high_14 - low_14)) * 100
                    results['%D'] = results['%K'].rolling(window=3).mean()
                    
                elif "ATR" in indicator:
                    high_low = self.current_data['high'] - self.current_data['low']
                    high_close = abs(self.current_data['high'] - self.current_data['close'].shift())
                    low_close = abs(self.current_data['low'] - self.current_data['close'].shift())
                    ranges = pd.concat([high_low, high_close, low_close], axis=1)
                    true_range = ranges.max(axis=1)
                    results['ATR'] = true_range.rolling(window=14).mean()
                    
                elif "OBV" in indicator:
                    obv = (np.sign(self.current_data['close'].diff()) * self.current_data['volume']).fillna(0).cumsum()
                    results['OBV'] = obv
            
            # Plot results
            plt.figure(figsize=(12, 6))
            plt.plot(self.current_data['close'], label='Close Price')
            
            for name, data in results.items():
                plt.plot(data, label=name)
                
            plt.title('Technical Analysis')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot
            plt.savefig('technical_analysis.png')
            plt.close()
            
            QMessageBox.information(self, "Success", 
                                  f"Technical analysis completed!\n"
                                  f"Plot saved as 'technical_analysis.png'")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during analysis: {str(e)}")
            logging.error(f"Error during analysis: {str(e)}")
            
    def run_backtest(self):
        """Run backtesting on the selected strategy."""
        try:
            if self.current_data is None:
                QMessageBox.warning(self, "Warning", "Please load data first!")
                return
                
            # Get strategy parameters
            initial_capital = float(self.initial_capital.text())
            risk_percentage = float(self.risk_percentage.text())
            stop_loss = float(self.stop_loss.text())
            take_profit = float(self.take_profit.text())
            
            # Initialize variables
            position = 0  # 0: no position, 1: long, -1: short
            capital = initial_capital
            trades = []
            
            # Calculate indicators
            sma_20 = self.current_data['close'].rolling(window=20).mean()
            sma_50 = self.current_data['close'].rolling(window=50).mean()
            
            # Simple moving average crossover strategy
            for i in range(50, len(self.current_data)):
                if position == 0:  # No position
                    if sma_20[i] > sma_50[i] and sma_20[i-1] <= sma_50[i-1]:
                        # Buy signal
                        entry_price = self.current_data['close'].iloc[i]
                        position = 1
                        shares = (capital * (risk_percentage / 100)) / entry_price
                        trades.append({
                            'type': 'buy',
                            'price': entry_price,
                            'shares': shares,
                            'date': self.current_data.index[i]
                        })
                elif position == 1:  # Long position
                    current_price = self.current_data['close'].iloc[i]
                    entry_price = trades[-1]['price']
                    
                    # Check stop loss and take profit
                    if (current_price <= entry_price * (1 - stop_loss/100) or 
                        current_price >= entry_price * (1 + take_profit/100) or
                        (sma_20[i] < sma_50[i] and sma_20[i-1] >= sma_50[i-1])):
                        # Sell
                        position = 0
                        shares = trades[-1]['shares']
                        profit = shares * (current_price - entry_price)
                        capital += profit
                        trades.append({
                            'type': 'sell',
                            'price': current_price,
                            'shares': shares,
                            'date': self.current_data.index[i],
                            'profit': profit
                        })
            
            # Calculate performance metrics
            total_trades = len([t for t in trades if t['type'] == 'sell'])
            profitable_trades = len([t for t in trades if t['type'] == 'sell' and t['profit'] > 0])
            total_profit = sum([t['profit'] for t in trades if t['type'] == 'sell'])
            win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Plot results
            plt.figure(figsize=(12, 6))
            plt.plot(self.current_data['close'], label='Close Price')
            plt.plot(sma_20, label='SMA 20')
            plt.plot(sma_50, label='SMA 50')
            
            # Plot buy and sell points
            buy_points = [t for t in trades if t['type'] == 'buy']
            sell_points = [t for t in trades if t['type'] == 'sell']
            
            plt.scatter([t['date'] for t in buy_points],
                       [t['price'] for t in buy_points],
                       color='green', marker='^', label='Buy')
            plt.scatter([t['date'] for t in sell_points],
                       [t['price'] for t in sell_points],
                       color='red', marker='v', label='Sell')
            
            plt.title('Backtest Results')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot
            plt.savefig('backtest_results.png')
            plt.close()
            
            message = (
                f"Backtest Results:\n\n"
                f"Initial Capital: ${initial_capital:,.2f}\n"
                f"Final Capital: ${capital:,.2f}\n"
                f"Total Profit: ${total_profit:,.2f}\n"
                f"Total Trades: {total_trades}\n"
                f"Profitable Trades: {profitable_trades}\n"
                f"Win Rate: {win_rate:.2f}%\n\n"
                f"Plot saved as 'backtest_results.png'"
            )
            
            QMessageBox.information(self, "Success", message)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during backtesting: {str(e)}")
            logging.error(f"Error during backtesting: {str(e)}")
            
    def start_live_trading(self):
        """Start live trading with the selected strategy."""
        try:
            # Get strategy parameters
            initial_capital = float(self.initial_capital.text())
            risk_percentage = float(self.risk_percentage.text())
            stop_loss = float(self.stop_loss.text())
            take_profit = float(self.take_profit.text())
            
            message = (
                "Live Trading Parameters:\n\n"
                f"Initial Capital: ${initial_capital:,.2f}\n"
                f"Risk per Trade: {risk_percentage}%\n"
                f"Stop Loss: {stop_loss}%\n"
                f"Take Profit: {take_profit}%\n\n"
                "Live trading feature is not implemented yet."
            )
            
            QMessageBox.information(self, "Info", message)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error starting live trading: {str(e)}")
            logging.error(f"Error starting live trading: {str(e)}")
            
    def browse_file(self):
        """Open file dialog to select data file."""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        
        data_source = self.data_source_combo.currentText()
        if data_source == "CSV":
            file_dialog.setNameFilter("CSV files (*.csv)")
        elif data_source == "Excel":
            file_dialog.setNameFilter("Excel files (*.xlsx *.xls)")
        elif data_source == "JSON":
            file_dialog.setNameFilter("JSON files (*.json)")
        elif data_source == "HDF5":
            file_dialog.setNameFilter("HDF5 files (*.h5 *.hdf5)")
        elif data_source == "NetCDF":
            file_dialog.setNameFilter("NetCDF files (*.nc)")
        elif data_source == "GeoJSON":
            file_dialog.setNameFilter("GeoJSON files (*.geojson)")
        elif data_source == "Shapefile":
            file_dialog.setNameFilter("Shapefiles (*.shp)")
        elif data_source == "SQLite":
            file_dialog.setNameFilter("SQLite files (*.db *.sqlite)")
        elif data_source == "DuckDB":
            file_dialog.setNameFilter("DuckDB files (*.duckdb)")
        elif data_source == "Keras Model":
            file_dialog.setNameFilter("Keras models (*.h5 *.keras)")
        else:
            file_dialog.setNameFilter("All files (*.*)")
        
        if file_dialog.exec_():
            filenames = file_dialog.selectedFiles()
            if filenames:
                self.file_path_input.setText(filenames[0])
                self.import_data() 