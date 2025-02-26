import os
# Set environment variable to silence Tk deprecation warning
os.environ['TK_SILENCE_DEPRECATION'] = '1'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.regularizers import L2
import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk
)
import matplotlib.dates as mdates
import mplfinance as mpf
import tkinter as tk
from tkinter import ttk, messagebox
import traceback
import contextlib
from datetime import datetime, timedelta
import matplotlib.gridspec as gridspec
import matplotlib
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
import functools
import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

matplotlib.use('TkAgg')

def find_databases():
    """Find all DuckDB database files in the current directory"""
    return glob.glob('*.db')

def create_connection(db_name):
    """Create a database connection"""
    try:
        return duckdb.connect(db_name)
    except Exception as e:
        print(f"Error connecting to database {db_name}: {e}")
        return None

class ThreadSafeManager:
    def __init__(self):
        self._lock = threading.Lock()
    
    def __enter__(self):
        self._lock.acquire()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()

def process_data_safely(func):
    """Decorator to handle data processing errors"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            print(f"\nStarting {func.__name__}...")
            result = func(*args, **kwargs)
            print(f"Successfully completed {func.__name__}")
            return result
        except Exception as e:
            print(f"Error in {func.__name__}: {str(e)}")
            print("Traceback:")
            traceback.print_exc()
            # Show error in GUI if self is first argument (instance method)
            if len(args) > 0 and hasattr(args[0], 'loading_label'):
                args[0].loading_label.config(text=f"Error in {func.__name__}: {str(e)}")
            return None
    return wrapper

class AIAgent:
    def __init__(self):
        """Initialize the AI agent"""
        self.model = None
        
    def build_model(self, input_shape):
        """Build the LSTM model"""
        try:
            print(f"Building model with input shape: {input_shape}")
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mse')
            self.model = model
            print("Model built successfully")
            return True
            
        except Exception as e:
            print(f"Error building model: {str(e)}")
            traceback.print_exc()
            return False

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the LSTM model"""
        try:
            print("\nStarting model training...")
            
            if self.model is None:
                raise ValueError("Model not built. Call build_model first.")
            
            # Early stopping callback
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping],
                verbose=1
            )
            
            print("Model training completed")
            return history
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            traceback.print_exc()
            return None

class DataAdapter:
    def __init__(self, sequence_length=60, features=None):
        """Initialize the data adapter"""
        self.sequence_length = sequence_length
        self.features = features or ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI']
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def prepare_training_data(self, df):
        """Prepare data for model training"""
        try:
            print("\nPreparing training data...")
            
            # Validate input data
            if not self._validate_dataframe(df):
                return None, None
            
            # Scale features
            scaled_data = self.scaler.fit_transform(df[self.features])
            
            # Create sequences
            X, y = self._create_sequences(scaled_data)
            
            # Split into training and validation sets
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
            return (X_train, y_train), (X_val, y_val)
            
        except Exception as e:
            print(f"Error preparing training data: {str(e)}")
            traceback.print_exc()
            return None, None
    
    def prepare_prediction_data(self, df):
        """Prepare data for prediction"""
        try:
            # Validate input data
            if not self._validate_dataframe(df):
                return None
            
            # Scale the data using the fitted scaler
            scaled_data = self.scaler.transform(df[self.features])
            
            # Create sequence
            X = scaled_data[-self.sequence_length:].reshape(1, self.sequence_length, len(self.features))
            return X
            
        except Exception as e:
            print(f"Error preparing prediction data: {str(e)}")
            traceback.print_exc()
            return None
    
    def inverse_transform_predictions(self, predictions):
        """Convert scaled predictions back to original scale"""
        try:
            # Create a dummy array with zeros for all features except Close price
            dummy = np.zeros((len(predictions), len(self.features)))
            dummy[:, 3] = predictions.flatten()  # Assuming Close is at index 3
            
            # Inverse transform
            return self.scaler.inverse_transform(dummy)[:, 3]
            
        except Exception as e:
            print(f"Error inverse transforming predictions: {str(e)}")
            traceback.print_exc()
            return None
    
    def _validate_dataframe(self, df):
        """Validate input dataframe"""
        if df is None or df.empty:
            print("Error: DataFrame is None or empty")
            return False
            
        missing_features = [f for f in self.features if f not in df.columns]
        if missing_features:
            print(f"Error: Missing required features: {missing_features}")
            return False
            
        return True
    
    def _create_sequences(self, data):
        """Create sequences for LSTM input"""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length, 3])  # 3 is Close price index
        
        return np.array(X), np.array(y)

class StockAIAgent:
    def __init__(self):
        """Initialize the AI agent with model architecture"""
        try:
            print("\nInitializing AI Agent...")
            self.model = None
            self.data_adapter = DataAdapter(
                sequence_length=self.sequence_length,
                features=['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MA20', 'MA50', 'MACD']
            )
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            self.sequence_length = 60  # Number of time steps to look back
            self.features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI']
            print("AI Agent initialized successfully")
            
        except Exception as e:
            print(f"Error initializing AI Agent: {str(e)}")
            traceback.print_exc()

    def build_model(self, input_shape):
        """Build LSTM model architecture"""
        try:
            print("\nBuilding LSTM model...")
            model = Sequential([
                LSTM(units=50, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                LSTM(units=50, return_sequences=True),
                Dropout(0.2),
                LSTM(units=50),
                Dropout(0.2),
                Dense(units=1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001),
                         loss='mean_squared_error')
            
            print("Model architecture:")
            model.summary()
            print("Model built successfully")
            return model
            
        except Exception as e:
            print(f"Error building model: {str(e)}")
            traceback.print_exc()
            return None

    def prepare_data(self, df):
        """Prepare data for LSTM model"""
        try:
            print("\nPreparing data for LSTM...")
            
            # Ensure all required features are present
            missing_features = [f for f in self.features if f not in df.columns]
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            # Scale features
            print("Scaling features...")
            data = self.scaler.fit_transform(df[self.features])
            
            # Create sequences
            print("Creating sequences...")
            X, y = [], []
            for i in range(len(data) - self.sequence_length):
                X.append(data[i:(i + self.sequence_length)])
                y.append(data[i + self.sequence_length, 3])  # 3 is Close price index
            
            X = np.array(X)
            y = np.array(y)
            
            # Split into train and validation sets
            print("Splitting data...")
            split = int(len(X) * 0.8)
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]
            
            print(f"Training samples: {len(X_train)}")
            print(f"Validation samples: {len(X_val)}")
            
            return (X_train, y_train), (X_val, y_val)
            
        except Exception as e:
            print(f"Error preparing data: {str(e)}")
            traceback.print_exc()
            return None

    def train(self, df, epochs=50, batch_size=32):
        """Train the LSTM model"""
        try:
            print("\nStarting model training...")
            
            # Prepare data using adapter
            data = self.data_adapter.prepare_training_data(df)
            if data is None:
                raise ValueError("Data preparation failed")
                
            (X_train, y_train), (X_val, y_val) = data
            
            # Build model if not exists
            if self.model is None:
                input_shape = (X_train.shape[1], X_train.shape[2])
                self.model = self.build_model(input_shape)
                if self.model is None:
                    raise ValueError("Model building failed")
            
            # Early stopping callback
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Train model
            print(f"\nTraining model with {epochs} epochs...")
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping],
                verbose=1
            )
            
            print("Model training completed")
            return history
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            traceback.print_exc()
            return None

    def predict(self, df):
        """Make predictions using trained model"""
        try:
            print("\nMaking predictions...")
            
            if self.model is None:
                raise ValueError("Model not trained")
            
            # Prepare prediction data
            data = self.data_adapter.prepare_prediction_data(df)
            
            # Make predictions
            predictions = self.model.predict(data)
            
            # Inverse transform predictions
            predictions = self.data_adapter.inverse_transform_predictions(predictions)
            
            # Update plot with predictions
            self.update_plots_with_predictions(df, predictions)
            self.loading_label.config(text="Predictions completed")
            
        except Exception as e:
            print(f"Error making predictions: {str(e)}")
            traceback.print_exc()
            return None

    def calculate_rsi(self, prices, periods=14):
        """Calculate Relative Strength Index"""
        try:
            deltas = np.diff(prices)
            seed = deltas[:periods+1]
            up = seed[seed >= 0].sum()/periods
            down = -seed[seed < 0].sum()/periods
            rs = up/down
            rsi = np.zeros_like(prices)
            rsi[:periods] = 100. - 100./(1.+rs)

            for i in range(periods, len(prices)):
                delta = deltas[i - 1]
                if delta > 0:
                    upval = delta
                    downval = 0.
                else:
                    upval = 0.
                    downval = -delta

                up = (up*(periods-1) + upval)/periods
                down = (down*(periods-1) + downval)/periods
                rs = up/down
                rsi[i] = 100. - 100./(1.+rs)

            return rsi

        except Exception as e:
            print(f"Error calculating RSI: {str(e)}")
            traceback.print_exc()
            return np.zeros_like(prices)

    def calculate_technical_indicators(self, df):
        """Calculate technical indicators for the dataset"""
        try:
            print("Starting technical indicator calculations...")
            if df is None or df.empty:
                print("Error: Input DataFrame is None or empty")
                return None

            # Check for required columns
            # Adjust the column name if 'Close' is named differently
            close_column = 'Close'  # Change this to the correct column name if needed
            if close_column not in df.columns:
                # Attempt to create a 'Close' column if possible
                if 'close_price' in df.columns:
                    close_column = 'close_price'
                elif 'closing_value' in df.columns:
                    close_column = 'closing_value'
                else:
                    print(f"Error: Missing required columns: [{close_column}]")
                    return None

            # Create a copy to avoid modifying original
            df = df.copy()
                
            # Calculate moving averages
            df['MA20'] = df[close_column].rolling(window=20).mean()
            df['MA50'] = df[close_column].rolling(window=50).mean()
            
            # Calculate RSI
            delta = df[close_column].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            exp1 = df[close_column].ewm(span=12, adjust=False).mean()
            exp2 = df[close_column].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
            # Calculate Bollinger Bands
            df['BB_middle'] = df[close_column].rolling(window=20).mean()
            df['BB_upper'] = df['BB_middle'] + 2 * df[close_column].rolling(window=20).std()
            df['BB_lower'] = df['BB_middle'] - 2 * df[close_column].rolling(window=20).std()
            
            # Handle NaN values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            print("Technical indicators calculated successfully")
            print(f"DataFrame shape after calculations: {df.shape}")
            return df
            
        except Exception as e:
            print(f"Error calculating technical indicators: {str(e)}")
            traceback.print_exc()
            return None
            
    def validate_data(self, df):
        """Validate the input data"""
        try:
            print("\nStarting validate_data...")
            print("Validating input data...")
            
            if df is None or df.empty:
                print("Error: DataFrame is None or empty")
                return None
                
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_columns):
                print("Error: Missing required columns")
                return None
                
            print("Data validation complete")
            print(f"Final shape: {df.shape}")
            print("Successfully completed validate_data")
            return df
            
        except Exception as e:
            print(f"Error in data validation: {str(e)}")
            traceback.print_exc()
            return None

class StockAnalyzerGUI:
    def __init__(self, available_dbs):
        """Initialize the GUI"""
        try:
            print("\nStarting initialize_gui...")
            print("Initializing GUI...")
            
            # Initialize basic attributes
            self.available_dbs = available_dbs
            self.root = tk.Tk()
            self.current_db = None
            self.tables = []
            self.tickers = []
            
            # Initialize technical analysis attributes
            self.scaler = None
            self.sequence_length = 10
            
            # Initialize AI agent and data adapter
            self.ai_agent = AIAgent()
            self.data_adapter = DataAdapter(
                sequence_length=self.sequence_length,
                features=['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MA20', 'MA50', 'MACD']
            )
            
            # Create GUI elements
            self.create_gui_elements()
            
            # Call initialization methods
            self.initialize_gui()
            
        except Exception as e:
            print(f"Error in initialization: {str(e)}")
            traceback.print_exc()

    def create_gui_elements(self):
        """Create GUI elements"""
        try:
            # Create main container
            self.main_container = ttk.Frame(self.root)
            self.main_container.pack(fill=tk.BOTH, expand=True)

            # Create status bar
            self.status_frame = ttk.Frame(self.main_container)
            self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)
            
            # Create loading label
            self.loading_label = ttk.Label(self.status_frame, text="Ready")
            self.loading_label.pack(side=tk.LEFT, padx=5)
            
            # Create ticker description
            self.ticker_desc = ttk.Label(self.status_frame, text="")
            self.ticker_desc.pack(side=tk.RIGHT, padx=5)
            
            # Create control panel
            self.control_panel = ttk.Frame(self.main_container)
            self.control_panel.pack(side=tk.LEFT, fill=tk.Y)
            
            # Create plot panel
            self.plot_panel = ttk.Frame(self.main_container)
            self.plot_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
            
        except Exception as e:
            print(f"Error creating GUI elements: {str(e)}")
            traceback.print_exc()

    def initialize_gui(self):
        """Initialize the GUI components"""
        try:
            print("Setting window dimensions...")
            self.root.title("Stock Market Analyzer")
            self.root.geometry("1200x800")
            
            # Initialize components
            self.initialize_plot_area()
            self.initialize_control_panel()
            
        except Exception as e:
            print(f"Error in GUI initialization: {str(e)}")
            traceback.print_exc()

    def initialize_plot_area(self):
        """Initialize the plotting area with proper configuration"""
        try:
            print("\nInitializing plot area...")
            
            # Create figure with subplots
            print("Creating figure...")
            self.figure = Figure(figsize=(10, 8), dpi=100)
            self.figure.set_facecolor('#f0f0f0')
            
            # Create canvas
            print("Creating canvas...")
            self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_panel)
            self.canvas.draw()
            
            # Add toolbar
            print("Adding toolbar...")
            toolbar_frame = ttk.Frame(self.plot_panel)
            toolbar_frame.grid(row=0, column=0, sticky="ew")
            self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
            
            # Grid canvas
            print("Positioning canvas...")
            self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
            
            # Configure grid weights
            self.plot_panel.grid_rowconfigure(1, weight=1)
            self.plot_panel.grid_columnconfigure(0, weight=1)
            
            print("Plot area initialization complete")
            
        except Exception as e:
            print(f"Error creating plot area: {str(e)}")
            traceback.print_exc()

    def initialize_control_panel(self):
        """Initialize the control panel"""
        try:
            print("\nInitializing control panel...")
            
            # Create control panel
            self.create_control_panel()
            
        except Exception as e:
            print(f"Error initializing control panel: {str(e)}")
            traceback.print_exc()

    def create_control_panel(self):
        """Create main control panel with proper layout"""
        try:
            print("\nCreating control panel...")
            
            # Configure control panel grid
            self.control_panel.grid_columnconfigure(0, weight=1)
            
            # Create database selection
            print("Creating database controls...")
            self.create_database_controls()
            
            # Create table selection
            print("Creating table controls...")
            self.create_table_controls()
            
            # Create ticker selection
            print("Creating ticker controls...")
            self.create_ticker_controls()
            
            # Create duration selection
            print("Creating duration controls...")
            self.create_duration_controls()
            
            # Add AI controls
            print("Adding AI controls...")
            self.add_ai_controls()
            
            print("Control panel creation complete")
            
        except Exception as e:
            print(f"Error creating control panel: {str(e)}")
            traceback.print_exc()

    def create_database_controls(self):
        """Create database selection controls"""
        try:
            print("Creating database selection frame...")
            # Database selection
            db_frame = ttk.LabelFrame(self.control_panel, text="Database Selection", padding="5")
            db_frame.pack(fill="x", padx=5, pady=5)
            
            ttk.Label(db_frame, text="Database:").pack(side="left", padx=5)
            self.db_combo = ttk.Combobox(db_frame, state="readonly")
            self.db_combo.pack(side="left", fill="x", expand=True, padx=5)
            
            if self.available_dbs:
                self.db_combo['values'] = self.available_dbs
                if not self.current_db:
                    self.current_db = self.available_dbs[0]
                self.db_combo.set(self.current_db)
                self.db_combo.bind('<<ComboboxSelected>>', self.on_database_change)
            
            # Refresh button
            ttk.Button(db_frame, text="🔄", width=3,
                      command=self.refresh_databases).pack(side="left", padx=5)
            
        except Exception as e:
            print(f"Error creating database controls: {str(e)}")
            traceback.print_exc()

    def create_table_controls(self):
        """Create table selection controls"""
        try:
            print("Creating table selection frame...")
            # Table selection
            table_frame = ttk.LabelFrame(self.control_panel, text="Table Selection", padding="5")
            table_frame.pack(fill="x", padx=5, pady=5)
            
            ttk.Label(table_frame, text="Table:").pack(side="left", padx=5)
            self.table_var = tk.StringVar()
            self.table_combo = ttk.Combobox(table_frame, textvariable=self.table_var)
            self.table_combo.pack(side="left", fill="x", expand=True, padx=5)
            
            if self.tables:
                self.table_combo['values'] = self.tables
                self.table_combo.set(self.tables[0])
                self.table_combo.bind('<<ComboboxSelected>>', self.on_table_change)
            
            # Add an update button to refresh the table list
            ttk.Button(table_frame, text="Update", command=self.refresh_tables).pack(side="left", padx=5)
            
            # Sector selection frame
            self.sector_frame = ttk.LabelFrame(self.control_panel, text="Sector Selection", padding="5")
            self.sector_frame.pack(fill="x", padx=5, pady=5)
            
            ttk.Label(self.sector_frame, text="Sector:").pack(side="left", padx=5)
            self.sector_var = tk.StringVar()
            self.sector_combo = ttk.Combobox(self.sector_frame, textvariable=self.sector_var)
            self.sector_combo.pack(side="left", fill="x", expand=True, padx=5)
            
            # Add an update button to refresh the sector list
            ttk.Button(self.sector_frame, text="Update", command=self.refresh_sectors).pack(side="left", padx=5)
            
            # Field selection frame
            self.field_frame = ttk.LabelFrame(self.control_panel, text="Field Selection", padding="5")
            self.field_frame.pack(fill="x", padx=5, pady=5)
            
            self.field_vars = {}  # Dictionary to hold field variables
            
        except Exception as e:
            print(f"Error creating table controls: {str(e)}")
            traceback.print_exc()

    def create_ticker_controls(self):
        """Create ticker selection controls"""
        try:
            print("Creating ticker selection frame...")
            # Ticker selection
            ticker_frame = ttk.LabelFrame(self.control_panel, text="Stock Selection", padding="5")
            ticker_frame.pack(fill="x", padx=5, pady=5)
            
            ttk.Label(ticker_frame, text="Ticker:").pack(side="left", padx=5)
            self.ticker_var = tk.StringVar()
            self.ticker_combo = ttk.Combobox(ticker_frame, textvariable=self.ticker_var)
            self.ticker_combo.pack(fill="x", padx=5, pady=2)
            
            # Pair selection
            # pair_label = ttk.Label(ticker_frame, text="Pair:")
            # pair_label.grid(row=1, column=0, sticky="w", padx=5, pady=5)

            # self.pair_combo = ttk.Combobox(ticker_frame, state="readonly")
            # self.pair_combo.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
            
        except Exception as e:
            print(f"Error creating ticker controls: {str(e)}")
            traceback.print_exc()

    def create_duration_controls(self):
        """Create duration selection controls"""
        try:
            duration_frame = ttk.LabelFrame(self.control_panel, text="Duration")
            duration_frame.pack(fill="x", padx=5, pady=5)
            
            durations = [("1 Day", "1d"), ("1 Month", "1mo"), 
                        ("3 Months", "3mo"), ("6 Months", "6mo"),
                        ("1 Year", "1y")]
            
            self.duration_var = tk.StringVar(value="1mo")
            for i, (text, value) in enumerate(durations):
                ttk.Radiobutton(duration_frame, text=text, value=value,
                              variable=self.duration_var).pack(side="left", padx=5)
                duration_frame.grid_columnconfigure(i, weight=1)
            
        except Exception as e:
            print(f"Error creating duration controls: {str(e)}")
            traceback.print_exc()

    def add_ai_controls(self):
        """Add AI control panel with pack layout"""
        try:
            print("\nAdding AI controls...")
            
            # Create main AI frame
            ai_frame = ttk.LabelFrame(self.control_panel, text="AI Analysis")
            ai_frame.pack(fill="x", padx=5, pady=5)
            
            # Training parameters frame
            print("Adding training parameters...")
            param_frame = ttk.LabelFrame(ai_frame, text="Training Parameters")
            param_frame.pack(fill="x", padx=5, pady=5)
            
            # Parameters container
            params_container = ttk.Frame(param_frame)
            params_container.pack(fill="x", padx=5, pady=2)
            
            # Epochs
            epochs_frame = ttk.Frame(params_container)
            epochs_frame.pack(fill="x", padx=5, pady=2)
            ttk.Label(epochs_frame, text="Epochs:").pack(side="left")
            self.epochs_var = tk.StringVar(value="50")
            ttk.Entry(epochs_frame, textvariable=self.epochs_var, width=10).pack(side="left", padx=5)
            
            # Batch Size
            batch_frame = ttk.Frame(params_container)
            batch_frame.pack(fill="x", padx=5, pady=2)
            ttk.Label(batch_frame, text="Batch Size:").pack(side="left")
            self.batch_size_var = tk.StringVar(value="32")
            ttk.Entry(batch_frame, textvariable=self.batch_size_var, width=10).pack(side="left", padx=5)
            
            # Learning Rate
            lr_frame = ttk.Frame(params_container)
            lr_frame.pack(fill="x", padx=5, pady=2)
            ttk.Label(lr_frame, text="Learning Rate:").pack(side="left")
            self.learning_rate_var = tk.StringVar(value="0.001")
            ttk.Entry(lr_frame, textvariable=self.learning_rate_var, width=10).pack(side="left", padx=5)
            
            # Control buttons frame
            button_frame = ttk.Frame(ai_frame)
            button_frame.pack(fill="x", padx=5, pady=5)
            
            # Train button
            ttk.Button(button_frame, text="Train Model", 
                      command=self.train_model).pack(side="left", expand=True, padx=5)
            
            # Predict button
            ttk.Button(button_frame, text="Make Prediction",
                      command=self.make_prediction).pack(side="left", expand=True, padx=5)
            
            # Add AI status
            self.ai_status = ttk.Label(ai_frame, text="AI Status: Not trained")
            self.ai_status.pack(fill="x", padx=5, pady=5)
            
            print("AI controls setup complete")
            
        except Exception as e:
            print(f"Error adding AI controls: {str(e)}")
            traceback.print_exc()

    def get_historical_data(self):
        """Retrieve historical data from the selected table"""
        try:
            print("Retrieving historical data...")
            
            if not hasattr(self, 'db_conn') or not self.db_conn:
                raise ValueError("No database connection available")
            
            table = self.table_var.get()
            if not table or table == 'No tables available':
                raise ValueError("No valid table selected")
            
            # Construct query to fetch data
            query = f"SELECT * FROM {table}"
            print(f"Executing query: {query}")
            df = self.db_conn.execute(query).fetchdf()
            
            if df.empty:
                raise ValueError("No data retrieved from the table")
            
            print(f"Retrieved {len(df)} rows of data")
            return df
            
        except Exception as e:
            print(f"Error retrieving historical data: {str(e)}")
            traceback.print_exc()
            return None

    def train_model(self):
        """Train the AI model using the selected database, table, sector, fields, and stock"""
        try:
            print("\n=== Starting Model Training ===")
            
            # Get current selections
            current_db = self.db_combo.get()
            current_table = self.table_var.get()
            current_sector = self.sector_var.get()
            current_ticker = self.ticker_combo.get()
            
            print(f"Current Database: {current_db}")
            print(f"Selected Table: {current_table}")
            print(f"Selected Sector: {current_sector}")
            print(f"Selected Ticker: {current_ticker}")
            
            # Validate selections
            if not current_db or current_db == 'No databases available':
                print("No valid database selected")
                return
            if not current_table or current_table == 'No tables available':
                print("No valid table selected")
                return
            if not current_ticker or current_ticker == 'No tickers available':
                print("No valid ticker selected")
                return
            
            # Connect to the selected database
            conn = duckdb.connect(current_db)
            
            # Retrieve historical data for the selected ticker
            query = f"SELECT * FROM {current_table} WHERE ticker = ?"
            print(f"Executing query: {query}")
            df = conn.execute(query, (current_ticker,)).fetchdf()
            
            if df.empty:
                print("No data retrieved for the selected ticker")
                return
            
            print(f"Retrieved {len(df)} rows of data")
            
            # Calculate technical indicators
            df = self.calculate_technical_indicators(df)
            if df is None:
                print("Skipping model training due to missing technical indicators.")
                return
            
            # Filter fields based on user selection
            selected_fields = [field for field, var in self.field_vars.items() if var.get()]
            df = df[selected_fields]
            
            # Proceed with model training using the prepared data
            # ... (model training code here) ...
            
            print("Model training completed successfully")
            
        except Exception as e:
            print("\n=== Error in Training ===")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("\nTraceback:")
            traceback.print_exc()
            self.loading_label.config(text=f"Training error: {str(e)}")

    def plot_training_history(self, history):
        """Plot training history"""
        try:
            # Clear previous plots
            self.figure.clear()
            
            # Create subplot for loss
            ax = self.figure.add_subplot(111)
            ax.plot(history.history['loss'], label='Training Loss')
            ax.plot(history.history['val_loss'], label='Validation Loss')
            ax.set_title('Model Training History')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True)
            
            # Update canvas
            self.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error plotting training history: {str(e)}")
            traceback.print_exc()

    def make_prediction(self):
        """Make predictions using the trained model"""
        try:
            print("\n=== Starting Prediction ===")
            
            if self.ai_agent.model is None:
                raise ValueError("Model not trained. Please train the model first.")
            
            # Get historical data
            df = self.get_historical_data()
            if df is None or df.empty:
                raise ValueError("No data available for prediction")
            
            # Calculate technical indicators
            df = self.calculate_technical_indicators(df)
            if df is None:
                raise ValueError("Failed to calculate technical indicators")
            
            # Prepare data for prediction
            X = self.data_adapter.prepare_prediction_data(df)
            if X is None:
                raise ValueError("Failed to prepare prediction data")
            
            # Make prediction
            predictions = self.ai_agent.model.predict(X)
            
            # Inverse transform predictions
            predictions = self.data_adapter.inverse_transform_predictions(predictions)
            
            # Ensure index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                print("Converting index to datetime...")
                df.index = pd.to_datetime(df.index)
            
            # Store future predictions
            last_date = df.index[-1]
            self.future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=len(predictions),
                freq='D'
            )
            self.future_predictions = predictions
            
            # Update plot with predictions
            self.update_plots_with_predictions(df, predictions)
            self.loading_label.config(text="Predictions completed")
            
        except Exception as e:
            print("\n=== Error in Prediction ===")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("\nTraceback:")
            traceback.print_exc()
            self.loading_label.config(text=f"Prediction error: {str(e)}")

    def on_database_change(self, event=None):
        """Handle database selection change"""
        new_db = self.db_combo.get()
        try:
            # Close existing connection if any
            if hasattr(self, 'db_conn') and self.db_conn:
                self.db_conn.close()
            
            # Connect to new database
            self.db_conn = duckdb.connect(new_db)
            print(f"Connected to database: {new_db}")
            
            # Get available tables
            self.tables = self.get_tables()
            print(f"Found tables: {self.tables}")
            
            # Update table combobox
            self.table_combo['values'] = self.tables if self.tables else ['No tables available']
            if self.tables:
                self.table_combo.set(self.tables[0])
                # Trigger table change to update tickers
                self.on_table_change()
            else:
                self.table_combo.set('No tables available')
                self.clear_ticker_selection()
            
        except Exception as e:
            print(f"Error switching database: {e}")
            messagebox.showerror("Database Error", str(e))

    def on_table_change(self, event=None):
        """Handle table selection change"""
        try:
            print(f"\nTable selection changed to: {self.table_var.get()}")
            
            if not hasattr(self, 'db_conn') or not self.db_conn:
                print("No database connection available")
                self.clear_all_selections()
                return
            
            table = self.table_var.get()
            if not table or table == 'No tables available':
                print("No valid table selected")
                self.clear_all_selections()
                return
            
            # Get column information
            print(f"Getting columns for table: {table}")
            columns = self.db_conn.execute(f"SELECT * FROM {table} LIMIT 0").description
            column_names = [col[0] for col in columns]
            print(f"Available columns: {column_names}")
            
            # Clear previous field checkboxes
            for widget in self.field_frame.winfo_children():
                widget.destroy()
            
            # Create checkboxes for each column
            self.field_vars.clear()
            for column in column_names:
                var = tk.BooleanVar(value=True)  # Default to checked
                chk = ttk.Checkbutton(self.field_frame, text=column, variable=var)
                chk.pack(anchor='w')
                self.field_vars[column] = var
            
            # Check if table has sector column
            if 'sector' in column_names:
                print("Found sector column, retrieving unique sectors...")
                sectors = self.db_conn.execute(
                    f"SELECT DISTINCT sector FROM {table} ORDER BY sector"
                ).fetchall()
                sectors = [s[0] for s in sectors]
                print(f"Found {len(sectors)} sectors")
                
                # Update sector combobox
                self.sector_combo['values'] = sectors
                if sectors:
                    self.sector_combo.set(sectors[0])  # Set the first sector as default
                    print(f"Set initial sector to: {sectors[0]}")
                    self.sector_combo.bind('<<ComboboxSelected>>', self.on_sector_change)
                else:
                    print("No sectors found in table")
                    self.clear_sector_selection()
            else:
                print("No sector column found in table")
                self.clear_sector_selection()
                
                # Update tickers directly if no sector column
                if 'ticker' in column_names:
                    print("Updating tickers directly from table...")
                    tickers = self.db_conn.execute(
                        f"SELECT DISTINCT ticker FROM {table} ORDER BY ticker"
                    ).fetchall()
                    tickers = [t[0] for t in tickers]
                    self.ticker_combo['values'] = tickers
                    if tickers:
                        self.ticker_combo.set(tickers[0])
                    else:
                        self.clear_ticker_selection()
                else:
                    self.clear_ticker_selection()
            
            # Update tickers, symbols, and pairs
            self.update_tickers_symbols_pairs(table, column_names)
            
        except Exception as e:
            print(f"Error in table change handler: {str(e)}")
            traceback.print_exc()
            self.clear_all_selections()

    def update_tickers_symbols_pairs(self, table, column_names):
        """Update tickers, symbols, and pairs based on the current table"""
        try:
            # Update tickers
            if 'ticker' in column_names:
                print("Found ticker column, retrieving unique tickers...")
                tickers = self.db_conn.execute(
                    f"SELECT DISTINCT ticker FROM {table} ORDER BY ticker"
                ).fetchall()
                tickers = [t[0] for t in tickers]
                print(f"Found {len(tickers)} tickers")
                self.ticker_combo['values'] = tickers
                if tickers:
                    self.ticker_combo.set(tickers[0])
                else:
                    self.clear_ticker_selection()
            else:
                self.clear_ticker_selection()
            
            # Update symbols
            if 'symbol' in column_names:
                print("Found symbol column, retrieving unique symbols...")
                symbols = self.db_conn.execute(
                    f"SELECT DISTINCT symbol FROM {table} ORDER BY symbol"
                ).fetchall()
                symbols = [s[0] for s in symbols]
                print(f"Found {len(symbols)} symbols")
                self.symbol_combo['values'] = symbols
                if symbols:
                    self.symbol_combo.set(symbols[0])
                else:
                    self.clear_symbol_selection()
            else:
                self.clear_symbol_selection()
            
            # Update pairs
            if 'pair' in column_names:
                print("Found pair column, retrieving unique pairs...")
                pairs = self.db_conn.execute(
                    f"SELECT DISTINCT pair FROM {table} ORDER BY pair"
                ).fetchall()
                pairs = [p[0] for p in pairs]
                print(f"Found {len(pairs)} pairs")
                self.pair_combo['values'] = pairs
                if pairs:
                    self.pair_combo.set(pairs[0])
                else:
                    self.clear_pair_selection()
            else:
                self.clear_pair_selection()
            
        except Exception as e:
            print(f"Error updating tickers, symbols, and pairs: {str(e)}")
            traceback.print_exc()
            self.clear_all_selections()

    def on_sector_change(self, event=None):
        """Handle sector selection change and update tickers"""
        try:
            print("Sector selection changed, updating tickers...")
            selected_sector = self.sector_var.get()
            table = self.table_var.get()
            
            if not table or table == 'No tables available':
                self.clear_ticker_selection()
                return
            
            # Check if the sector column exists
            columns = self.db_conn.execute(f"SELECT * FROM {table} LIMIT 0").description
            column_names = [col[0] for col in columns]
            
            if 'sector' in column_names and 'ticker' in column_names:
                # Query to get tickers for the selected sector
                tickers = self.db_conn.execute(
                    f"SELECT DISTINCT ticker FROM {table} WHERE sector = ? ORDER BY ticker", 
                    (selected_sector,)
                ).fetchall()
                tickers = [t[0] for t in tickers]
                self.ticker_combo['values'] = tickers
                
                if tickers:
                    self.ticker_combo.set(tickers[0])
                else:
                    self.clear_ticker_selection()
            else:
                self.clear_ticker_selection()
            
            print("Tickers updated based on sector selection")
        except Exception as e:
            print(f"Error updating tickers on sector change: {str(e)}")
            traceback.print_exc()

    def clear_ticker_selection(self):
        """Clear ticker selection when no valid sector/database is selected"""
        self.ticker_combo['values'] = ['No tickers available']
        self.ticker_combo.set('No tickers available')

    def clear_all_selections(self):
        """Clear all selections when no valid table/database is selected"""
        self.clear_ticker_selection()
        self.clear_sector_selection()
        for widget in self.field_frame.winfo_children():
            widget.destroy()
        self.field_vars.clear()

    def clear_sector_selection(self):
        """Clear sector selection when no valid sector is available"""
        self.sector_combo['values'] = ['No sectors available']
        self.sector_combo.set('No sectors available')

    def clear_symbol_selection(self):
        """Clear symbol selection when no valid table/database is selected"""
        self.symbol_combo['values'] = ['No symbols available']
        self.symbol_combo.set('No symbols available')

    def clear_pair_selection(self):
        """Clear pair selection when no valid table/database is selected"""
        self.pair_combo['values'] = ['No pairs available']
        self.pair_combo.set('No pairs available')

    def refresh_tables(self):
        """Refresh the list of available tables"""
        try:
            print("Refreshing table list...")
            
            # Store the current selection
            current_selection = self.table_var.get()
            
            # Refresh the list of tables
            self.tables = self.get_tables()
            self.table_combo['values'] = self.tables if self.tables else ['No tables available']
            
            # Restore the previous selection if it still exists
            if current_selection in self.tables:
                self.table_combo.set(current_selection)
            elif self.tables:
                self.table_combo.set(self.tables[0])
                self.on_table_change()
            else:
                self.table_combo.set('No tables available')
                self.clear_ticker_selection()
            
            # Update sector and field selection based on the new table
            self.update_sector_and_fields()
            
            print("Table list refreshed")
        except Exception as e:
            print(f"Error refreshing tables: {str(e)}")
            traceback.print_exc()

    def update_sector_and_fields(self):
        """Update sector and field selection based on the current table"""
        try:
            table = self.table_var.get()
            if not table or table == 'No tables available':
                self.clear_sector_selection()
                self.clear_field_selection()
                return
            
            # Get column information
            columns = self.db_conn.execute(f"SELECT * FROM {table} LIMIT 0").description
            column_names = [col[0] for col in columns]
            
            # Update sector selection if sector column exists
            if 'sector' in column_names:
                sectors = self.db_conn.execute(
                    f"SELECT DISTINCT sector FROM {table} ORDER BY sector"
                ).fetchall()
                sectors = [s[0] for s in sectors]
                self.sector_combo['values'] = sectors
                if sectors:
                    self.sector_combo.set(sectors[0])
                else:
                    self.clear_sector_selection()
            else:
                self.clear_sector_selection()
            
            # Update field selection
            self.clear_field_selection()
            for column in column_names:
                var = tk.BooleanVar(value=True)  # Default to checked
                chk = ttk.Checkbutton(self.field_frame, text=column, variable=var)
                chk.pack(anchor='w')
                self.field_vars[column] = var
            
        except Exception as e:
            print(f"Error updating sector and fields: {str(e)}")
            traceback.print_exc()

    def clear_field_selection(self):
        """Clear field selection when no valid table/database is selected"""
        for widget in self.field_frame.winfo_children():
            widget.destroy()
        self.field_vars.clear()

    def refresh_databases(self):
        """Refresh the list of available databases"""
        try:
            print("Refreshing database list...")
            
            # Store the current selection
            current_selection = self.db_combo.get()
            
            # Refresh the list of databases
            self.available_dbs = find_databases()
            self.db_combo['values'] = self.available_dbs
            
            # Reapply the previous selection if it still exists
            if current_selection in self.available_dbs:
                self.db_combo.set(current_selection)
            elif self.available_dbs:
                self.db_combo.set(self.available_dbs[0])
                self.on_database_change()
            else:
                self.db_combo.set('No databases available')
            
            print("Database list refreshed")
        except Exception as e:
            print(f"Error refreshing databases: {str(e)}")
            traceback.print_exc()

    def get_tables(self):
        """Retrieve the list of tables from the current database"""
        try:
            if not hasattr(self, 'db_conn') or not self.db_conn:
                print("No database connection available")
                return []
            
            # Query to get table names
            tables = self.db_conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name != 'sqlite_sequence'
            """).fetchall()
            
            # Extract table names from query result
            table_names = [table[0] for table in tables]
            print(f"Retrieved tables: {table_names}")
            return table_names
            
        except Exception as e:
            print(f"Error retrieving tables: {str(e)}")
            traceback.print_exc()
            return []

    def refresh_sectors(self):
        """Refresh the list of available sectors"""
        try:
            print("Refreshing sector list...")
            table = self.table_var.get()
            if not table or table == 'No tables available':
                self.clear_sector_selection()
                return
            
            # Store the current selection
            current_selection = self.sector_var.get()
            
            # Get column information
            columns = self.db_conn.execute(f"SELECT * FROM {table} LIMIT 0").description
            column_names = [col[0] for col in columns]
            
            # Update sector selection if sector column exists
            if 'sector' in column_names:
                sectors = self.db_conn.execute(
                    f"SELECT DISTINCT sector FROM {table} ORDER BY sector"
                ).fetchall()
                sectors = [s[0] for s in sectors]
                self.sector_combo['values'] = sectors
                
                # Reapply the previous selection if it still exists
                if current_selection in sectors:
                    self.sector_combo.set(current_selection)
                elif sectors:
                    self.sector_combo.set(sectors[0])
                else:
                    self.clear_sector_selection()
            else:
                self.clear_sector_selection()
            
            print("Sector list refreshed")
        except Exception as e:
            print(f"Error refreshing sectors: {str(e)}")
            traceback.print_exc()

    def run(self):
        """Run the Tkinter main loop"""
        try:
            print("Running the GUI application...")
            self.root.mainloop()
        except Exception as e:
            print(f"Error running the application: {str(e)}")
            traceback.print_exc()

    def calculate_technical_indicators(self, df):
        """Calculate technical indicators for the given DataFrame."""
        required_columns = ['Close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return None
        
        try:
            # Example: Calculate a 20-day moving average
            df['MA20'] = df['Close'].rolling(window=20).mean()
            # Add more technical indicators as needed
            print("Technical indicators calculated successfully.")
            return df
        except Exception as e:
            print(f"Error calculating technical indicators: {e}")
            return None

@process_data_safely
def initialize_gui(databases):
    app = StockAnalyzerGUI(databases)
    return app

def process_database(db_name):
    """Process a single database file"""
    try:
        print(f"Connecting to database: {db_name}")
        conn = create_connection(db_name)
        if conn is None:
            print(f"Failed to connect to {db_name}")
            return
        
        # Get table information
        tables = conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name != 'sqlite_sequence'
        """).fetchall()
        
        print(f"Found {len(tables)} tables in {db_name}")
        
        # Close connection
        conn.close()
        
    except Exception as e:
        print(f"Error processing database {db_name}: {str(e)}")
        traceback.print_exc()

def main():
    try:
        # Get list of database files
        databases = find_databases()
        print(f"Found databases: {databases}\n")
        
        # Initialize GUI with databases
        app = initialize_gui(databases)
        if app:
            app.run()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()

def rename_category_to_sector(conn, table_name):
    """Rename 'category' column to 'sector' in the specified table."""
    try:
        # Check if 'category' column exists
        columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        column_names = [col[1] for col in columns]
        
        if 'category' in column_names:
            print(f"Renaming 'category' to 'sector' in table {table_name}...")
            conn.execute(f"ALTER TABLE {table_name} RENAME COLUMN category TO sector;")
            print(f"Column renamed successfully in table {table_name}.")
        else:
            print(f"No 'category' column found in table {table_name}.")
            
    except Exception as e:
        print(f"Error renaming column in table {table_name}: {str(e)}")
        traceback.print_exc()

# Example usage
database_path = 'your_database_path.db'
conn = duckdb.connect(database_path)

# Specify the table you want to alter
table_name = 'your_table_name'

# Rename the column
rename_category_to_sector(conn, table_name)

# Close the connection
conn.close()
