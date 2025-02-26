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
                
            # Create a copy to avoid modifying original
            df = df.copy()
                
            # Calculate moving averages
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA50'] = df['Close'].rolling(window=50).mean()
            
            # Calculate RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
            # Calculate Bollinger Bands
            df['BB_middle'] = df['Close'].rolling(window=20).mean()
            df['BB_upper'] = df['BB_middle'] + 2 * df['Close'].rolling(window=20).std()
            df['BB_lower'] = df['BB_middle'] - 2 * df['Close'].rolling(window=20).std()
            
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
            self.setup_initial_database()
            
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

    def train_model(self):
        """Train the AI model"""
        try:
            print("\n=== Starting Model Training ===")
            print(f"Current Database: {self.current_db}")
            print(f"Selected Table: {self.table_var.get()}")
            print(f"Selected Ticker: {self.ticker_var.get()}")
            
            # Get historical data
            df = self.get_historical_data()
            if df is None or df.empty:
                return
            
            # Calculate technical indicators
            df = self.calculate_technical_indicators(df)
            if df is None:
                return
                
            # Prepare data for training using data adapter
            data = self.data_adapter.prepare_training_data(df)
            if data is None:
                raise ValueError("Data preparation failed")
                
            (X_train, y_train), (X_val, y_val) = data
            
            # Build model if not exists
            if self.ai_agent.model is None:
                input_shape = (X_train.shape[1], X_train.shape[2])
                self.ai_agent.build_model(input_shape)
            
            # Train model
            history = self.ai_agent.train(X_train, y_train, X_val, y_val)
            if history:
                self.plot_training_history(history)
                self.loading_label.config(text="Model training completed")
            
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
                self.clear_ticker_selection()
                return
            
            table = self.table_var.get()
            if not table or table == 'No tables available':
                print("No valid table selected")
                self.clear_ticker_selection()
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
            
            # Check if table has ticker column
            if 'ticker' in column_names:
                print("Found ticker column, retrieving unique tickers...")
                tickers = self.db_conn.execute(
                    f"SELECT DISTINCT ticker FROM {table} ORDER BY ticker"
                ).fetchall()
                tickers = [t[0] for t in tickers]
                print(f"Found {len(tickers)} tickers")
                
                # Update ticker combobox
                self.ticker_combo['values'] = tickers
                if tickers:
                    self.ticker_combo.set(tickers[0])
                    print(f"Set initial ticker to: {tickers[0]}")
                    # Start analysis automatically
                    self.start_analysis()
                else:
                    print("No tickers found in table")
                    self.clear_ticker_selection()
            else:
                print("No ticker column found in table")
                self.clear_ticker_selection()
            
        except Exception as e:
            print(f"Error in table change handler: {str(e)}")
            traceback.print_exc()
            self.clear_ticker_selection()

    def clear_ticker_selection(self):
        """Clear ticker selection when no valid table/database is selected"""
        self.ticker_combo['values'] = ['No tickers available']
        self.ticker_combo.set('No tickers available')
        self.ticker_desc.config(text="")

    def update_ticker_description(self):
        """Update the description for the currently selected ticker"""
        ticker = self.ticker_var.get()
        description = self.ticker_desc.get(ticker, "No description available")
        self.ticker_desc.config(text=description)

    def get_tables(self):
        """Get list of non-empty tables from current database"""
        try:
            print(f"\nQuerying tables from database: {self.current_db}")
            tables = self.db_conn.execute(
                """SELECT name FROM sqlite_master 
                   WHERE type='table' AND name != 'sqlite_sequence'"""
            ).fetchall()
            tables = [t[0] for t in tables]
            print(f"Found raw tables: {tables}")
            
            # Filter out empty tables and get table statistics
            non_empty_tables = []
            for table in tables:
                try:
                    print(f"\nAnalyzing table: {table}")
                    # Get column names first
                    columns = self.db_conn.execute(f"SELECT * FROM {table} LIMIT 0").description
                    column_names = [col[0] for col in columns]
                    print(f"Columns in {table}: {column_names}")
                    
                    # Build dynamic query based on available columns
                    count_query = f"SELECT COUNT(*) FROM {table}"
                    count = self.db_conn.execute(count_query).fetchone()[0]
                    
                    if count > 0:
                        non_empty_tables.append(table)
                        print(f"Table {table} statistics:")
                        print(f"  Total records: {count}")
                        
                        # Get date range if date column exists
                        date_column = None
                        for col in ['date', 'expiry', 'created_at']:
                            if col in column_names:
                                date_column = col
                                break
                        
                        if date_column:
                            date_query = f"""
                                SELECT 
                                    MIN({date_column}) as earliest_date,
                                    MAX({date_column}) as latest_date 
                                FROM {table}
                            """
                            date_range = self.db_conn.execute(date_query).fetchone()
                            if date_range and date_range[0] and date_range[1]:
                                print(f"  Date range: {date_range[0]} to {date_range[1]}")
                        
                        # Get sample of tickers if available
                        ticker_column = None
                        for col in ['ticker', 'symbol', 'pair']:
                            if col in column_names:
                                ticker_column = col
                                break
                        
                        if ticker_column:
                            tickers_query = f"""
                                SELECT DISTINCT {ticker_column}
                                FROM {table}
                                ORDER BY {ticker_column}
                                LIMIT 5
                            """
                            tickers = [row[0] for row in self.db_conn.execute(tickers_query).fetchall()]
                            print(f"  Sample tickers: {tickers}")
                    else:
                        print(f"Table {table} is empty")
                
                except Exception as e:
                    print(f"Error processing table {table}: {str(e)}")
                    continue
            
            print(f"\nFinal non-empty tables: {non_empty_tables}")
            return non_empty_tables
            
        except Exception as e:
            print(f"Error getting tables: {str(e)}")
            traceback.print_exc()
            return []

    def get_tickers(self, table_name):
        """Get list of unique tickers/symbols from the specified table"""
        try:
            # First get all column names for the table
            columns_query = """
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = ?
            """
            columns = [row[0].lower() for row in self.db_conn.execute(columns_query, [table_name]).fetchall()]
            print(f"Available columns in {table_name}: {columns}")
            
            # Try different common column names for tickers
            ticker_columns = ['ticker', 'symbol', 'pair']
            
            for col in ticker_columns:
                if col in columns:
                    query = f"""
                        SELECT DISTINCT {col}
                        FROM {table_name}
                        WHERE {col} IS NOT NULL
                        ORDER BY {col}
                    """
                    result = self.db_conn.execute(query).fetchall()
                    if result:
                        tickers = [row[0] for row in result]
                        print(f"Found tickers using column '{col}': {tickers[:5]}...")
                        return tickers
            
            # Special handling for market_data table
            if 'type' in columns and table_name == 'market_data':
                query = """
                    SELECT DISTINCT ticker 
                    FROM market_data 
                    WHERE type = 'forex'
                    ORDER BY ticker
                """
                result = self.db_conn.execute(query).fetchall()
                if result:
                    tickers = [row[0] for row in result]
                    print(f"Found forex pairs: {tickers[:5]}...")
                    return tickers
            
            print(f"No suitable ticker column found in {table_name}")
            return []
            
        except Exception as e:
            print(f"Error getting tickers: {str(e)}")
            traceback.print_exc()
            return []

    def get_historical_data(self):
        """Get historical price data for a ticker"""
        try:
            # Get column names first
            columns = self.db_conn.execute(f"SELECT * FROM {self.table_var.get()} LIMIT 0").description
            column_names = [col[0] for col in columns]
            
            # Determine date column
            date_column = None
            for col in ['date', 'expiry', 'created_at']:
                if col in column_names:
                    date_column = col
                    break
            
            if not date_column:
                raise ValueError(f"No date column found in table {self.table_var.get()}")
            
            # Map timeframe to interval
            interval_map = {
                '1d': '1 day',
                '1mo': '1 month',
                '3mo': '3 months',
                '6mo': '6 months',
                '1y': '1 year'
            }
            interval = interval_map.get(self.duration_var.get(), '1 month')
            
            # Build column list based on available columns
            select_columns = []
            column_mapping = {
                'date': date_column,
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
                'adj_close': 'Adj_Close'
            }
            
            for db_col, df_col in column_mapping.items():
                if db_col in column_names:
                    select_columns.append(f"{db_col} as {df_col}")
                elif db_col == 'adj_close' and 'close' in column_names:
                    select_columns.append(f"close as {df_col}")
            
            # Determine ticker column
            ticker_column = None
            for col in ['ticker', 'symbol', 'pair']:
                if col in column_names:
                    ticker_column = col
                    break
            
            if not ticker_column:
                raise ValueError(f"No ticker column found in table {self.table_var.get()}")
            
            # Build and execute query
            query = f"""
                SELECT {', '.join(select_columns)}
                FROM {self.table_var.get()}
                WHERE {ticker_column} = ?
                AND {date_column} >= CURRENT_DATE - INTERVAL '{interval}'
                ORDER BY {date_column}
            """
            
            print(f"Executing query: {query}")
            print(f"Parameters: {[self.ticker_var.get()]}")
            
            df = self.db_conn.execute(query, [self.ticker_var.get()]).df()
            if df is not None and not df.empty:
                print(f"Retrieved {len(df)} rows of data")
                print(f"Sample data:\n{df.head()}")
                
                # Ensure date column is datetime and set as index
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                
            return df
            
        except Exception as e:
            print(f"Error retrieving historical data: {str(e)}")
            traceback.print_exc()
            return None

    def start_analysis(self):
        """Start analysis for selected ticker"""
        try:
            ticker = self.ticker_var.get()
            duration = self.duration_var.get()
            
            print(f"\nStarting analysis for {ticker} over {duration} period")
            self.loading_label.config(text=f"Loading data for {ticker}...")
            
            # Get historical data
            print(f"Retrieving historical data from {self.table_var.get()}")
            df = self.get_historical_data()
            
            if df is not None and not df.empty:
                print(f"Retrieved {len(df)} records")
                print("Calculating technical indicators...")
                
                # Calculate indicators
                try:
                    df = self.calculate_technical_indicators(df)
                    if df is None:
                        raise ValueError("Invalid data after calculating indicators")
                    
                    # Update plots
                    print("Updating visualization...")
                    self.update_plots(df, ticker)
                    self.loading_label.config(text=f"Analysis complete for {ticker}")
                    print("Analysis complete")
                    
                except Exception as e:
                    print(f"Error calculating indicators: {str(e)}")
                    traceback.print_exc()
                    self.loading_label.config(text=f"Error in technical analysis: {str(e)}")
            else:
                print(f"No data available for {ticker}")
                self.loading_label.config(text=f"No data available for {ticker}")
                
        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            traceback.print_exc()
            self.loading_label.config(text=f"Error analyzing {ticker}: {str(e)}")

    @process_data_safely
    def update_plots(self, df, ticker):
        """Update all plots with current data"""
        try:
            if ticker == 'No tickers available' or self.table_var.get() == 'No tables available':
                print("Invalid ticker or table selection")
                return
            
            print(f"\nUpdating plots for {ticker} from {self.table_var.get()}")
            print(f"Data shape: {df.shape}")
            print("Columns available:", df.columns.tolist())
            
            # Clear previous plots
            print("Clearing previous plots...")
            self.figure.clear()
            
            # Create subplots with specific heights
            print("Creating subplot layout...")
            gs = self.figure.add_gridspec(2, 1, height_ratios=[3, 1])
            ax_price = self.figure.add_subplot(gs[0])  # Price plot (larger)
            ax_volume = self.figure.add_subplot(gs[1])  # Volume plot (smaller)
            
            # Convert date to datetime if it's not already
            print("Processing date column...")
            df['date'] = pd.to_datetime(df['date'])
            
            # Create OHLC data for mplfinance
            df_mpf = df.set_index('date')
            
            # Plot using mplfinance
            mpf.plot(df_mpf, type='candle', style='charles',
                    volume=True, 
                    ax=ax_price,
                    volume_panel=1,
                    volume_axis=ax_volume,
                    show_nontrading=False,
                    update_width_config=dict(candle_linewidth=0.6))
            
            # Add moving averages if we have enough data
            print("Adding technical indicators...")
            if len(df) > 50:
                df['MA20'] = df['Close'].rolling(window=20).mean()
                df['MA50'] = df['Close'].rolling(window=50).mean()
                ax_price.plot(df['date'], df['MA20'], label='20-day MA', color='blue', alpha=0.7)
                ax_price.plot(df['date'], df['MA50'], label='50-day MA', color='orange', alpha=0.7)
                print("Added moving averages")
            
            # Customize price plot
            print("Customizing price plot...")
            ax_price.set_title(f'{ticker} Price History')
            ax_price.set_ylabel('Price')
            ax_price.grid(True, alpha=0.3)
            ax_price.legend()
            
            # Format x-axis dates
            print("Formatting axes...")
            for ax in [ax_price, ax_volume]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Add RSI subplot if available
            if 'RSI' in df.columns:
                print("Adding RSI subplot...")
                gs = self.figure.add_gridspec(3, 1, height_ratios=[3, 1, 1])
                ax_rsi = self.figure.add_subplot(gs[2])
                ax_rsi.plot(df['date'], df['RSI'], color='purple', label='RSI')
                ax_rsi.axhline(y=70, color='r', linestyle='--', alpha=0.5)
                ax_rsi.axhline(y=30, color='g', linestyle='--', alpha=0.5)
                ax_rsi.set_title('RSI (14)')
                ax_rsi.set_ylabel('RSI')
                ax_rsi.set_ylim([0, 100])
                ax_rsi.grid(True, alpha=0.3)
            
            # Adjust layout and display
            print("Finalizing plot layout...")
            self.figure.tight_layout()
            self.canvas.draw()
            
            print("Plot update complete")
            self.loading_label.config(text=f"Updated plot for {ticker}")
            
        except Exception as e:
            print(f"Error updating plots: {str(e)}")
            traceback.print_exc()
            self.loading_label.config(text=f"Error updating plot: {str(e)}")

    def find_duckdb_databases(self):
        """Find all DuckDB database files in current directory"""
        try:
            dbs = []
            for file in os.listdir('.'):
                if file.endswith('.db'):
                    try:
                        # Try to connect to verify it's a valid DuckDB database
                        test_conn = duckdb.connect(file)
                        test_conn.close()
                        dbs.append(file)
                    except:
                        continue
            return dbs
        except Exception as e:
            print(f"Error finding databases: {str(e)}")
            return []

    def refresh_databases(self):
        """Refresh the list of available databases"""
        try:
            self.available_dbs = self.find_duckdb_databases()
            print(f"Refreshed database list: {self.available_dbs}")
            
            self.db_combo['values'] = self.available_dbs
            if self.available_dbs:
                if self.db_combo.get() not in self.available_dbs:
                    self.db_combo.set(self.available_dbs[0])
                    self.on_database_change()
        except Exception as e:
            print(f"Error refreshing databases: {e}")
            messagebox.showerror("Refresh Error", str(e))

    def run(self):
        """Start the GUI main loop"""
        try:
            print("Starting main loop...")
            self.root.mainloop()
        except Exception as e:
            print(f"Error in main loop: {str(e)}")
            traceback.print_exc()

    def update_tickers(self):
        """Update available tickers based on current table selection"""
        if not hasattr(self, 'db_conn') or not self.db_conn:
            self.clear_ticker_selection()
            return
            
        table = self.table_var.get()
        if not table or table == 'No tables available':
            self.clear_ticker_selection()
            return
            
        try:
            # Get column information
            columns = self.db_conn.execute(f"SELECT * FROM {table} LIMIT 0").description
            column_names = [col[0] for col in columns]
            
            # Check if table has ticker column
            if 'ticker' in column_names:
                # Get unique tickers from the table
                tickers = self.db_conn.execute(f"SELECT DISTINCT ticker FROM {table}").fetchall()
                tickers = [t[0] for t in tickers]
                
                # Update ticker combobox
                self.ticker_combo['values'] = tickers
                if tickers:
                    self.ticker_combo.set(tickers[0])
                    self.update_ticker_description()
                else:
                    self.clear_ticker_selection()
            else:
                self.clear_ticker_selection()
                
        except Exception as e:
            print(f"Error updating tickers: {e}")
            self.clear_ticker_selection()

    def update_plots_with_predictions(self, df, predictions):
        """Update plots to include AI predictions"""
        try:
            print("\nUpdating plots with predictions...")
            print(f"DataFrame length: {len(df)}, Predictions length: {len(predictions)}")
            
            # Create a copy of the dataframe
            df_pred = df.copy()
            
            # Create predictions array with NaN values
            pred_array = np.full(len(df_pred), np.nan)
            
            # Place the predictions at the end of the array
            pred_array[-len(predictions):] = predictions
            
            # Add predictions to dataframe
            df_pred['Predictions'] = pred_array
            
            # Update plots
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            # Plot actual prices
            ax.plot(df_pred.index, df_pred['Close'], label='Actual', color='blue')
            
            # Plot predictions (only the last few points)
            mask = ~np.isnan(pred_array)
            if np.any(mask):
                ax.plot(df_pred.index[mask], pred_array[mask], 
                       label='Predicted', color='red', linestyle='--')
            
            # Add future predictions if available
            if hasattr(self, 'future_dates') and hasattr(self, 'future_predictions'):
                ax.plot(self.future_dates, self.future_predictions, 
                       label='Future Predictions', color='green', linestyle=':')
            
            # Customize plot
            ax.set_title(f'Stock Price Prediction for {self.ticker_var.get()}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            ax.grid(True)
            
            # Format dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Update display
            self.figure.tight_layout()
            self.canvas.draw()
            
            print("Plot updated successfully with predictions")
            
        except Exception as e:
            print(f"Error updating plots with predictions: {str(e)}")
            traceback.print_exc()

    def create_plot_area(self):
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

    def create_status_bar(self):
        """Create status bar"""
        try:
            print("\nCreating status bar...")
            status_frame = ttk.Frame(self.root)
            status_frame.pack(fill="x", side="bottom", padx=5, pady=2)
            
            self.status_label = ttk.Label(status_frame, text="Ready")
            self.status_label.pack(side="left", padx=5)
            
            print("Status bar creation complete")
            
        except Exception as e:
            print(f"Error creating status bar: {str(e)}")
            traceback.print_exc()

    def setup_initial_database(self):
        """Setup initial database connection and populate controls"""
        try:
            print("\nSetting up initial database connection...")
            
            if not self.available_dbs:
                print("No databases available")
                self.loading_label.config(text="No databases found")
                return
            
            # Set initial database
            self.current_db = self.available_dbs[0]
            print(f"Selected initial database: {self.current_db}")
            
            # Connect to database
            print("Establishing connection...")
            self.db_conn = duckdb.connect(self.current_db)
            
            # Update database combo
            print("Updating database selection...")
            self.db_combo['values'] = self.available_dbs
            self.db_combo.set(self.current_db)
            
            # Get and set tables
            print("Getting available tables...")
            self.tables = self.get_tables()
            if self.tables:
                print(f"Found tables: {self.tables}")
                self.table_combo['values'] = self.tables
                self.table_combo.set(self.tables[0])
                
                # Get initial tickers
                print("Getting initial tickers...")
                self.refresh_tickers()
            else:
                print("No tables found")
                self.loading_label.config(text="No tables found in database")
            
            print("Initial database setup complete")
            
        except Exception as e:
            print(f"Error in initial database setup: {str(e)}")
            traceback.print_exc()
            self.loading_label.config(text=f"Database initialization error: {str(e)}")

    def refresh_tickers(self):
        """Refresh the list of available tickers for the current table"""
        try:
            if not self.current_db or not self.table_var.get():
                return
            
            query = f"""
                SELECT DISTINCT ticker 
                FROM {self.table_var.get()}
                ORDER BY ticker
            """
            
            tickers = self.db_conn.execute(query).fetchall()
            if tickers is not None and len(tickers) > 0:
                tickers = [t[0] for t in tickers]
                self.ticker_combo['values'] = tickers
                if tickers:
                    self.ticker_combo.set(tickers[0])
                    
        except Exception as e:
            print(f"Error refreshing tickers: {str(e)}")
            traceback.print_exc()

    def prepare_data_for_training(self, df):
        """Prepare data for LSTM training"""
        try:
            print("Preparing data for training...")
            
            # Calculate technical indicators
            df_processed = self.calculate_technical_indicators(df)
            if df_processed is None:
                print("Failed to calculate technical indicators")
                return None, None
            
            # Select features for training
            features = ['Close', 'Volume', 'MA20', 'MA50', 'RSI', 'MACD', 'Signal_Line', 
                       'BB_upper', 'BB_middle', 'BB_lower']
            
            # Ensure all features exist
            if not all(feature in df_processed.columns for feature in features):
                print(f"Missing features. Available columns: {df_processed.columns}")
                return None, None
            
            # Create the feature dataset
            data = df_processed[features].values
            
            # Scale the data
            self.scaler = MinMaxScaler()
            scaled_data = self.scaler.fit_transform(data)
            
            # Create sequences
            X = []
            y = []
            
            for i in range(self.sequence_length, len(scaled_data)):
                X.append(scaled_data[i-self.sequence_length:i])
                y.append(scaled_data[i, 0])  # Predict the Close price
                
            X = np.array(X)
            y = np.array(y)
            
            print(f"Data preparation complete. X shape: {X.shape}, y shape: {y.shape}")
            return X, y
            
        except Exception as e:
            print(f"Error preparing training data: {str(e)}")
            traceback.print_exc()
            return None, None

    def calculate_technical_indicators(self, df):
        """Calculate technical indicators for the dataset"""
        try:
            print("Starting technical indicator calculations...")
            
            # Create a copy to avoid modifying the original dataframe
            df = df.copy()
            
            # Calculate Moving Averages
            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA50'] = df['Close'].rolling(window=50).mean()
            
            # Calculate RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
            # Calculate Bollinger Bands
            df['BB_middle'] = df['Close'].rolling(window=20).mean()
            df['BB_upper'] = df['BB_middle'] + 2 * df['Close'].rolling(window=20).std()
            df['BB_lower'] = df['BB_middle'] - 2 * df['Close'].rolling(window=20).std()
            
            # Calculate Average True Range (ATR)
            high_low = df['High'] - df['Low']
            high_close = abs(df['High'] - df['Close'].shift())
            low_close = abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['ATR'] = true_range.rolling(window=14).mean()
            
            # Calculate Volume Moving Average
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            
            # Drop any rows with NaN values
            df = df.dropna()
            
            print("Technical indicators calculated successfully")
            return df
            
        except Exception as e:
            print(f"Error calculating technical indicators: {str(e)}")
            traceback.print_exc()
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
