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
                # First LSTM layer with input_shape=(sequence_length, n_features)
                LSTM(50, return_sequences=True, input_shape=input_shape),
                BatchNormalization(),
                Dropout(0.2),
                
                # Second LSTM layer
                LSTM(50, return_sequences=True),
                BatchNormalization(),
                Dropout(0.2),
                
                # Third LSTM layer
                LSTM(50),
                BatchNormalization(),
                Dropout(0.2),
                
                # Dense layers for prediction
                Dense(25, activation='relu'),
                Dense(1)
            ])
            
            # Compile model with Adam optimizer and MSE loss
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']  # Adding mean absolute error as a metric
            )
            
            self.model = model
            print("Model built successfully")
            model.summary()  # Print model architecture
            return True
            
        except Exception as e:
            print(f"Error building model: {str(e)}")
            traceback.print_exc()
            return False

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the LSTM model"""
        try:
            print("\nStarting model training...")
            print(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")
            print(f"Validation data shape: X_val={X_val.shape}, y_val={y_val.shape}")
            
            if self.model is None:
                raise ValueError("Model not built. Call build_model first.")
            
            # Early stopping callback with more patience
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
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
        self.sequence_length = sequence_length
        # Define features in lowercase to match DataFrame columns
        self.features = features or [
            'open', 'high', 'low', 'close', 'volume',
            'ma20', 'rsi', 'macd', 'ma50'  # All 9 features in lowercase
        ]
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def prepare_training_data(self, df):
        try:
            print("\nPreparing training data...")
            if df is None or df.empty:
                print("Error: DataFrame is empty")
                return None

            print("DataFrame columns:", df.columns.tolist())
            print("DataFrame head:\n", df.head())

            # Convert all column names to lowercase
            df.columns = df.columns.str.lower()
            
            # Find date column
            date_column = None
            for col in df.columns:
                if 'date' in col.lower():
                    date_column = col
                    print(f"Found date column: {col}")
                    break

            if date_column is None:
                raise ValueError("No date column found")

            # Set the date column as index
            df.set_index(date_column, inplace=True)
            
            # Convert index to datetime
            print("Index before datetime conversion:", df.index)
            df.index = pd.to_datetime(df.index)
            print("Converting index to datetime...")
            print("Index after datetime conversion:", df.index)
            df.sort_index(inplace=True)

            # Calculate technical indicators
            df = self.calculate_technical_indicators(df)
            
            # Verify all required features are present (case-insensitive)
            df_cols_lower = df.columns.str.lower()
            missing_features = [feat for feat in self.features if feat not in df_cols_lower]
            
            if missing_features:
                print(f"Error: Missing required features: {missing_features}")
                print("Available columns:", df.columns.tolist())
                return None

            # Create feature matrix using only the specified features (case-insensitive)
            feature_data = df[[col for col in df.columns if col.lower() in self.features]].values
            
            if len(feature_data) < self.sequence_length:
                print(f"Error: Not enough data points. Need at least {self.sequence_length}")
                return None

            # Scale features
            scaled_data = self.scaler.fit_transform(feature_data)
            
            # Create sequences
            X, y = self._create_sequences(scaled_data)
            
            if len(X) == 0:
                print("Error: No sequences created")
                return None

            # Split into training and validation sets
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
            print(f"Feature shape: X_train={X_train.shape}, X_val={X_val.shape}")
            return (X_train, y_train), (X_val, y_val)

        except Exception as e:
            print(f"Error preparing training data: {str(e)}")
            traceback.print_exc()
            return None
    
    def prepare_prediction_data(self, df):
        """Prepare data for prediction"""
        try:
            print("\nPreparing prediction data...")
            if df is None or df.empty:
                print("Error: DataFrame is empty")
                return None

            print("Initial DataFrame shape:", df.shape)
            print("DataFrame columns:", df.columns.tolist())

            # Convert all column names to lowercase
            df.columns = df.columns.str.lower()
            
            # Find and process date column
            date_column = next((col for col in df.columns if 'date' in col.lower()), None)
            if date_column:
                df.set_index(date_column, inplace=True)
                df.index = pd.to_datetime(df.index)
                df.sort_index(inplace=True)
            
            # Calculate technical indicators
            df = self.calculate_technical_indicators(df)
            
            # Verify required features
            required_features = ['open', 'high', 'low', 'close', 'volume', 'ma20', 'rsi', 'macd', 'ma50']
            missing_features = [feat for feat in required_features if feat not in df.columns.str.lower()]
            
            if missing_features:
                print(f"Error: Missing required features: {missing_features}")
                return None

            # Get the most recent sequence
            feature_data = df[required_features].values
            if len(feature_data) < self.sequence_length:
                print(f"Error: Not enough data points. Need at least {self.sequence_length}")
                return None

            # Scale the features
            scaled_data = self.scaler.fit_transform(feature_data)
            
            # Take the most recent sequence
            recent_sequence = scaled_data[-self.sequence_length:]
            
            # Reshape for prediction
            prediction_data = np.reshape(recent_sequence, (1, self.sequence_length, len(required_features)))
            
            print(f"Prediction data shape: {prediction_data.shape}")
            return prediction_data

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

    def calculate_technical_indicators(self, df):
        """Calculate technical indicators"""
        try:
            # Ensure column names are lowercase
            df.columns = df.columns.str.lower()
            
            # Calculate MA20
            df['ma20'] = df['close'].rolling(window=20).mean()
            
            # Calculate RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            
            # Calculate MA50
            df['ma50'] = df['close'].rolling(window=50).mean()
            
            print("Technical indicators calculated successfully.")
            print("DataFrame with technical indicators:\n", df.head())
            
            return df
            
        except Exception as e:
            print(f"Error calculating technical indicators: {str(e)}")
            traceback.print_exc()
            return df

class StockAIAgent:
    def __init__(self):
        """Initialize the AI agent with model architecture"""
        try:
            print("\nInitializing AI Agent...")
            self.model = None
            self.data_adapter = DataAdapter(
                sequence_length=self.sequence_length,
                features=['open', 'high', 'low', 'close', 'volume', 'rsi', 'ma20', 'ma50', 'macd']
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
            print("\n=== Starting Prediction ===")
            
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
            print("Retrieved columns:", df.columns)
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
                features=['open', 'high', 'low', 'close', 'volume', 'rsi', 'ma20', 'ma50', 'macd']
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
            
            # Create figure and axis
            self.fig = Figure(figsize=(10, 8), dpi=100)
            self.ax = self.fig.add_subplot(111)  # Add this line to create the axis
            self.fig.set_facecolor('#f0f0f0')
            
            # Create canvas
            print("Creating canvas...")
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_panel)
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
            
            # Initialize field_vars to store checkbox variables
            self.field_vars = {
                'Close': tk.BooleanVar(),
                'Open': tk.BooleanVar(),
                'High': tk.BooleanVar(),
                'Low': tk.BooleanVar(),
                'Volume': tk.BooleanVar()
            }

            # Create checkboxes for field selection
            for i, (field, var) in enumerate(self.field_vars.items()):
                checkbox = ttk.Checkbutton(self.field_frame, text=field, variable=var)
                checkbox.grid(row=i // 2, column=i % 2, sticky="w", padx=5, pady=5)
            
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
            
            # Initialize ticker listbox
            self.ticker_listbox = tk.Listbox(ticker_frame, selectmode=tk.MULTIPLE)
            self.ticker_listbox.pack(fill="x", padx=5, pady=2)
            
            # Add "Clear All" and "Select All" buttons
            clear_all_button = ttk.Button(ticker_frame, text="Clear All", command=self.clear_all_tickers)
            clear_all_button.pack(side="left", padx=5, pady=5)

            select_all_button = ttk.Button(ticker_frame, text="Select All", command=self.select_all_tickers)
            select_all_button.pack(side="left", padx=5, pady=5)
            
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
            print("Retrieved columns:", df.columns)
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
            selected_fields = self.get_selected_fields()  # Method to get selected fields from checkboxes
            
            print(f"Current Database: {current_db}")
            print(f"Selected Table: {current_table}")
            print(f"Selected Sector: {current_sector}")
            print(f"Selected Ticker: {current_ticker}")
            print(f"Selected Fields: {selected_fields}")
            
            # Validate selections
            if not current_db or current_db == 'No databases available':
                print("No valid database selected")
                return
            
            # Connect to the DuckDB database
            conn = create_connection(current_db)
            if conn is None:
                print("Failed to connect to the database")
                return
            
            # Construct query with selected fields
            fields_str = ', '.join(selected_fields)
            query = f"SELECT {fields_str} FROM {current_table} WHERE ticker = ?"
            df = conn.execute(query, (current_ticker,)).fetchdf()
            print(f"Retrieved {len(df)} rows of data")
            
            # Check for required columns
            required_columns = ['close', 'open', 'high', 'low', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"Error: Missing required columns: {missing_columns}")
                print("Skipping model training due to missing technical indicators.")
                return
            
            # Proceed with technical indicator calculations and model training
            df = self.calculate_technical_indicators(df)
            if df is None:
                print("Skipping model training due to missing technical indicators.")
                return
            
            # Check if data is retrieved
            if df is None or df.empty:
                print("No data retrieved from the table. Please check your query and selections.")
                self.loading_label.config(text="Training error: No data retrieved")
                return
            
            # Proceed with model training using the prepared data
            if df is not None:
                # Initialize AI agent if not already done
                if self.ai_agent.model is None:
                    input_shape = (self.sequence_length, len(selected_fields))
                    self.ai_agent.build_model(input_shape)
                
                # Prepare data using the data adapter
                data = self.data_adapter.prepare_training_data(df)
                if data is None:
                    print("Data preparation failed. Please check your data and try again.")
                    self.loading_label.config(text="Training error: Data preparation failed")
                    return
                
                (X_train, y_train), (X_val, y_val) = data
                
                # Train the model
                history = self.ai_agent.train(X_train, y_train, X_val, y_val, 
                                              epochs=int(self.epochs_var.get()), 
                                              batch_size=int(self.batch_size_var.get()))
                
                if history is not None:
                    print("Model training completed successfully")
                    self.ai_status.config(text="AI Status: Trained")
                else:
                    print("Model training failed")
                    self.ai_status.config(text="AI Status: Training failed")
            
        except Exception as e:
            print("\n=== Error in Training ===")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("\nTraceback:")
            traceback.print_exc()
            self.loading_label.config(text=f"Training error: {str(e)}")

    def get_selected_fields(self):
        """Retrieve the list of selected fields from checkboxes."""
        return [field for field, var in self.field_vars.items() if var.get()]

    def plot_training_history(self, history):
        """Plot training history"""
        try:
            # Clear previous plots
            self.fig.clear()
            
            # Create subplot for loss
            ax = self.fig.add_subplot(111)
            ax.plot(history.history['loss'], label='Training Loss')
            ax.plot(history.history['val_loss'], label='Validation Loss')
            ax.set_title('Model Training History')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True)
            
            # Update canvas
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error plotting training history: {str(e)}")
            traceback.print_exc()

    def make_prediction(self):
        """Make prediction using the trained model"""
        try:
            print("\n=== Starting Prediction ===")
            
            if self.ai_agent.model is None:
                raise ValueError("Model not trained yet")
            
            # Get current selections
            current_db = self.db_combo.get()
            current_table = self.table_var.get() if hasattr(self, 'table_var') else ""
            current_ticker = self.ticker_var.get()
            
            print(f"Making prediction for {current_ticker}")
            
            # Connect to database
            conn = create_connection(current_db)
            if not conn:
                raise ValueError("Failed to connect to database")
            
            try:
                # Construct query
                if current_table:
                    query = f"""
                        SELECT * FROM {current_table} 
                        WHERE ticker = ? 
                        ORDER BY date DESC 
                        LIMIT {self.data_adapter.sequence_length + 10}
                    """
                else:
                    query = f"""
                        SELECT * FROM {current_ticker} 
                        ORDER BY date DESC 
                        LIMIT {self.data_adapter.sequence_length + 10}
                    """
                
                print(f"Executing query: {query}")
                df = conn.execute(query, (current_ticker,) if current_table else ()).fetchdf()
                print(f"Retrieved {len(df)} rows of data")
                
                if df.empty:
                    raise ValueError("No data retrieved for prediction")
                
                # Prepare data for prediction
                prediction_data = self.data_adapter.prepare_prediction_data(df)
                if prediction_data is None:
                    raise ValueError("Failed to prepare prediction data")
                
                # Make prediction
                prediction = self.ai_agent.model.predict(prediction_data)
                
                # Inverse transform the prediction
                last_close = df['close'].iloc[-1]
                predicted_change = prediction[0][0]  # Assuming single prediction
                predicted_price = last_close * (1 + predicted_change)
                
                print(f"\nPrediction Results:")
                print(f"Current Price: ${last_close:.2f}")
                print(f"Predicted Price: ${predicted_price:.2f}")
                print(f"Predicted Change: {predicted_change*100:.2f}%")
                
                # Update plot with prediction
                self.update_plot_with_prediction(df, predicted_price)
                
                return predicted_price
                
            except Exception as e:
                print(f"Error in prediction: {str(e)}")
                traceback.print_exc()
                return None
                
            finally:
                conn.close()
                
        except Exception as e:
            print("\n=== Error in Prediction ===")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("\nTraceback:")
            traceback.print_exc()
            return None

    def update_plot_with_prediction(self, df, predicted_price):
        """Update plot with prediction"""
        try:
            self.ax.clear()
            
            # Plot historical data
            dates = df.index[-30:]  # Last 30 days
            prices = df['close'][-30:]
            self.ax.plot(dates, prices, label='Historical')
            
            # Add prediction point
            next_date = dates[-1] + pd.Timedelta(days=1)
            self.ax.scatter(next_date, predicted_price, color='red', label='Prediction')
            
            # Customize plot
            self.ax.set_title(f'Stock Price Prediction for {self.ticker_var.get()}')
            self.ax.set_xlabel('Date')
            self.ax.set_ylabel('Price')
            self.ax.legend()
            self.ax.grid(True)
            
            # Rotate x-axis labels
            plt.xticks(rotation=45)
            
            # Adjust layout and redraw
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error updating plot: {str(e)}")
            traceback.print_exc()

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
                    self.sector_combo.set(sectors[0])
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
        """Update tickers and symbols based on the current table"""
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
            
        except Exception as e:
            print(f"Error updating tickers and symbols: {str(e)}")
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
        required_columns = ['close', 'open', 'high', 'low', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return None
        
        try:
            # Example: Calculate a 20-day moving average
            df['MA20'] = df['close'].rolling(window=20).mean()
            # Add more technical indicators as needed
            print("Technical indicators calculated successfully.")
            return df
        except Exception as e:
            print(f"Error calculating technical indicators: {e}")
            return None

    def clear_all_tickers(self):
        """Clear all selections in the ticker listbox"""
        self.ticker_listbox.selection_clear(0, tk.END)

    def select_all_tickers(self):
        """Select all items in the ticker listbox"""
        self.ticker_listbox.selection_set(0, tk.END)

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
