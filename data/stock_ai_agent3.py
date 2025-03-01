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
from tensorflow.keras.losses import Huber

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
    """AI agent for stock price prediction."""
    
    def __init__(self, db_conn=None):
        """Initialize the AI agent."""
        self.db_conn = db_conn
        self.model = None
        self.history = None
        self.scaler = None
        self.prediction = None
        self.learning_rate = 0.001  # Add default learning rate
        self.batch_size = 32
        self.epochs = 50
        self.sequence_length = 10
        
    def build_model(self, input_shape):
        """Build and compile the LSTM model."""
        try:
            print(f"Building model with input shape: {input_shape}")
            model = Sequential()
            
            # First LSTM layer
            model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            
            # Second LSTM layer
            model.add(LSTM(50, return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            
            # Third LSTM layer
            model.add(LSTM(50, return_sequences=False))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            
            # Dense output layers
            model.add(Dense(25, activation='relu'))
            model.add(Dense(1))
            
            # Use the Huber() class instead of the string 'huber_loss'
            model.compile(optimizer=Adam(learning_rate=self.learning_rate), 
                         loss=Huber(), 
                         metrics=['mae'])
            
            print("Model built successfully")
            model.summary()
            
            # This is the critical fix - assign the model to self.model
            self.model = model
            
            return model
        except Exception as e:
            print(f"Error building model: {e}")
            traceback.print_exc()
            return None

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
        
    def prepare_training_data(self, data, sequence_length=10):
        """Prepare data for training by creating sequences."""
        try:
            print("\nPreparing training data...")
            print(f"DataFrame columns: {data.columns.tolist()}")
            print("DataFrame head:")
            print(data.head())
            
            # Check if we have a date column
            date_col = None
            for col in data.columns:
                if col.lower() in ['date', 'datetime', 'time']:
                    date_col = col
                    print(f"Found date column: {date_col}")
                    break
            
            # If we found a date column, set it as index
            if date_col:
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
                    data[date_col] = pd.to_datetime(data[date_col])
                
                # Set as index
                data = data.set_index(date_col)
                print(f"Index before datetime conversion: {data.index}")
                
                # Ensure index is datetime
                data.index = pd.to_datetime(data.index)
                print(f"Index after datetime conversion: {data.index}")
                
                # Sort by date
                data = data.sort_index()
            
            # Calculate technical indicators again (on clean data)
            data = self.calculate_technical_indicators(data)
            print("Technical indicators calculated successfully.")
            print("DataFrame with technical indicators:")
            print(data.head())
            
            # Remove non-numeric columns
            numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
            data = data[numeric_cols]
            
            # Handle any remaining NaN values
            data = data.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Normalize the data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)
            self.scaler = scaler
            
            # Create sequences for LSTM
            X, y = [], []
            for i in range(len(scaled_data) - sequence_length):
                X.append(scaled_data[i:i + sequence_length])
                y.append(scaled_data[i + sequence_length, data.columns.get_loc('close')])
            
            X, y = np.array(X), np.array(y)
            
            # Split into training and validation sets
            train_size = int(len(X) * 0.8)
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]
            
            print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
            print(f"Feature shape: X_train={X_train.shape}, X_val={X_val.shape}")
            
            return X_train, X_val, y_train, y_val
        except Exception as e:
            print(f"Error preparing training data: {e}")
            traceback.print_exc()
            return None, None, None, None
    
    def prepare_prediction_data(self, data):
        """Prepare data for prediction."""
        try:
            print("Preparing prediction data...")
            print(f"Initial DataFrame shape: {data.shape}")
            print(f"DataFrame columns: {list(data.columns)}")
            
            # Calculate technical indicators
            df = self.calculate_technical_indicators(data.copy())
            
            # Important: Save the original scaler feature names
            if hasattr(self, 'scaler') and hasattr(self.scaler, 'feature_names_in_'):
                print(f"Scaler was fitted with these features: {self.scaler.feature_names_in_}")
                feature_columns = list(self.scaler.feature_names_in_)
            else:
                # Fall back to default feature columns
                feature_columns = ['open', 'high', 'low', 'close', 'volume', 'MA20', 'ma50', 'rsi', 'macd', 'prediction']
            
            # Check if MA20 exists in dataframe (case matters!)
            if 'MA20' in df.columns:
                # If we need ma20 (lowercase) and only have MA20, create it
                if 'MA20' in feature_columns and 'ma20' not in df.columns:
                    df['ma20'] = df['MA20']
            elif 'ma20' in df.columns:
                # If we need MA20 (uppercase) and only have ma20, create it
                if 'MA20' in feature_columns and 'MA20' not in df.columns:
                    df['MA20'] = df['ma20']
                
            # Do the same for ma50/MA50
            if 'MA50' in df.columns and 'ma50' not in df.columns:
                df['ma50'] = df['MA50']
            elif 'ma50' in df.columns and 'MA50' not in df.columns:
                df['MA50'] = df['ma50']
            
            # Ensure all required columns exist
            for col in feature_columns:
                if col not in df.columns:
                    print(f"Adding missing column: {col}")
                    df[col] = 0  # Add missing columns with default values
            
            # Select only the necessary features in the same order as during training
            df = df[feature_columns]
            
            print(f"Final prediction features: {list(df.columns)}")
            
            # Scale the features
            if hasattr(self, 'scaler'):
                print("Using fitted scaler for prediction data")
                # Important: Use the same column order as during training
                df_scaled = pd.DataFrame(self.scaler.transform(df),
                                       columns=df.columns,
                                       index=df.index)
            else:
                print("Warning: No scaler found, using raw features")
                df_scaled = df
            
            # Create sequences for LSTM (same window size as training)
            X_samples = []
            for i in range(df_scaled.shape[0] - 9):
                X_samples.append(df_scaled.values[i:i+10])
            
            if not X_samples:
                print("Warning: Not enough data points for a prediction window")
                # If we don't have enough data for a full window, use what we have
                if df_scaled.shape[0] > 0:
                    # Pad with zeros if needed
                    pad_size = 10 - df_scaled.shape[0]
                    if pad_size > 0:
                        padding = np.zeros((pad_size, df_scaled.shape[1]))
                        padded_data = np.vstack((padding, df_scaled.values))
                    else:
                        padded_data = df_scaled.values
                    X_samples.append(padded_data[-10:])
            
            if X_samples:
                prediction_data = np.array(X_samples)
                print(f"Prediction data shape: {prediction_data.shape}")
                return prediction_data
            else:
                print("Error: Could not create prediction data")
                return None
        except Exception as e:
            print(f"Error preparing prediction data: {e}")
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

    def calculate_technical_indicators(self, data):
        """Calculate technical indicators for stock data."""
        try:
            # Create a working copy
            df = data.copy()
            
            # Handle any missing values in the input data
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Create a DataFrame with technical indicators
            if 'close' in df.columns:
                # Moving Averages
                df['ma20'] = df['close'].rolling(window=20).mean()
                df['ma50'] = df['close'].rolling(window=50).mean()
                
                # RSI - Relative Strength Index
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss.replace(0, 0.001)  # Avoid division by zero
                df['rsi'] = 100 - (100 / (1 + rs))
                
                # MACD - Moving Average Convergence Divergence
                ema12 = df['close'].ewm(span=12, adjust=False).mean()
                ema26 = df['close'].ewm(span=26, adjust=False).mean()
                df['macd'] = ema12 - ema26
            
            print("Technical indicators calculated successfully.")
            # Handle NaN values for all technical indicators
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            return df
        except Exception as e:
            print(f"Error calculating technical indicators: {e}")
            traceback.print_exc()
            return data

class StockAIAgent:
    def __init__(self):
        """Initialize the AI agent with model architecture"""
        try:
            print("\nInitializing AI Agent...")
            self.model = None
            self.sequence_length = 60  # Number of time steps to look back
            self.features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI']
            self.data_adapter = DataAdapter(
                sequence_length=self.sequence_length,
                features=['open', 'high', 'low', 'close', 'volume', 'rsi', 'ma20', 'ma50', 'macd']
            )
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            print("AI Agent initialized successfully")
            
        except Exception as e:
            print(f"Error initializing AI Agent: {str(e)}")
            traceback.print_exc()

    def build_model(self, input_shape):
        """Build and compile the LSTM model."""
        try:
            print(f"Building model with input shape: {input_shape}")
            model = Sequential()
            
            # First LSTM layer
            model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            
            # Second LSTM layer
            model.add(LSTM(50, return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            
            # Third LSTM layer
            model.add(LSTM(50, return_sequences=False))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            
            # Dense output layers
            model.add(Dense(25, activation='relu'))
            model.add(Dense(1))
            
            # Use the Huber() class instead of the string 'huber_loss'
            model.compile(optimizer=Adam(learning_rate=self.learning_rate), 
                         loss=Huber(), 
                         metrics=['mae'])
            
            print("Model built successfully")
            model.summary()
            
            # This is the critical fix - assign the model to self.model
            self.model = model
            
            return model
        except Exception as e:
            print(f"Error building model: {e}")
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
                
            (X_train, X_val, y_train, y_val) = data
            
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
                
    def update_plot_with_prediction(self, dates, prices, predicted_price):
        """Update the plot with predicted price."""
        try:
            if not dates or not prices:
                print("No data to plot")
                return
            
            # Make sure dates is a list of datetime objects, not integers
            if isinstance(dates[-1], (int, float)):
                print(f"Converting date from numeric: {dates[-1]}")
                # If it's an integer timestamp, convert to datetime
                try:
                    # Attempt to convert from milliseconds timestamp
                    last_date = pd.to_datetime(dates[-1], unit='ms')
                except:
                    # If that fails, try using the last date with an incremental day
                    if len(dates) > 1 and isinstance(dates[-2], pd.Timestamp):
                        last_date = dates[-2] + pd.Timedelta(days=1)
                    else:
                        # Last resort: use current date
                        last_date = pd.Timestamp.now()
                dates[-1] = last_date
            
            # Now we can safely add a timedelta
            next_date = dates[-1] + pd.Timedelta(days=1)
            
            # The rest of the plotting code continues as before
            dates_with_prediction = list(dates) + [next_date]
            prices_with_prediction = list(prices) + [predicted_price]
            
            # Clear the plot
            self.ax.clear()
            
            # Plot the historical data
            self.ax.plot(dates, prices, label='Historical Prices')
            
            # Highlight the prediction
            self.ax.plot([dates[-1], next_date], [prices[-1], predicted_price], 'r-', linewidth=2, label='Prediction')
            self.ax.scatter([next_date], [predicted_price], color='red', s=50)
            
            # Add labels and legend
            self.ax.set_xlabel('Date')
            self.ax.set_ylabel('Price')
            self.ax.set_title(f'Price Prediction')
            self.ax.legend()
            
            # Format x-axis to show dates nicely
            self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            self.ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.xticks(rotation=45)
            
            # Redraw the plot
            self.canvas.draw()
        except Exception as e:
            print(f"Error updating plot: {e}")
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
            
            # Check if table has ticker column
            if 'ticker' in [col.lower() for col in column_names]:
                print("Found ticker column, retrieving unique tickers...")
                ticker_col = next(col for col in column_names if col.lower() == 'ticker')
                tickers = self.db_conn.execute(
                    f"SELECT DISTINCT {ticker_col} FROM {table} ORDER BY {ticker_col}"
                ).fetchall()
                tickers = [t[0] for t in tickers]
                print(f"Found {len(tickers)} tickers")
                
                # Update ticker combobox
                self.ticker_combo['values'] = tickers
                if tickers:
                    self.ticker_combo.set(tickers[0])
                    
                # Update ticker listbox
                self.ticker_listbox.delete(0, tk.END)
                for ticker in tickers:
                    self.ticker_listbox.insert(tk.END, ticker)
            else:
                print("No ticker column found in table")
                self.clear_ticker_selection()
            
            # Check if table has sector column
            if 'sector' in [col.lower() for col in column_names]:
                print("Found sector column, retrieving unique sectors...")
                sector_col = next(col for col in column_names if col.lower() == 'sector')
                sectors = self.db_conn.execute(
                    f"SELECT DISTINCT {sector_col} FROM {table} ORDER BY {sector_col}"
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
            
        except Exception as e:
            print(f"Error in table change handler: {str(e)}")
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
            
            # Check if the sector and ticker columns exist
            columns = self.db_conn.execute(f"SELECT * FROM {table} LIMIT 0").description
            column_names = [col[0] for col in columns]
            column_names_lower = [col.lower() for col in column_names]
            
            if 'sector' in column_names_lower and 'ticker' in column_names_lower:
                # Get the actual column names (preserve case)
                sector_col = next(col for col in column_names if col.lower() == 'sector')
                ticker_col = next(col for col in column_names if col.lower() == 'ticker')
                
                # Query to get tickers for the selected sector
                tickers = self.db_conn.execute(
                    f"SELECT DISTINCT {ticker_col} FROM {table} WHERE {sector_col} = ? ORDER BY {ticker_col}", 
                    (selected_sector,)
                ).fetchall()
                tickers = [t[0] for t in tickers]
                
                # Update ticker combobox
                self.ticker_combo['values'] = tickers
                if tickers:
                    self.ticker_combo.set(tickers[0])
                else:
                    self.clear_ticker_selection()
                    
                # Update ticker listbox
                self.ticker_listbox.delete(0, tk.END)
                for ticker in tickers:
                    self.ticker_listbox.insert(tk.END, ticker)
                    
                print(f"Updated tickers based on sector '{selected_sector}': {len(tickers)} tickers found")
            else:
                self.clear_ticker_selection()
                print("Sector or ticker column not found in table")
            
        except Exception as e:
            print(f"Error updating tickers on sector change: {str(e)}")
            traceback.print_exc()

    def clear_ticker_selection(self):
        """Clear ticker selection when no valid sector/database is selected"""
        self.ticker_combo['values'] = ['No tickers available']
        self.ticker_combo.set('No tickers available')
        self.ticker_listbox.delete(0, tk.END)
        self.ticker_listbox.insert(tk.END, 'No tickers available')

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
            if 'sector' in [col.lower() for col in column_names]:
                sectors = self.db_conn.execute(
                    f"SELECT DISTINCT sector FROM {table} ORDER BY sector"
                ).fetchall()
                sectors = [s[0] for s in sectors if s[0] is not None]  # Filter out None values
                self.sector_combo['values'] = sectors
                if sectors:
                    # Use textvariable instead of direct set
                    self.sector_var.set(sectors[0])
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
                sectors = [s[0] for s in sectors if s[0] is not None]  # Filter out None values
                self.sector_combo['values'] = sectors
                
                # Reapply the previous selection if it still exists
                if current_selection in sectors:
                    self.sector_var.set(current_selection)
                elif sectors:
                    self.sector_var.set(sectors[0])
                    print(f"Set initial sector to: {sectors[0]}")
                    self.sector_combo.bind('<<ComboboxSelected>>', self.on_sector_change)
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

def initialize_control_panel(root, databases):
    """Initialize the control panel section of the GUI."""
    try:
        print("\nInitializing control panel...")
        
        # Create a frame for the control panel - now it will be on the left
        control_frame = tk.Frame(root)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        print("Creating control panel...")
        
        # Database selection section
        print("Creating database controls...")
        db_frame = tk.LabelFrame(control_frame, text="Database Selection")
        db_frame.pack(fill=tk.X, padx=5, pady=5)
        
        print("Creating database selection frame...")
        db_var = tk.StringVar(value=databases[0] if databases else "")
        db_dropdown = ttk.Combobox(db_frame, textvariable=db_var, values=databases)
        db_dropdown.pack(padx=5, pady=5, fill=tk.X)
        
        # Table selection section
        print("Creating table controls...")
        table_frame = tk.LabelFrame(control_frame, text="Table Selection")
        table_frame.pack(fill=tk.X, padx=5, pady=5)
        
        print("Creating table selection frame...")
        table_var = tk.StringVar()
        table_dropdown = ttk.Combobox(table_frame, textvariable=table_var)
        table_dropdown.pack(padx=5, pady=5, fill=tk.X)
        
        # Ticker selection section
        print("Creating ticker controls...")
        ticker_frame = tk.LabelFrame(control_frame, text="Ticker Selection")
        ticker_frame.pack(fill=tk.X, padx=5, pady=5)
        
        print("Creating ticker selection frame...")
        ticker_var = tk.StringVar()
        ticker_dropdown = ttk.Combobox(ticker_frame, textvariable=ticker_var)
        ticker_dropdown.pack(padx=5, pady=5, fill=tk.X)
        
        # Duration selection
        print("Creating duration controls...")
        duration_frame = tk.LabelFrame(control_frame, text="Duration")
        duration_frame.pack(fill=tk.X, padx=5, pady=5)
        duration_var = tk.StringVar(value="1 Year")
        duration_dropdown = ttk.Combobox(duration_frame, textvariable=duration_var, 
                                        values=["1 Week", "1 Month", "3 Months", "6 Months", "1 Year", "All"])
        duration_dropdown.pack(padx=5, pady=5, fill=tk.X)
        
        # AI controls section
        print("Adding AI controls...")
        ai_frame = tk.LabelFrame(control_frame, text="AI Controls")
        ai_frame.pack(fill=tk.X, padx=5, pady=5)
        
        print("Adding training parameters...")
        # Add AI model parameters
        epochs_var = tk.IntVar(value=50)
        epochs_label = tk.Label(ai_frame, text="Epochs:")
        epochs_label.pack(anchor=tk.W, padx=5, pady=2)
        epochs_entry = tk.Entry(ai_frame, textvariable=epochs_var, width=10)
        epochs_entry.pack(anchor=tk.W, padx=5, pady=2)
        
        batch_size_var = tk.IntVar(value=32)
        batch_size_label = tk.Label(ai_frame, text="Batch Size:")
        batch_size_label.pack(anchor=tk.W, padx=5, pady=2)
        batch_size_entry = tk.Entry(ai_frame, textvariable=batch_size_var, width=10)
        batch_size_entry.pack(anchor=tk.W, padx=5, pady=2)
        
        # Status label
        status_var = tk.StringVar(value="Ready")
        status_label = tk.Label(control_frame, textvariable=status_var, anchor=tk.W)
        status_label.pack(fill=tk.X, padx=5, pady=10)
        
        # Train and predict buttons
        train_button = tk.Button(ai_frame, text="Train Model", 
                                command=lambda: train_model(db_var.get(), table_var.get(), ticker_var.get(), 
                                                         epochs_var.get(), batch_size_var.get(), status_var))
        train_button.pack(padx=5, pady=5, fill=tk.X)
        
        predict_button = tk.Button(ai_frame, text="Make Prediction", 
                                 command=lambda: make_prediction(db_var.get(), table_var.get(), ticker_var.get(), status_var))
        predict_button.pack(padx=5, pady=5, fill=tk.X)
        
        print("AI controls setup complete")
        
        # Create a refresh button for database list
        refresh_button = tk.Button(db_frame, text="Refresh", 
                                 command=lambda: refresh_databases(db_dropdown, db_var))
        refresh_button.pack(padx=5, pady=5, fill=tk.X)
        
        # Configure dropdown events
        db_dropdown.bind("<<ComboboxSelected>>", lambda event: update_tables(db_var.get(), table_dropdown, table_var))
        table_dropdown.bind("<<ComboboxSelected>>", lambda event: update_tickers(db_var.get(), table_var.get(), ticker_dropdown, ticker_var))
        
        # Store references to UI elements in globals to access later
        global ui_elements
        ui_elements = {
            'db_var': db_var,
            'db_dropdown': db_dropdown,
            'table_var': table_var,
            'table_dropdown': table_dropdown,
            'ticker_var': ticker_var,
            'ticker_dropdown': ticker_dropdown,
            'duration_var': duration_var,
            'duration_dropdown': duration_dropdown,
            'epochs_var': epochs_var,
            'batch_size_var': batch_size_var,
            'status_var': status_var,
            'train_button': train_button,
            'predict_button': predict_button,
            'root': root
        }
        
        # Initial population of tables if a database is selected
        if db_var.get():
            update_tables(db_var.get(), table_dropdown, table_var)
        
        print("Control panel creation complete")
        return control_frame
        
    except Exception as e:
        print(f"Error initializing control panel: {e}")
        traceback.print_exc()
        return None

# Functions for UI events
def refresh_databases(db_dropdown, db_var):
    """Refresh the list of available databases"""
    try:
        print("Refreshing database list...")
        db_files = glob.glob('*.db')
        print(f"Found databases: {db_files}")
        db_dropdown['values'] = db_files
        if db_files and not db_var.get() in db_files:
            db_var.set(db_files[0])
        print("Database list refreshed")
    except Exception as e:
        print(f"Error refreshing database list: {e}")
        traceback.print_exc()

def update_tables(db_name, table_dropdown, table_var):
    """Update the list of tables in the selected database"""
    try:
        if not db_name:
            return
            
        print(f"Connecting to database: {db_name}")
        conn = create_connection(db_name)
        if conn:
            # Get list of tables
            query = "SELECT name FROM sqlite_master WHERE type='table'"
            tables = conn.execute(query).fetchall()
            table_list = [t[0] for t in tables]
            print(f"Retrieved tables: {table_list}")
            
            # Update the dropdown
            table_dropdown['values'] = table_list
            if table_list and (not table_var.get() or table_var.get() not in table_list):
                table_var.set(table_list[0])
                
            print("Table list refreshed")
            conn.close()
            print(f"Found tables: {table_list}")
            return table_list
    except Exception as e:
        print(f"Error updating tables: {e}")
        traceback.print_exc()
    return []

def update_tickers(db_name, table_name, ticker_dropdown, ticker_var):
    """Update the list of tickers in the selected table"""
    try:
        if not db_name or not table_name:
            return
            
        print(f"Getting columns for table: {table_name}")
        conn = create_connection(db_name)
        if conn:
            # Get column info
            query = f"PRAGMA table_info({table_name})"
            columns = conn.execute(query).fetchall()
            column_names = [col[1] for col in columns]
            print(f"Available columns: {column_names}")
            
            ticker_column = None
            sector_column = None
            
            # Find ticker and sector columns
            for col in column_names:
                if col.lower() == 'ticker':
                    ticker_column = col
                    print(f"Found ticker column, retrieving unique tickers...")
                if col.lower() == 'sector':
                    sector_column = col
                    print(f"Found sector column, retrieving unique sectors...")
            
            # Get unique tickers
            if ticker_column:
                query = f"SELECT DISTINCT {ticker_column} FROM {table_name}"
                tickers = conn.execute(query).fetchall()
                ticker_list = [t[0] for t in tickers]
                print(f"Found {len(ticker_list)} tickers")
                
                # Update the dropdown
                ticker_dropdown['values'] = ticker_list
                if ticker_list and (not ticker_var.get() or ticker_var.get() not in ticker_list):
                    ticker_var.set(ticker_list[0])
            
            conn.close()
    except Exception as e:
        print(f"Error updating tickers: {e}")
        traceback.print_exc()

def train_model(db_name, table_name, ticker, epochs, batch_size, status_var):
    """Train the AI model with selected parameters"""
    try:
        print("\n=== Starting Model Training ===")
        print(f"Current Database: {db_name}")
        print(f"Selected Table: {table_name}")
        print(f"Selected Ticker: {ticker}")
        
        status_var.set("Training model...")
        
        # Get the data
        conn = create_connection(db_name)
        if conn:
            # Check if ticker exists in table
            query = f"SELECT COUNT(*) FROM {table_name} WHERE ticker = ?"
            count = conn.execute(query, (ticker,)).fetchone()[0]
            if count == 0:
                error_msg = f"Error: Ticker '{ticker}' not found in table '{table_name}'"
                print(error_msg)
                status_var.set(error_msg)
                return
                
            # Get column info
            query = f"PRAGMA table_info({table_name})"
            columns = conn.execute(query).fetchall()
            column_names = [col[1] for col in columns]
            print(f"Available columns in {table_name}: {column_names}")
            
            # Select required fields
            fields = []
            for field in ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']:
                if field in column_names:
                    fields.append(field)
            
            print(f"Selected Fields: {fields}")
            
            # Execute query to get data
            query = f"SELECT {', '.join(fields)} FROM {table_name} WHERE ticker = ?"
            df = pd.read_sql_query(query, conn, params=(ticker,))
            print(f"Retrieved {len(df)} rows of data")
            
            # Create and train the model
            agent = AIAgent()
            
            # Prepare data
            data_adapter = DataAdapter()
            X_train, X_val, y_train, y_val = data_adapter.prepare_training_data(df)
            
            if X_train is not None:
                # Build model
                input_shape = (X_train.shape[1], X_train.shape[2])  # (sequence_length, features)
                agent.build_model(input_shape)
                
                # Train model
                agent.train(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size)
                
                # Save the trained model and scaler in global variables
                global trained_model, trained_scaler
                trained_model = agent.model
                trained_scaler = data_adapter.scaler
                
                status_var.set("Model training completed successfully")
                print("Model training completed successfully")
            else:
                status_var.set("Failed to prepare training data")
            
            conn.close()
    except Exception as e:
        error_msg = f"Error training model: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        status_var.set(error_msg)

def make_prediction(db_name, table_name, ticker, status_var):
    """Make predictions using the trained model"""
    global trained_model, trained_scaler, ui_elements, ax, canvas
    
    try:
        print("\n=== Starting Prediction ===")
        print(f"Making prediction for {ticker}")
        
        if trained_model is None:
            status_var.set("Error: Model not trained yet")
            return
            
        status_var.set("Making prediction...")
        
        # Get the most recent data for prediction
        conn = create_connection(db_name)
        if conn:
            # Get recent data - adjust limit based on your sequence length
            query = f"""
                        SELECT * FROM {table_name} 
                        WHERE ticker = ? 
                        ORDER BY date DESC 
                        LIMIT 20
                    """
            df = pd.read_sql_query(query, conn, params=(ticker,))
            print(f"Retrieved {len(df)} rows of data")
            
            # Sort by date ascending (oldest to newest)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
            
            # Prepare prediction data
            data_adapter = DataAdapter()
            data_adapter.scaler = trained_scaler  # Use the scaler from training
            
            X_pred = data_adapter.prepare_prediction_data(df)
            
            if X_pred is not None:
                # Make prediction
                predictions = trained_model.predict(X_pred)
                
                # Inverse transform to get actual values
                last_prediction = predictions[-1][0]
                
                # Get actual price for comparison
                current_price = df['close'].iloc[-1]
                
                # Calculate percentage change
                pct_change = (float(last_prediction) - current_price) / current_price * 100
                
                print("\nPrediction Results:")
                print(f"Current Price: ${current_price:.2f}")
                print(f"Predicted Price: ${float(last_prediction):.2f}")
                print(f"Predicted Change: {pct_change:.2f}%")
                
                status_var.set(f"Predicted: ${float(last_prediction):.2f} ({pct_change:+.2f}%)")
                
                # Update plot with prediction
                if 'date' in df.columns:
                    dates = df['date'].values
                    prices = df['close'].values
                    update_plot_with_prediction(dates, prices, float(last_prediction))
            else:
                status_var.set("Failed to prepare prediction data")
            
            conn.close()
    except Exception as e:
        error_msg = f"Error making prediction: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        status_var.set(error_msg)

def update_plot_with_prediction(dates, prices, predicted_price):
    """Update the plot with predicted price."""
    global ax, canvas
    
    try:
        if not dates.any() or not prices.any():
            print("No data to plot")
            return
            
        # Make sure dates is a list of datetime objects, not integers
        if isinstance(dates[0], (int, float, np.integer)):
            print(f"Converting date from numeric: {dates[0]}")
            # Convert timestamps to datetime
            dates = pd.to_datetime(dates)
            
        # Now we can safely add a timedelta
        next_date = dates[-1] + pd.Timedelta(days=1)
        
        # Clear the plot
        ax.clear()
        
        # Plot the historical data
        ax.plot(dates, prices, label='Historical Prices')
        
        # Highlight the prediction
        ax.plot([dates[-1], next_date], [prices[-1], predicted_price], 'r-', linewidth=2, label='Prediction')
        ax.scatter([next_date], [predicted_price], color='red', s=50)
        
        # Add labels and legend
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title(f'Price Prediction')
        ax.legend()
        
        # Format x-axis to show dates nicely
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)
        
        # Redraw the plot
        canvas.draw()
    except Exception as e:
        print(f"Error updating plot: {e}")
        traceback.print_exc()

def initialize_gui(database_files=None):
    """Initialize the GUI application."""
    try:
        print("\nStarting initialize_gui...")
        # Use provided database files or global variable
        databases = database_files or []
        
        # If we still don't have databases, look for them
        if not databases:
            print("No databases provided, searching for database files...")
            db_files = glob.glob('*.db')
            print(f"Found databases: {db_files}")
            databases = db_files
        
        print("Initializing GUI...")
        
        # Create the main application window
        root = tk.Tk()
        root.title("Stock Market Analyzer")
        root.geometry("1200x800")
        print("Setting window dimensions...")
        
        # Create a main container frame to hold controls and plot
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Initialize control panel on the left side
        control_frame = initialize_control_panel(main_frame, databases)
        
        # Create a frame for the plot area on the right side
        plot_frame = tk.Frame(main_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Initialize GUI components as global variables
        global ax, canvas, fig, trained_model, trained_scaler
        trained_model = None
        trained_scaler = None
        
        # Create plot area
        print("\nInitializing plot area...")
        fig, ax = plt.subplots(figsize=(10, 6))
        print("Creating canvas...")
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        print("Adding toolbar...")
        toolbar = NavigationToolbar2Tk(canvas, plot_frame)
        toolbar.update()
        print("Positioning canvas...")
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        print("Plot area initialization complete")
        
        print("Successfully completed initialize_gui")
        
        # Run the application
        print("Running the GUI application...")
        root.mainloop()
        
        return root, fig, ax, canvas
    except Exception as e:
        print(f"Error in initialize_gui: {e}")
        print("Traceback:")
        traceback.print_exc()
        return None

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
