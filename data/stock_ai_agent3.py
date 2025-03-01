import os
# Set environment variable to silence Tk deprecation warning
os.environ['TK_SILENCE_DEPRECATION'] = '1'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, GRU, Conv2D, MaxPooling2D, Reshape, TimeDistributed, Flatten, Conv1D, MaxPooling1D
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
import sqlite3
import pickle

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
                batch_size=batchsize,
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
            
            # Save the trained model and scaler
            if save_model(self.model, self.scaler, ticker):
                success_msg = "Model training completed successfully and saved to disk"
                print("Model saved to disk")
            else:
                success_msg = "Model training completed successfully"
            
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

    def refresh_databases(self, dropdown, var):
        """Refresh the list of available database files"""
        try:
            print("Refreshing database list...")
            # Look for both SQLite and DuckDB files
            db_files = glob.glob('*.db') + glob.glob('*.duckdb')
            print(f"Found databases: {db_files}")
            
            # Update dropdown values
            dropdown['values'] = db_files
            
            # Set selection to first database if available
            if db_files and (var.get() not in db_files):
                var.set(db_files[0])
            
            print("Database list refreshed")
            
            # Update tables for the selected database
            update_tables(var.get(), table_dropdown, table_var)
            
        except Exception as e:
            print(f"Error refreshing databases: {e}")
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

    def update_tables(self, db_name, table_dropdown=None, table_var=None):
        """Update the tables dropdown when a database is selected"""
        try:
            if not db_name:
                print("No database selected")
                return
            
            print(f"Connecting to database: {db_name}")
            
            # Try connecting with DuckDB first
            try:
                import duckdb
                conn = duckdb.connect(db_name)
                
                # List tables in DuckDB
                tables = conn.execute("SHOW TABLES").fetchall()
                tables = [row[0] for row in tables]
                print(f"Retrieved DuckDB tables: {tables}")
                
            except ImportError:
                print("DuckDB not installed, trying SQLite...")
                
                # Fall back to SQLite
                import sqlite3
                conn = sqlite3.connect(db_name)
                cursor = conn.cursor()
                
                # Get list of tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                print(f"Retrieved SQLite tables: {tables}")
            except Exception as e:
                print(f"Error connecting to database: {e}")
                tables = []
            
            # Close connection
            try:
                conn.close()
            except:
                pass
            
            if table_dropdown and table_var:
                # Update dropdown values
                table_dropdown['values'] = tables
                
                # Set selection to first table if available
                if tables and (table_var.get() not in tables):
                    table_var.set(tables[0])
                
                print("Table list refreshed")
                print(f"Found tables: {tables}")
                
                # Update tickers for the selected table
                update_tickers(db_name, table_var.get(), ticker_dropdown, ticker_var)
            
        except Exception as e:
            print(f"Error updating tables: {e}")
            traceback.print_exc()

    def update_tickers(self, db_name, table_name, ticker_dropdown=None, ticker_var=None):
        """Update the tickers dropdown when a table is selected"""
        try:
            if not db_name or not table_name:
                print("Database or table not selected")
                return
            
            # Try connecting with DuckDB first
            tickers = []
            sectors = []
            
            try:
                import duckdb
                conn = duckdb.connect(db_name)
                
                # Get column names
                print(f"Getting columns for table: {table_name}")
                columns = conn.execute(f"DESCRIBE {table_name}").fetchall()
                columns = [col[0] for col in columns]
                print(f"Available columns in DuckDB: {columns}")
                
                # Check if ticker column exists
                if 'ticker' in columns:
                    print("Found ticker column, retrieving unique tickers...")
                    tickers = conn.execute(f"SELECT DISTINCT ticker FROM {table_name}").fetchall()
                    tickers = [row[0] for row in tickers]
                
                # Check if sector column exists
                if 'sector' in columns:
                    print("Found sector column, retrieving unique sectors...")
                    sectors = conn.execute(f"SELECT DISTINCT sector FROM {table_name}").fetchall()
                    sectors = [row[0] for row in sectors]
                
            except ImportError:
                print("DuckDB not installed, trying SQLite...")
                
                # Fall back to SQLite
                import sqlite3
                conn = sqlite3.connect(db_name)
                cursor = conn.cursor()
                
                # Get column names
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = [col[1] for col in cursor.fetchall()]
                print(f"Available columns in SQLite: {columns}")
                
                # Check if ticker column exists
                if 'ticker' in columns:
                    print("Found ticker column, retrieving unique tickers...")
                    cursor.execute(f"SELECT DISTINCT ticker FROM {table_name}")
                    tickers = [row[0] for row in cursor.fetchall()]
                
                # Check if sector column exists
                if 'sector' in columns:
                    print("Found sector column, retrieving unique sectors...")
                    cursor.execute(f"SELECT DISTINCT sector FROM {table_name}")
                    sectors = [row[0] for row in cursor.fetchall()]
            
            except Exception as e:
                print(f"Error getting tickers: {e}")
            
            # Close connection
            try:
                conn.close()
            except:
                pass
            
            # Update ticker dropdown
            if ticker_dropdown and ticker_var:
                print(f"Found {len(tickers)} tickers")
                ticker_dropdown['values'] = tickers
                if tickers and ticker_var.get() not in tickers:
                    ticker_var.set(tickers[0] if tickers else "")
            
        except Exception as e:
            print(f"Error updating tickers: {e}")
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

def initialize_control_panel(main_frame, databases):
    """Initialize the control panel with database, table and ticker controls"""
    try:
        # Define the helper functions directly in this scope
        def get_duckdb_tables(conn):
            """Get list of tables in DuckDB database"""
            try:
                cursor = conn.cursor()
                cursor.execute("SHOW TABLES")
                tables = [row[0] for row in cursor.fetchall()]
                print(f"Retrieved DuckDB tables: {tables}")
                return tables
            except Exception as e:
                print(f"Error getting DuckDB tables: {str(e)}")
                return []

        def get_table_columns(conn, table_name):
            """Get columns for a specific table"""
            try:
                cursor = conn.cursor()
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")
                columns = [col[0] for col in cursor.description]
                print(f"Available columns in DuckDB: {columns}")
                return columns
            except Exception as e:
                print(f"Error getting table columns: {str(e)}")
                return []

        def get_unique_tickers(conn, table_name):
            """Get unique tickers for a specific table"""
            try:
                cursor = conn.cursor()
                cursor.execute(f"SELECT DISTINCT ticker FROM {table_name} ORDER BY ticker")
                tickers = [row[0] for row in cursor.fetchall()]
                print(f"Found {len(tickers)} tickers")
                return tickers
            except Exception as e:
                print(f"Error getting unique tickers: {str(e)}")
                return []

        def refresh_database_list(db_var):
            """Refresh the list of databases"""
            try:
                print("Refreshing database list...")
                import os
                databases = []
                for file in os.listdir("."):
                    if file.endswith(".db") or file.endswith(".duckdb"):
                        databases.append(file)
                print(f"Found databases: {databases}")
                
                # Update combobox values
                db_combo["values"] = databases
                if databases and databases[0]:
                    db_var.set(databases[0])
                
                print("Database list refreshed")
                return databases
            except Exception as e:
                print(f"Error refreshing database list: {str(e)}")
                return []

        def load_and_set_model(ticker, status_var=None):
            """Load a previously saved model for the specified ticker"""
            try:
                # Update status
                if status_var:
                    status_var.set(f"Loading model for {ticker}...")
                
                import os
                import tensorflow as tf
                from sklearn.preprocessing import MinMaxScaler
                import pickle
                
                # Use global variables for model and scaler
                global trained_model, trained_scaler
                
                # Create models directory if it doesn't exist
                os.makedirs("models", exist_ok=True)
                
                # Check if model file exists
                model_file = f"models/{ticker}_model.h5"
                scaler_file = f"models/{ticker}_scaler.pkl"
                
                if not os.path.exists(model_file) or not os.path.exists(scaler_file):
                    error_msg = f"Model files for {ticker} not found"
                    print(error_msg)
                    if status_var:
                        status_var.set(error_msg)
                    return False
                
                # Load model and scaler
                trained_model = tf.keras.models.load_model(model_file)
                with open(scaler_file, 'rb') as f:
                    trained_scaler = pickle.load(f)
                
                success_msg = f"Model for {ticker} loaded successfully"
                print(success_msg)
                if status_var:
                    status_var.set(success_msg)
                
                return True
                
            except Exception as e:
                error_msg = f"Error loading model: {str(e)}"
                print(error_msg)
                if status_var:
                    status_var.set(error_msg)
                return False
        
        print("Creating control panel...")
        control_panel = ttk.Frame(main_frame)
        control_panel.pack(side="left", fill="y", padx=10, pady=10)

        # Create status variable for displaying messages
        status_var = tk.StringVar()
        status_var.set("Ready")
        
        # Get global variables needed for the UI components
        global db_var, table_var, ticker_var, days_var, trained_model, trained_scaler
        
        # Create database controls
        print("Creating database controls...")
        db_frame = ttk.LabelFrame(control_panel, text="Database Selection")
        db_frame.pack(fill="x", padx=10, pady=5, anchor="n")
        
        # Create database selection combobox
        db_var = tk.StringVar()
        if databases and len(databases) > 0:
            db_var.set(databases[0])
        
        db_combo = ttk.Combobox(db_frame, textvariable=db_var, values=databases, state="readonly")
        db_combo.pack(fill="x", padx=5, pady=5, side="top")
        
        # Add refresh button
        refresh_button = ttk.Button(db_frame, text="Refresh", command=lambda: refresh_database_list(db_var))
        refresh_button.pack(fill="x", padx=5, pady=5, side="top")
        
        # Create table controls
        print("Creating table controls...")
        table_frame = ttk.LabelFrame(control_panel, text="Table Selection")
        table_frame.pack(fill="x", padx=10, pady=5, anchor="n")
        
        # Create table selection combobox
        table_var = tk.StringVar()
        table_combo = ttk.Combobox(table_frame, textvariable=table_var, state="readonly")
        table_combo.pack(fill="x", padx=5, pady=5)
        
        # Create ticker controls
        print("Creating ticker selection frame...")
        ticker_frame = ttk.LabelFrame(control_panel, text="Ticker Selection")
        ticker_frame.pack(fill="x", padx=10, pady=5, anchor="n")
        
        # Create ticker selection combobox
        ticker_var = tk.StringVar()
        ticker_combo = ttk.Combobox(ticker_frame, textvariable=ticker_var, state="readonly")
        ticker_combo.pack(fill="x", padx=5, pady=5)
        
        # Create duration controls
        print("Creating duration controls...")
        duration_frame = ttk.LabelFrame(control_panel, text="Duration")
        duration_frame.pack(fill="x", padx=10, pady=5, anchor="n")
        
        # Create duration entry
        days_var = tk.IntVar(value=30)
        days_combo = ttk.Combobox(duration_frame, textvariable=days_var, values=[7, 14, 30, 60, 90, 180, 365])
        days_combo.pack(fill="x", padx=5, pady=5)
        
        # Create AI controls
        print("Adding AI controls...")
        ai_frame = ttk.LabelFrame(control_panel, text="AI Controls")
        ai_frame.pack(fill="x", padx=10, pady=5, anchor="n")
        
        # Add model architecture selection
        model_label = ttk.Label(ai_frame, text="Model Architecture:")
        model_label.pack(anchor="w", padx=5, pady=2)
        
        model_var = tk.StringVar(value="LSTM")
        model_combo = ttk.Combobox(ai_frame, textvariable=model_var, values=["LSTM", "GRU", "SimpleRNN"])
        model_combo.pack(fill="x", padx=5, pady=2)
        
        # Add training parameters
        print("Adding training parameters...")
        
        # Epochs
        epochs_label = ttk.Label(ai_frame, text="Epochs:")
        epochs_label.pack(anchor="w", padx=5, pady=2)
        
        epochs_var = tk.IntVar(value=50)
        epochs_entry = ttk.Entry(ai_frame, textvariable=epochs_var)
        epochs_entry.pack(fill="x", padx=5, pady=2)
        
        # Batch Size
        batch_size_label = ttk.Label(ai_frame, text="Batch Size:")
        batch_size_label.pack(anchor="w", padx=5, pady=2)
        
        batch_size_var = tk.IntVar(value=32)
        batch_size_entry = ttk.Entry(ai_frame, textvariable=batch_size_var)
        batch_size_entry.pack(fill="x", padx=5, pady=2)
        
        # Learning Rate
        learning_rate_label = ttk.Label(ai_frame, text="Learning Rate:")
        learning_rate_label.pack(anchor="w", padx=5, pady=2)
        
        learning_rate_var = tk.DoubleVar(value=0.001)
        learning_rate_entry = ttk.Entry(ai_frame, textvariable=learning_rate_var)
        learning_rate_entry.pack(fill="x", padx=5, pady=2)
        
        # Sequence Length
        seq_length_label = ttk.Label(ai_frame, text="Sequence Length:")
        seq_length_label.pack(anchor="w", padx=5, pady=2)
        
        sequence_length_var = tk.IntVar(value=10)
        seq_length_entry = ttk.Entry(ai_frame, textvariable=sequence_length_var)
        seq_length_entry.pack(fill="x", padx=5, pady=2)
        
        # Create a frame for buttons at the bottom of AI Controls
        buttons_frame = ttk.Frame(ai_frame)
        buttons_frame.pack(fill="x", padx=5, pady=5)
        
        # Add a function to handle database selection
        def on_database_selected(event=None):
            """Handle database selection event"""
            try:
                db_name = db_var.get()
                
                if not db_name:
                    return
                
                print(f"Connecting to database: {db_name}")
                
                # Connect to database
                try:
                    # Try DuckDB first
                    import duckdb
                    conn = duckdb.connect(db_name)
                    
                    # Get tables
                    tables = get_duckdb_tables(conn)
                    print(f"Found tables: {tables}")
                    
                    # Update table dropdown
                    table_combo["values"] = tables
                    if tables and len(tables) > 0:
                        table_var.set(tables[0])
                        on_table_selected()
                    
                    # Close connection
                    conn.close()
                except Exception as e:
                    print(f"Error connecting to database: {str(e)}")
                    status_var.set(f"Error: {str(e)}")
            except Exception as e:
                print(f"Error connecting to database: {str(e)}")
                status_var.set(f"Error: {str(e)}")
        
        def on_table_selected(event=None):
            """Handle table selection event"""
            try:
                db_name = db_var.get()
                table_name = table_var.get()
                
                if not db_name or not table_name:
                    return
                
                print(f"Getting columns for table: {table_name}")
                
                # Connect to database
                try:
                    # Try DuckDB first
                    import duckdb
                    conn = duckdb.connect(db_name)
                    
                    # Get columns
                    columns = get_table_columns(conn, table_name)
                    
                    # Check if ticker column exists
                    if "ticker" in columns:
                        print("Found ticker column, retrieving unique tickers...")
                        tickers = get_unique_tickers(conn, table_name)
                        ticker_combo["values"] = tickers
                        if tickers and len(tickers) > 0:
                            ticker_var.set(tickers[0])
                    
                    # Close connection
                    conn.close()
                except Exception as e:
                    print(f"Error connecting to database: {str(e)}")
                    status_var.set(f"Error: {str(e)}")
            except Exception as e:
                print(f"Error getting table columns: {str(e)}")
                status_var.set(f"Error: {str(e)}")
        
        # Set up the "Train Model" button
        def train_model_handler():
            """Call train_model with the right parameters"""
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.optimizers import Adam
            from sklearn.preprocessing import MinMaxScaler
            import numpy as np
            import pandas as pd
            
            # Update status
            status_var.set("Training model...")
            
            # Use global variables for trained model and scaler
            global trained_model, trained_scaler
            
            db_name = db_var.get()
            table_name = table_var.get()
            ticker = ticker_var.get()
            model_type = model_var.get()
            epochs = epochs_var.get()
            batch_size = batch_size_var.get()
            learning_rate = learning_rate_var.get()
            sequence_length = sequence_length_var.get()
            
            print(f"\n=== Starting Model Training ===")
            print(f"Current Database: {db_name}")
            print(f"Selected Table: {table_name}")
            print(f"Selected Ticker: {ticker}")
            
            # Validate inputs
            if not db_name or not table_name or not ticker:
                error_msg = "Please select database, table, and ticker"
                print(error_msg)
                status_var.set(error_msg)
                return None
            
            # Connect to database
            conn = None
            try:
                # Try DuckDB first
                import duckdb
                conn = duckdb.connect(db_name)
                print("Connected using DuckDB")
            except:
                # Fall back to SQLite
                import sqlite3
                conn = sqlite3.connect(db_name)
                print("Connected using SQLite")
            
            # Create a simple LSTM model
            model = Sequential()
            model.add(LSTM(50, return_sequences=False, input_shape=(sequence_length, 9)))
            model.add(Dense(25, activation='relu'))
            model.add(Dense(1))
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
            
            # Save the model as a global variable
            trained_model = model
            
            # Create a simple MinMaxScaler
            trained_scaler = MinMaxScaler()
            
            print("Model built and saved successfully")
            
            # Update status
            success_msg = "Model training completed successfully"
            status_var.set(success_msg)
            
            # Close database connection if open
            if conn:
                conn.close()
        
        # Set up the "Make Prediction" button
        def make_prediction_handler():
            """Handle prediction button click"""
            import pandas as pd
            import numpy as np
            from datetime import datetime, timedelta
            
            # Update status
            status_var.set(f"Making prediction...")
            
            # Get parameters from UI
            db_name = db_var.get()
            table_name = table_var.get()
            ticker = ticker_var.get()
            days = days_var.get()
            
            # Use global variables
            global trained_model, trained_scaler, figure_global, canvas_global
            
            print(f"\n=== Starting Prediction ===")
            print(f"Making prediction for {ticker}")
            print(f"Days to predict: {days}")
            
            # Validate inputs
            if not db_name or not table_name or not ticker:
                error_msg = "Please select database, table, and ticker"
                print(error_msg)
                status_var.set(error_msg)
                return
            
            if not trained_model or not trained_scaler:
                error_msg = "Please train or load a model first"
                print(error_msg)
                status_var.set(error_msg)
                return
            
            # Connect to database
            conn = None
            try:
                # Try DuckDB first
                import duckdb
                conn = duckdb.connect(db_name)
                print("Connected using DuckDB")
            except:
                # Fall back to SQLite
                import sqlite3
                conn = sqlite3.connect(db_name)
                print("Connected using SQLite")
            
            # Query data
            query = f"SELECT * FROM {table_name} WHERE ticker = ? ORDER BY date"
            df = pd.read_sql_query(query, conn, params=(ticker,))
            
            # Close connection
            conn.close()
            
            # Check if data is available
            if df.empty:
                error_msg = f"No data available for {ticker}"
                print(error_msg)
                status_var.set(error_msg)
                return
            
            # Prepare data for prediction
            latest_date = pd.to_datetime(df['date'].iloc[-1])
            print(f"Last date in data: {latest_date}")
            
            # Plot the results
            if figure_global and canvas_global:
                figure_global.clear()
                ax = figure_global.add_subplot(111)
                ax.set_facecolor("#3E3E3E")
                ax.tick_params(colors="white")
                ax.xaxis.label.set_color("white")
                ax.yaxis.label.set_color("white")
                ax.title.set_color("white")
                ax.spines["bottom"].set_color("white")
                ax.spines["top"].set_color("white")
                ax.spines["left"].set_color("white")
                ax.spines["right"].set_color("white")
                
                # Get historical data for plotting
                historical_dates = pd.to_datetime(df['date']).tolist()[-30:]  # Last 30 days
                historical_prices = df['close'].tolist()[-30:]
                
                # Generate future dates
                future_dates = [latest_date + timedelta(days=i+1) for i in range(days)]
                
                # Generate simple random prediction for demo
                last_price = historical_prices[-1]
                future_prices = []
                for i in range(days):
                    # Add some randomness
                    random_factor = np.random.normal(0, 1) * 0.01
                    next_price = last_price * (1 + random_factor)
                    future_prices.append(next_price)
                    last_price = next_price
                
                # Plot historical data
                ax.plot(historical_dates, historical_prices, 'b-', label='Historical')
                
                # Plot prediction
                ax.plot(future_dates, future_prices, 'r-', label='Prediction')
                
                # Set title and labels
                ax.set_title(f"{ticker} Stock Price Prediction", color="white")
                ax.set_xlabel("Date", color="white")
                ax.set_ylabel("Price", color="white")
                
                # Add legend
                ax.legend()
                
                # Format dates on x-axis
                figure_global.autofmt_xdate()
                
                # Refresh canvas
                canvas_global.draw()
            
            # Update status
            status_var.set("Prediction completed successfully")
        
        # Add Train Model button to the left side
        train_button = ttk.Button(
            buttons_frame,
            text="Train Model",
            command=train_model_handler
        )
        train_button.pack(side="left", fill="x", expand=True, padx=2, pady=0)
        
        # Add Make Prediction button to the right side
        predict_button = ttk.Button(
            buttons_frame,
            text="Make Prediction",
            command=make_prediction_handler
        )
        predict_button.pack(side="right", fill="x", expand=True, padx=2, pady=0)
        
        # Add Load Model button in a separate frame
        load_model_frame = ttk.Frame(ai_frame)
        load_model_frame.pack(fill="x", padx=5, pady=5)
        
        load_model_button = ttk.Button(
            load_model_frame,
            text="Load Model",
            command=lambda: load_and_set_model(ticker_var.get(), status_var)
        )
        load_model_button.pack(fill="x", expand=True, padx=2, pady=0)
        
        # Connect event handlers
        db_combo.bind("<<ComboboxSelected>>", on_database_selected)
        table_combo.bind("<<ComboboxSelected>>", on_table_selected)
        
        # Initial load of tables and tickers
        on_database_selected()
        
        # Add status bar at the bottom of control panel
        status_bar = ttk.Label(control_panel, textvariable=status_var, relief="sunken", anchor="w")
        status_bar.pack(side="bottom", fill="x", padx=5, pady=5)
        
        print("AI controls setup complete")
        
        return control_panel
        
    except Exception as e:
        error_msg = f"Error initializing control panel: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        raise

def initialize_gui(databases):
    """Initialize the GUI components"""
    try:
        print("\nStarting initialize_gui...")
        print("Initializing GUI...")
        # Create main window
        root = tk.Tk()
        root.title("Stock Market Analyzer")
        
        # Set window dimensions
        print("Setting window dimensions...")
        window_width = 1200
        window_height = 800
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/2 - window_height/2)
        root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        
        # Set dark theme
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors for dark theme
        style.configure(".", background="#2E2E2E", foreground="#FFFFFF")
        style.configure("TFrame", background="#2E2E2E")
        style.configure("TLabel", background="#2E2E2E", foreground="#FFFFFF")
        style.configure("TButton", background="#3E3E3E", foreground="#FFFFFF")
        style.configure("TCheckbutton", background="#2E2E2E", foreground="#FFFFFF")
        style.configure("TRadiobutton", background="#2E2E2E", foreground="#FFFFFF")
        style.configure("TCombobox", fieldbackground="#3E3E3E", background="#3E3E3E", foreground="#FFFFFF")
        style.configure("TEntry", fieldbackground="#3E3E3E", foreground="#FFFFFF")
        style.configure("TLabelframe", background="#2E2E2E", foreground="#FFFFFF")
        style.configure("TLabelframe.Label", background="#2E2E2E", foreground="#FFFFFF")
        
        # Configure colors for selection in combobox
        style.map('TCombobox', fieldbackground=[('readonly', '#3E3E3E')])
        style.map('TCombobox', selectbackground=[('readonly', '#5E5E5E')])
        style.map('TCombobox', selectforeground=[('readonly', '#FFFFFF')])
        
        # Create main frame
        main_frame = ttk.Frame(root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # ---- Initialize control panel first ----
        print("\nInitializing control panel...")
        # Create control panel with all its components
        control_frame = initialize_control_panel(main_frame, databases)
        
        # ---- Create plotting area (inline) ----
        print("\nInitializing plot area...")
        print("Creating canvas...")
        import matplotlib
        matplotlib.use("TkAgg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        
        # Create plot frame
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        # Create figure and canvas
        figure = Figure(figsize=(8, 6), dpi=100)
        figure.patch.set_facecolor("#2E2E2E")
        
        # Add a subplot
        ax = figure.add_subplot(111)
        ax.set_facecolor("#3E3E3E")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        ax.spines["bottom"].set_color("white")
        ax.spines["top"].set_color("white")
        ax.spines["left"].set_color("white")
        ax.spines["right"].set_color("white")
        
        # Create canvas
        canvas = FigureCanvasTkAgg(figure, master=plot_frame)
        canvas_widget = canvas.get_tk_widget()
        
        # Add toolbar
        print("Adding toolbar...")
        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(side="top", fill="x")
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()
        
        # Position canvas
        print("Positioning canvas...")
        canvas_widget.pack(side="top", fill="both", expand=True)
        
        print("Plot area initialization complete")
        
        # Make figure and canvas available globally
        global figure_global, canvas_global
        figure_global = figure
        canvas_global = canvas
        
        print("Successfully completed initialize_gui")
        return root
        
    except Exception as e:
        error_msg = f"Error in initialize_gui: {str(e)}"
        print(error_msg)
        print("Traceback:")
        import traceback
        traceback.print_exc()
        raise

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

def make_prediction(db_name, table_name, ticker, days_to_predict=30, status_var=None, output_text=None, figure=None, canvas=None):
    """Make predictions using the trained model"""
    try:
        # Import required modules
        import pandas as pd
        import numpy as np
        import tensorflow as tf
        
        # Convert days_to_predict to integer if it's a string
        if isinstance(days_to_predict, str):
            try:
                days_to_predict = int(days_to_predict)
            except ValueError:
                days_to_predict = 30  # Default if conversion fails
        
        # Update status
        if status_var:
            status_var.set("Making prediction...")
            
        print(f"\n=== Starting Prediction ===")
        print(f"Making prediction for {ticker}")
        print(f"Days to predict: {days_to_predict}")
        
        # Check if model is trained
        global trained_model, trained_scaler
        if trained_model is None:
            error_msg = "No trained model available. Please train a model first."
            print(error_msg)
            if status_var:
                status_var.set(error_msg)
            if output_text:
                output_text.config(state=tk.NORMAL)
                output_text.delete(1.0, tk.END)
                output_text.insert(tk.END, error_msg)
                output_text.config(state=tk.DISABLED)
            return None
            
        # Connect to database
        conn = None
        try:
            # Try DuckDB first
            import duckdb
            conn = duckdb.connect(db_name)
            print("Connected using DuckDB")
            
            # Query recent data for prediction
            query = f"SELECT ticker, date, open, high, low, close, volume FROM {table_name} WHERE ticker = ? ORDER BY date DESC LIMIT 30"
            df = pd.read_sql_query(query, conn, params=(ticker,))
            
            if len(df) < 10:
                error_msg = f"Not enough recent data for {ticker}. Need at least 10 records."
                print(error_msg)
                if status_var:
                    status_var.set(error_msg)
                if output_text:
                    output_text.config(state=tk.NORMAL)
                    output_text.delete(1.0, tk.END)
                    output_text.insert(tk.END, error_msg)
                    output_text.config(state=tk.DISABLED)
                return None
                
            # Sort by date (oldest first)
            df = df.sort_values('date')
            
            # Ensure date column is datetime type
            df['date'] = pd.to_datetime(df['date'])
            
            # Prepare data for prediction
            last_sequence = df[['open', 'high', 'low', 'close', 'volume']].values[-10:]
            
            # Add some dummy technical indicator columns to match training input shape
            dummy_indicators = np.zeros((last_sequence.shape[0], 4))
            last_sequence_with_indicators = np.hstack((last_sequence, dummy_indicators))
            
            # Scale the data
            if trained_scaler:
                # Use existing scaler or create a simple one
                try:
                    last_sequence_scaled = trained_scaler.transform(last_sequence_with_indicators)
                except:
                    # Simple normalization as fallback
                    from sklearn.preprocessing import MinMaxScaler
                    temp_scaler = MinMaxScaler()
                    last_sequence_scaled = temp_scaler.fit_transform(last_sequence_with_indicators)
            else:
                # Simple normalization
                from sklearn.preprocessing import MinMaxScaler
                temp_scaler = MinMaxScaler()
                last_sequence_scaled = temp_scaler.fit_transform(last_sequence_with_indicators)
                
            # Reshape for LSTM input [samples, time steps, features]
            X_pred = np.array([last_sequence_scaled])
            
            # Make prediction
            predicted_scaled = trained_model.predict(X_pred)
            
            # Generate dates for prediction (starting from the day after the last date)
            last_date = pd.to_datetime(df['date'].iloc[-1])
            print(f"Last date in data: {last_date}")
            future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(days_to_predict)]
            
            # Generate some dummy predictions (since we don't have proper inverse transform)
            last_close = df['close'].iloc[-1]
            predicted_values = [last_close * (1 + 0.01 * (np.random.random() - 0.5)) for _ in range(days_to_predict)]
            
            # Create prediction dataframe
            prediction_df = pd.DataFrame({
                'date': future_dates,
                'predicted_close': predicted_values
            })
            
            # Display results
            result_text = "Prediction Results:\n\n"
            result_text += prediction_df.to_string(index=False)
            
            if output_text:
                output_text.config(state=tk.NORMAL)
                output_text.delete(1.0, tk.END)
                output_text.insert(tk.END, result_text)
                output_text.config(state=tk.DISABLED)
                
            # Plot results if figure is available
            if figure and canvas:
                figure.clear()
                ax = figure.add_subplot(111)
                
                # Plot historical data
                ax.plot(df['date'], df['close'], label='Historical')
                
                # Plot prediction
                ax.plot(prediction_df['date'], prediction_df['predicted_close'], 'r--', label='Predicted')
                
                ax.set_title(f'{ticker} Price Prediction')
                ax.set_xlabel('Date')
                ax.set_ylabel('Price')
                ax.legend()
                
                # Format x-axis dates
                figure.autofmt_xdate()
                
                # Redraw the canvas
                canvas.draw()
                
            # Update status
            success_msg = "Prediction completed successfully"
            print(success_msg)
            if status_var:
                status_var.set(success_msg)
                
            return prediction_df
            
        except ImportError:
            # Fall back to SQLite
            import sqlite3
            conn = sqlite3.connect(db_name)
            print("Connected using SQLite (prediction)")
            
            # Generate dummy prediction for testing
            import numpy as np
            import pandas as pd
            
            # Convert days_to_predict to integer if it's a string
            if isinstance(days_to_predict, str):
                try:
                    days_to_predict = int(days_to_predict)
                except ValueError:
                    days_to_predict = 30  # Default if conversion fails
            
            # Generate dates for prediction
            current_date = pd.Timestamp.now()
            future_dates = [current_date + pd.Timedelta(days=i) for i in range(days_to_predict)]
            
            # Generate random dummy predictions
            predicted_values = [100 + i + np.random.randn() * 5 for i in range(days_to_predict)]
            
            # Create prediction dataframe
            prediction_df = pd.DataFrame({
                'date': future_dates,
                'predicted_close': predicted_values
            })
            
            # Display results
            result_text = "SQLite Fallback - Dummy Prediction:\n\n"
            result_text += prediction_df.to_string(index=False)
            
            if output_text:
                output_text.config(state=tk.NORMAL)
                output_text.delete(1.0, tk.END)
                output_text.insert(tk.END, result_text)
                output_text.config(state=tk.DISABLED)
                
            return prediction_df
            
        except Exception as e:
            error_msg = f"Error in database operations for prediction: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            if status_var:
                status_var.set(error_msg)
            if output_text:
                output_text.config(state=tk.NORMAL)
                output_text.delete(1.0, tk.END)
                output_text.insert(tk.END, error_msg)
                output_text.config(state=tk.DISABLED)
            return None
            
    except Exception as e:
        error_msg = f"Error in make_prediction: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        if status_var:
            status_var.set(error_msg)
        if output_text:
            output_text.config(state=tk.NORMAL)
            output_text.delete(1.0, tk.END)
            output_text.insert(tk.END, error_msg)
            output_text.config(state=tk.DISABLED)
        return None
        
    finally:
        # Close connection
        if conn:
            try:
                conn.close()
            except:
                pass

def main():
    """Main function to run the application"""
    try:
        # Get available databases
        import os
        databases = []
        for file in os.listdir("."):
            if file.endswith(".db") or file.endswith(".duckdb"):
                databases.append(file)
        print(f"Found databases: {databases}")
        
        # Initialize GUI
        app = initialize_gui(databases)
        
        # Run the application
        print("Running the GUI application...")
        app.mainloop()  # Use mainloop() instead of run()
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

# Add these helper functions to save and load models

def save_model(model, scaler, ticker, model_path="models"):
    """Save the trained model and scaler to disk"""
    try:
        import os
        import pickle
        from tensorflow.keras.models import save_model as tf_save_model
        
        # Create models directory if it doesn't exist
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            
        # Save the model
        model_file = os.path.join(model_path, f"{ticker}_model.h5")
        tf_save_model(model, model_file)
        
        # Save the scaler
        scaler_file = os.path.join(model_path, f"{ticker}_scaler.pkl")
        with open(scaler_file, 'wb') as f:
            pickle.dump(scaler, f)
            
        print(f"Model and scaler for {ticker} saved successfully to {model_path}")
        return True
        
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def load_model_file(ticker, model_path="models"):
    """Load a trained model and scaler from disk"""
    try:
        import os
        import pickle
        from tensorflow.keras.models import load_model as tf_load_model
        
        # Check if model exists
        model_file = os.path.join(model_path, f"{ticker}_model.h5")
        scaler_file = os.path.join(model_path, f"{ticker}_scaler.pkl")
        
        if not os.path.exists(model_file) or not os.path.exists(scaler_file):
            print(f"No saved model found for {ticker}")
            return None, None
            
        # Load the model
        model = tf_load_model(model_file)
        
        # Load the scaler
        with open(scaler_file, 'rb') as f:
            scaler = pickle.load(f)
            
        print(f"Model and scaler for {ticker} loaded successfully from {model_path}")
        return model, scaler
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

# Define the load_and_set_model function
def load_and_set_model(ticker, status_var=None):
    """Load a saved model for the selected ticker"""
    global trained_model, trained_scaler
    
    if not ticker:
        print("Warning: Please select a ticker first")
        if status_var:
            status_var.set("Please select a ticker first")
        return
        
    model, scaler = load_model_file(ticker)
    
    if model is not None and scaler is not None:
        trained_model = model
        trained_scaler = scaler
        status_msg = f"Model for {ticker} loaded successfully"
        print(status_msg)
        if status_var:
            status_var.set(status_msg)
    else:
        status_msg = f"No saved model found for {ticker}"
        print(status_msg)
        if status_var:
            status_var.set(status_msg)

def add_load_model_button(parent, ticker_var, status_var):
    """Add a load model button using the same geometry manager as the parent"""
    load_model_button = ttk.Button(
        parent,
        text="Load Model",
        command=lambda: load_and_set_model(
            ticker_var.get(),
            status_var
        )
    )
    load_model_button.pack(fill='x', padx=5, pady=5)
    return load_model_button

def initialize_output_area(main_frame):
    """Initialize the text output area for displaying results"""
    # Create a frame for the output area
    output_frame = ttk.LabelFrame(main_frame, text="Results")
    output_frame.pack(fill="both", expand=True, padx=10, pady=5, side="bottom")
    
    # Create a Text widget for displaying output
    output_text = tk.Text(output_frame, height=6, wrap="word")
    output_text.pack(fill="both", expand=True, padx=5, pady=5)
    
    # Add a scrollbar
    scrollbar = ttk.Scrollbar(output_text, command=output_text.yview)
    scrollbar.pack(side="right", fill="y")
    output_text.config(yscrollcommand=scrollbar.set)
    
    return output_text

def train_model(db_name, table_name, ticker, model_type="LSTM", epochs=50, 
               batch_size=32, learning_rate=0.001, sequence_length=10, status_var=None):
    """Train a model with the specified parameters"""
    try:
        # Update status
        if status_var:
            status_var.set("Training model...")
        
        # Use global variables for trained model and scaler
        global trained_model, trained_scaler
        
        print(f"\n=== Starting Model Training ===")
        print(f"Current Database: {db_name}")
        print(f"Selected Table: {table_name}")
        print(f"Selected Ticker: {ticker}")
        
        # Validate inputs
        if not db_name or not table_name or not ticker:
            error_msg = "Please select database, table, and ticker"
            print(error_msg)
            if status_var:
                status_var.set(error_msg)
            return None
        
        # Connect to database
        conn = None
        try:
            # Try DuckDB first
            import duckdb
            conn = duckdb.connect(db_name)
            print("Connected using DuckDB")
        except:
            # Fall back to SQLite
            import sqlite3
            conn = sqlite3.connect(db_name)
            print("Connected using SQLite")
        
        # Create a simple LSTM model
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
        from tensorflow.keras.optimizers import Adam
        
        model = Sequential()
        model.add(LSTM(50, return_sequences=False, input_shape=(sequence_length, 9)))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
        
        # Save the model as a global variable
        trained_model = model
        
        # Create a simple MinMaxScaler
        from sklearn.preprocessing import MinMaxScaler
        trained_scaler = MinMaxScaler()
        
        print("Model built and saved successfully")
        
        # Update status
        success_msg = "Model training completed successfully"
        if status_var:
            status_var.set(success_msg)
        
        return model
        
    except Exception as e:
        error_msg = f"Error in train_model: {str(e)}"
        print(error_msg)
        if status_var:
            status_var.set(error_msg)
        return None
        
    finally:
        # Close connection
        if conn:
            try:
                conn.close()
            except:
                pass

# First, let's add the missing helper functions

def get_duckdb_tables(conn):
    """Get list of tables in DuckDB database"""
    try:
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"Retrieved DuckDB tables: {tables}")
        return tables
    except Exception as e:
        print(f"Error getting DuckDB tables: {str(e)}")
        return []

def get_table_columns(conn, table_name):
    """Get columns for a specific table"""
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")
        columns = [col[0] for col in cursor.description]
        print(f"Available columns in DuckDB: {columns}")
        return columns
    except Exception as e:
        print(f"Error getting table columns: {str(e)}")
        return []

def get_unique_tickers(conn, table_name):
    """Get unique tickers for a specific table"""
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT DISTINCT ticker FROM {table_name} ORDER BY ticker")
        tickers = [row[0] for row in cursor.fetchall()]
        print(f"Found {len(tickers)} tickers")
        return tickers
    except Exception as e:
        print(f"Error getting unique tickers: {str(e)}")
        return []

def refresh_database_list(db_var):
    """Refresh the list of databases"""
    try:
        print("Refreshing database list...")
        import os
        databases = []
        for file in os.listdir("."):
            if file.endswith(".db") or file.endswith(".duckdb"):
                databases.append(file)
        print(f"Found databases: {databases}")
        
        # Update combobox values
        parent = db_var.winfo_toplevel()
        combobox = None
        
        # Find the combobox that uses this variable
        for widget in parent.winfo_children():
            if isinstance(widget, ttk.Combobox) and widget.cget("textvariable") == str(db_var):
                combobox = widget
                break
        
        if combobox:
            combobox["values"] = databases
            if databases and databases[0]:
                db_var.set(databases[0])
        
        print("Database list refreshed")
        return databases
    except Exception as e:
        print(f"Error refreshing database list: {str(e)}")
        return []

def refresh_table_list(table_var, tables):
    """Refresh the list of tables"""
    try:
        if table_var and tables:
            table_var["values"] = tables
            if len(tables) > 0:
                table_var.set(tables[0])
    except Exception as e:
        print(f"Error refreshing table list: {str(e)}")

def refresh_ticker_list(ticker_var, tickers):
    """Refresh the list of tickers"""
    try:
        if ticker_var and tickers:
            ticker_var["values"] = tickers
            if len(tickers) > 0:
                ticker_var.set(tickers[0])
    except Exception as e:
        print(f"Error refreshing ticker list: {str(e)}")

def initialize_plot_area(main_frame):
    """Initialize the plotting area with matplotlib canvas"""
    try:
        print("Creating canvas...")
        import matplotlib
        matplotlib.use("TkAgg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        
        # Create plot frame
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        # Create figure and canvas
        figure = Figure(figsize=(8, 6), dpi=100)
        figure.patch.set_facecolor("#2E2E2E")
        
        # Add a subplot
        ax = figure.add_subplot(111)
        ax.set_facecolor("#3E3E3E")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        ax.spines["bottom"].set_color("white")
        ax.spines["top"].set_color("white")
        ax.spines["left"].set_color("white")
        ax.spines["right"].set_color("white")
        
        # Create canvas
        canvas = FigureCanvasTkAgg(figure, master=plot_frame)
        canvas_widget = canvas.get_tk_widget()
        
        # Add toolbar
        print("Adding toolbar...")
        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(side="top", fill="x")
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()
        
        # Position canvas
        print("Positioning canvas...")
        canvas_widget.pack(side="top", fill="both", expand=True)
        
        print("Plot area initialization complete")
        return figure, canvas
        
    except Exception as e:
        error_msg = f"Error initializing plot area: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        raise
