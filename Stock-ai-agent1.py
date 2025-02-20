import os
# Set environment variable to silence Tk deprecation warning
os.environ['TK_SILENCE_DEPRECATION'] = '1'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, GRU
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
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from sklearn.feature_selection import mutual_info_regression
import yfinance as yf
import time
import sqlite3
from sklearn.metrics import mean_squared_error

matplotlib.use('TkAgg')

def find_databases():
    """Find DuckDB database files"""
    print("\nSearching for database files...")
    try:
        all_files = glob.glob('*.db')
        valid_dbs = []
        
        for db_file in all_files:
            try:
                # Test if file is a valid DuckDB database
                conn = duckdb.connect(db_file)
                conn.execute("SHOW TABLES")
                conn.close()
                valid_dbs.append(db_file)
            except Exception as e:
                print(f"Skipping invalid database {db_file}: {str(e)}")
        
        print(f"Found valid databases: {valid_dbs}")
        return valid_dbs
    except Exception as e:
        print(f"Error searching for databases: {str(e)}")
        return []

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

    def is_trained(self):
        return self.model is not None
        
    def train(self, df):
        """Train the model on historical data"""
        try:
            # Implement your training logic here
            self.model = "trained"  # Placeholder
        except Exception as e:
            print(f"Training error: {str(e)}")
            raise
            
    def predict(self, df):
        """Make predictions using the trained model"""
        try:
            if not self.is_trained():
                raise ValueError("Model not trained")
            # Implement your prediction logic here
            return 0.0  # Placeholder
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            raise

class DataAdapter:
    def __init__(self, sequence_length=60, features=None):
        """Initialize the data adapter"""
        self.sequence_length = sequence_length
        self.features = features or ['open', 'high', 'low', 'close', 'volume', 'rsi']
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

    def prepare_duckdb_data(self, conn, table, ticker, start_date=None):
        """Prepare data from DuckDB database"""
        try:
            print(f"\nPreparing DuckDB data for {ticker}")
            
            # Build query based on available features
            feature_cols = ", ".join(self.features)
            query = f"""
                SELECT date, {feature_cols}
                FROM {table}
                WHERE ticker = ?
                {f"AND date >= ?" if start_date else ""}
                ORDER BY date
            """
            
            params = [ticker]
            if start_date:
                params.append(start_date)
            
            # Execute query and convert to DataFrame
            df = conn.execute(query, params).df()
            
            if df.empty:
                print(f"No data found for {ticker}")
                return None
                
            # Convert date to datetime and set as index
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            return self.prepare_training_data(df)
            
        except Exception as e:
            print(f"Error preparing DuckDB data: {str(e)}")
            traceback.print_exc()
            return None, None

class StockAIAgent:
    def __init__(self):
        """Initialize the AI agent with model architecture"""
        try:
            print("\nInitializing AI Agent...")
            self.model = None
            self.data_adapter = DataAdapter(
                sequence_length=self.sequence_length,
                features=[
                    'open', 'high', 'low', 'close', 'volume',
                    'daily_return', 'volatility', 'volume_momentum',
                    'price_momentum_14d', 'price_momentum_30d',
                    'true_range', 'obv', 'cci', 'roc', 'adl', 'cmf',
                    'EMA_5', 'EMA_20', 'EMA_50', 'MFI',
                    'BB_width', 'VWAP', 'Volume_MA_Ratio',
                    'Price_Momentum', 'Volume_Force', 'Trend_Strength'
                ]
            )
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            self.sequence_length = 60  # Number of time steps to look back
            self.features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI']
            print("AI Agent initialized successfully")
            
            # Add DuckDB-specific features
            self.duckdb_features = [
                'open', 'high', 'low', 'close', 'volume',
                'daily_return', 'volatility', 'volume_momentum',
                'price_momentum_14d', 'price_momentum_30d',
                'true_range', 'obv', 'cci', 'roc', 'adl', 'cmf',
                'EMA_5', 'EMA_20', 'EMA_50', 'MFI',
                'BB_width', 'VWAP', 'Volume_MA_Ratio',
                'Price_Momentum', 'Volume_Force', 'Trend_Strength'
            ]
            
            # Enhanced model configuration
            self.model_config = {
                'lstm_units': [100, 50, 50],  # Increased units in LSTM layers
                'dropout_rates': [0.3, 0.3, 0.3],  # Adjusted dropout for regularization
                'dense_units': [32, 16, 1],  # Added dense layers for better feature extraction
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100,
                'patience': 15,  # Early stopping patience
                'validation_split': 0.2
            }
            
            # Feature importance tracking
            self.feature_importance = {}
            
            # Industry classification
            self.industry_groups = {}
            self.sector_performance = {}
            
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

    def prepare_data(self, df, target_column='close', sequence_length=60):
        """Prepare data for training"""
        try:
            print("\nPreparing data for training...")
            
            # Check if required features exist
            required_features = ['open', 'high', 'low', 'close', 'volume']
            if not all(feature in df.columns for feature in required_features):
                print(f"Error: Missing required features: {[f for f in required_features if f not in df.columns]}")
                return None
            
            # Calculate technical indicators
            df = self.calculate_technical_indicators(df)
            if df is None:
                return None
            
            # Create feature list
            feature_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'daily_return', 'volatility', 'volume_momentum',
                'price_momentum_14d', 'price_momentum_30d',
                'true_range', 'obv', 'cci', 'roc', 'adl', 'cmf',
                'EMA_5', 'EMA_20', 'EMA_50', 'MFI', 'BB_width',
                'VWAP', 'Volume_MA_Ratio', 'Price_Momentum',
                'Volume_Force', 'Trend_Strength'
            ]
            
            # Check if all features are present
            missing_features = [f for f in feature_columns if f not in df.columns]
            if missing_features:
                print(f"Error: Missing required features: {missing_features}")
                return None
            
            # Prepare features and target
            features = df[feature_columns].values
            target = df[target_column].values
            
            # Scale the data
            self.feature_scaler = MinMaxScaler()
            self.target_scaler = MinMaxScaler()
            
            scaled_features = self.feature_scaler.fit_transform(features)
            scaled_target = self.target_scaler.fit_transform(target.reshape(-1, 1))
            
            # Create sequences
            X, y = [], []
            for i in range(len(scaled_features) - sequence_length):
                X.append(scaled_features[i:(i + sequence_length)])
                y.append(scaled_target[i + sequence_length])
            
            X = np.array(X)
            y = np.array(y)
            
            # Split into training and validation sets
            split = int(len(X) * 0.8)
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]
            
            print(f"Data preparation completed:")
            print(f"Training set shape: {X_train.shape}")
            print(f"Validation set shape: {X_val.shape}")
            
            return (X_train, y_train), (X_val, y_val)
            
        except Exception as e:
            print(f"Error preparing data: {str(e)}")
            traceback.print_exc()
            return None

    def train_model(self, data, ticker):
        """Train the model with the prepared data"""
        try:
            if data is None:
                print("No data available for training")
                return None
            
            (X_train, y_train), (X_val, y_val) = data
            
            # Build model
            model = Sequential([
                LSTM(units=self.model_config['lstm_units'][0], 
                     return_sequences=True,
                     input_shape=(X_train.shape[1], X_train.shape[2])),
                BatchNormalization(),
                Dropout(self.model_config['dropout_rates'][0]),
                
                LSTM(units=self.model_config['lstm_units'][1], 
                     return_sequences=True),
                BatchNormalization(),
                Dropout(self.model_config['dropout_rates'][1]),
                
                LSTM(units=self.model_config['lstm_units'][2]),
                BatchNormalization(),
                Dropout(self.model_config['dropout_rates'][2]),
                
                Dense(units=self.model_config['dense_units'][0], activation='relu'),
                Dense(units=self.model_config['dense_units'][1], activation='relu'),
                Dense(units=self.model_config['dense_units'][2], activation='linear')
            ])
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=self.model_config['learning_rate']),
                loss='mse',
                metrics=['mae']
            )
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.model_config['patience'],
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                )
            ]
            
            # Train model
            print(f"\nTraining model for {ticker}...")
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.model_config['epochs'],
                batch_size=self.model_config['batch_size'],
                callbacks=callbacks,
                verbose=1
            )
            
            print("Model training completed")
            return model, history
            
        except Exception as e:
            print("\n=== Error in Training ===")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("\nTraceback:")
            traceback.print_exc()
            return None, None

    def process_ticker(self, ticker):
        """Process a single ticker"""
        try:
            print(f"\nProcessing {ticker}...")
            
            # Download data
            stock = yf.Ticker(ticker)
            df = stock.history(period="2y")
            
            if df.empty:
                print(f"No data available for {ticker}")
                return None
            
            # Prepare data
            data = self.prepare_data(df)
            if data is None:
                return None
            
            # Train model
            model, history = self.train_model(data, ticker)
            if model is None:
                return None
            
            return {
                'model': model,
                'history': history,
                'data': data
            }
            
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
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
            
            # Standardize column names to lowercase
            df.columns = [col.lower() for col in df.columns]
            
            # Check if we have the required columns
            required_columns = ['close', 'volume']
            if not all(col in df.columns for col in required_columns):
                print(f"Missing required columns. Available columns: {df.columns.tolist()}")
                return None
                
            # Calculate basic indicators
            df['daily_return'] = df['close'].pct_change()
            df['volatility'] = df['daily_return'].rolling(window=20).std()
            
            # Moving averages
            df['ma5'] = df['close'].rolling(window=5).mean()
            df['ma20'] = df['close'].rolling(window=20).mean()
            df['ma50'] = df['close'].rolling(window=50).mean()
            
            # Clean up any NaN values
            df.fillna(method='ffill', inplace=True)
            df.fillna(0, inplace=True)
            
            print("Technical indicators calculated successfully")
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

    def process_duckdb_data(self, conn, table, ticker):
        """Process data directly from DuckDB"""
        try:
            print(f"\nProcessing DuckDB data for {ticker}")
            
            # Get enhanced market data
            df = self.get_enhanced_market_data(ticker)
            if df is None:
                return None
                
            # Calculate advanced indicators
            df = self.calculate_advanced_indicators(df)
            if df is None:
                return None
                
            # Prepare data for training using data adapter
            return self.data_adapter.prepare_duckdb_data(conn, table, ticker)
            
        except Exception as e:
            print(f"Error processing DuckDB data: {str(e)}")
            traceback.print_exc()
            return None

    def get_duckdb_query_status(self, conn, table):
        """Get status of DuckDB table"""
        try:
            stats = conn.execute(f"""
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(DISTINCT ticker) as unique_tickers,
                    MIN(date) as earliest_date,
                    MAX(date) as latest_date
                FROM {table}
            """).fetchone()
            
            return {
                'total_records': stats[0],
                'unique_tickers': stats[1],
                'date_range': f"{stats[2]} to {stats[3]}"
            }
            
        except Exception as e:
            print(f"Error getting DuckDB status: {str(e)}")
            return None

    def plot_training_metrics(self, history):
        """Plot detailed training metrics"""
        try:
            print("\nPlotting training metrics...")
            
            # Clear previous plots
            self.figure.clear()
            
            # Create subplot grid
            gs = gridspec.GridSpec(2, 2)
            
            # Loss plot
            ax1 = self.figure.add_subplot(gs[0, :])
            ax1.plot(history.history['loss'], label='Training Loss')
            ax1.plot(history.history['val_loss'], label='Validation Loss')
            ax1.set_title('Model Loss During Training')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Learning curve
            ax2 = self.figure.add_subplot(gs[1, 0])
            ax2.plot(history.history['loss'])
            ax2.set_title('Learning Curve')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.grid(True)
            
            # Validation performance
            ax3 = self.figure.add_subplot(gs[1, 1])
            val_loss = history.history['val_loss']
            ax3.plot(val_loss)
            ax3.axhline(y=min(val_loss), color='r', linestyle='--', 
                       label=f'Best: {min(val_loss):.4f}')
            ax3.set_title('Validation Performance')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Validation Loss')
            ax3.legend()
            ax3.grid(True)
            
            # Update display
            self.figure.tight_layout()
            self.canvas.draw()
            print("Training metrics plot updated")
            
        except Exception as e:
            print(f"Error plotting training metrics: {str(e)}")
            traceback.print_exc()

    def plot_predictions_analysis(self, df, predictions, future_predictions=None):
        """Plot comprehensive prediction analysis"""
        try:
            print("\nPlotting prediction analysis...")
            
            # Clear previous plots
            self.figure.clear()
            
            # Create subplot grid
            gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1])
            
            # 1. Main price plot with predictions
            ax_main = self.figure.add_subplot(gs[0, :])
            
            # Plot actual prices
            ax_main.plot(df.index, df['close'], label='Actual', color='blue')
            
            # Plot model predictions
            if len(predictions) > 0:
                prediction_dates = df.index[-len(predictions):]
                ax_main.plot(prediction_dates, predictions, 
                           label='Predicted', color='red', linestyle='--')
            
            # Plot future predictions if available
            if future_predictions is not None and hasattr(self, 'future_dates'):
                ax_main.plot(self.future_dates, future_predictions,
                           label='Future Forecast', color='green', linestyle=':')
            
            ax_main.set_title(f'Stock Price Prediction Analysis - {self.ticker_var.get()}')
            ax_main.legend()
            ax_main.grid(True)
            
            # 2. Prediction Error Analysis
            ax_error = self.figure.add_subplot(gs[1, 0])
            if len(predictions) > 0:
                actual = df['close'].iloc[-len(predictions):]
                error = actual - predictions
                ax_error.plot(prediction_dates, error, color='red')
                ax_error.axhline(y=0, color='black', linestyle='--')
                ax_error.set_title('Prediction Error')
                ax_error.grid(True)
            
            # 3. Error Distribution
            ax_dist = self.figure.add_subplot(gs[1, 1])
            if len(predictions) > 0:
                ax_dist.hist(error, bins=30, color='blue', alpha=0.7)
                ax_dist.set_title('Error Distribution')
                ax_dist.grid(True)
            
            # 4. Technical Indicators
            ax_tech = self.figure.add_subplot(gs[2, :])
            
            # Plot RSI if available
            if 'RSI' in df.columns:
                ax_tech.plot(df.index, df['RSI'], label='RSI', color='purple')
                ax_tech.axhline(y=70, color='r', linestyle='--', alpha=0.5)
                ax_tech.axhline(y=30, color='g', linestyle='--', alpha=0.5)
            
            # Plot MACD if available
            if all(x in df.columns for x in ['MACD', 'Signal_Line']):
                ax_tech.plot(df.index, df['MACD'], label='MACD', color='blue')
                ax_tech.plot(df.index, df['Signal_Line'], label='Signal', color='orange')
            
            ax_tech.set_title('Technical Indicators')
            ax_tech.legend()
            ax_tech.grid(True)
            
            # Update display
            self.figure.tight_layout()
            self.canvas.draw()
            print("Prediction analysis plot updated")
            
        except Exception as e:
            print(f"Error plotting prediction analysis: {str(e)}")
            traceback.print_exc()

    def plot_model_evaluation(self, y_true, y_pred):
        """Plot model evaluation metrics"""
        try:
            print("\nPlotting model evaluation metrics...")
            
            # Clear previous plots
            self.figure.clear()
            
            # Create subplot grid
            gs = gridspec.GridSpec(2, 2)
            
            # 1. Scatter plot of predicted vs actual values
            ax1 = self.figure.add_subplot(gs[0, 0])
            ax1.scatter(y_true, y_pred, alpha=0.5)
            ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                    'r--', lw=2)
            ax1.set_title('Predicted vs Actual Values')
            ax1.set_xlabel('Actual Values')
            ax1.set_ylabel('Predicted Values')
            ax1.grid(True)
            
            # 2. Residual plot
            ax2 = self.figure.add_subplot(gs[0, 1])
            residuals = y_pred - y_true
            ax2.scatter(y_pred, residuals, alpha=0.5)
            ax2.axhline(y=0, color='r', linestyle='--')
            ax2.set_title('Residual Plot')
            ax2.set_xlabel('Predicted Values')
            ax2.set_ylabel('Residuals')
            ax2.grid(True)
            
            # 3. Residual histogram
            ax3 = self.figure.add_subplot(gs[1, 0])
            ax3.hist(residuals, bins=30, alpha=0.7)
            ax3.set_title('Residual Distribution')
            ax3.grid(True)
            
            # 4. Error metrics
            ax4 = self.figure.add_subplot(gs[1, 1])
            ax4.axis('off')
            
            # Calculate metrics
            mse = np.mean((y_true - y_pred) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_true - y_pred))
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            # Display metrics
            metrics_text = (
                f"Model Performance Metrics:\n\n"
                f"MSE: {mse:.4f}\n"
                f"RMSE: {rmse:.4f}\n"
                f"MAE: {mae:.4f}\n"
                f"MAPE: {mape:.2f}%"
            )
            ax4.text(0.1, 0.5, metrics_text, fontsize=10, 
                    verticalalignment='center')
            
            # Update display
            self.figure.tight_layout()
            self.canvas.draw()
            print("Model evaluation plot updated")
            
        except Exception as e:
            print(f"Error plotting model evaluation: {str(e)}")
            traceback.print_exc()

    def build_enhanced_model(self, input_shape):
        """Build enhanced LSTM model architecture"""
        try:
            print("\nBuilding enhanced LSTM model...")
            
            model = Sequential([
                # First LSTM layer with increased units
                LSTM(units=self.model_config['lstm_units'][0],
                     return_sequences=True,
                     input_shape=input_shape),
                BatchNormalization(),
                Dropout(self.model_config['dropout_rates'][0]),
                
                # Second LSTM layer
                LSTM(units=self.model_config['lstm_units'][1],
                     return_sequences=True),
                BatchNormalization(),
                Dropout(self.model_config['dropout_rates'][1]),
                
                # Third LSTM layer
                LSTM(units=self.model_config['lstm_units'][2]),
                BatchNormalization(),
                Dropout(self.model_config['dropout_rates'][2]),
                
                # Dense layers for better feature extraction
                Dense(self.model_config['dense_units'][0], activation='relu'),
                Dense(self.model_config['dense_units'][1], activation='relu'),
                Dense(self.model_config['dense_units'][2], activation='linear')
            ])
            
            # Use Adam optimizer with learning rate scheduling
            optimizer = Adam(learning_rate=self.model_config['learning_rate'])
            model.compile(optimizer=optimizer, loss='huber')  # Huber loss for robustness
            
            print("Enhanced model architecture:")
            model.summary()
            return model
            
        except Exception as e:
            print(f"Error building enhanced model: {str(e)}")
            traceback.print_exc()
            return None

    def feature_selection(self, X, y):
        """Perform feature selection using mutual information"""
        try:
            print("\nPerforming feature selection...")
            
            # Reshape X for feature selection
            X_reshaped = X.reshape(X.shape[0], -1)
            
            # Calculate mutual information scores
            mi_scores = mutual_info_regression(X_reshaped, y)
            
            # Create feature importance dictionary
            feature_names = [f'feature_{i}' for i in range(X_reshaped.shape[1])]
            self.feature_importance = dict(zip(feature_names, mi_scores))
            
            # Sort features by importance
            sorted_features = sorted(self.feature_importance.items(), 
                                  key=lambda x: x[1], reverse=True)
            
            print("\nFeature importance scores:")
            for feature, score in sorted_features[:10]:  # Show top 10
                print(f"{feature}: {score:.4f}")
                
            return sorted_features
            
        except Exception as e:
            print(f"Error in feature selection: {str(e)}")
            traceback.print_exc()
            return None

    def train_with_cross_validation(self, X, y, n_splits=5):
        """Train model with time-series cross validation"""
        try:
            print("\nStarting time-series cross validation...")
            
            # Create time series split
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            # Store metrics for each fold
            fold_metrics = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                print(f"\nTraining fold {fold + 1}/{n_splits}")
                
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Build and train model
                model = self.build_enhanced_model((X_train.shape[1], X_train.shape[2]))
                
                # Callbacks
                callbacks = [
                    EarlyStopping(monitor='val_loss', 
                                patience=self.model_config['patience'],
                                restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_loss', 
                                    factor=0.5,
                                    patience=5,
                                    min_lr=0.00001)
                ]
                
                # Train model
                history = model.fit(
                    X_train, y_train,
                    epochs=self.model_config['epochs'],
                    batch_size=self.model_config['batch_size'],
                    validation_data=(X_val, y_val),
                    callbacks=callbacks,
                    verbose=1
                )
                
                # Evaluate model
                val_loss = model.evaluate(X_val, y_val, verbose=0)
                fold_metrics.append(val_loss)
                
                print(f"Fold {fold + 1} validation loss: {val_loss:.4f}")
            
            # Calculate average performance
            avg_loss = np.mean(fold_metrics)
            std_loss = np.std(fold_metrics)
            
            print(f"\nCross-validation results:")
            print(f"Average loss: {avg_loss:.4f} (±{std_loss:.4f})")
            
            return avg_loss, std_loss
            
        except Exception as e:
            print(f"Error in cross validation: {str(e)}")
            traceback.print_exc()
            return None

    def ensemble_predict(self, X, n_models=5):
        """Make predictions using an ensemble of models"""
        try:
            print("\nMaking ensemble predictions...")
            
            predictions = []
            
            for i in range(n_models):
                print(f"Training model {i+1}/{n_models} for ensemble...")
                
                # Build and train model with slightly different configurations
                model = self.build_enhanced_model((X.shape[1], X.shape[2]))
                model.fit(X, y,
                         epochs=self.model_config['epochs'],
                         batch_size=self.model_config['batch_size'],
                         verbose=0)
                
                # Make predictions
                pred = model.predict(X)
                predictions.append(pred)
            
            # Average predictions
            ensemble_pred = np.mean(predictions, axis=0)
            
            # Calculate prediction uncertainty
            pred_std = np.std(predictions, axis=0)
            
            print("Ensemble predictions complete")
            return ensemble_pred, pred_std
            
        except Exception as e:
            print(f"Error in ensemble prediction: {str(e)}")
            traceback.print_exc()
            return None, None

    def group_tickers_by_industry(self, tickers):
        """Group tickers by industry sector"""
        try:
            print("\nGrouping tickers by industry...")
            
            industry_groups = {}
            failed_tickers = []
            
            for ticker in tickers:
                try:
                    # Get company info using yfinance
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    
                    # Get industry and sector information
                    industry = info.get('industry', 'Unknown')
                    sector = info.get('sector', 'Unknown')
                    
                    # Create industry key
                    industry_key = f"{sector}_{industry}"
                    
                    if industry_key not in industry_groups:
                        industry_groups[industry_key] = {
                            'tickers': [],
                            'sector': sector,
                            'industry': industry,
                            'market_cap_total': 0,
                            'correlation_matrix': None
                        }
                    
                    # Add ticker to group
                    industry_groups[industry_key]['tickers'].append(ticker)
                    industry_groups[industry_key]['market_cap_total'] += info.get('marketCap', 0)
                    
                    time.sleep(0.1)  # Rate limiting
                    
                except Exception as e:
                    print(f"Error processing {ticker}: {str(e)}")
                    failed_tickers.append(ticker)
                    continue
            
            self.industry_groups = industry_groups
            return industry_groups, failed_tickers
            
        except Exception as e:
            print(f"Error grouping tickers: {str(e)}")
            traceback.print_exc()
            return None, None

    def analyze_industry_correlations(self, industry_group):
        """Analyze correlations within industry group"""
        try:
            tickers = industry_group['tickers']
            all_data = []
            
            # Get historical data for all tickers in group
            for ticker in tickers:
                df = self.get_historical_data(ticker)
                if df is not None and not df.empty:
                    all_data.append(df['close'].rename(ticker))
            
            if all_data:
                # Create correlation matrix
                data = pd.concat(all_data, axis=1)
                correlation_matrix = data.corr()
                
                return correlation_matrix
            return None
            
        except Exception as e:
            print(f"Error analyzing correlations: {str(e)}")
            return None

    def process_industry_group(self, industry_key, group_data):
        """Process all tickers in an industry group"""
        try:
            print(f"\nProcessing {industry_key} group with {len(group_data['tickers'])} tickers")
            
            # Analyze industry correlations
            correlation_matrix = self.analyze_industry_correlations(group_data)
            group_data['correlation_matrix'] = correlation_matrix
            
            # Sort tickers by market cap
            tickers = group_data['tickers']
            results = {}
            
            # Process large caps first (they often lead the industry)
            large_caps = tickers[:int(len(tickers) * 0.2)]  # Top 20%
            print(f"Processing {len(large_caps)} large-cap stocks first...")
            large_cap_results = self.process_ticker_batch(large_caps, batch_size=3)
            if large_cap_results:
                results.update(large_cap_results)
            
            # Process remaining tickers
            remaining_tickers = tickers[int(len(tickers) * 0.2):]
            print(f"Processing {len(remaining_tickers)} remaining stocks...")
            remaining_results = self.process_ticker_batch(remaining_tickers, batch_size=5)
            if remaining_results:
                results.update(remaining_results)
            
            # Calculate industry performance metrics
            self.calculate_industry_metrics(industry_key, results)
            
            return results
            
        except Exception as e:
            print(f"Error processing industry group: {str(e)}")
            traceback.print_exc()
            return None

    def calculate_industry_metrics(self, industry_key, results):
        """Calculate industry-wide performance metrics"""
        try:
            metrics = {
                'average_return': [],
                'volatility': [],
                'correlation': [],
                'prediction_accuracy': []
            }
            
            for ticker, ticker_results in results.items():
                if ticker_results:
                    metrics['prediction_accuracy'].append(
                        ticker_results.get('model_performance', {}).get('accuracy', 0)
                    )
                    # Add other metrics as needed
            
            self.sector_performance[industry_key] = {
                'avg_prediction_accuracy': np.mean(metrics['prediction_accuracy']),
                'std_prediction_accuracy': np.std(metrics['prediction_accuracy']),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"Error calculating industry metrics: {str(e)}")

    def process_all_industries(self, tickers):
        """Process all tickers by industry groups"""
        try:
            print("\nStarting industry-based processing...")
            
            # Group tickers by industry
            industry_groups, failed_tickers = self.group_tickers_by_industry(tickers)
            if not industry_groups:
                raise ValueError("Failed to group tickers by industry")
            
            all_results = {}
            
            # Process each industry group
            for industry_key, group_data in industry_groups.items():
                print(f"\nProcessing industry: {industry_key}")
                
                # Process the industry group
                results = self.process_industry_group(industry_key, group_data)
                if results:
                    all_results[industry_key] = results
                
                # Optimize memory between groups
                self.optimize_memory_usage()
            
            # Handle failed tickers
            if failed_tickers:
                print(f"\nProcessing {len(failed_tickers)} failed tickers...")
                failed_results = self.process_ticker_batch(failed_tickers)
                if failed_results:
                    all_results['unclassified'] = failed_results
            
            return all_results
            
        except Exception as e:
            print(f"Error in industry processing: {str(e)}")
            traceback.print_exc()
            return None

class StockAnalyzerGUI:
    def __init__(self, databases):
        """Initialize the GUI application"""
        try:
            print("Starting GUI initialization...")
            
            # Initialize basic attributes
            self.root = tk.Tk()
            self.root.title("Stock Market Analyzer")
            self.root.geometry("1200x800")
            
            # Initialize database attributes
            self.conn = None
            self.current_db = None
            self.databases = databases
            self.valid_databases = []
            
            # Initialize Tkinter variables
            self.database_var = tk.StringVar()
            self.table_var = tk.StringVar()
            self.ticker_var = tk.StringVar()
            self.duration_var = tk.StringVar(value='6mo')
            self.model_var = tk.StringVar(value='lstm')
            self.epochs_var = tk.StringVar(value='100')
            self.batch_var = tk.StringVar(value='32')
            self.seq_var = tk.StringVar(value='60')
            
            # Initialize AI components
            self.model = None
            self.scaler = None
            self.sequence_length = None
            self.current_data = None
            
            # Initialize methods
            self._setup_methods()
            
            # Validate databases
            for db in databases:
                try:
                    conn = duckdb.connect(db)
                    conn.execute("SHOW TABLES")
                    conn.close()
                    self.valid_databases.append(db)
                except Exception as e:
                    print(f"Skipping invalid database {db}: {str(e)}")
            
            if not self.valid_databases:
                raise ValueError("No valid databases found")
            
            # Initialize data structures
            self.available_dbs = self.valid_databases.copy()
            self.available_tables = []
            self.available_tickers = []
            self.tables = []
            
            # Set up the main layout
            self.setup_main_layout()
            
            # Create GUI components
            self.create_plot_area()
            self.create_control_panel()
            
            # Set up initial database connection
            self.setup_database_connection()
            
            print("GUI initialization completed successfully")
            
        except Exception as e:
            print(f"Error during GUI initialization: {str(e)}")
            raise

    def _setup_methods(self):
        """Set up all required methods"""
        # Training methods
        self.train_model = self._train_model
        self.make_prediction = self._make_prediction
        
        # Database methods
        self.connect_to_database = self._connect_to_database
        self.refresh_database_connection = self._refresh_database_connection
        self.execute_query = self._execute_query
        self.fetch_data = self._fetch_data
        
        # GUI update methods
        self.update_ai_status = self._update_ai_status
        self.update_progress = self._update_progress
        self.enable_prediction = self._enable_prediction
        
        # Event handlers
        self.on_database_change = self._on_database_change
        self.on_duration_change = self._on_duration_change
        self.apply_custom_duration = self._apply_custom_duration

    def _train_model(self):
        """Internal implementation of train_model"""
        # ... existing train_model implementation ...
        pass

    def _make_prediction(self):
        """Internal implementation of make_prediction"""
        # ... existing make_prediction implementation ...
        pass

    def _connect_to_database(self, database):
        """Internal implementation of connect_to_database"""
        # ... existing connect_to_database implementation ...
        pass

    def _refresh_database_connection(self):
        """Internal implementation of refresh_database_connection"""
        # ... existing refresh_database_connection implementation ...
        pass

    def _execute_query(self, query, params=None):
        """Internal implementation of execute_query"""
        # ... existing execute_query implementation ...
        pass

    def _fetch_data(self, query, params=None):
        """Internal implementation of fetch_data"""
        # ... existing fetch_data implementation ...
        pass

    def _update_ai_status(self, message, is_error=False):
        """Internal implementation of update_ai_status"""
        # ... existing update_ai_status implementation ...
        pass

    def _update_progress(self, value):
        """Internal implementation of update_progress"""
        # ... existing update_progress implementation ...
        pass

    def _enable_prediction(self, enable=True):
        """Internal implementation of enable_prediction"""
        # ... existing enable_prediction implementation ...
        pass

    def _on_database_change(self, event=None):
        """Internal implementation of on_database_change"""
        # ... existing on_database_change implementation ...
        pass

    def _on_duration_change(self, event=None):
        """Internal implementation of on_duration_change"""
        # ... existing on_duration_change implementation ...
        pass

    def _apply_custom_duration(self):
        """Internal implementation of apply_custom_duration"""
        # ... existing apply_custom_duration implementation ...
        pass

    def setup_main_layout(self):
        """Set up the main window layout"""
        try:
            print("Setting up GUI layout...")
            
            # Create main container
            self.main_frame = ttk.Frame(self.root)
            self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Create left sidebar for controls
            self.sidebar_frame = ttk.Frame(self.main_frame, width=250)
            self.sidebar_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
            self.sidebar_frame.pack_propagate(False)
            
            # Create right side for plots
            self.plot_frame = ttk.Frame(self.main_frame)
            self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            print("GUI layout setup complete")
            
        except Exception as e:
            print(f"Error in GUI layout setup: {str(e)}")
            raise

    def create_control_panel(self):
        """Create the control panel with all widgets"""
        try:
            print("Creating control panel...")
            
            # Create database controls
            self.create_database_controls()
            ttk.Separator(self.sidebar_frame, orient='horizontal').pack(fill='x', pady=10)
            
            # Create table controls
            self.create_table_controls()
            ttk.Separator(self.sidebar_frame, orient='horizontal').pack(fill='x', pady=10)
            
            # Create ticker controls
            self.create_ticker_controls()
            ttk.Separator(self.sidebar_frame, orient='horizontal').pack(fill='x', pady=10)
            
            # Create duration controls
            self.create_duration_controls()
            ttk.Separator(self.sidebar_frame, orient='horizontal').pack(fill='x', pady=10)
            
            # Create AI controls
            self.create_ai_controls()
            
            print("Control panel created successfully")
            
        except Exception as e:
            print(f"Error creating control panel: {str(e)}")
            raise

    def create_database_controls(self):
        """Create database selection controls"""
        try:
            # Create database frame
            db_frame = ttk.LabelFrame(self.sidebar_frame, text="Database Selection")
            db_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Add database list label
            ttk.Label(db_frame, text="Available Databases:").pack(fill=tk.X, padx=5, pady=(5,0))
            
            # Database listbox with scrollbar
            db_list_frame = ttk.Frame(db_frame)
            db_list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            db_scrollbar = ttk.Scrollbar(db_list_frame)
            db_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            self.db_listbox = tk.Listbox(
                db_list_frame,
                height=5,
                yscrollcommand=db_scrollbar.set
            )
            self.db_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            db_scrollbar.config(command=self.db_listbox.yview)
            
            # Populate database listbox
            for db in self.valid_databases:
                self.db_listbox.insert(tk.END, db)
            
            # Database dropdown
            self.db_combo = ttk.Combobox(
                db_frame,
                textvariable=self.database_var,
                values=self.valid_databases,
                state='readonly'
            )
            self.db_combo.pack(fill=tk.X, padx=5, pady=5)
            
            # Bind database selection event
            self.db_combo.bind('<<ComboboxSelected>>', self.on_database_change)
            
            # Database controls
            button_frame = ttk.Frame(db_frame)
            button_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Refresh button
            refresh_btn = ttk.Button(
                button_frame,
                text="Refresh Databases",
                command=self.refresh_databases
            )
            refresh_btn.pack(side=tk.LEFT, padx=(0,5))
            
            # Connect button
            connect_btn = ttk.Button(
                button_frame,
                text="Connect",
                command=self.refresh_database_connection
            )
            connect_btn.pack(side=tk.LEFT)
            
            # Status label
            self.db_status_label = ttk.Label(
                db_frame,
                text="",
                wraplength=200
            )
            self.db_status_label.pack(fill=tk.X, padx=5, pady=5)
            
        except Exception as e:
            print(f"Error creating database controls: {str(e)}")
            raise

    def on_database_change(self, event=None):
        """Handle database selection change"""
        try:
            new_db = self.database_var.get()
            if new_db != self.current_db:
                print(f"\nConnected to database: {new_db}")
                self.current_db = new_db
                self.refresh_database_connection()
                self.update_available_tables()
                self.update_available_tickers()
                self.db_status_label.config(
                    text=f"Connected to: {new_db}",
                    foreground="green"
                )
        except Exception as e:
            print(f"Error in database change handler: {str(e)}")
            self.db_status_label.config(
                text=f"Connection error: {str(e)}",
                foreground="red"
            )

    def refresh_database_connection(self):
        """Refresh the current database connection"""
        try:
            print(f"Connecting to database: {self.current_db}")
            
            # Close existing connection if any
            if self.conn:
                self.conn.close()
            
            # Create new connection
            self.conn = duckdb.connect(self.current_db)
            
            # Update status
            self.db_status_label.config(
                text=f"Connected to: {self.current_db}",
                foreground="green"
            )
            
            # Update available tables
            self.update_available_tables()
            
        except Exception as e:
            print(f"Error connecting to database: {str(e)}")
            self.db_status_label.config(
                text=f"Connection error: {str(e)}",
                foreground="red"
            )
            raise

    def create_table_controls(self):
        """Create table selection controls"""
        try:
            # Create table frame
            table_frame = ttk.LabelFrame(self.sidebar_frame, text="Table Selection")
            table_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Table dropdown
            table_select = ttk.Frame(table_frame)
            table_select.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Label(table_select, text="Select Table:").pack(side=tk.LEFT, padx=(0,5))
            
            self.table_combo = ttk.Combobox(
                table_select,
                textvariable=self.table_var,
                values=self.available_tables,
                state='readonly'
            )
            self.table_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            # Bind table selection event
            self.table_combo.bind('<<ComboboxSelected>>', self.on_table_change)
            
            # Table status label
            self.table_status_label = ttk.Label(
                table_frame,
                text="No table selected",
                wraplength=200
            )
            self.table_status_label.pack(fill=tk.X, padx=5, pady=5)
            
        except Exception as e:
            print(f"Error creating table controls: {str(e)}")
            raise

    def create_ticker_controls(self):
        """Create ticker selection controls"""
        ticker_frame = ttk.LabelFrame(self.sidebar_frame, text="Ticker Selection")
        ticker_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Add ticker list label
        ttk.Label(ticker_frame, text="Available Tickers:").pack(fill=tk.X, padx=5, pady=(5,0))
        
        # Ticker listbox with scrollbar
        ticker_list_frame = ttk.Frame(ticker_frame)
        ticker_list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ticker_scrollbar = ttk.Scrollbar(ticker_list_frame)
        ticker_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.ticker_listbox = tk.Listbox(ticker_list_frame, height=8,
                                       yscrollcommand=ticker_scrollbar.set)
        self.ticker_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ticker_scrollbar.config(command=self.ticker_listbox.yview)
        
        # Ticker dropdown
        self.ticker_combo = ttk.Combobox(
            ticker_frame,
            textvariable=self.ticker_var,
            state='readonly'
        )
        self.ticker_combo.pack(fill=tk.X, padx=5, pady=5)

    def update_listboxes(self):
        """Update both listboxes with current values"""
        try:
            if self.table_listbox and self.ticker_listbox:
                # Update table listbox
                self.table_listbox.delete(0, tk.END)
                for table in self.tables:
                    self.table_listbox.insert(tk.END, table)
                    
                # Update ticker listbox
                self.ticker_listbox.delete(0, tk.END)
                for ticker in self.available_tickers:
                    self.ticker_listbox.insert(tk.END, ticker)
        except Exception as e:
            print(f"Error updating listboxes: {str(e)}")

    def on_table_change(self, event=None):
        """Handle table selection change"""
        try:
            table = self.table_var.get()
            print(f"\nSelected table: {table}")
            
            if not table:
                return
            
            # Validate table schema
            required_columns = ['date', 'ticker', 'close', 'volume']
            if not self.validate_table_schema(table, required_columns):
                raise ValueError("Table missing required columns")
            
            # Update available tickers
            self.update_available_tickers()
            
            # Update status
            if hasattr(self, 'table_status_label'):
                self.table_status_label.config(
                    text=f"Using table: {table}",
                    foreground="green"
                )
            
        except Exception as e:
            print(f"Error in table change handler: {str(e)}")
            if hasattr(self, 'table_status_label'):
                self.table_status_label.config(
                    text=f"Error: {str(e)}",
                    foreground="red"
                )

    def create_plot_area(self):
        """Create the matplotlib figure and canvas"""
        print("Creating plot area...")
        try:
            # Create figure frame
            self.figure_frame = ttk.Frame(self.plot_frame)
            self.figure_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Create figure with larger size and better DPI
            self.figure = Figure(figsize=(12, 8), dpi=100)
            self.canvas = FigureCanvasTkAgg(self.figure, master=self.figure_frame)
            self.canvas.draw()
            
            # Pack the canvas
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add toolbar
            self.toolbar_frame = ttk.Frame(self.plot_frame)
            self.toolbar_frame.pack(fill=tk.X, padx=5)
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
            self.toolbar.update()
            
            # Create subplots
            self.setup_subplots()
            
            print("Plot area created successfully")
            
        except Exception as e:
            print(f"Error creating plot area: {str(e)}")
            raise

    def setup_subplots(self):
        """Set up the initial subplots"""
        try:
            # Clear any existing plots
            self.figure.clear()
            
            # Create main price subplot
            self.ax1 = self.figure.add_subplot(211)  # Price plot (top)
            self.ax1.set_title("Stock Price")
            self.ax1.set_xlabel("Date")
            self.ax1.set_ylabel("Price")
            self.ax1.grid(True)
            
            # Create volume subplot
            self.ax2 = self.figure.add_subplot(212)  # Volume plot (bottom)
            self.ax2.set_title("Volume")
            self.ax2.set_xlabel("Date")
            self.ax2.set_ylabel("Volume")
            self.ax2.grid(True)
            
            # Adjust layout
            self.figure.tight_layout()
            
            # Store current plots
            self.current_price_plot = None
            self.current_volume_plot = None
            
            # Draw the canvas
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error setting up subplots: {str(e)}")
            raise

    def update_plot(self, df=None):
        """Update the plot with new data"""
        try:
            if df is None or df.empty:
                return
                
            # Clear existing plots
            self.ax1.clear()
            self.ax2.clear()
            
            # Plot price data
            self.current_price_plot = self.ax1.plot(
                df.index, 
                df['close'],
                label='Close Price',
                color='blue',
                linewidth=1
            )
            
            # Plot volume data
            self.current_volume_plot = self.ax2.bar(
                df.index,
                df['volume'],
                label='Volume',
                color='gray',
                alpha=0.5
            )
            
            # Set titles and labels
            ticker = self.ticker_var.get()
            self.ax1.set_title(f"{ticker} Stock Price")
            self.ax1.set_xlabel("Date")
            self.ax1.set_ylabel("Price")
            self.ax1.grid(True)
            
            self.ax2.set_title(f"{ticker} Trading Volume")
            self.ax2.set_xlabel("Date")
            self.ax2.set_ylabel("Volume")
            self.ax2.grid(True)
            
            # Format dates on x-axis
            self.figure.autofmt_xdate()
            
            # Adjust layout
            self.figure.tight_layout()
            
            # Redraw canvas
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error updating plot: {str(e)}")
            self.loading_label.config(text=f"Error updating plot: {str(e)}")

    def refresh_databases(self):
        """Refresh the list of available databases"""
        try:
            print("\nRefreshing database list...")
            
            # Get current databases
            current_dbs = glob.glob('*.db')
            
            # Validate databases
            self.valid_databases = []
            for db in current_dbs:
                try:
                    conn = duckdb.connect(db)
                    conn.execute("SHOW TABLES")
                    conn.close()
                    self.valid_databases.append(db)
                except Exception as e:
                    print(f"Skipping invalid database {db}: {str(e)}")
            
            # Update the database list
            self.databases = self.valid_databases.copy()
            
            # Update the combobox values
            self.db_combo['values'] = self.valid_databases
            
            # Update the listbox
            self.db_listbox.delete(0, tk.END)
            for db in self.valid_databases:
                self.db_listbox.insert(tk.END, db)
            
            # If current database is not in list, select first available
            if self.current_db not in self.valid_databases and self.valid_databases:
                self.database_var.set(self.valid_databases[0])
                self.current_db = self.valid_databases[0]
                self.refresh_database_connection()
            
            # Update status
            if self.valid_databases:
                self.db_status_label.config(
                    text=f"Found {len(self.valid_databases)} valid databases",
                    foreground="green"
                )
            else:
                self.db_status_label.config(
                    text="No valid databases found",
                    foreground="red"
                )
            
            print(f"Refreshed database list: {self.valid_databases}")
            
        except Exception as e:
            error_msg = f"Error refreshing databases: {str(e)}"
            print(error_msg)
            self.db_status_label.config(
                text=error_msg,
                foreground="red"
            )
            raise

    def is_valid_database(self, db_file):
        """Check if a file is a valid DuckDB database"""
        try:
            conn = duckdb.connect(db_file)
            conn.execute("SHOW TABLES")
            tables = conn.fetchall()
            conn.close()
            return len(tables) > 0
        except Exception as e:
            print(f"Invalid database {db_file}: {str(e)}")
            return False

    def validate_database_connection(self, db_file):
        """Validate database connection and check for required tables/columns"""
        try:
            # Try to connect
            conn = duckdb.connect(db_file)
            
            # Get tables
            tables = conn.execute("SHOW TABLES").fetchall()
            if not tables:
                conn.close()
                return False, "No tables found in database"
            
            # Check each table for required columns
            for table in tables:
                table_name = table[0]
                columns = conn.execute(f"DESCRIBE {table_name}").fetchall()
                column_names = [col[0].lower() for col in columns]
                
                # Check for minimum required columns
                required_columns = ['date', 'ticker', 'close']
                missing_columns = [col for col in required_columns 
                                 if not any(existing.startswith(col) 
                                          for existing in column_names)]
                
                if not missing_columns:  # Found a valid table
                    conn.close()
                    return True, "Valid database"
            
            conn.close()
            return False, "No tables with required columns found"
            
        except Exception as e:
            return False, f"Database validation error: {str(e)}"

    def cleanup_database_connection(self):
        """Clean up database connection when closing"""
        try:
            if self.conn:
                self.conn.close()
                print("Database connection closed")
        except Exception as e:
            print(f"Error closing database connection: {str(e)}")

    def create_duration_controls(self):
        """Create time duration selection controls"""
        try:
            # Create duration frame
            duration_frame = ttk.LabelFrame(self.sidebar_frame, text="Time Duration")
            duration_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Duration options
            self.duration_options = {
                '1 Month': '1mo',
                '3 Months': '3mo',
                '6 Months': '6mo',
                '1 Year': '1y',
                '2 Years': '2y',
                '5 Years': '5y'
            }
            
            # Create radio buttons for duration selection
            self.duration_var.set('6mo')  # Default to 6 months
            
            # Create duration buttons frame with grid layout
            duration_buttons_frame = ttk.Frame(duration_frame)
            duration_buttons_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Organize buttons in a grid (2 columns)
            row = 0
            col = 0
            for text, value in self.duration_options.items():
                rb = ttk.Radiobutton(
                    duration_buttons_frame,
                    text=text,
                    value=value,
                    variable=self.duration_var,
                    command=self.on_duration_change
                )
                rb.grid(row=row, column=col, padx=5, pady=2, sticky='w')
                col += 1
                if col > 1:  # Move to next row after 2 columns
                    col = 0
                    row += 1
            
            # Custom duration frame
            custom_frame = ttk.Frame(duration_frame)
            custom_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Custom duration entry
            ttk.Label(custom_frame, text="Custom Days:").pack(side=tk.LEFT, padx=(0,5))
            
            self.custom_duration = ttk.Entry(custom_frame, width=10)
            self.custom_duration.pack(side=tk.LEFT, padx=(0,5))
            
            # Apply custom duration button
            apply_btn = ttk.Button(
                custom_frame,
                text="Apply",
                command=self.apply_custom_duration
            )
            apply_btn.pack(side=tk.LEFT)
            
            # Duration status label
            self.duration_status = ttk.Label(
                duration_frame,
                text="Current: 6 Months",
                wraplength=200
            )
            self.duration_status.pack(fill=tk.X, padx=5, pady=5)
            
        except Exception as e:
            print(f"Error creating duration controls: {str(e)}")
            raise

    def on_duration_change(self, event=None):
        """Handle duration selection change"""
        try:
            duration = self.duration_var.get()
            duration_text = next(
                (k for k, v in self.duration_options.items() if v == duration),
                duration
            )
            print(f"Duration changed to: {{{duration_text}}} {duration}")
            
            # Update status label
            self.duration_status.config(
                text=f"Current: {duration_text}",
                foreground="green"
            )
            
            # Update plot if we have data
            if hasattr(self, 'current_data') and self.current_data is not None:
                self.update_plot(self.current_data)
                
        except Exception as e:
            print(f"Error in duration change handler: {str(e)}")
            self.duration_status.config(
                text=f"Error: {str(e)}",
                foreground="red"
            )

    def apply_custom_duration(self):
        """Apply custom duration from entry field"""
        try:
            # Get custom duration
            custom_days = self.custom_duration.get().strip()
            
            if not custom_days:
                raise ValueError("Please enter number of days")
                
            # Validate input
            days = int(custom_days)
            if days <= 0:
                raise ValueError("Days must be positive")
                
            # Set custom duration
            custom_value = f"{days}d"
            self.duration_var.set(custom_value)
            
            # Update status
            self.duration_status.config(
                text=f"Current: {days} Days",
                foreground="green"
            )
            
            print(f"Custom duration set to: {days} days")
            
            # Update plot if we have data
            if hasattr(self, 'current_data') and self.current_data is not None:
                self.update_plot(self.current_data)
                
        except ValueError as ve:
            error_msg = str(ve)
            print(f"Invalid custom duration: {error_msg}")
            self.duration_status.config(
                text=f"Error: {error_msg}",
                foreground="red"
            )
        except Exception as e:
            print(f"Error applying custom duration: {str(e)}")
            self.duration_status.config(
                text=f"Error: {str(e)}",
                foreground="red"
            )

    def get_duration_days(self, duration):
        """Convert duration string to number of days"""
        try:
            # Handle custom duration format (e.g., "30d")
            if duration.endswith('d'):
                return int(duration[:-1])
                
            # Handle standard durations
            duration_days = {
                '1mo': 30,
                '3mo': 90,
                '6mo': 180,
                '1y': 365,
                '2y': 730,
                '5y': 1825
            }
            
            return duration_days.get(duration, 180)  # Default to 6 months
            
        except Exception as e:
            print(f"Error converting duration: {str(e)}")
            return 180  # Default to 6 months on error

    def create_ai_controls(self):
        """Create AI model controls"""
        try:
            # Create AI control frame
            ai_frame = ttk.LabelFrame(self.sidebar_frame, text="AI Controls")
            ai_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Model configuration frame
            config_frame = ttk.Frame(ai_frame)
            config_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Model type selection
            model_frame = ttk.Frame(config_frame)
            model_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(model_frame, text="Model Type:").pack(side=tk.LEFT, padx=(0,5))
            
            self.model_var = tk.StringVar(value="lstm")
            model_combo = ttk.Combobox(
                model_frame,
                textvariable=self.model_var,
                values=["lstm", "gru", "transformer"],
                state="readonly",
                width=15
            )
            model_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            # Training parameters frame
            params_frame = ttk.LabelFrame(ai_frame, text="Training Parameters")
            params_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Epochs
            epoch_frame = ttk.Frame(params_frame)
            epoch_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(epoch_frame, text="Epochs:").pack(side=tk.LEFT, padx=(0,5))
            
            self.epochs_var = tk.StringVar(value="100")
            self.epochs_entry = ttk.Entry(
                epoch_frame,
                textvariable=self.epochs_var,
                width=10
            )
            self.epochs_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            # Batch size
            batch_frame = ttk.Frame(params_frame)
            batch_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(batch_frame, text="Batch Size:").pack(side=tk.LEFT, padx=(0,5))
            
            self.batch_var = tk.StringVar(value="32")
            self.batch_entry = ttk.Entry(
                batch_frame,
                textvariable=self.batch_var,
                width=10
            )
            self.batch_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            # Sequence length
            seq_frame = ttk.Frame(params_frame)
            seq_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(seq_frame, text="Sequence Length:").pack(side=tk.LEFT, padx=(0,5))
            
            self.seq_var = tk.StringVar(value="60")
            self.seq_entry = ttk.Entry(
                seq_frame,
                textvariable=self.seq_var,
                width=10
            )
            self.seq_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            # Training controls
            train_frame = ttk.Frame(ai_frame)
            train_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Train button
            self.train_btn = ttk.Button(
                train_frame,
                text="Train Model",
                command=self.train_model
            )
            self.train_btn.pack(side=tk.LEFT, padx=(0,5))
            
            # Predict button
            self.predict_btn = ttk.Button(
                train_frame,
                text="Make Prediction",
                command=self.make_prediction,
                state="disabled"  # Initially disabled until model is trained
            )
            self.predict_btn.pack(side=tk.LEFT)
            
            # Progress bar
            self.progress_var = tk.DoubleVar()
            self.progress_bar = ttk.Progressbar(
                ai_frame,
                variable=self.progress_var,
                maximum=100
            )
            self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
            
            # Status label
            self.ai_status_label = ttk.Label(
                ai_frame,
                text="Ready",
                wraplength=200
            )
            self.ai_status_label.pack(fill=tk.X, padx=5, pady=5)
            
            print("AI controls created successfully")
            
        except Exception as e:
            print(f"Error creating AI controls: {str(e)}")
            raise

    def update_ai_status(self, message, is_error=False):
        """Update AI status label with message"""
        try:
            self.ai_status_label.config(
                text=message,
                foreground="red" if is_error else "green"
            )
            self.root.update_idletasks()
        except Exception as e:
            print(f"Error updating AI status: {str(e)}")

    def update_progress(self, value):
        """Update progress bar value"""
        try:
            self.progress_var.set(value)
            self.root.update_idletasks()
        except Exception as e:
            print(f"Error updating progress: {str(e)}")

    def get_model_params(self):
        """Get current model parameters"""
        try:
            return {
                'model_type': self.model_var.get(),
                'epochs': int(self.epochs_var.get()),
                'batch_size': int(self.batch_var.get()),
                'sequence_length': int(self.seq_var.get())
            }
        except ValueError as ve:
            raise ValueError(f"Invalid parameter value: {str(ve)}")
        except Exception as e:
            raise Exception(f"Error getting model parameters: {str(e)}")

    def enable_prediction(self, enable=True):
        """Enable or disable prediction button"""
        try:
            self.predict_btn.config(state="normal" if enable else "disabled")
        except Exception as e:
            print(f"Error updating prediction button state: {str(e)}")

    def make_prediction(self):
        """Make predictions using the trained model"""
        try:
            if not self.is_trained():
                raise ValueError("Model not trained yet")
                
            print("\n=== Making Prediction ===")
            self.update_ai_status("Preparing prediction data...")
            
            # Get latest data
            ticker = self.ticker_var.get()
            duration = self.duration_var.get()
            df = self.get_historical_data(ticker, duration)
            
            if df is None or df.empty:
                raise ValueError("No data available for prediction")
            
            # Scale data
            scaled_data = self.scaler.transform(df[['close', 'volume']])
            
            # Create sequence
            X = scaled_data[-self.sequence_length:]
            X = np.array([X])
            
            # Make prediction
            self.update_ai_status("Making prediction...")
            prediction = self.model.predict(X)
            
            # Inverse transform
            last_close = df['close'].iloc[-1]
            predicted_close = self.scaler.inverse_transform(
                np.array([[prediction[0][0], 0]])
            )[0][0]
            
            # Calculate change
            change = ((predicted_close - last_close) / last_close) * 100
            
            # Update status with prediction
            status_text = (
                f"Prediction Complete\n"
                f"Current: ${last_close:.2f}\n"
                f"Predicted: ${predicted_close:.2f}\n"
                f"Change: {change:+.2f}%"
            )
            self.update_ai_status(status_text)
            
            # Update plot with prediction
            self.plot_prediction(df, predicted_close)
            
            print("=== Prediction Complete ===")
            
        except Exception as e:
            print("\n=== Error in Prediction ===")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("\nTraceback:")
            traceback.print_exc()
            
            self.update_ai_status(f"Prediction Error: {str(e)}", is_error=True)

    def plot_prediction(self, df, predicted_value):
        """Plot the prediction on the chart"""
        try:
            # Get the last date and create next date
            last_date = df.index[-1]
            next_date = last_date + pd.Timedelta(days=1)
            
            # Create prediction point
            pred_x = [last_date, next_date]
            pred_y = [df['close'].iloc[-1], predicted_value]
            
            # Plot prediction line
            self.ax1.plot(pred_x, pred_y, 'r--', linewidth=2, label='Prediction')
            self.ax1.plot([next_date], [predicted_value], 'ro', label='Predicted Price')
            
            # Add legend
            self.ax1.legend()
            
            # Redraw canvas
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error plotting prediction: {str(e)}")

    def run(self):
        """Start the GUI application"""
        try:
            print("Starting main loop...")
            
            # Center the window on screen
            self.center_window()
            
            # Configure window close handler
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            
            # Start the main event loop
            self.root.mainloop()
            
        except Exception as e:
            print(f"Error in main loop: {str(e)}")
            traceback.print_exc()
        finally:
            self.cleanup()

    def center_window(self):
        """Center the window on the screen"""
        try:
            # Update window size
            self.root.update_idletasks()
            
            # Get screen dimensions
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            
            # Get window dimensions
            window_width = self.root.winfo_width()
            window_height = self.root.winfo_height()
            
            # Calculate position
            x = (screen_width - window_width) // 2
            y = (screen_height - window_height) // 2
            
            # Set window position
            self.root.geometry(f"+{x}+{y}")
            
        except Exception as e:
            print(f"Error centering window: {str(e)}")

    def on_closing(self):
        """Handle window closing"""
        try:
            print("\nClosing application...")
            
            # Cleanup resources
            self.cleanup()
            
            # Destroy the window
            self.root.destroy()
            
        except Exception as e:
            print(f"Error during closing: {str(e)}")
            self.root.destroy()

    def cleanup(self):
        """Clean up resources"""
        try:
            # Close database connection
            self.cleanup_database_connection()
            
            # Clean up AI resources
            if hasattr(self, 'model') and self.model is not None:
                try:
                    import tensorflow as tf
                    tf.keras.backend.clear_session()
                except Exception as e:
                    print(f"Error clearing Keras session: {str(e)}")
            
            # Clean up matplotlib resources
            if hasattr(self, 'figure') and self.figure is not None:
                try:
                    import matplotlib.pyplot as plt
                    plt.close(self.figure)
                except Exception as e:
                    print(f"Error closing matplotlib figure: {str(e)}")
            
            print("Cleanup completed")
            
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

    def handle_error(self, error, context=""):
        """Handle errors in the GUI"""
        try:
            error_msg = f"Error in {context}: {str(error)}"
            print(error_msg)
            
            # Update status label if available
            if hasattr(self, 'loading_label'):
                self.loading_label.config(
                    text=error_msg,
                    foreground="red"
                )
            
            # Show error in AI status if available
            if hasattr(self, 'ai_status_label'):
                self.update_ai_status(error_msg, is_error=True)
            
            # Reset progress if available
            if hasattr(self, 'progress_var'):
                self.update_progress(0)
            
            # Re-enable controls if needed
            if hasattr(self, 'train_btn'):
                self.train_btn.config(state="normal")
            
            if hasattr(self, 'predict_btn'):
                self.enable_prediction(self.is_trained())
            
        except Exception as e:
            print(f"Error in error handler: {str(e)}")

    def setup_database_connection(self, database=None):
        """Set up the initial database connection"""
        try:
            print("Setting up initial database connection...")
            
            # Use provided database or first available
            if database is None and self.valid_databases:
                database = self.valid_databases[0]
            
            if database:
                print(f"Connecting to database: {database}")
                success = self._connect_to_database(database)
                
                if success:
                    self.current_db = database
                    self.database_var.set(database)
                    
                    # Update available tables
                    self.update_available_tables()
                    
                    print("Initial database setup complete")
                else:
                    raise ValueError(f"Failed to connect to database: {database}")
            else:
                print("No valid database available")
                raise ValueError("No valid database available")
                
        except Exception as e:
            print(f"Error in initial database setup: {str(e)}")
            raise

    def _connect_to_database(self, database):
        """Connect to a specific database"""
        try:
            # Close existing connection if any
            if self.conn:
                self.conn.close()
                self.conn = None
            
            # Create new connection
            self.conn = duckdb.connect(database)
            
            # Test connection
            self.conn.execute("SELECT 1")
            
            print(f"Connected to database: {database}")
            return True
            
        except Exception as e:
            print(f"Error connecting to database {database}: {str(e)}")
            self.conn = None
            self.current_db = None
            return False

    def validate_table_schema(self, table_name, required_columns):
        """Validate that a table has required columns"""
        try:
            print(f"Validating schema for table: {table_name}")
            
            # Get table schema
            query = f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = '{table_name}'
            """
            result = self.conn.execute(query).fetchall()
            
            # Extract column names (case insensitive)
            columns = [col[0].lower() for col in result]
            print(f"Found columns: {columns}")
            
            # Check for required columns
            missing_columns = []
            for required in required_columns:
                if not any(col.startswith(required.lower()) for col in columns):
                    missing_columns.append(required)
            
            if missing_columns:
                print(f"Missing required columns: {missing_columns}")
                return False
            
            print(f"Table {table_name} validation successful")
            return True
            
        except Exception as e:
            print(f"Error validating table {table_name}: {str(e)}")
            return False

    def validate_tables(self):
        """Validate tables have required columns"""
        try:
            valid_tables = []
            required_columns = ['date', 'ticker', 'close', 'volume']
            
            print("\nValidating tables...")
            for table in self.available_tables:
                try:
                    if self.validate_table_schema(table, required_columns):
                        valid_tables.append(table)
                        print(f"Table {table} is valid")
                    else:
                        print(f"Table {table} is invalid (missing required columns)")
                except Exception as e:
                    print(f"Table {table} validation failed: {str(e)}")
            
            self.tables = valid_tables
            
            if not self.tables:
                print("No valid tables found")
                raise ValueError("No valid tables found with required columns")
            
            # Update status
            if hasattr(self, 'table_status_label'):
                self.table_status_label.config(
                    text=f"Found {len(self.tables)} valid tables",
                    foreground="green"
                )
            
            print(f"Valid tables: {self.tables}")
            return True
            
        except Exception as e:
            print(f"Error validating tables: {str(e)}")
            if hasattr(self, 'table_status_label'):
                self.table_status_label.config(
                    text=f"Error: {str(e)}",
                    foreground="red"
                )
            raise

    def get_table_schema(self, table_name):
        """Get schema information for a table"""
        try:
            query = f"""
                SELECT column_name, data_type, is_nullable, 
                       column_default, character_maximum_length
                FROM information_schema.columns 
                WHERE table_name = '{table_name}'
                ORDER BY ordinal_position
            """
            result = self.conn.execute(query).fetchall()
            
            schema = []
            for col in result:
                schema.append({
                    'name': col[0],
                    'type': col[1],
                    'nullable': col[2] == 'YES',
                    'default': col[3],
                    'max_length': col[4]
                })
            
            return schema
            
        except Exception as e:
            print(f"Error getting schema for table {table_name}: {str(e)}")
            raise

    def update_available_tables(self):
        """Update the list of available tables from current database"""
        try:
            print("\nUpdating available tables...")
            
            if not self.conn:
                self._connect_to_database(self.current_db)
                if not self.conn:
                    raise ValueError("No database connection")
            
            # Get list of tables
            query = """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'main'
            """
            result = self.conn.execute(query).fetchall()
            
            # Extract table names
            self.available_tables = [row[0] for row in result]
            print(f"Found tables: {self.available_tables}")
            
            if not self.available_tables:
                raise ValueError("No tables found in database")
            
            # Update table dropdown values
            if hasattr(self, 'table_combo'):
                self.table_combo['values'] = self.available_tables
                
                # If current table not in list, select first available
                if self.table_var.get() not in self.available_tables and self.available_tables:
                    self.table_var.set(self.available_tables[0])
                    self.on_table_change()
            
            # Validate tables
            self.validate_tables()
            
        except Exception as e:
            print(f"Error updating available tables: {str(e)}")
            if hasattr(self, 'table_status_label'):
                self.table_status_label.config(
                    text=f"Error: {str(e)}",
                    foreground="red"
                )
            raise

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
