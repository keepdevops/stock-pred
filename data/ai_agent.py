import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Input, GRU, 
                                   Bidirectional, Conv1D, MaxPooling1D, 
                                   GlobalAveragePooling1D, LayerNormalization,
                                   MultiHeadAttention)
import duckdb
from datetime import datetime, timedelta
import joblib
import tensorflow as tf
import os
import tkinter as tk
from tkinter import ttk, messagebox

class TickerAIAgent:
    def __init__(self, table_name='stock_metrics', connection=None, model_type=None):
        self.table_name = table_name
        self.model_type = model_type  # Store the model type
        self.model = None
        self.scaler = MinMaxScaler()
        self.sequence_length = 10  # Reduced from 60 to 10 for shorter sequences
        
        # Define available model architectures
        self.model_architectures = {
            'lstm': self._build_lstm_model,
            'gru': self._build_gru_model,
            'simple': self._build_simple_model,
            'deep': self._build_deep_model,
            'bidirectional': self._build_bidirectional_model,
            'transformer': self._build_transformer_model,
            'cnn_lstm': self._build_cnn_lstm_model,
            'attention': self._build_attention_model
        }
        
        try:
            # Use the provided connection or create a new one
            self.conn = connection if connection else duckdb.connect('historical_market_data.db')
            self.owns_connection = connection is None
            print("AI agent using database connection")
            
            # Get actual columns from the table
            self.verify_table_structure()
            
            # Define numeric columns based on actual table structure
            self.numeric_columns = [
                col for col in self.available_columns 
                if col not in ['date', 'ticker', 'symbol', 'pair', 'sector', 'updated_at']
            ]
            print(f"Available numeric columns: {self.numeric_columns}")
            
        except Exception as e:
            print(f"Error initializing AI agent: {e}")
            raise

    def verify_table_structure(self):
        """Verify table exists and has required columns"""
        try:
            # Get table columns
            columns = self.conn.execute(f"SELECT * FROM {self.table_name} LIMIT 0").df().columns
            print(f"Found columns in table: {columns.tolist()}")
            
            # Special handling for forex table first
            if self.table_name == 'historical_forex':
                if 'pair' in columns:
                    self.ticker_column = 'pair'
                    print(f"Using '{self.ticker_column}' as forex pair identifier")
                    self.available_columns = columns
                    return
            
            # Check for ticker identifier column (symbol or ticker)
            ticker_column = None
            possible_ticker_columns = ['symbol', 'ticker']
            
            for col in possible_ticker_columns:
                if col in columns:
                    ticker_column = col
                    break
            
            if not ticker_column:
                raise ValueError(f"Missing required ticker identifier column. Need one of: {possible_ticker_columns}")
            
            # Check for date column
            if 'date' not in columns:
                raise ValueError("Missing required column 'date'")
            
            # Store the column names for later use
            self.ticker_column = ticker_column
            self.available_columns = columns
            
            print(f"Using '{self.ticker_column}' as ticker identifier")
            print(f"Available columns for analysis: {columns.tolist()}")
            
        except Exception as e:
            print(f"Error verifying table structure: {e}")
            raise

    def get_available_columns(self):
        """Get list of available numeric columns for analysis"""
        try:
            # Query the table to get a sample of data
            query = f"""
                SELECT *
                FROM {self.table_name}
                LIMIT 1
            """
            sample = self.conn.execute(query).df()
            
            # Identify numeric columns
            numeric_cols = []
            for col in sample.columns:
                if col not in ['date', self.ticker_column, 'sector', 'updated_at']:
                    try:
                        # Try to convert a sample to numeric
                        query = f"""
                            SELECT {col}
                            FROM {self.table_name}
                            WHERE {col} IS NOT NULL
                                AND CAST({col} AS VARCHAR) != ''
                            LIMIT 1
                        """
                        value = self.conn.execute(query).df()[col].iloc[0]
                        pd.to_numeric(value)
                        numeric_cols.append(col)
                    except:
                        continue
            
            return numeric_cols
        except Exception as e:
            print(f"Error getting available columns: {e}")
            return []

    def prepare_data(self, ticker, column):
        """Prepare data for training/prediction"""
        try:
            # Verify column exists
            if column not in self.available_columns:
                raise ValueError(f"Column '{column}' not found in table. Available columns: {self.available_columns.tolist()}")
            
            # Get data from database using the correct ticker column name
            query = f"""
                SELECT date, {column}
                FROM {self.table_name}
                WHERE {self.ticker_column} = ?
                    AND {column} IS NOT NULL
                    AND CAST({column} AS VARCHAR) != ''
                ORDER BY date ASC
            """
            df = self.conn.execute(query, [ticker]).df()
            
            if df.empty:
                raise ValueError(f"No data found for {ticker} in column {column}")
            
            # Convert column to numeric, handling any non-numeric values
            try:
                df[column] = pd.to_numeric(df[column], errors='coerce')
                df = df.dropna()  # Remove any rows where conversion failed
            except Exception as e:
                raise ValueError(f"Column {column} contains non-numeric values: {str(e)}")
            
            if df.empty:
                raise ValueError(f"No numeric data found for {ticker} in column {column}")
            
            print(f"\nPreparing data for {ticker} using {column}")
            print(f"Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"Number of records: {len(df)}")
            
            if len(df) < self.sequence_length + 1:
                raise ValueError(f"Not enough data points. Need at least {self.sequence_length + 1} points, but got {len(df)}")
            
            # Scale the data
            values = df[column].values.reshape(-1, 1)
            scaled_values = self.scaler.fit_transform(values)
            
            # Create sequences
            X, y = [], []
            for i in range(len(scaled_values) - self.sequence_length):
                X.append(scaled_values[i:(i + self.sequence_length)])
                y.append(scaled_values[i + self.sequence_length])
            
            return np.array(X), np.array(y), df['date'].values
            
        except Exception as e:
            print(f"Data preparation error: {e}")
            raise

    def _build_lstm_model(self, input_shape, params):
        """Build LSTM model"""
        model = Sequential([
            Input(shape=input_shape),
            LSTM(params['units'], return_sequences=True),
            Dropout(params['dropout']),
            LSTM(params['units']),
            Dropout(params['dropout']),
            Dense(25),
            Dense(1)
        ])
        return model

    def _build_gru_model(self, input_shape, params):
        """Build GRU model"""
        model = Sequential([
            Input(shape=input_shape),
            GRU(params['units'], return_sequences=True),
            Dropout(params['dropout']),
            GRU(params['units']),
            Dense(1)
        ])
        return model

    def _build_simple_model(self, input_shape, params):
        """Build simple model"""
        model = Sequential([
            Input(shape=input_shape),
            LSTM(params['units']),
            Dense(1)
        ])
        return model

    def _build_deep_model(self, input_shape, params):
        """Build deep model"""
        model = Sequential([
            Input(shape=input_shape),
            LSTM(params['units'], return_sequences=True),
            Dropout(params['dropout']),
            LSTM(params['units'], return_sequences=True),
            Dropout(params['dropout']),
            LSTM(params['units']),
            Dense(50),
            Dense(25),
            Dense(1)
        ])
        return model

    def _build_bidirectional_model(self, input_shape, params):
        """Build bidirectional model"""
        model = Sequential([
            Input(shape=input_shape),
            Bidirectional(LSTM(params['units'], return_sequences=True)),
            Dropout(params['dropout']),
            Bidirectional(LSTM(params['units'])),
            Dense(1)
        ])
        return model

    def _build_transformer_model(self, input_shape, params):
        """Build transformer model"""
        inputs = Input(shape=input_shape)
        x = inputs
        
        # Add positional encoding
        pos_encoding = tf.keras.layers.Embedding(
            input_dim=input_shape[0],
            output_dim=params['units']
        )(tf.range(start=0, limit=input_shape[0], delta=1))
        x = tf.keras.layers.Add()([x, pos_encoding])
        
        # Transformer layers
        for _ in range(2):
            x = MultiHeadAttention(
                num_heads=4,
                key_dim=params['units']
            )(x, x)
            x = LayerNormalization()(x)
        
        x = GlobalAveragePooling1D()(x)
        x = Dense(params['units'], activation='relu')(x)
        outputs = Dense(1)(x)
        
        return Model(inputs=inputs, outputs=outputs)

    def _build_cnn_lstm_model(self, input_shape, params):
        """Build CNN-LSTM model"""
        model = Sequential([
            Input(shape=input_shape),
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            LSTM(params['units']),
            Dense(1)
        ])
        return model

    def _build_attention_model(self, input_shape, params):
        """Build attention model"""
        inputs = Input(shape=input_shape)
        lstm_out = LSTM(params['units'], return_sequences=True)(inputs)
        attention = MultiHeadAttention(
            num_heads=4,
            key_dim=params['units']
        )(lstm_out, lstm_out)
        x = GlobalAveragePooling1D()(attention)
        outputs = Dense(1)(x)
        return Model(inputs=inputs, outputs=outputs)

    def train(self, ticker, column, model_type='lstm', params=None):
        """Train model for given ticker and column"""
        try:
            # Set default parameters if none provided
            if params is None:
                params = {
                    'units': 50,
                    'dropout': 0.2,
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 100
                }
            
            if model_type not in self.model_architectures:
                raise ValueError(f"Unknown model type. Available: {list(self.model_architectures.keys())}")
            
            print(f"\nTraining {model_type} model for {ticker} using {column}")
            
            # Prepare data
            X, y, _ = self.prepare_data(ticker, column)
            
            if len(X) < self.sequence_length:
                raise ValueError(f"Not enough data points for {ticker} in {column}. Need at least {self.sequence_length} points.")
            
            # Build model
            self.model = self.model_architectures[model_type](
                input_shape=(X.shape[1], 1),
                params=params
            )
            
            # Compile model
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
                loss='mse'
            )
            
            # Create models directory
            os.makedirs('models', exist_ok=True)
            
            # Train model
            history = self.model.fit(
                X, y,
                batch_size=params['batch_size'],
                epochs=params['epochs'],
                validation_split=0.2,
                verbose=1
            )
            
            # Save components
            model_path = f'models/{ticker}_{column}_{model_type}_model.keras'
            scaler_path = f'models/{ticker}_{column}_scaler.joblib'
            params_path = f'models/{ticker}_{column}_params.joblib'
            
            self.model.save(model_path)
            joblib.dump(self.scaler, scaler_path)
            joblib.dump(params, params_path)
            
            print(f"\nModel saved: {model_path}")
            return history
            
        except Exception as e:
            print(f"Training error: {e}")
            raise

    def predict(self, ticker, column='value', model_type='lstm', prediction_days=30):
        """Generate predictions"""
        try:
            model_path = f'models/{ticker}_{column}_{model_type}_model.keras'
            scaler_path = f'models/{ticker}_{column}_scaler.joblib'
            
            if not os.path.exists(model_path):
                raise ValueError(f"Model not found. Please train first.")
            
            self.model = tf.keras.models.load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            
            X, _, dates = self.prepare_data(ticker, column)
            
            last_sequence = X[-1:]
            predictions = []
            current_sequence = last_sequence.copy()
            
            for _ in range(prediction_days):
                pred = self.model.predict(current_sequence, verbose=0)
                predictions.append(pred[0, 0])
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[0, -1] = pred
            
            predictions = self.scaler.inverse_transform(
                np.array(predictions).reshape(-1, 1)
            )
            
            future_dates = pd.date_range(
                start=pd.to_datetime(dates[-1]) + pd.Timedelta(days=1),
                periods=prediction_days,
                freq='D'
            )
            
            return future_dates, predictions.flatten()
            
        except Exception as e:
            print(f"Prediction error: {e}")
            raise

    def cleanup(self):
        """Cleanup resources"""
        try:
            # Only close the connection if we created it
            if self.owns_connection and self.conn:
                self.conn.close()
                print("Database connection closed")
        except Exception as e:
            print(f"Error closing database connection: {e}")

    def setup_fields(self):
        try:
            current_table = self.table_var.get()
            print(f"Setting up fields for table: {current_table}")
            
            fields = self.conn.execute(f"SELECT * FROM {current_table} LIMIT 0").df().columns
            print(f"Fields retrieved: {fields}")
            
            # Filter out non-numeric and special fields
            excluded_fields = ['id', 'symbol', 'industry', 'date', 'updated_at']
            available_fields = [field for field in fields if field not in excluded_fields]
            print(f"Available fields: {available_fields}")
            
            # Clear existing field frame if it exists
            if hasattr(self, 'field_frame'):
                print("Destroying existing field frame")
                self.field_frame.destroy()
            
            # Create new field frame
            self.field_frame = ttk.LabelFrame(self.main_frame, text="Select Fields", padding="5")
            self.field_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
            
            # Create checkboxes for fields
            self.field_vars = {}
            for i, field in enumerate(available_fields):
                display_name = field.replace('_', ' ').title()
                var = tk.BooleanVar(value=field in ['value', 'sector'])
                self.field_vars[field] = var
                cb = ttk.Checkbutton(self.field_frame, text=display_name, variable=var)
                cb.grid(row=i//3, column=i%3, sticky=tk.W, padx=5, pady=2)
                
                # Add tooltip for the field
                field_tooltip = self.get_field_tooltip(field)
                ToolTip(cb, field_tooltip)
            
        except Exception as e:
            print(f"Error setting up fields: {e}")
            messagebox.showerror("Error", f"Failed to setup fields: {e}") 