import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, GRU, Bidirectional, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, Conv1D, MaxPooling1D, Concatenate
import duckdb
from datetime import datetime, timedelta
import joblib
import tensorflow as tf
import os

class TickerAIAgent:
    def __init__(self, table_name='stock_metrics'):
        self.table_name = table_name
        self.model = None
        self.scaler = MinMaxScaler()
        self.sequence_length = 60  # Number of time steps to look back
        
        # Define model architectures
        self.model_architectures = {
            'lstm': self._build_lstm_model,
            'gru': self._build_gru_model,
            'simple': self._build_simple_model,
            'deep': self._build_deep_model,
            'bidirectional': self._build_bidirectional_model,
            'transformer': self._build_transformer_model,
            'cnn_lstm': self._build_cnn_lstm_model,
            'dual_lstm': self._build_dual_lstm_model,
            'attention_lstm': self._build_attention_lstm_model,
            'hybrid': self._build_hybrid_model
        }
        
        # Initialize database connection
        try:
            self.conn = duckdb.connect('historical_market_data.db')
            print("Successfully connected to database")
        except Exception as e:
            print(f"Error connecting to database: {e}")
            raise

    def _build_lstm_model(self, input_shape, params):
        """Standard LSTM model"""
        return Sequential([
            Input(shape=input_shape),
            LSTM(params['units'], return_sequences=True),
            Dropout(params['dropout']),
            LSTM(params['units'], return_sequences=False),
            Dropout(params['dropout']),
            Dense(25),
            Dense(1)
        ])

    def _build_gru_model(self, input_shape, params):
        """GRU-based model"""
        return Sequential([
            Input(shape=input_shape),
            GRU(params['units'], return_sequences=True),
            Dropout(params['dropout']),
            GRU(params['units'], return_sequences=False),
            Dropout(params['dropout']),
            Dense(25),
            Dense(1)
        ])

    def _build_simple_model(self, input_shape, params):
        """Simplified model for faster training"""
        return Sequential([
            Input(shape=input_shape),
            LSTM(params['units'], return_sequences=False),
            Dense(25),
            Dense(1)
        ])

    def _build_deep_model(self, input_shape, params):
        """Deeper network for complex patterns"""
        return Sequential([
            Input(shape=input_shape),
            LSTM(params['units'] * 2, return_sequences=True),
            Dropout(params['dropout']),
            LSTM(params['units'] * 2, return_sequences=True),
            Dropout(params['dropout']),
            LSTM(params['units'] * 2, return_sequences=False),
            Dropout(params['dropout']),
            Dense(50),
            Dense(25),
            Dense(1)
        ])

    def _build_bidirectional_model(self, input_shape, params):
        """Bidirectional LSTM model"""
        return Sequential([
            Input(shape=input_shape),
            Bidirectional(LSTM(params['units'], return_sequences=True)),
            Dropout(params['dropout']),
            Bidirectional(LSTM(params['units'], return_sequences=False)),
            Dropout(params['dropout']),
            Dense(25),
            Dense(1)
        ])

    def _build_transformer_model(self, input_shape, params):
        """Transformer-based model"""
        return Sequential([
            Input(shape=input_shape),
            MultiHeadAttention(num_heads=params['num_heads'], key_dim=params['key_dim']),
            LayerNormalization(epsilon=1e-6),
            Dropout(params['dropout']),
            MultiHeadAttention(num_heads=params['num_heads'], key_dim=params['key_dim']),
            LayerNormalization(epsilon=1e-6),
            Dropout(params['dropout']),
            GlobalAveragePooling1D(),
            Dense(params['units'], activation=params['activation']),
            Dropout(params['dropout']),
            Dense(25, activation=params['activation']),
            Dense(1)
        ])

    def _build_cnn_lstm_model(self, input_shape, params):
        """CNN-LSTM hybrid model"""
        return Sequential([
            Input(shape=input_shape),
            Conv1D(filters=params['filters'], kernel_size=params['kernel_size'], 
                   padding='same', activation=params['activation']),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=params['filters']//2, kernel_size=params['kernel_size'], 
                   padding='same', activation=params['activation']),
            LSTM(params['units'], return_sequences=True),
            Dropout(params['dropout']),
            LSTM(params['units'], return_sequences=False),
            Dense(25),
            Dense(1)
        ])

    def _build_dual_lstm_model(self, input_shape, params):
        """Dual-path LSTM model"""
        input_layer = Input(shape=input_shape)
        lstm1 = LSTM(params['units'], return_sequences=True)(input_layer)
        lstm2 = LSTM(params['units'], return_sequences=False)(lstm1)
        
        lstm3 = LSTM(params['units']//2, return_sequences=False)(input_layer)
        
        merged = Concatenate()([lstm2, lstm3])
        dense1 = Dense(25)(merged)
        output = Dense(1)(dense1)
        
        return Model(inputs=input_layer, outputs=output)

    def _build_attention_lstm_model(self, input_shape, params):
        """LSTM with attention mechanism"""
        return Sequential([
            Input(shape=input_shape),
            LSTM(params['units'], return_sequences=True),
            MultiHeadAttention(num_heads=params['num_heads'], key_dim=params['key_dim']),
            GlobalAveragePooling1D(),
            Dropout(params['dropout']),
            Dense(25),
            Dense(1)
        ])

    def _build_hybrid_model(self, input_shape, params):
        """Hybrid model combining multiple approaches"""
        return Sequential([
            Input(shape=input_shape),
            Conv1D(filters=params['filters']//2, kernel_size=params['kernel_size'], 
                   padding='same', activation=params['activation']),
            Bidirectional(LSTM(params['units'], return_sequences=True)),
            MultiHeadAttention(num_heads=params['num_heads'], key_dim=params['key_dim']),
            LayerNormalization(epsilon=1e-6),
            GlobalAveragePooling1D(),
            Dense(params['units'], activation=params['activation']),
            Dropout(params['dropout']),
            Dense(25),
            Dense(1)
        ])

    def prepare_data(self, ticker, column='value'):
        """Prepare data for training/prediction"""
        try:
            # Get data from database
            query = f"""
                SELECT date, {column}
                FROM {self.table_name}
                WHERE symbol = ?
                ORDER BY date ASC
            """
            df = self.conn.execute(query, [ticker]).df()
            
            if df.empty:
                raise ValueError(f"No data found for ticker {ticker}")
            
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

    def build_model(self, input_shape, params):
        """Build the model architecture"""
        model = Sequential([
            Input(shape=input_shape),
            LSTM(params['units'], return_sequences=True),
            Dropout(params['dropout']),
            LSTM(params['units'], return_sequences=False),
            Dropout(params['dropout']),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
            loss='mse'
        )
        
        return model

    def train(self, ticker, column='value', params=None):
        """Train the model"""
        try:
            if params is None:
                params = {
                    'units': 50,
                    'dropout': 0.2,
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 100
                }
            
            # Prepare data
            X, y, _ = self.prepare_data(ticker, column)
            
            # Build model
            self.model = self.build_model(
                input_shape=(X.shape[1], 1),
                params=params
            )
            
            # Train model
            history = self.model.fit(
                X, y,
                batch_size=params['batch_size'],
                epochs=params['epochs'],
                validation_split=0.2,
                verbose=1
            )
            
            # Save model and scaler
            os.makedirs('models', exist_ok=True)
            self.model.save(f'models/{ticker}_{column}_model.keras')
            joblib.dump(self.scaler, f'models/{ticker}_{column}_scaler.joblib')
            joblib.dump(params, f'models/{ticker}_{column}_params.joblib')
            
            return history
            
        except Exception as e:
            print(f"Training error: {e}")
            raise

    def predict(self, ticker, column='value', prediction_days=30):
        """Generate predictions"""
        try:
            # Load the saved model and scaler
            try:
                model_path = f'models/{ticker}_{column}_model.keras'
                self.model = tf.keras.models.load_model(model_path)
                self.scaler = joblib.load(f'models/{ticker}_{column}_scaler.joblib')
            except Exception as e:
                raise ValueError(f"Failed to load model for {ticker}. Please train the model first.")
            
            # Prepare data
            X, _, dates = self.prepare_data(ticker, column)
            last_sequence = X[-1:]
            predictions = []
            current_sequence = last_sequence.copy()
            
            # Generate predictions
            for _ in range(prediction_days):
                pred = self.model.predict(current_sequence, verbose=0)
                predictions.append(pred[0, 0])
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[0, -1] = pred
            
            # Inverse transform predictions
            predictions = self.scaler.inverse_transform(
                np.array(predictions).reshape(-1, 1)
            )
            
            # Generate future dates
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
            self.conn.close()
            print("Database connection closed")
        except Exception as e:
            print(f"Error closing database connection: {e}") 