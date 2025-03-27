import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, GRU, Conv1D, MaxPooling1D, Flatten, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import os
import tensorflow.keras as keras

class StockAIAgent:
    """AI agent for stock market analysis and prediction."""
    
    def __init__(self, config=None):
        """
        Initialize the Stock AI Agent.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.config = config or {}
        self.current_model = None
        self.model_history = None
        self.scaler = MinMaxScaler()
        self.model_dir = Path(self.config.get('model_dir', 'models'))
        self.model_dir.mkdir(exist_ok=True)
        
        # Define available model architectures
        self.available_models = {
            'lstm': self._build_lstm_model,
            'gru': self._build_gru_model,
            'transformer': self._build_transformer_model,
            'cnn': self._build_cnn_model
        }
        
        # Define column name mappings for different data sources
        self.column_mappings = {
            'default': {
                'open': ['open', 'Open', 'OPEN', 'o', 'O'],
                'high': ['high', 'High', 'HIGH', 'h', 'H'],
                'low': ['low', 'Low', 'LOW', 'l', 'L'],
                'close': ['close', 'Close', 'CLOSE', 'c', 'C'],
                'volume': ['volume', 'Volume', 'VOLUME', 'v', 'V']
            },
            'yahoo': {
                'open': ['Open'],
                'high': ['High'],
                'low': ['Low'],
                'close': ['Close'],
                'volume': ['Volume']
            },
            'alpha_vantage': {
                'open': ['1. open'],
                'high': ['2. high'],
                'low': ['3. low'],
                'close': ['4. close'],
                'volume': ['5. volume']
            }
        }
        
        # Define technical indicators to calculate
        self.technical_indicators = {
            'SMA': ['sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_100', 'sma_200'],
            'EMA': ['ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200'],
            'RSI': ['rsi_6', 'rsi_14', 'rsi_21'],
            'MACD': ['macd', 'macd_signal', 'macd_hist'],
            'BB': ['bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position'],
            'ATR': ['atr_14', 'atr_21'],
            'OBV': ['obv', 'obv_ema'],
            'ADX': ['adx_14', 'adx_21', 'di_plus', 'di_minus'],
            'Stochastic': ['stoch_k', 'stoch_d', 'stoch_rsi_k', 'stoch_rsi_d'],
            'CCI': ['cci_14', 'cci_21'],
            'MFI': ['mfi_14', 'mfi_21'],
            'ROC': ['roc_10', 'roc_21'],
            'MOM': ['mom_10', 'mom_21'],
            'Williams': ['williams_r_14', 'williams_r_21'],
            'Ichimoku': ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b'],
            'Volume': ['volume_sma', 'volume_ema', 'volume_obv', 'volume_ad'],
            'Price': ['price_change', 'price_change_pct', 'price_volatility'],
            'Support/Resistance': ['support_level', 'resistance_level']
        }
        
    def get_available_models(self) -> List[str]:
        """Get list of available model types.
        
        Returns:
            List of model type names
        """
        return list(self.available_models.keys())
        
    def _build_lstm_model(self, input_shape: Tuple[int, int]) -> Model:
        """Build LSTM model architecture.
        
        Args:
            input_shape: Shape of input data (timesteps, features)
            
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            LSTM(64, input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.config.get('learning_rate', 0.001)),
            loss='mse',
            metrics=['mae']
        )
        
        return model
        
    def _build_gru_model(self, input_shape: Tuple[int, int]) -> Model:
        """Build GRU model architecture.
        
        Args:
            input_shape: Shape of input data (timesteps, features)
            
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            GRU(64, input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            GRU(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.config.get('learning_rate', 0.001)),
            loss='mse',
            metrics=['mae']
        )
        
        return model
        
    def _build_transformer_model(self, input_shape: Tuple[int, int]) -> Model:
        """Build Transformer model architecture.
        
        Args:
            input_shape: Shape of input data (timesteps, features)
            
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = Input(shape=input_shape)
        
        # Multi-head attention
        attention = MultiHeadAttention(
            num_heads=4,
            key_dim=32
        )(inputs, inputs)
        
        # Add & Norm
        x = LayerNormalization(epsilon=1e-6)(attention + inputs)
        
        # Feed Forward
        ffn = Dense(128, activation='relu')(x)
        ffn = Dense(input_shape[1])(ffn)
        
        # Add & Norm
        x = LayerNormalization(epsilon=1e-6)(ffn + x)
        
        # Global average pooling
        x = GlobalAveragePooling1D()(x)
        
        # Output layers
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=self.config.get('learning_rate', 0.001)),
            loss='mse',
            metrics=['mae']
        )
        
        return model
        
    def _build_cnn_model(self, input_shape: Tuple[int, int]) -> Model:
        """Build CNN model architecture.
        
        Args:
            input_shape: Shape of input data (timesteps, features)
            
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            Conv1D(64, 3, activation='relu', input_shape=input_shape),
            MaxPooling1D(2),
            Conv1D(32, 3, activation='relu'),
            MaxPooling1D(2),
            Flatten(),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.config.get('learning_rate', 0.001)),
            loss='mse',
            metrics=['mae']
        )
        
        return model
        
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the given data."""
        try:
            self.logger.info("Calculating technical indicators")
            
            # Ensure we have the required OHLCV data
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                raise ValueError("Missing required OHLCV columns for technical indicators")
            
            # Calculate SMAs and EMAs
            for period in [5, 10, 20, 50, 100, 200]:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            
            # Calculate RSI
            for period in [6, 14, 21]:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # Calculate Bollinger Bands
            for period in [20]:
                df[f'bb_middle'] = df['close'].rolling(window=period).mean()
                bb_std = df['close'].rolling(window=period).std()
                df[f'bb_upper'] = df[f'bb_middle'] + (bb_std * 2)
                df[f'bb_lower'] = df[f'bb_middle'] - (bb_std * 2)
                df['bb_width'] = (df[f'bb_upper'] - df[f'bb_lower']) / df[f'bb_middle']
                df['bb_position'] = (df['close'] - df[f'bb_lower']) / (df[f'bb_upper'] - df[f'bb_lower'])
            
            # Calculate ATR
            for period in [14, 21]:
                high_low = df['high'] - df['low']
                high_close = np.abs(df['high'] - df['close'].shift())
                low_close = np.abs(df['low'] - df['close'].shift())
                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = np.max(ranges, axis=1)
                df[f'atr_{period}'] = true_range.rolling(period).mean()
            
            # Calculate OBV
            df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
            df['obv_ema'] = df['obv'].ewm(span=20, adjust=False).mean()
            
            # Calculate Stochastic Oscillator
            for period in [14]:
                low_14 = df['low'].rolling(window=period).min()
                high_14 = df['high'].rolling(window=period).max()
                df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
                df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
            
            # Calculate Stochastic RSI
            for period in [14]:
                stoch_rsi = df[f'rsi_{period}'].rolling(window=period).apply(
                    lambda x: (x - x.min()) / (x.max() - x.min()) * 100
                )
                df['stoch_rsi_k'] = stoch_rsi
                df['stoch_rsi_d'] = stoch_rsi.rolling(window=3).mean()
            
            # Calculate CCI
            for period in [14, 21]:
                tp = (df['high'] + df['low'] + df['close']) / 3
                tp_ma = tp.rolling(window=period).mean()
                tp_std = tp.rolling(window=period).std()
                df[f'cci_{period}'] = (tp - tp_ma) / (0.015 * tp_std)
            
            # Calculate Money Flow Index
            for period in [14, 21]:
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                money_flow = typical_price * df['volume']
                
                positive_flow = pd.Series(0, index=df.index)
                negative_flow = pd.Series(0, index=df.index)
                
                positive_flow[typical_price > typical_price.shift(1)] = money_flow[typical_price > typical_price.shift(1)]
                negative_flow[typical_price < typical_price.shift(1)] = money_flow[typical_price < typical_price.shift(1)]
                
                positive_mf = positive_flow.rolling(window=period).sum()
                negative_mf = negative_flow.rolling(window=period).sum()
                
                df[f'mfi_{period}'] = 100 - (100 / (1 + positive_mf / negative_mf))
            
            # Calculate Rate of Change
            for period in [10, 21]:
                df[f'roc_{period}'] = df['close'].pct_change(periods=period) * 100
            
            # Calculate Momentum
            for period in [10, 21]:
                df[f'mom_{period}'] = df['close'].diff(periods=period)
            
            # Calculate Williams %R
            for period in [14, 21]:
                highest_high = df['high'].rolling(window=period).max()
                lowest_low = df['low'].rolling(window=period).min()
                df[f'williams_r_{period}'] = ((highest_high - df['close']) / (highest_high - lowest_low)) * -100
            
            # Calculate Ichimoku Cloud
            df['tenkan_sen'] = (df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2
            df['kijun_sen'] = (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2
            df['senkou_span_a'] = (df['tenkan_sen'] + df['kijun_sen']) / 2
            df['senkou_span_b'] = (df['high'].rolling(window=52).max() + df['low'].rolling(window=52).min()) / 2
            
            # Calculate Volume Indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ema'] = df['volume'].ewm(span=20, adjust=False).mean()
            
            # Calculate Price-based Indicators
            df['price_change'] = df['close'].diff()
            df['price_change_pct'] = df['close'].pct_change() * 100
            df['price_volatility'] = df['price_change'].rolling(window=20).std()
            
            # Calculate Support/Resistance Levels (simplified)
            df['support_level'] = df['low'].rolling(window=20).min()
            df['resistance_level'] = df['high'].rolling(window=20).max()
            
            # Fill NaN values with forward fill then backward fill
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            self.logger.info("Technical indicators calculated successfully")
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {str(e)}")
            raise
        
    def _map_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map column names to standard format based on available columns."""
        self.logger.info("Mapping column names to standard format")
        
        # Get all column names from the dataframe
        available_columns = df.columns.tolist()
        
        # Initialize mapping dictionary
        column_map = {}
        
        # Try to find matching columns for each required field
        for standard_name, possible_names in self.column_mappings['default'].items():
            for col in available_columns:
                if col in possible_names:
                    column_map[col] = standard_name
                    break
        
        # Rename columns if mapping found
        if column_map:
            df = df.rename(columns=column_map)
            self.logger.info(f"Column mapping applied: {column_map}")
        else:
            self.logger.warning("No column mapping found. Using original column names.")
            
        return df
        
    def prepare_data_for_training(self, df):
        """
        Prepare data for model training.
        
        Args:
            df (pd.DataFrame): Input DataFrame with stock data
            
        Returns:
            tuple: (X, y) arrays for training
        """
        try:
            logging.info("Preparing data for model training...")
            
            # Get features and target from config
            features = self.config.get('features', ['open', 'high', 'low', 'close', 'volume'])
            target = self.config.get('target', 'close')
            sequence_length = self.config.get('sequence_length', 60)
            
            # Check if all required features are present
            missing_features = [f for f in features if f not in df.columns]
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
                
            # Create sequences for LSTM
            feature_data = df[features].values
            target_data = df[target].values
            
            X, y = [], []
            for i in range(len(df) - sequence_length):
                X.append(feature_data[i:(i + sequence_length)])
                y.append(target_data[i + sequence_length])
                
            X = np.array(X)
            y = np.array(y)
            
            # Normalize data
            X_reshaped = X.reshape((X.shape[0] * X.shape[1], X.shape[2]))
            X_scaled = self.scaler.fit_transform(X_reshaped)
            X = X_scaled.reshape(X.shape)
            
            logging.info(f"Data prepared: X shape {X.shape}, y shape {y.shape}")
            return X, y
            
        except Exception as e:
            logging.error(f"Error preparing data: {str(e)}")
            raise
            
    def build_model(self, input_shape):
        """
        Build and return the LSTM model.
        
        Args:
            input_shape (tuple): Shape of input data (timesteps, features)
            
        Returns:
            keras.Model: Compiled LSTM model
        """
        try:
            model = Sequential([
                LSTM(50, activation='relu', input_shape=input_shape, return_sequences=True),
                Dropout(0.2),
                LSTM(50, activation='relu', return_sequences=False),
                Dropout(0.2),
                Dense(25, activation='relu'),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            self.current_model = model
            return model
            
        except Exception as e:
            logging.error(f"Error building model: {str(e)}")
            raise
            
    def train_model(self, data, epochs=100, batch_size=32):
        """
        Train the model on the provided data.
        
        Args:
            data (pd.DataFrame): The input data for training
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            dict: Training history
        """
        try:
            logging.info("Starting model training...")
            
            # Prepare data for training
            X_train, y_train = self.prepare_data_for_training(data)
            
            if X_train is None or y_train is None:
                raise ValueError("Failed to prepare training data")
                
            # Build model if not already built
            if self.current_model is None:
                self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
                
            # Train the model
            history = self.current_model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                verbose=1
            )
            
            # Save the model
            self.save_model('models/stock_model.h5')
            
            logging.info("Model training completed successfully")
            return history.history
            
        except Exception as e:
            logging.error(f"Error during model training: {str(e)}")
            raise
            
    def predict(self, data):
        """
        Make predictions using the trained model.
        
        Args:
            data (pd.DataFrame): Input data for prediction
            
        Returns:
            np.ndarray: Predicted values
        """
        try:
            if self.current_model is None:
                raise ValueError("Model not trained. Please train the model first.")
                
            # Get features from config
            features = self.config.get('features', ['open', 'high', 'low', 'close', 'volume'])
            sequence_length = self.config.get('sequence_length', 60)
            
            # Check if we have enough data
            if len(data) < sequence_length:
                raise ValueError(f"Not enough data for prediction. Need at least {sequence_length} rows.")
                
            # Prepare the last sequence for prediction
            feature_data = data[features].values
            X = feature_data[-sequence_length:].reshape(1, sequence_length, len(features))
            
            # Normalize the data
            X_reshaped = X.reshape((X.shape[0] * X.shape[1], X.shape[2]))
            X_scaled = self.scaler.transform(X_reshaped)
            X = X_scaled.reshape(X.shape)
            
            # Make prediction
            prediction = self.current_model.predict(X, verbose=0)
            
            return prediction
            
        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            raise
            
    def save_model(self, filepath):
        """
        Save the trained model to a file.
        
        Args:
            filepath (str): Path to save the model
        """
        try:
            if self.current_model is None:
                raise ValueError("No model to save")
                
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save the model
            self.current_model.save(filepath)
            logging.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            raise
            
    def load_model(self, filepath):
        """
        Load a trained model from a file.
        
        Args:
            filepath (str): Path to the saved model
        """
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Model file not found: {filepath}")
                
            self.current_model = tf.keras.models.load_model(filepath)
            logging.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise 