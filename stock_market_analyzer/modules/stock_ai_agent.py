import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
import os
import pickle
from typing import Dict, List, Optional, Any
import tensorflow as tf
import xgboost as xgb
import traceback

class StockAIAgent:
    """AI agent for stock market analysis and prediction."""
    
    def __init__(self, data_loader):
        """Initialize the AI agent with a data loader."""
        self.logger = logging.getLogger(__name__)
        self.data_loader = data_loader
        
        # Default configuration
        self.lookback_days = 60
        self.prediction_days = 5
        self.training_config = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'validation_split': 0.2
        }
        
        # Load available models
        self.models = self.load_available_models()
        self.current_model = None
        
        self.logger.info(f"Loaded {len(self.models)} available models")
    
    def load_available_models(self):
        """Load available models from the models directory."""
        try:
            models = []
            models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
            
            # Ensure models directory exists
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)
            
            # Add default models
            models.extend([
                {'name': 'LSTM', 'type': 'deep_learning'},
                {'name': 'GRU', 'type': 'deep_learning'},
                {'name': 'XGBoost', 'type': 'ensemble'},
                {'name': 'RandomForest', 'type': 'ensemble'},
                {'name': 'LightGBM', 'type': 'ensemble'},
                {'name': 'Prophet', 'type': 'statistical'},
                {'name': 'ARIMA', 'type': 'statistical'},
                {'name': 'SARIMA', 'type': 'statistical'},
                {'name': 'VAR', 'type': 'statistical'},
                {'name': 'Transformer', 'type': 'deep_learning'},
                {'name': 'Ensemble', 'type': 'ensemble'}
            ])
            
            return models
            
        except Exception as e:
            self.logger.error(f"Error loading available models: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    def get_available_models(self):
        """Get list of available models."""
        return self.models
    
    def get_current_model(self):
        """Get the current model."""
        return self.current_model
    
    async def analyze_technical(self, data: pd.DataFrame, lookback: int = None) -> Dict:
        """Run technical analysis on the data."""
        try:
            lookback = lookback or self.lookback_days
            
            # Calculate technical indicators
            results = {}
            
            # Moving averages
            results['SMA_20'] = data['close'].rolling(window=20).mean().iloc[-1]
            results['SMA_50'] = data['close'].rolling(window=50).mean().iloc[-1]
            results['EMA_20'] = data['close'].ewm(span=20).mean().iloc[-1]
            
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            results['RSI'] = (100 - (100 / (1 + rs))).iloc[-1]
            
            # MACD
            exp1 = data['close'].ewm(span=12, adjust=False).mean()
            exp2 = data['close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            results['MACD'] = macd.iloc[-1]
            results['MACD_Signal'] = signal.iloc[-1]
            
            # Bollinger Bands
            sma = data['close'].rolling(window=20).mean()
            std = data['close'].rolling(window=20).std()
            results['BB_Upper'] = (sma + (std * 2)).iloc[-1]
            results['BB_Lower'] = (sma - (std * 2)).iloc[-1]
            
            # Volume analysis
            results['Volume_SMA'] = data['volume'].rolling(window=20).mean().iloc[-1]
            results['Volume_Change'] = (data['volume'].iloc[-1] / data['volume'].iloc[-2] - 1) * 100
            
            # Price momentum
            results['Price_Change'] = (data['close'].iloc[-1] / data['close'].iloc[-2] - 1) * 100
            results['Price_Change_5D'] = (data['close'].iloc[-1] / data['close'].iloc[-6] - 1) * 100
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in technical analysis: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    async def analyze_fundamental(self, symbol: str) -> Dict:
        """Run fundamental analysis on the stock."""
        try:
            # Use yfinance for fundamental data
            import yfinance as yf
            
            # Get stock info
            stock = yf.Ticker(symbol)
            info = stock.info
            
            # Extract relevant metrics
            results = {
                'Market Cap': info.get('marketCap'),
                'PE Ratio': info.get('trailingPE'),
                'Forward PE': info.get('forwardPE'),
                'PEG Ratio': info.get('pegRatio'),
                'Price to Book': info.get('priceToBook'),
                'Dividend Yield': info.get('dividendYield'),
                'Profit Margin': info.get('profitMargins'),
                'Operating Margin': info.get('operatingMargins'),
                'Return on Equity': info.get('returnOnEquity'),
                'Revenue Growth': info.get('revenueGrowth'),
                'Earnings Growth': info.get('earningsGrowth'),
                'Quick Ratio': info.get('quickRatio'),
                'Current Ratio': info.get('currentRatio'),
                'Debt to Equity': info.get('debtToEquity')
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in fundamental analysis: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    async def train_model(self, model_name: str, epochs: int, batch_size: int, callback=None):
        """Train a model with the given parameters."""
        try:
            self.logger.info(f"Training model {model_name}")
            
            # Get model class
            model_class = self.get_model_class(model_name)
            if model_class is None:
                raise ValueError(f"Model {model_name} not found")
            
            # Create and train model
            self.current_model = model_class(
                lookback_days=self.lookback_days,
                prediction_days=self.prediction_days,
                **self.training_config
            )
            
            # Train the model
            history = await self.current_model.train(
                epochs=epochs,
                batch_size=batch_size,
                callback=callback
            )
            
            return history
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def get_model_class(self, model_name: str):
        """Get the model class for the given name."""
        try:
            # Import model classes
            from .models import (
                LSTMModel, GRUModel, XGBoostModel, RandomForestModel,
                LightGBMModel, ProphetModel, ARIMAModel, SARIMAModel,
                VARModel, TransformerModel, EnsembleModel
            )
            
            # Map model names to classes
            model_map = {
                'LSTM': LSTMModel,
                'GRU': GRUModel,
                'XGBoost': XGBoostModel,
                'RandomForest': RandomForestModel,
                'LightGBM': LightGBMModel,
                'Prophet': ProphetModel,
                'ARIMA': ARIMAModel,
                'SARIMA': SARIMAModel,
                'VAR': VARModel,
                'Transformer': TransformerModel,
                'Ensemble': EnsembleModel
            }
            
            return model_map.get(model_name)
            
        except Exception as e:
            self.logger.error(f"Error getting model class: {str(e)}")
            return None

    def set_active_model(self, model_name: str):
        """Set the active model by name."""
        if not model_name:
            raise ValueError("Model name cannot be empty")
            
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
            
        self.current_model = model_name
        self.logger.info(f"Set active model to: {model_name}")
        
    def get_active_model(self) -> Optional[Any]:
        """Get the currently active model."""
        return self.current_model
        
    def save_model(self, model_path: str, model: Optional[Any] = None) -> None:
        """Save the current model to disk.
        
        Args:
            model_path: Path where to save the model
            model: Optional model to save. If None, uses self.current_model
        """
        try:
            # Use provided model or active model
            model_to_save = model if model is not None else self.current_model
            if model_to_save is None:
                raise ValueError("No model to save")
                
            # Try saving as Keras model first
            try:
                if isinstance(model_to_save, (tf.keras.Model, tf.keras.Sequential)):
                    model_to_save.save(model_path)
                    self.logger.info(f"Saved Keras model to {model_path}")
                else:
                    # For non-Keras models, use pickle
                    with open(model_path, 'wb') as f:
                        pickle.dump(model_to_save, f)
                    self.logger.info(f"Saved model to {model_path}")
            except Exception as e:
                self.logger.error(f"Error saving model: {e}")
                raise ValueError(f"Failed to save model: {str(e)}")
                    
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise
        
    def prepare_training_data(self, data: pd.DataFrame) -> tuple:
        """Prepare data for model training."""
        try:
            self.logger.info("Preparing training data")
            
            # Normalize column names to lowercase if needed
            column_mapping = {}
            for col in data.columns:
                if col.lower() in ['open', 'high', 'low', 'close', 'volume']:
                    column_mapping[col] = col.lower()
            
            # Create a copy of the dataframe with normalized column names
            data_normalized = data.copy()
            if column_mapping:
                data_normalized = data_normalized.rename(columns=column_mapping)
            
            # Check if required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data_normalized.columns]
            
            if missing_columns:
                self.logger.warning(f"Missing required columns: {missing_columns}")
                # Try to find capitalized versions
                for col in missing_columns:
                    cap_col = col.capitalize()
                    if cap_col in data.columns:
                        data_normalized[col] = data[cap_col]
                        self.logger.info(f"Using {cap_col} for {col}")
            
            # Create feature scaler if not exists
            if not hasattr(self, 'feature_scaler'):
                self.feature_scaler = MinMaxScaler()
            
            # Prepare features
            features = data_normalized[['open', 'high', 'low', 'close', 'volume']].values
            scaled_features = self.feature_scaler.fit_transform(features)
            
            # Create sequences
            X, y = [], []
            for i in range(self.lookback_days, len(scaled_features)):
                # Get sequence of lookback_days
                sequence = scaled_features[i - self.lookback_days:i]
                X.append(sequence)
                # Target is the next day's closing price
                y.append(scaled_features[i, 3])  # 3 is the index of 'close' in our features
            
            X = np.array(X)
            y = np.array(y)
            
            # Log shapes for debugging
            self.logger.info(f"Prepared training data: X shape {X.shape}, y shape {y.shape}")
            
            # Ensure X has the correct shape for LSTM (samples, time steps, features)
            if len(X.shape) == 2:
                X = X.reshape(X.shape[0], X.shape[1], 1)
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {e}")
            raise
            
    def build_model(self):
        """Build the model."""
        try:
            self.model = LGBMRegressor(
                n_estimators=self.training_config['n_estimators'],
                learning_rate=self.training_config['learning_rate'],
                objective='regression',
                n_jobs=-1
            )
            self.logger.info("Model built successfully")
                
        except Exception as e:
            self.logger.error(f"Error building model: {e}")
            raise
            
    def train(self, data: pd.DataFrame, validation_data: pd.DataFrame = None):
        """Train the model with the provided data."""
        try:
            # Prepare training data
            X_train, y_train = self.prepare_training_data(data)
            
            # Build model if not already built
            if self.model is None:
                self.build_model()
                
            # Prepare validation data if provided
            eval_set = None
            if validation_data is not None:
                X_val, y_val = self.prepare_training_data(validation_data)
                eval_set = [(X_val, y_val)]
                
            # Train the model
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                eval_metric='l2'
            )
            
            self.logger.info("Model training completed")
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            raise
            
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model."""
        try:
            if self.model is None:
                raise ValueError("Model has not been trained yet")
                
            # Prepare data for prediction
            X, _ = self.prepare_training_data(data)
            
            # Make predictions
            scaled_predictions = self.model.predict(X)
            
            # Reshape predictions to match scaler shape
            reshaped_predictions = np.zeros((len(scaled_predictions), 2))
            reshaped_predictions[:, 0] = scaled_predictions  # Use predictions directly
            
            # Inverse transform predictions
            predictions = self.feature_scaler.inverse_transform(reshaped_predictions)[:, 0]
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {e}")
            raise
            
    def predict_future(self, data: pd.DataFrame, days: int = None) -> np.ndarray:
        """Predict future values based on the last known sequence."""
        try:
            if self.current_model is None:
                raise ValueError("No active model has been set. Please load or train a model first.")
                
            if days is None:
                days = self.prediction_days
                
            # Normalize column names to lowercase if needed
            column_mapping = {}
            for col in data.columns:
                if col.lower() in ['open', 'high', 'low', 'close', 'volume']:
                    column_mapping[col] = col.lower()
            
            # Create a copy of the dataframe with normalized column names
            data_normalized = data.copy()
            if column_mapping:
                data_normalized = data_normalized.rename(columns=column_mapping)
            
            # Check if required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data_normalized.columns]
            
            if missing_columns:
                self.logger.warning(f"Missing required columns: {missing_columns}")
                # Try to find capitalized versions
                for col in missing_columns:
                    cap_col = col.capitalize()
                    if cap_col in data.columns:
                        data_normalized[col] = data[cap_col]
                        self.logger.info(f"Using {cap_col} for {col}")
            
            # Prepare the last sequence
            features = data_normalized[['open', 'high', 'low', 'close', 'volume']].values[-self.lookback_days:]
            
            # Create feature scaler if not exists
            if not hasattr(self, 'feature_scaler'):
                self.feature_scaler = MinMaxScaler()
                self.feature_scaler.fit(features)
                
            scaled_features = self.feature_scaler.transform(features)
            
            # Reshape based on model type
            if isinstance(self.current_model, xgb.XGBRegressor):
                last_sequence = scaled_features.reshape(1, -1)  # Flatten for XGBoost
            else:  # For Keras models (LSTM, Transformer)
                last_sequence = scaled_features.reshape(1, self.lookback_days, scaled_features.shape[1])
            
            predictions = []
            current_sequence = last_sequence.copy()
            
            for _ in range(days):
                # Predict next value
                prediction = self.current_model.predict(current_sequence)
                
                # Handle different output shapes based on model type
                if isinstance(prediction, list):
                    next_pred = prediction[0]
                elif isinstance(prediction, np.ndarray):
                    # If prediction is multi-dimensional, flatten to single value
                    if prediction.ndim > 1:
                        next_pred = prediction[0]
                        if isinstance(next_pred, np.ndarray) and len(next_pred) > 0:
                            next_pred = next_pred[0]
                    else:
                        next_pred = prediction[0]
                else:
                    next_pred = prediction
                
                # Ensure next_pred is a scalar
                if hasattr(next_pred, 'shape') and next_pred.shape:
                    next_pred = float(next_pred)
                
                predictions.append(next_pred)
                
                # Update sequence for next prediction
                if isinstance(self.current_model, xgb.XGBRegressor):
                    # Update for XGBoost
                    new_row = np.zeros(scaled_features.shape[1])
                    new_row[3] = next_pred  # Set close price (index 3)
                    current_sequence = np.roll(current_sequence.reshape(-1, scaled_features.shape[1]), -1, axis=0)
                    current_sequence[-1] = new_row
                    current_sequence = current_sequence.reshape(1, -1)
                else:
                    # Update for Keras models
                    new_row = scaled_features[-1].copy()
                    new_row[3] = next_pred  # Set close price (index 3)
                    current_sequence = np.roll(current_sequence.reshape(self.lookback_days, scaled_features.shape[1]), -1, axis=0)
                    current_sequence[-1] = new_row
                    current_sequence = current_sequence.reshape(1, self.lookback_days, scaled_features.shape[1])
            
            # Reshape predictions for inverse transform
            if len(predictions) > 0:
                # Create a properly shaped array for inverse transformation
                reshaped_predictions = np.zeros((len(predictions), scaled_features.shape[1]))
                
                # Convert predictions to numpy array of floats explicitly
                predictions_array = np.array(predictions, dtype=float)
                
                # Set close price (index 3)
                reshaped_predictions[:, 3] = predictions_array
                
                # Use the last row's values for other columns
                for i in range(scaled_features.shape[1]):
                    if i != 3:  # Skip close price
                        reshaped_predictions[:, i] = scaled_features[-1, i]
                
                # Inverse transform predictions
                final_predictions = self.feature_scaler.inverse_transform(reshaped_predictions)[:, 3]
                return final_predictions
            else:
                return np.array([])
            
        except Exception as e:
            self.logger.error(f"Error predicting future values: {e}")
            raise
            
    def evaluate(self, test_data: pd.DataFrame) -> dict:
        """Evaluate the model's performance."""
        try:
            if self.model is None:
                raise ValueError("Model has not been trained yet")
                
            # Prepare test data
            X_test, y_test = self.prepare_training_data(test_data)
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            return {
                'mse': mse,
                'mae': mae,
                'rmse': np.sqrt(mse)
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {e}")
            raise

    async def analyze(self, data: pd.DataFrame) -> dict:
        """Perform complete analysis including training, prediction, and evaluation."""
        try:
            # Split data into training and test sets
            train_size = int(len(data) * (1 - self.training_config['validation_split']))
            train_data = data[:train_size]
            test_data = data[train_size:]
            
            # Train the model
            await self.train_model(self.current_model, self.training_config['n_estimators'], 32)
            
            # Make predictions
            predictions = self.predict(test_data)
            
            # Evaluate the model
            evaluation = self.evaluate(test_data)
            
            # Predict future values
            future_predictions = self.predict_future(data)
            
            # Combine results
            analysis_results = {
                'evaluation_metrics': evaluation,
                'last_prediction': predictions[-1],
                'future_predictions': future_predictions.tolist(),
                'model_info': self.training_config
            }
            
            self.logger.info("Analysis completed successfully")
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error performing analysis: {e}")
            raise

    async def analyze_technical(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform technical analysis on the data."""
        try:
            self.logger.info("Starting technical analysis")
            
            # Normalize column names to lowercase if needed
            column_mapping = {}
            for col in data.columns:
                if col.lower() in ['open', 'high', 'low', 'close', 'volume']:
                    column_mapping[col] = col.lower()
            
            # Create a copy of the dataframe with normalized column names
            data_normalized = data.copy()
            if column_mapping:
                data_normalized = data_normalized.rename(columns=column_mapping)
            
            # Check if required columns exist
            required_columns = ['close']
            missing_columns = [col for col in required_columns if col not in data_normalized.columns]
            
            if missing_columns:
                self.logger.warning(f"Missing required columns: {missing_columns}")
                # Try to find capitalized versions
                for col in missing_columns:
                    cap_col = col.capitalize()
                    if cap_col in data.columns:
                        data_normalized[col] = data[cap_col]
                        self.logger.info(f"Using {cap_col} for {col}")
            
            # Import our technical indicators module
            from modules.technical_indicators import TechnicalIndicators, PatternRecognition, SentimentAnalysis
            
            # Calculate technical indicators
            results = {}
            
            # Moving averages
            results['sma_20'] = data_normalized['close'].rolling(window=20).mean().iloc[-1]
            results['sma_50'] = data_normalized['close'].rolling(window=50).mean().iloc[-1]
            results['sma_200'] = data_normalized['close'].rolling(window=200).mean().iloc[-1]
            
            # RSI
            delta = data_normalized['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            results['rsi'] = 100 - (100 / (1 + rs)).iloc[-1]
            
            # MACD
            exp1 = data_normalized['close'].ewm(span=12, adjust=False).mean()
            exp2 = data_normalized['close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            results['macd'] = macd.iloc[-1]
            results['macd_signal'] = signal.iloc[-1]
            
            # Bollinger Bands
            middle_band = data_normalized['close'].rolling(window=20).mean()
            std = data_normalized['close'].rolling(window=20).std()
            results['bb_upper'] = middle_band.iloc[-1] + (std.iloc[-1] * 2)
            results['bb_middle'] = middle_band.iloc[-1]
            results['bb_lower'] = middle_band.iloc[-1] - (std.iloc[-1] * 2)
            
            # Volume indicators
            results['volume_ma'] = data_normalized['volume'].rolling(window=20).mean().iloc[-1]
            results['volume_ratio'] = data_normalized['volume'].iloc[-1] / results['volume_ma']
            
            # Price momentum
            results['daily_return'] = data_normalized['close'].pct_change().iloc[-1]
            results['volatility'] = data_normalized['close'].pct_change().std() * np.sqrt(252)
            
            # Add advanced indicators if data has sufficient length
            if len(data) >= 60:
                # Add Ichimoku Cloud
                try:
                    ichimoku_df = TechnicalIndicators.add_ichimoku_cloud(data_normalized)
                    results['tenkan_sen'] = ichimoku_df['tenkan_sen'].iloc[-1]
                    results['kijun_sen'] = ichimoku_df['kijun_sen'].iloc[-1]
                    results['senkou_span_a'] = ichimoku_df['senkou_span_a'].iloc[-1]
                    results['senkou_span_b'] = ichimoku_df['senkou_span_b'].iloc[-1]
                except Exception as e:
                    self.logger.warning(f"Error calculating Ichimoku Cloud: {e}")
                
                # Add VWAP
                try:
                    vwap_df = TechnicalIndicators.add_vwap(data_normalized)
                    results['vwap'] = vwap_df['vwap'].iloc[-1]
                except Exception as e:
                    self.logger.warning(f"Error calculating VWAP: {e}")
                
                # Detect patterns
                try:
                    patterns_df = PatternRecognition.find_all_patterns(data_normalized)
                    # Extract pattern results from the last row
                    for col in ['head_shoulders', 'double_top', 'double_bottom', 
                               'ascending_triangle', 'descending_triangle', 'symmetrical_triangle']:
                        if col in patterns_df.columns:
                            results[col] = bool(patterns_df[col].iloc[-1])
                except Exception as e:
                    self.logger.warning(f"Error detecting patterns: {e}")
                
                # Add sentiment analysis if symbol is available
                try:
                    symbol = data_normalized.get('symbol', 'Unknown')
                    if hasattr(data_normalized, 'name'):
                        symbol = data_normalized.name
                    sentiment = SentimentAnalysis.get_combined_sentiment(symbol)
                    results.update(sentiment)
                except Exception as e:
                    self.logger.warning(f"Error calculating sentiment: {e}")
            
            self.logger.info("Technical analysis completed successfully")
            return {'technical': results}
            
        except Exception as e:
            self.logger.error(f"Error in technical analysis: {e}")
            raise ValueError(f"Technical analysis failed: {str(e)}")
            
    async def analyze_fundamental(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform fundamental analysis on the data."""
        try:
            self.logger.info("Starting fundamental analysis")
            
            # Calculate fundamental metrics
            results = {}
            
            # Price-to-Earnings ratio (if available)
            if 'pe_ratio' in data.columns:
                results['pe_ratio'] = data['pe_ratio'].iloc[-1]
                
            # Price-to-Book ratio (if available)
            if 'pb_ratio' in data.columns:
                results['pb_ratio'] = data['pb_ratio'].iloc[-1]
                
            # Dividend yield (if available)
            if 'dividend_yield' in data.columns:
                results['dividend_yield'] = data['dividend_yield'].iloc[-1]
                
            # Market cap (if available)
            if 'market_cap' in data.columns:
                results['market_cap'] = data['market_cap'].iloc[-1]
                
            # Revenue growth (if available)
            if 'revenue_growth' in data.columns:
                results['revenue_growth'] = data['revenue_growth'].iloc[-1]
                
            # Profit margin (if available)
            if 'profit_margin' in data.columns:
                results['profit_margin'] = data['profit_margin'].iloc[-1]
                
            self.logger.info("Fundamental analysis completed successfully")
            return {'fundamental': results}
            
        except Exception as e:
            self.logger.error(f"Error in fundamental analysis: {e}")
            raise ValueError(f"Fundamental analysis failed: {str(e)}")
            
    async def analyze_sentiment(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform sentiment analysis on the data."""
        try:
            self.logger.info("Starting sentiment analysis")
            
            # Import our sentiment analysis module
            from modules.technical_indicators import SentimentAnalysis
            
            # Calculate sentiment metrics
            results = {}
            
            # Get symbol information
            symbol = data.get('symbol', 'Unknown')
            if hasattr(data, 'name'):
                symbol = data.name
            
            # Get comprehensive sentiment analysis
            sentiment_results = SentimentAnalysis.get_combined_sentiment(symbol)
            results.update(sentiment_results)
            
            # Add traditional indicators as well
            # Social media sentiment (if available in data)
            if 'social_sentiment' in data.columns:
                results['historic_social_sentiment'] = data['social_sentiment'].iloc[-1]
                
            # News sentiment (if available in data)
            if 'news_sentiment' in data.columns:
                results['historic_news_sentiment'] = data['news_sentiment'].iloc[-1]
                
            # Market sentiment indicators
            results['price_momentum'] = data['close'].pct_change(periods=5).iloc[-1]
            results['volume_momentum'] = data['volume'].pct_change(periods=5).iloc[-1]
            
            # Volatility as a sentiment indicator
            results['volatility'] = data['close'].pct_change().std() * np.sqrt(252)
            
            self.logger.info("Sentiment analysis completed successfully")
            return {'sentiment': results}
            
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {e}")
            raise ValueError(f"Sentiment analysis failed: {str(e)}")

    def create_model(self, model_type: str, params: dict) -> Any:
        """Create a new model of the specified type with given parameters."""
        try:
            self.logger.info(f"Creating {model_type} model with parameters: {params}")
            self.current_model = model_type
            
            if model_type == 'LSTM':
                # Ensure input_dim is set with a default value if not provided
                if 'input_dim' not in params:
                    # Default to 5 features (OHLCV)
                    params['input_dim'] = 5
                    self.logger.info(f"Using default input_dim of {params['input_dim']}")
                
                # Create LSTM model
                model = tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(self.lookback_days, params['input_dim'])),
                    tf.keras.layers.LSTM(
                        units=params.get('hidden_dim', 64),
                        return_sequences=params.get('num_layers', 2) > 1
                    ),
                    tf.keras.layers.Dropout(params.get('dropout', 0.2))
                ])
                
                # Add additional LSTM layers if specified
                for _ in range(params.get('num_layers', 2) - 1):
                    model.add(tf.keras.layers.LSTM(units=params.get('hidden_dim', 64), return_sequences=True))
                    model.add(tf.keras.layers.Dropout(params.get('dropout', 0.2)))
                
                # Add output layer
                model.add(tf.keras.layers.Dense(1))
                
                # Compile model
                model.compile(optimizer='adam', loss='mse')
                self.logger.info("LSTM model created successfully")
                
            elif model_type == 'XGBoost':
                # Create XGBoost model
                model = xgb.XGBRegressor(
                    learning_rate=params.get('learning_rate', 0.01),
                    max_depth=params.get('max_depth', 6),
                    n_estimators=params.get('n_trees', 100),
                    objective='reg:squarederror'
                )
                self.logger.info("XGBoost model created successfully")
                
            elif model_type == 'Transformer':
                # Create Transformer model using Functional API
                if 'input_dim' not in params:
                    # Default to 5 features (OHLCV)
                    params['input_dim'] = 5
                    self.logger.info(f"Using default input_dim of {params['input_dim']}")
                
                inputs = tf.keras.Input(shape=(self.lookback_days, params['input_dim']))
                
                # Initial dense layer
                x = tf.keras.layers.Dense(params.get('hidden_dim', 128))(inputs)
                
                # Transformer blocks
                for _ in range(params.get('n_layers', 2)):
                    # Multi-head attention
                    attn_output = tf.keras.layers.MultiHeadAttention(
                        num_heads=params.get('n_heads', 4),
                        key_dim=params.get('hidden_dim', 128) // params.get('n_heads', 4)
                    )(x, x)
                    
                    # Add & Norm
                    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
                    
                    # Feed-forward network
                    ffn_output = tf.keras.layers.Dense(params.get('hidden_dim', 128))(x)
                    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)
                    
                    # Dropout
                    x = tf.keras.layers.Dropout(params.get('dropout', 0.1))(x)
                
                # Output layer
                outputs = tf.keras.layers.Dense(1)(x)
                
                # Create model
                model = tf.keras.Model(inputs=inputs, outputs=outputs)
                model.compile(optimizer='adam', loss='mse')
                self.logger.info("Transformer model created successfully")
                
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error creating {model_type} model: {e}")
            raise

    def train_model(self, model, X, y, epochs=100, batch_size=32, validation_split=0.2):
        """Train the model with progress updates."""
        try:
            self.logger.info(f"Starting model training with {epochs} epochs")
            
            # Store the model as active model
            self.current_model = model
            
            # Initialize training history
            history = {
                'loss': [],
                'val_loss': [],
                'best_val_loss': float('inf'),
                'patience': 10,
                'patience_counter': 0
            }
            
            # Split data into training and validation sets
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            if isinstance(model, xgb.XGBRegressor):
                # XGBoost training
                eval_set = [(X_train.reshape(X_train.shape[0], -1), y_train),
                           (X_val.reshape(X_val.shape[0], -1), y_val)]
                
                model.fit(
                    X_train.reshape(X_train.shape[0], -1),  # Flatten for XGBoost
                    y_train,
                    eval_set=eval_set,
                    verbose=False
                )
                
                # Get predictions for history
                train_pred = model.predict(X_train.reshape(X_train.shape[0], -1))
                val_pred = model.predict(X_val.reshape(X_val.shape[0], -1))
                
                history['loss'] = [mean_squared_error(y_train, train_pred)]
                history['val_loss'] = [mean_squared_error(y_val, val_pred)]
                
            else:
                # Keras model training
                for epoch in range(epochs):
                    # Train for one epoch
                    train_result = model.fit(
                        X_train, y_train,
                        batch_size=batch_size,
                        epochs=1,
                        validation_data=(X_val, y_val),
                        verbose=0
                    )
                    
                    # Get loss values
                    train_loss = train_result.history['loss'][0]
                    val_loss = train_result.history['val_loss'][0]
                    
                    # Update history
                    history['loss'].append(train_loss)
                    history['val_loss'].append(val_loss)
                    
                    # Log progress
                    self.logger.info(f"Epoch {epoch + 1}/{epochs}: loss={train_loss:.4f}, val_loss={val_loss:.4f}")
                    
                    # Early stopping check
                    if val_loss < history['best_val_loss']:
                        history['best_val_loss'] = val_loss
                        history['patience_counter'] = 0
                        # Save best model
                        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', f"best_model_{self.current_model}.keras")
                        model.save(model_path)
                        self.logger.info(f"Saved best model to {model_path}")
                    else:
                        history['patience_counter'] += 1
                        if history['patience_counter'] >= history['patience']:
                            self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                            break
            
            self.logger.info("Model training completed")
            return history
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            raise ValueError(f"Error training model: {str(e)}")

    def load_model(self, model_path: str) -> None:
        """Load a saved model from disk."""
        try:
            # Check if the file exists
            if not os.path.exists(model_path):
                self.logger.error(f"Model file not found: {model_path}")
                raise FileNotFoundError(f"Model file not found: {model_path}")
                
            # Try loading as Keras model first
            try:
                # Define custom objects for all model types
                custom_objects = {
                    'MultiHeadAttention': tf.keras.layers.MultiHeadAttention,
                    'LayerNormalization': tf.keras.layers.LayerNormalization,
                    'Dense': tf.keras.layers.Dense,
                    'Dropout': tf.keras.layers.Dropout,
                    'LSTM': tf.keras.layers.LSTM,
                    'Input': tf.keras.layers.Input,
                    'Flatten': tf.keras.layers.Flatten
                }
                
                # Determine if this is a transformer model based on filename
                is_transformer = "transformer" in model_path.lower()
                
                # Load the model with appropriate configuration
                if is_transformer:
                    self.current_model = tf.keras.models.load_model(
                        model_path,
                        custom_objects=custom_objects,
                        compile=False  # Don't compile transformer models to avoid parameter mismatches
                    )
                else:
                    self.current_model = tf.keras.models.load_model(model_path)
                    
                self.logger.info(f"Loaded Keras model from {model_path}")
            except Exception as keras_error:
                self.logger.warning(f"Failed to load as Keras model: {keras_error}")
                # If that fails, try loading as pickle
                try:
                    with open(model_path, 'rb') as f:
                        self.current_model = pickle.load(f)
                    self.logger.info(f"Loaded model from {model_path}")
                except Exception as pickle_error:
                    self.logger.error(f"Error loading model as pickle: {pickle_error}")
                    raise ValueError(f"Failed to load model: {str(keras_error)}. Also failed as pickle: {str(pickle_error)}")
                    
        except FileNotFoundError as e:
            self.logger.error(f"Model file not found: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

    def load_model_by_name(self, model_name: str) -> None:
        """Load a model by its name.
        
        Args:
            model_name: Name of the model to load
        """
        try:
            # Check if model exists
            if model_name in self.models:
                model_path = self.models[model_name]['path']
                self.load_model(model_path)
                self.logger.info(f"Loaded model {model_name} from {model_path}")
            else:
                self.logger.error(f"Model not found: {model_name}")
                raise ValueError(f"Model not found: {model_name}")
                
        except Exception as e:
            self.logger.error(f"Error loading model {model_name}: {e}")
            raise

    def get_model_names(self) -> List[str]:
        """Get list of available model names."""
        try:
            # Get all .keras files in the models directory
            model_files = [f for f in os.listdir(os.path.dirname(__file__)) if f.endswith('.keras')]
            # Remove the .keras extension
            model_names = [os.path.splitext(f)[0] for f in model_files]
            self.logger.info(f"Found {len(model_names)} model(s): {model_names}")
            return model_names if model_names else ["No models available"]
        except Exception as e:
            self.logger.error(f"Error getting model names: {str(e)}")
            return ["No models available"]

    def get_model_type(self, model_name: str) -> Optional[str]:
        """Get the type of a model by its name."""
        try:
            # Check if model exists
            model_path = os.path.join(os.path.dirname(__file__), f"{model_name}.keras")
            if not os.path.exists(model_path):
                self.logger.warning(f"Model file not found: {model_path}")
                return None
            
            # Load model configuration
            model = tf.keras.models.load_model(model_path)
            
            # Determine model type based on architecture
            if any('lstm' in layer.name.lower() for layer in model.layers):
                return 'LSTM'
            elif any('transformer' in layer.name.lower() for layer in model.layers):
                return 'Transformer'
            elif any('xgboost' in layer.name.lower() for layer in model.layers):
                return 'XGBoost'
            else:
                return 'Unknown'
            
        except Exception as e:
            self.logger.error(f"Error determining model type for {model_name}: {str(e)}")
            return None 