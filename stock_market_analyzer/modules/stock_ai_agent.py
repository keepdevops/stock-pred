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

class StockAIAgent:
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        self.active_model = None
        self.available_models = {}
        self.load_available_models()
        
        # Model parameters
        self.lookback_days = config.get('lookback_days', 60)
        self.prediction_days = config.get('prediction_days', 5)
        
        # Training parameters
        self.n_estimators = config.get('training', {}).get('n_estimators', 100)
        self.learning_rate = config.get('training', {}).get('learning_rate', 0.1)
        self.validation_split = config.get('training', {}).get('validation_split', 0.2)
        
    def load_available_models(self):
        """Load available models from the models directory."""
        try:
            # Create models directory if it doesn't exist
            os.makedirs(self.models_dir, exist_ok=True)
            
            # Scan for model files
            for file in os.listdir(self.models_dir):
                if file.endswith('.h5'):  # Keras model files
                    model_name = os.path.splitext(file)[0]
                    model_path = os.path.join(self.models_dir, file)
                    self.available_models[model_name] = {
                        'path': model_path,
                        'type': 'keras',
                        'name': model_name
                    }
                elif file.endswith('.pkl'):  # Scikit-learn model files
                    model_name = os.path.splitext(file)[0]
                    model_path = os.path.join(self.models_dir, file)
                    self.available_models[model_name] = {
                        'path': model_path,
                        'type': 'sklearn',
                        'name': model_name
                    }
                    
            self.logger.info(f"Loaded {len(self.available_models)} available models")
            
        except Exception as e:
            self.logger.error(f"Error loading available models: {e}")
            raise
        
    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        return list(self.available_models.keys())
        
    def set_active_model(self, model_name: str):
        """Set the active model by name."""
        if model_name not in self.available_models:
            raise ValueError(f"Unknown model: {model_name}")
            
        model_info = self.available_models[model_name]
        try:
            if model_info['type'] == 'keras':
                self.active_model = tf.keras.models.load_model(model_info['path'])
            elif model_info['type'] == 'sklearn':
                with open(model_info['path'], 'rb') as f:
                    self.active_model = pickle.load(f)
                    
            self.logger.info(f"Set active model to: {model_name}")
            
        except Exception as e:
            self.logger.error(f"Error loading model {model_name}: {e}")
            raise
        
    def get_active_model(self) -> Optional[Any]:
        """Get the currently active model."""
        return self.active_model
        
    def save_model(self, model: Any, name: str):
        """Save a model to the models directory."""
        try:
            if isinstance(model, tf.keras.Model):
                model_path = os.path.join(self.models_dir, f"{name}.h5")
                model.save(model_path)
                self.available_models[name] = {
                    'path': model_path,
                    'type': 'keras',
                    'name': name
                }
            else:
                model_path = os.path.join(self.models_dir, f"{name}.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                self.available_models[name] = {
                    'path': model_path,
                    'type': 'sklearn',
                    'name': name
                }
                
            self.logger.info(f"Saved model: {name}")
            
        except Exception as e:
            self.logger.error(f"Error saving model {name}: {e}")
            raise
        
    def prepare_data(self, data: pd.DataFrame) -> tuple:
        """Prepare data for training or prediction."""
        try:
            # Extract features
            features = data[['close', 'volume']].values
            
            # Scale features
            scaled_features = self.scaler.fit_transform(features)
            
            X, y = [], []
            for i in range(self.lookback_days, len(scaled_features)):
                X.append(scaled_features[i - self.lookback_days:i].flatten())  # Flatten for LGBM
                y.append(scaled_features[i, 0])  # Predict only the next day's closing price
                
            return np.array(X), np.array(y)
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {e}")
            raise
            
    def build_model(self):
        """Build the model."""
        try:
            self.model = LGBMRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
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
            X_train, y_train = self.prepare_data(data)
            
            # Build model if not already built
            if self.model is None:
                self.build_model()
                
            # Prepare validation data if provided
            eval_set = None
            if validation_data is not None:
                X_val, y_val = self.prepare_data(validation_data)
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
            X, _ = self.prepare_data(data)
            
            # Make predictions
            scaled_predictions = self.model.predict(X)
            
            # Reshape predictions to match scaler shape
            reshaped_predictions = np.zeros((len(scaled_predictions), 2))
            reshaped_predictions[:, 0] = scaled_predictions  # Use predictions directly
            
            # Inverse transform predictions
            predictions = self.scaler.inverse_transform(reshaped_predictions)[:, 0]
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {e}")
            raise
            
    def predict_future(self, data: pd.DataFrame, days: int = None) -> np.ndarray:
        """Predict future values based on the last known sequence."""
        try:
            if self.model is None:
                raise ValueError("Model has not been trained yet")
                
            if days is None:
                days = self.prediction_days
                
            # Prepare the last sequence
            features = data[['close', 'volume']].values[-self.lookback_days:]
            scaled_features = self.scaler.transform(features)
            last_sequence = scaled_features.reshape(1, -1)  # Flatten for LGBM
            
            predictions = []
            current_sequence = last_sequence.copy()
            
            for _ in range(days):
                # Predict next value
                next_pred = self.model.predict(current_sequence)[0]
                predictions.append(next_pred)  # Take the predicted value
                
                # Update sequence for next prediction
                new_row = np.zeros(2)
                new_row[0] = next_pred
                new_row[1] = current_sequence[0, -1]  # Keep last volume
                current_sequence = np.roll(current_sequence.reshape(-1, 2), -1, axis=0)
                current_sequence[-1] = new_row
                current_sequence = current_sequence.reshape(1, -1)
                
            # Reshape predictions for inverse transform
            reshaped_predictions = np.zeros((len(predictions), 2))
            reshaped_predictions[:, 0] = predictions
            
            # Inverse transform predictions
            final_predictions = self.scaler.inverse_transform(reshaped_predictions)[:, 0]
            
            return final_predictions
            
        except Exception as e:
            self.logger.error(f"Error predicting future values: {e}")
            raise
            
    def evaluate(self, test_data: pd.DataFrame) -> dict:
        """Evaluate the model's performance."""
        try:
            if self.model is None:
                raise ValueError("Model has not been trained yet")
                
            # Prepare test data
            X_test, y_test = self.prepare_data(test_data)
            
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

    def analyze(self, data: pd.DataFrame) -> dict:
        """Perform complete analysis including training, prediction, and evaluation."""
        try:
            # Split data into training and test sets
            train_size = int(len(data) * (1 - self.validation_split))
            train_data = data[:train_size]
            test_data = data[train_size:]
            
            # Train the model
            self.train(train_data)
            
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
                'model_info': {
                    'lookback_days': self.lookback_days,
                    'prediction_days': self.prediction_days,
                    'n_estimators': self.n_estimators,
                    'learning_rate': self.learning_rate
                }
            }
            
            self.logger.info("Analysis completed successfully")
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error performing analysis: {e}")
            raise

    def analyze_technical(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform technical analysis on the data."""
        try:
            self.logger.info("Starting technical analysis")
            
            # Calculate technical indicators
            results = {}
            
            # Moving averages
            results['sma_20'] = data['close'].rolling(window=20).mean().iloc[-1]
            results['sma_50'] = data['close'].rolling(window=50).mean().iloc[-1]
            results['sma_200'] = data['close'].rolling(window=200).mean().iloc[-1]
            
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            results['rsi'] = 100 - (100 / (1 + rs)).iloc[-1]
            
            # MACD
            exp1 = data['close'].ewm(span=12, adjust=False).mean()
            exp2 = data['close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            results['macd'] = macd.iloc[-1]
            results['macd_signal'] = signal.iloc[-1]
            
            # Bollinger Bands
            middle_band = data['close'].rolling(window=20).mean()
            std = data['close'].rolling(window=20).std()
            results['bb_upper'] = middle_band.iloc[-1] + (std.iloc[-1] * 2)
            results['bb_middle'] = middle_band.iloc[-1]
            results['bb_lower'] = middle_band.iloc[-1] - (std.iloc[-1] * 2)
            
            # Volume indicators
            results['volume_ma'] = data['volume'].rolling(window=20).mean().iloc[-1]
            results['volume_ratio'] = data['volume'].iloc[-1] / results['volume_ma']
            
            # Price momentum
            results['daily_return'] = data['close'].pct_change().iloc[-1]
            results['volatility'] = data['close'].pct_change().std() * np.sqrt(252)
            
            self.logger.info("Technical analysis completed successfully")
            return {'technical': results}
            
        except Exception as e:
            self.logger.error(f"Error in technical analysis: {e}")
            raise ValueError(f"Technical analysis failed: {str(e)}")
            
    def analyze_fundamental(self, data: pd.DataFrame) -> Dict[str, Any]:
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
            
    def analyze_sentiment(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform sentiment analysis on the data."""
        try:
            self.logger.info("Starting sentiment analysis")
            
            # Calculate sentiment metrics
            results = {}
            
            # Social media sentiment (if available)
            if 'social_sentiment' in data.columns:
                results['social_sentiment'] = data['social_sentiment'].iloc[-1]
                
            # News sentiment (if available)
            if 'news_sentiment' in data.columns:
                results['news_sentiment'] = data['news_sentiment'].iloc[-1]
                
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