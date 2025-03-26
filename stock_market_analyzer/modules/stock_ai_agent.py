import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging

class StockAIAgent:
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.model = None
        self.scaler = MinMaxScaler()
        
        # Model parameters
        self.lookback_days = config.get('lookback_days', 60)
        self.prediction_days = config.get('prediction_days', 5)
        
        # Training parameters
        self.n_estimators = config.get('training', {}).get('n_estimators', 100)
        self.learning_rate = config.get('training', {}).get('learning_rate', 0.1)
        self.validation_split = config.get('training', {}).get('validation_split', 0.2)
        
        # Available models
        self.available_models = {
            'lightgbm': {
                'name': 'LightGBM Regressor',
                'type': 'regression',
                'parameters': {
                    'n_estimators': self.n_estimators,
                    'learning_rate': self.learning_rate
                }
            }
        }
        
        # Active model tracking
        self.active_model_id = None
        
    def list_models(self) -> dict:
        """List all available models and their configurations."""
        return self.available_models
        
    def get_active_model(self) -> dict:
        """Get the currently active model configuration."""
        if self.model is None:
            return None
            
        return {
            'type': 'lightgbm',
            'name': 'LightGBM Regressor',
            'parameters': {
                'n_estimators': self.n_estimators,
                'learning_rate': self.learning_rate,
                'lookback_days': self.lookback_days,
                'prediction_days': self.prediction_days
            },
            'status': 'trained' if self.model is not None else 'untrained'
        }
        
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
            predictions = self.predict(test_data)
            actual = test_data['close'].values[self.lookback_days:]
            
            # Calculate metrics
            evaluation = {
                'mse': mean_squared_error(actual, predictions),
                'mae': mean_absolute_error(actual, predictions),
                'rmse': np.sqrt(mean_squared_error(actual, predictions)),
                'mape': np.mean(np.abs((actual - predictions) / actual)) * 100
            }
            
            self.logger.info("Model evaluation completed")
            return evaluation
            
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