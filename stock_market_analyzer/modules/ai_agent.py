import pandas as pd
import numpy as np
import math
from typing import Dict, Any, Optional, List, Tuple
import logging
from .models.model_factory import ModelFactory
from .models.transformer_model import TransformerStockPredictor
from .models.lstm_model import LSTMStockPredictor
from .models.xgboost_model import XGBoostStockPredictor
from .models.model_tuner import ModelTuner
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

class AIAgent:
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.model_factory = ModelFactory()
        self.active_model_id = None
        self.scaler = MinMaxScaler()
        
        # Initialize default models
        self._initialize_models()
        
        # Initialize model tuners
        self.model_tuners = {}
        for model_type in ['transformer', 'lstm', 'xgboost']:
            self.model_tuners[model_type] = ModelTuner(model_type)
        
    def _initialize_models(self):
        """Initialize default models with configuration."""
        try:
            # Initialize transformer model
            transformer_params = self.config.get('transformer_params', {
                'input_dim': 6,
                'd_model': 64,
                'nhead': 4,
                'num_layers': 2,
                'dim_feedforward': 256,
                'dropout': 0.1
            })
            self.model_factory.create_model('transformer', transformer_params)
            
            # Initialize LSTM model
            lstm_params = self.config.get('lstm_params', {
                'input_dim': 6,
                'hidden_dim': 64,
                'num_layers': 2,
                'dropout': 0.1
            })
            self.model_factory.create_model('lstm', lstm_params)
            
            # Initialize XGBoost model
            xgb_params = self.config.get('xgb_params', {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            })
            self.model_factory.create_model('xgboost', xgb_params)
            
            # Set transformer as default active model
            self.active_model_id = 'transformer'
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")
            raise
            
    def set_active_model(self, model_id: str):
        """Set the active model for predictions."""
        if model_id in self.model_factory.models:
            self.active_model_id = model_id
            self.logger.info(f"Set active model to {model_id}")
        else:
            raise ValueError(f"Model {model_id} not found")
            
    def get_active_model(self) -> Any:
        """Get the currently active model."""
        return self.model_factory.get_model(self.active_model_id)
        
    def train_model(
        self,
        data: pd.DataFrame,
        model_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, List[float]]:
        """
        Train a specific model or the active model.
        
        Args:
            data: Training data
            model_id: Optional model ID to train (if None, uses active model)
            **kwargs: Additional training parameters
            
        Returns:
            Training history
        """
        model_id = model_id or self.active_model_id
        model = self.model_factory.get_model(model_id)
        
        if model is None:
            raise ValueError(f"Model {model_id} not found")
            
        try:
            # Prepare data
            X = self._prepare_features(data)
            y = data['close'].values
            
            # Train model
            history = model.train(X, y, **kwargs)
            self.logger.info(f"Model {model_id} trained successfully")
            return history
            
        except Exception as e:
            self.logger.error(f"Error training model {model_id}: {e}")
            raise
            
    def predict(
        self,
        data: pd.DataFrame,
        model_id: Optional[str] = None
    ) -> np.ndarray:
        """
        Make predictions using a specific model or the active model.
        
        Args:
            data: Input data for prediction
            model_id: Optional model ID to use (if None, uses active model)
            
        Returns:
            Array of predictions
        """
        model_id = model_id or self.active_model_id
        model = self.model_factory.get_model(model_id)
        
        if model is None:
            raise ValueError(f"Model {model_id} not found")
            
        try:
            # Prepare features
            X = self._prepare_features(data)
            
            # Make predictions
            predictions = model.predict(X)
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error making predictions with model {model_id}: {e}")
            raise
            
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for model input."""
        # Convert column names to lowercase for consistent handling
        data.columns = data.columns.str.lower()
        
        # Select relevant features
        features = ['open', 'high', 'low', 'close', 'volume']
        
        # Add technical indicators
        data = data.copy()
        
        # Use existing moving averages if available, otherwise calculate them
        if 'sma_5' in data.columns:
            data['ma5'] = data['sma_5']
        elif 'sma5' in data.columns:
            data['ma5'] = data['sma5']
        else:
            data['ma5'] = data['close'].rolling(window=5).mean()
            
        if 'sma_20' in data.columns:
            data['ma20'] = data['sma_20']
        elif 'sma20' in data.columns:
            data['ma20'] = data['sma20']
        else:
            data['ma20'] = data['close'].rolling(window=20).mean()
        
        # Use existing RSI if available, otherwise calculate it
        if 'rsi' in data.columns:
            pass  # Use existing RSI
        else:
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
        
        # Use existing MACD if available, otherwise calculate it
        if 'macd' in data.columns and 'macd_signal' in data.columns:
            data['signal'] = data['macd_signal']  # Rename to match expected feature name
        elif 'macd' in data.columns and 'signal_line' in data.columns:
            data['signal'] = data['signal_line']  # Alternative name for MACD signal
        else:
            exp1 = data['close'].ewm(span=12, adjust=False).mean()
            exp2 = data['close'].ewm(span=26, adjust=False).mean()
            data['macd'] = exp1 - exp2
            data['signal'] = data['macd'].ewm(span=9, adjust=False).mean()
        
        # Add new features to the list
        features.extend(['ma5', 'ma20', 'rsi', 'macd', 'signal'])
        
        # Handle missing values
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Convert to numpy array
        return data[features].values
        
    def save_model(self, model_id: str, path: str):
        """Save a specific model to disk."""
        self.model_factory.save_model(model_id, self.model_factory.get_model(model_id), path)
        
    def load_model(self, model_id: str, model_type: str, path: str, model_params: Optional[Dict[str, Any]] = None):
        """Load a model from disk."""
        self.model_factory.load_model(model_id, model_type, path, model_params)
        
    def list_models(self) -> Dict[str, str]:
        """List all available models and their types."""
        return self.model_factory.list_models()
        
    def tune_model(
        self,
        data: pd.DataFrame,
        model_id: Optional[str] = None,
        n_trials: int = 100,
        n_splits: int = 5
    ) -> Tuple[Dict[str, Any], float]:
        """
        Tune hyperparameters for a specific model or the active model.
        
        Args:
            data: Training data
            model_id: Optional model ID to tune (if None, uses active model)
            n_trials: Number of optimization trials
            n_splits: Number of splits for cross-validation
            
        Returns:
            Tuple of (best parameters, best validation loss)
        """
        model_id = model_id or self.active_model_id
        model = self.model_factory.get_model(model_id)
        
        if model is None:
            raise ValueError(f"Model {model_id} not found")
            
        try:
            # Prepare features
            X = self._prepare_features(data)
            
            # Get model type
            model_type = model.__class__.__name__.lower()
            if 'transformer' in model_type:
                model_type = 'transformer'
            elif 'lstm' in model_type:
                model_type = 'lstm'
            elif 'xgboost' in model_type:
                model_type = 'xgboost'
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
            # Run tuning
            tuner = self.model_tuners[model_type]
            best_params, best_loss = tuner.tune(X, n_trials, n_splits)
            
            return best_params, best_loss
            
        except Exception as e:
            self.logger.error(f"Error tuning model {model_id}: {e}")
            raise
            
    def update_model_params(
        self,
        params: Dict[str, Any],
        model_id: Optional[str] = None
    ):
        """
        Update parameters for a specific model or the active model.
        
        Args:
            params: New parameter values
            model_id: Optional model ID to update (if None, uses active model)
        """
        model_id = model_id or self.active_model_id
        model = self.model_factory.get_model(model_id)
        
        if model is None:
            raise ValueError(f"Model {model_id} not found")
            
        try:
            # Update model parameters
            for param, value in params.items():
                if hasattr(model, param):
                    setattr(model, param, value)
                    
            self.logger.info(f"Updated parameters for model {model_id}")
            
        except Exception as e:
            self.logger.error(f"Error updating model parameters: {e}")
            raise

    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        return list(self.model_factory.models.keys())
        
    def create_lstm_model(self, input_shape: tuple) -> Sequential:
        """Create a new LSTM model."""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
        
    def prepare_data(self, data: pd.DataFrame, sequence_length: int = 10) -> tuple:
        """Prepare data for LSTM model."""
        # Scale the data
        scaled_data = self.scaler.fit_transform(data[['close']].values)
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:(i + sequence_length)])
            y.append(scaled_data[i + sequence_length])
            
        return np.array(X), np.array(y)
        
    def train_model_lstm(self, model: Sequential, epochs: int = 100, batch_size: int = 32):
        """Train the specified LSTM model."""
        try:
            if not hasattr(self, 'training_data'):
                raise ValueError("No training data available")
                
            X, y = self.training_data
            
            # Train the model
            model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.1)
            
            self.logger.info(f"Model trained successfully for {epochs} epochs")
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            raise
            
    def tune_model_lstm(self, model: Sequential):
        """Tune the LSTM model hyperparameters."""
        try:
            # Implement hyperparameter tuning logic here
            # For now, just log that tuning was attempted
            self.logger.info("Model tuning attempted")
            
        except Exception as e:
            self.logger.error(f"Error tuning model: {e}")
            raise
            
    def predict_lstm(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Make predictions using the active LSTM model."""
        try:
            if self.active_model_id != 'lstm':
                raise ValueError("No active LSTM model selected")
                
            # Prepare data for prediction
            scaled_data = self.scaler.transform(data[['close']].values)
            
            # Create sequences for prediction
            X = []
            for i in range(len(scaled_data) - 10):  # Using sequence length of 10
                X.append(scaled_data[i:(i + 10)])
            X = np.array(X)
            
            # Make predictions
            predictions = self.model_factory.get_model('lstm').predict(X)
            
            # Inverse transform predictions
            predictions = self.scaler.inverse_transform(predictions)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {e}")
            return None
            
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform technical analysis on the data."""
        try:
            results = {
                'technical_indicators': self.calculate_technical_indicators(data),
                'trend_analysis': self.analyze_trend(data),
                'volatility_analysis': self.analyze_volatility(data)
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error performing analysis: {e}")
            return {}
            
    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical indicators."""
        try:
            indicators = {}
            
            # Calculate moving averages
            indicators['ma5'] = data['close'].rolling(window=5).mean().iloc[-1]
            indicators['ma20'] = data['close'].rolling(window=20).mean().iloc[-1]
            
            # Calculate RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs)).iloc[-1]
            
            # Calculate MACD
            exp1 = data['close'].ewm(span=12, adjust=False).mean()
            exp2 = data['close'].ewm(span=26, adjust=False).mean()
            indicators['macd'] = exp1.iloc[-1] - exp2.iloc[-1]
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}")
            return {}
            
    def analyze_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price trends."""
        try:
            # Calculate trend direction
            ma20 = data['close'].rolling(window=20).mean()
            current_price = data['close'].iloc[-1]
            
            trend = {
                'direction': 'up' if current_price > ma20.iloc[-1] else 'down',
                'strength': abs(current_price - ma20.iloc[-1]) / ma20.iloc[-1] * 100
            }
            
            return trend
            
        except Exception as e:
            self.logger.error(f"Error analyzing trend: {e}")
            return {}
            
    def analyze_volatility(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze price volatility."""
        try:
            # Calculate daily returns
            returns = data['close'].pct_change()
            
            volatility = {
                'daily_volatility': returns.std() * np.sqrt(252),  # Annualized
                'max_drawdown': (data['close'] / data['close'].cummax() - 1).min()
            }
            
            return volatility
            
        except Exception as e:
            self.logger.error(f"Error analyzing volatility: {e}")
            return {} 