import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
from .models.model_factory import ModelFactory
from .models.transformer_model import TransformerStockPredictor
from .models.lstm_model import LSTMStockPredictor
from .models.xgboost_model import XGBoostStockPredictor
from .models.model_tuner import ModelTuner

class AIAgent:
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.model_factory = ModelFactory()
        self.active_model_id = None
        
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
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1
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
        # Select relevant features
        features = ['open', 'high', 'low', 'close', 'volume']
        
        # Add technical indicators
        data = data.copy()
        
        # Calculate moving averages
        data['ma5'] = data['close'].rolling(window=5).mean()
        data['ma20'] = data['close'].rolling(window=20).mean()
        
        # Calculate RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
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