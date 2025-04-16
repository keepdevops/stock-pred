import os
import json
import logging
import traceback
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from .lstm_model import LSTMModel
from .xgboost_model import XGBoostModel
from .transformer_model import TransformerModel
from .model_factory import ModelFactory
from .data_preprocessor import DataPreprocessor

class ModelManager:
    """Manager class for handling stock prediction models."""
    
    def __init__(self):
        """Initialize the model manager."""
        self.logger = logging.getLogger(__name__)
        self.model_factory = ModelFactory()
        self.data_preprocessor = DataPreprocessor()
        self.models: Dict[str, Dict[str, Any]] = {}
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
        
    def create_model(self, ticker: str, model_type: str, params: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new model for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            model_type: Type of model to create
            params: Optional model parameters
            
        Returns:
            Model ID
        """
        try:
            model_id = f"{ticker}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create model instance
            model = self.model_factory.create_model(model_type, params)
            
            # Store model
            self.models[model_id] = {
                'ticker': ticker,
                'type': model_type,
                'model': model,
                'params': params or {},
                'created_at': datetime.now(),
                'metrics': {}
            }
            
            self.logger.info(f"Created new {model_type} model for {ticker}")
            return model_id
            
        except Exception as e:
            self.logger.error(f"Error creating model: {e}")
            raise
            
    def train_model(self, ticker: str, model_type: str, epochs: int = 100, batch_size: int = 32) -> Dict[str, float]:
        """
        Train a model for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            model_type: Type of model to train
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Training metrics
        """
        try:
            # Get or create model
            model_id = self._get_model_id(ticker, model_type)
            if not model_id:
                model_id = self.create_model(ticker, model_type)
                
            model_data = self.models[model_id]
            model = model_data['model']
            
            # Load and preprocess data
            data = self.data_preprocessor.load_data(ticker)
            X_train, y_train, X_val, y_val = self.data_preprocessor.prepare_data(data)
            
            # Train model
            history = model.train(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val)
            )
            
            # Store metrics
            metrics = {
                'train_loss': history.history['loss'][-1],
                'val_loss': history.history['val_loss'][-1],
                'train_mae': history.history['mae'][-1],
                'val_mae': history.history['val_mae'][-1]
            }
            model_data['metrics'] = metrics
            
            # Save model
            model_path = os.path.join(self.model_dir, f"{model_id}.h5")
            model.save(model_path)
            
            self.logger.info(f"Trained {model_type} model for {ticker}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            raise
            
    def predict(self, ticker: str, model_type: str, days: int = 7) -> List[float]:
        """
        Make predictions with a trained model.
        
        Args:
            ticker: Stock ticker symbol
            model_type: Type of model to use
            days: Number of days to predict
            
        Returns:
            List of predicted values
        """
        try:
            model_id = self._get_model_id(ticker, model_type)
            if not model_id:
                raise ValueError(f"No trained model found for {ticker}")
                
            model_data = self.models[model_id]
            model = model_data['model']
            
            # Load and preprocess data
            data = self.data_preprocessor.load_data(ticker)
            X = self.data_preprocessor.prepare_prediction_data(data, days)
            
            # Make predictions
            predictions = model.predict(X)
            
            self.logger.info(f"Made predictions with {model_type} model for {ticker}")
            return predictions.tolist()
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {e}")
            raise
            
    def get_model_details(self, ticker: str, model_type: str) -> Dict[str, Any]:
        """
        Get details of a trained model.
        
        Args:
            ticker: Stock ticker symbol
            model_type: Type of model
            
        Returns:
            Model details
        """
        try:
            model_id = self._get_model_id(ticker, model_type)
            if not model_id:
                raise ValueError(f"No trained model found for {ticker}")
                
            model_data = self.models[model_id]
            return {
                'ticker': model_data['ticker'],
                'type': model_data['type'],
                'created_at': model_data['created_at'],
                'metrics': model_data['metrics'],
                'params': model_data['params']
            }
            
        except Exception as e:
            self.logger.error(f"Error getting model details: {e}")
            raise
            
    def get_trained_models(self) -> List[Dict[str, Any]]:
        """
        Get list of all trained models.
        
        Returns:
            List of model details
        """
        return [
            {
                'ticker': data['ticker'],
                'type': data['type'],
                'created_at': data['created_at']
            }
            for data in self.models.values()
        ]
        
    def _get_model_id(self, ticker: str, model_type: str) -> Optional[str]:
        """Get model ID for a ticker and model type."""
        for model_id, data in self.models.items():
            if data['ticker'] == ticker and data['type'] == model_type:
                return model_id
        return None
        
    def _load_models_metadata(self) -> Dict[str, Any]:
        """Load models metadata from file."""
        metadata_path = os.path.join(self.model_dir, "models_metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading models metadata: {e}")
        return {}
        
    def _save_models_metadata(self):
        """Save models metadata to file."""
        metadata_path = os.path.join(self.model_dir, "models_metadata.json")
        try:
            with open(metadata_path, 'w') as f:
                json.dump(self.models_metadata, f, indent=4)
        except Exception as e:
            self.logger.error(f"Error saving models metadata: {e}")
            
    def _get_model_path(self, ticker: str, model_type: str) -> str:
        """Get the path for a model file."""
        return os.path.join(self.model_dir, f"{ticker}_{model_type.replace(' ', '_')}.model")
        
    def _prepare_data(self, ticker: str, lookback: int = 60) -> Optional[tuple]:
        """Prepare data for model training."""
        try:
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback * 2)  # Get more data for training
            data = yf.download(ticker, start=start_date, end=end_date)
            
            if data.empty:
                self.logger.error(f"No data available for {ticker}")
                return None
                
            # Prepare features
            data['Returns'] = data['Close'].pct_change()
            data['Volatility'] = data['Returns'].rolling(window=5).std()
            data['Volume_Change'] = data['Volume'].pct_change()
            
            # Drop NaN values
            data = data.dropna()
            
            # Create sequences
            X = []
            y = []
            for i in range(len(data) - lookback):
                X.append(data[['Returns', 'Volatility', 'Volume_Change']].iloc[i:i+lookback].values)
                y.append(data['Close'].iloc[i+lookback])
                
            X = np.array(X)
            y = np.array(y)
            
            # Split into train and validation sets
            split = int(0.8 * len(X))
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]
            
            return X_train, X_val, y_train, y_val
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {e}")
            self.logger.error(traceback.format_exc())
            return None
            
    def remove_model(self, ticker: str, model_type: str):
        """Remove a trained model."""
        try:
            model_id = self._get_model_id(ticker, model_type)
            if not model_id:
                raise ValueError(f"No model found for {ticker} - {model_type}")
                
            # Remove model file
            model_path = self._get_model_path(ticker, model_type)
            if os.path.exists(model_path):
                os.remove(model_path)
                
            # Remove from metadata
            del self.models[model_id]
            self._save_models_metadata()
            
        except Exception as e:
            self.logger.error(f"Error removing model: {e}")
            self.logger.error(traceback.format_exc())
            raise
            
    def clear_models(self):
        """Remove all trained models."""
        try:
            # Remove all model files
            for model_id in list(self.models.keys()):
                ticker, model_type = model_id.split('_', 1)
                model_path = self._get_model_path(ticker, model_type)
                if os.path.exists(model_path):
                    os.remove(model_path)
                    
            # Clear metadata
            self.models = {}
            self._save_models_metadata()
            
        except Exception as e:
            self.logger.error(f"Error clearing models: {e}")
            self.logger.error(traceback.format_exc())
            raise 