import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow import keras
import os

@dataclass
class StrategyTrainingData:
    """Represents training data generated from trading strategies."""
    features: np.ndarray
    labels: np.ndarray
    strategy_name: str
    performance_metrics: Dict[str, float]

class StrategyTrainer:
    """Trains models using data from trading strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the strategy trainer.
        
        Args:
            config: Configuration dictionary containing training parameters
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.training_data = []
        self.models = {}
        
    def prepare_training_data(self, strategy_results: pd.DataFrame, strategy_name: str) -> StrategyTrainingData:
        """Prepare training data from strategy results.
        
        Args:
            strategy_results: DataFrame containing strategy backtest results
            strategy_name: Name of the strategy
            
        Returns:
            StrategyTrainingData object containing features and labels
        """
        try:
            # Calculate features
            features = self._calculate_features(strategy_results)
            
            # Calculate labels (future returns)
            labels = self._calculate_labels(strategy_results)
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(strategy_results)
            
            return StrategyTrainingData(
                features=features,
                labels=labels,
                strategy_name=strategy_name,
                performance_metrics=metrics
            )
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {e}")
            raise
            
    def _calculate_features(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate features from strategy results.
        
        Args:
            data: DataFrame containing strategy results
            
        Returns:
            Array of features
        """
        features = []
        
        # Technical indicators
        features.extend([
            data['close'].pct_change().values,
            data['volume'].pct_change().values,
            data['position'].values,
            data['returns'].values
        ])
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            ma = data['close'].rolling(window=window).mean()
            features.append(ma.values)
            
        # Volatility
        volatility = data['close'].rolling(window=20).std()
        features.append(volatility.values)
        
        # Stack features
        features = np.column_stack(features)
        
        # Handle NaN values
        features = np.nan_to_num(features, nan=0)
        
        return features
        
    def _calculate_labels(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate labels (future returns) from strategy results.
        
        Args:
            data: DataFrame containing strategy results
            
        Returns:
            Array of labels
        """
        # Calculate future returns (next day)
        future_returns = data['close'].shift(-1).pct_change().values
        
        # Handle NaN values
        future_returns = np.nan_to_num(future_returns, nan=0)
        
        return future_returns
        
    def _calculate_performance_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics for the strategy.
        
        Args:
            data: DataFrame containing strategy results
            
        Returns:
            Dictionary of performance metrics
        """
        # Calculate total return
        total_return = (data['portfolio_value'].iloc[-1] / self.config['initial_capital']) - 1
        
        # Calculate Sharpe ratio
        risk_free_rate = self.config.get('risk_free_rate', 0.02)
        excess_returns = data['returns'] - (risk_free_rate / 252)
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        
        # Calculate maximum drawdown
        cumulative_returns = (1 + data['returns']).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        # Calculate win rate
        winning_trades = (data['returns'] > 0).sum()
        total_trades = (data['position'] != 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate
        }
        
    def train_model(self, training_data: StrategyTrainingData) -> keras.Model:
        """Train a model using strategy training data.
        
        Args:
            training_data: StrategyTrainingData object containing features and labels
            
        Returns:
            Trained Keras model
        """
        try:
            # Create model architecture
            model = self._create_model(input_shape=(training_data.features.shape[1],))
            
            # Prepare data
            X = training_data.features
            y = training_data.labels
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train model
            model.fit(
                X_train, y_train,
                batch_size=self.config.get('batch_size', 32),
                epochs=self.config.get('epochs', 100),
                validation_data=(X_test, y_test),
                verbose=1
            )
            
            # Store model
            self.models[training_data.strategy_name] = model
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            raise
            
    def _create_model(self, input_shape: Tuple[int, ...]) -> keras.Model:
        """Create model architecture.
        
        Args:
            input_shape: Shape of input features
            
        Returns:
            Keras model
        """
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=input_shape),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.get('learning_rate', 0.001)),
            loss='mse',
            metrics=['mae']
        )
        
        return model
        
    def predict(self, strategy_name: str, features: np.ndarray) -> np.ndarray:
        """Make predictions using a trained model.
        
        Args:
            strategy_name: Name of the strategy
            features: Array of features
            
        Returns:
            Array of predictions
        """
        if strategy_name not in self.models:
            raise ValueError(f"No trained model found for strategy: {strategy_name}")
            
        model = self.models[strategy_name]
        return model.predict(features)
        
    def save_models(self, directory: str) -> None:
        """Save trained models to disk.
        
        Args:
            directory: Directory to save models
        """
        for strategy_name, model in self.models.items():
            model_path = f"{directory}/{strategy_name}_model.h5"
            model.save(model_path)
            self.logger.info(f"Saved model for strategy {strategy_name} to {model_path}")
            
    def load_models(self, directory: str) -> None:
        """Load trained models from disk.
        
        Args:
            directory: Directory containing saved models
        """
        for strategy_name in self.models.keys():
            model_path = f"{directory}/{strategy_name}_model.h5"
            if os.path.exists(model_path):
                self.models[strategy_name] = keras.models.load_model(model_path)
                self.logger.info(f"Loaded model for strategy {strategy_name} from {model_path}") 