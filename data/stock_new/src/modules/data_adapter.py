import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass

@dataclass
class TrainingData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    scaler: MinMaxScaler

class DataAdapter:
    def __init__(self, sequence_length: int = 10):
        """Initialize DataAdapter with sequence length for time series."""
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.features: List[str] = []
        self._feature_scalers: Dict[str, MinMaxScaler] = {}
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the dataset."""
        df = df.copy()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate Moving Averages
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['MA50'] = df['close'].rolling(window=50).mean()
        
        # Calculate MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Calculate Bollinger Bands
        df['BB_middle'] = df['close'].rolling(window=20).mean()
        df['BB_upper'] = df['BB_middle'] + 2 * df['close'].rolling(window=20).std()
        df['BB_lower'] = df['BB_middle'] - 2 * df['close'].rolling(window=20).std()
        
        return df
    
    def prepare_training_data(
        self,
        df: pd.DataFrame,
        features: List[str],
        validation_split: float = 0.2
    ) -> TrainingData:
        """Prepare sequences for training with validation split."""
        self.features = features
        
        # Calculate technical indicators if needed
        if any(indicator in features for indicator in ['RSI', 'MACD', 'MA20', 'MA50']):
            df = self.calculate_technical_indicators(df)
        
        # Extract features and scale
        data = df[features].values
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:i + self.sequence_length])
            y.append(scaled_data[i + self.sequence_length, features.index('close')])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split into training and validation sets
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        return TrainingData(X_train, y_train, X_val, y_val, self.scaler)
    
    def prepare_prediction_data(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None
    ) -> np.ndarray:
        """Prepare sequences for prediction."""
        features = features or self.features
        
        # Calculate technical indicators if needed
        if any(indicator in features for indicator in ['RSI', 'MACD', 'MA20', 'MA50']):
            df = self.calculate_technical_indicators(df)
        
        # Extract features and scale
        data = df[features].values
        scaled_data = self.scaler.transform(data)
        
        # Create sequences
        X = []
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:i + self.sequence_length])
        
        return np.array(X)
    
    def inverse_transform_predictions(
        self,
        predictions: np.ndarray,
        feature_index: int = 0
    ) -> np.ndarray:
        """Convert scaled predictions back to original scale."""
        # Create a dummy array with the same shape as the training data
        dummy = np.zeros((len(predictions), len(self.features)))
        dummy[:, feature_index] = predictions
        
        # Inverse transform
        return self.scaler.inverse_transform(dummy)[:, feature_index]
    
    def get_feature_names(self) -> List[str]:
        """Return list of available features including technical indicators."""
        return [
            'open', 'high', 'low', 'close', 'volume',
            'RSI', 'MACD', 'Signal_Line',
            'MA20', 'MA50',
            'BB_middle', 'BB_upper', 'BB_lower'
        ]
    
    def normalize_returns(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        method: str = 'min-max'
    ) -> Dict[str, float]:
        """Calculate normalized returns using different methods."""
        match method.lower():
            case 'min-max':
                norm_actual = (actual - actual.min()) / (actual.max() - actual.min())
                norm_predicted = (predicted - predicted.min()) / (predicted.max() - predicted.min())
            case 'z-score':
                norm_actual = (actual - actual.mean()) / actual.std()
                norm_predicted = (predicted - predicted.mean()) / predicted.std()
            case 'softmax':
                norm_actual = np.exp(actual) / np.sum(np.exp(actual))
                norm_predicted = np.exp(predicted) / np.sum(np.exp(predicted))
            case _:
                raise ValueError(f"Unsupported normalization method: {method}")
        
        return {
            'actual_norm': float(norm_actual.mean()),
            'predicted_norm': float(norm_predicted.mean()),
            'correlation': float(np.corrcoef(norm_actual, norm_predicted)[0, 1])
        } 