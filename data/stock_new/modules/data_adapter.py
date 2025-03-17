import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List

class DataAdapter:
    def __init__(self, sequence_length: int = 10):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.features = []
    
    def prepare_training_data(self, df: pd.DataFrame, features: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for training."""
        self.features = features
        data = df[features].values
        
        # Scale data
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:i + self.sequence_length])
            y.append(scaled_data[i + self.sequence_length, 0])  # Predict 'close' price
            
        return np.array(X), np.array(y)
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI, MACD, and other technical indicators."""
        df = df.copy()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        
        return df 