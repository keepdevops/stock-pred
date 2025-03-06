"""
Data preparation and transformation for model training and prediction
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import traceback
from utils.technical_indicators import calculate_technical_indicators

class DataAdapter:
    def __init__(self, sequence_length=10, features=None):
        self.sequence_length = sequence_length
        self.features = features or ['open', 'high', 'low', 'close', 'volume', 'ma20', 'ma50', 'rsi', 'macd']
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def prepare_training_data(self, data):
        """Prepare data for training by creating sequences"""
        try:
            print("\nPreparing training data...")
            df = data.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()

            # Calculate technical indicators
            df = calculate_technical_indicators(df)
            df = df[self.features].ffill().bfill().fillna(0)
            scaled_data = self.scaler.fit_transform(df)

            X, y = [], []
            for i in range(len(scaled_data) - self.sequence_length):
                X.append(scaled_data[i:i + self.sequence_length])
                y.append(scaled_data[i + self.sequence_length, 3])  # 'close' at index 3
            X, y = np.array(X), np.array(y)

            train_size = int(len(X) * 0.8)
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]

            print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
            return X_train, X_val, y_train, y_val
        except Exception as e:
            print(f"Error preparing training data: {e}")
            traceback.print_exc()
            return None, None, None, None

    def prepare_prediction_data(self, data):
        """Prepare data for prediction"""
        try:
            print("\nPreparing prediction data...")
            df = data.copy()
            df['date'] = pd.to_datetime(df['date'])
            
            # Sort by date
            if 'date' in df.columns:
                df = df.sort_values('date')
            
            # Calculate technical indicators
            df = calculate_technical_indicators(df)
            
            # Make sure we have all required features
            for feature in self.features:
                if feature not in df.columns and feature != 'date':
                    print(f"Warning: Feature '{feature}' not found in data. Adding zeros.")
                    df[feature] = 0
                    
            # Handle NaN values
            df = df[self.features].ffill().bfill().fillna(0)
            
            # Scale the data
            scaled_data = self.scaler.transform(df)

            # Create sequences
            X = []
            for i in range(len(scaled_data) - self.sequence_length + 1):
                X.append(scaled_data[i:i + self.sequence_length])
            return np.array(X) if X else None
            
        except Exception as e:
            print(f"Error preparing prediction data: {e}")
            traceback.print_exc()
            return None

    def inverse_transform_predictions(self, predictions):
        """Convert scaled predictions back to original scale"""
        try:
            dummy = np.zeros((len(predictions), len(self.features)))
            dummy[:, 3] = predictions.flatten()  # 'close' at index 3
            return self.scaler.inverse_transform(dummy)[:, 3]
        except Exception as e:
            print(f"Error inverse transforming predictions: {e}")
            return None
