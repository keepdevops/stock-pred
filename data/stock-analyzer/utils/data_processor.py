"""
Data processing utilities for stock price prediction
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def prepare_data(df, sequence_length=10, test_size=0.2, target_column='close'):
    """
    Prepare data for LSTM model training
    
    Args:
        df: DataFrame with stock data
        sequence_length: Number of previous days to use for prediction
        test_size: Portion of data to use for testing
        target_column: Column to predict (default is 'close' for closing price)
        
    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    # Make sure the target column is lowercase
    target_column = target_column.lower()
    
    # Convert column names to lowercase
    df.columns = [col.lower() for col in df.columns]
    
    # Make sure we have date and target columns
    required_columns = ['date', target_column]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame. Available columns: {df.columns.tolist()}")
    
    # Sort by date
    df = df.sort_values('date')
    
    # Select features
    feature_columns = []
    
    # Core price features
    price_columns = ['open', 'high', 'low', 'close']
    for col in price_columns:
        if col in df.columns:
            feature_columns.append(col)
    
    # Include volume if available
    if 'volume' in df.columns:
        feature_columns.append('volume')
    
    # Create feature set
    data = df[feature_columns].values
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Create sequences
    X, y = create_sequences(scaled_data, sequence_length, target_column_idx=feature_columns.index(target_column))
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    return X_train, X_test, y_train, y_test, scaler

def create_sequences(data, sequence_length, target_column_idx=3):
    """
    Create sequences for LSTM training
    
    Args:
        data: Scaled data
        sequence_length: Length of sequences
        target_column_idx: Index of target column in data (default is 3 for 'close')
        
    Returns:
        X, y arrays for training
    """
    X = []
    y = []
    
    for i in range(len(data) - sequence_length):
        # Get sequence of features
        seq = data[i:i + sequence_length]
        
        # Get target value (next day's closing price)
        target = data[i + sequence_length, target_column_idx]
        
        X.append(seq)
        y.append(target)
    
    return np.array(X), np.array(y)

def inverse_transform_predictions(predictions, scaler, feature_idx=3):
    """
    Transform predictions back to original scale
    
    Args:
        predictions: Scaled predictions
        scaler: Scaler used to scale the data
        feature_idx: Index of the feature being predicted (default is 3 for 'close')
        
    Returns:
        Predictions in original scale
    """
    # Create a dummy array for inverse transformation
    dummy = np.zeros((len(predictions), scaler.scale_.shape[0]))
    dummy[:, feature_idx] = predictions.flatten()
    
    # Inverse transform
    dummy_inverse = scaler.inverse_transform(dummy)
    
    # Return only the predicted values
    return dummy_inverse[:, feature_idx] 