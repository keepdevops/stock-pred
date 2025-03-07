"""
LSTM model creation and training utilities
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

def create_lstm_model(input_shape, neurons=50, dropout=0.2, layers=2):
    """
    Create an LSTM model for stock price prediction using Keras recommendations
    
    Args:
        input_shape: Shape of input data (sequence_length, features)
        neurons: Number of neurons in LSTM layers
        dropout: Dropout rate
        layers: Number of LSTM layers
        
    Returns:
        Compiled Keras model
    """
    # Use the recommended approach with Input layer
    inputs = Input(shape=input_shape)
    
    # First LSTM layer
    if layers == 1:
        x = LSTM(neurons)(inputs)
        x = Dropout(dropout)(x)
    else:
        x = LSTM(neurons, return_sequences=True)(inputs)
        x = Dropout(dropout)(x)
        
        # Middle LSTM layers
        for i in range(layers - 2):
            x = LSTM(neurons, return_sequences=True)(x)
            x = Dropout(dropout)(x)
        
        # Last LSTM layer
        x = LSTM(neurons)(x)
        x = Dropout(dropout)(x)
    
    # Output layer
    outputs = Dense(1)(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, learning_rate=0.001):
    """
    Train the LSTM model
    
    Args:
        model: Keras model to train
        X_train: Training features
        y_train: Training target values
        X_val: Validation features
        y_val: Validation target values
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for Adam optimizer
        
    Returns:
        Training history
    """
    # Compile model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    # Early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )
    
    return history

def plot_training_history(history):
    """
    Plot training and validation loss
    
    Args:
        history: Training history from model.fit()
    """
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    return plt.gcf()  # Return the current figure

def predict_future_prices(model, last_sequence, scaler, days_to_predict=30):
    """
    Predict future stock prices
    
    Args:
        model: Trained Keras model
        last_sequence: Last sequence of data (normalized)
        scaler: The scaler used to normalize the data
        days_to_predict: Number of days to predict
        
    Returns:
        Array of predicted prices
    """
    # Make a copy of the last sequence
    curr_sequence = last_sequence.copy()
    
    # List to store predictions
    predicted_prices = []
    
    # Predict future prices
    for _ in range(days_to_predict):
        # Get prediction for next day (scaled)
        pred = model.predict(curr_sequence.reshape(1, curr_sequence.shape[0], curr_sequence.shape[1]), verbose=0)
        
        # Add prediction to the list
        predicted_prices.append(pred[0][0])
        
        # Update sequence with the new prediction
        # Create a new row with the prediction and zeros for other features
        new_row = np.zeros((1, curr_sequence.shape[1]))
        new_row[0, 0] = pred[0][0]  # Assuming first column is the closing price
        
        # Remove first row and add the new prediction at the end
        curr_sequence = np.vstack((curr_sequence[1:], new_row))
    
    # Convert scaled predictions back to original scale
    predicted_prices = np.array(predicted_prices).reshape(-1, 1)
    
    # If using MinMaxScaler, we need to create an array with the right shape for inverse_transform
    if hasattr(scaler, 'data_min_'):  # Check if it's a MinMaxScaler or StandardScaler
        # For MinMaxScaler
        dummy_array = np.zeros((len(predicted_prices), scaler.scale_.shape[0]))
        dummy_array[:, 0] = predicted_prices.flatten()
        predicted_prices = scaler.inverse_transform(dummy_array)[:, 0]
    else:
        # For StandardScaler or other scalers with direct inverse_transform
        predicted_prices = scaler.inverse_transform(predicted_prices).flatten()
    
    return predicted_prices
