"""
LSTM model implementation
"""
import tensorflow as tf

def create_lstm_model(input_shape, neurons=50, num_layers=2, dropout_rate=0.2):
    """Create an LSTM model with the specified architecture"""
    model = tf.keras.Sequential()
    
    # First layer with explicit Input shape
    model.add(tf.keras.layers.Input(shape=input_shape))
    model.add(tf.keras.layers.LSTM(neurons, return_sequences=(num_layers > 1)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(dropout_rate))
    
    # Additional LSTM layers if requested
    for i in range(1, num_layers):
        return_sequences = i < num_layers - 1  # Only last layer doesn't return sequences
        model.add(tf.keras.layers.LSTM(neurons, return_sequences=return_sequences))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(dropout_rate))
    
    # Output layer
    model.add(tf.keras.layers.Dense(25, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    
    return model
