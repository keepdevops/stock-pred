"""
CNN-LSTM model implementation
"""
import tensorflow as tf

def create_cnn_lstm_model(input_shape, filters=64, kernel_size=3, lstm_units=50, dropout_rate=0.2):
    """Create a CNN-LSTM model with the specified architecture"""
    model = tf.keras.Sequential()
    
    # CNN layers with explicit Input shape
    model.add(tf.keras.layers.Input(shape=input_shape))
    model.add(tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    
    # LSTM layer
    model.add(tf.keras.layers.LSTM(lstm_units))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(dropout_rate))
    
    # Output layer
    model.add(tf.keras.layers.Dense(25, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    
    return model
