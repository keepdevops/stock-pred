import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import logging

class LSTMModel:
    """LSTM model for stock price prediction."""
    
    def __init__(self, input_shape=(60, 5), units=50, dropout=0.2, learning_rate=0.001):
        """Initialize the LSTM model."""
        self.logger = logging.getLogger(__name__)
        self.input_shape = input_shape
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.model = self._build_model()
        
    def _build_model(self):
        """Build the LSTM model."""
        model = Sequential([
            LSTM(units=self.units, return_sequences=True, input_shape=self.input_shape),
            Dropout(self.dropout),
            LSTM(units=self.units, return_sequences=False),
            Dropout(self.dropout),
            Dense(units=1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        return model
        
    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.1):
        """Train the LSTM model."""
        try:
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=1
            )
            return history
            
        except Exception as e:
            self.logger.error(f"Error training LSTM model: {str(e)}")
            raise
            
    def predict(self, X):
        """Make predictions using the trained model."""
        try:
            return self.model.predict(X)
            
        except Exception as e:
            self.logger.error(f"Error making predictions with LSTM model: {str(e)}")
            raise
            
    def save(self, filepath):
        """Save the model to a file."""
        try:
            self.model.save(filepath)
            self.logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving LSTM model: {str(e)}")
            raise
            
    def load(self, filepath):
        """Load the model from a file."""
        try:
            self.model = tf.keras.models.load_model(filepath)
            self.logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading LSTM model: {str(e)}")
            raise 