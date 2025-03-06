"""
Factory for creating different types of models
"""
import os
import json
import tensorflow as tf
from models.lstm_model import create_lstm_model
from models.gru_model import create_gru_model
from models.bilstm_model import create_bilstm_model
from models.cnn_lstm_model import create_cnn_lstm_model

class ModelFactory:
    @staticmethod
    def create_model(model_type, input_shape, **kwargs):
        """Create a model based on the specified type"""
        if model_type == "LSTM":
            return create_lstm_model(input_shape, **kwargs)
        elif model_type == "GRU":
            return create_gru_model(input_shape, **kwargs)
        elif model_type == "BiLSTM":
            return create_bilstm_model(input_shape, **kwargs)
        elif model_type == "CNN-LSTM":
            return create_cnn_lstm_model(input_shape, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def save_model(model, path, metadata=None):
        """Save model and metadata"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model
        model.save(path)
        
        # Save metadata if provided
        if metadata:
            metadata_path = f"{path}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
    
    @staticmethod
    def load_model(path):
        """Load model and metadata"""
        # Load the model
        model = tf.keras.models.load_model(path)
        
        # Load metadata if it exists
        metadata = None
        metadata_path = f"{path}_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return model, metadata
