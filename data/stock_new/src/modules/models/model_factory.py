from typing import Dict, Any, Tuple
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Input, Bidirectional,
    Conv1D, MaxPooling1D, Dropout, LayerNormalization,
    MultiHeadAttention
)

class ModelFactory:
    """Factory class for creating different types of neural network models."""
    
    @staticmethod
    def create_model(
        model_type: str,
        input_shape: Tuple[int, int],
        config: Dict[str, Any]
    ) -> Model:
        """Create and return a specified model type."""
        creators = {
            'LSTM': ModelFactory.create_lstm_model,
            'GRU': ModelFactory.create_gru_model,
            'BiLSTM': ModelFactory.create_bilstm_model,
            'CNN-LSTM': ModelFactory.create_cnn_lstm_model,
            'Transformer': ModelFactory.create_transformer_model
        }
        
        if model_type not in creators:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return creators[model_type](input_shape, config)
    
    @staticmethod
    def create_lstm_model(
        input_shape: Tuple[int, int],
        config: Dict[str, Any]
    ) -> Model:
        """Create LSTM model."""
        inputs = Input(shape=input_shape)
        x = inputs
        
        # Add LSTM layers
        for i, units in enumerate(config.get('lstm_units', [50, 50])):
            return_sequences = i < len(config.get('lstm_units', [50, 50])) - 1
            x = LSTM(
                units=units,
                return_sequences=return_sequences,
                dropout=config.get('dropout', 0.2)
            )(x)
        
        # Add Dense layers
        for units in config.get('dense_units', [32]):
            x = Dense(units=units, activation='relu')(x)
            x = Dropout(config.get('dropout', 0.2))(x)
        
        outputs = Dense(units=1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=config.get('optimizer', 'adam'),
            loss=config.get('loss', 'mse')
        )
        
        return model
    
    @staticmethod
    def create_gru_model(
        input_shape: Tuple[int, int],
        config: Dict[str, Any]
    ) -> Model:
        """Create GRU model."""
        inputs = Input(shape=input_shape)
        x = inputs
        
        # Add GRU layers
        for i, units in enumerate(config.get('gru_units', [50, 50])):
            return_sequences = i < len(config.get('gru_units', [50, 50])) - 1
            x = GRU(
                units=units,
                return_sequences=return_sequences,
                dropout=config.get('dropout', 0.2)
            )(x)
        
        # Add Dense layers
        for units in config.get('dense_units', [32]):
            x = Dense(units=units, activation='relu')(x)
            x = Dropout(config.get('dropout', 0.2))(x)
        
        outputs = Dense(units=1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=config.get('optimizer', 'adam'),
            loss=config.get('loss', 'mse')
        )
        
        return model
    
    @staticmethod
    def create_bilstm_model(
        input_shape: Tuple[int, int],
        config: Dict[str, Any]
    ) -> Model:
        """Create Bidirectional LSTM model."""
        inputs = Input(shape=input_shape)
        x = inputs
        
        # Add Bidirectional LSTM layers
        for i, units in enumerate(config.get('lstm_units', [50, 50])):
            return_sequences = i < len(config.get('lstm_units', [50, 50])) - 1
            x = Bidirectional(
                LSTM(
                    units=units,
                    return_sequences=return_sequences,
                    dropout=config.get('dropout', 0.2)
                )
            )(x)
        
        # Add Dense layers
        for units in config.get('dense_units', [32]):
            x = Dense(units=units, activation='relu')(x)
            x = Dropout(config.get('dropout', 0.2))(x)
        
        outputs = Dense(units=1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=config.get('optimizer', 'adam'),
            loss=config.get('loss', 'mse')
        )
        
        return model
    
    @staticmethod
    def create_cnn_lstm_model(
        input_shape: Tuple[int, int],
        config: Dict[str, Any]
    ) -> Model:
        """Create CNN-LSTM hybrid model."""
        inputs = Input(shape=input_shape)
        x = inputs
        
        # Add CNN layers
        for filters in config.get('cnn_filters', [64, 32]):
            x = Conv1D(
                filters=filters,
                kernel_size=config.get('kernel_size', 3),
                activation='relu',
                padding='same'
            )(x)
            x = MaxPooling1D(pool_size=2)(x)
        
        # Add LSTM layers
        for i, units in enumerate(config.get('lstm_units', [50])):
            return_sequences = i < len(config.get('lstm_units', [50])) - 1
            x = LSTM(
                units=units,
                return_sequences=return_sequences,
                dropout=config.get('dropout', 0.2)
            )(x)
        
        # Add Dense layers
        for units in config.get('dense_units', [32]):
            x = Dense(units=units, activation='relu')(x)
            x = Dropout(config.get('dropout', 0.2))(x)
        
        outputs = Dense(units=1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=config.get('optimizer', 'adam'),
            loss=config.get('loss', 'mse')
        )
        
        return model
    
    @staticmethod
    def create_transformer_model(
        input_shape: Tuple[int, int],
        config: Dict[str, Any]
    ) -> Model:
        """Create Transformer model."""
        inputs = Input(shape=input_shape)
        x = inputs
        
        # Add positional encoding (simplified version)
        x = Dense(config.get('d_model', 64))(x)  # Project to d_model dimensions
        
        # Add Transformer layers
        for _ in range(config.get('num_layers', 2)):
            # Multi-head attention
            attention_output = MultiHeadAttention(
                num_heads=config.get('num_heads', 4),
                key_dim=config.get('d_model', 64) // config.get('num_heads', 4)
            )(x, x)
            x = LayerNormalization(epsilon=1e-6)(x + attention_output)
            
            # Feed-forward network
            ffn = Dense(config.get('d_model', 64) * 4, activation='relu')(x)
            ffn = Dense(config.get('d_model', 64))(ffn)
            x = LayerNormalization(epsilon=1e-6)(x + ffn)
        
        # Global average pooling
        x = tf.reduce_mean(x, axis=1)
        
        # Add Dense layers
        for units in config.get('dense_units', [32]):
            x = Dense(units=units, activation='relu')(x)
            x = Dropout(config.get('dropout', 0.2))(x)
        
        outputs = Dense(units=1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=config.get('optimizer', 'adam'),
            loss=config.get('loss', 'mse')
        )
        
        return model 