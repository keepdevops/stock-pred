import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_networks import MLPRegressor
from sklearn.svm import SVR
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, MultiHeadAttention, LayerNormalization
import tensorflow as tf
import logging

class BaseMLModel:
    """Base class for all ML models."""
    def __init__(self, name):
        self.name = name
        self.model = None
        self.scaler = MinMaxScaler()
        self.logger = logging.getLogger(__name__)
        
    def prepare_data(self, data: pd.DataFrame, target_col='close', sequence_length=10):
        """Prepare data for training/prediction."""
        try:
            # Create features
            features = ['open', 'high', 'low', 'close', 'volume']
            X = data[features].values
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Create sequences
            X_sequences = []
            y = []
            
            for i in range(len(X_scaled) - sequence_length):
                X_sequences.append(X_scaled[i:(i + sequence_length)])
                y.append(X_scaled[i + sequence_length, features.index(target_col)])
                
            return np.array(X_sequences), np.array(y)
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {e}")
            raise
            
    def train(self, data: pd.DataFrame, epochs=100, batch_size=32):
        """Train the model."""
        raise NotImplementedError
        
    def predict(self, data: pd.DataFrame, sequence_length=10):
        """Make predictions."""
        raise NotImplementedError
        
    def evaluate(self, data: pd.DataFrame):
        """Evaluate model performance."""
        raise NotImplementedError

class LSTMModel(BaseMLModel):
    """LSTM model for time series prediction."""
    def __init__(self):
        super().__init__("LSTM")
        self.sequence_length = 10
        
    def build_model(self, input_shape):
        """Build LSTM model architecture."""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
        
    def train(self, data: pd.DataFrame, epochs=100, batch_size=32):
        """Train the LSTM model."""
        try:
            X, y = self.prepare_data(data, sequence_length=self.sequence_length)
            self.model = self.build_model((X.shape[1], X.shape[2]))
            self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.1)
            return True
        except Exception as e:
            self.logger.error(f"Error training LSTM model: {e}")
            return False
            
    def predict(self, data: pd.DataFrame):
        """Make predictions using the LSTM model."""
        try:
            X, _ = self.prepare_data(data, sequence_length=self.sequence_length)
            predictions = self.model.predict(X)
            return self.scaler.inverse_transform(predictions)
        except Exception as e:
            self.logger.error(f"Error making predictions: {e}")
            return None
            
    def evaluate(self, data: pd.DataFrame):
        """Evaluate LSTM model performance."""
        try:
            X, y = self.prepare_data(data, sequence_length=self.sequence_length)
            loss = self.model.evaluate(X, y)
            return {'loss': loss}
        except Exception as e:
            self.logger.error(f"Error evaluating model: {e}")
            return None

class RandomForestModel(BaseMLModel):
    """Random Forest model for time series prediction."""
    def __init__(self):
        super().__init__("Random Forest")
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    def train(self, data: pd.DataFrame, epochs=None, batch_size=None):
        """Train the Random Forest model."""
        try:
            X, y = self.prepare_data(data)
            self.model.fit(X.reshape(X.shape[0], -1), y)
            return True
        except Exception as e:
            self.logger.error(f"Error training Random Forest model: {e}")
            return False
            
    def predict(self, data: pd.DataFrame):
        """Make predictions using the Random Forest model."""
        try:
            X, _ = self.prepare_data(data)
            predictions = self.model.predict(X.reshape(X.shape[0], -1))
            return self.scaler.inverse_transform(predictions.reshape(-1, 1))
        except Exception as e:
            self.logger.error(f"Error making predictions: {e}")
            return None
            
    def evaluate(self, data: pd.DataFrame):
        """Evaluate Random Forest model performance."""
        try:
            X, y = self.prepare_data(data)
            score = self.model.score(X.reshape(X.shape[0], -1), y)
            return {'score': score}
        except Exception as e:
            self.logger.error(f"Error evaluating model: {e}")
            return None

class GradientBoostingModel(BaseMLModel):
    """Gradient Boosting model for time series prediction."""
    def __init__(self):
        super().__init__("Gradient Boosting")
        self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        
    def train(self, data: pd.DataFrame, epochs=None, batch_size=None):
        """Train the Gradient Boosting model."""
        try:
            X, y = self.prepare_data(data)
            self.model.fit(X.reshape(X.shape[0], -1), y)
            return True
        except Exception as e:
            self.logger.error(f"Error training Gradient Boosting model: {e}")
            return False
            
    def predict(self, data: pd.DataFrame):
        """Make predictions using the Gradient Boosting model."""
        try:
            X, _ = self.prepare_data(data)
            predictions = self.model.predict(X.reshape(X.shape[0], -1))
            return self.scaler.inverse_transform(predictions.reshape(-1, 1))
        except Exception as e:
            self.logger.error(f"Error making predictions: {e}")
            return None
            
    def evaluate(self, data: pd.DataFrame):
        """Evaluate Gradient Boosting model performance."""
        try:
            X, y = self.prepare_data(data)
            score = self.model.score(X.reshape(X.shape[0], -1), y)
            return {'score': score}
        except Exception as e:
            self.logger.error(f"Error evaluating model: {e}")
            return None

class SVMModel(BaseMLModel):
    """Support Vector Machine model for time series prediction."""
    def __init__(self):
        super().__init__("SVM")
        self.model = SVR(kernel='rbf')
        
    def train(self, data: pd.DataFrame, epochs=None, batch_size=None):
        """Train the SVM model."""
        try:
            X, y = self.prepare_data(data)
            self.model.fit(X.reshape(X.shape[0], -1), y)
            return True
        except Exception as e:
            self.logger.error(f"Error training SVM model: {e}")
            return False
            
    def predict(self, data: pd.DataFrame):
        """Make predictions using the SVM model."""
        try:
            X, _ = self.prepare_data(data)
            predictions = self.model.predict(X.reshape(X.shape[0], -1))
            return self.scaler.inverse_transform(predictions.reshape(-1, 1))
        except Exception as e:
            self.logger.error(f"Error making predictions: {e}")
            return None
            
    def evaluate(self, data: pd.DataFrame):
        """Evaluate SVM model performance."""
        try:
            X, y = self.prepare_data(data)
            score = self.model.score(X.reshape(X.shape[0], -1), y)
            return {'score': score}
        except Exception as e:
            self.logger.error(f"Error evaluating model: {e}")
            return None

class TransformerModel(BaseMLModel):
    """Transformer model for time series prediction."""
    def __init__(self):
        super().__init__("Transformer")
        self.sequence_length = 10
        self.d_model = 64
        self.num_heads = 8
        self.ff_dim = 256
        self.num_transformer_blocks = 4
        self.mlp_units = [128, 64]
        self.dropout = 0.1
        self.mlp_dropout = 0.2
        
    def transformer_encoder(self, inputs):
        """Create a transformer encoder."""
        # Normalization and Attention
        x = LayerNormalization(epsilon=1e-6)(inputs)
        attention_output = MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.d_model, dropout=self.dropout
        )(x, x)
        x = attention_output + inputs
        
        # Feed Forward Network
        x = LayerNormalization(epsilon=1e-6)(x)
        x = Dense(self.ff_dim, activation="relu")(x)
        x = Dropout(self.dropout)(x)
        x = Dense(self.d_model)(x)
        return x + inputs
        
    def build_model(self, input_shape):
        """Build transformer model architecture."""
        inputs = Input(shape=input_shape)
        
        # Expand dimension for attention
        x = Dense(self.d_model)(inputs)
        
        # Transformer blocks
        for _ in range(self.num_transformer_blocks):
            x = self.transformer_encoder(x)
            
        # Global average pooling
        x = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        
        # MLP layers
        for dim in self.mlp_units:
            x = Dense(dim, activation="relu")(x)
            x = Dropout(self.mlp_dropout)(x)
            
        # Output layer
        outputs = Dense(1)(x)
        
        model = Model(inputs, outputs)
        model.compile(
            optimizer="adam",
            loss="mse",
            metrics=["mae"]
        )
        return model
        
    def train(self, data: pd.DataFrame, epochs=100, batch_size=32):
        """Train the transformer model."""
        try:
            X, y = self.prepare_data(data, sequence_length=self.sequence_length)
            self.model = self.build_model((X.shape[1], X.shape[2]))
            
            # Add early stopping and learning rate scheduling
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    patience=10,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                )
            ]
            
            history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.1,
                callbacks=callbacks,
                verbose=1
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Error training transformer model: {e}")
            return False
            
    def predict(self, data: pd.DataFrame):
        """Make predictions using the transformer model."""
        try:
            X, _ = self.prepare_data(data, sequence_length=self.sequence_length)
            predictions = self.model.predict(X)
            return self.scaler.inverse_transform(predictions)
        except Exception as e:
            self.logger.error(f"Error making predictions: {e}")
            return None
            
    def evaluate(self, data: pd.DataFrame):
        """Evaluate transformer model performance."""
        try:
            X, y = self.prepare_data(data, sequence_length=self.sequence_length)
            results = self.model.evaluate(X, y)
            return {
                'loss': results[0],
                'mae': results[1]
            }
        except Exception as e:
            self.logger.error(f"Error evaluating model: {e}")
            return None

class MLModelManager:
    """Manager class for ML models."""
    def __init__(self):
        self.models = {
            'LSTM': LSTMModel(),
            'Random Forest': RandomForestModel(),
            'Gradient Boosting': GradientBoostingModel(),
            'SVM': SVMModel(),
            'Transformer': TransformerModel()
        }
        self.active_model = None
        self.logger = logging.getLogger(__name__)
        
    def get_available_models(self):
        """Get list of available models."""
        return list(self.models.keys())
        
    def set_active_model(self, model_name: str):
        """Set the active model."""
        if model_name in self.models:
            self.active_model = self.models[model_name]
            return True
        return False
        
    def get_active_model(self):
        """Get the currently active model."""
        return self.active_model
        
    def train_model(self, data: pd.DataFrame, epochs=100, batch_size=32):
        """Train the active model."""
        if self.active_model is None:
            raise ValueError("No active model selected")
        return self.active_model.train(data, epochs, batch_size)
        
    def predict(self, data: pd.DataFrame):
        """Make predictions using the active model."""
        if self.active_model is None:
            raise ValueError("No active model selected")
        return self.active_model.predict(data)
        
    def evaluate_model(self, data: pd.DataFrame):
        """Evaluate the active model."""
        if self.active_model is None:
            raise ValueError("No active model selected")
        return self.active_model.evaluate(data) 