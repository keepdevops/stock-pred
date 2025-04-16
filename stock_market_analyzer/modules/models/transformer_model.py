import torch
import torch.nn as nn
import numpy as np
import math
from typing import Tuple, Optional, Dict, Any
import logging
import tensorflow as tf
from tensorflow.keras import layers, Model
import joblib

class StockTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 6,  # OHLCV + technical indicators
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        output_dim: int = 1
    ):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, seq_len, input_dim)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.output_proj(x[:, -1, :])  # Use last sequence output
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

class TransformerStockPredictor:
    def __init__(
        self,
        input_dim: int = 6,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.logger = logging.getLogger(__name__)
        self.device = device
        
        # Initialize model
        self.model = StockTransformer(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        ).to(device)
        
        # Initialize optimizer and loss function
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
    def prepare_data(
        self,
        data: np.ndarray,
        sequence_length: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare data for training/prediction."""
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(data[i + sequence_length, 4])  # Predict close price
            
        X = torch.FloatTensor(np.array(X)).to(self.device)
        y = torch.FloatTensor(np.array(y)).to(self.device)
        return X, y
        
    def train(
        self,
        data: np.ndarray,
        sequence_length: int = 10,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2
    ) -> dict:
        """Train the model."""
        self.model.train()
        
        # Prepare data
        X, y = self.prepare_data(data, sequence_length)
        
        # Split into train and validation sets
        train_size = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            total_train_loss = 0
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i + batch_size]
                batch_y = y_train[i:i + batch_size]
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs.squeeze(), batch_y)
                loss.backward()
                self.optimizer.step()
                
                total_train_loss += loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = self.criterion(val_outputs.squeeze(), y_val)
            
            avg_train_loss = total_train_loss / (len(X_train) / batch_size)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss.item())
            
            if (epoch + 1) % 10 == 0:
                self.logger.info(
                    f"Epoch [{epoch+1}/{epochs}], "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {val_loss.item():.4f}"
                )
        
        return history
        
    def predict(
        self,
        data: np.ndarray,
        sequence_length: int = 10
    ) -> np.ndarray:
        """Make predictions."""
        self.model.eval()
        
        # Prepare data
        X, _ = self.prepare_data(data, sequence_length)
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(X)
            predictions = predictions.cpu().numpy()
            
        return predictions
        
    def save_model(self, path: str):
        """Save the model to disk."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        
    def load_model(self, path: str):
        """Load the model from disk."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

class TransformerModel:
    def __init__(self, 
                 input_shape: Tuple[int, int],
                 num_heads: int = 4,
                 ff_dim: int = 32,
                 num_transformer_blocks: int = 2,
                 dropout_rate: float = 0.1,
                 **kwargs):
        """
        Initialize Transformer model for stock prediction.
        
        Args:
            input_shape: Shape of input data (sequence_length, num_features)
            num_heads: Number of attention heads
            ff_dim: Dimension of feed-forward network
            num_transformer_blocks: Number of transformer blocks
            dropout_rate: Dropout rate
            **kwargs: Additional model parameters
        """
        self.input_shape = input_shape
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.dropout_rate = dropout_rate
        self.logger = logging.getLogger(__name__)
        
        # Build model
        self.model = self._build_model()
        
    def _transformer_encoder(self, inputs: tf.Tensor) -> tf.Tensor:
        """Create a transformer encoder block."""
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.input_shape[1]
        )(inputs, inputs)
        
        # Add & normalize
        attention_output = layers.Dropout(self.dropout_rate)(attention_output)
        attention_output = layers.LayerNormalization(
            epsilon=1e-6
        )(inputs + attention_output)
        
        # Feed-forward network
        ffn_output = layers.Dense(self.ff_dim, activation='relu')(attention_output)
        ffn_output = layers.Dense(self.input_shape[1])(ffn_output)
        ffn_output = layers.Dropout(self.dropout_rate)(ffn_output)
        
        # Add & normalize
        return layers.LayerNormalization(
            epsilon=1e-6
        )(attention_output + ffn_output)
        
    def _build_model(self) -> Model:
        """Build the transformer model."""
        inputs = layers.Input(shape=self.input_shape)
        
        # Positional encoding
        x = layers.Dense(self.input_shape[1])(inputs)
        
        # Transformer blocks
        for _ in range(self.num_transformer_blocks):
            x = self._transformer_encoder(x)
            
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Output layer
        outputs = layers.Dense(1)(x)
        
        return Model(inputs=inputs, outputs=outputs)
        
    def compile(self, learning_rate: float = 0.001):
        """Compile the model."""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
    def fit(self, 
            X: np.ndarray, 
            y: np.ndarray,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            epochs: int = 100,
            batch_size: int = 32,
            callbacks: Optional[list] = None) -> Dict[str, list]:
        """
        Train the model.
        
        Args:
            X: Training features
            y: Training targets
            validation_data: Optional validation data
            epochs: Number of epochs
            batch_size: Batch size
            callbacks: Optional callbacks
            
        Returns:
            Dictionary containing training history
        """
        try:
            history = self.model.fit(
                X, y,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            self.logger.info("Model training completed successfully")
            return {
                'train_loss': history.history['loss'],
                'val_loss': history.history.get('val_loss', []),
                'train_mae': history.history['mae'],
                'val_mae': history.history.get('val_mae', [])
            }
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            return {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []}
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Array of predictions
        """
        try:
            return self.model.predict(X)
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            return np.array([])
            
    def save(self, path: str):
        """Save model to file."""
        try:
            self.model.save(path)
            self.logger.info(f"Model saved to {path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            
    def load(self, path: str):
        """Load model from file."""
        try:
            self.model = tf.keras.models.load_model(path)
            self.logger.info(f"Model loaded from {path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'input_shape': self.input_shape,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'num_transformer_blocks': self.num_transformer_blocks,
            'dropout_rate': self.dropout_rate
        } 