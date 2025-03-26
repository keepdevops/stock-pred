import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import logging

class LSTMStockPredictor(nn.Module):
    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        """
        Initialize LSTM model for stock prediction.
        
        Args:
            input_dim: Number of input features (OHLCV = 5)
            hidden_dim: Number of hidden units in LSTM layers
            num_layers: Number of LSTM layers
            dropout: Dropout rate for regularization
        """
        super(LSTMStockPredictor, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.logger = logging.getLogger(__name__)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Fully connected layer for prediction
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            hidden: Optional initial hidden state
            
        Returns:
            Tuple of (predictions, hidden_state)
        """
        # Initialize hidden state if not provided
        if hidden is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
            hidden = (h0, c0)
            
        # Forward pass through LSTM
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Get predictions from last time step
        predictions = self.fc(lstm_out[:, -1, :])
        
        return predictions, hidden
        
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions on numpy array input.
        
        Args:
            x: Input array of shape (sequence_length, input_dim)
            
        Returns:
            Numpy array of predictions
        """
        self.eval()  # Set model to evaluation mode
        
        try:
            with torch.no_grad():
                # Convert input to tensor and add batch dimension
                x = torch.FloatTensor(x).unsqueeze(0)
                
                # Make prediction
                predictions, _ = self.forward(x)
                
                # Convert to numpy array and remove batch dimension
                return predictions.numpy().squeeze()
                
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            return np.array([])
            
    def save(self, path: str):
        """Save model state to file."""
        try:
            torch.save(self.state_dict(), path)
            self.logger.info(f"Model saved to {path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            
    def load(self, path: str):
        """Load model state from file."""
        try:
            self.load_state_dict(torch.load(path))
            self.logger.info(f"Model loaded from {path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            
class LSTMTrainer:
    def __init__(self, model: LSTMStockPredictor, learning_rate: float = 0.001):
        """
        Initialize trainer for LSTM model.
        
        Args:
            model: LSTMStockPredictor instance
            learning_rate: Learning rate for optimization
        """
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.logger = logging.getLogger(__name__)
        
    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Perform single training step.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            y: Target tensor of shape (batch_size, 1)
            
        Returns:
            Loss value for this step
        """
        # Set model to training mode
        self.model.train()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        predictions, _ = self.model(x)
        
        # Calculate loss
        loss = self.criterion(predictions, y)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        self.optimizer.step()
        
        return loss.item()
        
    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Average loss for this epoch
        """
        total_loss = 0
        num_batches = len(train_loader)
        
        for x_batch, y_batch in train_loader:
            loss = self.train_step(x_batch, y_batch)
            total_loss += loss
            
        return total_loss / num_batches
        
    def evaluate(self, val_loader: torch.utils.data.DataLoader) -> float:
        """
        Evaluate model on validation data.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                predictions, _ = self.model(x_batch)
                loss = self.criterion(predictions, y_batch)
                total_loss += loss.item()
                
        return total_loss / num_batches 