import os
import pickle
import tkinter as tk
from tkinter import messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class Trainer:
    def train_model(self, ticker, model_type, X_train, y_train, X_test, y_test, epochs=50, batch_size=32, **kwargs):
        """Train a machine learning model and automatically save it
        
        Args:
            ticker (str): The ticker symbol for the data
            model_type (str): Type of model to train (LSTM, GRU, etc.)
            X_train: Training input data
            y_train: Training target data
            X_test: Testing input data
            y_test: Testing target data
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            **kwargs: Additional parameters for model creation
            
        Returns:
            tuple: (trained_model, training_history)
        """
        print(f"Training {model_type} model for {ticker}...")
        
        # Create the model
        model = self._create_model(model_type, X_train.shape[1:], **kwargs)
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # Auto-save the model
        saved_path = self._save_model(model, history.history, ticker, model_type)
        
        return model, history, saved_path

    def _save_model(self, model, history, ticker, model_type):
        """Automatically save the model and its training history
        
        Args:
            model: The trained model to save
            history: Training history dictionary
            ticker: Ticker symbol
            model_type: Type of model (LSTM, GRU, etc.)
            
        Returns:
            str: Path to the saved model
        """
        try:
            import os
            import time
            import pickle
            
            # Generate timestamp for unique filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_filename = f"{ticker}_{model_type}_{timestamp}.keras"
            
            # Determine models directory
            models_dir = getattr(self, 'models_dir', None)
            if not models_dir:
                # Try different paths
                if hasattr(self, 'config') and hasattr(self.config, 'models_dir'):
                    models_dir = self.config.models_dir
                else:
                    # Default path
                    models_dir = "/Users/moose/stock-pred/data/stock-analyzer/models"
                    
            # Make sure the directory exists
            os.makedirs(models_dir, exist_ok=True)
                    
            # Full paths for model and history
            model_path = os.path.join(models_dir, model_filename)
            history_path = model_path.replace('.keras', '_history.pkl')
            
            # Save the model
            print(f"Auto-saving model to: {model_path}")
            model.save(model_path, save_format='keras')
            
            # Save the history
            print(f"Saving training history to: {history_path}")
            with open(history_path, 'wb') as f:
                pickle.dump(history, f)
                
            print(f"Model and history successfully saved")
            return model_path
            
        except Exception as e:
            print(f"Error saving model: {e}")
            import traceback
            traceback.print_exc()
            return None 

def auto_save_model(model, history, ticker, model_type, models_dir=None):
    """Automatically save model and training history
    
    Args:
        model: The trained model
        history: Training history (from model.fit)
        ticker: Ticker symbol
        model_type: Type of model (LSTM, GRU, etc.)
        models_dir: Directory to save models (optional)
    
    Returns:
        str: Path of saved model
    """
    import os
    import time
    import pickle
    
    try:
        # Generate timestamp for filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_filename = f"{ticker}_{model_type}_{timestamp}.keras"
        
        # Set default models directory if not provided
        if not models_dir:
            models_dir = "/Users/moose/stock-pred/data/stock-analyzer/models"
            
        # Create directory if it doesn't exist
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            print(f"Created models directory: {models_dir}")
        
        # Full paths for model and history
        model_path = os.path.join(models_dir, model_filename)
        history_path = model_path.replace('.keras', '_history.pkl')
        
        # Save model
        print(f"Auto-saving model to: {model_path}")
        model.save(model_path, save_format='keras')
        
        # Save history
        if hasattr(history, 'history'):
            history_data = history.history
        else:
            history_data = history
            
        with open(history_path, 'wb') as f:
            pickle.dump(history_data, f)
        print(f"Training history saved to: {history_path}")
        
        return model_path
    
    except Exception as e:
        print(f"Error auto-saving model: {e}")
        import traceback
        traceback.print_exc()
        return None 