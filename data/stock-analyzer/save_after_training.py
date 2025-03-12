"""
Force Save Trained Models with .keras Extension
This script adds a hook to save models after training
"""
import os
import time
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

class SaveModelCallback(Callback):
    """Callback to save model after training with proper .keras extension"""
    
    def __init__(self, ticker, model_type, models_dir=None):
        self.ticker = ticker
        self.model_type = model_type
        self.models_dir = models_dir or "/Users/moose/stock-pred/data/stock-analyzer/models"
        os.makedirs(self.models_dir, exist_ok=True)
        
    def on_train_end(self, logs=None):
        """Save model and history when training ends"""
        print("\n===== AUTOMATICALLY SAVING MODEL WITH .KERAS EXTENSION =====")
        
        try:
            # Generate filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_filename = f"{self.ticker}_{self.model_type}_{timestamp}.keras"
            
            # Get save paths
            model_path = os.path.join(self.models_dir, model_filename)
            history_path = model_path.replace('.keras', '_history.pkl')
            
            # Save model
            print(f"Saving model to: {model_path}")
            self.model.save(model_path, save_format='keras')
            print(f"Model saved successfully: {os.path.exists(model_path)}")
            
            # Save history
            print(f"Saving history to: {history_path}")
            with open(history_path, 'wb') as f:
                pickle.dump(self.model.history.history, f)
            print(f"History saved successfully: {os.path.exists(history_path)}")
            
            print(f"Model and history saved as: {model_filename}")
            print("===== SAVE COMPLETE =====\n")
            
        except Exception as e:
            print(f"ERROR saving model: {e}")
            import traceback
            traceback.print_exc()
            print("===== SAVE FAILED =====\n") 