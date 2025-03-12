"""
Force Save Models with .keras Extension
This module provides a function to force-save models to ensure they use the .keras extension
"""
import os
import time
import pickle
import tensorflow as tf

def force_save_trained_model(model, history, ticker, model_type):
    """
    Force-save a trained model with the proper .keras extension
    
    Args:
        model: The trained TensorFlow model
        history: The training history object
        ticker: The stock ticker symbol
        model_type: The model type (LSTM, GRU, etc.)
        
    Returns:
        tuple: (model_path, history_path) if successful, (None, None) otherwise
    """
    print("\n===== FORCE SAVING MODEL WITH .KERAS EXTENSION =====")
    
    try:
        # Create timestamp for unique filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_filename = f"{ticker}_{model_type}_{timestamp}.keras"
        print(f"Generated filename: {model_filename}")
        
        # Get models directory
        models_dir = "/Users/moose/stock-pred/data/stock-analyzer/models"
        os.makedirs(models_dir, exist_ok=True)
        print(f"Models directory: {models_dir}")
        
        # Full paths
        model_path = os.path.join(models_dir, model_filename)
        history_path = model_path.replace('.keras', '_history.pkl')
        print(f"Model path: {model_path}")
        print(f"History path: {history_path}")
        
        # Save model
        print("Saving model...")
        model.save(model_path, save_format='keras')
        print(f"Model saved: {os.path.exists(model_path)}")
        
        # Save history
        print("Saving history...")
        history_data = history.history if hasattr(history, 'history') else history
        with open(history_path, 'wb') as f:
            pickle.dump(history_data, f)
        print(f"History saved: {os.path.exists(history_path)}")
        
        print("===== MODEL SAVED SUCCESSFULLY =====\n")
        return model_path, history_path
    
    except Exception as e:
        print(f"ERROR saving model: {e}")
        import traceback
        traceback.print_exc()
        print("===== MODEL SAVE FAILED =====\n")
        return None, None 