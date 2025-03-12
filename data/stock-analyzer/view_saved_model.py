"""
View Saved Model - Utility to load and view saved models
"""
import os
import glob
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

MODELS_DIR = "/Users/moose/stock-pred/data/stock-analyzer/models"

def list_all_saved_models():
    """List all saved model files with .keras extension"""
    pattern = os.path.join(MODELS_DIR, "*_model.keras")
    model_files = glob.glob(pattern)
    
    print(f"Found {len(model_files)} models:")
    for model_file in model_files:
        print(f"  - {os.path.basename(model_file)}")
    
    return model_files

def view_model(ticker, model_type="lstm"):
    """View a specific model by ticker and type"""
    model_filename = f"{ticker}_{model_type}_model.keras"
    model_path = os.path.join(MODELS_DIR, model_filename)
    history_path = model_path.replace(".keras", "_history.pkl")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found: {model_filename}")
        print("Checking for alternative files...")
        
        # Try finding any model with this ticker
        pattern = os.path.join(MODELS_DIR, f"{ticker}_*.keras")
        matches = glob.glob(pattern)
        if matches:
            model_path = matches[0]
            history_path = model_path.replace(".keras", "_history.pkl")
            print(f"Found alternative model: {os.path.basename(model_path)}")
        else:
            print(f"No models found for ticker: {ticker}")
            return
    
    # Load the model
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Successfully loaded model from: {model_path}")
        print("\nModel Summary:")
        model.summary()
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load and display training history
    if os.path.exists(history_path):
        try:
            with open(history_path, "rb") as f:
                history = pickle.load(f)
            
            # Plot training history
            plt.figure(figsize=(12, 6))
            
            # Plot loss
            plt.subplot(1, 2, 1)
            plt.plot(history['loss'], label='Training Loss')
            if 'val_loss' in history:
                plt.plot(history['val_loss'], label='Validation Loss')
            plt.title(f"{ticker} Model Loss")
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            # Plot other metrics if available
            metrics = [key for key in history.keys() if key not in ['loss', 'val_loss']]
            if metrics:
                plt.subplot(1, 2, 2)
                for metric in metrics:
                    plt.plot(history[metric], label=metric)
                plt.title(f"{ticker} Model Metrics")
                plt.xlabel('Epoch')
                plt.ylabel('Value')
                plt.legend()
            
            plt.tight_layout()
            plt.show()
            
            print("\nTraining History:")
            for key, values in history.items():
                print(f"  {key}: {values[-1]:.6f} (final), {np.min(values):.6f} (min), {np.max(values):.6f} (max)")
        
        except Exception as e:
            print(f"Error loading history: {e}")
    else:
        print(f"No training history found at: {history_path}")

if __name__ == "__main__":
    # Example usage
    list_all_saved_models()
    
    # Prompt user for ticker to view
    ticker = input("\nEnter ticker to view (e.g., AAPL): ").strip().upper()
    model_type = input("Enter model type (lstm/gru) [default: lstm]: ").strip().lower() or "lstm"
    
    view_model(ticker, model_type) 