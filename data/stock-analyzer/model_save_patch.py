"""
Model Save Patch
Monkey patches Keras' fit method to auto-save models after training
"""
import os
import time
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model

# Store the original fit method
original_fit = tf.keras.Model.fit

def patched_fit(self, *args, **kwargs):
    """Patched version of fit that auto-saves models after training"""
    # Call the original fit method to perform training
    history = original_fit(self, *args, **kwargs)
    
    # After training completes, auto-save the model
    try:
        print("\n===== AUTO-SAVING MODEL WITH .KERAS EXTENSION =====")
        
        # Create models directory if it doesn't exist
        models_dir = "/Users/moose/stock-pred/data/stock-analyzer/models"
        os.makedirs(models_dir, exist_ok=True)
        
        # Try to get ticker from the UI context
        ticker = "unknown"
        try:
            # Look for the ticker in the parent frames
            import tkinter as tk
            for widget in tk._default_root.winfo_children():
                if hasattr(widget, 'nametowidget'):
                    ticker_widget = None
                    try:
                        ticker_widget = widget.nametowidget(".!frame.!frame.ticker_combobox")
                        if ticker_widget and hasattr(ticker_widget, 'get'):
                            ticker = ticker_widget.get()
                            break
                    except:
                        pass
        except:
            pass
        
        # Determine model type from the layer types
        if any('lstm' in str(layer).lower() for layer in self.layers):
            model_type = "LSTM"
        elif any('gru' in str(layer).lower() for layer in self.layers):
            model_type = "GRU"
        else:
            model_type = "Model"
        
        # Create filenames
        model_filename = f"{ticker}_{model_type}_{time.strftime('%Y%m%d_%H%M%S')}.keras"
        model_path = os.path.join(models_dir, model_filename)
        history_path = model_path.replace(".keras", "_history.pkl")
        
        # Save the model
        print(f"Saving model to: {model_path}")
        self.save(model_path, save_format='keras')
        print(f"Model saved successfully: {os.path.exists(model_path)}")
        
        # Save the training history
        print(f"Saving history to: {history_path}")
        with open(history_path, 'wb') as f:
            pickle.dump(history.history, f)
        print(f"History saved successfully: {os.path.exists(history_path)}")
        
        # Refresh the model list if possible
        try:
            from ui.training_panel import refresh_model_list
            refresh_model_list()
            print("Model list refreshed")
        except:
            pass
        
        print(f"Model and history saved as: {model_filename}")
        print("===== AUTO-SAVE COMPLETE =====\n")
    
    except Exception as e:
        print(f"ERROR during auto-save: {str(e)}")
        import traceback
        traceback.print_exc()
        print("===== AUTO-SAVE FAILED =====\n")
    
    # Return the original history
    return history

# Apply the monkey patch
def apply_model_save_patch():
    print("\n=================================================")
    print("| Model Auto-Save patch applied successfully      |")
    print("| All trained models will be saved automatically  |")
    print("=================================================\n")
    tf.keras.Model.fit = patched_fit 