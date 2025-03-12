"""
Model Save Patch
Monkey patches Keras' fit method to auto-save models after training
"""
import os
import time
import pickle
import tensorflow as tf
import traceback
import pandas as pd
import numpy as np

# Store the original fit method
original_fit = tf.keras.Model.fit

# Global variable to store the current ticker - can be set by the application
current_ticker = None

def set_current_ticker(ticker):
    """Allow setting the current ticker from outside"""
    global current_ticker
    current_ticker = ticker
    print(f"Current ticker set to: {ticker}")

def find_ticker_in_data(X_data):
    """Try to detect ticker from training data"""
    try:
        # If X_data is a pandas DataFrame with 'ticker' column
        if isinstance(X_data, pd.DataFrame) and 'ticker' in X_data.columns:
            tickers = X_data['ticker'].unique()
            if len(tickers) > 0:
                return tickers[0]
                
        # If X_data is related to a ticker in some other way
        if hasattr(X_data, 'name') and isinstance(X_data.name, str):
            return X_data.name
            
        # If X_data is a dict with ticker info
        if isinstance(X_data, dict) and 'ticker' in X_data:
            return X_data['ticker']
    except:
        pass
    return None

def detect_ticker_from_ui():
    """Aggressive approach to find ticker in UI"""
    try:
        import tkinter as tk
        root = tk._default_root
        if not root:
            return None
            
        # Find all comboboxes and entries that might contain ticker info
        potential_ticker_widgets = []
        
        def scan_widget(widget, depth=0):
            if depth > 10:  # Limit recursion
                return
                
            # Check widget attributes
            if hasattr(widget, 'get') and callable(widget.get):
                try:
                    value = widget.get()
                    # If it looks like a ticker (uppercase, 1-5 chars)
                    if (isinstance(value, str) and value.isupper() and 
                        len(value) >= 1 and len(value) <= 5):
                        potential_ticker_widgets.append((widget, value))
                except:
                    pass
                    
            # Check if it's named with 'ticker' in the name
            widget_name = str(widget).lower()
            if 'ticker' in widget_name and hasattr(widget, 'get'):
                try:
                    value = widget.get()
                    if value and isinstance(value, str):
                        potential_ticker_widgets.append((widget, value))
                except:
                    pass
            
            # Recursively scan children
            if hasattr(widget, 'winfo_children'):
                for child in widget.winfo_children():
                    scan_widget(child, depth+1)
        
        # Start scanning from root
        scan_widget(root)
        
        # Return the most likely ticker
        if potential_ticker_widgets:
            # Sort by likelihood (prefer longer tickers that are all uppercase)
            potential_ticker_widgets.sort(
                key=lambda x: (x[1].isupper(), len(x[1])), 
                reverse=True
            )
            return potential_ticker_widgets[0][1]
    except:
        pass
    return None

def patched_fit(self, *args, **kwargs):
    """Patched version of fit that auto-saves models after training"""
    # Attempt to find ticker from args before training
    ticker_from_args = None
    if args and len(args) > 0:
        ticker_from_args = find_ticker_in_data(args[0])
    
    # Call the original fit method to perform training
    history = original_fit(self, *args, **kwargs)
    
    # After training completes, auto-save the model
    try:
        print("\n===== AUTO-SAVING MODEL WITH .KERAS EXTENSION =====")
        
        # Create models directory if it doesn't exist
        models_dir = "/Users/moose/stock-pred/data/stock-analyzer/models"
        os.makedirs(models_dir, exist_ok=True)
        
        # Try multiple methods to get the ticker (in order of preference)
        ticker = "unknown"
        
        # Method 1: Use the global variable if it was set
        global current_ticker
        if current_ticker:
            ticker = current_ticker
            print(f"Using globally set ticker: {ticker}")
            
        # Method 2: Use ticker found in training data arguments
        elif ticker_from_args:
            ticker = ticker_from_args
            print(f"Using ticker from training data: {ticker}")
            
        # Method 3: Check for ticker information in the stack frames
        else:
            try:
                import inspect
                frame = inspect.currentframe()
                while frame:
                    if 'self' in frame.f_locals:
                        # Check for ticker_var
                        if hasattr(frame.f_locals['self'], 'ticker_var'):
                            if hasattr(frame.f_locals['self'].ticker_var, 'get'):
                                value = frame.f_locals['self'].ticker_var.get()
                                if value:
                                    ticker = value
                                    print(f"Found ticker in stack (ticker_var): {ticker}")
                                    break
                        
                        # Check for selected_ticker
                        if hasattr(frame.f_locals['self'], 'selected_ticker'):
                            value = frame.f_locals['self'].selected_ticker
                            if value:
                                ticker = value
                                print(f"Found ticker in stack (selected_ticker): {ticker}")
                                break
                    
                    # Check for ticker in local variables
                    if 'ticker' in frame.f_locals and isinstance(frame.f_locals['ticker'], str):
                        ticker = frame.f_locals['ticker']
                        print(f"Found ticker in local variables: {ticker}")
                        break
                        
                    frame = frame.f_back
            except Exception as e:
                print(f"Error finding ticker in stack: {e}")
        
        # Method 4: Look for ticker in UI widgets
        if ticker == "unknown":
            ui_ticker = detect_ticker_from_ui()
            if ui_ticker:
                ticker = ui_ticker
                print(f"Found ticker in UI widgets: {ticker}")
                
        # Method 5: Look for a _ticker attribute in the model
        if ticker == "unknown" and hasattr(self, '_ticker'):
            ticker = self._ticker
            print(f"Found ticker in model: {ticker}")
        
        # Determine model type from the layer types (use lowercase as requested)
        if any('lstm' in str(layer).lower() for layer in self.layers):
            model_type = "lstm"
        elif any('gru' in str(layer).lower() for layer in self.layers):
            model_type = "gru"
        else:
            model_type = "model"
        
        # Create filename using the requested format: {ticker}_lstm_model.keras
        model_filename = f"{ticker}_{model_type}_model.keras"
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
        
        # Also save a copy with the standard name for compatibility (uppercase for backward compatibility)
        standard_model_type = model_type.upper()
        compat_model_path = os.path.join(models_dir, standard_model_type)
        compat_history_path = os.path.join(models_dir, f"{standard_model_type}_history.pkl")
        
        try:
            self.save(compat_model_path, save_format='keras')
            with open(compat_history_path, 'wb') as f:
                pickle.dump(history.history, f)
            print(f"Compatibility copies saved at: {compat_model_path}")
        except Exception as e:
            print(f"Note: Could not save compatibility copies: {e}")
        
        print(f"Model and history saved as: {model_filename}")
        print("===== AUTO-SAVE COMPLETE =====\n")
    
    except Exception as e:
        print(f"ERROR during auto-save: {str(e)}")
        traceback.print_exc()
        print("===== AUTO-SAVE FAILED =====\n")
    
    # Return the original history
    return history

def patch_train_function():
    """Patch the training function to capture the ticker"""
    # Use this function to patch the specific training function in your code
    # that contains the ticker information
    print("Training function patch would be implemented here")
    # Example (pseudocode):
    # original_train = YourClass._train_model
    # def patched_train(self, *args, **kwargs):
    #     set_current_ticker(self.ticker_var.get())
    #     return original_train(self, *args, **kwargs)
    # YourClass._train_model = patched_train

def apply_model_save_patch():
    """Apply the model save patch to the application"""
    print("\n=================================================")
    print("| Model Auto-Save patch applied successfully      |")
    print("| Models will be saved as: ticker_lstm_model.keras|")
    print("=================================================\n")
    tf.keras.Model.fit = patched_fit 