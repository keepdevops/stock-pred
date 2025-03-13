"""
Direct patch to ensure models are saved with their ticker names
"""
import os
import time
import pickle
import inspect
import importlib
from tensorflow import keras

# Global variables to store current training information
_current_ticker = None
_current_model_type = None

def set_current_ticker(ticker):
    """Set the current ticker being trained"""
    global _current_ticker
    _current_ticker = ticker
    print(f"Current ticker being trained: {ticker}")

def get_current_ticker():
    """Get the current ticker being trained"""
    return _current_ticker

def set_current_model_type(model_type):
    """Set the current model type being trained"""
    global _current_model_type
    _current_model_type = model_type
    print(f"Current model type: {model_type}")

def get_current_model_type():
    """Get the current model type being trained"""
    return _current_model_type

def find_ticker_in_context():
    """Try to find ticker in execution context"""
    ticker = None
    
    # Check if we already have a saved ticker
    if _current_ticker:
        return _current_ticker
    
    # Look through call stack to find ticker information
    frame = inspect.currentframe()
    while frame:
        # Look for ticker in local variables
        for var_name, var_val in frame.f_locals.items():
            if var_name == 'ticker' and isinstance(var_val, str):
                ticker = var_val
                print(f"Found ticker in local variables: {ticker}")
                return ticker
            
            # Check common ticker variable patterns
            if var_name == 'selected_ticker' and isinstance(var_val, str):
                ticker = var_val
                print(f"Found selected_ticker: {ticker}")
                return ticker
            
        # Check if we can find UI widgets with ticker info
        if 'self' in frame.f_locals:
            obj = frame.f_locals['self']
            # Check for ticker_var.get()
            if hasattr(obj, 'ticker_var') and hasattr(obj.ticker_var, 'get'):
                try:
                    ticker = obj.ticker_var.get()
                    if ticker:
                        print(f"Found ticker from ticker_var: {ticker}")
                        return ticker
                except:
                    pass
            
            # Check for selected_ticker attribute
            if hasattr(obj, 'selected_ticker'):
                ticker = obj.selected_ticker
                if ticker:
                    print(f"Found ticker from selected_ticker attribute: {ticker}")
                    return ticker
            
            # Check for ticker_combo.get()
            if hasattr(obj, 'ticker_combo') and hasattr(obj.ticker_combo, 'get'):
                try:
                    ticker = obj.ticker_combo.get()
                    if ticker:
                        print(f"Found ticker from ticker_combo: {ticker}")
                        return ticker
                except:
                    pass
        
        # Move up the call stack
        frame = frame.f_back
    
    # Default ticker name if we can't find it
    print("Could not find ticker in context, using 'unknown'")
    return 'unknown'

def detect_model_type(model):
    """Detect what type of model (LSTM, GRU, etc.) based on layers"""
    model_type = "unknown"
    
    # First check if we already have a saved model type
    if _current_model_type:
        return _current_model_type
    
    # Check model layers for LSTM, GRU, etc.
    for layer in model.layers:
        layer_type = layer.__class__.__name__.lower()
        if "lstm" in layer_type:
            model_type = "lstm"
            break
        elif "gru" in layer_type:
            model_type = "gru"
            break
    
    print(f"Detected model type: {model_type}")
    return model_type

def get_save_paths(ticker, model_type):
    """Generate appropriate save paths with ticker and model type"""
    # Get models directory
    models_dir = "/Users/moose/stock-pred/data/stock-analyzer/models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created models directory: {models_dir}")
    
    # Create filename with ticker and model_type
    base_filename = f"{ticker}_{model_type}_model"
    model_path = os.path.join(models_dir, f"{base_filename}.keras")
    history_path = os.path.join(models_dir, f"{base_filename}_history.pkl")
    
    # Also create the model type path (for compatibility)
    model_type_path = os.path.join(models_dir, f"{model_type}.keras")
    history_type_path = os.path.join(models_dir, f"{model_type}_history.pkl")
    
    return model_path, history_path, model_type_path, history_type_path

def patched_save(self, filepath, *args, **kwargs):
    """
    Patched version of Model.save that ensures ticker name is included
    """
    original_filepath = filepath
    
    # Always use .keras extension
    if not filepath.endswith('.keras'):
        filepath = filepath + '.keras'
    
    # Make sure save_format is set to 'keras'
    kwargs['save_format'] = 'keras'
    
    # If filepath doesn't contain ticker, add it
    ticker = find_ticker_in_context()
    model_type = detect_model_type(self)
    
    if ticker != 'unknown' and f"{ticker}_" not in os.path.basename(filepath):
        # Create new filepath with ticker included
        models_dir = os.path.dirname(filepath) or "/Users/moose/stock-pred/data/stock-analyzer/models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        # Keep the original filename if specified explicitly
        if original_filepath != "/Users/moose/stock-pred/data/stock-analyzer/models/test_model.keras":
            model_filename = f"{ticker}_{model_type}_model.keras"
            filepath = os.path.join(models_dir, model_filename)
            print(f"\n>>> PATCHED SAVE: Including ticker in filename. New path: {filepath}")
    
    print(f"\n>>> PATCHED SAVE: Saving model to: {filepath}")
    
    # Call the original save method
    try:
        result = original_save(self, filepath, *args, **kwargs)
        print(f">>> PATCHED SAVE: Model saved successfully: {os.path.exists(filepath)}")
        
        # Also save the model history if available
        try:
            if hasattr(self, 'history') and self.history:
                history_path = filepath.replace('.keras', '_history.pkl')
                with open(history_path, 'wb') as f:
                    pickle.dump(self.history, f)
                print(f">>> PATCHED SAVE: History saved to: {history_path}")
        except Exception as e:
            print(f">>> PATCHED SAVE: Could not save history: {e}")
            
        return result
    except Exception as e:
        print(f">>> PATCHED SAVE ERROR: {str(e)}")
        raise

# Find actual training functions to monkey patch
def patch_training_methods():
    """Find and patch common training methods"""
    # Look for _train_model in ui.training_panel
    try:
        training_module = importlib.import_module('ui.training_panel')
        
        for name, obj in inspect.getmembers(training_module):
            if inspect.isclass(obj):
                # Look for training methods that might be creating models
                for method_name in dir(obj):
                    if callable(getattr(obj, method_name)) and ('train' in method_name.lower() or 'create' in method_name.lower()):
                        original_method = getattr(obj, method_name)
                        
                        def patched_method(self, *args, **kwargs):
                            """Patched training method that captures ticker"""
                            print(f"Intercepted training method: {method_name}")
                            
                            # Try to get ticker
                            ticker = None
                            if hasattr(self, 'ticker_var') and hasattr(self.ticker_var, 'get'):
                                ticker = self.ticker_var.get()
                            elif hasattr(self, 'selected_ticker'):
                                ticker = self.selected_ticker
                            elif len(args) > 0 and isinstance(args[0], str):
                                ticker = args[0]
                            
                            # Set the current ticker
                            if ticker:
                                set_current_ticker(ticker)
                            
                            # Call original method
                            result = original_method(self, *args, **kwargs)
                            
                            # If result is a model, save it with ticker name
                            if isinstance(result, keras.Model):
                                model_type = detect_model_type(result)
                                set_current_model_type(model_type)
                                
                                ticker = get_current_ticker() or 'unknown'
                                model_path, history_path, _, _ = get_save_paths(ticker, model_type)
                                
                                # Save model with ticker name
                                try:
                                    result.save(model_path, save_format='keras')
                                    print(f"Auto-saved model with ticker name: {model_path}")
                                    
                                    # Save history if available
                                    if hasattr(result, 'history') and result.history:
                                        with open(history_path, 'wb') as f:
                                            pickle.dump(result.history, f)
                                        print(f"Auto-saved history with ticker name: {history_path}")
                                except Exception as e:
                                    print(f"Error auto-saving model with ticker: {e}")
                            
                            # If result is a tuple containing model and history
                            elif isinstance(result, tuple) and len(result) >= 2 and isinstance(result[0], keras.Model):
                                model, history = result[0], result[1]
                                model_type = detect_model_type(model)
                                set_current_model_type(model_type)
                                
                                ticker = get_current_ticker() or 'unknown'
                                model_path, history_path, _, _ = get_save_paths(ticker, model_type)
                                
                                # Save model with ticker name
                                try:
                                    model.save(model_path, save_format='keras')
                                    print(f"Auto-saved model with ticker name: {model_path}")
                                    
                                    # Save history
                                    with open(history_path, 'wb') as f:
                                        pickle.dump(history, f)
                                    print(f"Auto-saved history with ticker name: {history_path}")
                                except Exception as e:
                                    print(f"Error auto-saving model with ticker: {e}")
                            
                            return result
                        
                        # Apply patch
                        setattr(obj, method_name, patched_method)
                        print(f"Patched training method: {name}.{method_name}")
        
    except ImportError:
        print("Could not import training module.")
    
    # Also directly patch Keras model.fit to capture ticker
    original_fit = keras.Model.fit
    
    def patched_fit(self, *args, **kwargs):
        """Patched model.fit method to ensure model is saved with ticker name after training"""
        # Save current ticker in case we need it later
        ticker = get_current_ticker() or find_ticker_in_context()
        if ticker:
            set_current_ticker(ticker)
        
        # Call original fit method
        result = original_fit(self, *args, **kwargs)
        
        # Automatically save the model with ticker name
        if hasattr(result, 'history'):
            model_type = detect_model_type(self)
            set_current_model_type(model_type)
            
            ticker = get_current_ticker() or 'unknown'
            model_path, history_path, model_type_path, history_type_path = get_save_paths(ticker, model_type)
            
            # For backwards compatibility, also save with just the model type
            try:
                self.save(model_type_path, save_format='keras')
                
                # Save history
                with open(history_type_path, 'wb') as f:
                    pickle.dump(result.history, f)
                print(f"Saved model with type: {model_type_path}")
            except Exception as e:
                print(f"Error saving model with type: {e}")
        
        return result
    
    # Apply fit patch
    keras.Model.fit = patched_fit
    print("Patched keras.Model.fit method for ticker tracking during training")

def apply_ticker_save_patch():
    """Apply all patches to ensure models are saved with ticker names"""
    global original_save
    
    print("")
    print("=================================================")
    print("| Ticker Model Save patch applied                |")
    print("| Models will be saved with ticker_modeltype.keras|")
    print("=================================================")
    print("")
    
    # Patch Model.save
    original_save = keras.Model.save
    keras.Model.save = patched_save
    
    # Patch training methods
    patch_training_methods()
    
    # Find and patch direct training methods in controller modules
    try:
        # Look for controller module
        controller_module = importlib.import_module('controller.model_controller')
        
        for name, obj in inspect.getmembers(controller_module):
            if inspect.isclass(obj) and 'controller' in name.lower():
                for method_name in dir(obj):
                    if callable(getattr(obj, method_name)) and ('train' in method_name.lower() or 'create' in method_name.lower()):
                        original_method = getattr(obj, method_name)
                        
                        def patched_controller_method(self, *args, **kwargs):
                            """Patched controller method that tracks the ticker"""
                            print(f"Intercepted controller method: {method_name}")
                            
                            # Try to get ticker from various sources
                            ticker = None
                            if len(args) > 0 and isinstance(args[0], str):
                                ticker = args[0]
                            elif 'ticker' in kwargs:
                                ticker = kwargs['ticker']
                            elif hasattr(self, 'selected_ticker'):
                                ticker = self.selected_ticker
                            
                            # Set the current ticker
                            if ticker:
                                set_current_ticker(ticker)
                            
                            # Call original method
                            return original_method(self, *args, **kwargs)
                        
                        # Apply patch
                        setattr(obj, method_name, patched_controller_method)
                        print(f"Patched controller method: {name}.{method_name}")
    
    except ImportError:
        print("Could not import controller module.")
    
    print("Ticker save patches applied successfully")