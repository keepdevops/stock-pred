"""
Model View Patch - Enhances model viewing to support .keras files
"""
import os
import glob
import tensorflow as tf
from set_ticker import set_ticker_for_training

def find_model_file(model_name, model_dir):
    """Find a model file with .keras extension or without"""
    print(f"Looking for model: {model_name} in {model_dir}")
    
    # Check with .keras extension first
    keras_path = os.path.join(model_dir, f"{model_name}.keras")
    if os.path.exists(keras_path):
        print(f"Found model at {keras_path}")
        return keras_path
        
    # Check for ticker_model_type.keras pattern
    model_with_type_path = os.path.join(model_dir, f"{model_name}_lstm_model.keras")
    if os.path.exists(model_with_type_path):
        print(f"Found model at {model_with_type_path}")
        return model_with_type_path
        
    # Check without extension (legacy)
    base_path = os.path.join(model_dir, model_name)
    if os.path.exists(base_path):
        print(f"Found legacy model at {base_path}")
        return base_path
    
    # Check all files with ticker prefix
    pattern = os.path.join(model_dir, f"{model_name}_*.keras")
    keras_files = glob.glob(pattern)
    if keras_files:
        print(f"Found model at {keras_files[0]}")
        return keras_files[0]
    
    # Not found
    return None

def find_history_file(model_path, model_dir=None, model_name=None):
    """Find the history file for a given model path"""
    if not model_path:
        return None
        
    print(f"Looking for history for model: {model_path}")
    
    # If model ends with _model.keras
    if model_path.endswith('_model.keras'):
        history_path = model_path.replace('_model.keras', '_model_history.pkl')
        if os.path.exists(history_path):
            print(f"Found history at {history_path}")
            return history_path
    
    # If model ends with .keras
    if model_path.endswith('.keras'):
        history_path = model_path.replace('.keras', '_history.pkl')
        if os.path.exists(history_path):
            print(f"Found history at {history_path}")
            return history_path
    
    # Try standard format with _history.pkl suffix
    history_path = f"{model_path}_history.pkl"
    if os.path.exists(history_path):
        print(f"Found history at {history_path}")
        return history_path
        
    # Try standard format with .history suffix
    history_path = f"{model_path}.history"
    if os.path.exists(history_path):
        print(f"Found history at {history_path}")
        return history_path
    
    # Not found
    return None

def load_model_with_extension(model_path):
    """Load a model with proper handling of .keras extension"""
    try:
        if model_path.endswith('.keras'):
            print(f"Loading .keras model from: {model_path}")
            return tf.keras.models.load_model(model_path, compile=True)
        else:
            # Try loading without format specification first
            try:
                return tf.keras.models.load_model(model_path, compile=True)
            except:
                # If that fails, try with h5 format
                return tf.keras.models.load_model(model_path, compile=True)
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None

# Monkey patch functions to create a direct fix
def monkey_patch_model_controller():
    """
    Patch the ModelController class to properly handle .keras extensions
    """
    try:
        # Import your model controller
        from ui.model_controller import ModelController
        
        # Save original methods
        original_view_model = ModelController.view_model
        
        # Define patched methods
        def patched_view_model(self, model_name):
            """Patched version of view_model that handles .keras extension"""
            print(f"Patched view_model called for {model_name}")
            model_path = find_model_file(model_name, self.models_directory)
            
            if not model_path:
                print(f"Model {model_name} not found")
                return None
                
            history_path = find_history_file(model_path, self.models_directory, model_name)
            
            # Load the model and history
            model = load_model_with_extension(model_path)
            
            # Continue with original function for displaying
            if model:
                print(f"Model loaded successfully from {model_path}")
                # Call original to handle the rest
                return original_view_model(self, model_name)
            else:
                print("Failed to load model")
                return None
        
        # Apply patches
        ModelController.view_model = patched_view_model
        print("ModelController.view_model successfully patched")
        
    except Exception as e:
        print(f"Failed to patch ModelController: {e}")
        import traceback
        traceback.print_exc()

def apply_view_model_patch():
    """Apply the view model patch to the application"""
    print("\n=================================================")
    print("| Model Viewer patch applied                      |")
    print("| Enhanced to find ticker_lstm_model.keras files  |")
    print("=================================================\n")
    monkey_patch_model_controller() 

def _train_model(self, ticker, epochs, *args, **kwargs):
    # Set ticker before training
    set_ticker_for_training(ticker)
    
    # Rest of your training code... 