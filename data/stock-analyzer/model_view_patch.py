"""
Model View Patch - Enhances model viewing to support .keras files
"""
import os
import glob
import tensorflow as tf

def find_model_file(model_name, model_dir):
    """Find a model file with the new naming pattern"""
    print(f"Looking for model: {model_name} in {model_dir}")
    
    # Check for exact match with new naming pattern
    exact_path = os.path.join(model_dir, f"{model_name}_model.keras")
    if os.path.exists(exact_path):
        print(f"Found exact model match at: {exact_path}")
        return exact_path
        
    # List of possible paths to check
    paths_to_check = [
        os.path.join(model_dir, f"{model_name}.keras"),           # Model with .keras extension
        os.path.join(model_dir, model_name),                      # Model without extension
        os.path.join(model_dir, f"{model_name}_lstm_model.keras"),# New pattern with lstm
        os.path.join(model_dir, f"{model_name}_gru_model.keras"), # New pattern with gru
        *glob.glob(os.path.join(model_dir, f"{model_name}_*.keras")), # Any model with ticker prefix
    ]
    
    # Check each path
    for path in paths_to_check:
        if os.path.exists(path):
            print(f"Found model at: {path}")
            return path
    
    print(f"No model found for {model_name}")
    return None

def find_history_file(model_path, model_name, model_dir):
    """Find the history file for a given model file"""
    print(f"Looking for history for model: {model_path}")
    
    # Direct replacement for new naming pattern
    if model_path.endswith('_model.keras'):
        history_path = model_path.replace('_model.keras', '_model_history.pkl')
        if os.path.exists(history_path):
            print(f"Found history at: {history_path}")
            return history_path
    
    # List of possible history paths
    paths_to_check = [
        model_path.replace('.keras', '_history.pkl') if model_path.endswith('.keras') else f"{model_path}_history.pkl",
        os.path.join(model_dir, f"{model_name}_history.pkl"),
        os.path.join(model_dir, f"{model_name}.history"),
        os.path.join(model_dir, f"{model_name}_lstm_model_history.pkl"), # New pattern with lstm
        os.path.join(model_dir, f"{model_name}_gru_model_history.pkl"),  # New pattern with gru
    ]
    
    # Check each path
    for path in paths_to_check:
        if os.path.exists(path):
            print(f"Found history at: {path}")
            return path
    
    print(f"No history found for {model_name}")
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
                
            history_path = find_history_file(model_path, model_name, self.models_directory)
            
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