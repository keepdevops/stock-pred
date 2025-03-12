"""
Patch for Model Viewer to Handle .keras Extension
"""
import os
import pickle
import tensorflow as tf

def load_model_with_history(model_name, models_dir):
    """
    Load a model and its history with proper .keras extension handling
    
    Args:
        model_name: Name of the model to load
        models_dir: Directory containing models
        
    Returns:
        tuple: (model, history) if successful, (None, None) otherwise
    """
    print(f"\n===== LOADING MODEL: {model_name} =====")
    
    # Ensure .keras extension
    if not model_name.endswith('.keras'):
        keras_path = os.path.join(models_dir, model_name + '.keras')
        if os.path.exists(keras_path):
            model_path = keras_path
            print(f"Added .keras extension, using: {model_path}")
        else:
            # Try original path as fallback
            model_path = os.path.join(models_dir, model_name)
            print(f"Using fallback path: {model_path}")
    else:
        model_path = os.path.join(models_dir, model_name)
        print(f"Model already has .keras extension: {model_path}")
    
    # Check if file exists
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None, None
    
    # Load model
    try:
        print(f"Loading model from: {model_path}")
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully")
        
        # Try to load history
        history = None
        history_paths = [
            model_path.replace('.keras', '_history.pkl'),
            model_path + '_history.pkl',
            model_path.replace('.keras', '.history'),
            model_path + '.history'
        ]
        
        for path in history_paths:
            print(f"Checking for history at: {path}")
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        history = pickle.load(f)
                    print(f"History loaded from: {path}")
                    break
                except Exception as e:
                    print(f"Error loading history from {path}: {e}")
        
        return model, history
    
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None 