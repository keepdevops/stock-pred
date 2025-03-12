# Patch for the function that loads models for viewing
def patch_model_viewer():
    import os
    from pathlib import Path
    
    def find_keras_model(model_name):
        """Find a model file with or without .keras extension"""
        models_dir = "/Users/moose/stock-pred/data/stock-analyzer/models"
        
        # Check with .keras extension first
        keras_path = os.path.join(models_dir, f"{model_name}.keras")
        if os.path.exists(keras_path):
            return keras_path
            
        # Check without extension
        base_path = os.path.join(models_dir, model_name)
        if os.path.exists(base_path):
            return base_path
            
        # Check all files with .keras extension for matching pattern
        files = list(Path(models_dir).glob(f"*{model_name}*.keras"))
        if files:
            return str(files[0])
            
        return None
    
    # Find and patch the model loading function
    # This would require identifying the exact function to patch 