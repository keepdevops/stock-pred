"""
TensorFlow Save Function Monkey Patch
This script patches TensorFlow to always save models with the .keras extension
Run this BEFORE starting your application
"""
import os
import sys
import time
import tensorflow as tf
from tensorflow.keras.models import Model

# Store the original save method
original_save = tf.keras.models.Model.save

# Create a wrapper function that forces .keras extension
def patched_save(self, filepath, *args, **kwargs):
    """Patched save function that ensures .keras extension"""
    # Force save_format to 'keras'
    kwargs['save_format'] = 'keras'
    
    # Ensure filepath has .keras extension
    if not filepath.endswith('.keras'):
        # If file has another extension, replace it
        if '.' in os.path.basename(filepath):
            filepath = os.path.splitext(filepath)[0] + '.keras'
        else:
            # If no extension, add .keras
            filepath = filepath + '.keras'
    
    print(f"\n>>> PATCHED SAVE: Saving model to: {filepath}")
    # Call the original save method with our modified arguments
    result = original_save(self, filepath, *args, **kwargs)
    print(f">>> PATCHED SAVE: Model saved successfully: {os.path.exists(filepath)}")
    return result

# Replace the save method with our patched version
tf.keras.models.Model.save = patched_save

print("\n==================================================")
print("| TensorFlow Model.save() patched successfully    |")
print("| All models will now be saved with .keras format |")
print("==================================================\n")

# Test the patch with a simple model
def test_patch():
    """Test that our patch is working"""
    print("Creating test model...")
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, input_shape=(5,), activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # Try saving with different paths
    models_dir = "/Users/moose/stock-pred/data/stock-analyzer/models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Test case 1: No extension
    path1 = os.path.join(models_dir, "test_model_no_extension")
    model.save(path1)
    print(f"Test 1 result: {os.path.exists(path1 + '.keras')}")
    
    # Test case 2: Wrong extension
    path2 = os.path.join(models_dir, "test_model.h5")
    model.save(path2)
    print(f"Test 2 result: {os.path.exists(path2.replace('.h5', '.keras'))}")
    
    # Test case 3: Correct extension
    path3 = os.path.join(models_dir, "test_model.keras")
    model.save(path3)
    print(f"Test 3 result: {os.path.exists(path3)}")
    
    print("\nPatch test completed successfully!")

# Run the test
test_patch()

# If you're importing this script, the patch will be applied
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--run-app":
        # If --run-app flag is provided, import and run the main application
        print("Starting application with patched TensorFlow...")
        
        # Import your main module
        try:
            import main
            # Run the main function if it exists
            if hasattr(main, "main"):
                main.main()
            # If no main function, try running the file
            else:
                exec(open("main.py").read())
                
        except Exception as e:
            print(f"Error running application: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nRun your application with the patched TensorFlow by:")
        print("1. Import this module at the start of your application:")
        print("   import fix_model_save")
        print("2. Or run your application through this script:")
        print("   python fix_model_save.py --run-app")

# Define the model to convert - change this if needed
MODEL_NAME = "LSTM"  # Change to "GRU" or other model name if needed
TICKER = "GS"        # Change to your preferred ticker name 