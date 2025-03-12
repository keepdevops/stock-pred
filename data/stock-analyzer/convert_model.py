"""
Ultra simple script to convert existing models to .keras format
Run this script directly with: python convert_model.py
"""
import os
import time
import tensorflow as tf

print("\n=== DIRECT MODEL CONVERTER ===")

# Define the model to convert - change this if needed
MODEL_NAME = "LSTM"
TICKER = "GS"  # Change to your preferred ticker name

# Directory paths
models_dir = "/Users/moose/stock-pred/data/stock-analyzer/models"
source_path = os.path.join(models_dir, MODEL_NAME)

# Verify the source model exists
print(f"Looking for model: {source_path}")
if not os.path.exists(source_path):
    print(f"ERROR: Source model not found: {source_path}")
    exit(1)

# Create target filename with timestamp
timestamp = time.strftime("%Y%m%d_%H%M%S")
target_filename = f"{TICKER}_{MODEL_NAME}_{timestamp}.keras"
target_path = os.path.join(models_dir, target_filename)

print(f"Source model: {source_path}")
print(f"Target path: {target_path}")

try:
    # Load the model
    print("Loading model...")
    model = tf.keras.models.load_model(source_path)
    print("Model loaded successfully!")
    
    # Save with .keras extension
    print(f"Saving model as: {target_filename}")
    model.save(target_path, save_format='keras')
    
    if os.path.exists(target_path):
        print(f"\n✅ SUCCESS: Model converted and saved as: {target_filename}")
        print(f"File size: {os.path.getsize(target_path)/1024/1024:.2f} MB")
    else:
        print("\n❌ ERROR: File was not created")
        
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n=== CONVERSION COMPLETE ===") 