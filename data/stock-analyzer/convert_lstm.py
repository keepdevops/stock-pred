"""
Simple script to convert LSTM model to .keras format
"""
import os
import time
import tensorflow as tf

# Load existing LSTM model
model_path = "/Users/moose/stock-pred/data/stock-analyzer/models/LSTM"
print(f"Loading model from: {model_path}")
model = tf.keras.models.load_model(model_path)

# Create new filename with .keras extension
models_dir = "/Users/moose/stock-pred/data/stock-analyzer/models"
timestamp = time.strftime("%Y%m%d_%H%M%S")
new_filename = f"GS_LSTM_{timestamp}.keras"
new_path = os.path.join(models_dir, new_filename)

# Save with .keras extension
print(f"Saving as: {new_filename}")
model.save(new_path, save_format='keras')

# Verify it worked
if os.path.exists(new_path):
    print(f"SUCCESS! Model saved as: {new_filename}")
    print(f"Size: {os.path.getsize(new_path)/1024/1024:.2f} MB")
else:
    print("ERROR: Failed to save model") 