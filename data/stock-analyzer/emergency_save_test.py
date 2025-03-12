"""
Emergency Model Save Test
This script will create a simple model and save it to verify functionality
"""
import os
import time
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# 1. Set up paths
MODELS_DIR = "/Users/moose/stock-pred/data/stock-analyzer/models"
timestamp = time.strftime("%Y%m%d_%H%M%S")
model_filename = f"EMERGENCY_TEST_{timestamp}.keras"
model_path = os.path.join(MODELS_DIR, model_filename)
history_path = model_path.replace('.keras', '_history.pkl')

print(f"\n==== EMERGENCY MODEL SAVE TEST ====")
print(f"Models directory: {MODELS_DIR}")
print(f"Model filename: {model_filename}")

# 2. Create directory if needed
os.makedirs(MODELS_DIR, exist_ok=True)
print(f"Directory exists: {os.path.exists(MODELS_DIR)}")

# 3. Create a simple model
print("\nCreating test model...")
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(10, 5)),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# 4. Create dummy training history
history = {
    'loss': [0.05, 0.03, 0.02, 0.01],
    'val_loss': [0.06, 0.04, 0.03, 0.02]
}

# 5. Save the model
print("\nSaving model...")
try:
    model.save(model_path, save_format='keras')
    print(f"✓ Model saved successfully: {model_path}")
    print(f"  File exists: {os.path.exists(model_path)}")
    if os.path.exists(model_path):
        print(f"  File size: {os.path.getsize(model_path)/1024/1024:.2f} MB")
except Exception as e:
    print(f"✗ Error saving model: {e}")
    import traceback
    traceback.print_exc()

# 6. Save the history
print("\nSaving history...")
try:
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"✓ History saved successfully: {history_path}")
    print(f"  File exists: {os.path.exists(history_path)}")
    if os.path.exists(history_path):
        print(f"  File size: {os.path.getsize(history_path)/1024:.2f} KB")
except Exception as e:
    print(f"✗ Error saving history: {e}")
    import traceback
    traceback.print_exc()

# 7. Try to load the model back
print("\nVerifying model loading...")
try:
    loaded_model = tf.keras.models.load_model(model_path)
    print(f"✓ Model loaded successfully")
    # Verify it's the same structure
    loaded_model.summary()
except Exception as e:
    print(f"✗ Error loading model: {e}")
    import traceback
    traceback.print_exc()

print("\n==== EMERGENCY SAVE TEST COMPLETE ====")
print(f"If everything worked, you should see the model: {model_filename}")
print(f"in your models directory: {MODELS_DIR}")
