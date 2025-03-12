"""
Standalone script to test model creation, training, and saving
Run this directly to test without depending on the UI
"""
import os
import time
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Configuration 
TICKER = "TEST"
MODEL_TYPE = "LSTM"
MODELS_DIR = "/Users/moose/stock-pred/data/stock-analyzer/models"
EPOCHS = 5

print("\n===== STANDALONE MODEL SAVE TEST =====")

# 1. Create sample data
print("Creating sample data...")
X_train = np.random.random((100, 10, 5))  # 100 samples, 10 timesteps, 5 features
y_train = np.random.random((100, 1))      # 100 target values
X_test = np.random.random((20, 10, 5))    # 20 test samples
y_test = np.random.random((20, 1))        # 20 test target values

# 2. Create and compile model
print("Creating model...")
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(10, 5)))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# 3. Train the model
print(f"\nTraining model for {EPOCHS} epochs...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=10,
    verbose=1
)

# 4. Prepare paths
timestamp = time.strftime("%Y%m%d_%H%M%S")
model_filename = f"{TICKER}_{MODEL_TYPE}_{timestamp}.keras"
model_path = os.path.join(MODELS_DIR, model_filename)
history_path = model_path.replace('.keras', '_history.pkl')

# 5. Ensure directory exists
os.makedirs(MODELS_DIR, exist_ok=True)
print(f"\nSaving to models directory: {MODELS_DIR}")
print(f"Model exists: {os.path.exists(MODELS_DIR)}")

# 6. Save model
print(f"Saving model to: {model_path}")
model.save(model_path, save_format='keras')
print(f"Model saved: {os.path.exists(model_path)}")

# 7. Save history
print(f"Saving history to: {history_path}")
with open(history_path, 'wb') as f:
    pickle.dump(history.history, f)
print(f"History saved: {os.path.exists(history_path)}")

# 8. Test loading
print("\nVerifying model loading...")
loaded_model = tf.keras.models.load_model(model_path)
print("Model loaded successfully!")

print("\n===== TEST COMPLETED SUCCESSFULLY =====")
print(f"Model saved as: {model_filename}")
print(f"Check your models list in the application")
