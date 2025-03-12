"""
Test script to verify model saving functionality
Run this directly with: python test_model_save.py
"""
import os
import time
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

def test_model_save():
    print("\n===== MODEL SAVE TEST =====")
    
    # 1. Create a simple model
    print("Creating test model...")
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(10, 5)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # Print model summary
    model.summary()
    
    # 2. Create dummy history
    history = {
        'loss': [0.1, 0.05, 0.02, 0.01],
        'val_loss': [0.15, 0.08, 0.03, 0.02]
    }
    
    # 3. Set up paths
    models_dir = "/Users/moose/stock-pred/data/stock-analyzer/models"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_filename = f"TEST_MODEL_{timestamp}.keras"
    model_path = os.path.join(models_dir, model_filename)
    history_path = model_path.replace('.keras', '_history.pkl')
    
    print(f"Models directory: {models_dir}")
    print(f"Model path: {model_path}")
    
    # 4. Ensure directory exists
    os.makedirs(models_dir, exist_ok=True)
    print(f"Directory exists: {os.path.exists(models_dir)}")
    
    # 5. Save model
    print("Saving model...")
    try:
        model.save(model_path, save_format='keras')
        print(f"Model saved successfully: {os.path.exists(model_path)}")
        if os.path.exists(model_path):
            print(f"Model file size: {os.path.getsize(model_path)/1024/1024:.2f} MB")
    except Exception as e:
        print(f"ERROR saving model: {e}")
        import traceback
        traceback.print_exc()
    
    # 6. Save history
    print("Saving history...")
    try:
        with open(history_path, 'wb') as f:
            pickle.dump(history, f)
        print(f"History saved successfully: {os.path.exists(history_path)}")
        if os.path.exists(history_path):
            print(f"History file size: {os.path.getsize(history_path)/1024:.2f} KB")
    except Exception as e:
        print(f"ERROR saving history: {e}")
        import traceback
        traceback.print_exc()
    
    # 7. Try loading the saved model
    print("Testing model loading...")
    try:
        loaded_model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        print("Model layers:")
        for i, layer in enumerate(loaded_model.layers):
            print(f"  Layer {i}: {layer.name} - {type(layer).__name__}")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
    
    print("===== TEST COMPLETE =====\n")
    return model_path, history_path

if __name__ == "__main__":
    model_path, history_path = test_model_save()
    print(f"Test model saved to: {model_path}")
    print(f"Test history saved to: {history_path}") 