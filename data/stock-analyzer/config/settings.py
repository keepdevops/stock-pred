"""
Application settings and configuration
"""

# Database settings
DEFAULT_DB_TYPE = "duckdb"

# Model settings
DEFAULT_SEQUENCE_LENGTH = 10
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_MODEL_TYPE = "LSTM"
DEFAULT_NEURONS = 50
DEFAULT_LAYERS = 2
DEFAULT_DROPOUT = 0.2

# UI settings
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
DARK_BG = "#2E2E2E"
DARKER_BG = "#1E1E1E"
LIGHT_TEXT = "white"
ACCENT_COLOR = "#4A6CD4"

# Features for prediction
DEFAULT_FEATURES = [
    'open', 'high', 'low', 'close', 'volume', 
    'ma20', 'ma50', 'rsi', 'macd'
]

# Prediction settings
DEFAULT_PREDICTION_DAYS = 30
