"""Machine learning models for stock market analysis."""

from .lstm_model import LSTMModel
from .transformer_model import TransformerModel
from .model_factory import ModelFactory
from .model_tuner import ModelTuner

# Try to import XGBoostModel, but don't fail if it's not available
try:
    from .xgboost_model import XGBoostModel
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBoostModel = None

__all__ = [
    'LSTMModel',
    'TransformerModel',
    'ModelFactory',
    'ModelTuner'
]

# Only add XGBoostModel to __all__ if it's available
if XGBOOST_AVAILABLE:
    __all__.append('XGBoostModel') 