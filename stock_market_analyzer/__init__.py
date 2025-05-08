"""
Stock Market Analyzer package.
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import modules
from .modules.gui import StockGUI
from .modules.database import DatabaseConnector
from .modules.data_loader import DataLoader
from .modules.stock_ai_agent import StockAIAgent
from .modules.trading.real_trading_agent import RealTradingAgent
from .modules.message_bus import MessageBus
from .modules.tabs import (
    BaseTab,
    DataTab,
    AnalysisTab,
    ChartsTab,
    ModelsTab,
    PredictionsTab,
    ImportTab,
    SettingsTab,
    HelpTab,
    TradingTab
)

__all__ = [
    'StockGUI',
    'DatabaseConnector',
    'DataLoader',
    'StockAIAgent',
    'RealTradingAgent',
    'MessageBus',
    'BaseTab',
    'DataTab',
    'AnalysisTab',
    'ChartsTab',
    'ModelsTab',
    'PredictionsTab',
    'ImportTab',
    'SettingsTab',
    'HelpTab',
    'TradingTab'
]

# Default configuration
config = {
    'model_type': 'lstm',
    'sequence_length': 60,
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.001,
    'use_technical_indicators': True,
    'additional_features': []
} 