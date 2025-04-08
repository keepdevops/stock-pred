"""
Stock Market Analyzer package.
"""

# Remove circular imports
# from .modules.gui import StockGUI
# from .modules.database import DatabaseConnector
# from .modules.data_loader import DataLoader
# from .modules.stock_ai_agent import StockAIAgent
# from .modules.trading.real_trading_agent import RealTradingAgent

__all__ = [
    'StockGUI',
    'DatabaseConnector',
    'DataLoader',
    'StockAIAgent',
    'RealTradingAgent'
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