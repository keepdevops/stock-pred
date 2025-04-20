"""Stock Market Analyzer modules package."""

import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Remove circular imports
# from .database import DatabaseConnector
# from .data_loader import DataLoader
# from .stock_ai_agent import StockAIAgent
# from .trading.real_trading_agent import RealTradingAgent
# from .gui import StockGUI

# Import message bus
from .message_bus import MessageBus

# Import tabs module
from .tabs import (
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

from .database import DatabaseConnector
from .data_loader import DataLoader
from .data_service import DataService
from .stock_ai_agent import StockAIAgent
from .trading.real_trading_agent import RealTradingAgent

__all__ = [
    'BaseTab',
    'DataTab',
    'AnalysisTab',
    'ChartsTab',
    'ModelsTab',
    'PredictionsTab',
    'ImportTab',
    'SettingsTab',
    'HelpTab',
    'TradingTab',
    'MessageBus',
    'DatabaseConnector',
    'DataLoader',
    'DataService',
    'StockAIAgent',
    'RealTradingAgent'
] 