import sys
import logging
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

from stock_market_analyzer.config.config_manager import ConfigurationManager
from stock_market_analyzer.modules.gui import StockGUI
from stock_market_analyzer.modules.database import DatabaseConnector
from stock_market_analyzer.modules.data_loader import DataLoader
from stock_market_analyzer.modules.stock_ai_agent import StockAIAgent
from stock_market_analyzer.modules.trading.real_trading_agent import RealTradingAgent

# Rest of the code remains the same 