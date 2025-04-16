"""
Tabs module for the Stock Market Analyzer application.
"""

from .base_tab import BaseTab
from .monitor_tab import MonitorTab
from .data_tab import DataTab
from .analysis_tab import AnalysisTab
from .charts_tab import ChartsTab
from .models_tab import ModelsTab
from .predictions_tab import PredictionsTab
from .settings_tab import SettingsTab
from .trading_tab import TradingTab
from .import_tab import ImportTab
from .help_tab import HelpTab

__all__ = [
    'BaseTab',
    'MonitorTab',
    'DataTab',
    'AnalysisTab',
    'ChartsTab',
    'ModelsTab',
    'PredictionsTab',
    'SettingsTab',
    'TradingTab',
    'ImportTab',
    'HelpTab'
] 