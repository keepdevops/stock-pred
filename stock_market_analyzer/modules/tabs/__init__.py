"""
Tabs module for the Stock Market Analyzer application.
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from .base_tab import BaseTab
from .data_tab import DataTab
from .analysis_tab import AnalysisTab
from .charts_tab import ChartsTab
from .models_tab import ModelsTab
from .predictions_tab import PredictionsTab
from .import_tab import ImportTab
from .settings_tab import SettingsTab
from .help_tab import HelpTab
from .trading_tab import TradingTab

# Map of tab names to their classes
TAB_CLASSES = {
    "DataTab": DataTab,
    "AnalysisTab": AnalysisTab,
    "ChartsTab": ChartsTab,
    "ModelsTab": ModelsTab,
    "PredictionsTab": PredictionsTab,
    "ImportTab": ImportTab,
    "SettingsTab": SettingsTab,
    "HelpTab": HelpTab,
    "TradingTab": TradingTab
}

def get_tab_class(tab_name: str) -> type:
    """Get the tab class for a given tab name.
    
    Args:
        tab_name: The name of the tab to get the class for.
        
    Returns:
        The tab class.
        
    Raises:
        ValueError: If the tab name is not found.
    """
    if tab_name not in TAB_CLASSES:
        raise ValueError(f"Unknown tab name: {tab_name}")
    return TAB_CLASSES[tab_name]

__all__ = [
    "BaseTab",
    "DataTab",
    "AnalysisTab",
    "ChartsTab",
    "ModelsTab",
    "PredictionsTab",
    "ImportTab",
    "SettingsTab",
    "HelpTab",
    "TradingTab",
    "get_tab_class"
] 