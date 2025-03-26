"""
Stock Market Analyzer - A comprehensive tool for stock market analysis and trading.
"""

import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Version information
__version__ = '1.0.0'
__author__ = 'Stock Market Analyzer Team'
__license__ = 'MIT'

# Import main components
from .config import ConfigurationManager

__all__ = ['ConfigurationManager'] 