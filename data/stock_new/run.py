import json
import logging
import tkinter as tk
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add the project root directory to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.gui.data_collector_gui import DataCollectorGUI
from src.config.config_manager import ConfigManager, DataCollectionConfig
from src.database.database_manager import DatabaseManager

class Config:
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self.load()

    def load(self):
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                self.data = json.load(f)
        except FileNotFoundError:
            self.data = self._create_default_config()
            self.save()

    def save(self):
        """Save current configuration to JSON file."""
        with open(self.config_path, 'w') as f:
            json.dump(self.data, f, indent=2)

    def get(self, key, default=None):
        """Get a configuration value."""
        try:
            keys = key.split('.')
            value = self.data
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key, value):
        """Set a configuration value."""
        keys = key.split('.')
        data = self.data
        for k in keys[:-1]:
            if k not in data:
                data[k] = {}
            data = data[k]
        data[keys[-1]] = value

    def _create_default_config(self):
        """Create default configuration."""
        return {
            "database": {
                "path": "data/market_data.duckdb",
                "backup_path": "data/backups/"
            },
            "tickers": [],
            "data_collection": {
                "historical": {
                    "default_start_date": "2023-03-18",
                    "default_end_date": "2025-03-18",
                    "batch_size": 100
                },
                "realtime": {
                    "default_interval": 60
                }
            },
            "logging": {
                "level": "INFO",
                "file_path": "logs/app.log"
            },
            "gui": {
                "window": {
                    "title": "Stock Market Data Collector",
                    "width": 800,
                    "height": 600
                }
            }
        }

def setup_logging():
    """Setup logging configuration."""
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('logs/app.log')
    
    # Create formatters
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(log_format)
    file_handler.setFormatter(log_format)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    # Initialize database manager
    db_manager = DatabaseManager()

    # Create and run GUI
    app = DataCollectorGUI(db_manager=db_manager, logger=logger)
    app.run()  # This line is crucial to show the GUI

if __name__ == "__main__":
    main() 