import json
import logging
import tkinter as tk
import os
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.gui.data_collector_gui import DataCollectorGUI
from src.config.config_manager import ConfigManager
from src.database.database_connector import DataCollector

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

def setup_logging(config):
    """Setup logging configuration."""
    log_path = Path(config.get('logging.file_path', 'logs/app.log'))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=config.get('logging.level', 'INFO'),
        format=config.get('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

def main():
    try:
        # Load configuration
        config = Config()
        
        # Setup logging
        setup_logging(config)
        
        # Create database directory if it doesn't exist
        db_path = Path(config.get('database.path'))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        db = DataCollector(str(db_path))
        
        # Create and configure root window
        root = tk.Tk()
        root.title(config.get('gui.window.title', 'Stock Market Data Collector'))
        root.geometry(f"{config.get('gui.window.width', 800)}x{config.get('gui.window.height', 600)}")
        
        # Initialize GUI
        app = DataCollectorGUI(root, config, db)
        
        # Start application
        root.mainloop()
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Error in main: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 