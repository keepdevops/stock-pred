import json
import logging
from pathlib import Path
from typing import Dict, Any

class ConfigurationManager:
    def __init__(self, config_path: str = "config/config.json"):
        self.logger = logging.getLogger(__name__)
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.load_config()

    def load_config(self):
        """Load configuration from JSON file."""
        try:
            if not self.config_path.exists():
                self.logger.warning(f"Config file not found at {self.config_path}. Creating default config.")
                self.create_default_config()
            
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
                
            # Ensure required sections exist
            required_sections = ['data_collection', 'data_processing', 'visualization']
            for section in required_sections:
                if section not in self.config:
                    self.config[section] = {}
                    
            self.logger.info("Configuration loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            self.logger.info("Using default configuration")
            self.create_default_config()

    def create_default_config(self):
        """Create default configuration."""
        try:
            default_config = {
                "data_collection": {
                    "api_key": "",
                    "retry_attempts": 3,
                    "retry_delay": 5,
                    "timeout": 30
                },
                "data_processing": {
                    "database": {
                        "path": "data/market_data.duckdb",
                        "type": "duckdb"
                    },
                    "batch_size": 100,
                    "cache_enabled": True,
                    "cache_duration": 3600
                },
                "visualization": {
                    "theme": "dark",
                    "default_chart_type": "candlestick",
                    "indicators": ["SMA", "EMA", "RSI"]
                }
            }
            
            # Create config directory if it doesn't exist
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save default config
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
                
            self.config = default_config
            self.logger.info("Default configuration created")
            
        except Exception as e:
            self.logger.error(f"Error creating default configuration: {e}")
            raise

    def get(self, section: str, key: str = None, default: Any = None) -> Any:
        """Get configuration value."""
        try:
            if section not in self.config:
                self.logger.warning(f"Section '{section}' not found in config")
                return default
                
            if key is None:
                return self.config[section]
                
            return self.config[section].get(key, default)
            
        except Exception as e:
            self.logger.error(f"Error getting config value: {e}")
            return default

    def set(self, section: str, key: str, value: Any) -> bool:
        """Set configuration value."""
        try:
            if section not in self.config:
                self.config[section] = {}
                
            self.config[section][key] = value
            
            # Save updated config
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting config value: {e}")
            return False

    def __getattr__(self, name: str) -> Any:
        """Allow accessing config sections as attributes."""
        if name in self.config:
            return self.config[name]
        raise AttributeError(f"Configuration section '{name}' not found") 