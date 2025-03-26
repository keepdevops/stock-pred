import json
import logging
from pathlib import Path

class ConfigurationManager:
    def __init__(self, config_file='config.json'):
        self.logger = logging.getLogger(__name__)
        self.config_file = Path(config_file)
        self.config = self._load_config()

    def _load_config(self):
        """Load configuration from JSON file."""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            else:
                self.logger.warning(f"Config file {self.config_file} not found. Using default configuration.")
                return self._create_default_config()
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return self._create_default_config()

    def _create_default_config(self):
        """Create and return default configuration."""
        default_config = {
            'database': {
                'path': 'stocks.db',
                'type': 'sqlite'
            },
            'data': {
                'source': 'yahoo',
                'symbols_file': 'symbols.txt',
                'start_date': '2020-01-01',
                'end_date': 'today'
            },
            'trading': {
                'mode': 'simulation',
                'initial_balance': 10000,
                'risk_per_trade': 0.02
            },
            'ai': {
                'model_type': 'lstm',
                'lookback_days': 60,
                'prediction_days': 5,
                'training': {
                    'epochs': 100,
                    'batch_size': 32,
                    'validation_split': 0.2
                }
            }
        }
        
        # Save default config
        try:
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=4)
            self.logger.info(f"Default configuration saved to {self.config_file}")
        except Exception as e:
            self.logger.error(f"Error saving default config: {e}")
        
        return default_config

    def get(self, key, default=None):
        """Get configuration value by key."""
        try:
            keys = key.split('.')
            value = self.config
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            self.logger.warning(f"Configuration key '{key}' not found. Using default: {default}")
            return default

    def set(self, key, value):
        """Set configuration value."""
        try:
            keys = key.split('.')
            config = self.config
            for k in keys[:-1]:
                config = config.setdefault(k, {})
            config[keys[-1]] = value
            
            # Save updated config
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            self.logger.info(f"Configuration updated: {key} = {value}")
        except Exception as e:
            self.logger.error(f"Error updating config: {e}")

    def get_database_config(self):
        """Get database configuration."""
        return self.get('database', {})

    def get_data_config(self):
        """Get data source configuration."""
        return self.get('data', {})

    def get_trading_config(self):
        """Get trading configuration."""
        return self.get('trading', {})

    def get_ai_config(self):
        """Get AI model configuration."""
        return self.get('ai', {}) 