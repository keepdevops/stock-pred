import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigurationManager:
    """Class for managing application configuration."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.config_file = config_file or 'config.json'
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default."""
        try:
            config_path = Path(self.config_file)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                self.logger.info(f"Configuration loaded from {self.config_file}")
                return config
            else:
                config = self._create_default_config()
                self._save_config(config)
                self.logger.info(f"Default configuration created and saved to {self.config_file}")
                return config
                
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            config = self._create_default_config()
            self.logger.info("Using default configuration")
            return config
            
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration."""
        return {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'name': 'stock_market',
                'user': 'postgres',
                'password': '',
                'table_prefix': 'stock_'
            },
            'data_source': {
                'provider': 'yahoo',
                'api_key': '',
                'start_date': '2020-01-01',
                'end_date': 'today',
                'symbols_file': 'symbols.txt'
            },
            'ai_model': {
                'model_type': 'lstm',
                'sequence_length': 60,
                'features': [
                    'open',
                    'high',
                    'low',
                    'close',
                    'volume'
                ],
                'target': 'close',
                'train_split': 0.8,
                'batch_size': 32,
                'epochs': 100,
                'learning_rate': 0.001,
                'model_dir': 'models',
                'use_technical_indicators': True,
                'additional_features': []
            },
            'trading': {
                'mode': 'simulation',
                'initial_capital': 100000.0,
                'position_size': 0.1,
                'stop_loss': 0.02,
                'take_profit': 0.05,
                'max_positions': 5
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': 'stock_market.log'
            }
        }
        
    def _save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
            self.logger.info(f"Configuration saved to {self.config_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration."""
        return self.config.get('database', {})
        
    def get_data_config(self) -> Dict[str, Any]:
        """Get data source configuration."""
        return self.config.get('data_source', {})
        
    def get_ai_config(self) -> Dict[str, Any]:
        """Get AI model configuration."""
        return self.config.get('ai_model', {})
        
    def get_trading_config(self) -> Dict[str, Any]:
        """Get trading configuration."""
        return self.config.get('trading', {})
        
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.config.get('logging', {})
        
    def update_config(self, section: str, key: str, value: Any) -> bool:
        """Update configuration value."""
        try:
            if section not in self.config:
                self.config[section] = {}
            self.config[section][key] = value
            self._save_config(self.config)
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")
            return False
            
    def get_config(self) -> Dict[str, Any]:
        """Get entire configuration."""
        return self.config 