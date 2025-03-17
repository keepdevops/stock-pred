import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class ModelConfig:
    """Configuration for neural network models."""
    model_type: str = 'LSTM'
    sequence_length: int = 10
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    layers: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.layers is None:
            self.layers = {
                'lstm_units': [50, 50],
                'dense_units': [32],
                'dropout': 0.2
            }

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    learning_rate: float = 0.001
    optimizer: str = 'adam'
    loss: str = 'mse'
    metrics: list = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ['mae', 'mse']

@dataclass
class TradingConfig:
    """Configuration for trading parameters."""
    initial_budget: float = 10000.0
    max_position_size: float = 0.1
    max_open_trades: int = 5
    profit_tiers: list = None
    stop_loss: float = -2.0
    
    def __post_init__(self):
        if self.profit_tiers is None:
            self.profit_tiers = [5.0, 8.0, 9.0]

class ConfigManager:
    """Manage application configuration."""
    
    def __init__(self, config_dir: Path = None):
        self.config_dir = config_dir or Path.home() / '.stock_analyzer'
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_config = ModelConfig()
        self.training_config = TrainingConfig()
        self.trading_config = TradingConfig()
        
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from file."""
        config_file = self.config_dir / 'config.json'
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
                
                self.model_config = ModelConfig(**config.get('model', {}))
                self.training_config = TrainingConfig(**config.get('training', {}))
                self.trading_config = TradingConfig(**config.get('trading', {}))
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        config = {
            'model': asdict(self.model_config),
            'training': asdict(self.training_config),
            'trading': asdict(self.trading_config)
        }
        
        config_file = self.config_dir / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
    
    def update_config(
        self,
        section: str,
        updates: Dict[str, Any]
    ) -> None:
        """Update specific configuration section."""
        config_map = {
            'model': self.model_config,
            'training': self.training_config,
            'trading': self.trading_config
        }
        
        if section not in config_map:
            raise ValueError(f"Unknown configuration section: {section}")
        
        for key, value in updates.items():
            if hasattr(config_map[section], key):
                setattr(config_map[section], key, value)
        
        self.save_config()
    
    def get_config(self, section: str) -> Dict[str, Any]:
        """Get configuration for specific section."""
        config_map = {
            'model': self.model_config,
            'training': self.training_config,
            'trading': self.trading_config
        }
        
        if section not in config_map:
            raise ValueError(f"Unknown configuration section: {section}")
        
        return asdict(config_map[section]) 