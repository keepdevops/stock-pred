import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

@dataclass
class SystemConfig:
    version: str
    description: str
    date: str

@dataclass
class HistoricalConfig:
    enabled: bool
    points: int
    start_date: str
    end_date: str

@dataclass
class DataSourceConfig:
    name: str
    type: str
    retry_attempts: int
    retry_backoff_base: int
    api_key: Optional[str] = None
    secret_key: Optional[str] = None

@dataclass
class RealtimeConfig:
    enabled: bool
    source: str
    available_sources: list[DataSourceConfig]

@dataclass
class ParallelConfig:
    enabled: bool
    max_workers: int

@dataclass
class DataCollectionConfig:
    tickers: list[str]
    historical: HistoricalConfig
    realtime: RealtimeConfig
    parallel_processing: ParallelConfig

@dataclass
class DatabaseConfig:
    type: str
    path: str
    index_columns: list[str]

@dataclass
class DataProcessingConfig:
    database: DatabaseConfig

@dataclass
class LogFileConfig:
    name: str
    path: str
    level: str

@dataclass
class LoggingConfig:
    enabled: bool
    files: list[LogFileConfig]

class ConfigurationManager:
    """Manages application configuration."""
    
    def __init__(self, config_path: str = "config/data_collection.json"):
        self.config_path = Path(config_path)
        self.logger = logging.getLogger("ConfigManager")
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        self.load_configuration()
    
    def load_configuration(self) -> None:
        """Load configuration from JSON file."""
        try:
            with open(self.config_path) as f:
                config_data = json.load(f)
            
            self.system_config = SystemConfig(**config_data["system_config"])
            
            # Load data collection config
            dc_config = config_data["data_collection"]
            self.data_collection = DataCollectionConfig(
                tickers=dc_config["tickers"],
                historical=HistoricalConfig(**dc_config["historical"]),
                realtime=RealtimeConfig(
                    enabled=dc_config["realtime"]["enabled"],
                    source=dc_config["realtime"]["source"],
                    available_sources=[
                        DataSourceConfig(**src) 
                        for src in dc_config["realtime"]["available_sources"]
                    ]
                ),
                parallel_processing=ParallelConfig(**dc_config["parallel_processing"])
            )
            
            # Load data processing config
            self.data_processing = DataProcessingConfig(
                database=DatabaseConfig(**config_data["data_processing"]["database"])
            )
            
            # Load logging config
            log_config = config_data["logging"]
            self.logging = LoggingConfig(
                enabled=log_config["enabled"],
                files=[LogFileConfig(**f) for f in log_config["files"]]
            )
            
            self.logger.info("Configuration loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            raise
    
    def save_configuration(self) -> None:
        """Save current configuration to file."""
        try:
            config_data = {
                "system_config": {
                    "version": self.system_config.version,
                    "description": self.system_config.description,
                    "date": self.system_config.date
                },
                "data_collection": {
                    "tickers": self.data_collection.tickers,
                    "historical": {
                        "enabled": self.data_collection.historical.enabled,
                        "points": self.data_collection.historical.points,
                        "start_date": self.data_collection.historical.start_date,
                        "end_date": self.data_collection.historical.end_date
                    },
                    "realtime": {
                        "enabled": self.data_collection.realtime.enabled,
                        "source": self.data_collection.realtime.source,
                        "available_sources": [
                            {
                                "name": src.name,
                                "type": src.type,
                                "retry_attempts": src.retry_attempts,
                                "retry_backoff_base": src.retry_backoff_base,
                                "api_key": src.api_key,
                                "secret_key": src.secret_key
                            }
                            for src in self.data_collection.realtime.available_sources
                        ]
                    },
                    "parallel_processing": {
                        "enabled": self.data_collection.parallel_processing.enabled,
                        "max_workers": self.data_collection.parallel_processing.max_workers
                    }
                },
                "data_processing": {
                    "database": {
                        "type": self.data_processing.database.type,
                        "path": self.data_processing.database.path,
                        "index_columns": self.data_processing.database.index_columns
                    }
                },
                "logging": {
                    "enabled": self.logging.enabled,
                    "files": [
                        {
                            "name": f.name,
                            "path": f.path,
                            "level": f.level
                        }
                        for f in self.logging.files
                    ]
                }
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(config_data, f, indent=4)
            
            self.logger.info("Configuration saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {str(e)}")
            raise
    
    def update_section(self, section: str, updates: Dict[str, Any]) -> None:
        """Update a specific section of the configuration."""
        if not hasattr(self, section):
            raise ValueError(f"Invalid configuration section: {section}")
        
        current_section = getattr(self, section)
        for key, value in updates.items():
            if hasattr(current_section, key):
                setattr(current_section, key, value)
        
        self.save_configuration() 