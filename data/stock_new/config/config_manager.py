import json
from pathlib import Path
from typing import Dict, List, Any, Optional
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
    available_sources: List[DataSourceConfig]

@dataclass
class ParallelConfig:
    enabled: bool
    max_workers: int

@dataclass
class DataCollectionConfig:
    tickers: List[str]
    historical: HistoricalConfig
    realtime: RealtimeConfig
    parallel_processing: ParallelConfig

@dataclass
class CacheConfig:
    enabled: bool = True
    database: str = "sqlite"
    path: str = "cache/realtime_cache.db"
    expiry_seconds: int = 300
    max_entries: int = 1000

@dataclass
class CleaningConfig:
    lowercase: bool = True
    remove_special_chars: bool = True
    standardize_dates: str = "YYYY-MM-DD"
    fill_missing: str = "0"

@dataclass
class ValidationConfig:
    enabled: bool = True
    batch_size: int = 10
    required_columns: List[str] = None
    date_format: str = "YYYY-MM-DD"
    numeric_fields: List[str] = None

    def __post_init__(self):
        if self.required_columns is None:
            self.required_columns = ["date", "open", "high", "low", "close", "volume"]
        if self.numeric_fields is None:
            self.numeric_fields = ["open", "high", "low", "close", "volume"]

@dataclass
class DatabaseConfig:
    type: str = "duckdb"
    path: str = "data/market_data.duckdb"
    index_columns: List[str] = None

    def __post_init__(self):
        if self.index_columns is None:
            self.index_columns = ["date"]

@dataclass
class TickerMixingCombination:
    name: str
    tickers: List[str]
    fields: List[str]
    filters: Dict[str, str] = None
    aggregations: Dict[str, str] = None

    def __post_init__(self):
        if self.filters is None:
            self.filters = {}
        if self.aggregations is None:
            self.aggregations = {}

@dataclass
class TickerMixingConfig:
    combinations: List[TickerMixingCombination] = None
    output_format: str = "csv"
    output_path: str = "data/"

    def __post_init__(self):
        if self.combinations is None:
            self.combinations = [
                TickerMixingCombination(
                    name="default_mix",
                    tickers=["AAPL", "GOOG"],
                    fields=["date", "close"],
                    filters={},
                    aggregations={"close": "AVG"}
                )
            ]

@dataclass
class GuiConfig:
    theme: str = "clam"
    available_themes: List[str] = None
    window_size: str = "800x600"

    def __post_init__(self):
        if self.available_themes is None:
            self.available_themes = ["clam", "alt", "default"]

@dataclass
class LogFileConfig:
    name: str
    path: str
    level: str

@dataclass
class LoggingConfig:
    enabled: bool
    files: List[LogFileConfig]

@dataclass
class DataProcessingConfig:
    cleaning: CleaningConfig = None
    validation: ValidationConfig = None
    database: DatabaseConfig = None

    def __post_init__(self):
        if self.cleaning is None:
            self.cleaning = CleaningConfig()
        if self.validation is None:
            self.validation = ValidationConfig()
        if self.database is None:
            self.database = DatabaseConfig()

class ConfigurationManager:
    """Manages application configuration."""
    
    def __init__(self, config_path: str = "config/data_collection.json"):
        self.config_path = Path(config_path)
        self.logger = logging.getLogger("ConfigManager")
        
        if not self.config_path.exists():
            self.create_default_config()
        
        self.load_configuration()
    
    def create_default_config(self) -> None:
        """Create default configuration file."""
        default_config = {
            "system_config": {
                "version": "1.0.0",
                "description": "Configuration for market data collection",
                "date": "2025-03-17"
            },
            "data_collection": {
                "tickers": ["AAPL", "GOOG", "MSFT"],
                "historical": {
                    "enabled": True,
                    "points": 720,
                    "start_date": "2023-03-17",
                    "end_date": "2025-03-17"
                },
                "realtime": {
                    "enabled": False,
                    "source": "yahoo",
                    "available_sources": [
                        {
                            "name": "yahoo",
                            "type": "http",
                            "retry_attempts": 3,
                            "retry_backoff_base": 2
                        }
                    ]
                },
                "parallel_processing": {
                    "enabled": True,
                    "max_workers": 10
                }
            },
            "data_processing": {
                "cleaning": {
                    "lowercase": True,
                    "remove_special_chars": True,
                    "standardize_dates": "YYYY-MM-DD",
                    "fill_missing": "0"
                },
                "validation": {
                    "enabled": True,
                    "batch_size": 10,
                    "required_columns": [
                        "date", "open", "high", "low", "close", "volume"
                    ],
                    "date_format": "YYYY-MM-DD",
                    "numeric_fields": [
                        "open", "high", "low", "close", "volume"
                    ]
                },
                "database": {
                    "type": "duckdb",
                    "path": "data/market_data.duckdb",
                    "index_columns": ["date"]
                }
            },
            "gui_settings": {
                "theme": "clam",
                "available_themes": ["clam", "alt", "default"],
                "window_size": "800x600"
            },
            "ticker_mixing": {
                "combinations": [
                    {
                        "name": "tech_portfolio",
                        "tickers": ["AAPL", "GOOG"],
                        "fields": ["date", "close"],
                        "filters": {"date": "> '2024-01-01'"},
                        "aggregations": {"close": "AVG"}
                    }
                ],
                "output_format": "csv",
                "output_path": "data/"
            },
            "logging": {
                "enabled": True,
                "files": [
                    {
                        "name": "data_collection",
                        "path": "logs/data_collection.log",
                        "level": "INFO"
                    }
                ]
            }
        }
        
        self.config_path.parent.mkdir(exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=4)
        
        self.logger.info(f"Created default configuration file: {self.config_path}")
    
    def load_configuration(self) -> None:
        """Load configuration from JSON file."""
        try:
            with open(self.config_path) as f:
                config_data = json.load(f)
            
            # System Config
            self.system_config = SystemConfig(**config_data["system_config"])
            
            # Data Collection Config
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
            
            # Cache Config (optional)
            self.cache_settings = CacheConfig()
            if "cache_settings" in config_data:
                self.cache_settings = CacheConfig(**config_data["cache_settings"])
            
            # Data Processing Config (with defaults)
            self.data_processing = DataProcessingConfig()
            if "data_processing" in config_data:
                dp_config = config_data["data_processing"]
                if "cleaning" in dp_config:
                    self.data_processing.cleaning = CleaningConfig(**dp_config["cleaning"])
                if "validation" in dp_config:
                    self.data_processing.validation = ValidationConfig(**dp_config["validation"])
                if "database" in dp_config:
                    self.data_processing.database = DatabaseConfig(**dp_config["database"])
            
            # Ticker Mixing Config (optional)
            self.ticker_mixing = TickerMixingConfig()
            if "ticker_mixing" in config_data:
                tm_config = config_data["ticker_mixing"]
                combinations = [
                    TickerMixingCombination(**combo)
                    for combo in tm_config.get("combinations", [])
                ]
                self.ticker_mixing = TickerMixingConfig(
                    combinations=combinations,
                    output_format=tm_config.get("output_format", "csv"),
                    output_path=tm_config.get("output_path", "data/")
                )
            
            # GUI Settings (optional)
            self.gui_settings = GuiConfig()
            if "gui_settings" in config_data:
                self.gui_settings = GuiConfig(**config_data["gui_settings"])
            
            # Logging Config
            log_config = config_data.get("logging", {
                "enabled": True,
                "files": [{
                    "name": "data_collection",
                    "path": "logs/data_collection.log",
                    "level": "INFO"
                }]
            })
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
                    "cleaning": {
                        "lowercase": self.data_processing.cleaning.lowercase,
                        "remove_special_chars": self.data_processing.cleaning.remove_special_chars,
                        "standardize_dates": self.data_processing.cleaning.standardize_dates,
                        "fill_missing": self.data_processing.cleaning.fill_missing
                    },
                    "validation": {
                        "enabled": self.data_processing.validation.enabled,
                        "batch_size": self.data_processing.validation.batch_size,
                        "required_columns": self.data_processing.validation.required_columns,
                        "date_format": self.data_processing.validation.date_format,
                        "numeric_fields": self.data_processing.validation.numeric_fields
                    },
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

    def get_data_source_config(self, source_name: str) -> Optional[DataSourceConfig]:
        """Get configuration for a specific data source."""
        for source in self.data_collection.realtime.available_sources:
            if source.name == source_name:
                return source
        return None 

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
) 