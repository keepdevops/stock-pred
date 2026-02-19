import json
from pathlib import Path
from types import SimpleNamespace
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
    
    def __init__(self, config_path: str = "config/system_config.json"):
        self.logger = logging.getLogger("ConfigManager")
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration from JSON file."""
        try:
            config_file = Path(self.config_path)
            
            if not config_file.exists():
                self.create_default_config()
            
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            self.logger.info("Configuration loaded successfully")
            return config
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            return self.get_default_config()
    
    def create_default_config(self):
        """Create default configuration file."""
        config = self.get_default_config()
        
        config_file = Path(self.config_path)
        config_file.parent.mkdir(exist_ok=True)
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
    
    def get_default_config(self):
        """Get default configuration."""
        return {
            "database": {
                "path": "data/market_data.duckdb",
                "type": "duckdb"
            },
            "data_collection": {
                "period": "2y",
                "interval": "1d",
                "retries": 3,
                "retry_delay": 5
            },
            "data_processing": {
                "database": {
                    "type": "duckdb",
                    "path": "data/market_data.duckdb",
                    "index_columns": ["date"]
                }
            },
            "gui": {
                "window_size": "1200x800",
                "theme": "default"
            }
        }

    @property
    def data_processing(self):
        """Get data processing configuration (database, cleaning, validation)."""
        dp = self.config.get("data_processing") or {}
        db = dp.get("database") or {}
        # Support flat config keys (e.g. system_config.json with db_path)
        path = db.get("path") or self.config.get("db_path") or "data/market_data.duckdb"
        return SimpleNamespace(
            database=SimpleNamespace(
                path=path,
                type=db.get("type", "duckdb"),
                index_columns=db.get("index_columns", ["date"])
            ),
            cleaning=getattr(self, "_cleaning_config", None),
            validation=getattr(self, "_validation_config", None),
        )

    @property
    def gui_settings(self):
        """Get GUI settings (window_size, theme) for main.py run()."""
        gui = self.config.get("gui") or {}
        return SimpleNamespace(
            window_size=gui.get("window_size", "1200x800"),
            theme=gui.get("theme", "default")
        )

    @property
    def database(self):
        """Get database configuration."""
        return self.config.get("database", {})

    @property
    def data_collection(self):
        """Get data collection configuration."""
        base = self.config.get("data_collection") or {}
        # Support flat config (e.g. system_config.json: data_period, data_interval)
        if not base and (self.config.get("data_period") or self.config.get("data_interval")):
            base = {
                "period": self.config.get("data_period", "2y"),
                "interval": self.config.get("data_interval", "1d"),
                "retries": self.config.get("retries", 3),
                "retry_delay": self.config.get("retry_delay", 5),
            }
        if not base:
            base = {"period": "2y", "interval": "1d", "retries": 3, "retry_delay": 5}
        return base

    @property
    def gui(self):
        """Get GUI configuration."""
        return self.config.get("gui", {})

    def get_data_source_config(self, source_name: str) -> Optional[DataSourceConfig]:
        """Get configuration for a specific data source."""
        for source in self.data_collection.get("realtime", {}).get("available_sources", []):
            if source.name == source_name:
                return source
        return None 

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
) 