import polars as pl
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union
import json
from datetime import datetime, timedelta

class TickerMixer:
    def __init__(self, database_path: str = "data/database/market_data.parquet",
                 config_path: str = "config.json"):
        self.logger = logging.getLogger(__name__)
        self.database_path = Path(database_path)
        self.config = self._load_config(config_path)
        
        # Initialize lazy frame for efficient queries
        self._initialize_lazy_frame()

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.warning(f"{config_path} not found, using defaults")
            return {"combinations": []}

    def _initialize_lazy_frame(self) -> None:
        """Initialize lazy frame for efficient queries"""
        try:
            if not self.database_path.exists():
                raise FileNotFoundError(f"Database not found at {self.database_path}")
            
            self.lazy_frame = pl.scan_parquet(self.database_path)
            self.logger.info("Lazy frame initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing lazy frame: {e}")
            raise

    def execute_combination(self, combination_name: str) -> Optional[pl.DataFrame]:
        """Execute a predefined combination query"""
        try:
            # Find the combination in config
            combination = next(
                (c for c in self.config["combinations"] if c["name"] == combination_name),
                None
            )
            
            if not combination:
                raise ValueError(f"Combination '{combination_name}' not found")
            
            # Build and execute query
            query = self._build_query(combination)
            result = self._execute_query(query)
            
            self.logger.info(f"Successfully executed combination '{combination_name}'")
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing combination '{combination_name}': {e}")
            return None

    def _build_query(self, combination: Dict) -> pl.LazyFrame:
        """Build a query from combination parameters"""
        query = self.lazy_frame
        
        # Filter tickers
        if combination.get("tickers"):
            query = query.filter(pl.col("ticker").is_in(combination["tickers"]))
        
        # Apply date filters
        if "date_range" in combination:
            date_range = combination["date_range"]
            if date_range.get("start"):
                query = query.filter(pl.col("date") >= date_range["start"])
            if date_range.get("end"):
                query = query.filter(pl.col("date") <= date_range["end"])
        
        # Select fields
        if combination.get("fields"):
            query = query.select(["date", "ticker"] + combination["fields"])
        
        # Apply custom filters
        if "filters" in combination:
            query = self._apply_filters(query, combination["filters"])
        
        # Apply aggregations
        if "aggregations" in combination:
            query = self._apply_aggregations(query, combination["aggregations"])
        
        return query

    def _apply_filters(self, query: pl.LazyFrame, filters: Dict) -> pl.LazyFrame:
        """Apply custom filters to query"""
        for field, conditions in filters.items():
            for operator, value in conditions.items():
                if operator == "gt":
                    query = query.filter(pl.col(field) > value)
                elif operator == "lt":
                    query = query.filter(pl.col(field) < value)
                elif operator == "eq":
                    query = query.filter(pl.col(field) == value)
                elif operator == "between":
                    query = query.filter(
                        (pl.col(field) >= value[0]) & (pl.col(field) <= value[1])
                    )
        return query

    def _apply_aggregations(self, query: pl.LazyFrame, aggregations: Dict) -> pl.LazyFrame:
        """Apply aggregations to query"""
        agg_expressions = []
        
        for field, operations in aggregations.items():
            for operation in operations:
                if operation == "sum":
                    agg_expressions.append(pl.col(field).sum().alias(f"{field}_sum"))
                elif operation == "mean":
                    agg_expressions.append(pl.col(field).mean().alias(f"{field}_mean"))
                elif operation == "std":
                    agg_expressions.append(pl.col(field).std().alias(f"{field}_std"))
                elif operation == "min":
                    agg_expressions.append(pl.col(field).min().alias(f"{field}_min"))
                elif operation == "max":
                    agg_expressions.append(pl.col(field).max().alias(f"{field}_max"))
        
        if agg_expressions:
            group_by = ["ticker"]
            if "time_window" in aggregations:
                window = aggregations["time_window"]
                if window == "daily":
                    group_by.append(pl.col("date").cast(pl.Date))
                elif window == "weekly":
                    group_by.append((pl.col("date").cast(pl.Date) - pl.duration(days=6)).alias("week"))
                elif window == "monthly":
                    group_by.append(pl.col("date").dt.strftime("%Y-%m").alias("month"))
            
            query = query.groupby(group_by).agg(agg_expressions)
        
        return query

    def _execute_query(self, query: pl.LazyFrame) -> pl.DataFrame:
        """Execute the lazy query and return results"""
        return query.collect()

    def create_combination(self, name: str, tickers: List[str], fields: List[str],
                         filters: Optional[Dict] = None,
                         aggregations: Optional[Dict] = None) -> None:
        """Create a new combination and save to config"""
        try:
            combination = {
                "name": name,
                "tickers": tickers,
                "fields": fields,
                "filters": filters or {},
                "aggregations": aggregations or {}
            }
            
            # Add or update combination
            existing_combinations = self.config.get("combinations", [])
            existing_idx = next(
                (i for i, c in enumerate(existing_combinations) if c["name"] == name),
                None
            )
            
            if existing_idx is not None:
                existing_combinations[existing_idx] = combination
            else:
                existing_combinations.append(combination)
            
            self.config["combinations"] = existing_combinations
            
            # Save updated config
            with open("config.json", 'w') as f:
                json.dump(self.config, f, indent=4)
            
            self.logger.info(f"Created combination '{name}'")
            
        except Exception as e:
            self.logger.error(f"Error creating combination: {e}")
            raise

    def get_available_combinations(self) -> List[str]:
        """Get list of available combination names"""
        return [c["name"] for c in self.config.get("combinations", [])]

    def calculate_correlation(self, ticker1: str, ticker2: str, field: str = "close",
                            window_days: int = 30) -> Optional[float]:
        """Calculate correlation between two tickers"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=window_days)
            
            query = (
                self.lazy_frame
                .filter(pl.col("ticker").is_in([ticker1, ticker2]))
                .filter(pl.col("date").cast(pl.Date).is_between(start_date, end_date))
                .select(["date", "ticker", field])
                .pivot(index="date", columns="ticker", values=field)
                .select([
                    pl.col(ticker1).corr(pl.col(ticker2)).alias("correlation")
                ])
            )
            
            result = query.collect()
            return result["correlation"][0] if not result.is_empty() else None
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation: {e}")
            return None

    def calculate_returns(self, tickers: List[str], period: str = "daily") -> Optional[pl.DataFrame]:
        """Calculate returns for given tickers"""
        try:
            query = self.lazy_frame.filter(pl.col("ticker").is_in(tickers))
            
            if period == "daily":
                returns = (
                    query.groupby("ticker")
                    .agg([
                        (pl.col("close") / pl.col("close").shift(1) - 1)
                        .alias("return")
                    ])
                )
            elif period == "weekly":
                returns = (
                    query.groupby(["ticker", pl.col("date").dt.strftime("%Y-%W")])
                    .agg([
                        (pl.col("close").last() / pl.col("close").first() - 1)
                        .alias("return")
                    ])
                )
            
            return returns.collect()
            
        except Exception as e:
            self.logger.error(f"Error calculating returns: {e}")
            return None

def main():
    # Example usage
    mixer = TickerMixer()
    
    # Create a sample combination
    mixer.create_combination(
        name="tech_stocks",
        tickers=["AAPL", "MSFT", "GOOGL"],
        fields=["close", "volume"],
        filters={"volume": {"gt": 1000000}},
        aggregations={
            "close": ["mean", "std"],
            "time_window": "weekly"
        }
    )
    
    # Execute the combination
    result = mixer.execute_combination("tech_stocks")
    if result is not None:
        print(result)
    
    # Calculate correlation
    correlation = mixer.calculate_correlation("AAPL", "MSFT")
    print(f"Correlation between AAPL and MSFT: {correlation}")

if __name__ == "__main__":
    main() 