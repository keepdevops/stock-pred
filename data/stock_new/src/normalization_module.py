import numpy as np
import polars as pl
from typing import List, Dict, Optional, Tuple, Union
import logging
from datetime import datetime
import json
from pathlib import Path
import scipy.stats as stats
import time

class TickerNormalizer:
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.cache = {}
        self.cache_config = {
            "max_size": 1000,
            "ttl": 3600  # 1 hour
        }
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load normalization configuration"""
        default_config = {
            "ticker_normalization": {
                "normalization_methods": {
                    "softmax": {"enabled": True, "parameters": {"temperature": 1.0}},
                    "sigmoid": {"enabled": True, "parameters": {"slope": 1.0, "shift": 0.0}},
                    "min_max": {"enabled": True, "parameters": {"feature_range": [0, 1]}},
                    "z_score": {"enabled": True, "parameters": {"output_range": "unbounded"}}
                }
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading config: {e}")
                return default_config
        return default_config

    def normalize_data(self, 
                      data: Union[pl.DataFrame, np.ndarray], 
                      method: str,
                      column: Optional[str] = None,
                      **kwargs) -> Union[pl.DataFrame, np.ndarray]:
        """
        Normalize data using specified method
        
        Args:
            data: Input data (DataFrame or numpy array)
            method: Normalization method ('softmax', 'sigmoid', 'min_max', 'z_score')
            column: Column name if using DataFrame
            **kwargs: Additional parameters for normalization
        """
        try:
            # Convert DataFrame column to numpy array if needed
            if isinstance(data, pl.DataFrame) and column:
                values = data[column].to_numpy()
            else:
                values = np.array(data)
            
            # Apply normalization
            if method == "softmax":
                normalized = self._softmax_normalize(values, **kwargs)
            elif method == "sigmoid":
                normalized = self._sigmoid_normalize(values, **kwargs)
            elif method == "min_max":
                normalized = self._minmax_normalize(values, **kwargs)
            elif method == "z_score":
                normalized = self._zscore_normalize(values, **kwargs)
            elif method == "robust_scale":
                normalized = self._robust_scale_normalize(values, **kwargs)
            elif method == "decimal_scaling":
                normalized = self._decimal_scaling_normalize(values)
            elif method == "log":
                normalized = self._log_normalize(values, **kwargs)
            else:
                raise ValueError(f"Unknown normalization method: {method}")
            
            # Return in original format
            if isinstance(data, pl.DataFrame) and column:
                return data.with_columns([
                    pl.Series(name=f"{column}_normalized", values=normalized)
                ])
            return normalized
            
        except Exception as e:
            self.logger.error(f"Error in normalize_data: {e}")
            raise

    def _softmax_normalize(self, 
                          values: np.ndarray, 
                          temperature: float = 1.0) -> np.ndarray:
        """
        Apply softmax normalization
        
        Args:
            values: Input values
            temperature: Softmax temperature parameter
        """
        try:
            # Avoid overflow by subtracting max
            exp_values = np.exp((values - np.max(values)) / temperature)
            return exp_values / np.sum(exp_values)
            
        except Exception as e:
            self.logger.error(f"Error in softmax normalization: {e}")
            raise

    def _sigmoid_normalize(self, 
                         values: np.ndarray, 
                         slope: float = 1.0, 
                         shift: float = 0.0) -> np.ndarray:
        """
        Apply sigmoid normalization
        
        Args:
            values: Input values
            slope: Sigmoid curve slope
            shift: Horizontal shift
        """
        try:
            return 1 / (1 + np.exp(-slope * (values - shift)))
            
        except Exception as e:
            self.logger.error(f"Error in sigmoid normalization: {e}")
            raise

    def _minmax_normalize(self, 
                         values: np.ndarray, 
                         feature_range: Tuple[float, float] = (0, 1),
                         min_value: Optional[float] = None,
                         max_value: Optional[float] = None) -> np.ndarray:
        """
        Apply min-max normalization
        
        Args:
            values: Input values
            feature_range: Output range tuple (min, max)
            min_value: Optional preset minimum value
            max_value: Optional preset maximum value
        """
        try:
            min_val = min_value if min_value is not None else np.min(values)
            max_val = max_value if max_value is not None else np.max(values)
            
            if min_val == max_val:
                return np.full_like(values, feature_range[0])
            
            scaled = (values - min_val) / (max_val - min_val)
            return scaled * (feature_range[1] - feature_range[0]) + feature_range[0]
            
        except Exception as e:
            self.logger.error(f"Error in min-max normalization: {e}")
            raise

    def _zscore_normalize(self, 
                         values: np.ndarray,
                         mean: Optional[float] = None,
                         std: Optional[float] = None) -> np.ndarray:
        """
        Apply z-score normalization
        
        Args:
            values: Input values
            mean: Optional preset mean value
            std: Optional preset standard deviation
        """
        try:
            mean_val = mean if mean is not None else np.mean(values)
            std_val = std if std is not None else np.std(values)
            
            if std_val == 0:
                return np.zeros_like(values)
            
            return (values - mean_val) / std_val
            
        except Exception as e:
            self.logger.error(f"Error in z-score normalization: {e}")
            raise

    def _robust_scale_normalize(self, 
                              values: np.ndarray, 
                              quantile_range: Tuple[float, float] = (25.0, 75.0)) -> np.ndarray:
        """
        Robust scaling using quantiles instead of min-max
        """
        try:
            q_low, q_high = np.percentile(values, quantile_range)
            scale = q_high - q_low
            if scale == 0:
                scale = 1.0
            return (values - q_low) / scale
        except Exception as e:
            self.logger.error(f"Error in robust scaling: {e}")
            raise

    def _decimal_scaling_normalize(self, values: np.ndarray) -> np.ndarray:
        """
        Normalize by moving decimal points
        """
        try:
            max_abs = np.max(np.abs(values))
            n_digits = len(str(int(max_abs)))
            return values / (10 ** n_digits)
        except Exception as e:
            self.logger.error(f"Error in decimal scaling: {e}")
            raise

    def _log_normalize(self, 
                      values: np.ndarray, 
                      base: float = np.e,
                      handle_zeros: bool = True) -> np.ndarray:
        """
        Apply log normalization with zero handling
        """
        try:
            if handle_zeros:
                min_nonzero = np.min(values[values > 0]) if np.any(values > 0) else 1
                values = values + min_nonzero
            return np.log(values) / np.log(base)
        except Exception as e:
            self.logger.error(f"Error in log normalization: {e}")
            raise

    def normalize_ticker_data(self, 
                            df: pl.DataFrame, 
                            columns: List[str],
                            methods: List[str]) -> pl.DataFrame:
        """
        Normalize multiple columns with specified methods
        
        Args:
            df: Input DataFrame
            columns: List of columns to normalize
            methods: List of normalization methods for each column
        """
        try:
            result = df.clone()
            
            for column, method in zip(columns, methods):
                if method not in self.config["ticker_normalization"]["normalization_methods"]:
                    raise ValueError(f"Unknown normalization method: {method}")
                
                if not self.config["ticker_normalization"]["normalization_methods"][method]["enabled"]:
                    self.logger.warning(f"Method {method} is disabled in config")
                    continue
                
                params = self.config["ticker_normalization"]["normalization_methods"][method]["parameters"]
                result = self.normalize_data(result, method, column, **params)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in normalize_ticker_data: {e}")
            raise

    def save_normalization_metadata(self, 
                                  ticker: str,
                                  data_source: str,
                                  start_date: str,
                                  end_date: str) -> None:
        """Save normalization metadata to config"""
        try:
            metadata = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "data_source": data_source,
                "time_period": {
                    "start_date": start_date,
                    "end_date": end_date
                }
            }
            
            self.config["ticker_normalization"]["ticker_symbol"] = ticker
            self.config["ticker_normalization"]["metadata"] = metadata
            
            # Save updated config
            if hasattr(self, 'config_path') and self.config_path:
                with open(self.config_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
                    
        except Exception as e:
            self.logger.error(f"Error saving normalization metadata: {e}")
            raise

    def rolling_normalize(self, 
                         df: pl.DataFrame,
                         column: str,
                         window_size: int,
                         method: str) -> pl.DataFrame:
        """
        Apply normalization over rolling windows
        """
        try:
            result = df.clone()
            values = df[column].to_numpy()
            normalized = np.zeros_like(values, dtype=float)
            
            for i in range(len(values)):
                start_idx = max(0, i - window_size + 1)
                window = values[start_idx:i + 1]
                normalized[i] = self.normalize_data(window, method)[-1]
            
            return result.with_columns([
                pl.Series(name=f"{column}_rolling_norm", values=normalized)
            ])
        except Exception as e:
            self.logger.error(f"Error in rolling normalization: {e}")
            raise

    def cross_sectional_normalize(self,
                                df: pl.DataFrame,
                                columns: List[str],
                                method: str,
                                groupby: str = "date") -> pl.DataFrame:
        """
        Normalize across multiple tickers at each timestamp
        """
        try:
            result = df.clone()
            
            for col in columns:
                normalized = (
                    df.groupby(groupby)
                    .agg([
                        pl.col(col)
                        .map_elements(lambda x: self.normalize_data(x, method))
                        .alias(f"{col}_cross_norm")
                    ])
                )
                result = result.join(normalized, on=groupby)
            
            return result
        except Exception as e:
            self.logger.error(f"Error in cross-sectional normalization: {e}")
            raise

    def adaptive_normalize(self,
                         df: pl.DataFrame,
                         column: str,
                         volatility_window: int = 20) -> pl.DataFrame:
        """
        Normalize based on local volatility
        """
        try:
            result = df.clone()
            values = df[column].to_numpy()
            
            # Calculate rolling volatility
            rolling_std = np.array([
                np.std(values[max(0, i-volatility_window):i+1])
                for i in range(len(values))
            ])
            
            # Adjust normalization based on volatility
            normalized = values / (rolling_std + 1e-8)  # Avoid division by zero
            
            return result.with_columns([
                pl.Series(name=f"{column}_adaptive_norm", values=normalized)
            ])
        except Exception as e:
            self.logger.error(f"Error in adaptive normalization: {e}")
            raise

    def compute_normalization_stats(self,
                                  original: np.ndarray,
                                  normalized: np.ndarray) -> Dict:
        """
        Compute statistics about the normalization
        """
        try:
            return {
                "original_stats": {
                    "mean": float(np.mean(original)),
                    "std": float(np.std(original)),
                    "min": float(np.min(original)),
                    "max": float(np.max(original)),
                    "skew": float(stats.skew(original)),
                    "kurtosis": float(stats.kurtosis(original))
                },
                "normalized_stats": {
                    "mean": float(np.mean(normalized)),
                    "std": float(np.std(normalized)),
                    "min": float(np.min(normalized)),
                    "max": float(np.max(normalized)),
                    "skew": float(stats.skew(normalized)),
                    "kurtosis": float(stats.kurtosis(normalized))
                },
                "transformation_quality": {
                    "correlation": float(np.corrcoef(original, normalized)[0, 1]),
                    "information_retention": float(
                        1 - np.sum((original - normalized)**2) / np.sum(original**2)
                    )
                }
            }
        except Exception as e:
            self.logger.error(f"Error computing normalization stats: {e}")
            raise

    def validate_normalization(self,
                             original: np.ndarray,
                             normalized: np.ndarray,
                             method: str) -> bool:
        """
        Validate normalization results
        """
        try:
            if method == "min_max":
                return (
                    np.allclose(np.min(normalized), 0) and
                    np.allclose(np.max(normalized), 1)
                )
            elif method == "z_score":
                return (
                    np.allclose(np.mean(normalized), 0, atol=1e-2) and
                    np.allclose(np.std(normalized), 1, atol=1e-2)
                )
            elif method == "softmax":
                return np.allclose(np.sum(normalized), 1)
            elif method == "sigmoid":
                return np.all((normalized >= 0) & (normalized <= 1))
            return True
        except Exception as e:
            self.logger.error(f"Error validating normalization: {e}")
            raise

    def _get_cache_key(self, values: np.ndarray, method: str, **kwargs) -> str:
        """Generate cache key from input parameters"""
        return f"{hash(values.tobytes())}:{method}:{hash(str(kwargs))}"

    def normalize_data_with_cache(self,
                                values: np.ndarray,
                                method: str,
                                **kwargs) -> np.ndarray:
        """
        Normalize with caching for performance
        """
        try:
            cache_key = self._get_cache_key(values, method, **kwargs)
            
            if cache_key in self.cache:
                cache_entry = self.cache[cache_key]
                if time.time() - cache_entry["timestamp"] < self.cache_config["ttl"]:
                    return cache_entry["result"]
            
            result = self.normalize_data(values, method, **kwargs)
            
            # Cache the result
            self.cache[cache_key] = {
                "result": result,
                "timestamp": time.time()
            }
            
            # Clean cache if too large
            if len(self.cache) > self.cache_config["max_size"]:
                oldest_key = min(
                    self.cache.keys(),
                    key=lambda k: self.cache[k]["timestamp"]
                )
                del self.cache[oldest_key]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in cached normalization: {e}")
            raise

def main():
    # Example usage
    normalizer = TickerNormalizer()
    
    # Create sample data
    data = pl.DataFrame({
        "date": pd.date_range(start="2023-01-01", periods=100),
        "close": np.random.randn(100) * 10 + 100,
        "volume": np.random.randint(1000, 10000, 100)
    })
    
    # Normalize multiple columns
    normalized_data = normalizer.normalize_ticker_data(
        df=data,
        columns=["close", "volume"],
        methods=["min_max", "z_score"]
    )
    
    print("Original data:")
    print(data.head())
    print("\nNormalized data:")
    print(normalized_data.head())
    
    # Save metadata
    normalizer.save_normalization_metadata(
        ticker="AAPL",
        data_source="yahoo",
        start_date="2023-01-01",
        end_date="2023-04-01"
    )

if __name__ == "__main__":
    main() 