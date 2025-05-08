import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class TradingStrategy:
    """Represents a trading strategy with its parameters."""
    name: str
    description: str
    parameters: Dict[str, Any]
    performance_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        self.performance_metrics = {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0
        }

class StrategyOptimizer:
    """Optimizes trading strategies using historical data."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the strategy optimizer.
        
        Args:
            config: Configuration dictionary containing optimization parameters
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.strategies = []
        self.best_strategy = None
        
    def add_strategy(self, strategy: TradingStrategy) -> None:
        """Add a trading strategy to optimize.
        
        Args:
            strategy: TradingStrategy object to add
        """
        self.strategies.append(strategy)
        
    def optimize_strategies(self, historical_data: pd.DataFrame) -> List[TradingStrategy]:
        """Optimize all trading strategies using historical data.
        
        Args:
            historical_data: DataFrame containing historical price data
            
        Returns:
            List of optimized strategies sorted by performance
        """
        self.logger.info("Starting strategy optimization...")
        
        for strategy in self.strategies:
            # Run backtest for each strategy
            results = self._backtest_strategy(strategy, historical_data)
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(results)
            strategy.performance_metrics = metrics
            
            self.logger.info(f"Strategy {strategy.name} performance: {metrics}")
            
        # Sort strategies by total return
        self.strategies.sort(key=lambda x: x.performance_metrics['total_return'], reverse=True)
        self.best_strategy = self.strategies[0]
        
        return self.strategies
        
    def _backtest_strategy(self, strategy: TradingStrategy, data: pd.DataFrame) -> pd.DataFrame:
        """Run backtest for a specific strategy.
        
        Args:
            strategy: TradingStrategy to backtest
            data: Historical price data
            
        Returns:
            DataFrame containing backtest results
        """
        results = pd.DataFrame(index=data.index)
        results['position'] = 0
        results['returns'] = 0.0
        results['portfolio_value'] = self.config['initial_capital']
        
        # Apply strategy-specific logic
        if strategy.name == 'Moving Average Crossover':
            results = self._ma_crossover_strategy(data, strategy.parameters, results)
        elif strategy.name == 'RSI Strategy':
            results = self._rsi_strategy(data, strategy.parameters, results)
        elif strategy.name == 'Bollinger Bands':
            results = self._bollinger_bands_strategy(data, strategy.parameters, results)
        elif strategy.name == 'AI-Powered':
            results = self._ai_strategy(data, strategy.parameters, results)
            
        return results
        
    def _ma_crossover_strategy(self, data: pd.DataFrame, params: Dict[str, Any], results: pd.DataFrame) -> pd.DataFrame:
        """Implement Moving Average Crossover strategy."""
        short_window = params.get('short_window', 20)
        long_window = params.get('long_window', 50)
        
        # Calculate moving averages
        data['SMA_short'] = data['close'].rolling(window=short_window).mean()
        data['SMA_long'] = data['close'].rolling(window=long_window).mean()
        
        # Generate signals
        results['position'] = np.where(data['SMA_short'] > data['SMA_long'], 1, -1)
        results['position'] = results['position'].fillna(0)
        
        # Calculate returns
        results['returns'] = results['position'].shift(1) * data['close'].pct_change()
        results['portfolio_value'] = (1 + results['returns']).cumprod() * self.config['initial_capital']
        
        return results
        
    def _rsi_strategy(self, data: pd.DataFrame, params: Dict[str, Any], results: pd.DataFrame) -> pd.DataFrame:
        """Implement RSI-based strategy."""
        rsi_period = params.get('rsi_period', 14)
        overbought = params.get('overbought', 70)
        oversold = params.get('oversold', 30)
        
        # Calculate RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Generate signals
        results['position'] = np.where(data['RSI'] < oversold, 1,
                                     np.where(data['RSI'] > overbought, -1, 0))
        results['position'] = results['position'].fillna(0)
        
        # Calculate returns
        results['returns'] = results['position'].shift(1) * data['close'].pct_change()
        results['portfolio_value'] = (1 + results['returns']).cumprod() * self.config['initial_capital']
        
        return results
        
    def _bollinger_bands_strategy(self, data: pd.DataFrame, params: Dict[str, Any], results: pd.DataFrame) -> pd.DataFrame:
        """Implement Bollinger Bands strategy."""
        window = params.get('window', 20)
        num_std = params.get('num_std', 2)
        
        # Calculate Bollinger Bands
        data['SMA'] = data['close'].rolling(window=window).mean()
        data['STD'] = data['close'].rolling(window=window).std()
        data['Upper'] = data['SMA'] + (data['STD'] * num_std)
        data['Lower'] = data['SMA'] - (data['STD'] * num_std)
        
        # Generate signals
        results['position'] = np.where(data['close'] < data['Lower'], 1,
                                     np.where(data['close'] > data['Upper'], -1, 0))
        results['position'] = results['position'].fillna(0)
        
        # Calculate returns
        results['returns'] = results['position'].shift(1) * data['close'].pct_change()
        results['portfolio_value'] = (1 + results['returns']).cumprod() * self.config['initial_capital']
        
        return results
        
    def _ai_strategy(self, data: pd.DataFrame, params: Dict[str, Any], results: pd.DataFrame) -> pd.DataFrame:
        """Implement AI-powered trading strategy."""
        # Use the AI model to generate predictions
        predictions = params.get('ai_model').predict(data)
        
        # Generate signals based on predictions
        results['position'] = np.where(predictions > data['close'], 1, -1)
        results['position'] = results['position'].fillna(0)
        
        # Calculate returns
        results['returns'] = results['position'].shift(1) * data['close'].pct_change()
        results['portfolio_value'] = (1 + results['returns']).cumprod() * self.config['initial_capital']
        
        return results
        
    def _calculate_performance_metrics(self, results: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics for a strategy.
        
        Args:
            results: DataFrame containing backtest results
            
        Returns:
            Dictionary of performance metrics
        """
        # Calculate total return
        total_return = (results['portfolio_value'].iloc[-1] / self.config['initial_capital']) - 1
        
        # Calculate Sharpe ratio
        risk_free_rate = self.config.get('risk_free_rate', 0.02)  # 2% annual risk-free rate
        excess_returns = results['returns'] - (risk_free_rate / 252)  # Daily risk-free rate
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        
        # Calculate maximum drawdown
        cumulative_returns = (1 + results['returns']).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        # Calculate win rate
        winning_trades = (results['returns'] > 0).sum()
        total_trades = (results['position'] != 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate profit factor
        gross_profit = results[results['returns'] > 0]['returns'].sum()
        gross_loss = abs(results[results['returns'] < 0]['returns'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }
        
    def get_strategy_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations for trading strategies.
        
        Returns:
            List of strategy recommendations with explanations
        """
        recommendations = []
        
        for strategy in self.strategies[:3]:  # Top 3 strategies
            recommendation = {
                'strategy': strategy.name,
                'description': strategy.description,
                'performance': strategy.performance_metrics,
                'recommendation': self._generate_recommendation(strategy)
            }
            recommendations.append(recommendation)
            
        return recommendations
        
    def _generate_recommendation(self, strategy: TradingStrategy) -> str:
        """Generate a recommendation for a strategy based on its performance.
        
        Args:
            strategy: TradingStrategy to analyze
            
        Returns:
            String containing the recommendation
        """
        metrics = strategy.performance_metrics
        
        if metrics['total_return'] > 0.2:  # >20% return
            if metrics['sharpe_ratio'] > 1.5:  # Good risk-adjusted return
                return "Strong Buy - Excellent risk-adjusted returns with high total return"
            else:
                return "Buy - High returns but consider risk management"
        elif metrics['total_return'] > 0:
            if metrics['win_rate'] > 0.6:  # High win rate
                return "Hold - Consistent performance with moderate returns"
            else:
                return "Hold - Positive returns but monitor risk"
        else:
            return "Sell - Strategy not performing well" 