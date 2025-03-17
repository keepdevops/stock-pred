from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from alpaca_trade_api.rest import REST
from alpaca_trade_api.entity import Order
from dataclasses import dataclass
from ..stock_ai_agent import StockAIAgent
from ..database import DatabaseConnector

@dataclass
class TradeConfig:
    """Configuration for trading strategies."""
    profit_tiers: List[float]  # e.g., [5.0, 8.0, 9.0] for 5%, 8%, 9% gains
    stop_loss: float  # e.g., -2.0 for 2% loss
    max_position_size: float  # e.g., 0.1 for 10% of budget
    max_open_trades: int  # e.g., 5 trades at once

class RealTradingAgent:
    """Executes live trades based on AI predictions and trading strategies."""
    
    def __init__(
        self,
        ai_agent: StockAIAgent,
        api_key: str,
        api_secret: str,
        base_url: str,
        budget: float = 10000.0,
        strategy: str = 'tiered_sell'
    ):
        self.ai_agent = ai_agent
        self.broker_api = REST(api_key, api_secret, base_url)
        self.budget = budget
        self.strategy = strategy
        self.logger = logging.getLogger(__name__)
        
        # Default trading configuration
        self.trade_config = TradeConfig(
            profit_tiers=[5.0, 8.0, 9.0],
            stop_loss=-2.0,
            max_position_size=0.1,
            max_open_trades=5
        )
        
        # Initialize trade log
        self.trade_log: Dict[str, List[Any]] = {
            'timestamp': [],
            'ticker': [],
            'action': [],
            'quantity': [],
            'price': [],
            'total': [],
            'strategy_signal': []
        }
    
    def fetch_live_data(
        self,
        ticker: str,
        timeframe: str = '1Min',
        limit: int = 100
    ) -> pd.DataFrame:
        """Fetch real-time market data."""
        try:
            # Get latest bars
            bars = self.broker_api.get_bars(
                ticker,
                timeframe,
                limit=limit,
                adjustment='raw'
            ).df
            
            # Format data
            bars = bars.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            
            return bars
            
        except Exception as e:
            self.logger.error(f"Error fetching live data for {ticker}: {e}")
            return pd.DataFrame()
    
    def execute_trade(
        self,
        action: str,
        ticker: str,
        quantity: float,
        order_type: str = 'market'
    ) -> Optional[Order]:
        """Execute a trade order."""
        try:
            # Check if we can trade
            if not self.broker_api.get_clock().is_open:
                self.logger.warning("Market is closed")
                return None
            
            # Execute order
            order = self.broker_api.submit_order(
                symbol=ticker,
                qty=quantity,
                side=action,
                type=order_type,
                time_in_force='day'
            )
            
            # Log trade
            self._log_trade(ticker, action, quantity, order.filled_avg_price)
            
            return order
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return None
    
    def apply_strategy(
        self,
        predictions: np.ndarray,
        current_price: float,
        ticker: str
    ) -> Dict[str, Any]:
        """Apply trading strategy based on predictions."""
        try:
            match self.strategy:
                case 'tiered_sell':
                    return self._apply_tiered_sell_strategy(
                        predictions,
                        current_price,
                        ticker
                    )
                case 'mean_reversion':
                    return self._apply_mean_reversion_strategy(
                        predictions,
                        current_price,
                        ticker
                    )
                case _:
                    raise ValueError(f"Unknown strategy: {self.strategy}")
                    
        except Exception as e:
            self.logger.error(f"Error applying strategy: {e}")
            return {'action': 'hold', 'quantity': 0, 'reason': str(e)}
    
    def monitor_trades(self) -> Dict[str, float]:
        """Monitor open positions and account status."""
        try:
            account = self.broker_api.get_account()
            positions = self.broker_api.list_positions()
            
            # Calculate metrics
            total_equity = float(account.equity)
            buying_power = float(account.buying_power)
            open_positions = len(positions)
            
            # Calculate P&L
            total_pl = sum(
                float(pos.unrealized_pl)
                for pos in positions
            )
            
            return {
                'equity': total_equity,
                'buying_power': buying_power,
                'open_positions': open_positions,
                'total_pl': total_pl
            }
            
        except Exception as e:
            self.logger.error(f"Error monitoring trades: {e}")
            return {}
    
    def save_trade_data(self, db_path: str) -> bool:
        """Save trade log to database."""
        try:
            # Create DataFrame from trade log
            trade_df = pd.DataFrame(self.trade_log)
            
            # Connect to database
            db_connector = DatabaseConnector()
            if not db_connector.create_connection(db_path):
                return False
            
            # Save to database
            return db_connector.update_stock_data(
                trade_df,
                'trade_history',
                if_exists='append'
            )
            
        except Exception as e:
            self.logger.error(f"Error saving trade data: {e}")
            return False
    
    def _log_trade(
        self,
        ticker: str,
        action: str,
        quantity: float,
        price: float
    ) -> None:
        """Log trade details."""
        self.trade_log['timestamp'].append(datetime.now())
        self.trade_log['ticker'].append(ticker)
        self.trade_log['action'].append(action)
        self.trade_log['quantity'].append(quantity)
        self.trade_log['price'].append(price)
        self.trade_log['total'].append(quantity * price)
        self.trade_log['strategy_signal'].append(self.strategy)
    
    def _apply_tiered_sell_strategy(
        self,
        predictions: np.ndarray,
        current_price: float,
        ticker: str
    ) -> Dict[str, Any]:
        """Implement tiered sell strategy."""
        try:
            # Get position if exists
            position = self.broker_api.get_position(ticker)
            avg_entry = float(position.avg_entry_price)
            
            # Calculate returns
            current_return = (current_price - avg_entry) / avg_entry * 100
            
            # Check stop loss
            if current_return <= self.trade_config.stop_loss:
                return {
                    'action': 'sell',
                    'quantity': position.qty,
                    'reason': 'stop_loss'
                }
            
            # Check profit tiers
            for tier in self.trade_config.profit_tiers:
                if current_return >= tier:
                    sell_quantity = float(position.qty) * (1/len(self.trade_config.profit_tiers))
                    return {
                        'action': 'sell',
                        'quantity': sell_quantity,
                        'reason': f'profit_tier_{tier}'
                    }
            
            # Check for new entry
            predicted_return = (predictions[0] - current_price) / current_price * 100
            if predicted_return > 2.0:  # Minimum expected return
                max_position = self.budget * self.trade_config.max_position_size
                quantity = max_position / current_price
                return {
                    'action': 'buy',
                    'quantity': quantity,
                    'reason': 'new_position'
                }
            
            return {'action': 'hold', 'quantity': 0, 'reason': 'no_signal'}
            
        except Exception as e:
            self.logger.error(f"Error in tiered sell strategy: {e}")
            return {'action': 'hold', 'quantity': 0, 'reason': str(e)}
    
    def _apply_mean_reversion_strategy(
        self,
        predictions: np.ndarray,
        current_price: float,
        ticker: str
    ) -> Dict[str, Any]:
        """Implement mean reversion strategy."""
        # Implementation for mean reversion strategy
        # This is a placeholder for future enhancement
        return {'action': 'hold', 'quantity': 0, 'reason': 'not_implemented'} 