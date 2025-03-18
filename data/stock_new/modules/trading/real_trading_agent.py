import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import pandas as pd

from modules.database import DatabaseConnector

@dataclass
class TradeConfig:
    """Configuration for trading parameters."""
    initial_budget: float = 10000.0
    max_position_size: float = 0.1
    max_open_trades: int = 5
    stop_loss: float = -0.02  # 2% stop loss
    take_profit: float = 0.05  # 5% take profit
    risk_per_trade: float = 0.01  # 1% risk per trade

class RealTradingAgent:
    """Handles real-time trading operations."""
    
    def __init__(
        self,
        db_connector: DatabaseConnector,
        logger: Optional[logging.Logger] = None,
        config: Optional[TradeConfig] = None
    ):
        self.db = db_connector
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or TradeConfig()
        
        self.current_positions: Dict[str, Dict] = {}
        self.trade_history: List[Dict] = []
        self._is_trading = False
    
    def start_trading(self) -> None:
        """Start the trading system."""
        try:
            if self._is_trading:
                self.logger.warning("Trading system is already running")
                return
            
            self._is_trading = True
            self.logger.info("Trading system started")
            
        except Exception as e:
            self.logger.error(f"Error starting trading system: {str(e)}")
            raise
    
    def stop_trading(self) -> None:
        """Stop the trading system."""
        try:
            if not self._is_trading:
                self.logger.warning("Trading system is not running")
                return
            
            self._is_trading = False
            self.logger.info("Trading system stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping trading system: {str(e)}")
            raise
    
    def place_order(
        self,
        ticker: str,
        order_type: str,
        quantity: float,
        price: float
    ) -> bool:
        """Place a trading order."""
        try:
            # Validate order
            if order_type not in ['BUY', 'SELL']:
                raise ValueError(f"Invalid order type: {order_type}")
            
            # Check if we can place the order
            if order_type == 'BUY':
                if len(self.current_positions) >= self.config.max_open_trades:
                    self.logger.warning("Maximum number of open trades reached")
                    return False
                
                position_size = quantity * price
                if position_size > self.config.initial_budget * self.config.max_position_size:
                    self.logger.warning("Order exceeds maximum position size")
                    return False
            
            # Record the order
            order = {
                'ticker': ticker,
                'type': order_type,
                'quantity': quantity,
                'price': price,
                'timestamp': datetime.now(),
                'status': 'EXECUTED'  # In a real system, this would be more complex
            }
            
            self.trade_history.append(order)
            
            # Update positions
            if order_type == 'BUY':
                self.current_positions[ticker] = {
                    'quantity': quantity,
                    'entry_price': price,
                    'current_price': price,
                    'stop_loss': price * (1 + self.config.stop_loss),
                    'take_profit': price * (1 + self.config.take_profit)
                }
            elif order_type == 'SELL' and ticker in self.current_positions:
                del self.current_positions[ticker]
            
            self.logger.info(f"Order placed successfully: {order}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error placing order: {str(e)}")
            return False
    
    def update_positions(self, market_data: Dict[str, float]) -> None:
        """Update current positions with latest market data."""
        try:
            for ticker, price in market_data.items():
                if ticker in self.current_positions:
                    position = self.current_positions[ticker]
                    position['current_price'] = price
                    
                    # Check stop loss and take profit
                    if price <= position['stop_loss']:
                        self.logger.info(f"Stop loss triggered for {ticker}")
                        self.place_order(ticker, 'SELL', position['quantity'], price)
                    
                    elif price >= position['take_profit']:
                        self.logger.info(f"Take profit triggered for {ticker}")
                        self.place_order(ticker, 'SELL', position['quantity'], price)
            
        except Exception as e:
            self.logger.error(f"Error updating positions: {str(e)}")
    
    def get_position_summary(self) -> pd.DataFrame:
        """Get summary of current positions."""
        try:
            if not self.current_positions:
                return pd.DataFrame()
            
            summary = []
            for ticker, position in self.current_positions.items():
                current_value = position['quantity'] * position['current_price']
                entry_value = position['quantity'] * position['entry_price']
                profit_loss = current_value - entry_value
                profit_loss_pct = (profit_loss / entry_value) * 100
                
                summary.append({
                    'ticker': ticker,
                    'quantity': position['quantity'],
                    'entry_price': position['entry_price'],
                    'current_price': position['current_price'],
                    'current_value': current_value,
                    'profit_loss': profit_loss,
                    'profit_loss_pct': profit_loss_pct,
                    'stop_loss': position['stop_loss'],
                    'take_profit': position['take_profit']
                })
            
            return pd.DataFrame(summary)
            
        except Exception as e:
            self.logger.error(f"Error getting position summary: {str(e)}")
            return pd.DataFrame()
    
    def get_trade_history(self) -> pd.DataFrame:
        """Get trading history."""
        try:
            if not self.trade_history:
                return pd.DataFrame()
            
            return pd.DataFrame(self.trade_history)
            
        except Exception as e:
            self.logger.error(f"Error getting trade history: {str(e)}")
            return pd.DataFrame()
    
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate trading performance metrics."""
        try:
            if not self.trade_history:
                return {}
            
            df = self.get_trade_history()
            
            # Calculate basic metrics
            metrics = {
                'total_trades': len(df),
                'winning_trades': len(df[df['price'] > df['price'].shift(1)]),
                'losing_trades': len(df[df['price'] < df['price'].shift(1)]),
                'win_rate': 0.0,
                'average_profit': 0.0,
                'average_loss': 0.0,
                'profit_factor': 0.0
            }
            
            if metrics['total_trades'] > 0:
                metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades']
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            return {} 