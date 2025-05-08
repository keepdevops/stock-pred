import logging
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime, timedelta

class RealTradingAgent:
    """Class for handling real trading operations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.initial_capital = config.get('initial_capital', 100000.0)
        self.current_capital = self.initial_capital
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.trade_history: List[Dict[str, Any]] = []
        
    def execute_trade(self, symbol: str, action: str, quantity: int, price: float) -> bool:
        """
        Execute a trade with the given parameters.
        
        Args:
            symbol: Stock symbol
            action: 'buy' or 'sell'
            quantity: Number of shares
            price: Price per share
            
        Returns:
            bool: True if trade was successful, False otherwise
        """
        try:
            if action not in ['buy', 'sell']:
                self.logger.error(f"Invalid action: {action}")
                return False
                
            if quantity <= 0:
                self.logger.error(f"Invalid quantity: {quantity}")
                return False
                
            if price <= 0:
                self.logger.error(f"Invalid price: {price}")
                return False
                
            trade_value = quantity * price
            
            if action == 'buy':
                if trade_value > self.current_capital:
                    self.logger.error(f"Insufficient capital for buy order: {trade_value} > {self.current_capital}")
                    return False
                    
                self.current_capital -= trade_value
                if symbol not in self.positions:
                    self.positions[symbol] = {'quantity': 0, 'avg_price': 0}
                    
                # Update position
                current_position = self.positions[symbol]
                total_quantity = current_position['quantity'] + quantity
                total_value = (current_position['quantity'] * current_position['avg_price']) + trade_value
                current_position['avg_price'] = total_value / total_quantity
                current_position['quantity'] = total_quantity
                
            else:  # sell
                if symbol not in self.positions or self.positions[symbol]['quantity'] < quantity:
                    self.logger.error(f"Insufficient shares for sell order: {quantity} > {self.positions.get(symbol, {}).get('quantity', 0)}")
                    return False
                    
                self.current_capital += trade_value
                self.positions[symbol]['quantity'] -= quantity
                
                if self.positions[symbol]['quantity'] == 0:
                    del self.positions[symbol]
                    
            # Record trade
            self.trade_history.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'price': price,
                'value': trade_value,
                'remaining_capital': self.current_capital
            })
            
            self.logger.info(f"Successfully executed {action} order for {quantity} shares of {symbol} at ${price:.2f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return False
            
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current position for a symbol."""
        return self.positions.get(symbol)
        
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        try:
            total_value = self.current_capital
            
            for symbol, position in self.positions.items():
                # Get current price from market data
                current_price = self._get_current_price(symbol)
                if current_price:
                    total_value += position['quantity'] * current_price
                    
            return total_value
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio value: {e}")
            return self.current_capital
            
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        # This is a placeholder - in a real implementation, you would fetch the current price
        # from your market data provider
        return None
        
    def get_trade_history(self) -> List[Dict[str, Any]]:
        """Get list of executed trades."""
        return self.trade_history
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Calculate trading performance metrics."""
        try:
            if not self.trade_history:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0,
                    'profit_loss': 0,
                    'return_on_capital': 0
                }
                
            total_trades = len(self.trade_history)
            winning_trades = sum(1 for trade in self.trade_history if trade['value'] > 0)
            losing_trades = total_trades - winning_trades
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            profit_loss = self.get_portfolio_value() - self.initial_capital
            return_on_capital = (profit_loss / self.initial_capital) * 100 if self.initial_capital > 0 else 0
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'profit_loss': profit_loss,
                'return_on_capital': return_on_capital
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {} 