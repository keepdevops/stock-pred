import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd

class TradingAgent:
    def __init__(self, initial_balance: float = 100000.0):
        self.logger = logging.getLogger(__name__)
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions: Dict[str, float] = {}  # symbol -> quantity
        self.trading_history: List[Dict] = []
        
    def get_balance(self) -> float:
        """Get current account balance."""
        return self.current_balance
        
    def get_positions(self) -> Dict[str, float]:
        """Get current positions."""
        return self.positions.copy()
        
    def get_trading_history(self) -> List[Dict]:
        """Get trading history."""
        return self.trading_history.copy()
        
    def calculate_position_size(self, symbol: str, current_price: float) -> float:
        """Calculate maximum position size based on risk management rules."""
        try:
            # Risk 1% of account per trade
            risk_amount = self.current_balance * 0.01
            
            # Use a 2% stop loss
            stop_loss = current_price * 0.02
            
            # Calculate position size
            position_size = risk_amount / stop_loss
            
            # Round to nearest whole share
            position_size = round(position_size)
            
            # Ensure we don't exceed available balance
            max_shares = self.current_balance / current_price
            position_size = min(position_size, max_shares)
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0
            
    def place_buy_order(self, symbol: str, size: float, price: float) -> bool:
        """Place a buy order."""
        try:
            # Validate order
            if size <= 0:
                raise ValueError("Order size must be greater than 0")
                
            total_cost = size * price
            if total_cost > self.current_balance:
                raise ValueError("Insufficient funds")
                
            # Execute order
            self.positions[symbol] = self.positions.get(symbol, 0) + size
            self.current_balance -= total_cost
            
            # Record trade
            self.trading_history.append({
                'timestamp': datetime.now(),
                'type': 'BUY',
                'symbol': symbol,
                'size': size,
                'price': price,
                'total': total_cost
            })
            
            self.logger.info(f"Buy order executed: {size} shares of {symbol} @ ${price:.2f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error placing buy order: {e}")
            return False
            
    def place_sell_order(self, symbol: str, size: float, price: float) -> bool:
        """Place a sell order."""
        try:
            # Validate order
            if size <= 0:
                raise ValueError("Order size must be greater than 0")
                
            current_position = self.positions.get(symbol, 0)
            if size > current_position:
                raise ValueError("Insufficient shares")
                
            # Execute order
            self.positions[symbol] = current_position - size
            self.current_balance += size * price
            
            # Record trade
            self.trading_history.append({
                'timestamp': datetime.now(),
                'type': 'SELL',
                'symbol': symbol,
                'size': size,
                'price': price,
                'total': size * price
            })
            
            self.logger.info(f"Sell order executed: {size} shares of {symbol} @ ${price:.2f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error placing sell order: {e}")
            return False
            
    def get_position_value(self, symbol: str, current_price: float) -> float:
        """Get current value of a position."""
        return self.positions.get(symbol, 0) * current_price
        
    def get_total_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Get total portfolio value including cash."""
        portfolio_value = self.current_balance
        
        for symbol, quantity in self.positions.items():
            if symbol in prices:
                portfolio_value += quantity * prices[symbol]
                
        return portfolio_value
        
    def calculate_pnl(self, symbol: str, current_price: float) -> Tuple[float, float]:
        """Calculate profit/loss for a position."""
        try:
            position = self.positions.get(symbol, 0)
            if position == 0:
                return 0.0, 0.0
                
            # Get average entry price from trading history
            buy_trades = [t for t in self.trading_history 
                         if t['symbol'] == symbol and t['type'] == 'BUY']
            
            if not buy_trades:
                return 0.0, 0.0
                
            total_cost = sum(t['total'] for t in buy_trades)
            total_shares = sum(t['size'] for t in buy_trades)
            avg_entry_price = total_cost / total_shares
            
            # Calculate P&L
            pnl = (current_price - avg_entry_price) * position
            pnl_percentage = (current_price - avg_entry_price) / avg_entry_price * 100
            
            return pnl, pnl_percentage
            
        except Exception as e:
            self.logger.error(f"Error calculating P&L: {e}")
            return 0.0, 0.0
            
    def get_trade_summary(self) -> Dict:
        """Get summary of trading activity."""
        try:
            summary = {
                'total_trades': len(self.trading_history),
                'buy_trades': len([t for t in self.trading_history if t['type'] == 'BUY']),
                'sell_trades': len([t for t in self.trading_history if t['type'] == 'SELL']),
                'current_balance': self.current_balance,
                'positions': len(self.positions),
                'initial_balance': self.initial_balance,
                'total_pnl': self.current_balance - self.initial_balance
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating trade summary: {e}")
            return {} 