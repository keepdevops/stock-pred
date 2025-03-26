import logging
from datetime import datetime
import pandas as pd

class RealTradingAgent:
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.mode = config.get('mode', 'simulation')
        self.initial_balance = config.get('initial_balance', 10000)
        self.risk_per_trade = config.get('risk_per_trade', 0.02)
        
        self.balance = self.initial_balance
        self.positions = {}  # symbol -> {size, entry_price}
        self.trade_history = []
        
    def place_buy_order(self, symbol: str, size: float, price: float = None):
        """Place a buy order."""
        try:
            if self.mode == 'simulation':
                if price is None:
                    raise ValueError("Price must be provided in simulation mode")
                    
                # Check if we have enough balance
                cost = size * price
                if cost > self.balance:
                    raise ValueError(f"Insufficient balance for trade: {cost} > {self.balance}")
                    
                # Update balance and positions
                self.balance -= cost
                if symbol in self.positions:
                    # Average down
                    total_size = self.positions[symbol]['size'] + size
                    total_cost = (self.positions[symbol]['size'] * self.positions[symbol]['entry_price'] + 
                                size * price)
                    self.positions[symbol] = {
                        'size': total_size,
                        'entry_price': total_cost / total_size
                    }
                else:
                    self.positions[symbol] = {
                        'size': size,
                        'entry_price': price
                    }
                
                # Record trade
                self.trade_history.append({
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'type': 'buy',
                    'size': size,
                    'price': price,
                    'balance': self.balance
                })
                
                self.logger.info(f"Buy order placed: {size} {symbol} @ {price}")
                return True
                
            else:
                raise NotImplementedError("Real trading not implemented yet")
                
        except Exception as e:
            self.logger.error(f"Error placing buy order: {e}")
            return False
            
    def place_sell_order(self, symbol: str, size: float = None, price: float = None):
        """Place a sell order."""
        try:
            if self.mode == 'simulation':
                if price is None:
                    raise ValueError("Price must be provided in simulation mode")
                    
                if symbol not in self.positions:
                    raise ValueError(f"No position in {symbol}")
                    
                if size is None:
                    size = self.positions[symbol]['size']
                elif size > self.positions[symbol]['size']:
                    raise ValueError(f"Insufficient position size: {size} > {self.positions[symbol]['size']}")
                    
                # Update balance and positions
                proceeds = size * price
                self.balance += proceeds
                
                if size == self.positions[symbol]['size']:
                    del self.positions[symbol]
                else:
                    self.positions[symbol]['size'] -= size
                    
                # Record trade
                self.trade_history.append({
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'type': 'sell',
                    'size': size,
                    'price': price,
                    'balance': self.balance
                })
                
                self.logger.info(f"Sell order placed: {size} {symbol} @ {price}")
                return True
                
            else:
                raise NotImplementedError("Real trading not implemented yet")
                
        except Exception as e:
            self.logger.error(f"Error placing sell order: {e}")
            return False
            
    def get_position(self, symbol: str) -> dict:
        """Get current position for a symbol."""
        return self.positions.get(symbol, {'size': 0, 'entry_price': 0})
        
    def get_balance(self) -> float:
        """Get current balance."""
        return self.balance
        
    def get_trading_history(self) -> pd.DataFrame:
        """Get trading history as a DataFrame."""
        return pd.DataFrame(self.trade_history)
        
    def calculate_position_size(self, symbol: str, price: float) -> float:
        """Calculate position size based on risk management rules."""
        try:
            risk_amount = self.balance * self.risk_per_trade
            position_size = risk_amount / price
            return round(position_size, 2)
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0
            
    def get_portfolio_value(self, current_prices: dict) -> float:
        """Calculate total portfolio value including open positions."""
        try:
            portfolio_value = self.balance
            
            for symbol, position in self.positions.items():
                if symbol in current_prices:
                    portfolio_value += position['size'] * current_prices[symbol]
                    
            return portfolio_value
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio value: {e}")
            return self.balance
            
    def get_portfolio_summary(self) -> dict:
        """Get summary of current portfolio state."""
        return {
            'initial_balance': self.initial_balance,
            'current_balance': self.balance,
            'open_positions': len(self.positions),
            'total_trades': len(self.trade_history)
        } 