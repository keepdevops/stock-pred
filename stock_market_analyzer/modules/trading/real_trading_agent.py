import logging
from datetime import datetime
import pandas as pd
from typing import Dict, List
import traceback
import os
import time
from PyQt6.QtCore import QTimer
import subprocess
from stock_market_analyzer.process_manager import ProcessManager
from ..tabs.monitor_tab import MonitorTab
from ..message_bus import MessageBus

def setup_logging(tab_name):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{tab_name.lower()}_tab.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

class RealTradingAgent:
    """Real trading agent for executing trades."""
    
    def __init__(self, data_loader):
        """Initialize the trading agent with a data loader."""
        self.logger = logging.getLogger(__name__)
        self.data_loader = data_loader
        self.message_bus = MessageBus()
        
        # Default configuration
        self.mode = 'simulation'
        self.initial_balance = 100000.0
        self.risk_per_trade = 0.02
        self.current_balance = self.initial_balance
        self.positions = {}
        self.order_history = []
        
        self.logger.info("Trading agent initialized in simulation mode")
        
        self.heartbeat_timer = QTimer()
        self.heartbeat_timer.timeout.connect(self.send_heartbeat)
        self.heartbeat_timer.start(5000)  # every 5 seconds
    
    async def buy(self, symbol: str, amount: float, price: float = None) -> Dict:
        """Execute a buy order."""
        try:
            # Get current price if not provided
            if price is None:
                data = await self.data_loader.load_data(symbol)
                if data.empty:
                    raise ValueError(f"No data available for {symbol}")
                price = data['close'].iloc[-1]
            
            # Calculate order cost
            order_cost = amount * price
            
            # Check if we have enough balance
            if order_cost > self.current_balance:
                raise ValueError(f"Insufficient funds. Required: ${order_cost:.2f}, Available: ${self.current_balance:.2f}")
            
            # Execute order
            order = {
                'symbol': symbol,
                'type': 'buy',
                'amount': amount,
                'price': price,
                'time': datetime.now(),
                'status': 'executed'
            }
            
            # Update positions and balance
            if symbol in self.positions:
                self.positions[symbol]['amount'] += amount
                self.positions[symbol]['cost'] += order_cost
            else:
                self.positions[symbol] = {
                    'amount': amount,
                    'cost': order_cost
                }
            
            self.current_balance -= order_cost
            self.order_history.append(order)
            
            self.logger.info(f"Buy order executed: {amount} {symbol} @ ${price:.2f}")
            return order
            
        except Exception as e:
            self.logger.error(f"Error executing buy order: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    async def sell(self, symbol: str, amount: float, price: float = None) -> Dict:
        """Execute a sell order."""
        try:
            # Check if we have the position
            if symbol not in self.positions:
                raise ValueError(f"No position in {symbol}")
            
            # Check if we have enough shares
            if amount > self.positions[symbol]['amount']:
                raise ValueError(f"Insufficient shares. Required: {amount}, Available: {self.positions[symbol]['amount']}")
            
            # Get current price if not provided
            if price is None:
                data = await self.data_loader.load_data(symbol)
                if data.empty:
                    raise ValueError(f"No data available for {symbol}")
                price = data['close'].iloc[-1]
            
            # Execute order
            order = {
                'symbol': symbol,
                'type': 'sell',
                'amount': amount,
                'price': price,
                'time': datetime.now(),
                'status': 'executed'
            }
            
            # Calculate order value
            order_value = amount * price
            
            # Update positions and balance
            self.positions[symbol]['amount'] -= amount
            if self.positions[symbol]['amount'] == 0:
                del self.positions[symbol]
            
            self.current_balance += order_value
            self.order_history.append(order)
            
            self.logger.info(f"Sell order executed: {amount} {symbol} @ ${price:.2f}")
            return order
            
        except Exception as e:
            self.logger.error(f"Error executing sell order: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def get_position(self, symbol: str) -> Dict:
        """Get current position for a symbol."""
        return self.positions.get(symbol, {'amount': 0, 'cost': 0})
    
    def get_balance(self) -> float:
        """Get current balance."""
        return self.current_balance
    
    def get_portfolio_value(self) -> float:
        """Get total portfolio value including positions."""
        try:
            total_value = self.current_balance
            
            # Add value of all positions
            for symbol, position in self.positions.items():
                data = self.data_loader.load_data(symbol)
                if not data.empty:
                    current_price = data['close'].iloc[-1]
                    position_value = position['amount'] * current_price
                    total_value += position_value
            
            return total_value
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio value: {str(e)}")
            return self.current_balance
    
    def get_order_history(self) -> List[Dict]:
        """Get order history."""
        return self.order_history

    def get_trading_history(self) -> pd.DataFrame:
        """Get trading history as a DataFrame."""
        return pd.DataFrame(self.order_history)
        
    def calculate_position_size(self, symbol: str, price: float) -> float:
        """Calculate position size based on risk management rules."""
        try:
            risk_amount = self.current_balance * self.risk_per_trade
            position_size = risk_amount / price
            return round(position_size, 2)
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0
            
    def get_portfolio_summary(self) -> dict:
        """Get summary of current portfolio state."""
        return {
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'open_positions': len(self.positions),
            'total_trades': len(self.order_history)
        }

    def send_heartbeat(self):
        """Send heartbeat message to monitor."""
        self.message_bus.publish("ProcessMonitor", "heartbeat", {
            "tab": self.__class__.__name__.replace("Tab", ""),
            "pid": os.getpid(),
            "timestamp": time.time()
        }) 