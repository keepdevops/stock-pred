import sys
import os
import logging
from typing import Dict, Optional
from datetime import datetime

from ..process_manager import ProcessManager
from .base_agent import BaseAgent

class RealTradingAgent(BaseAgent):
    """Real-time trading agent that manages trading operations."""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.process_manager = ProcessManager()
        self._setup_logging()
        
    def _setup_logging(self):
        """Set up logging for the trading agent."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Create handlers
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler('trading_agent.log')
        
        # Create formatters and add it to handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers to the logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
    def start(self):
        """Start the trading agent and its associated processes."""
        try:
            self.logger.info("Starting trading agent...")
            self.process_manager.start_tab("trading")
            self.logger.info("Trading agent started successfully")
        except Exception as e:
            self.logger.error(f"Failed to start trading agent: {str(e)}")
            raise
            
    def stop(self):
        """Stop the trading agent and its associated processes."""
        try:
            self.logger.info("Stopping trading agent...")
            self.process_manager.stop_tab("trading")
            self.logger.info("Trading agent stopped successfully")
        except Exception as e:
            self.logger.error(f"Failed to stop trading agent: {str(e)}")
            raise
            
    def restart(self):
        """Restart the trading agent and its associated processes."""
        try:
            self.logger.info("Restarting trading agent...")
            self.process_manager.restart_tab("trading")
            self.logger.info("Trading agent restarted successfully")
        except Exception as e:
            self.logger.error(f"Failed to restart trading agent: {str(e)}")
            raise
            
    def check_status(self) -> Dict:
        """Check the status of the trading agent and its processes."""
        try:
            status = self.process_manager.check_status("trading")
            self.logger.info(f"Trading agent status: {status}")
            return status
        except Exception as e:
            self.logger.error(f"Failed to check trading agent status: {str(e)}")
            raise 