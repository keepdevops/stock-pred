import os
import sys
import json
import logging
import traceback
import argparse
import time
from typing import Optional, Dict, Any
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QObject, pyqtSignal, QTimer
from modules.message_bus import MessageBus
from modules.tabs import get_tab_class

def setup_python_path(args):
    """Set up the Python path using command line arguments.
    
    Args:
        args: The parsed command line arguments.
    """
    # Clear any existing paths
    sys.path = []
    
    # Get conda environment paths
    conda_env = args.conda_env
    if conda_env:
        conda_site_packages = os.path.join(conda_env, 'lib', 'python3.12', 'site-packages')
        conda_lib = os.path.join(conda_env, 'lib', 'python3.12')
        conda_lib_dynload = os.path.join(conda_env, 'lib', 'python3.12', 'lib-dynload')
    else:
        conda_site_packages = os.path.join(args.python_dir, 'lib', 'python3.12', 'site-packages')
        conda_lib = os.path.join(args.python_dir, 'lib', 'python3.12')
        conda_lib_dynload = os.path.join(args.python_dir, 'lib', 'python3.12', 'lib-dynload')
    
    # Add paths in order of precedence
    paths = [
        conda_lib,           # Python standard library
        conda_lib_dynload,   # Python dynamic load library
        conda_site_packages, # Conda site-packages
        args.package_root,   # Package root
        args.project_root,   # Project root
        args.modules_dir,    # Modules directory
        args.python_dir,     # Python directory
    ]
    
    # Add each path if it exists and isn't already in sys.path
    for path in paths:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
            
    # Add any existing PYTHONPATH
    if "PYTHONPATH" in os.environ:
        for path in os.environ["PYTHONPATH"].split(os.pathsep):
            if path and path not in sys.path:
                sys.path.append(path)

def main():
    """Main entry point for the tab worker."""
    try:
        # Parse command line arguments first
        parser = argparse.ArgumentParser()
        parser.add_argument("--project-root", required=True)
        parser.add_argument("--package-root", required=True)
        parser.add_argument("--modules-dir", required=True)
        parser.add_argument("--python-dir", required=True)
        parser.add_argument("--conda-env", required=True)
        parser.add_argument("tab_name")
        args = parser.parse_args()
        
        # Set up Python path
        setup_python_path(args)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(__name__)
        
        # Create application
        app = QApplication(sys.argv)
        
        # Create and set up worker
        worker = TabWorker()
        worker.setup_tab(args.tab_name)
        
        # Start event loop
        logger.info(f"Starting worker for {args.tab_name}")
        sys.exit(app.exec())
        
    except Exception as e:
        logger.error(f"Error in worker: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

class TabWorker(QObject):
    """Worker class for running tabs in separate processes."""
    
    # Signals for communication with the parent process
    message_received = pyqtSignal(str, str, dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self):
        """Initialize the tab worker."""
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.message_bus = MessageBus()
        self.tab_instance = None
        
    def setup_tab(self, tab_name: str):
        """Set up the tab instance.
        
        Args:
            tab_name: The name of the tab to set up.
        """
        try:
            # Get the tab class
            tab_class = get_tab_class(tab_name)
            if not tab_class:
                raise ValueError(f"Unknown tab class: {tab_name}")
                
            # Create the tab instance with message bus
            self.tab_instance = tab_class(message_bus=self.message_bus)
            self.tab_instance.show()  # Make sure the tab is visible
            
            # Send initial connection status
            self.message_bus.publish("connection", "status_update", {
                "tab": tab_name,
                "status": "connected"
            })
            
            self.logger.info(f"Set up tab instance for {tab_name}")
            
            # Subscribe to messages
            self.message_bus.subscribe("all", self.handle_message)
            
        except Exception as e:
            self.logger.error(f"Error setting up tab: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            # Send error status
            self.message_bus.publish("connection", "status_update", {
                "tab": tab_name,
                "status": "error",
                "error": str(e)
            })
            raise
            
    def handle_message(self, sender: str, message_type: str, data: Dict[str, Any]):
        """Handle incoming messages.
        
        Args:
            sender: The sender of the message.
            message_type: The type of message.
            data: The message data.
        """
        try:
            if self.tab_instance:
                self.tab_instance.handle_message(sender, message_type, data)
        except Exception as e:
            self.logger.error(f"Error handling message: {str(e)}")
            self.logger.error(traceback.format_exc())
            
    def cleanup(self):
        """Clean up resources."""
        try:
            if self.tab_instance:
                self.tab_instance.cleanup()
                
            # Send disconnect status
            self.message_bus.publish("connection", "status_update", {
                "tab": self.tab_instance.__class__.__name__,
                "status": "disconnected"
            })
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            self.logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 