import logging
from typing import Dict, List, Callable, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Message:
    """Represents a message in the message bus system."""
    type: str
    data: Any
    timestamp: datetime
    source: str

class MessageBus:
    """Message bus for handling communication between different components."""
    
    def __init__(self):
        """Initialize the message bus."""
        self.logger = logging.getLogger(__name__)
        self.subscribers: Dict[str, List[Callable]] = {}
        self.message_history: List[Message] = []
        
    def subscribe(self, message_type: str, callback: Callable):
        """
        Subscribe to messages of a specific type.
        
        Args:
            message_type: Type of messages to subscribe to
            callback: Function to call when a message of the specified type is received
        """
        if message_type not in self.subscribers:
            self.subscribers[message_type] = []
        self.subscribers[message_type].append(callback)
        self.logger.debug(f"New subscriber for message type: {message_type}")
        
    def unsubscribe(self, message_type: str, callback: Callable):
        """
        Unsubscribe from messages of a specific type.
        
        Args:
            message_type: Type of messages to unsubscribe from
            callback: Function to remove from subscribers
        """
        if message_type in self.subscribers:
            self.subscribers[message_type].remove(callback)
            self.logger.debug(f"Removed subscriber for message type: {message_type}")
            
    def publish(self, message_type: str, data: Any, source: str = "unknown"):
        """
        Publish a message to all subscribers.
        
        Args:
            message_type: Type of the message
            data: Message data
            source: Source of the message
        """
        message = Message(
            type=message_type,
            data=data,
            timestamp=datetime.now(),
            source=source
        )
        
        self.message_history.append(message)
        
        if message_type in self.subscribers:
            for callback in self.subscribers[message_type]:
                try:
                    callback(source, message_type, data)
                except Exception as e:
                    self.logger.error(f"Error in subscriber callback: {e}")
                    
    def get_message_history(self, message_type: str = None) -> List[Message]:
        """
        Get message history, optionally filtered by message type.
        
        Args:
            message_type: Optional message type to filter by
            
        Returns:
            List of messages matching the criteria
        """
        if message_type:
            return [msg for msg in self.message_history if msg.type == message_type]
        return self.message_history
        
    def clear_history(self):
        """Clear the message history."""
        self.message_history.clear()
        self.logger.debug("Message history cleared")
        
    def get_subscribers(self, message_type: str = None) -> Dict[str, List[Callable]]:
        """
        Get subscribers, optionally filtered by message type.
        
        Args:
            message_type: Optional message type to filter by
            
        Returns:
            Dictionary of subscribers
        """
        if message_type:
            return {message_type: self.subscribers.get(message_type, [])}
        return self.subscribers 