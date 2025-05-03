from collections import deque
from typing import List

class MemoryBuffer:
    """Manages a rolling buffer of past evaluations for context awareness."""
    
    def __init__(self, max_size: int = 5):
        """Initialize the memory buffer.
        
        Args:
            max_size: Maximum number of past evaluations to keep
        """
        self.buffer = deque(maxlen=max_size)
    
    def add(self, item: str) -> None:
        """Add a new evaluation to the buffer.
        
        Args:
            item: Evaluation result to add
        """
        self.buffer.append(item)
    
    def get_context(self) -> str:
        """Get the current context from the buffer.
        
        Returns:
            Formatted context string combining past evaluations
        """
        return "\n\n".join(self.buffer)
    
    def clear(self) -> None:
        """Clear the memory buffer."""
        self.buffer.clear() 