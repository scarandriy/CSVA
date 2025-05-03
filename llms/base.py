from abc import ABC, abstractmethod

class LLM(ABC):
    """Abstract base class for LLM models."""
    
    def __init__(self, system_prompt: str = ""):
        """Initialize the LLM with a system prompt.
        
        Args:
            system_prompt: Initial system prompt for the model
        """
        self.system_prompt = system_prompt
    
    @abstractmethod
    def evaluate(self, prompt: str) -> str:
        """Evaluate the given prompt and return the model's response.
        
        Args:
            prompt: Input prompt to evaluate
            
        Returns:
            Model's response as a string
        """
        pass 