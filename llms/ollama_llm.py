from ollama import Client
from .base import LLM

class OllamaLLM(LLM):
    """Ollama-based LLM implementation."""
    
    def __init__(self, model_name: str, system_prompt: str = ""):
        """Initialize the Ollama LLM.
        
        Args:
            model_name: Name of the Ollama model to use
            system_prompt: Initial system prompt
        """
        super().__init__(system_prompt)
        self.model_name = model_name
        self.client = Client()
    
    def evaluate(self, prompt: str) -> str:
        """Evaluate the prompt using the Ollama model.
        
        Args:
            prompt: Input prompt to evaluate
            
        Returns:
            Model's response
        """
        full_prompt = f"{self.system_prompt}\n\n{prompt}"
        response = self.client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": full_prompt}]
        )
        return response['message']['content'] 