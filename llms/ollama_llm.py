# llms/ollama_llm.py
from ollama import Client
from .base import LLM

class OllamaLLM(LLM):
    """
    Ollama wrapper supporting:
      • text‐only models (Mistral)
      • multimodal models (Gemma3, LLaVA, etc.)
    """

    def __init__(self, model_name: str, system_prompt: str = "", params: dict = None):
        super().__init__(system_prompt)
        
        self.model_name = model_name
        self.options    = params or {}

    def evaluate(self, prompt: str = None, image_path: str = None) -> str:
        # 1) Build the chat messages
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # 2) Attach user message, embedding the image if provided
        if prompt is not None or image_path:
            user_msg = {"role": "user", "content": prompt or ""}
            if image_path:
                # Ollama expects 'images' inside the message dict
                user_msg["images"] = [image_path]
            messages.append(user_msg)

        # 3) Prepare call args
        call_kwargs = {
            "model":    self.model_name,
            "messages": messages,
        }
        if self.options:
            call_kwargs["options"] = self.options

        # 4) Invoke
        self.client     = Client()
        resp = self.client.chat(**call_kwargs)
        return resp["message"]["content"]