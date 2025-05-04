# processors/base.py
from abc import ABC, abstractmethod

class ImageProcessor(ABC):
    """Defines the interface for all image processors."""
    
    @abstractmethod
    def process(self, image_path: str):
        """
        Args:
            image_path: Path to the image file.
        Returns:
            Either OCR’d text (str) or—if you're using a built-in vision model—a marker
            that tells the LLM wrapper “send the raw image.”
        """
        pass
