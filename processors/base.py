from abc import ABC, abstractmethod
from typing import Any

class ImageProcessor(ABC):
    """Abstract base class for image processing strategies."""
    
    @abstractmethod
    def process(self, image_path: str) -> Any:
        """Process an image and return the result.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Processed result (embedding for CLIP, text for OCR)
        """
        pass 