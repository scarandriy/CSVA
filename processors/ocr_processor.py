import easyocr
from typing import Optional, List
from .base import ImageProcessor

class OCRImageProcessor(ImageProcessor):
    """OCR-based image processor using EasyOCR."""
    
    def __init__(self, languages: Optional[List[str]] = None):
        """Initialize the OCR processor.
        
        Args:
            languages: List of languages to detect (default: ['en'])
        """
        self.languages = languages or ['en']
        self.reader = easyocr.Reader(self.languages)
    
    def process(self, image_path: str) -> str:
        """Extract text from an image using OCR.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text from the image
        """
        result = self.reader.readtext(image_path)
        return " ".join([text[1] for text in result]) 