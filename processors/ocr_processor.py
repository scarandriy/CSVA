# processors/ocr_processor.py
from typing import List, Optional
from .base import ImageProcessor

class OCRImageProcessor(ImageProcessor):
    """Extract plain text from the screenshot."""
    
    def __init__(self, languages: Optional[List[str]] = None):
        self.languages = languages or ["en"]
    
    def process(self, image_path: str) -> str:
        results = []
        return " ".join([line[1] for line in results])
    
