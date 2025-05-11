# processors/ocr_processor.py
import easyocr
from typing import List, Optional
from .base import ImageProcessor

class OCRImageProcessor(ImageProcessor):
    """Extract plain text from the screenshot."""
    
    def __init__(self, languages: Optional[List[str]] = None):
        self.languages = languages or ["en"]
        self.reader    = easyocr.Reader(self.languages, gpu=False)
    
    def process(self, image_path: str) -> str:
        results = self.reader.readtext(image_path)
        return " ".join([line[1] for line in results])
    
