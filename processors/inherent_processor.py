# processors/inherent_processor.py
from .base import ImageProcessor

class InherentImageProcessor(ImageProcessor):
    """
    For an LLM with native vision (Gemma, LLaVA, etc.).
    We return the path itself so that the LLM wrapper can
    detect and attach the image.
    """
    def process(self, image_path: str):
        # We return a non‐str marker so that pipeline can
        # tell this is an image, not OCR text.
        return {"image_path": image_path}
