from typing import List, Optional
from processors.base import ImageProcessor
from llms.base import LLM
from core.memory import MemoryBuffer
from core.logger import EvaluationLogger

class ScamEvaluatorPipeline:
    """Main pipeline for evaluating images for scam detection."""
    
    def __init__(
        self,
        image_processor: ImageProcessor,
        llm: LLM,
        memory_buffer: Optional[MemoryBuffer] = None,
        logger: Optional[EvaluationLogger] = None
    ):
        """Initialize the pipeline.
        
        Args:
            image_processor: Strategy for processing images (CLIP or OCR)
            llm: Language model for evaluation
            memory_buffer: Optional buffer for context awareness
            logger: Optional logger for results
        """
        self.image_processor = image_processor
        self.llm = llm
        self.memory_buffer = memory_buffer or MemoryBuffer()
        self.logger = logger or EvaluationLogger()
    
    def run_single(self, image_path: str) -> str:
        """Process and evaluate a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Model's evaluation result
        """
        # Process the image
        processed = self.image_processor.process(image_path)
        
        # Build the prompt
        if isinstance(processed, str):  # OCR case
            context = self.memory_buffer.get_context()
            prompt = f"{context}\n\nImage text:\n{processed}\n\nEvaluate this image for scam from 1 to 5."
            extracted_text = processed
        else:  # CLIP case
            prompt = processed
            extracted_text = None
        
        # Get model evaluation
        response = self.llm.evaluate(prompt)
        
        # Update memory and log results
        self.memory_buffer.add(response)
        self.logger.log({
            "image": image_path,
            "extracted_text": extracted_text,
            "response": response
        })
        
        return response
    
    def run_batch(self, image_paths: List[str]) -> List[str]:
        """Process and evaluate multiple images.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            List of evaluation results
        """
        return [self.run_single(path) for path in image_paths] 