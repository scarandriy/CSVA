from processors.ocr_processor import OCRImageProcessor
from llms.ollama_llm import OllamaLLM
from core.pipeline import ScamEvaluatorPipeline
from core.memory import MemoryBuffer
from core.logger import EvaluationLogger

def main():
    # Initialize components
    ocr_processor = OCRImageProcessor()
    llm = OllamaLLM(
        model_name="mistral",
        system_prompt="Evaluate the following image for scam likelihood from 1 to 5. "
                     "Provide a brief explanation for your rating."
    )
    memory = MemoryBuffer(max_size=5)
    logger = EvaluationLogger()

    # Create pipeline
    pipeline = ScamEvaluatorPipeline(
        image_processor=ocr_processor,
        llm=llm,
        memory_buffer=memory,
        logger=logger
    )

    # Example usage with multiple images
    image_paths = [
        "data/screenshots/shot1.png",
        "data/screenshots/shot2.png"
    ]

    # Process images
    results = pipeline.run_batch(image_paths)
    
    # Print results
    for image_path, result in zip(image_paths, results):
        print(f"\nImage: {image_path}")
        print(f"Result: {result}")

if __name__ == "__main__":
    main() 