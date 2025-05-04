# core/pipeline.py
from typing import List
from processors.base            import ImageProcessor
from processors.inherent_processor import InherentImageProcessor
from llms.base                  import LLM
from core.memory                import MemoryBuffer
from core.logger                import EvaluationLogger
from utils.screenshot_capturer  import capture
from utils.resource_governor    import ResourceGovernor

class ScamEvaluatorPipeline:
    def __init__(
        self,
        image_processor: ImageProcessor,
        llm: LLM,
        memory_buffer: MemoryBuffer = None,
        logger: EvaluationLogger  = None,
    ):
        self.image_processor = image_processor
        self.llm             = llm
        self.memory_buffer   = memory_buffer or MemoryBuffer()
        self.logger          = logger or EvaluationLogger()

    def run_single(self, image_path: str) -> str:
        if image_path is None:
            image_path = capture()
        proc_out = self.image_processor.process(image_path)

        # --- multimodal path (Gemma3 etc.) ---
        if isinstance(proc_out, dict) and "image_path" in proc_out:
            resp = self.llm.evaluate(
                prompt=None,
                image_path=proc_out["image_path"]
            )
            extracted = None

        # --- OCR‐only path (Mistral) ---
        else:
            text = proc_out  # str
            context = self.memory_buffer.get_context()
            prompt  = (
                f"{context}\n\n"
                f"Image text:\n{text}\n\n"
                "Evaluate this image for scam (1–5):"
            )
            resp = self.llm.evaluate(prompt=prompt)
            extracted = text

        # update memory & log
        self.memory_buffer.add(resp)
        self.logger.log({
            "image":         image_path,
            "extracted_text": extracted,
            "response":      resp
        })

        return resp

    def run_batch(self, image_paths: List[str]) -> List[str]:
        return [self.run_single(p) for p in image_paths]
