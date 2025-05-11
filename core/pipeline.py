# core/pipeline.py
from typing import List
from processors.base            import ImageProcessor
from processors.inherent_processor import InherentImageProcessor
from llms.base                  import LLM
from core.memory                import MemoryBuffer
from core.logger                import EvaluationLogger
from utils.screenshot_capturer  import capture
from captions.blip_captioner import FastCaptioner
from processors.ocr_processor import OCRImageProcessor


import json


class ScamEvaluatorPipeline:
    def __init__(
        self,
        image_processor: ImageProcessor,
        llm: LLM,
        small_llm: LLM,
        memory_buffer: MemoryBuffer = None,
        logger: EvaluationLogger  = None,
    ):
        
        self.image_processor = image_processor
        self.llm             = llm
        self.small_llm       = small_llm
        self.memory_buffer   = memory_buffer or MemoryBuffer()
        self.logger          = logger or EvaluationLogger()
        #self.captioner       = FastCaptioner(threads=4)  
        self.ocr_processor   = OCRImageProcessor()
    


    def run_single(self, image_path: str) -> str:

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
        if len(image_paths)==0:
            image_paths.append(capture())
        return [self.run_single(p) for p in image_paths]
    

    def run_multi(self, image_paths: list[str]) -> str:
        ocr_texts = [self.ocr_processor.process(p) for p in image_paths]
        for i in ocr_texts:
            print(' ----', i)
        # 2) Normalize via small (text-only) LLM
        combined = "\n---\n".join(ocr_texts)
        context_prompt = (
            f"{combined}\n\n"
            "<end_of_context>"
        )
        context = self.small_llm.evaluate(prompt=context_prompt)
        print(context)
            


        select_prompt = (
            "You are a scam-detection assistant.\n\n"
            f"User context:\n{context}\n\n"
            "Given these sequential image descriptions, pick the INDEX (0-based) of the "
            "image most likely to be fraudulent or high-risk.\n"
            "Respond ONLY with JSON in the format {\"index\": <number>}.\n\n"
        )
        sel_resp = self.llm.evaluate(prompt=select_prompt)
        try:
            sel = json.loads(sel_resp)
            print(sel_resp)
            idx = int(sel.get("index", 1))
        except Exception:
            idx = 1
        # clamp and convert to zero-based
        focus_idx = max(1, min(idx, len(image_paths))) - 1
        focus_path = image_paths[focus_idx]

        # 4) Send that single image to the heavy multimodal LLM
        deep_resp = self.llm.evaluate(
            prompt=None,
            image_path=focus_path
        )

        # 5) Log everything
        self.logger.log({
            "ocr_texts": ocr_texts,
            "context": context,
            "selection_json": sel_resp,
            "deep_response": deep_resp
        })
        return deep_resp