# core/pipeline.py
from typing import List
from processors.base            import ImageProcessor
from processors.inherent_processor import InherentImageProcessor
from llms.base                  import LLM
from core.memory                import MemoryBuffer
from core.logger                import EvaluationLogger
from utils.screenshot_capturer  import capture
from captions.blip_captioner import FastCaptioner

import json


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
        self.captioner       = FastCaptioner(threads=4)      


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
        # caption every screenshot (fast)
        captions = [self.captioner.caption(p) for p in image_paths]

        # ask the LLM to rate the captions (text-only)
        bullet = "\n".join(f"{i+1}. {c}" for i, c in enumerate(captions))
        prompt = (
            "You are a scam-detection assistant.\n\n"
            "For each caption, give a risk 0-10 and a short reason.\n"
            "Return JSON as described.\n\nCaptions:\n" + bullet
        )
        txt_resp = self.llm.evaluate(prompt=prompt)     # ← no images
        try:
            scores = json.loads(txt_resp)
        except Exception:
            # fallback: treat all low risk if parsing fails
            scores = [{"idx": i+1, "risk": 1, "why": "parse-error"} for i in range(len(captions))]

        # choose the caption with highest risk
        top = max(scores, key=lambda d: d["risk"])
        focus_idx, top_risk = top["idx"]-1, top["risk"]
        focus_path = image_paths[focus_idx]

        # only if ≥7, run heavy multimodal analysis on that screenshot
        if top_risk >= 7:
            deep_prompt = (
                "Full visual analysis of high-risk screenshot.\n"
                f"Caption: {captions[focus_idx]}\n"
                "Respond with a JSON report {risk_level, scam_type, reason}."
            )
            deep_resp = self.llm.evaluate(
                prompt=deep_prompt,
                image_paths=[focus_path]
            )
        else:
            deep_resp = "All captions rated low/medium risk."

        # log once
        self.logger.log({
            "captions": captions,
            "rankings": scores,
            "deep_response": deep_resp
        })
        return deep_resp
