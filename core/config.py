# core/config.py
import yaml
from processors.ocr_processor      import OCRImageProcessor
from processors.inherent_processor import InherentImageProcessor
from llms.ollama_llm              import OllamaLLM

def load_full_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def make_image_processor(name: str):
    name = name.lower()
    if name == "ocr":
        return OCRImageProcessor()
    if name == "inherent":
        return InherentImageProcessor()
    raise ValueError(f"Unknown image_processor: {name}")

def make_llm(cfg: dict) -> OllamaLLM:
    return OllamaLLM(
        model_name    = cfg["model_name"],
        system_prompt = cfg.get("system_prompt", ""),
        params        = cfg.get("params", {}),
    )
