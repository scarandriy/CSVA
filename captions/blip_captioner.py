# captions/blip_captioner.py  (BLIP-1)
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch, pathlib

class FastCaptioner:
    def __init__(self, device="cpu", threads=4, target=384):
        torch.set_num_threads(threads)
        self.device  = device
        self.target  = target                         
        self.proc = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            use_fast=True)                            
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base").to(device)

    def _load_and_resize(self, path: str) -> Image.Image:
        img = Image.open(path).convert("RGB")
        img.thumbnail((self.target, self.target))     
        return img

    @torch.no_grad()
    def caption(self, img_path: str) -> str:
        img = self._load_and_resize(img_path)
        inputs = self.proc(images=img, return_tensors="pt").to(self.device)

        ids = self.model.generate(
            **inputs,
            max_length=25,
            num_beams=4,
            repetition_penalty=1.1,
        )
        return self.proc.decode(ids[0], skip_special_tokens=True)
