import torch, PIL.Image as Image
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
)
#NOT FAST
class Blip2Captioner:
    """
    Captioner based on Salesforce/blip2-opt-2.7b.
    • fp16 on M-series (mps) ≈ 1.2 s / screenshot
    • int8 (bitsandbytes) on 8 GB GPU ≈ 0.4 s / screenshot
    """

    def __init__(
        self,
        device: str = "cpu",    # "cpu" | "mps" | "cuda"
        precision: str = "bf16",# "bf16" | "fp16" | "int8"
        max_len: int = 35,
    ):
        name = "Salesforce/blip2-opt-2.7b"
        self.proc = Blip2Processor.from_pretrained(name)

        if precision == "int8":
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                name,
                quantization_config=bnb_config,
                device_map="auto",
            )
        else:
            dtype = torch.bfloat16 if precision == "bf16" else torch.float16
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                name,
                torch_dtype=dtype,
                device_map={"": device},
            )

        self.device  = device
        self.max_len = max_len
        torch.set_grad_enabled(False)

    def caption(self, img_path: str) -> str:
        img = Image.open(img_path).convert("RGB")

        # ↓↓↓ resize *before* processor to keep UI text legible
        img = img.resize((384, 384), Image.BICUBIC)

        inputs = self.proc(images=img, return_tensors="pt").to(self.device)
        ids    = self.model.generate(
            **inputs,
            max_new_tokens=self.max_len,
            num_beams=4,
            repetition_penalty=1.1,
        )
        return self.proc.decode(ids[0], skip_special_tokens=True)
