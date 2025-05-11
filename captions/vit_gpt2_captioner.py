# captions/vit_gpt2_captioner.py  (ViT-GPT2)
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import torch

class ViTGPT2Captioner:
    def __init__(
        self,
        model_name: str = "nlpconnect/vit-gpt2-image-captioning",
        device: str = "cpu",
        threads: int = 4,
        target: int = 384,
        max_length: int = 25,
        num_beams: int = 4,
    ):
        torch.set_num_threads(threads)
        self.device = device
        self.target = target
        self.max_length = max_length
        self.num_beams = num_beams

        # load the encoder-decoder model
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
        # the ViT preprocessing module
        self.feature_extractor = ViTImageProcessor.from_pretrained(model_name)
        # the GPT-2 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _load_and_resize(self, path: str) -> Image.Image:
        img = Image.open(path).convert("RGB")
        img.thumbnail((self.target, self.target))
        return img

    @torch.no_grad()
    def caption(self, img_path: str) -> str:
        img = self._load_and_resize(img_path)
        pixel_values = self.feature_extractor(images=img, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        output_ids = self.model.generate(
            pixel_values,
            max_length=self.max_length,
            num_beams=self.num_beams,
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
