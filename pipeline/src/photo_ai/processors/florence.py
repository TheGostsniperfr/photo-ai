"""Florence-2-large — multi-task: captions, tags, OCR, object detection."""

from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
import re


TASK_CAPTION = "<MORE_DETAILED_CAPTION>"
TASK_TAGS = "<GENERATE_TAGS>"
TASK_OCR = "<OCR>"


class FlorenceProcessor:
    def __init__(self, model_name: str = "microsoft/Florence-2-large"):
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.float16 if self._device == "cuda" else torch.float32
        self._processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self._dtype,
            trust_remote_code=True,
        ).to(self._device)
        self._model.eval()

    def _run(self, image: Image.Image, task: str) -> str:
        inputs = self._processor(text=task, images=image, return_tensors="pt").to(
            self._device, self._dtype
        )
        with torch.inference_mode():
            ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512,
                num_beams=3,
            )
        return self._processor.batch_decode(ids, skip_special_tokens=False)[0]

    def process(self, path: Path) -> dict:
        image = Image.open(path).convert("RGB")

        caption_raw = self._run(image, TASK_CAPTION)
        caption = self._processor.post_process_generation(
            caption_raw, task=TASK_CAPTION, image_size=image.size
        ).get(TASK_CAPTION, "").strip()

        tags_raw = self._run(image, TASK_TAGS)
        tags_parsed = self._processor.post_process_generation(
            tags_raw, task=TASK_TAGS, image_size=image.size
        ).get(TASK_TAGS, "")
        tags = [t.strip().lower() for t in re.split(r"[,;]", tags_parsed) if t.strip()]

        ocr_raw = self._run(image, TASK_OCR)
        ocr = self._processor.post_process_generation(
            ocr_raw, task=TASK_OCR, image_size=image.size
        ).get(TASK_OCR, "").strip()

        return {"caption": caption, "tags": tags, "ocr_text": ocr or None}
