"""CLIP ViT-L/14 — generates semantic embeddings for Qdrant."""

from pathlib import Path
from PIL import Image
from sentence_transformers import SentenceTransformer
import torch


class ClipProcessor:
    MODEL_NAME = "clip-ViT-L-14"

    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = SentenceTransformer(self.MODEL_NAME, device=device)

    def embed_image(self, path: Path) -> list[float]:
        img = Image.open(path).convert("RGB")
        embedding = self._model.encode(img, convert_to_numpy=True)
        return embedding.tolist()

    def embed_text(self, text: str) -> list[float]:
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
