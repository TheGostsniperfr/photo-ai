"""Qwen2.5-VL via Ollama — rich captions for quality_tier=full."""

import base64
from pathlib import Path
import httpx


class OllamaProcessor:
    PROMPT = (
        "Describe this photo in detail. Include: people present and their activities, "
        "location type and setting, objects, mood/atmosphere, any visible text. "
        "Be concise but complete. Answer in English."
    )

    def __init__(self, base_url: str, model: str):
        self._url = base_url.rstrip("/") + "/api/generate"
        self._model = model

    def caption(self, path: Path) -> str:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        resp = httpx.post(
            self._url,
            json={
                "model": self._model,
                "prompt": self.PROMPT,
                "images": [b64],
                "stream": False,
                "options": {"temperature": 0.2},
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
