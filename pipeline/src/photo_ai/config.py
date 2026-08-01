from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PHOTO_AI_", env_file=".env")

    photos_path: Path = Path("/photos")
    database_url: str = "postgresql://photoai:changeme@localhost:5432/photoai"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "photos"
    ollama_url: str = "http://localhost:11434"

    # Runner API
    runner_host: str = "0.0.0.0"
    runner_port: int = 8765
    runner_auth_token: str = "changeme"

    # Model quality tier: "fast" (Florence-2 only) | "full" (Florence-2 + Qwen2.5-VL)
    quality_tier: str = "fast"

    # Florence-2 model variant
    florence_model: str = "microsoft/Florence-2-large"

    # Ollama model for rich captions (quality_tier=full)
    ollama_model: str = "qwen2.5vl:7b"

    # InsightFace similarity threshold (0-1, lower = stricter matching)
    face_similarity_threshold: float = 0.45

    # Skip photos already processed (False = force reprocess all)
    incremental: bool = True
