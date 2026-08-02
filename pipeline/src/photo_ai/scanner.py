"""Discover and batch-process photos, tracking state in PostgreSQL."""

import asyncio
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Callable

import httpx
from PIL import Image
from PIL.ExifTags import TAGS as EXIF_TAGS
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

from photo_ai.album_engine import AlbumEngine, PhotoMeta
from photo_ai.config import Settings
from photo_ai.db import Database
from photo_ai.immich_client import ImmichClient
from photo_ai.vector_store import VectorStore
from photo_ai.processors.clip import ClipProcessor
from photo_ai.processors.florence import FlorenceProcessor
from photo_ai.processors.faces import FaceProcessor
from photo_ai.processors.ollama import OllamaProcessor
from photo_ai.processors.xmp import XmpWriter

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".tiff", ".tif", ".bmp", ".cr2", ".nef", ".arw", ".dng", ".orf", ".rw2"}

console = Console()


def _extract_taken_at(path: Path) -> datetime | None:
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            for tag_id, value in exif.items():
                if EXIF_TAGS.get(tag_id) == "DateTimeOriginal":
                    return datetime.strptime(str(value), "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class PhotoScanner:
    def __init__(self, settings: Settings, db: Database, vector_store: VectorStore):
        self._settings = settings
        self._db = db
        self._vs = vector_store
        self._clip = ClipProcessor()
        self._florence = FlorenceProcessor(settings.florence_model)
        self._faces = FaceProcessor(settings.face_similarity_threshold)
        self._xmp = XmpWriter()
        self._ollama = OllamaProcessor(settings.ollama_url, settings.ollama_model) if settings.quality_tier == "full" else None

    def _discover(self) -> list[Path]:
        photos = []
        for base in self._settings.photos_paths:
            for root, _, files in os.walk(base):
                for f in files:
                    p = Path(root) / f
                    if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                        photos.append(p)
        return sorted(photos)

    async def _process_one(self, path: Path) -> dict:
        stat = path.stat()
        mtime = stat.st_mtime
        sha256 = await asyncio.to_thread(_sha256, path)

        if self._settings.incremental and await self._db.is_processed(str(path), mtime, sha256):
            return {"skipped": True}

        # Florence-2: caption + tags + OCR
        result = await asyncio.to_thread(self._florence.process, path)
        caption = result["caption"]
        tags = result["tags"]
        ocr_text = result.get("ocr_text")

        # Optional: richer caption via Qwen2.5-VL
        if self._ollama:
            caption = await asyncio.to_thread(self._ollama.caption, path)

        # CLIP embedding → Qdrant
        embedding = await asyncio.to_thread(self._clip.embed_image, path)
        await self._vs.upsert(str(path), embedding, {"caption": caption, "tags": tags})

        # Face detection
        detected = await asyncio.to_thread(self._faces.detect, path)
        face_names: list[str] = []
        stored_faces = []
        for face in detected:
            all_faces = await self._db.list_face_clusters()
            # Simple nearest-neighbour matching against stored embeddings
            # Full clustering happens offline; here we just store for later
            stored_faces.append({"bbox": face.bbox, "confidence": face.confidence})

        # XMP sidecar
        await asyncio.to_thread(
            self._xmp.write,
            path,
            caption=caption,
            tags=tags,
            people=face_names or None,
            ocr_text=ocr_text,
        )

        taken_at = await asyncio.to_thread(_extract_taken_at, path)

        # Persist to DB
        await self._db.upsert_photo(
            str(path), mtime, sha256,
            caption=caption, tags=tags, faces=stored_faces, ocr_text=ocr_text,
            taken_at=taken_at,
        )

        return {"caption": caption, "tags": tags, "faces": len(detected)}

    async def run(
        self,
        full: bool = False,
        progress_callback: Callable[[int, int, str], None] | None = None,
        report_url: str | None = None,
    ) -> dict:
        photos = self._discover()
        total = len(photos)
        done = 0
        errors = 0

        console.print(f"[bold green]Found {total} photos[/bold green]")

        if full:
            # Force reprocess everything
            self._settings.incremental = False

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing...", total=total)

            for path in photos:
                try:
                    result = await self._process_one(path)
                    if not result.get("skipped"):
                        done += 1
                except Exception as e:
                    errors += 1
                    console.print(f"[red]Error processing {path}: {e}[/red]")

                progress.advance(task)
                processed = total - photos.index(path) - 1

                if progress_callback:
                    progress_callback(total - processed, total, str(path))

        summary = {"total": total, "processed": done, "errors": errors}

        if self._settings.immich_url and self._settings.immich_api_key:
            immich = ImmichClient(
                self._settings.immich_url,
                self._settings.immich_api_key,
                self._settings.immich_path_prefix,
            )
            try:
                rows = await self._db.list_all_photos()
                metas = [
                    PhotoMeta(
                        path=r["path"],
                        caption=r["caption"] or "",
                        tags=r["tags"] if isinstance(r["tags"], list) else json.loads(r["tags"] or "[]"),
                        taken_at=r["taken_at"],
                        asset_id=r["immich_asset_id"],
                    )
                    for r in rows
                ]
                engine = AlbumEngine(immich)
                album_result = await engine.run(metas)
                # Cache resolved asset IDs to avoid re-fetching next run
                for meta in metas:
                    if meta.asset_id:
                        await self._db.update_immich_asset_id(meta.path, meta.asset_id)
                summary["albums"] = album_result
            finally:
                await immich.close()

        if report_url:
            try:
                async with httpx.AsyncClient() as client:
                    await client.post(report_url, json=summary, timeout=10)
            except Exception:
                pass

        return summary
