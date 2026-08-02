"""PostgreSQL state tracking for processed photos and face identities."""

import asyncpg
import json
from pathlib import Path


CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS photos (
    path             TEXT PRIMARY KEY,
    mtime            DOUBLE PRECISION NOT NULL,
    sha256           TEXT NOT NULL,
    processed_at     TIMESTAMPTZ DEFAULT now(),
    caption          TEXT,
    tags             JSONB DEFAULT '[]',
    faces            JSONB DEFAULT '[]',
    duplicate_of     TEXT,
    ocr_text         TEXT,
    taken_at         TIMESTAMPTZ,
    immich_asset_id  TEXT
);

CREATE TABLE IF NOT EXISTS faces (
    id          SERIAL PRIMARY KEY,
    name        TEXT,
    cluster_id  TEXT NOT NULL,
    embedding   BYTEA NOT NULL,
    photo_path  TEXT NOT NULL REFERENCES photos(path) ON DELETE CASCADE,
    bbox        JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_photos_mtime    ON photos(mtime);
CREATE INDEX IF NOT EXISTS idx_photos_taken_at ON photos(taken_at);
CREATE INDEX IF NOT EXISTS idx_faces_cluster   ON faces(cluster_id);
CREATE INDEX IF NOT EXISTS idx_faces_name      ON faces(name);

-- Migrate existing tables
ALTER TABLE photos ADD COLUMN IF NOT EXISTS taken_at        TIMESTAMPTZ;
ALTER TABLE photos ADD COLUMN IF NOT EXISTS immich_asset_id TEXT;
"""


class Database:
    def __init__(self, url: str):
        self._url = url
        self._pool: asyncpg.Pool | None = None

    async def init(self) -> None:
        self._pool = await asyncpg.create_pool(self._url, min_size=2, max_size=10)
        async with self._pool.acquire() as conn:
            await conn.execute(CREATE_TABLES)

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()

    async def get_photo(self, path: str) -> dict | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM photos WHERE path = $1", path)
            return dict(row) if row else None

    async def upsert_photo(self, path: str, mtime: float, sha256: str, **fields) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO photos (path, mtime, sha256, caption, tags, faces, duplicate_of, ocr_text, taken_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (path) DO UPDATE SET
                    mtime        = EXCLUDED.mtime,
                    sha256       = EXCLUDED.sha256,
                    processed_at = now(),
                    caption      = EXCLUDED.caption,
                    tags         = EXCLUDED.tags,
                    faces        = EXCLUDED.faces,
                    duplicate_of = EXCLUDED.duplicate_of,
                    ocr_text     = EXCLUDED.ocr_text,
                    taken_at     = EXCLUDED.taken_at
                """,
                path,
                mtime,
                sha256,
                fields.get("caption"),
                json.dumps(fields.get("tags", [])),
                json.dumps(fields.get("faces", [])),
                fields.get("duplicate_of"),
                fields.get("ocr_text"),
                fields.get("taken_at"),
            )

    async def is_processed(self, path: str, mtime: float, sha256: str) -> bool:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT 1 FROM photos WHERE path = $1 AND mtime = $2 AND sha256 = $3",
                path, mtime, sha256,
            )
            return row is not None

    async def get_stats(self) -> dict:
        async with self._pool.acquire() as conn:
            photos = await conn.fetchval("SELECT COUNT(*) FROM photos")
            named_faces = await conn.fetchval("SELECT COUNT(DISTINCT name) FROM faces WHERE name IS NOT NULL")
            total_faces = await conn.fetchval("SELECT COUNT(*) FROM faces")
            dupes = await conn.fetchval("SELECT COUNT(*) FROM photos WHERE duplicate_of IS NOT NULL")
            return {
                "total_photos": photos,
                "named_people": named_faces,
                "total_faces": total_faces,
                "duplicates": dupes,
            }

    async def list_all_photos(self) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT path, caption, tags, taken_at, immich_asset_id
                FROM photos
                WHERE caption IS NOT NULL
                ORDER BY taken_at NULLS LAST
                """
            )
            return [dict(r) for r in rows]

    async def update_immich_asset_id(self, path: str, asset_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE photos SET immich_asset_id = $1 WHERE path = $2",
                asset_id, path,
            )

    async def list_face_clusters(self) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT cluster_id, name, COUNT(*) as count
                FROM faces
                GROUP BY cluster_id, name
                ORDER BY count DESC
                """
            )
            return [dict(r) for r in rows]

    async def name_cluster(self, cluster_id: str, name: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE faces SET name = $1 WHERE cluster_id = $2", name, cluster_id
            )
