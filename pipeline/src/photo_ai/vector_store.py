"""Qdrant vector store for semantic photo search."""

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
import uuid


VECTOR_SIZE = 768  # CLIP ViT-L/14


class VectorStore:
    def __init__(self, url: str, collection: str):
        self._client = AsyncQdrantClient(url=url)
        self._collection = collection

    async def init(self) -> None:
        existing = [c.name for c in (await self._client.get_collections()).collections]
        if self._collection not in existing:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )

    async def upsert(self, path: str, embedding: list[float], payload: dict) -> None:
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, path))
        await self._client.upsert(
            collection_name=self._collection,
            points=[PointStruct(id=point_id, vector=embedding, payload={"path": path, **payload})],
        )

    async def search(self, embedding: list[float], limit: int = 10) -> list[dict]:
        results = await self._client.search(
            collection_name=self._collection,
            query_vector=embedding,
            limit=limit,
            with_payload=True,
        )
        return [{"path": r.payload["path"], "score": r.score, **r.payload} for r in results]

    async def close(self) -> None:
        await self._client.close()
