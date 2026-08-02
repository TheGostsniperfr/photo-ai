"""Thin async client for the Immich REST API."""

import httpx


class ImmichClient:
    def __init__(self, base_url: str, api_key: str, path_prefix: str = "/mnt"):
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"x-api-key": api_key},
            timeout=30.0,
        )
        # Pipeline paths (/photos/...) → Immich paths (/mnt/photos/...)
        self._prefix = path_prefix

    def _immich_path(self, pipeline_path: str) -> str:
        return self._prefix + pipeline_path

    async def find_asset_id(self, pipeline_path: str) -> str | None:
        resp = await self._http.post("/api/search/metadata", json={
            "originalPath": self._immich_path(pipeline_path),
        })
        if not resp.is_success:
            return None
        items = resp.json().get("assets", {}).get("items", [])
        return items[0]["id"] if items else None

    async def set_description(self, asset_id: str, description: str) -> None:
        await self._http.put(f"/api/assets/{asset_id}", json={"description": description})

    async def get_albums(self) -> list[dict]:
        resp = await self._http.get("/api/albums")
        resp.raise_for_status()
        return resp.json()

    async def create_album(self, name: str, asset_ids: list[str], description: str = "") -> str:
        resp = await self._http.post("/api/albums", json={
            "albumName": name,
            "description": description,
            "assetIds": asset_ids,
        })
        resp.raise_for_status()
        return resp.json()["id"]

    async def add_assets_to_album(self, album_id: str, asset_ids: list[str]) -> None:
        await self._http.put(f"/api/albums/{album_id}/assets", json={"ids": asset_ids})

    async def close(self) -> None:
        await self._http.aclose()
