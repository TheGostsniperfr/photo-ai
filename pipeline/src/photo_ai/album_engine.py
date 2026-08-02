"""Auto-album creation: temporal event clustering + semantic theme detection."""

import json
from dataclasses import dataclass
from datetime import datetime, timedelta

from rich.console import Console

from photo_ai.immich_client import ImmichClient

console = Console()

# Keywords matched against Florence-2 captions + tags (lowercase)
THEMES: dict[str, list[str]] = {
    "Noel": [
        "christmas", "christmas tree", "gift", "santa", "decoration",
        "ornament", "wreath", "advent",
    ],
    "Anniversaire": [
        "birthday", "birthday cake", "candle", "celebration", "party", "balloon",
    ],
    "Musee": [
        "painting", "sculpture", "gallery", "museum", "artwork", "exhibition",
        "portrait", "canvas",
    ],
    "Vacances - Plage": [
        "beach", "sea", "ocean", "sand", "vacation", "sunbathing",
        "waves", "coastline", "swimsuit",
    ],
    "Nature - Randonnee": [
        "forest", "mountain", "hiking", "trail", "nature", "landscape",
        "valley", "waterfall", "cliff",
    ],
    "Restaurant - Repas": [
        "restaurant", "food", "meal", "dinner", "lunch", "dish",
        "cuisine", "plate", "menu",
    ],
    "Sport": [
        "sport", "running", "cycling", "gym", "workout", "fitness",
        "race", "match", "stadium",
    ],
    "Animaux": [
        "dog", "cat", "animal", "pet", "bird", "rabbit", "horse",
        "puppy", "kitten", "wildlife",
    ],
    "Concert - Festival": [
        "concert", "festival", "stage", "music", "crowd", "performer",
        "microphone", "audience",
    ],
    "Neige - Hiver": [
        "snow", "snowflake", "winter", "skiing", "sledding", "frost", "ice", "blizzard",
    ],
}


@dataclass
class PhotoMeta:
    path: str
    caption: str
    tags: list[str]
    taken_at: datetime | None
    asset_id: str | None = None


class AlbumEngine:
    def __init__(
        self,
        immich: ImmichClient,
        event_gap_hours: int = 6,
        min_event_photos: int = 5,
        min_theme_photos: int = 3,
    ):
        self._immich = immich
        self._gap = timedelta(hours=event_gap_hours)
        self._min_event = min_event_photos
        self._min_theme = min_theme_photos
        self._album_cache: dict[str, str] = {}  # name → album_id

    async def run(self, photos: list[PhotoMeta]) -> dict:
        albums = await self._immich.get_albums()
        self._album_cache = {a["albumName"]: a["id"] for a in albums}

        # Resolve Immich asset IDs for photos that don't have one cached
        missing = [p for p in photos if p.asset_id is None]
        console.print(f"[blue]Resolving {len(missing)} Immich asset IDs...[/blue]")
        for photo in missing:
            photo.asset_id = await self._immich.find_asset_id(photo.path)

        valid = [p for p in photos if p.asset_id]
        console.print(f"[green]{len(valid)} photos matched in Immich[/green]")

        # Push Florence-2 descriptions to Immich
        pushed = sum(1 for p in valid if p.caption)
        for photo in valid:
            if photo.caption:
                await self._immich.set_description(photo.asset_id, photo.caption)
        console.print(f"[green]Pushed {pushed} descriptions[/green]")

        event_count = await self._create_event_albums(valid)
        theme_count = await self._create_theme_albums(valid)

        return {
            "descriptions_pushed": pushed,
            "event_albums": event_count,
            "theme_albums": theme_count,
        }

    async def _create_event_albums(self, photos: list[PhotoMeta]) -> int:
        dated = sorted(
            [p for p in photos if p.taken_at is not None],
            key=lambda p: p.taken_at,
        )
        if not dated:
            return 0

        events: list[list[PhotoMeta]] = []
        current = [dated[0]]
        for photo in dated[1:]:
            if photo.taken_at - current[-1].taken_at > self._gap:
                events.append(current)
                current = []
            current.append(photo)
        events.append(current)

        created = 0
        for event in events:
            if len(event) < self._min_event:
                continue
            date_str = event[0].taken_at.strftime("%d %B %Y")
            asset_ids = [p.asset_id for p in event if p.asset_id]
            await self._upsert_album(f"Event - {date_str}", asset_ids)
            created += 1

        console.print(f"[green]{created} event albums[/green]")
        return created

    async def _create_theme_albums(self, photos: list[PhotoMeta]) -> int:
        buckets: dict[str, list[str]] = {t: [] for t in THEMES}

        for photo in photos:
            if not photo.asset_id:
                continue
            text = (photo.caption + " " + " ".join(photo.tags)).lower()
            for theme, keywords in THEMES.items():
                if any(kw in text for kw in keywords):
                    buckets[theme].append(photo.asset_id)

        created = 0
        for theme, ids in buckets.items():
            if len(ids) < self._min_theme:
                continue
            await self._upsert_album(theme, ids)
            created += 1

        console.print(f"[green]{created} theme albums[/green]")
        return created

    async def _upsert_album(self, name: str, asset_ids: list[str]) -> None:
        if name in self._album_cache:
            await self._immich.add_assets_to_album(self._album_cache[name], asset_ids)
        else:
            album_id = await self._immich.create_album(name, asset_ids)
            self._album_cache[name] = album_id
