"""photo-ai CLI — scan-new | scan-all | scan-file | serve | stats | faces"""

import asyncio
import click
from rich.console import Console
from rich.table import Table

from photo_ai.config import Settings
from photo_ai.db import Database
from photo_ai.vector_store import VectorStore
from photo_ai.scanner import PhotoScanner

console = Console()


def _get_stack(settings: Settings) -> tuple[Database, VectorStore]:
    db = Database(settings.database_url)
    vs = VectorStore(settings.qdrant_url, settings.qdrant_collection)
    return db, vs


async def _init(settings: Settings) -> tuple[Database, VectorStore]:
    db, vs = _get_stack(settings)
    await db.init()
    await vs.init()
    return db, vs


@click.group()
def cli():
    pass


@cli.command("scan-new")
@click.option("--report-url", envvar="PHOTO_AI_REPORT_URL", default=None)
def scan_new(report_url: str | None):
    """Process only new or modified photos."""
    settings = Settings()
    settings.incremental = True

    async def _run():
        db, vs = await _init(settings)
        scanner = PhotoScanner(settings, db, vs)
        summary = await scanner.run(full=False, report_url=report_url)
        console.print(f"[green]Done:[/green] {summary}")
        await db.close()
        await vs.close()

    asyncio.run(_run())


@cli.command("scan-all")
@click.option("--report-url", envvar="PHOTO_AI_REPORT_URL", default=None)
def scan_all(report_url: str | None):
    """Full rescan — ignores previous processing state."""
    settings = Settings()
    settings.incremental = False

    async def _run():
        db, vs = await _init(settings)
        scanner = PhotoScanner(settings, db, vs)
        summary = await scanner.run(full=True, report_url=report_url)
        console.print(f"[green]Done:[/green] {summary}")
        await db.close()
        await vs.close()

    asyncio.run(_run())


@cli.command("scan-file")
@click.argument("file_path")
def scan_file(file_path: str):
    """Process a single photo file."""
    from pathlib import Path
    settings = Settings()
    settings.incremental = False

    async def _run():
        db, vs = await _init(settings)
        scanner = PhotoScanner(settings, db, vs)
        result = await scanner._process_one(Path(file_path))
        console.print(result)
        await db.close()
        await vs.close()

    asyncio.run(_run())


@cli.command("serve")
def serve():
    """Run as webhook receiver (runner mode)."""
    import uvicorn
    from photo_ai.runner import create_app
    settings = Settings()
    app = create_app(settings)
    uvicorn.run(app, host=settings.runner_host, port=settings.runner_port)


@cli.command("stats")
def stats():
    """Print processing statistics."""
    settings = Settings()

    async def _run():
        db, _ = _get_stack(settings)
        await db.init()
        s = await db.get_stats()
        await db.close()

        table = Table(title="Photo AI Stats")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        table.add_row("Total photos processed", str(s["total_photos"]))
        table.add_row("Named people", str(s["named_people"]))
        table.add_row("Total faces detected", str(s["total_faces"]))
        table.add_row("Duplicates found", str(s["duplicates"]))
        console.print(table)

    asyncio.run(_run())


@cli.command("faces")
def faces():
    """List face clusters (named and unnamed)."""
    settings = Settings()

    async def _run():
        db, _ = _get_stack(settings)
        await db.init()
        clusters = await db.list_face_clusters()
        await db.close()

        table = Table(title="Face Clusters")
        table.add_column("Cluster ID")
        table.add_column("Name")
        table.add_column("Photo count", justify="right")
        for c in clusters:
            table.add_row(c["cluster_id"], c["name"] or "[dim]unknown[/dim]", str(c["count"]))
        console.print(table)

    asyncio.run(_run())


@cli.command("name-face")
@click.argument("cluster_id")
@click.argument("name")
def name_face(cluster_id: str, name: str):
    """Assign a name to a face cluster."""
    settings = Settings()

    async def _run():
        db, _ = _get_stack(settings)
        await db.init()
        await db.name_cluster(cluster_id, name)
        await db.close()
        console.print(f"[green]Cluster {cluster_id} named '{name}'[/green]")

    asyncio.run(_run())


if __name__ == "__main__":
    cli()
