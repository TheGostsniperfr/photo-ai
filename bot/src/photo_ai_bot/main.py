"""Discord bot — photo-ai dashboard, trigger, status reporting."""

import asyncio
import os
from datetime import datetime
from typing import Optional

import discord
from discord import app_commands
import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict


class BotSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BOT_")

    discord_token: str
    discord_guild_id: int

    runner_url: str = "http://photo-ai-runner.photo-ai.svc.cluster.local:8765"
    runner_auth_token: str = "changeme"

    ollama_url: str = "http://ollama-external-service.ollama.svc.cluster.local:11434"


settings = BotSettings()

intents = discord.Intents.default()
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)
GUILD = discord.Object(id=settings.discord_guild_id)

_http = httpx.AsyncClient(
    headers={"Authorization": f"Bearer {settings.runner_auth_token}"},
    base_url=settings.runner_url,
    timeout=30.0,
)


def _runner_available() -> bool:
    try:
        r = httpx.get(f"{settings.runner_url}/healthz", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _ollama_available() -> bool:
    try:
        r = httpx.get(f"{settings.ollama_url}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


@client.event
async def on_ready():
    await tree.sync(guild=GUILD)
    print(f"Logged in as {client.user}")


@tree.command(name="photos-scan-new", description="Process only new/modified photos", guild=GUILD)
async def scan_new(interaction: discord.Interaction):
    await _trigger(interaction, "scan-new")


@tree.command(name="photos-scan-all", description="Full rescan of all photos (slow)", guild=GUILD)
async def scan_all(interaction: discord.Interaction):
    await _trigger(interaction, "scan-all")


async def _trigger(interaction: discord.Interaction, mode: str):
    await interaction.response.defer(ephemeral=False)

    if not _ollama_available():
        await interaction.followup.send(
            embed=_embed("Workstation Offline", "Ollama is not reachable — is the workstation on?", discord.Color.red())
        )
        return

    try:
        resp = await _http.post("/api/run", json={"mode": mode})
        resp.raise_for_status()
        job_id = resp.json()["job_id"]
    except Exception as e:
        await interaction.followup.send(
            embed=_embed("Runner Error", str(e), discord.Color.red())
        )
        return

    msg = await interaction.followup.send(
        embed=_embed("Pipeline Started", f"`{mode}` — job `{job_id}`\nPolling for updates...", discord.Color.blue())
    )

    # Poll until done
    for _ in range(120):  # max 10 minutes
        await asyncio.sleep(5)
        try:
            status_resp = await _http.get(f"/api/status/{job_id}")
            status = status_resp.json()
        except Exception:
            continue

        if status["status"] == "done":
            r = status.get("result", {})
            await msg.edit(embed=_embed(
                "Done",
                f"Processed **{r.get('processed', '?')}** photos out of **{r.get('total', '?')}**\n"
                f"Errors: {r.get('errors', 0)}",
                discord.Color.green(),
            ))
            return

        if status["status"] == "error":
            await msg.edit(embed=_embed("Error", status.get("error", "unknown"), discord.Color.red()))
            return

    await msg.edit(embed=_embed("Timeout", "Pipeline is still running — check logs.", discord.Color.yellow()))


@tree.command(name="photos-stats", description="Show photo library statistics", guild=GUILD)
async def stats(interaction: discord.Interaction):
    await interaction.response.defer()
    try:
        resp = await _http.get("/api/status")
        jobs = resp.json()
        running = [j for j in jobs if j["status"] == "running"]
    except Exception:
        running = []

    embed = discord.Embed(title="Photo AI — Status", color=discord.Color.blurple())
    embed.add_field(name="Runner", value="Online" if _runner_available() else "Offline", inline=True)
    embed.add_field(name="Ollama", value="Online" if _ollama_available() else "Offline", inline=True)
    embed.add_field(name="Active jobs", value=str(len(running)), inline=True)
    embed.timestamp = datetime.utcnow()
    await interaction.followup.send(embed=embed)


@tree.command(name="photos-search", description="Semantic search across your photos", guild=GUILD)
@app_commands.describe(query="Natural language query, e.g. 'friends at a party'")
async def search(interaction: discord.Interaction, query: str):
    await interaction.response.send_message(
        embed=_embed("Search", f"Semantic search for: `{query}`\n*(not yet implemented — add Qdrant HTTP endpoint)*", discord.Color.yellow())
    )


def _embed(title: str, description: str, color: discord.Color) -> discord.Embed:
    e = discord.Embed(title=title, description=description, color=color)
    e.timestamp = datetime.utcnow()
    return e


def main():
    client.run(settings.discord_token)


if __name__ == "__main__":
    main()
