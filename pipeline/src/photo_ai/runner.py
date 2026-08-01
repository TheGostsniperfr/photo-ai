"""FastAPI webhook receiver — accepts trigger from Discord bot / k8s CronJob."""

import asyncio
import uuid
from datetime import datetime
from typing import Literal

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from photo_ai.config import Settings
from photo_ai.db import Database
from photo_ai.vector_store import VectorStore
from photo_ai.scanner import PhotoScanner


class RunRequest(BaseModel):
    mode: Literal["scan-new", "scan-all"] = "scan-new"
    report_url: str | None = None


class JobStatus(BaseModel):
    job_id: str
    status: Literal["running", "done", "error"]
    started_at: str
    finished_at: str | None = None
    result: dict | None = None
    error: str | None = None


_jobs: dict[str, JobStatus] = {}
_lock = asyncio.Lock()


def create_app(settings: Settings) -> FastAPI:
    app = FastAPI(title="photo-ai runner", docs_url=None)
    bearer = HTTPBearer()

    def _verify_token(credentials: HTTPAuthorizationCredentials = Depends(bearer)):
        if credentials.credentials != settings.runner_auth_token:
            raise HTTPException(status_code=401, detail="Invalid token")

    @app.post("/api/run", dependencies=[Depends(_verify_token)])
    async def run(req: RunRequest) -> dict:
        job_id = str(uuid.uuid4())
        status = JobStatus(
            job_id=job_id,
            status="running",
            started_at=datetime.utcnow().isoformat(),
        )
        async with _lock:
            _jobs[job_id] = status

        asyncio.create_task(_execute(job_id, req, settings))
        return {"job_id": job_id}

    @app.get("/api/status/{job_id}", dependencies=[Depends(_verify_token)])
    async def get_status(job_id: str) -> JobStatus:
        if job_id not in _jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        return _jobs[job_id]

    @app.get("/api/status")
    async def list_jobs() -> list[JobStatus]:
        return list(_jobs.values())

    @app.get("/healthz")
    async def health() -> dict:
        return {"status": "ok"}

    return app


async def _execute(job_id: str, req: RunRequest, settings: Settings) -> None:
    try:
        db = Database(settings.database_url)
        vs = VectorStore(settings.qdrant_url, settings.qdrant_collection)
        await db.init()
        await vs.init()

        scanner = PhotoScanner(settings, db, vs)
        result = await scanner.run(
            full=(req.mode == "scan-all"),
            report_url=req.report_url,
        )

        await db.close()
        await vs.close()

        async with _lock:
            _jobs[job_id].status = "done"
            _jobs[job_id].finished_at = datetime.utcnow().isoformat()
            _jobs[job_id].result = result

    except Exception as e:
        async with _lock:
            _jobs[job_id].status = "error"
            _jobs[job_id].finished_at = datetime.utcnow().isoformat()
            _jobs[job_id].error = str(e)
