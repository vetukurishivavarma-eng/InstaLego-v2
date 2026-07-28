"""FastAPI delivery layer for the landtitle pipeline.

Run locally with:
    uvicorn landtitle.api.app:app --reload

See the README's "Web API (local)" section for the full local-run
instructions, required env vars, and frontend location.

This module only ever imports and calls into landtitle.pipeline /
landtitle.opinion.pdf_export -- it does not modify or re-implement any
pipeline logic. Submission is synchronous only in the sense of returning a
job id immediately; the actual (slow, LLM-driven) pipeline run happens on a
background worker thread managed by jobs.JobManager.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from landtitle.api.jobs import DONE, PipelineInputError, job_manager
from landtitle.api.models import JobCreateResponse, JobStatusResponse
from landtitle.config import LOG_LEVEL
from landtitle.opinion.pdf_export import export_opinion_pdf
from landtitle.pipeline import run_pipeline

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Land Title Due Diligence Engine API",
    description="Thin HTTP layer over the landtitle pipeline: submit documents, poll status, download the draft opinion PDF.",
    version="1.0.0",
)

# CORS: the frontend is hosted separately (Vercel) and calls this backend
# cross-origin over an ngrok tunnel. Defaulting to "*" for now since this API
# has no auth/session/cookie state (file-in, file-out only) -- TODO tighten
# this to the actual https://*.vercel.app origin once it's known.
_cors_origins = [o.strip() for o in os.environ.get("LANDTITLE_CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

_PDF_MAGIC = b"%PDF-"


def _validate_pdf(filename: Optional[str], content: bytes, label: str) -> None:
    if not content:
        raise HTTPException(status_code=400, detail=f"{label}: uploaded file is empty.")
    if not filename or not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail=f"{label}: only .pdf files are accepted.")
    if not content.startswith(_PDF_MAGIC):
        raise HTTPException(status_code=400, detail=f"{label}: file content does not look like a PDF.")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/opinions", response_model=JobCreateResponse, status_code=202)
async def create_opinion_job(
    sale_deed: List[UploadFile] = File(..., description="One or more Sale Deed PDFs, chronological order (oldest first)"),
    revenue_record: Optional[UploadFile] = File(None, description="Optional Pahani/Adangal PDF"),
    ec: Optional[UploadFile] = File(None, description="Optional Encumbrance Certificate PDF"),
) -> JobCreateResponse:
    sale_deed_payload: list[tuple[str, bytes]] = []
    for f in sale_deed:
        content = await f.read()
        _validate_pdf(f.filename, content, "Sale Deed")
        sale_deed_payload.append((f.filename or "sale_deed.pdf", content))
        logger.info("Upload received: sale deed %r (%d bytes)", f.filename, len(content))

    revenue_record_payload = None
    if revenue_record is not None and revenue_record.filename:
        content = await revenue_record.read()
        _validate_pdf(revenue_record.filename, content, "Revenue Record")
        revenue_record_payload = (revenue_record.filename, content)
        logger.info("Upload received: revenue record %r (%d bytes)", revenue_record.filename, len(content))

    ec_payload = None
    if ec is not None and ec.filename:
        content = await ec.read()
        _validate_pdf(ec.filename, content, "Encumbrance Certificate")
        ec_payload = (ec.filename, content)
        logger.info("Upload received: EC %r (%d bytes)", ec.filename, len(content))

    try:
        record = job_manager.create_job(sale_deed_payload, revenue_record_payload, ec_payload)
    except PipelineInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return JobCreateResponse(job_id=record.job_id, status=record.status)


@app.get("/opinions/{job_id}", response_model=JobStatusResponse)
async def get_opinion_job(job_id: str) -> JobStatusResponse:
    record = job_manager.get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found (unknown id, or its retention window has elapsed).")
    return JobStatusResponse(
        job_id=record.job_id,
        status=record.status,
        error=record.error,
        sale_deed_count=record.sale_deed_count,
        has_revenue_record=record.has_revenue_record,
        has_ec=record.has_ec,
        created_at=record.created_at,
        finished_at=record.finished_at,
    )


@app.get("/opinions/{job_id}/download")
async def download_opinion(job_id: str):
    record = job_manager.get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found (unknown id, or its retention window has elapsed).")
    if record.status != DONE:
        raise HTTPException(status_code=409, detail=f"Job is not ready yet (status={record.status}).")
    if not record.output_path or not os.path.isfile(record.output_path):
        raise HTTPException(status_code=410, detail="Result has already been cleaned up (retention window elapsed).")

    job_manager.mark_downloaded(job_id)
    return FileResponse(
        record.output_path,
        media_type="application/pdf",
        filename="opinion_draft.pdf",
    )


@app.post("/generate-opinion")
def generate_opinion_sync(
    sale_deed: List[UploadFile] = File(..., description="One or more Sale Deed PDFs, chronological order (oldest first)"),
    revenue_record: Optional[UploadFile] = File(None, description="Optional Pahani/Adangal PDF"),
    ec: Optional[UploadFile] = File(None, description="Optional Encumbrance Certificate PDF"),
):
    """Synchronous companion to POST /opinions: no job id, no polling -- just
    upload, wait, and get the PDF back directly. Defined as a plain `def`
    (NOT `async def`) so Starlette dispatches it to its worker threadpool
    instead of the event loop; the pipeline run below blocks for several
    minutes and must never do that on the loop thread that also serves
    /health, /opinions/*, etc.

    UploadFile.read() is async and unusable from a sync def (no event loop
    to await against on a worker thread), so uploads are read via the
    underlying `.file` (a plain, blocking file object) instead -- see the
    UploadFile docstring in fastapi for this exact pattern.
    """
    tmp_dir = tempfile.mkdtemp(prefix="landtitle_sync_")
    cleanup_deferred = False
    try:
        sale_deed_paths: list[str] = []
        for i, f in enumerate(sale_deed):
            content = f.file.read()
            _validate_pdf(f.filename, content, "Sale Deed")
            path = os.path.join(tmp_dir, f"sale_deed_{i}.pdf")
            Path(path).write_bytes(content)
            sale_deed_paths.append(path)
            logger.info("Sync upload received: sale deed %r (%d bytes)", f.filename, len(content))

        revenue_record_path = None
        if revenue_record is not None and revenue_record.filename:
            content = revenue_record.file.read()
            _validate_pdf(revenue_record.filename, content, "Revenue Record")
            revenue_record_path = os.path.join(tmp_dir, "revenue_record.pdf")
            Path(revenue_record_path).write_bytes(content)
            logger.info("Sync upload received: revenue record %r (%d bytes)", revenue_record.filename, len(content))

        ec_path = None
        if ec is not None and ec.filename:
            content = ec.file.read()
            _validate_pdf(ec.filename, content, "Encumbrance Certificate")
            ec_path = os.path.join(tmp_dir, "ec.pdf")
            Path(ec_path).write_bytes(content)
            logger.info("Sync upload received: EC %r (%d bytes)", ec.filename, len(content))

        output_path = os.path.join(tmp_dir, "opinion_draft.pdf")
        try:
            opinion, _flags = run_pipeline(sale_deed_paths, revenue_record_path, ec_path)
            export_opinion_pdf(opinion, output_path)
        except Exception as exc:  # noqa: BLE001 - deliberate failure boundary: never leak a traceback to the client
            logger.exception("Synchronous opinion generation failed")
            return JSONResponse(status_code=500, content={"detail": f"Pipeline failed: {str(exc)[:300]}"})

        cleanup_deferred = True  # FileResponse streams after we return -- the BackgroundTask below cleans up instead
        return FileResponse(
            output_path,
            media_type="application/pdf",
            filename="opinion_draft.pdf",
            background=BackgroundTask(shutil.rmtree, tmp_dir, ignore_errors=True),
        )
    finally:
        if not cleanup_deferred:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# --- Static frontend -------------------------------------------------------
# Mounted LAST and at "/" so it acts as a catch-all fallback: Starlette
# matches routes in registration order, so the explicit /opinions* routes
# above are always matched first, and only paths that don't match one of
# them fall through to serving frontend/index.html (or other static assets).
_FRONTEND_DIR = Path(__file__).resolve().parents[3] / "frontend"
if _FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")
else:
    logger.warning("Frontend directory not found at %s; static UI will not be served.", _FRONTEND_DIR)
