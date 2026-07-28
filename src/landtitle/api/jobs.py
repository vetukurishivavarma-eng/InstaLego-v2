"""In-memory async job tracking for the opinion-generation API.

Design notes (see also app.py and the project README's "Web API" section):

- `run_pipeline()` makes several slow, sequential LLM calls against what may
  be a personal/unstable endpoint (see landtitle.llm.client.QwenClient) --
  it must never run on the request-handling thread, or a client-side HTTP
  timeout would fire long before the pipeline finishes. Each job runs on a
  worker thread from a bounded ThreadPoolExecutor instead; the HTTP layer
  only ever creates a job record and returns its id immediately.
- Uploaded document bytes are written under the OS temp directory
  (tempfile.mkdtemp), one subdirectory per job -- never inside the repo.
  Filenames on disk are fixed/sanitized (sale_deed_0.pdf, ...), never the
  client-supplied filename, so a crafted filename can't escape the job
  directory or collide with another job.
- Input PDFs are deleted the moment the pipeline has consumed them
  (success or failure) -- only the generated opinion PDF may outlive that,
  and only until it's downloaded or the retention window elapses, whichever
  a background reaper thread finds first.
- Nothing here logs document content: log lines carry job ids, filenames,
  and byte counts only.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from landtitle.opinion.pdf_export import export_opinion_pdf
from landtitle.pipeline import run_pipeline

logger = logging.getLogger(__name__)

# --- Job status values ---
PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"

# --- Tunables (env-overridable; local single-process use, see README) ---
JOB_RETENTION_SECONDS = float(os.environ.get("LANDTITLE_JOB_RETENTION_SECONDS", "1800"))  # 30 min
JOB_MAX_WORKERS = int(os.environ.get("LANDTITLE_JOB_MAX_WORKERS", "2"))
_CLEANUP_INTERVAL_SECONDS = max(5.0, min(JOB_RETENTION_SECONDS / 2, 60.0))


class PipelineInputError(ValueError):
    """Raised for bad/missing upload input -- surfaced as HTTP 400."""


@dataclass
class JobRecord:
    job_id: str
    job_dir: str
    status: str = PENDING
    error: Optional[str] = None
    output_path: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    downloaded: bool = False
    # Metadata only -- never document content -- kept for the status endpoint.
    sale_deed_count: int = 0
    has_revenue_record: bool = False
    has_ec: bool = False


class JobManager:
    def __init__(self, retention_seconds: float = JOB_RETENTION_SECONDS, max_workers: int = JOB_MAX_WORKERS) -> None:
        self._retention_seconds = retention_seconds
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="landtitle-job")
        self._stop_event = threading.Event()
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True, name="landtitle-job-reaper")
        self._cleanup_thread.start()

    # --- public API -----------------------------------------------------

    def create_job(
        self,
        sale_deeds: list[tuple[str, bytes]],
        revenue_record: Optional[tuple[str, bytes]],
        ec: Optional[tuple[str, bytes]],
    ) -> JobRecord:
        """`sale_deeds`/`revenue_record`/`ec` are (original_filename, content)
        pairs already validated by the caller (app.py) to look like PDFs.
        Writes them to a fresh job directory under the OS temp dir and
        schedules the pipeline run; returns immediately with a PENDING job."""
        if not sale_deeds:
            raise PipelineInputError("At least one Sale Deed file is required.")

        job_id = uuid.uuid4().hex
        job_dir = tempfile.mkdtemp(prefix=f"landtitle_job_{job_id}_")

        sale_deed_paths: list[str] = []
        for i, (_name, content) in enumerate(sale_deeds):
            path = os.path.join(job_dir, f"sale_deed_{i}.pdf")
            Path(path).write_bytes(content)
            sale_deed_paths.append(path)

        revenue_record_path = None
        if revenue_record is not None:
            revenue_record_path = os.path.join(job_dir, "revenue_record.pdf")
            Path(revenue_record_path).write_bytes(revenue_record[1])

        ec_path = None
        if ec is not None:
            ec_path = os.path.join(job_dir, "ec.pdf")
            Path(ec_path).write_bytes(ec[1])

        record = JobRecord(
            job_id=job_id,
            job_dir=job_dir,
            sale_deed_count=len(sale_deeds),
            has_revenue_record=revenue_record is not None,
            has_ec=ec is not None,
        )
        with self._lock:
            self._jobs[job_id] = record

        logger.info(
            "Job %s created: %d sale deed(s), revenue_record=%s, ec=%s",
            job_id, len(sale_deeds), revenue_record is not None, ec is not None,
        )

        self._executor.submit(self._run, job_id, sale_deed_paths, revenue_record_path, ec_path)
        return record

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._jobs.get(job_id)

    def mark_downloaded(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.downloaded = True

    def shutdown(self) -> None:
        """Best-effort teardown for tests/process exit."""
        self._stop_event.set()
        self._executor.shutdown(wait=False, cancel_futures=True)

    # --- internals --------------------------------------------------------

    def _run(
        self,
        job_id: str,
        sale_deed_paths: list[str],
        revenue_record_path: Optional[str],
        ec_path: Optional[str],
    ) -> None:
        record = self.get_job(job_id)
        if record is None:
            return  # job was reaped before it even started; nothing to do

        record.status = RUNNING
        logger.info("Job %s: pipeline started", job_id)
        try:
            opinion, _flags = run_pipeline(sale_deed_paths, revenue_record_path, ec_path)
        except Exception as exc:  # noqa: BLE001 - deliberately broad: a job-level failure boundary
            record.error = str(exc)
            record.status = FAILED
            logger.warning("Job %s: pipeline failed: %s", job_id, exc)
        else:
            try:
                output_path = os.path.join(record.job_dir, "opinion_draft.pdf")
                export_opinion_pdf(opinion, output_path)
            except Exception as exc:  # noqa: BLE001
                record.error = str(exc)
                record.status = FAILED
                logger.warning("Job %s: PDF export failed: %s", job_id, exc)
            else:
                record.output_path = output_path
                record.status = DONE
                logger.info("Job %s: done", job_id)
        finally:
            record.finished_at = time.time()
            # Input documents are consumed at this point (success or
            # failure) -- delete them immediately regardless of outcome.
            for path in [*sale_deed_paths, revenue_record_path, ec_path]:
                if path and os.path.isfile(path):
                    try:
                        os.remove(path)
                    except OSError:
                        logger.warning("Job %s: could not remove input file", job_id)

    def _cleanup_loop(self) -> None:
        while not self._stop_event.wait(_CLEANUP_INTERVAL_SECONDS):
            self._reap_expired()

    def _reap_expired(self) -> None:
        now = time.time()
        with self._lock:
            expired = [
                (job_id, record.job_dir)
                for job_id, record in self._jobs.items()
                if record.status in (DONE, FAILED)
                and record.finished_at is not None
                and (now - record.finished_at) > self._retention_seconds
            ]
            for job_id, _job_dir in expired:
                del self._jobs[job_id]
        for job_id, job_dir in expired:
            shutil.rmtree(job_dir, ignore_errors=True)
            logger.info("Job %s: expired, cleaned up", job_id)


# Process-wide singleton -- single-process, in-memory job tracking as
# documented in the task spec (a first cut; not multi-process safe).
job_manager = JobManager()
