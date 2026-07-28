"""Tests for the FastAPI delivery layer (src/landtitle/api/).

These test the job/API/file-handling layer only -- landtitle.pipeline.run_pipeline
is monkeypatched in every test so nothing here makes a real LLM call or does
real OCR (that's already covered elsewhere against real documents, per the
project README). export_opinion_pdf is NOT mocked: it's fast, pure-code, and
exercising it for real lets the download tests check an actual PDF comes back.
"""
from __future__ import annotations

import os
import time

import pytest
from fastapi.testclient import TestClient

import landtitle.api.jobs as jobs_module
from landtitle.api.app import app
from landtitle.schemas import Flag, LegalOpinion

client = TestClient(app)

_MINIMAL_PDF = b"%PDF-1.4\n%fake test fixture, not a real document\n%%EOF"

_FAKE_OPINION = LegalOpinion(
    property_summary="Fictional 200 sq. yard plot for test purposes only.",
    chain_of_ownership="Single vendor to single purchaser, no gaps.",
    encumbrance_status="No active encumbrances found in the test fixture EC.",
    legal_compliance_check="Registered instrument; no compliance issues raised.",
    risk_flags_narrative="No risk flags raised in this test run.",
    overall_recommendation="Clear Title",
)


def _pdf_files(n_sale_deeds: int = 1, revenue_record: bool = False, ec: bool = False):
    files = [("sale_deed", (f"deed_{i}.pdf", _MINIMAL_PDF, "application/pdf")) for i in range(n_sale_deeds)]
    if revenue_record:
        files.append(("revenue_record", ("pahani.pdf", _MINIMAL_PDF, "application/pdf")))
    if ec:
        files.append(("ec", ("ec.pdf", _MINIMAL_PDF, "application/pdf")))
    return files


def _wait_for_status(job_id: str, target_statuses, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        resp = client.get(f"/opinions/{job_id}")
        assert resp.status_code == 200
        last = resp.json()
        if last["status"] in target_statuses:
            return last
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not reach {target_statuses} in time, last={last}")


# --- happy path -------------------------------------------------------------

def test_submit_poll_download_success(monkeypatch):
    captured_paths = {}

    def fake_run_pipeline(sale_deed_paths, revenue_record_path, ec_path, client=None, retriever=None, survey_number_mapping=None):
        # Record paths so we can assert on cleanup timing after this returns.
        captured_paths["sale_deed_paths"] = list(sale_deed_paths)
        captured_paths["revenue_record_path"] = revenue_record_path
        captured_paths["ec_path"] = ec_path
        # Inputs must still exist while the pipeline is "running".
        for p in sale_deed_paths:
            assert os.path.isfile(p)
        if revenue_record_path:
            assert os.path.isfile(revenue_record_path)
        return _FAKE_OPINION, [Flag(issue="test issue", severity="low", detail="test detail")]

    monkeypatch.setattr(jobs_module, "run_pipeline", fake_run_pipeline)

    resp = client.post("/opinions", files=_pdf_files(n_sale_deeds=2, revenue_record=True, ec=True))
    assert resp.status_code == 202
    body = resp.json()
    job_id = body["job_id"]
    assert body["status"] in ("pending", "running")

    status = _wait_for_status(job_id, {"done", "failed"})
    assert status["status"] == "done"
    assert status["error"] is None
    assert status["sale_deed_count"] == 2
    assert status["has_revenue_record"] is True
    assert status["has_ec"] is True

    # Input files must be gone immediately once the pipeline has consumed them.
    for p in captured_paths["sale_deed_paths"]:
        assert not os.path.isfile(p)
    assert not os.path.isfile(captured_paths["revenue_record_path"])

    dl = client.get(f"/opinions/{job_id}/download")
    assert dl.status_code == 200
    assert dl.headers["content-type"] == "application/pdf"
    assert dl.content.startswith(b"%PDF")


def test_download_before_done_returns_409(monkeypatch):
    started = {"go": False}

    def slow_pipeline(*args, **kwargs):
        while not started["go"]:
            time.sleep(0.02)
        return _FAKE_OPINION, []

    monkeypatch.setattr(jobs_module, "run_pipeline", slow_pipeline)

    resp = client.post("/opinions", files=_pdf_files())
    job_id = resp.json()["job_id"]

    # Give the worker thread a moment to pick the job up and move it to RUNNING,
    # but it's blocked on `started["go"]` so it can't finish yet.
    time.sleep(0.1)
    status = client.get(f"/opinions/{job_id}").json()
    assert status["status"] in ("pending", "running")

    dl = client.get(f"/opinions/{job_id}/download")
    assert dl.status_code == 409

    started["go"] = True
    _wait_for_status(job_id, {"done"})


# --- failure path ------------------------------------------------------------

def test_pipeline_failure_recorded_as_failed_job(monkeypatch):
    def failing_pipeline(*args, **kwargs):
        raise RuntimeError("LLM_API_BASE_URL is not set")

    monkeypatch.setattr(jobs_module, "run_pipeline", failing_pipeline)

    resp = client.post("/opinions", files=_pdf_files())
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    status = _wait_for_status(job_id, {"done", "failed"})
    assert status["status"] == "failed"
    assert "LLM_API_BASE_URL" in status["error"]

    dl = client.get(f"/opinions/{job_id}/download")
    assert dl.status_code == 409

    # Server must not have crashed / the app must still serve other requests.
    assert client.get("/health").status_code == 200


def test_pdf_export_failure_also_recorded_as_failed(monkeypatch):
    def ok_pipeline(*args, **kwargs):
        return _FAKE_OPINION, []

    def failing_export(opinion, output_path):
        raise OSError("disk full (simulated)")

    monkeypatch.setattr(jobs_module, "run_pipeline", ok_pipeline)
    monkeypatch.setattr(jobs_module, "export_opinion_pdf", failing_export)

    resp = client.post("/opinions", files=_pdf_files())
    job_id = resp.json()["job_id"]

    status = _wait_for_status(job_id, {"done", "failed"})
    assert status["status"] == "failed"
    assert "disk full" in status["error"]


# --- upload validation --------------------------------------------------------

def test_rejects_non_pdf_content():
    files = [("sale_deed", ("deed.pdf", b"not actually a pdf", "application/pdf"))]
    resp = client.post("/opinions", files=files)
    assert resp.status_code == 400
    assert "PDF" in resp.json()["detail"]


def test_rejects_wrong_extension():
    files = [("sale_deed", ("deed.txt", _MINIMAL_PDF, "text/plain"))]
    resp = client.post("/opinions", files=files)
    assert resp.status_code == 400


def test_rejects_empty_file():
    files = [("sale_deed", ("deed.pdf", b"", "application/pdf"))]
    resp = client.post("/opinions", files=files)
    assert resp.status_code == 400


def test_requires_at_least_one_sale_deed():
    # No `sale_deed` field at all -- FastAPI's own required-field validation
    # (422) should reject this before it ever reaches job creation.
    resp = client.post("/opinions", files=[])
    assert resp.status_code == 422


def test_optional_documents_can_be_omitted(monkeypatch):
    def fake_run_pipeline(sale_deed_paths, revenue_record_path, ec_path, client=None, retriever=None, survey_number_mapping=None):
        assert revenue_record_path is None
        assert ec_path is None
        return _FAKE_OPINION, []

    monkeypatch.setattr(jobs_module, "run_pipeline", fake_run_pipeline)

    resp = client.post("/opinions", files=_pdf_files(n_sale_deeds=1))
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    status = _wait_for_status(job_id, {"done", "failed"})
    assert status["status"] == "done"
    assert status["has_revenue_record"] is False
    assert status["has_ec"] is False


# --- unknown job ---------------------------------------------------------------

def test_unknown_job_id_returns_404():
    assert client.get("/opinions/does-not-exist").status_code == 404
    assert client.get("/opinions/does-not-exist/download").status_code == 404


# --- no content leakage ---------------------------------------------------------

def test_status_response_never_contains_uploaded_bytes(monkeypatch):
    secret_marker = b"SUPER-SECRET-DOCUMENT-CONTENT-MARKER"

    def fake_run_pipeline(sale_deed_paths, revenue_record_path, ec_path, client=None, retriever=None, survey_number_mapping=None):
        return _FAKE_OPINION, []

    monkeypatch.setattr(jobs_module, "run_pipeline", fake_run_pipeline)

    files = [("sale_deed", ("deed.pdf", b"%PDF-1.4\n" + secret_marker + b"\n%%EOF", "application/pdf"))]
    resp = client.post("/opinions", files=files)
    job_id = resp.json()["job_id"]

    status = _wait_for_status(job_id, {"done", "failed"})
    assert secret_marker.decode() not in str(status)
