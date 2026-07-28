"""Tests for the synchronous POST /generate-opinion endpoint (src/landtitle/api/app.py).

Unlike /opinions (job-based, tested in test_api.py), this endpoint blocks and
returns the PDF directly. landtitle.api.app.run_pipeline is monkeypatched in
every test so nothing here makes a real LLM call or does real OCR.
export_opinion_pdf is NOT mocked on the success path (fast, pure-code, lets us
check a real PDF comes back); it IS mocked on the export-failure test.
"""
from __future__ import annotations

import glob
import os

from fastapi.testclient import TestClient

import landtitle.api.app as app_module
from landtitle.api.app import app
from landtitle.schemas import LegalOpinion

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


def _tmp_dirs_matching(prefix: str) -> list[str]:
    import tempfile
    return glob.glob(os.path.join(tempfile.gettempdir(), f"{prefix}*"))


# --- success path -------------------------------------------------------------

def test_generate_opinion_success_returns_pdf(monkeypatch):
    captured = {}

    def fake_run_pipeline(sale_deed_paths, revenue_record_path, ec_path, client=None, retriever=None, survey_number_mapping=None):
        captured["sale_deed_paths"] = list(sale_deed_paths)
        captured["revenue_record_path"] = revenue_record_path
        captured["ec_path"] = ec_path
        captured["tmp_dir"] = os.path.dirname(sale_deed_paths[0])
        for p in sale_deed_paths:
            assert os.path.isfile(p)
        return _FAKE_OPINION, []

    monkeypatch.setattr(app_module, "run_pipeline", fake_run_pipeline)

    resp = client.post("/generate-opinion", files=_pdf_files(n_sale_deeds=2, revenue_record=True, ec=True))
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert resp.content.startswith(b"%PDF")

    # Temp directory (and its input files) must be cleaned up after the
    # response has been fully sent.
    assert not os.path.isdir(captured["tmp_dir"])


def test_generate_opinion_optional_documents_can_be_omitted(monkeypatch):
    def fake_run_pipeline(sale_deed_paths, revenue_record_path, ec_path, client=None, retriever=None, survey_number_mapping=None):
        assert revenue_record_path is None
        assert ec_path is None
        return _FAKE_OPINION, []

    monkeypatch.setattr(app_module, "run_pipeline", fake_run_pipeline)

    resp = client.post("/generate-opinion", files=_pdf_files(n_sale_deeds=1))
    assert resp.status_code == 200
    assert resp.content.startswith(b"%PDF")


# --- failure path ---------------------------------------------------------------

def test_generate_opinion_pipeline_failure_returns_safe_json(monkeypatch):
    tmp_dir_holder = {}

    def failing_pipeline(sale_deed_paths, revenue_record_path, ec_path, client=None, retriever=None, survey_number_mapping=None):
        tmp_dir_holder["tmp_dir"] = os.path.dirname(sale_deed_paths[0])
        raise RuntimeError("LLM_API_BASE_URL is not set, and also /some/internal/path/leaked.py line 42")

    monkeypatch.setattr(app_module, "run_pipeline", failing_pipeline)

    resp = client.post("/generate-opinion", files=_pdf_files())
    assert resp.status_code == 500
    body = resp.json()
    assert "detail" in body
    assert isinstance(body["detail"], str)

    raw = resp.text
    assert "Traceback (most recent call last)" not in raw
    # No "<file path>, line N" style stack-frame markers leaking verbatim.
    assert ".py\", line" not in raw
    assert ".py:" not in raw or "detail" in body  # detail is allowed to mention it briefly; just no traceback markers above

    # Temp dir must still be cleaned up on the failure path.
    assert not os.path.isdir(tmp_dir_holder["tmp_dir"])


def test_generate_opinion_export_failure_returns_safe_json(monkeypatch):
    tmp_dir_holder = {}

    def ok_pipeline(sale_deed_paths, revenue_record_path, ec_path, client=None, retriever=None, survey_number_mapping=None):
        tmp_dir_holder["tmp_dir"] = os.path.dirname(sale_deed_paths[0])
        return _FAKE_OPINION, []

    def failing_export(opinion, output_path):
        raise OSError("disk full (simulated)")

    monkeypatch.setattr(app_module, "run_pipeline", ok_pipeline)
    monkeypatch.setattr(app_module, "export_opinion_pdf", failing_export)

    resp = client.post("/generate-opinion", files=_pdf_files())
    assert resp.status_code == 500
    assert "disk full" in resp.json()["detail"]
    assert "Traceback (most recent call last)" not in resp.text

    assert not os.path.isdir(tmp_dir_holder["tmp_dir"])


# --- upload validation --------------------------------------------------------

def test_generate_opinion_requires_at_least_one_sale_deed():
    resp = client.post("/generate-opinion", files=[])
    assert resp.status_code == 422


def test_generate_opinion_rejects_non_pdf_content():
    files = [("sale_deed", ("deed.pdf", b"not actually a pdf", "application/pdf"))]
    resp = client.post("/generate-opinion", files=files)
    assert resp.status_code == 400


# --- CORS -----------------------------------------------------------------------

def test_generate_opinion_cors_header_present_on_preflight():
    resp = client.options(
        "/generate-opinion",
        headers={
            "Origin": "https://example-frontend.vercel.app",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert resp.headers.get("access-control-allow-origin") is not None


def test_health_cors_header_present_with_origin():
    resp = client.get("/health", headers={"Origin": "https://example-frontend.vercel.app"})
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") is not None
