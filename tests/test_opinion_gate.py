import pytest

from landtitle.opinion.generator import finalize_opinion
from landtitle.opinion.pdf_export import export_opinion_pdf
from landtitle.schemas import LegalOpinion


def test_new_opinion_defaults_to_draft():
    opinion = LegalOpinion(overall_recommendation="Clear Title")
    assert opinion.is_draft is True
    assert opinion.reviewed_by is None


def test_finalize_requires_non_empty_reviewer():
    opinion = LegalOpinion(overall_recommendation="Clear Title")
    with pytest.raises(ValueError):
        finalize_opinion(opinion, "")
    with pytest.raises(ValueError):
        finalize_opinion(opinion, "   ")


def test_finalize_clears_draft_flag():
    opinion = LegalOpinion(overall_recommendation="Clear Title")
    finalized = finalize_opinion(opinion, "Adv. K. Rao")
    assert finalized.is_draft is False
    assert finalized.reviewed_by == "Adv. K. Rao"
    # original object must be untouched (finalize returns a copy)
    assert opinion.is_draft is True


def test_export_draft_pdf_contains_watermark(tmp_path):
    opinion = LegalOpinion(
        property_summary="Test property.",
        overall_recommendation="Title with Minor Issues",
    )
    output = tmp_path / "draft.pdf"
    export_opinion_pdf(opinion, str(output))
    assert output.exists()
    assert output.stat().st_size > 0


def test_export_finalized_pdf_shows_reviewer(tmp_path):
    opinion = finalize_opinion(LegalOpinion(overall_recommendation="Clear Title"), "Adv. K. Rao")
    output = tmp_path / "final.pdf"
    export_opinion_pdf(opinion, str(output))
    assert output.exists()
    assert output.stat().st_size > 0


def test_export_pdf_survives_smart_punctuation(tmp_path):
    # Confirmed live: real LLM output uses curly quotes/dashes (e.g. a right
    # single quote in "GPA's") which crash fpdf2's latin-1-only core font
    # unless sanitized first.
    opinion = LegalOpinion(
        property_summary="Free of GPA’s agreements — per the “title” search…",
        overall_recommendation="Clear Title",
    )
    output = tmp_path / "smart_punct.pdf"
    export_opinion_pdf(opinion, str(output))
    assert output.exists()
    assert output.stat().st_size > 0
