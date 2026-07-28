"""Structural validation for the real, uploadable negative-case fixtures in
`fixtures/negative_cases/`.

Context: `scripts/trap_test.py` proves the verification layer works, but only
by mutating in-memory Python objects (e.g. `revenue_record.model_copy(...)`,
or appending a fake `ECEntry` in code) -- there was no fixture FILE that,
uploaded through the real pipeline (OCR/text-intake -> extraction ->
verification), independently triggers those same checks. These tests close
that gap for the parts of the pipeline that are pure code (no LLM/network
required):

- For the Revenue Record fixture: extraction is flat-text LLM extraction
  (see `extraction/revenue_record.py`) with no pure-code parsing step before
  the LLM call, so there is nothing structural to parse here. We can only
  confirm the fixture file exists, intake reads it via the `.txt`
  passthrough path (`ocr/intake.py::extract_document`), and it contains the
  deliberately-mismatched land_extent value. The LLM extraction itself is
  NOT exercised by this test -- that requires a live model endpoint and is
  a deliberate separate step (see fixtures/negative_cases/README.md).

- For the Encumbrance Certificate fixture: extraction goes through
  `extraction/ec_tables.py::parse_fixed_width_text_table()` +
  `flatten_table()` BEFORE any LLM involvement, so that path is fully
  testable here with no LLM/network needed. We confirm the fixture parses
  into the intended rows, with the mortgage entry's executant/claimant
  correctly separated (not scrambled), and that running the parsed rows
  through the real `active_encumbrance_check()` produces exactly the
  "Unreleased mortgage" high-severity flag with no spurious extras.
"""
from __future__ import annotations

from pathlib import Path

from landtitle.extraction.ec_tables import flatten_table, parse_fixed_width_text_table
from landtitle.ocr.intake import extract_document, full_text
from landtitle.schemas import ECEntry, EncumbranceCertificate
from landtitle.verification.checks import active_encumbrance_check

_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "negative_cases"
_REVENUE_RECORD_FIXTURE = _FIXTURES_DIR / "revenue_record_extent_mismatch.txt"
_EC_FIXTURE = _FIXTURES_DIR / "ec_unreleased_mortgage.txt"


# --- revenue_record_extent_mismatch.txt --------------------------------------
#
# Not pure-code testable beyond intake passthrough + presence of the
# deliberately-wrong value: extraction itself is an LLM call.


def test_revenue_record_fixture_exists_and_reads_via_txt_passthrough():
    assert _REVENUE_RECORD_FIXTURE.exists()
    pages = extract_document(str(_REVENUE_RECORD_FIXTURE))
    assert len(pages) == 1
    assert pages[0].source == "text"
    assert len(pages[0].text) > 0


def test_revenue_record_fixture_contains_deliberately_mismatched_extent():
    """The real cached Sale Deed's land_extent is 329.61 square yards. This
    fixture's Pahani land_extent (150.00 Sq. Yards) differs by ~54%, far
    exceeding checks.py::_EXTENT_TOLERANCE_RATIO (2%), and is meant to
    trigger `extent_match()`'s "Land extent mismatch between Sale Deed and
    Revenue Record" medium-severity flag once run through real extraction."""
    text = full_text(extract_document(str(_REVENUE_RECORD_FIXTURE)))
    assert "150.00 Sq. Yards" in text
    assert "329.61" not in text  # must not accidentally match the real deed's extent


# --- ec_unreleased_mortgage.txt ----------------------------------------------
#
# EC extraction is pure-code table parsing before any LLM involvement, so
# this fixture is fully structurally testable end-to-end.


def _parse_fixture_entries() -> list[ECEntry]:
    text = full_text(extract_document(str(_EC_FIXTURE)))
    rows = parse_fixed_width_text_table(text)
    labeled = flatten_table(rows)
    return [ECEntry(**lr.fields) for lr in labeled]


def test_ec_fixture_exists_and_reads_via_txt_passthrough():
    assert _EC_FIXTURE.exists()
    pages = extract_document(str(_EC_FIXTURE))
    assert len(pages) == 1
    assert pages[0].source == "text"


def test_ec_fixture_parses_four_real_rows_and_drops_nil_placeholder():
    text = full_text(extract_document(str(_EC_FIXTURE)))
    rows = parse_fixed_width_text_table(text)
    assert len(rows) == 4
    doc_numbers = [r[1] for r in rows]  # canonical order: sl_no, document_number, ...
    assert doc_numbers == ["3311/2001", "3312/2001", "812/2013", "550/2015"]


def test_ec_fixture_mortgage_row_parties_correctly_separated_not_scrambled():
    """The mortgage row's executant (mortgagor, the then-owner from the
    prior GPA sale deed) and claimant (mortgagee, the bank) must land in
    the correct fields -- this is exactly the confirmed bug class
    ec_tables.py exists to prevent (see its module docstring)."""
    entries = _parse_fixture_entries()
    mortgage_entries = [e for e in entries if e.document_number == "550/2015"]
    assert len(mortgage_entries) == 1
    mortgage = mortgage_entries[0]
    assert mortgage.nature_of_document == "Simple Mortgage"
    assert "Ananya Devi Reddy" in mortgage.executants_sellers
    assert "Cooperative Bank" in mortgage.claimants_buyers
    # No cross-contamination between the two party fields.
    assert "Cooperative Bank" not in mortgage.executants_sellers
    assert "Ananya Devi Reddy" not in mortgage.claimants_buyers


def test_ec_fixture_gpa_continuation_still_attributed_to_executants():
    """Row 3's GPA continuation lines (wrapping across two lines that don't
    line up with any header column) must land under Executants, not
    Claimants -- the same tricky case covered generically in
    test_ec_text_table.py, confirmed here against this specific fixture."""
    entries = _parse_fixture_entries()
    deed = next(e for e in entries if e.document_number == "812/2013")
    assert "GPA holder" in deed.executants_sellers
    assert deed.claimants_buyers.strip() == "Fictional Ananya Devi Reddy"


def test_ec_fixture_triggers_unreleased_mortgage_flag_via_real_check():
    """Runs the parsed fixture entries through the real, unmodified
    `active_encumbrance_check()` (verification/checks.py) end-to-end and
    confirms exactly the expected single high-severity flag -- no spurious
    litigation flag from the dropped NIL placeholder row, no flag for the
    three clean sale-deed entries."""
    entries = _parse_fixture_entries()
    ec = EncumbranceCertificate(entries=entries)
    flags = active_encumbrance_check(ec)
    assert len(flags) == 1
    assert flags[0].issue == "Unreleased mortgage entry in Encumbrance Certificate"
    assert flags[0].severity == "high"
    assert "550/2015" in flags[0].detail
