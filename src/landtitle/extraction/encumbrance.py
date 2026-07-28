"""Encumbrance Certificate extraction — orchestrates table-aware pre-processing
(ec_tables.py) before ever calling the LLM. See ec_tables.py module docstring
for the confirmed bug this avoids.
"""
from __future__ import annotations

from pathlib import Path

from landtitle.extraction.ec_tables import (
    LabeledRow,
    extract_tables_pdfplumber,
    extract_tables_scanned,
    flatten_table,
    parse_fixed_width_text_table,
)
from landtitle.llm.client import QwenClient
from landtitle.ocr.intake import PageText
from landtitle.schemas import ECEntry, ECEntryBatch, EncumbranceCertificate

ROWS_PER_LLM_BATCH = 8

_ROW_CLEANUP_SYSTEM_PROMPT = """You are cleaning up already-structured rows from an \
Encumbrance Certificate table. Each row below has been pre-labeled by field (Document \
Number, Date, Nature of Document, Executants (Sellers), Claimants (Buyers), Consideration) \
using the table's actual column structure — the labels are already correct and MUST NOT be \
swapped or reinterpreted. Your only job is to:
- Copy each labeled value into the matching JSON field.
- Trim OCR noise/line-wrap artifacts within a value (e.g. join a name split across two lines).
- Never move a value from "Executants (Sellers)" into "Claimants (Buyers)" or vice versa.
- Never introduce a person, date, or fact not present in the labeled row.
- If a row is missing a field, leave it null.
"""

_METADATA_SYSTEM_PROMPT = """You are extracting document-level metadata (not table rows) \
from the header/footer text of an Encumbrance Certificate. Rules:
- Never introduce a fact not explicitly present in the text.
- If a field is not present, leave it null.
"""

_MORTGAGE_KEYWORDS = [
    "mortgage", "hypothecation", "equitable mortgage", "simple mortgage",
    "usufructuary", "charge", "deposit of title deeds",
]
_LITIGATION_KEYWORDS = [
    "attachment", "lis pendens", "court order", "injunction", "decree",
    "suit", "litigation", "prohibitory order",
]


def _is_scanned(pdf_path: str) -> bool:
    from landtitle.ocr.intake import extract_pages_pypdf, MIN_CHARS_FOR_TEXT_PAGE

    pages = extract_pages_pypdf(pdf_path)
    if not pages:
        return True
    text_pages = sum(1 for p in pages if len(p.text) >= MIN_CHARS_FOR_TEXT_PAGE)
    return text_pages < len(pages) / 2


def _extract_labeled_rows(path: str) -> list[LabeledRow]:
    if path.lower().endswith(".txt"):
        text = Path(path).read_text(encoding="utf-8")
        rows = parse_fixed_width_text_table(text)
        return flatten_table(rows) if rows else []

    tables = extract_tables_scanned(path) if _is_scanned(path) else extract_tables_pdfplumber(path)
    labeled_rows: list[LabeledRow] = []
    for table in tables:
        labeled_rows.extend(flatten_table(table))
    return labeled_rows


def _batch(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def _clean_entries_via_llm(labeled_rows: list[LabeledRow], client: QwenClient) -> list[ECEntry]:
    entries: list[ECEntry] = []
    for batch in _batch(labeled_rows, ROWS_PER_LLM_BATCH):
        rows_text = "\n\n".join(f"Row {r.row_index + 1}:\n{r.to_labeled_text()}" for r in batch)
        user_prompt = (
            f"Convert these {len(batch)} pre-labeled EC rows into a JSON list of entries, "
            f"one per row, in the same order:\n\n{rows_text}"
        )
        result = client.extract_structured(_ROW_CLEANUP_SYSTEM_PROMPT, user_prompt, ECEntryBatch)
        entries.extend(result.entries)
    return entries


def _non_table_text(pages: list[PageText]) -> str:
    # Metadata (certificate number, issuing office, property description,
    # search period) lives in prose header/footer text, not table rows.
    return "\n\n".join(p.text for p in pages if p.text)[:4000]


def _compute_flags(entries: list[ECEntry]) -> tuple[bool, bool]:
    """Deterministic keyword match — never delegated to the LLM, since
    whether a specific nature-of-document string names a mortgage or
    litigation entry is an objective pattern match, not a judgment call."""
    natures = " ".join((e.nature_of_document or "").lower() for e in entries)
    mortgage_found = any(kw in natures for kw in _MORTGAGE_KEYWORDS)
    litigation_found = any(kw in natures for kw in _LITIGATION_KEYWORDS)
    return mortgage_found, litigation_found


def extract_encumbrance_certificate(pdf_path: str, pages: list[PageText], client: QwenClient) -> EncumbranceCertificate:
    labeled_rows = _extract_labeled_rows(pdf_path)
    entries = _clean_entries_via_llm(labeled_rows, client) if labeled_rows else []

    metadata_prompt = (
        f"Extract certificate_number, issuing_office, property_description, and search_period "
        f"from this Encumbrance Certificate header/footer text:\n\n{_non_table_text(pages)}"
    )
    metadata = client.extract_structured(_METADATA_SYSTEM_PROMPT, metadata_prompt, EncumbranceCertificate)

    mortgage_found, litigation_found = _compute_flags(entries)

    metadata.entries = entries
    metadata.mortgage_entries_found = mortgage_found
    metadata.litigation_entries_found = litigation_found
    return metadata
