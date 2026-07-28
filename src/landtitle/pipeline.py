"""End-to-end orchestration: intake -> extraction -> verification -> legal
grounding -> opinion generation.

Large-document handling: rather than truncating an OCR'd document to fit
memory (the fix used ad hoc in Kaggle testing, where a 15-page document was
trimmed to 5 pages to avoid OOM), this pipeline chunks extraction-layer text
via ocr.intake.chunk_text and merges the per-chunk structured results. This
is the "proper chunking/pagination strategy" called for in the build spec.
"""
from __future__ import annotations

import argparse
import re
from typing import TypeVar

from pydantic import BaseModel

from landtitle.config import CHUNK_OVERLAP_CHARS, MAX_CHARS_PER_EXTRACTION_CHUNK, MAX_CITATIONS_PER_OPINION
from landtitle.extraction.encumbrance import extract_encumbrance_certificate
from landtitle.extraction.revenue_record import extract_revenue_record
from landtitle.extraction.sale_deed import extract_sale_deed
from landtitle.legal.retrieval import LegalRetriever, RetrievedChunk
from landtitle.llm.client import QwenClient
from landtitle.ocr.intake import chunk_text, extract_document, full_text
from landtitle.opinion.generator import generate_opinion
from landtitle.opinion.pdf_export import export_opinion_pdf
from landtitle.schemas import EncumbranceCertificate, Flag, Party, RevenueRecord, SaleDeed
from landtitle.verification.checks import (
    active_encumbrance_check,
    boundary_match,
    extent_match,
    name_similarity_check,
    owner_chain_continuity_check,
    survey_number_match,
)

T = TypeVar("T", bound=BaseModel)

_HONORIFIC_RE = re.compile(r"^(sri|smt|kum|dr|mr|mrs|m/s)\.?\s*", re.IGNORECASE)
_RELATION_SPLIT_RE = re.compile(r",|\bS/O\.?\b|\bW/O\.?\b|\bD/O\.?\b", re.IGNORECASE)
# Below this, two Party.name strings are treated as unrelated rather than
# risking a false merge (see _merge_parties docstring for why this can't be
# the same NAME_SIMILARITY_THRESHOLD used for cross-document matching).
_GIVEN_NAME_MERGE_THRESHOLD = 70

# Confirmed live: a real Sale Deed's recital names each vendor once, then
# later refers back to them collectively by number ("Vendor No.1, 2 and
# Vendor No.3 through their GPA holder ... do hereby sell") when describing
# a joint act. Extraction sometimes captures that back-reference itself as a
# NEW Party rather than recognizing it as pointing at already-listed
# sellers — a "Vendee"-only entry with no actual name shows the same
# pattern. A name built entirely from role/number/filler words with no
# other content is dropped as a phantom party rather than kept as a
# fabricated additional person.
_PLACEHOLDER_PARTY_TOKENS = {
    "vendor", "vendors", "vendee", "vendees", "purchaser", "purchasers",
    "seller", "sellers", "buyer", "buyers", "executant", "executants",
    "claimant", "claimants", "party", "parties",
    "no", "and", "the", "said", "through", "their", "gpa", "holder", "aforesaid",
}


def _is_placeholder_party_name(name: str) -> bool:
    tokens = re.findall(r"[a-zA-Z]+", name.lower())
    return bool(tokens) and all(t in _PLACEHOLDER_PARTY_TOKENS for t in tokens)


def _given_name(name: str) -> str:
    """Extract the name segment before the S/o|W/o|D/o relation clause (or
    comma), stripped of honorifics. This is the part of an Indian party name
    that identifies the person themselves, as opposed to the relation clause
    which names a DIFFERENT person (their father/husband)."""
    head = _RELATION_SPLIT_RE.split(name, maxsplit=1)[0]
    return _HONORIFIC_RE.sub("", head.strip()).strip(" .,").lower()


def _merge_parties(parties: list[Party]) -> list[Party]:
    """Dedupe Party entries by fuzzy given-name match, not exact repr.

    Confirmed live: a real 15-page Sale Deed restates every party's name
    several times throughout the document (recitals, clauses, execution
    section). Chunking that document for extraction means each restatement
    gets independently re-extracted, and OCR/LLM variance produces slightly
    different strings for the SAME person each time (e.g. a relation tag
    flipped, a stray OCR character) — exact-repr dedup let 3+ garbled
    duplicates of the same 2 real people through.

    Matching on the FULL name (as verification/checks.py's
    name_similarity_check does for cross-document comparison) does not work
    here and was confirmed live to be actively unsafe: Indian naming puts a
    relative's name in the "S/o"/"W/o" clause, so two DIFFERENT real people
    in the same family (e.g. a father and son) share most of their name's
    tokens and can score as high on token_sort_ratio as genuine OCR noise on
    the SAME person does — there is no safe threshold on the full string.
    Comparing only the given-name segment (before that relation clause)
    resolved this: same-person OCR variants still scored 80-100, while
    real distinct father/son pairs scored under 10, on the actual data this
    bug was found on.

    When two entries match, keeps whichever has more fields filled in and
    backfills any still-missing field from the other, so a chunk that
    happened to catch `represented_by_gpa` isn't lost to a chunk that named
    the same person without it. Phantom parties built entirely from
    role/number words (e.g. "Vendor No.1, 2 and 3", a bare "Vendee") are
    dropped first — see _is_placeholder_party_name.
    """
    merged: list[Party] = []
    for party in parties:
        if _is_placeholder_party_name(party.name):
            continue
        candidate_given_name = _given_name(party.name)
        match_idx = None
        if candidate_given_name:
            match_idx = next(
                (
                    i
                    for i, kept in enumerate(merged)
                    if name_similarity_check(
                        candidate_given_name, _given_name(kept.name), threshold=_GIVEN_NAME_MERGE_THRESHOLD
                    )[0]
                ),
                None,
            )
        if match_idx is None:
            merged.append(party)
            continue

        kept = merged[match_idx]
        filled = lambda p: sum(1 for v in (p.relation, p.address, p.represented_by_gpa) if v)
        primary, secondary = (kept, party) if filled(kept) >= filled(party) else (party, kept)
        merged[match_idx] = Party(
            name=primary.name,
            relation=primary.relation or secondary.relation,
            address=primary.address or secondary.address,
            represented_by_gpa=primary.represented_by_gpa or secondary.represented_by_gpa,
        )
    return merged


def _merge_extractions(results: list[T]) -> T:
    """Merge structured results from multiple text chunks of the SAME
    document: for list fields, concatenate and dedupe (fuzzy-by-name for
    Party lists, exact string repr otherwise); for scalar fields, take the
    first non-empty value across chunks (earlier chunks are more likely to
    contain the document's opening/key facts)."""
    if len(results) == 1:
        return results[0]

    merged_data: dict = {}
    for field_name in type(results[0]).model_fields:
        values = [getattr(r, field_name) for r in results]
        if isinstance(values[0], list):
            all_items = [item for v in values for item in v]
            if all_items and isinstance(all_items[0], Party):
                merged_data[field_name] = _merge_parties(all_items)
            else:
                merged_list = []
                seen = set()
                for item in all_items:
                    key = item.model_dump_json() if isinstance(item, BaseModel) else repr(item)
                    if key not in seen:
                        seen.add(key)
                        merged_list.append(item)
                merged_data[field_name] = merged_list
        else:
            merged_data[field_name] = next((v for v in values if v), values[0])
    return type(results[0]).model_validate(merged_data)


def extract_sale_deed_document(pdf_path: str, client: QwenClient) -> SaleDeed:
    pages = extract_document(pdf_path)
    text = full_text(pages)
    chunks = chunk_text(text, MAX_CHARS_PER_EXTRACTION_CHUNK, CHUNK_OVERLAP_CHARS)
    results = [extract_sale_deed(chunk, client) for chunk in chunks]
    return _merge_extractions(results)


def extract_revenue_record_document(pdf_path: str, client: QwenClient) -> RevenueRecord:
    pages = extract_document(pdf_path)
    text = full_text(pages)
    chunks = chunk_text(text, MAX_CHARS_PER_EXTRACTION_CHUNK, CHUNK_OVERLAP_CHARS)
    results = [extract_revenue_record(chunk, client) for chunk in chunks]
    return _merge_extractions(results)


def extract_ec_document(pdf_path: str, client: QwenClient) -> EncumbranceCertificate:
    pages = extract_document(pdf_path)
    return extract_encumbrance_certificate(pdf_path, pages, client)


def run_verification(
    sale_deeds: list[SaleDeed],
    revenue_record: RevenueRecord | None,
    ec: EncumbranceCertificate | None,
    survey_number_mapping: dict[str, str] | None = None,
) -> list[Flag]:
    flags: list[Flag] = []

    if len(sale_deeds) >= 2:
        flags += owner_chain_continuity_check(sale_deeds)

    if sale_deeds and revenue_record is not None:
        latest_deed = sale_deeds[-1]
        flags += survey_number_match(latest_deed, revenue_record, mapping=survey_number_mapping)
        flags += extent_match(latest_deed, revenue_record)
        flags += boundary_match(latest_deed.boundaries, revenue_record.boundaries, "Sale Deed", "Revenue Record")

    if ec is not None:
        flags += active_encumbrance_check(ec)

    return flags


_BASE_LEGAL_QUERIES = [
    "requirements for a valid registered sale deed",
    "effect of an unreleased mortgage on transfer of property",
    "limitation period for challenging a registered sale deed",
]


def select_citations(
    flags: list[Flag],
    retriever: LegalRetriever,
    max_citations: int = MAX_CITATIONS_PER_OPINION,
) -> list[RetrievedChunk]:
    """Gather legal citations for the opinion prompt, capped to keep that
    prompt within the target GPU's memory budget (see config.py). Flag-
    specific queries run first so a case's actual risk flags always get a
    citation slot before the generic boilerplate queries do."""
    queries = [flag.issue for flag in flags] + list(_BASE_LEGAL_QUERIES)
    citations: list[RetrievedChunk] = []
    seen_labels = set()
    for query in queries:
        for chunk in retriever.hybrid_retrieve(query, k=3):
            if chunk.citation_label not in seen_labels:
                citations.append(chunk)
                seen_labels.add(chunk.citation_label)
    return citations[:max_citations]


def run_pipeline(
    sale_deed_paths: list[str],
    revenue_record_path: str | None,
    ec_path: str | None,
    client: QwenClient | None = None,
    retriever: LegalRetriever | None = None,
    survey_number_mapping: dict[str, str] | None = None,
):
    from landtitle.schemas import LegalOpinion

    client = client or QwenClient()
    retriever = retriever or LegalRetriever()

    sale_deeds = [extract_sale_deed_document(p, client) for p in sale_deed_paths]
    revenue_record = extract_revenue_record_document(revenue_record_path, client) if revenue_record_path else None
    ec = extract_ec_document(ec_path, client) if ec_path else None

    flags = run_verification(sale_deeds, revenue_record, ec, survey_number_mapping)

    citations = select_citations(flags, retriever)

    facts = {
        "sale_deeds": [d.model_dump() for d in sale_deeds],
        "revenue_record": revenue_record.model_dump() if revenue_record else None,
        "encumbrance_certificate": ec.model_dump() if ec else None,
    }

    opinion: LegalOpinion = generate_opinion(facts, flags, citations, client)
    return opinion, flags


def main() -> None:
    parser = argparse.ArgumentParser(description="Land Title Due Diligence Engine")
    parser.add_argument("--sale-deed", action="append", default=[], help="Path to a Sale Deed PDF (repeatable, chronological order)")
    parser.add_argument("--revenue-record", help="Path to a Pahani/Adangal PDF")
    parser.add_argument("--ec", help="Path to an Encumbrance Certificate PDF")
    parser.add_argument("--output", default="opinion_draft.pdf", help="Output PDF path")
    parser.add_argument("--reviewed-by", default=None, help="If set, finalizes the opinion under this reviewer name before export")
    args = parser.parse_args()

    opinion, flags = run_pipeline(args.sale_deed, args.revenue_record, args.ec)

    if args.reviewed_by:
        from landtitle.opinion.generator import finalize_opinion
        opinion = finalize_opinion(opinion, args.reviewed_by)

    export_opinion_pdf(opinion, args.output)
    print(f"Wrote {args.output} (draft={opinion.is_draft}); {len(flags)} verification flag(s) found.")


if __name__ == "__main__":
    main()
