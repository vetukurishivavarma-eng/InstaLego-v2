"""Cross-document verification — deliberately PURE CODE, no LLM.

Confirmed failure mode this avoids: when asked to independently identify
"risk flags" without strict rules, the LLM repeatedly treated NORMAL
chain-of-title events (different owners at different times, different
transaction dates) as false "inconsistencies" — e.g. flagging that two
co-buyers received different documents on the same date, or flagging the
very fact of an ownership change as needing "verification." A normal chain
of title always has different parties at different points — that is not a
conflict.

Real conflicts are narrow and specific:
- the same document number showing different details across two sources
- a seller in a later document who never appears as a buyer in an earlier one
- an encumbrance present in one document but silently absent in another for
  the same period
- a genuine survey/plot number mismatch for what should be the same property

Every function here implements one of those narrow, deterministic checks.
If an LLM is ever used downstream for explanation, it must be fed the flags
this module already found — never allowed to invent flags of its own.
"""
from __future__ import annotations

import re
from datetime import datetime

from rapidfuzz import fuzz

from landtitle.config import NAME_SIMILARITY_THRESHOLD
from landtitle.schemas import Boundaries, EncumbranceCertificate, Flag, RevenueRecord, SaleDeed

# --- Unit normalization for extent comparisons ------------------------------

_SQFT_PER_UNIT = {
    "sq ft": 1.0, "sq feet": 1.0, "sqft": 1.0, "square feet": 1.0,
    "sq yard": 9.0, "sq yards": 9.0, "sqyd": 9.0, "square yards": 9.0,
    "sq meter": 10.7639, "sq meters": 10.7639, "sq metre": 10.7639, "sqm": 10.7639, "square meters": 10.7639,
    "acre": 43560.0, "acres": 43560.0,
    "cent": 435.6, "cents": 435.6,
    "gunta": 1089.0, "guntas": 1089.0,
    "are": 1076.39, "ares": 1076.39,
}
_EXTENT_TOLERANCE_RATIO = 0.02  # 2% tolerance for rounding/transliteration noise


def _parse_extent_to_sqft(extent: str | None) -> float | None:
    if not extent:
        return None
    match = re.search(r"([\d,]+\.?\d*)\s*([a-zA-Z. ]+)", extent.strip())
    if not match:
        return None
    number = float(match.group(1).replace(",", ""))
    unit = match.group(2).strip().lower().rstrip(".")
    factor = _SQFT_PER_UNIT.get(unit)
    if factor is None:
        return None
    return number * factor


# --- Name similarity ---------------------------------------------------------


def name_similarity_check(name1: str | None, name2: str | None, threshold: int = NAME_SIMILARITY_THRESHOLD) -> tuple[bool, float]:
    """Returns (is_match, score). Uses fuzzy matching to tolerate minor
    spelling/transliteration variance between documents — this is expected
    and normal, not itself proof of a real conflict."""
    if not name1 or not name2:
        return False, 0.0
    score = fuzz.token_sort_ratio(name1.strip().lower(), name2.strip().lower())
    return score >= threshold, score


# --- Individual checks --------------------------------------------------------


def survey_number_match(sale_deed: SaleDeed, revenue_record: RevenueRecord, mapping: dict[str, str] | None = None) -> list[Flag]:
    """Sale Deeds use municipal numbering (e.g. 'Plot No.4, Municipal No.
    6-3-862/3'); Pahani uses revenue survey numbering (e.g. '82/A/4'). These
    are different systems and will never match directly — do not compare
    them for equality. This check only confirms both are present, and (if a
    correspondence mapping is supplied) that the mapping holds."""
    flags: list[Flag] = []
    deed_no, revenue_no = sale_deed.survey_number, revenue_record.survey_number

    if not deed_no or not revenue_no:
        flags.append(Flag(
            issue="Survey number cross-reference incomplete",
            severity="low",
            detail=f"Sale Deed survey/plot number: {deed_no or 'missing'}; Pahani survey number: {revenue_no or 'missing'}. "
                   f"Cannot cross-verify without both present.",
            documents_compared=["Sale Deed", "Revenue Record"],
        ))
        return flags

    if mapping is not None:
        expected = mapping.get(deed_no)
        if expected is not None and expected != revenue_no:
            flags.append(Flag(
                issue="Survey number mapping mismatch",
                severity="high",
                detail=f"Sale Deed number '{deed_no}' is expected to map to revenue survey number "
                       f"'{expected}' per the supplied reference, but the Pahani shows '{revenue_no}'.",
                documents_compared=["Sale Deed", "Revenue Record"],
            ))
    return flags


def extent_match(sale_deed: SaleDeed, revenue_record: RevenueRecord) -> list[Flag]:
    deed_sqft = _parse_extent_to_sqft(sale_deed.land_extent)
    revenue_sqft = _parse_extent_to_sqft(revenue_record.land_extent)

    if deed_sqft is None or revenue_sqft is None:
        return [Flag(
            issue="Land extent could not be cross-verified",
            severity="low",
            detail=f"Sale Deed land_extent: {sale_deed.land_extent!r}; Pahani land_extent: {revenue_record.land_extent!r}. "
                   f"Could not parse one or both into a comparable unit.",
            documents_compared=["Sale Deed", "Revenue Record"],
        )]

    diff_ratio = abs(deed_sqft - revenue_sqft) / max(deed_sqft, revenue_sqft)
    if diff_ratio > _EXTENT_TOLERANCE_RATIO:
        return [Flag(
            issue="Land extent mismatch between Sale Deed and Revenue Record",
            severity="medium",
            detail=f"Sale Deed: {sale_deed.land_extent} (~{deed_sqft:.1f} sq ft); "
                   f"Pahani: {revenue_record.land_extent} (~{revenue_sqft:.1f} sq ft); "
                   f"difference {diff_ratio:.1%} exceeds {_EXTENT_TOLERANCE_RATIO:.0%} tolerance.",
            documents_compared=["Sale Deed", "Revenue Record"],
        )]
    return []


def boundary_match(boundaries1: Boundaries, boundaries2: Boundaries, doc1_label: str, doc2_label: str) -> list[Flag]:
    flags = []
    for direction, val1, val2 in (
        ("North", boundaries1.north, boundaries2.north),
        ("South", boundaries1.south, boundaries2.south),
        ("East", boundaries1.east, boundaries2.east),
        ("West", boundaries1.west, boundaries2.west),
    ):
        if not val1 or not val2:
            continue  # missing data is not itself a conflict
        is_match, score = name_similarity_check(val1, val2)
        if not is_match:
            flags.append(Flag(
                issue=f"{direction} boundary mismatch",
                severity="medium",
                detail=f"{doc1_label}: '{val1}' vs {doc2_label}: '{val2}' (similarity {score:.0f}%).",
                documents_compared=[doc1_label, doc2_label],
            ))
    return flags


_MORTGAGE_RELEASE_KEYWORDS = ["release", "discharge", "satisfaction of mortgage", "reconveyance", "cancellation of mortgage"]


def active_encumbrance_check(ec: EncumbranceCertificate) -> list[Flag]:
    """Flag any unreleased mortgage or pending litigation entry. A mortgage
    entry is considered released only if a later entry in the same EC
    explicitly references release/discharge — the mere presence of a
    mortgage entry decades ago, on its own, is not itself the flag; an
    unreleased one is."""
    flags: list[Flag] = []

    def _sort_key(entry):
        for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(entry.date, fmt)
            except (ValueError, TypeError):
                continue
        return datetime.min

    entries_sorted = sorted(ec.entries, key=_sort_key)
    open_mortgages = []
    for entry in entries_sorted:
        nature = (entry.nature_of_document or "").lower()
        if any(kw in nature for kw in _MORTGAGE_RELEASE_KEYWORDS):
            if open_mortgages:
                open_mortgages.pop(0)  # earliest open mortgage is released
            continue
        if any(kw in nature for kw in ("mortgage", "hypothecation", "charge", "usufructuary")):
            open_mortgages.append(entry)

    for entry in open_mortgages:
        flags.append(Flag(
            issue="Unreleased mortgage entry in Encumbrance Certificate",
            severity="high",
            detail=f"Document No. {entry.document_number or 'unknown'} dated {entry.date or 'unknown'} "
                   f"({entry.nature_of_document}) has no later release/discharge entry in the same EC.",
            documents_compared=["Encumbrance Certificate"],
        ))

    litigation_keywords = ["attachment", "lis pendens", "court order", "injunction", "decree", "suit", "prohibitory order"]
    for entry in entries_sorted:
        nature = (entry.nature_of_document or "").lower()
        if any(kw in nature for kw in litigation_keywords):
            flags.append(Flag(
                issue="Litigation/attachment entry found in Encumbrance Certificate",
                severity="high",
                detail=f"Document No. {entry.document_number or 'unknown'} dated {entry.date or 'unknown'}: {entry.nature_of_document}.",
                documents_compared=["Encumbrance Certificate"],
            ))

    return flags


def owner_chain_continuity_check(sale_deeds: list[SaleDeed]) -> list[Flag]:
    """Verify each deed's buyer(s) become the next deed's seller(s), in
    chronological order. Deeds must be pre-sorted or will be sorted here by
    registration_date; a gap or unexplained jump is flagged, but the mere
    fact that parties differ between deeds is expected and not itself a flag."""
    flags: list[Flag] = []
    if len(sale_deeds) < 2:
        return flags

    def _sort_key(deed: SaleDeed):
        for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(deed.registration_date, fmt)
            except (ValueError, TypeError):
                continue
        return datetime.min

    deeds_sorted = sorted(sale_deeds, key=_sort_key)

    for i in range(len(deeds_sorted) - 1):
        current, nxt = deeds_sorted[i], deeds_sorted[i + 1]
        current_label = f"Sale Deed {current.document_number or i}"
        next_label = f"Sale Deed {nxt.document_number or i + 1}"

        for buyer in current.buyers:
            found = any(name_similarity_check(buyer.name, seller.name)[0] for seller in nxt.sellers)
            if not found:
                flags.append(Flag(
                    issue="Chain of title gap",
                    severity="high",
                    detail=f"Buyer '{buyer.name}' in {current_label} does not appear as a seller in {next_label}.",
                    documents_compared=[current_label, next_label],
                ))
    return flags
