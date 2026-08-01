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
    """Verify each deed's seller(s) can be traced to a buyer in an earlier
    submitted deed. The mere fact that parties differ between deeds is
    expected and not itself a flag; only a genuine missing link is.

    Two resolution strategies, tried in order for each deed:
    1. If this deed's own `prior_title_deed_references` resolves (by
       document number, via `_matching_document_number`) to another
       submitted deed, that IS this deed's self-declared predecessor --
       compare THAT specific deed's buyer(s) against this deed's seller(s).
       This is the strongest signal: it checks what the deed itself claims,
       not an assumption about ordering.
    2. Otherwise, compare this deed's seller(s) against the buyer(s) of the
       WHOLE SET of submitted deeds with a strictly earlier registration_date
       (not one single "nearest" deed) -- if none match, flag a gap.

    Deliberately NOT based on sorting all deeds by registration_date and
    comparing date-adjacent PAIRS (the prior implementation). Confirmed live
    with this project's own synthetic 3-deed test set: Deed A and Deed B
    share the exact same registration_date (15-07-2012) -- a real, expected
    situation (same-day back-to-back registrations), not a data error. A
    tie on the sort key makes Python's stable sort fall back to whatever
    order the deeds happened to be passed in to this function, which has no
    logical relationship to the real chain. This produced a false-positive
    "Chain of title gap" pairing Deed B's buyer against Deed A's seller --
    the wrong comparison entirely -- purely because of input order, while
    the real A->B link (which both deeds' own recitals agree on) went
    unchecked. Strategy 1 above sidesteps the tie using the deeds' own
    evidence instead of guessing an order; strategy 2 sidesteps it by
    comparing against the whole earlier-dated SET rather than picking one
    deed as "the" predecessor, so a tie within that set (as long as it
    doesn't include the deed under test) is harmless.

    A deed with no captured `sellers` is skipped entirely (nothing to
    verify a link FROM) -- consistent with this module's "missing data is
    not itself a conflict" principle elsewhere (see e.g. `boundary_match`).
    A deed whose prior reference does NOT resolve to any submitted deed
    (including a near-miss, e.g. one digit off) falls through to strategy 2
    rather than being flagged directly here -- an unresolvable reference is
    `unevidenced_prior_reference_check`'s job, since without a resolved
    predecessor there is no specific comparison to make.

    Strategy 2's date key falls back to the YEAR embedded in the deed's own
    `document_number` (the standard Indian sub-registrar "NNNN/YYYY" format)
    when `registration_date` itself didn't parse. Confirmed live, with this
    project's own real 3-deed test set: a deed's text can state its
    execution date ("made and executed on this the Nth day of ...") without
    ever separately stating a distinct registration date for its OWN
    document number, and extraction correctly leaves `registration_date`
    null rather than guessing -- but that then made strategy 2 skip the
    deed entirely (no date key at all), silently losing a genuine,
    real chain-of-title break that strategy 2 exists to catch. The
    document number's year suffix is reliably present (it's how the
    document is legally identified) even when a separate date sentence
    is absent, so it's a safe fallback signal -- year-only granularity
    can't order two deeds within the SAME year, but that's fine here:
    same-year deeds are exactly the case strategy 1 (reference-based
    linking) is meant to resolve; strategy 2 only needs to place deeds
    from DIFFERENT years in the right relative order.

    Confirmed live: naively mixing an exact parsed date with a year-only
    fallback (defaulting the missing month/day to January 1st) is NOT safe
    to compare directly with `<` -- it made a same-year deed with only a
    year-only fallback key look "earlier" than another same-year deed that
    has a real, later-in-the-year exact date, purely because Jan 1st sorts
    before any other day in that year. `_date_key` therefore returns
    `(year, exact_datetime_or_None)`, and `_is_strictly_earlier` below only
    trusts day-level ordering when BOTH sides have an exact date; a
    year-only key is only ever compared at year granularity."""
    flags: list[Flag] = []
    if len(sale_deeds) < 2:
        return flags

    def _date_key(deed: SaleDeed) -> tuple[int, datetime | None] | None:
        for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d"):
            try:
                exact = datetime.strptime(deed.registration_date, fmt)
                return (exact.year, exact)
            except (ValueError, TypeError):
                continue
        if deed.document_number:
            match = re.search(r"/(\d{4})\b", deed.document_number)
            if match:
                return (int(match.group(1)), None)
        return None

    def _is_strictly_earlier(key_a: tuple[int, datetime | None], key_b: tuple[int, datetime | None]) -> bool:
        year_a, exact_a = key_a
        year_b, exact_b = key_b
        if year_a != year_b:
            return year_a < year_b
        if exact_a is not None and exact_b is not None:
            return exact_a < exact_b
        return False  # same year, at least one side is a year-only estimate -- can't say

    known_document_numbers = [d.document_number.strip() for d in sale_deeds if d.document_number]
    deeds_by_docnum = {d.document_number.strip(): d for d in sale_deeds if d.document_number}
    dated = [(d, _date_key(d)) for d in sale_deeds]

    for deed in sale_deeds:
        if not deed.sellers:
            continue  # nothing to verify a link from

        deed_label = f"Sale Deed {deed.document_number or '(document number not captured)'}"

        resolved_predecessor = None
        for reference in deed.prior_title_deed_references:
            if not reference or not reference.strip():
                continue
            docnum = _matching_document_number(reference, known_document_numbers)
            if docnum and deeds_by_docnum[docnum] is not deed:
                resolved_predecessor = deeds_by_docnum[docnum]
                break

        if resolved_predecessor is not None:
            predecessor_label = f"Sale Deed {resolved_predecessor.document_number}"
            found = any(
                name_similarity_check(buyer.name, seller.name)[0]
                for buyer in resolved_predecessor.buyers
                for seller in deed.sellers
            )
            if not found:
                buyer_names = ", ".join(b.name for b in resolved_predecessor.buyers) or "none captured"
                seller_names = ", ".join(s.name for s in deed.sellers) or "none captured"
                flags.append(Flag(
                    issue="Chain of title gap",
                    severity="high",
                    detail=(
                        f"{deed_label} recites {predecessor_label} as its predecessor, but "
                        f"{predecessor_label}'s buyer(s) ({buyer_names}) do not match {deed_label}'s "
                        f"seller(s) ({seller_names})."
                    ),
                    documents_compared=[predecessor_label, deed_label],
                ))
            continue

        my_key = _date_key(deed)
        if my_key is None:
            continue
        earlier_deeds = [d for d, k in dated if k is not None and _is_strictly_earlier(k, my_key)]
        if not earlier_deeds:
            continue

        found = any(
            name_similarity_check(buyer.name, seller.name)[0]
            for earlier in earlier_deeds
            for buyer in earlier.buyers
            for seller in deed.sellers
        )
        if not found:
            earlier_labels = ", ".join(f"Sale Deed {d.document_number or '(unlabeled)'}" for d in earlier_deeds)
            seller_names = ", ".join(s.name for s in deed.sellers) or "none captured"
            flags.append(Flag(
                issue="Chain of title gap",
                severity="high",
                detail=(
                    f"{deed_label}'s seller(s) ({seller_names}) do not match any buyer among the "
                    f"earlier submitted deed(s) by registration date ({earlier_labels}), and no "
                    f"prior-title reference on {deed_label} resolves to one of them either."
                ),
                documents_compared=[deed_label] + [
                    f"Sale Deed {d.document_number}" for d in earlier_deeds if d.document_number
                ],
            ))
    return flags


# --- Prior-title-reference evidencing ----------------------------------------

_DOC_NUMBER_RE = re.compile(r"\d+/\d+")


def _document_number_candidates(reference: str) -> list[str]:
    """A `prior_title_deed_references` entry is often a full descriptive
    sentence copied from a deed's recital (e.g. "Document No. 1123/1998,
    dated 02-06-1998, registered at the office of the Sub-Registrar..."),
    not a bare document number. Pull out the number/year-shaped substring
    (the standard sub-registrar document-number format) for matching; if
    none is found, fall back to the whole trimmed string so an
    unconventionally-formatted reference can still match via containment."""
    matches = _DOC_NUMBER_RE.findall(reference)
    return matches if matches else [reference.strip()]


def _matching_document_number(reference: str, known_document_numbers: list[str]) -> str | None:
    """Return the known document number `reference` resolves to (by exact or
    substring match on the number-shaped candidates `_document_number_
    candidates` extracts), or None. Factored out of `_reference_is_evidenced`
    so callers that need to know WHICH document matched -- not just whether
    one did -- can reuse the exact same matching rule (see
    `owner_chain_continuity_check`, which needs the actual deed object to
    compare parties against)."""
    candidates = _document_number_candidates(reference)
    for known in known_document_numbers:
        if not known:
            continue
        known_norm = known.strip()
        for candidate in candidates:
            if not candidate:
                continue
            if candidate == known_norm or candidate in known_norm or known_norm in candidate:
                return known_norm
    return None


def _reference_is_evidenced(reference: str, known_document_numbers: list[str]) -> bool:
    if _matching_document_number(reference, known_document_numbers) is not None:
        return True
    # Last-resort fuzzy compare on the full strings, for the case where
    # neither side yields a clean number-shaped substring to compare.
    return any(
        fuzz.token_sort_ratio(reference.lower(), known.strip().lower()) >= NAME_SIMILARITY_THRESHOLD
        for known in known_document_numbers if known
    )


def unevidenced_prior_reference_check(
    sale_deeds: list[SaleDeed], ec: EncumbranceCertificate | None
) -> list[Flag]:
    """A Sale Deed's own recital often narrates how its seller acquired
    title by naming an earlier document (`prior_title_deed_references`),
    e.g. "Document No. 1123/1998, dated 02-06-1998" describing how the
    1998 vendor themselves came to own the property. That recital, on its
    own, proves nothing — it is the current deed's self-report of a prior
    transaction, not independent evidence that the transaction happened as
    described or that title actually passed cleanly through it.

    This check flags a prior-title reference only when NOTHING actually
    submitted for this diligence corroborates it: neither another submitted
    Sale Deed's own `document_number`, nor an entry in the submitted
    Encumbrance Certificate. That is a real, common chain-of-title gap in
    Indian land-title practice (the referenced document was simply never
    part of this diligence bundle) — but it is deliberately flagged as
    "not independently evidenced," not as a confirmed defect: severity is
    kept low, since the honest read of this signal is "pull the referenced
    document and confirm," not "title is bad." A deed with no stated prior
    reference (empty list) is not flagged at all — that is either root/
    original title or simply a field the extraction didn't capture, not a
    chain gap.
    """
    flags: list[Flag] = []
    if not sale_deeds:
        return flags

    known_document_numbers = [d.document_number for d in sale_deeds if d.document_number]
    if ec is not None:
        known_document_numbers += [e.document_number for e in ec.entries if e.document_number]

    for deed in sale_deeds:
        deed_label = f"Sale Deed {deed.document_number or '(document number not captured)'}"
        for reference in deed.prior_title_deed_references:
            if not reference or not reference.strip():
                continue
            if _reference_is_evidenced(reference, known_document_numbers):
                continue
            documents_compared = [deed_label]
            if ec is not None:
                documents_compared.append("Encumbrance Certificate")
            flags.append(Flag(
                issue="Prior title reference not independently evidenced",
                severity="low",
                detail=(
                    f"{deed_label} recites a prior title reference ('{reference.strip()}') that does "
                    f"not match the document_number of any other submitted Sale Deed or Encumbrance "
                    f"Certificate entry. This does not mean the referenced transaction is invalid -- "
                    f"only that it is not corroborated by the documents actually submitted for this "
                    f"diligence; the underlying document should be obtained and reviewed to close the "
                    f"chain of title."
                ),
                documents_compared=documents_compared,
            ))
    return flags
