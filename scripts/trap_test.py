"""Trap test: verify Layer 3 (verification/checks.py) genuinely catches
deliberately-injected inconsistencies when run against REAL extracted data
(from scripts/_debug_opinion_cache.pkl), not just clean synthetic objects
like tests/test_verification.py uses.

This does NOT call the LLM (verification is pure code) and does NOT need a
second real Sale Deed -- it takes the one real cached extraction we have and
deliberately corrupts specific fields to inject known-answer inconsistencies,
then confirms run_verification() reports them, while unrelated checks stay
quiet (no false positives from the injection itself).

Still NOT a substitute for the real multi-deed end-to-end pipeline run
(blocked on sourcing Document No. 2609/2000 or 2610/2000) -- that exercises
owner_chain_continuity_check() with two genuinely separate real documents.
This script instead builds a *synthetic second deed* using the real buyer
name from the cached deed (including its real formatting/honorifics), to at
least test the fuzzy-name-matching logic against real-shaped names rather
than clean placeholders like "Ramesh Kumar".
"""
from __future__ import annotations

import copy
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from landtitle.schemas import ECEntry, EncumbranceCertificate, Party, RevenueRecord, SaleDeed
from landtitle.verification.checks import (
    active_encumbrance_check,
    boundary_match,
    extent_match,
    owner_chain_continuity_check,
    survey_number_match,
)
from landtitle.pipeline import _merge_parties, run_verification

CACHE_PATH = Path(__file__).resolve().parent / "_debug_opinion_cache.pkl"


def load_real_objects():
    """Load the cached real extraction and re-apply the CURRENT
    _merge_parties() to sellers/buyers before use. The cache predates the
    party-merge/placeholder-drop fix (built 2026-07-27 23:43 IST, fix landed
    ~00:33 IST the next day), so its stored buyers/sellers still show the
    pre-fix un-merged state (e.g. a bare "Vendee" placeholder alongside 2
    OCR-garbled repeats of the same real buyer). Re-running today's
    _merge_parties on those same raw real strings reproduces exactly what a
    fresh extraction would produce now, without needing a live LLM call."""
    with open(CACHE_PATH, "rb") as f:
        cached = pickle.load(f)
    facts = cached["facts"]
    deed_data = dict(facts["sale_deeds"][0])
    deed_data["sellers"] = [p.model_dump() for p in _merge_parties([Party(**p) for p in deed_data["sellers"]])]
    deed_data["buyers"] = [p.model_dump() for p in _merge_parties([Party(**p) for p in deed_data["buyers"]])]
    sale_deed = SaleDeed(**deed_data)
    revenue_record = RevenueRecord(**facts["revenue_record"]) if facts["revenue_record"] else None
    ec = EncumbranceCertificate(**facts["encumbrance_certificate"]) if facts["encumbrance_certificate"] else None
    return sale_deed, revenue_record, ec


def report(label, flags, expect_nonempty, expect_issue_substring=None, expect_severity=None):
    ok_presence = bool(flags) == expect_nonempty
    ok_content = True
    if expect_issue_substring is not None:
        ok_content = any(expect_issue_substring.lower() in f.issue.lower() for f in flags)
    if expect_severity is not None:
        ok_content = ok_content and any(f.severity == expect_severity for f in flags)
    status = "PASS" if (ok_presence and ok_content) else "FAIL"
    print(f"[{status}] {label}: {len(flags)} flag(s)")
    for f in flags:
        print(f"          - [{f.severity}] {f.issue}: {f.detail}")
    return status == "PASS"


def main():
    sale_deed, revenue_record, ec = load_real_objects()
    print("=== Loaded real cached extraction ===")
    print(f"Sale Deed buyer(s): {[b.name for b in sale_deed.buyers]}")
    print(f"Sale Deed land_extent: {sale_deed.land_extent!r}  survey_number: {sale_deed.survey_number!r}")
    print(f"Sale Deed boundaries: {sale_deed.boundaries.model_dump()}")
    print(f"Revenue land_extent: {revenue_record.land_extent!r}  survey_number: {revenue_record.survey_number!r}")
    print(f"Revenue boundaries: {revenue_record.boundaries.model_dump()}")
    print(f"EC entries: {[(e.document_number, e.nature_of_document) for e in ec.entries]}")

    results = []

    print("\n=== BASELINE (real, unmodified data) ===")
    baseline_flags = run_verification([sale_deed], revenue_record, ec)
    for f in baseline_flags:
        print(f"  - [{f.severity}] {f.issue}: {f.detail}")

    print("\n=== TRAP 1: extent_match -- inject a real mismatch ===")
    bad_revenue = revenue_record.model_copy(update={"land_extent": "50 sq yards"})
    flags = extent_match(sale_deed, bad_revenue)
    results.append(report("extent mismatch (deed vs corrupted revenue)", flags, True, "mismatch", "medium"))

    print("\n=== TRAP 2: boundary_match -- inject a real single-side mismatch ===")
    bad_boundaries = sale_deed.boundaries.model_copy(update={"north": "COMPLETELY DIFFERENT LANDMARK XYZ"})
    flags = boundary_match(bad_boundaries, revenue_record.boundaries, "Sale Deed", "Revenue Record")
    results.append(report("north boundary mismatch (injected)", flags, True, "North", "medium"))
    # control: the other 3 sides are untouched real data -- must NOT fire
    untouched_sides_flags = [f for f in flags if "North" not in f.issue]
    results.append(report("  control: other 3 real boundary sides stay quiet", untouched_sides_flags, False))

    print("\n=== TRAP 3: active_encumbrance_check -- inject an unreleased mortgage ===")
    ec_with_mortgage = ec.model_copy(update={"entries": ec.entries + [
        ECEntry(document_number="INJECTED-1", date="01-01-2020", nature_of_document="Simple Mortgage"),
    ]})
    flags = active_encumbrance_check(ec_with_mortgage)
    results.append(report("unreleased mortgage (injected)", flags, True, "Unreleased mortgage", "high"))

    print("\n=== TRAP 4: active_encumbrance_check -- same mortgage, now released ===")
    ec_with_release = ec_with_mortgage.model_copy(update={"entries": ec_with_mortgage.entries + [
        ECEntry(document_number="INJECTED-2", date="01-01-2021", nature_of_document="Discharge of Mortgage"),
    ]})
    flags = active_encumbrance_check(ec_with_release)
    mortgage_flags = [f for f in flags if "mortgage" in f.issue.lower()]
    results.append(report("  control: released mortgage must NOT flag", mortgage_flags, False))

    print("\n=== TRAP 5: active_encumbrance_check -- inject litigation entry ===")
    ec_with_litigation = ec.model_copy(update={"entries": ec.entries + [
        ECEntry(document_number="INJECTED-3", date="01-01-2022", nature_of_document="Court Attachment Order"),
    ]})
    flags = active_encumbrance_check(ec_with_litigation)
    results.append(report("litigation entry (injected)", flags, True, "Litigation", "high"))

    print("\n=== TRAP 6: owner_chain_continuity_check -- real buyer name, continuous chain ===")
    real_buyer = sale_deed.buyers[0]
    deed2_continuous = SaleDeed(
        document_number="SYNTH-D2", registration_date="01-01-2023",
        sellers=[Party(name=real_buyer.name)],  # exact real name -> should match
        buyers=[Party(name="Synthetic New Owner")],
    )
    flags = owner_chain_continuity_check([sale_deed, deed2_continuous])
    results.append(report("continuous chain (real buyer name reused verbatim)", flags, False))

    print("\n=== TRAP 7: owner_chain_continuity_check -- real buyer name, OCR-style garble ===")
    garbled_name = real_buyer.name.upper().replace("O", "0")  # crude OCR-noise simulation
    deed2_garbled = SaleDeed(
        document_number="SYNTH-D2b", registration_date="01-01-2023",
        sellers=[Party(name=garbled_name)],
        buyers=[Party(name="Synthetic New Owner")],
    )
    flags = owner_chain_continuity_check([sale_deed, deed2_garbled])
    results.append(report(f"continuous chain (garbled variant: {garbled_name!r})", flags, False))

    print("\n=== TRAP 8: owner_chain_continuity_check -- real buyer name, genuine gap ===")
    deed2_gap = SaleDeed(
        document_number="SYNTH-D2c", registration_date="01-01-2023",
        sellers=[Party(name="Totally Unrelated Person")],
        buyers=[Party(name="Synthetic New Owner")],
    )
    flags = owner_chain_continuity_check([sale_deed, deed2_gap])
    results.append(report("chain gap (injected, unrelated seller)", flags, True, "Chain of title gap", "high"))

    print(f"\n=== SUMMARY: {sum(results)}/{len(results)} trap checks behaved as expected ===")
    if not all(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
