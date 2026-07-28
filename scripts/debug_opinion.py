"""Debug helper: run intake->extraction->verification->retrieval ONCE and cache
the result to disk, then let us retry just the opinion-generation call (the
step that's been failing with a server-side 500) as many times as needed
without repeating slow OCR + multiple extraction LLM calls each time.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from landtitle.legal.retrieval import LegalRetriever, build_citation_context
from landtitle.llm.client import QwenClient
from landtitle.pipeline import (
    extract_ec_document,
    extract_revenue_record_document,
    extract_sale_deed_document,
    run_verification,
    select_citations,
)

CACHE_PATH = Path(__file__).resolve().parent / "_debug_opinion_cache.pkl"


def build_and_cache():
    client = QwenClient()
    retriever = LegalRetriever()

    sale_deeds = [extract_sale_deed_document(
        "C:/Users/Shiva/Downloads/Prior-Owner-2-Sale-Deed.pdf", client)]
    revenue_record = extract_revenue_record_document(
        "C:/Users/Shiva/Downloads/sample_pahani_revenue_record.txt", client)
    ec = extract_ec_document(
        "C:/Users/Shiva/Downloads/sample_encumbrance_certificate.txt", client)

    flags = run_verification(sale_deeds, revenue_record, ec)
    citations = select_citations(flags, retriever)

    facts = {
        "sale_deeds": [d.model_dump() for d in sale_deeds],
        "revenue_record": revenue_record.model_dump() if revenue_record else None,
        "encumbrance_certificate": ec.model_dump() if ec else None,
    }

    with open(CACHE_PATH, "wb") as f:
        pickle.dump({"facts": facts, "flags": flags, "citations": citations}, f)
    print(f"Cached facts/flags/citations to {CACHE_PATH}")
    return facts, flags, citations


def load_cache():
    with open(CACHE_PATH, "rb") as f:
        d = pickle.load(f)
    return d["facts"], d["flags"], d["citations"]


def report_sizes(facts, flags, citations):
    from landtitle.opinion.generator import SYSTEM_PROMPT

    facts_json = json.dumps(facts, indent=2, default=str)
    citation_context = build_citation_context(citations)
    print(f"num sale_deeds: {len(facts.get('sale_deeds') or [])}")
    print(f"num flags: {len(flags)}")
    print(f"num citations: {len(citations)}")
    print(f"facts json chars: {len(facts_json)}")
    print(f"citation_context chars: {len(citation_context)}")
    print(f"system prompt chars: {len(SYSTEM_PROMPT)}")
    total = len(facts_json) + len(citation_context) + len(SYSTEM_PROMPT)
    print(f"rough total chars (facts+citations+system, excl. schema/flags text): {total}")
    print(f"rough token estimate (chars/4): {total // 4}")


def call_with_budget(facts, flags, citations, client, max_new_tokens):
    """Bisection helper: replicate generate_opinion's prompt construction but
    with an explicit, overridable max_new_tokens (bypassing the fixed
    config.OPINION_MAX_NEW_TOKENS) so we can binary-search the real ceiling."""
    from landtitle.opinion.generator import (
        SYSTEM_PROMPT,
        _ensure_all_flags_present,
        _ensure_all_transactions_present,
        _OpinionDraftFields,
    )

    verified_flags_text = (
        "\n".join(f"- [{f.severity.upper()}] {f.issue}: {f.detail}" for f in flags)
        if flags else "(none — no inconsistencies were detected by the verification layer)"
    )
    citation_context = build_citation_context(citations)
    user_prompt = f"""EXTRACTED DOCUMENT FACTS:
{json.dumps(facts, indent=2, default=str)}

VERIFIED FLAGS (the ONLY risk flags you may report):
{verified_flags_text}

LEGAL CITATIONS (use ONLY these exact labels when citing a section):
{citation_context}

Draft the opinion sections: Property Summary, Chain of Ownership, Encumbrance Status, \
Legal Compliance Check, a Risk Flags narrative (restating only the VERIFIED FLAGS above, or \
"No inconsistencies found" if none), and an Overall Recommendation — exactly one of \
"Clear Title", "Title with Minor Issues", or "Not Recommended"."""
    draft = client.extract_structured(SYSTEM_PROMPT, user_prompt, _OpinionDraftFields, max_new_tokens=max_new_tokens)
    from landtitle.schemas import LegalOpinion

    return LegalOpinion(
        property_summary=draft.property_summary,
        chain_of_ownership=_ensure_all_transactions_present(draft.chain_of_ownership, facts),
        encumbrance_status=draft.encumbrance_status,
        legal_compliance_check=draft.legal_compliance_check,
        risk_flags_narrative=_ensure_all_flags_present(draft.risk_flags_narrative, flags),
        overall_recommendation=draft.overall_recommendation,
        is_draft=True,
        reviewed_by=None,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Re-run OCR/extraction/retrieval instead of using cache")
    parser.add_argument("--sizes-only", action="store_true", help="Just print prompt size stats, don't call the LLM")
    parser.add_argument("--no-citations", action="store_true", help="Test with citations stripped out")
    parser.add_argument("--num-citations", type=int, default=None, help="Cap citations to this many for the test call (bisection)")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override max_new_tokens for the test call (bisection)")
    parser.add_argument("--export", default=None, help="If the call succeeds, export the opinion to this PDF path")
    args = parser.parse_args()

    if args.rebuild or not CACHE_PATH.exists():
        facts, flags, citations = build_and_cache()
    else:
        facts, flags, _stale_citations = load_cache()
        # Re-derive citations from cached flags on every load (fast: local
        # FAISS lookup, no OCR/LLM) so this always reflects the current
        # select_citations() cap/priority logic in pipeline.py, not whatever
        # was true when the cache was built.
        citations = select_citations(flags, LegalRetriever())
        print(f"Loaded cache from {CACHE_PATH} (citations re-selected fresh)")

    if args.no_citations:
        citations = []
    elif args.num_citations is not None:
        citations = citations[:args.num_citations]

    report_sizes(facts, flags, citations)

    if args.sizes_only:
        sys.exit(0)

    from landtitle.config import OPINION_MAX_NEW_TOKENS

    client = QwenClient()
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else OPINION_MAX_NEW_TOKENS
    print(f"\nCalling opinion generation (num_citations={len(citations)}, max_new_tokens={max_new_tokens})...")
    try:
        opinion = call_with_budget(facts, flags, citations, client, max_new_tokens)
        print("SUCCESS")
        print(opinion.model_dump_json(indent=2))
        if args.export:
            from landtitle.opinion.pdf_export import export_opinion_pdf

            export_opinion_pdf(opinion, args.export)
            print(f"\nExported PDF to {args.export}")
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
