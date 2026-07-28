"""Opinion generation — combines extracted facts (Layer 2), verification
flags (Layer 3), and legal grounding (Layer 4) into a structured draft
opinion.

Non-negotiable product/liability requirement: this tool must never present
output as a final, reliance-ready legal opinion without a human review
step. It is a research/first-draft accelerator for a licensed practitioner,
not an autonomous opinion generator. `LegalOpinion.is_draft` defaults True
and can only be cleared via `finalize_opinion()`, which requires an explicit
non-empty reviewer name — the LLM's output is never trusted to set this
field itself (see `_OpinionDraftFields` below, which the LLM writes into
instead of the full `LegalOpinion`).
"""
from __future__ import annotations

import json
import logging

from pydantic import BaseModel

from landtitle.config import OPINION_MAX_NEW_TOKENS
from landtitle.legal.retrieval import RetrievedChunk, build_citation_context
from landtitle.llm.client import QwenClient
from landtitle.schemas import ECEntry, Flag, LegalOpinion

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are drafting a preliminary land title due-diligence opinion for review \
by a licensed legal practitioner. This is a first-draft accelerator, NOT a final legal opinion. \
Follow these rules exactly — they were derived from confirmed failure cases in prior testing:

1. Never introduce a person, relationship, date, or fact not explicitly present in the extracted \
document facts provided to you. "W/o [Name]" / "S/o [Name]" indicates relationship only, NOT \
co-ownership, unless that person is separately named as a party to a transaction.
2. Treat each document number as a separate legal transaction. Never merge multiple document \
numbers into one combined narrative. Your Chain of Ownership MUST mention every prior transaction \
on record — including every Sale Deed entry in the Encumbrance Certificate and every reference in \
"prior_title_deed_references" — each by its own document number. Do not stop at the most recent \
transaction just because it supersedes the earlier ones; the full chain back to the earliest \
recorded transaction is required.
3. Only report a Risk Flag if it appears in the "VERIFIED FLAGS" list provided below. Do not \
invent, infer, or independently identify any additional risk flags — a normal chain of title has \
different parties at different points in time, and that is not itself a conflict. If the VERIFIED \
FLAGS list is non-empty, you MUST restate every single flag in it in your narrative, regardless of \
severity — never omit a verified flag even if it seems minor. Only say "No inconsistencies found" \
if the VERIFIED FLAGS list is completely empty.
4. Cite legal provisions using ONLY the exact Act name + Section number shown in the "LEGAL \
CITATIONS" labels below (e.g. "[Transfer of Property Act, Section 55]"). Do not use any other \
section number even if you recall one from general knowledge or training.
5. Do not sign this output with any name or identity of your own. Do not make any statement about \
who or what you are.
6. This draft requires human review before it can be relied upon. Do not state or imply that this \
opinion is final or ready for reliance.
"""


class _OpinionDraftFields(BaseModel):
    """What the LLM is allowed to write. Deliberately excludes `is_draft`
    and `reviewed_by` — those are set exclusively by this module's own code."""

    property_summary: str = ""
    chain_of_ownership: str = ""
    encumbrance_status: str = ""
    legal_compliance_check: str = ""
    risk_flags_narrative: str = ""
    overall_recommendation: str = ""


def _ensure_all_flags_present(narrative: str, flags: list[Flag]) -> str:
    """Guarantee every verified flag is textually present in the narrative,
    the same way legal citations are injected as verified labels rather than
    left to the LLM's discretion (see legal/retrieval.py). Confirmed live:
    the model silently dropped a real low-severity flag from its narrative
    despite the prompt instructing it to restate every flag — an LLM
    complying with a rule is not something this project treats as
    guaranteed (see verification/checks.py's "never LLM judgment"
    principle). A simple case-insensitive substring check on `flag.issue` is
    used rather than fuzzy matching: a false negative here just appends a
    slightly redundant line, which is a safe failure mode, whereas a false
    positive from a looser fuzzy match could let a real omission slip
    through — unacceptable for a verified risk flag."""
    if not flags:
        return narrative

    missing = [f for f in flags if f.issue.lower() not in narrative.lower()]
    if not missing:
        return narrative

    logger.warning(
        "Model omitted %d verified flag(s) from its narrative, re-injecting: %s",
        len(missing), [f.issue for f in missing],
    )
    missing_lines = "\n".join(f"- [{f.severity.upper()}] {f.issue}: {f.detail}" for f in missing)
    if narrative.strip().lower() in ("", "no inconsistencies found.", "no inconsistencies found"):
        # The model's narrative added nothing of value on top of the
        # boilerplate empty-case phrase — replace it outright instead of
        # appending after a now-contradictory "no inconsistencies" sentence.
        return missing_lines
    return f"{narrative}\n\n{missing_lines}"


def _ensure_all_transactions_present(narrative: str, facts: dict) -> str:
    """Guarantee every real Sale Deed transaction on record is mentioned in
    the chain-of-ownership narrative by document number, the same way
    verified flags are guaranteed present in the risk narrative (see
    _ensure_all_flags_present) — an LLM instructed to include every
    transaction is not something this project treats as guaranteed.

    Confirmed live: a real Encumbrance Certificate had 3 Sale Deed entries
    spanning the property's full history (an original builder sale split
    across 2 documents, then a later resale), but the model's own narrative
    jumped straight to the most recent transaction and silently dropped the
    2 earlier ones — not a merge/conflation, but an outright omission of
    real transactions it had been given."""
    ec = facts.get("encumbrance_certificate") or {}
    entries = [
        ECEntry(**e) for e in ec.get("entries", [])
        if e.get("document_number") and e.get("nature_of_document") and "sale deed" in e["nature_of_document"].lower()
    ]
    missing = [e for e in entries if e.document_number.lower() not in narrative.lower()]
    if not missing:
        return narrative

    logger.warning(
        "Model omitted %d real Sale Deed transaction(s) from the chain-of-ownership "
        "narrative, re-injecting: %s",
        len(missing), [e.document_number for e in missing],
    )
    missing_lines = "\n".join(
        f"- Document No.{e.document_number}"
        + (f", dated {e.date}" if e.date else "")
        + (f": {e.executants_sellers} to {e.claimants_buyers}" if e.executants_sellers or e.claimants_buyers else "")
        + (f", consideration {e.consideration}" if e.consideration else "")
        for e in missing
    )
    return f"{narrative}\n\nAdditional prior transactions on record (per Encumbrance Certificate):\n{missing_lines}"


def generate_opinion(
    facts: dict,
    flags: list[Flag],
    citations: list[RetrievedChunk],
    client: QwenClient,
) -> LegalOpinion:
    logger.info("Generating opinion: %d flag(s), %d citation(s)", len(flags), len(citations))
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

    draft = client.extract_structured(
        SYSTEM_PROMPT, user_prompt, _OpinionDraftFields, max_new_tokens=OPINION_MAX_NEW_TOKENS
    )

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


def finalize_opinion(opinion: LegalOpinion, reviewed_by: str) -> LegalOpinion:
    """The only sanctioned way to clear `is_draft`. Raises if no reviewer
    name is supplied — this is the product's non-negotiable review gate,
    not a formality."""
    if not reviewed_by or not reviewed_by.strip():
        raise ValueError("finalize_opinion requires a non-empty reviewed_by name — the review gate cannot be bypassed.")
    return opinion.model_copy(update={"is_draft": False, "reviewed_by": reviewed_by.strip()})
