"""Structured extraction schemas — one per validated document type.

Field choices here encode confirmed bug fixes from manual testing, not just
convenient modeling:

- `Party.represented_by_gpa` — without this explicit field, the model conflates
  a GPA (General Power of Attorney) holder's identity with the actual vendor.
- `SaleDeed.land_extent` / `SaleDeed.built_up_area` are kept as separate fields —
  a combined "extent" field caused the model to return built-up area (sq ft)
  when land extent (sq yards) was asked for, and vice versa.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Boundaries(BaseModel):
    north: Optional[str] = None
    south: Optional[str] = None
    east: Optional[str] = None
    west: Optional[str] = None


class Party(BaseModel):
    name: str
    relation: Optional[str] = None  # e.g. "S/o", "W/o", "D/o" + referenced name
    address: Optional[str] = None
    represented_by_gpa: Optional[str] = Field(
        default=None,
        description=(
            "If this party is represented by a General Power of Attorney holder, "
            "the GPA holder's name and GPA document reference. None if the party "
            "acts in person. Keep separate from `name` — never merge identities."
        ),
    )


class SaleDeed(BaseModel):
    document_type: str = "Sale Deed"
    sellers: list[Party] = Field(default_factory=list)
    buyers: list[Party] = Field(default_factory=list)
    land_extent: Optional[str] = Field(
        default=None,
        description="Land extent only, as a numeric quantity with its unit (square yards, square "
        "feet, acres, cents, etc.) copied exactly as written in the source text. Never combined "
        "with built-up area. Never a placeholder or illustrative figure -- if no land extent is "
        "stated, leave this null.",
    )
    built_up_area: Optional[str] = Field(
        default=None,
        description="Built-up/constructed area only, as a numeric quantity with its unit copied "
        "exactly as written in the source text. Never combined with land extent. Never a "
        "placeholder or illustrative figure -- if no built-up area is stated, leave this null.",
    )
    boundaries: Boundaries = Field(default_factory=Boundaries)
    sale_consideration: Optional[str] = None
    registration_date: Optional[str] = None
    document_number: Optional[str] = None
    prior_title_deed_references: list[str] = Field(default_factory=list)
    encumbrance_declaration: Optional[str] = None
    sub_registrar_office: Optional[str] = None
    property_description: Optional[str] = None
    survey_number: Optional[str] = Field(
        default=None,
        description="Municipal/plot numbering as it appears on the deed (typically a plot number "
        "and/or municipal door or premises number), copied exactly as written in the source text. "
        "This is NOT the revenue survey number used in Pahani records — do not conflate the two "
        "systems. Never a placeholder or illustrative value -- if not stated, leave this null.",
    )


class ECEntry(BaseModel):
    """One row of an Encumbrance Certificate. Extracted from table structure,
    never from flattened prose — see ec_tables.py for why."""

    document_number: Optional[str] = None
    date: Optional[str] = None
    nature_of_document: Optional[str] = None
    executants_sellers: Optional[str] = None
    claimants_buyers: Optional[str] = None
    consideration: Optional[str] = None


class ECEntryBatch(BaseModel):
    """Wrapper used when asking the LLM to clean up a batch of already
    table-structured, pre-labeled EC rows (see ec_tables.py)."""

    entries: list[ECEntry] = Field(default_factory=list)


class EncumbranceCertificate(BaseModel):
    document_type: str = "Encumbrance Certificate"
    certificate_number: Optional[str] = None
    issuing_office: Optional[str] = None
    property_description: Optional[str] = None
    search_period: Optional[str] = None
    entries: list[ECEntry] = Field(default_factory=list)
    mortgage_entries_found: bool = False
    litigation_entries_found: bool = False


class RevenueRecord(BaseModel):
    document_type: str = "Revenue Record"
    mandal: Optional[str] = None
    district: Optional[str] = None
    village: Optional[str] = None
    revenue_year: Optional[str] = None
    survey_number: Optional[str] = Field(
        default=None,
        description="Revenue survey numbering as it appears in the Pahani (a slash-separated "
        "survey/sub-division number), copied exactly as written in the source text. This is NOT "
        "the municipal/plot numbering used in Sale Deeds. Never a placeholder or illustrative "
        "value -- if not stated, leave this null.",
    )
    pattadar_name: Optional[str] = None
    pattadar_relation: Optional[str] = None
    land_extent: Optional[str] = None
    classification: Optional[str] = None
    land_use: Optional[str] = None
    possession_type: Optional[str] = None
    mutation_date: Optional[str] = None
    mutation_order_number: Optional[str] = None
    prior_pattadar: Optional[str] = None
    bank_loan_entry: Optional[str] = None
    court_attachment: Optional[str] = None
    government_dues_pending: Optional[str] = None
    tax_arrears: Optional[str] = None
    boundaries: Boundaries = Field(default_factory=Boundaries)
    assigned_land_status: Optional[str] = None
    pending_litigation: Optional[str] = None


class Flag(BaseModel):
    issue: str
    severity: str  # "high" | "medium" | "low"
    detail: str
    documents_compared: list[str] = Field(default_factory=list)


class LegalOpinion(BaseModel):
    property_summary: str = ""
    chain_of_ownership: str = ""
    encumbrance_status: str = ""
    legal_compliance_check: str = ""
    risk_flags_narrative: str = ""
    overall_recommendation: str = ""  # "Clear Title" | "Title with Minor Issues" | "Not Recommended"

    # Non-negotiable human-review gate (see project README). Never set True
    # by anything other than opinion.finalize_opinion() after an explicit
    # reviewer name is supplied — the LLM must never be trusted to set this.
    is_draft: bool = True
    reviewed_by: Optional[str] = None
