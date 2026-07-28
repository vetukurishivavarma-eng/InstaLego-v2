from landtitle.schemas import Boundaries, ECEntry, EncumbranceCertificate, Party, RevenueRecord, SaleDeed
from landtitle.verification.checks import (
    active_encumbrance_check,
    boundary_match,
    extent_match,
    name_similarity_check,
    owner_chain_continuity_check,
    survey_number_match,
)


def make_sale_deed(**kwargs) -> SaleDeed:
    defaults = dict(sellers=[], buyers=[], land_extent=None, boundaries=Boundaries())
    defaults.update(kwargs)
    return SaleDeed(**defaults)


def make_revenue_record(**kwargs) -> RevenueRecord:
    defaults = dict(land_extent=None, boundaries=Boundaries())
    defaults.update(kwargs)
    return RevenueRecord(**defaults)


# --- name_similarity_check ---------------------------------------------------


def test_name_similarity_exact_match():
    is_match, score = name_similarity_check("Ramesh Kumar", "Ramesh Kumar")
    assert is_match
    assert score == 100


def test_name_similarity_minor_variance_passes():
    is_match, score = name_similarity_check("Venkata Ramana Reddy", "Venkat Ramana Reddy")
    assert is_match


def test_name_similarity_different_names_fails():
    is_match, score = name_similarity_check("Ramesh Kumar", "Suresh Babu")
    assert not is_match


def test_name_similarity_missing_name():
    is_match, score = name_similarity_check(None, "Ramesh Kumar")
    assert not is_match
    assert score == 0.0


# --- survey_number_match ------------------------------------------------------


def test_survey_number_match_both_present_no_mapping_no_flag():
    deed = make_sale_deed(survey_number="Plot No.4, Municipal No.6-3-862/3")
    revenue = make_revenue_record(survey_number="82/A/4")
    flags = survey_number_match(deed, revenue)
    assert flags == []


def test_survey_number_match_missing_flags_low():
    deed = make_sale_deed(survey_number=None)
    revenue = make_revenue_record(survey_number="82/A/4")
    flags = survey_number_match(deed, revenue)
    assert len(flags) == 1
    assert flags[0].severity == "low"


def test_survey_number_match_mapping_mismatch_flags_high():
    deed = make_sale_deed(survey_number="Plot No.4, Municipal No.6-3-862/3")
    revenue = make_revenue_record(survey_number="99/Z/9")
    mapping = {"Plot No.4, Municipal No.6-3-862/3": "82/A/4"}
    flags = survey_number_match(deed, revenue, mapping=mapping)
    assert len(flags) == 1
    assert flags[0].severity == "high"


# --- extent_match --------------------------------------------------------------


def test_extent_match_equivalent_units_no_flag():
    deed = make_sale_deed(land_extent="266 sq yards")
    revenue = make_revenue_record(land_extent="2394 sq ft")  # 266*9 = 2394
    flags = extent_match(deed, revenue)
    assert flags == []


def test_extent_match_mismatch_flags_medium():
    deed = make_sale_deed(land_extent="266 sq yards")
    revenue = make_revenue_record(land_extent="1200 sq ft")
    flags = extent_match(deed, revenue)
    assert len(flags) == 1
    assert flags[0].severity == "medium"


def test_extent_match_unparseable_flags_low():
    deed = make_sale_deed(land_extent="some ambiguous value")
    revenue = make_revenue_record(land_extent="82/A/4")
    flags = extent_match(deed, revenue)
    assert len(flags) == 1
    assert flags[0].severity == "low"


# --- boundary_match --------------------------------------------------------------


def test_boundary_match_matching_no_flags():
    b1 = Boundaries(north="Road", south="Plot 5", east="Plot 3", west="Drain")
    b2 = Boundaries(north="Road", south="Plot 5", east="Plot 3", west="Drain")
    assert boundary_match(b1, b2, "Sale Deed", "Pahani") == []


def test_boundary_match_mismatch_flags():
    b1 = Boundaries(north="Road")
    b2 = Boundaries(north="Canal")
    flags = boundary_match(b1, b2, "Sale Deed", "Pahani")
    assert len(flags) == 1
    assert "North" in flags[0].issue


def test_boundary_match_missing_side_not_flagged():
    b1 = Boundaries(north=None)
    b2 = Boundaries(north="Canal")
    assert boundary_match(b1, b2, "Sale Deed", "Pahani") == []


# --- active_encumbrance_check ------------------------------------------------------


def test_active_encumbrance_check_released_mortgage_no_flag():
    ec = EncumbranceCertificate(entries=[
        ECEntry(document_number="1", date="01-01-2000", nature_of_document="Simple Mortgage"),
        ECEntry(document_number="2", date="01-01-2005", nature_of_document="Discharge of Mortgage"),
    ])
    flags = active_encumbrance_check(ec)
    assert flags == []


def test_active_encumbrance_check_unreleased_mortgage_flags_high():
    ec = EncumbranceCertificate(entries=[
        ECEntry(document_number="1", date="01-01-2000", nature_of_document="Simple Mortgage"),
    ])
    flags = active_encumbrance_check(ec)
    assert len(flags) == 1
    assert flags[0].severity == "high"
    assert "Unreleased mortgage" in flags[0].issue


def test_active_encumbrance_check_litigation_flags_high():
    ec = EncumbranceCertificate(entries=[
        ECEntry(document_number="3", date="01-01-2010", nature_of_document="Court Attachment Order"),
    ])
    flags = active_encumbrance_check(ec)
    assert len(flags) == 1
    assert "Litigation" in flags[0].issue


# --- owner_chain_continuity_check -----------------------------------------------------


def test_owner_chain_continuous_no_flags():
    deed1 = make_sale_deed(
        document_number="D1", registration_date="01-01-2000",
        buyers=[Party(name="Ramesh Kumar")],
    )
    deed2 = make_sale_deed(
        document_number="D2", registration_date="01-01-2010",
        sellers=[Party(name="Ramesh Kumar")], buyers=[Party(name="Suresh Babu")],
    )
    flags = owner_chain_continuity_check([deed1, deed2])
    assert flags == []


def test_owner_chain_gap_flags_high():
    deed1 = make_sale_deed(
        document_number="D1", registration_date="01-01-2000",
        buyers=[Party(name="Ramesh Kumar")],
    )
    deed2 = make_sale_deed(
        document_number="D2", registration_date="01-01-2010",
        sellers=[Party(name="Someone Else Entirely")], buyers=[Party(name="Suresh Babu")],
    )
    flags = owner_chain_continuity_check([deed1, deed2])
    assert len(flags) == 1
    assert flags[0].severity == "high"


def test_owner_chain_different_buyers_same_date_not_flagged_as_conflict():
    """Confirmed failure mode: co-buyers on the same date, or simply
    different parties across time, must NOT be treated as a conflict on
    their own — only a genuine missing link should be flagged."""
    deed1 = make_sale_deed(
        document_number="D1", registration_date="01-01-2000",
        buyers=[Party(name="Ramesh Kumar"), Party(name="Suresh Babu")],
    )
    deed2 = make_sale_deed(
        document_number="D2", registration_date="01-01-2000",
        sellers=[Party(name="Ramesh Kumar"), Party(name="Suresh Babu")],
        buyers=[Party(name="New Owner")],
    )
    flags = owner_chain_continuity_check([deed1, deed2])
    assert flags == []
