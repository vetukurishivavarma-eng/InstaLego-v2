from unittest.mock import MagicMock

from landtitle.opinion.generator import (
    _ensure_all_flags_present,
    _ensure_all_transactions_present,
    generate_opinion,
)
from landtitle.schemas import Flag

# Fictional EC data reproducing the real structural pattern (a builder's sale
# split across 2 documents, then a later resale) without any real person's
# name.
_EC_FACTS = {
    "encumbrance_certificate": {
        "entries": [
            {
                "document_number": "2609/2000",
                "date": "09-11-2000",
                "nature_of_document": "Sale Deed",
                "executants_sellers": "M/s Sunrise Realty",
                "claimants_buyers": "Kaushalya Venkataramanian & S. Suresh",
                "consideration": "Rs. 42,00,000/-",
            },
            {
                "document_number": "2610/2000",
                "date": "09-11-2000",
                "nature_of_document": "Sale Deed",
                "executants_sellers": "M/s Sunrise Realty",
                "claimants_buyers": "R. Venkataramanian",
                "consideration": "Rs. 38,50,000/-",
            },
            {
                "document_number": "400/2012",
                "date": "07-03-2012",
                "nature_of_document": "Sale Deed",
                "executants_sellers": "Kaushalya Venkataramanian, Priyanka Rajaraman",
                "claimants_buyers": "Priyanka Rajaraman",
                "consideration": "Rs. 1,70,00,000/-",
            },
        ]
    }
}

_LAND_EXTENT_FLAG = Flag(
    issue="Land extent could not be cross-verified",
    severity="low",
    detail="Sale Deed land_extent: '329.61 square yards'; Pahani land_extent: "
    "'329.61 Sq. Yards (0.068 Acres)'. Could not parse one or both into a comparable unit.",
    documents_compared=["Sale Deed", "Revenue Record"],
)


def test_ensure_all_flags_present_appends_missing_flag():
    # Confirmed live: the model said "No inconsistencies found." despite a
    # real low-severity verified flag existing — silently dropping it.
    narrative = _ensure_all_flags_present("No inconsistencies found.", [_LAND_EXTENT_FLAG])
    assert "Land extent could not be cross-verified" in narrative
    assert "no inconsistencies" not in narrative.lower()


def test_ensure_all_flags_present_leaves_correct_narrative_untouched():
    narrative = _ensure_all_flags_present(
        "One low-severity flag: Land extent could not be cross-verified due to unit mismatch.",
        [_LAND_EXTENT_FLAG],
    )
    assert narrative.count("Land extent could not be cross-verified") == 1


def test_ensure_all_flags_present_appends_after_unrelated_narrative():
    narrative = _ensure_all_flags_present(
        "The chain of title appears continuous and well-documented.", [_LAND_EXTENT_FLAG]
    )
    assert "chain of title appears continuous" in narrative
    assert "Land extent could not be cross-verified" in narrative


def test_ensure_all_flags_present_noop_when_no_flags():
    assert _ensure_all_flags_present("No inconsistencies found.", []) == "No inconsistencies found."


def test_ensure_all_transactions_present_appends_missing_earlier_transactions():
    # Confirmed live: the model jumped straight to the most recent
    # transaction (400/2012) and silently dropped the 2 earlier real
    # transactions from a real EC's entries.
    narrative = _ensure_all_transactions_present(
        "The chain begins with the sale registered under Document No.400/2012.", _EC_FACTS
    )
    assert "2609/2000" in narrative
    assert "2610/2000" in narrative
    assert "400/2012" in narrative  # original text preserved, not replaced


def test_ensure_all_transactions_present_noop_when_all_already_mentioned():
    complete = (
        "The chain begins with 2609/2000 and 2610/2000 (the original builder sales), "
        "followed by 400/2012 (the resale)."
    )
    assert _ensure_all_transactions_present(complete, _EC_FACTS) == complete


def test_ensure_all_transactions_present_ignores_non_sale_deed_entries():
    facts = {
        "encumbrance_certificate": {
            "entries": [
                {"document_number": "MTG/55", "nature_of_document": "Mortgage", "date": "01-01-2010"},
            ]
        }
    }
    # A mortgage entry isn't a chain-of-title transaction — must not be
    # injected into the chain-of-ownership narrative.
    narrative = _ensure_all_transactions_present("Some narrative.", facts)
    assert narrative == "Some narrative."


def test_ensure_all_transactions_present_noop_with_no_ec_data():
    assert _ensure_all_transactions_present("Some narrative.", {}) == "Some narrative."


def test_generate_opinion_guarantees_flag_even_if_model_omits_it():
    """End-to-end through generate_opinion() with a mocked client that
    reproduces the exact real failure: a non-empty VERIFIED FLAGS list, but
    the model's own narrative claims none were found."""
    client = MagicMock()
    client.extract_structured.return_value = MagicMock(
        property_summary="",
        chain_of_ownership="",
        encumbrance_status="",
        legal_compliance_check="",
        risk_flags_narrative="No inconsistencies found.",
        overall_recommendation="Clear Title",
    )

    opinion = generate_opinion(facts={}, flags=[_LAND_EXTENT_FLAG], citations=[], client=client)

    assert "Land extent could not be cross-verified" in opinion.risk_flags_narrative
    assert opinion.is_draft is True


def test_generate_opinion_guarantees_earlier_transactions_even_if_model_omits_them():
    """End-to-end through generate_opinion() reproducing the exact real
    failure: the model's own chain-of-ownership narrative jumped straight to
    the most recent transaction, dropping 2 real earlier ones from the EC."""
    client = MagicMock()
    client.extract_structured.return_value = MagicMock(
        property_summary="",
        chain_of_ownership="The sale was registered under Document No.400/2012.",
        encumbrance_status="",
        legal_compliance_check="",
        risk_flags_narrative="No inconsistencies found.",
        overall_recommendation="Clear Title",
    )

    opinion = generate_opinion(facts=_EC_FACTS, flags=[], citations=[], client=client)

    assert "2609/2000" in opinion.chain_of_ownership
    assert "2610/2000" in opinion.chain_of_ownership
