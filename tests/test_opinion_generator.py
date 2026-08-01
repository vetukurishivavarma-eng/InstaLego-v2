from unittest.mock import MagicMock

from landtitle.opinion.generator import (
    _ensure_all_flags_present,
    _ensure_all_transactions_present,
    _ensure_no_unverified_relationship_claims,
    _ensure_prior_reference_parties_not_claimed,
    _ensure_verified_party_identities_present,
    generate_opinion,
)
from landtitle.schemas import Flag

# Fictional Sale Deed facts reproducing the real structural pattern (a party
# named alongside a relation clause naming a different, non-transacting
# relative) without any real person's name.
_SALE_DEED_FACTS = {
    "sale_deeds": [
        {
            "document_number": None,
            "sellers": [{"name": "Sri. V. Nagendra Rao", "relation": "S/o Sri. V. Krishna Rao"}],
            "buyers": [{"name": "Smt. G. Padma", "relation": "W/o Sri. G. Ramesh"}],
        }
    ]
}

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


def test_ensure_verified_party_identities_present_appends_missing_parties():
    # Confirmed live: given a seller/buyer extracted exactly right (name and
    # relation clause correctly separated), the model still wrote the
    # relation-clause names (a father, a husband) into the narrative as if
    # they were the transacting parties.
    narrative = _ensure_verified_party_identities_present(
        "The chain of ownership begins with Sri. V. Krishna Rao, who sold the property "
        "to Sri. G. Ramesh through Smt. G. Padma.",
        _SALE_DEED_FACTS,
    )
    assert "Sri. V. Nagendra Rao" in narrative
    assert "Smt. G. Padma" in narrative


def test_ensure_verified_party_identities_present_noop_when_already_correct():
    correct = "Sri. V. Nagendra Rao sold the property to Smt. G. Padma."
    assert _ensure_verified_party_identities_present(correct, _SALE_DEED_FACTS) == correct


def test_ensure_verified_party_identities_present_noop_with_no_sale_deed_data():
    assert _ensure_verified_party_identities_present("Some narrative.", {}) == "Some narrative."


def test_generate_opinion_guarantees_correct_party_identities_even_if_model_swaps_them():
    """End-to-end through generate_opinion() reproducing the exact real
    failure: the model's own chain-of-ownership narrative substituted the
    relation-clause names (a father, a husband) for the actual seller/buyer,
    despite both being extracted correctly and despite an explicit prompt
    rule against exactly this confusion."""
    client = MagicMock()
    client.extract_structured.return_value = MagicMock(
        property_summary="",
        chain_of_ownership=(
            "The chain of ownership begins with Sri. V. Krishna Rao, who sold the property "
            "to Sri. G. Ramesh through Smt. G. Padma."
        ),
        encumbrance_status="",
        legal_compliance_check="",
        risk_flags_narrative="No inconsistencies found.",
        overall_recommendation="Clear Title",
    )

    opinion = generate_opinion(facts=_SALE_DEED_FACTS, flags=[], citations=[], client=client)

    assert "Sri. V. Nagendra Rao" in opinion.chain_of_ownership
    assert "Smt. G. Padma" in opinion.chain_of_ownership


_PRIOR_REFERENCE_ONLY_FACTS = {
    "sale_deeds": [
        {
            "document_number": "890/2005",
            "sellers": [{"name": "Sri. M. Ravindra Babu", "relation": "S/o Sri. M. Seetharamaiah"}],
            "buyers": [{"name": "Sri. D. Prasad Rao", "relation": "S/o Sri. D. Nageswara Rao"}],
            "prior_title_deed_references": ["Document No. 77/1990, dated 05-03-1990"],
        }
    ]
}


def test_ensure_prior_reference_parties_not_claimed_appends_disclaimer():
    # Confirmed live, twice: given a prior reference that is a bare document
    # number/date with no party data, the model still invented a transferor
    # for it -- once by reusing a relation-clause name (the seller's own
    # father) for a transaction the source actually describes as a family
    # partition, once by inventing an unrelated name entirely to paper over
    # a real chain gap. This must always append regardless of what specific
    # narrative text preceded it.
    narrative = (
        "Sri. M. Seetharamaiah transferred the property to Sri. M. Ravindra Babu "
        "via Document No. 77/1990."
    )
    result = _ensure_prior_reference_parties_not_claimed(narrative, _PRIOR_REFERENCE_ONLY_FACTS)
    assert "77/1990" in result
    assert "were not independently extracted or verified" in result.lower()


def test_ensure_prior_reference_parties_not_claimed_noop_with_no_references():
    facts = {"sale_deeds": [{"document_number": "890/2005", "sellers": [], "buyers": [],
                              "prior_title_deed_references": []}]}
    narrative = "A clean narrative with nothing to caveat."
    assert _ensure_prior_reference_parties_not_claimed(narrative, facts) == narrative


def test_ensure_prior_reference_parties_not_claimed_idempotent():
    once = _ensure_prior_reference_parties_not_claimed("Some narrative.", _PRIOR_REFERENCE_ONLY_FACTS)
    twice = _ensure_prior_reference_parties_not_claimed(once, _PRIOR_REFERENCE_ONLY_FACTS)
    assert once == twice  # must not append a second disclaimer


def test_generate_opinion_appends_prior_reference_disclaimer_even_if_model_fabricates_transferor():
    """End-to-end through generate_opinion() reproducing the exact real
    failure: the model invented a transferor (a relation-clause name) for a
    prior reference it was never given party data for, for a transaction
    the source describes as a family partition, not a purchase."""
    client = MagicMock()
    client.extract_structured.return_value = MagicMock(
        property_summary="",
        chain_of_ownership=(
            "Sri. M. Seetharamaiah transferred the property to Sri. M. Ravindra Babu "
            "via Document No. 77/1990."
        ),
        encumbrance_status="",
        legal_compliance_check="",
        risk_flags_narrative="No inconsistencies found.",
        overall_recommendation="Clear Title",
    )

    opinion = generate_opinion(facts=_PRIOR_REFERENCE_ONLY_FACTS, flags=[], citations=[], client=client)

    assert "were not independently extracted or verified" in opinion.chain_of_ownership.lower()
    assert "77/1990" in opinion.chain_of_ownership


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


# Fictional facts reproducing the real structural pattern (two transacting
# parties whose own relation clauses each name a different, non-transacting
# relative -- i.e. genuinely unrelated to each other) without any real name.
_UNRELATED_PARTIES_FACTS = {
    "sale_deeds": [
        {
            "document_number": "5551/2018",
            "sellers": [{"name": "Sri. N. Somasekhar", "relation": "S/o Sri. N. Venkatappa"}],
            "buyers": [{"name": "Sri. P. Ravi Teja", "relation": "S/o Sri. P. Chalapathi Rao"}],
        }
    ]
}

# Fictional facts reproducing a genuine family sale: the Buyer's own relation
# clause actually names the Seller.
_GENUINE_FAMILY_SALE_FACTS = {
    "sale_deeds": [
        {
            "document_number": "700/2010",
            "sellers": [{"name": "Sri. B. Koteswara Rao", "relation": None}],
            "buyers": [{"name": "Sri. B. Srinivas", "relation": "S/o Sri. B. Koteswara Rao"}],
        }
    ]
}


def test_ensure_no_unverified_relationship_claims_appends_disclaimer_for_unrelated_parties():
    # Confirmed live: given a Seller and Buyer whose own relation clauses
    # each name a different, non-transacting relative (genuinely unrelated
    # to each other), the model still wrote that the Buyer acquired the
    # property "from his father" -- misreading the Buyer's own relation
    # clause as describing the Seller.
    narrative = (
        "Sri. P. Ravi Teja acquired the property from his father, Sri. N. Somasekhar, "
        "under Document No. 5551/2018."
    )
    result = _ensure_no_unverified_relationship_claims(narrative, _UNRELATED_PARTIES_FACTS)
    assert "no relationship between seller(s)" in result.lower()
    assert "5551/2018" in result


def test_ensure_no_unverified_relationship_claims_restates_genuine_relationship():
    narrative = "Sri. B. Koteswara Rao sold the property to Sri. B. Srinivas."
    result = _ensure_no_unverified_relationship_claims(narrative, _GENUINE_FAMILY_SALE_FACTS)
    assert 'S/o Sri. B. Koteswara Rao' in result
    assert "no relationship between seller(s)" not in result.lower()


def test_ensure_no_unverified_relationship_claims_noop_with_no_sale_deed_data():
    assert _ensure_no_unverified_relationship_claims("Some narrative.", {}) == "Some narrative."


def test_ensure_no_unverified_relationship_claims_noop_when_buyers_missing():
    facts = {"sale_deeds": [{"document_number": "1/2020", "sellers": [{"name": "Sri. X"}], "buyers": []}]}
    assert _ensure_no_unverified_relationship_claims("Some narrative.", facts) == "Some narrative."


def test_ensure_no_unverified_relationship_claims_idempotent():
    once = _ensure_no_unverified_relationship_claims("Some narrative.", _UNRELATED_PARTIES_FACTS)
    twice = _ensure_no_unverified_relationship_claims(once, _UNRELATED_PARTIES_FACTS)
    assert once == twice


def test_generate_opinion_guarantees_relationship_disclaimer_even_if_model_fabricates_it():
    """End-to-end through generate_opinion() reproducing the exact real
    failure: the model invented a father-son relationship between two
    genuinely unrelated transacting parties by misreading the Buyer's own
    relation clause as describing the Seller."""
    client = MagicMock()
    client.extract_structured.return_value = MagicMock(
        property_summary="",
        chain_of_ownership=(
            "Sri. P. Ravi Teja acquired the property from his father, Sri. N. Somasekhar, "
            "under Document No. 5551/2018."
        ),
        encumbrance_status="",
        legal_compliance_check="",
        risk_flags_narrative="No inconsistencies found.",
        overall_recommendation="Clear Title",
    )

    opinion = generate_opinion(facts=_UNRELATED_PARTIES_FACTS, flags=[], citations=[], client=client)

    assert "no relationship between seller(s)" in opinion.chain_of_ownership.lower()


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
