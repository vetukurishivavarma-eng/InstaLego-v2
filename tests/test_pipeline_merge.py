from landtitle.pipeline import _is_placeholder_party_name, _merge_extractions, _merge_parties, run_verification
from landtitle.schemas import Party, SaleDeed

# Fictional names below reproduce the exact structural pattern found in a
# real Sale Deed (OCR noise on repeated restatements of the same person,
# plus a father/son sharing a surname) without using any real person's name.


def test_merge_parties_dedupes_ocr_garbled_repeats_of_same_person():
    parties = [
        Party(name="Smt. Kamala W/o Sri. R. Venkataraman", relation="W/o", address="No.7, Lotus Villa"),
        Party(name="KAMALA, W/O. SRI. R. VENKATARAAMAN"),  # OCR-garbled repeat, no other fields
        Party(name="Smt. Kamala W/o R. Venkataraman"),  # near-identical repeat
    ]
    merged = _merge_parties(parties)
    assert len(merged) == 1
    assert merged[0].address == "No.7, Lotus Villa"


def test_merge_parties_keeps_distinct_people_separate():
    parties = [
        Party(name="Smt. Kamala W/o Sri. R. Venkataraman"),
        Party(name="Sri. Suresh S/o Sri. R. Venkataraman"),
    ]
    merged = _merge_parties(parties)
    assert len(merged) == 2


def test_merge_parties_does_not_conflate_father_and_son_sharing_a_surname():
    # The real bug this guards against: naive full-name fuzzy matching scored
    # a father/son pair (sharing a surname via the S/o clause) almost as high
    # as genuine OCR noise on one person, which would have silently dropped a
    # real party from the sellers list.
    parties = [
        Party(name="Sri. Suresh S/o Sri. R. Venkataraman"),
        Party(name="Sri. R. VENKATARAMAN S/o S.Krishnamurthy"),
        Party(name="SRI. R. VENKATARAMAN, W/O. SRI. $s. KRISHNAMURTHY"),  # OCR variant of the father
    ]
    merged = _merge_parties(parties)
    assert len(merged) == 2
    given_names = {p.name for p in merged}
    assert any("Suresh" in n for n in given_names)


def test_merge_parties_backfills_gpa_from_a_different_chunk():
    # A later chunk caught the GPA detail the first chunk's extraction missed
    # for the same real person — must not be lost.
    parties = [
        Party(name="Smt. Kamala W/o Sri. R. Venkataraman"),
        Party(
            name="Smt. Kamala W/o Sri. R. Venkataraman",
            represented_by_gpa="Sri. R. Venkataraman S/o S.Krishnamurthy, ENDT814/G1/2012",
        ),
    ]
    merged = _merge_parties(parties)
    assert len(merged) == 1
    assert merged[0].represented_by_gpa == "Sri. R. Venkataraman S/o S.Krishnamurthy, ENDT814/G1/2012"


def test_is_placeholder_party_name_detects_numbered_collective_reference():
    # Confirmed live: a real Sale Deed's back-reference to already-named
    # vendors was extracted as if it were itself a new seller.
    assert _is_placeholder_party_name("Vendor No.1, 2 and Vendor No.3 through their GPA holder")
    assert _is_placeholder_party_name("Vendee")
    assert _is_placeholder_party_name("The said Vendors")


def test_is_placeholder_party_name_leaves_real_names_alone():
    assert not _is_placeholder_party_name("Smt. Kamala W/o Sri. R. Venkataraman")
    assert not _is_placeholder_party_name("Sri. Suresh S/o Sri. R. Venkataraman")


def test_merge_parties_drops_placeholder_entries():
    parties = [
        Party(name="Smt. Kamala W/o Sri. R. Venkataraman"),
        Party(name="Sri. Suresh S/o Sri. R. Venkataraman"),
        Party(name="Vendor No.1, 2 and Vendor No.3 through their GPA holder"),
    ]
    merged = _merge_parties(parties)
    assert len(merged) == 2
    assert all("Vendor No." not in p.name for p in merged)


def test_merge_extractions_uses_fuzzy_dedup_for_sale_deed_sellers():
    chunk1 = SaleDeed(
        sellers=[Party(name="Smt. Kamala W/o Sri. R. Venkataraman", address="No.7, Lotus Villa")],
        buyers=[Party(name="Smt. Priya Rajan")],
        land_extent="329.61 square yards",
    )
    chunk2 = SaleDeed(
        sellers=[Party(name="KAMALA, W/O. SRI. R. VENKATARAAMAN")],  # same person, garbled
        buyers=[Party(name="Smt. Priya Rajan")],
        document_number="400/2012",
    )
    merged = _merge_extractions([chunk1, chunk2])
    assert len(merged.sellers) == 1
    assert merged.sellers[0].address == "No.7, Lotus Villa"
    assert len(merged.buyers) == 1
    assert merged.document_number == "400/2012"
    assert merged.land_extent == "329.61 square yards"


def test_run_verification_calls_unevidenced_prior_reference_check():
    # Integration-level check that run_verification() actually wires in
    # unevidenced_prior_reference_check(), not just that the unit function
    # behaves correctly in isolation -- a Sale Deed whose recital names a
    # prior document that nothing else submitted evidences must surface a
    # flag out of the full run_verification() call.
    deed = SaleDeed(
        document_number="D2",
        prior_title_deed_references=["Document No. 1123/1998, dated 02-06-1998"],
    )
    flags = run_verification([deed], revenue_record=None, ec=None)
    assert any("not independently evidenced" in f.issue for f in flags)
