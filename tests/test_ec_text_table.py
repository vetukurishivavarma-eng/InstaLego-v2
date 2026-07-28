from landtitle.extraction.ec_tables import flatten_table, parse_fixed_width_text_table

# A trimmed version of the real hand-typed EC fixture used to validate this
# pipeline: whitespace-aligned columns (not a real PDF table), including the
# deliberately tricky case of a GPA reference wrapping across two
# continuation lines, a claimant continuation line, and "NIL" placeholder
# rows that are narrative remarks rather than real transactions.
_FIXTURE_TEXT = """\
Sl.No.  Document No.    Date        Nature of Document     Executant(s)                Claimant(s)                 Consideration
1.      2609/2000       09-11-2000  Sale Deed               M/s Sunrise Realty          Kaushalya Venkataramanian Rs. 42,00,000/-
                                                                                          & S. Suresh (undivided share)

2.      2610/2000       09-11-2000  Sale Deed               M/s Sunrise Realty          R. Venkataramanian          Rs. 38,50,000/-
                                                                                          (undivided share)

3.      400/2012        07-03-2012  Sale Deed               Kaushalya Venkataramanian, Priyanka Rajaraman          Rs. 1,70,00,000/-
                                     (through GPA holder      S. Suresh, R. Venkataramanian
                                     R. Venkataramanian)

4.      NIL             --          No mortgage entries found for this property in the search period.

5.      NIL             --          No court attachment or litigation entries found for this property in the search period.
"""


def test_parses_three_real_rows_and_drops_nil_placeholders():
    rows = parse_fixed_width_text_table(_FIXTURE_TEXT)
    assert len(rows) == 3
    doc_numbers = [r[1] for r in rows]  # canonical order: sl_no, document_number, ...
    assert doc_numbers == ["2609/2000", "2610/2000", "400/2012"]


def test_claimant_continuation_line_attributed_to_claimants_not_executants():
    rows = parse_fixed_width_text_table(_FIXTURE_TEXT)
    row1 = rows[0]
    # canonical order: sl_no, document_number, date, nature, executants, claimants, consideration
    executants, claimants = row1[4], row1[5]
    assert "S. Suresh" in claimants
    assert "S. Suresh" not in executants
    assert "Kaushalya Venkataramanian" in claimants


def test_gpa_continuation_attributed_to_executants_not_claimants():
    """The deliberately tricky case: a GPA reference wraps across two
    continuation lines whose indentation does not line up with any header
    column. This must still land under Executants (Sellers), never
    Claimants (Buyers) — conflating the two is exactly the confirmed bug
    ec_tables.py exists to prevent."""
    rows = parse_fixed_width_text_table(_FIXTURE_TEXT)
    row3 = rows[2]
    executants, claimants = row3[4], row3[5]
    assert "GPA holder" in executants
    assert "S. Suresh" in executants
    assert claimants.strip() == "Priyanka Rajaraman"
    assert "GPA" not in claimants


def test_nil_rows_excluded_from_flattened_output():
    rows = parse_fixed_width_text_table(_FIXTURE_TEXT)
    labeled = flatten_table(rows)
    assert len(labeled) == 3
    all_text = " ".join(lr.to_labeled_text() for lr in labeled)
    assert "mortgage entries found" not in all_text.lower()
    assert "litigation entries found" not in all_text.lower()


def test_no_recognizable_header_returns_empty():
    assert parse_fixed_width_text_table("just some random prose\nwith no table at all") == []
