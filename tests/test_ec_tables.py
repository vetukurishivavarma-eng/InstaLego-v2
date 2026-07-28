from landtitle.extraction.ec_tables import build_column_mapping, flatten_table, looks_like_header_row


def test_header_row_detected():
    header = ["Sl.No", "Document No.", "Date", "Nature", "Executants", "Claimants", "Consideration"]
    assert looks_like_header_row(header)


def test_non_header_row_not_detected():
    row = ["1", "1234/2001", "01-01-2001", "Sale Deed", "Ramesh Kumar", "Suresh Babu", "500000"]
    assert not looks_like_header_row(row)


def test_column_mapping_maps_expected_columns():
    header = ["Sl.No", "Document No.", "Date", "Nature", "Executants", "Claimants", "Consideration"]
    mapping = build_column_mapping(header)
    assert mapping[1] == "document_number"
    assert mapping[2] == "date"
    assert mapping[3] == "nature_of_document"
    assert mapping[4] == "executants_sellers"
    assert mapping[5] == "claimants_buyers"
    assert mapping[6] == "consideration"


def test_flatten_table_produces_prelabeled_rows_not_a_single_blob():
    table = [
        ["Sl.No", "Document No.", "Date", "Nature", "Executants", "Claimants", "Consideration"],
        ["1", "1234/2001", "01-01-2001", "Sale Deed", "Ramesh Kumar\nS/o Venkaiah", "Suresh Babu", "500000"],
        ["2", "5678/2010", "05-05-2010", "Gift Deed", "Suresh Babu", "Lakshmi Devi", "0"],
    ]
    rows = flatten_table(table)
    assert len(rows) == 2

    row1_text = rows[0].to_labeled_text()
    assert "Executants (Sellers): Ramesh Kumar S/o Venkaiah" in row1_text
    assert "Claimants (Buyers): Suresh Babu" in row1_text
    # Executant and claimant must never end up on the same labeled line —
    # this is exactly the confusion this module exists to prevent.
    assert row1_text.count("Executants (Sellers)") == 1
    assert row1_text.count("Claimants (Buyers)") == 1

    row2_text = rows[1].to_labeled_text()
    assert "Executants (Sellers): Suresh Babu" in row2_text
    assert "Claimants (Buyers): Lakshmi Devi" in row2_text


def test_flatten_table_falls_back_to_canonical_order_without_header():
    table = [
        ["1", "1234/2001", "01-01-2001", "Sale Deed", "Ramesh Kumar", "Suresh Babu", "500000"],
    ]
    rows = flatten_table(table)
    assert len(rows) == 1
    text = rows[0].to_labeled_text()
    assert "Executants (Sellers): Ramesh Kumar" in text
    assert "Claimants (Buyers): Suresh Babu" in text
