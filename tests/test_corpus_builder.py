from landtitle.legal.corpus_builder import is_footnote_line, split_into_sections, strip_footnotes


def test_footnote_line_with_subs_detected():
    assert is_footnote_line("1. Subs. by Act 39 of 1948, s. 5, for s. 88.")


def test_footnote_line_with_omitted_by_detected():
    assert is_footnote_line("2. The words ‘’ omitted by Act 48 of 2001, s. 10.")


def test_real_section_start_not_a_footnote():
    assert not is_footnote_line("55. Rights and liabilities of buyer and seller.")


def test_plain_prose_line_not_a_footnote():
    assert not is_footnote_line("In the absence of a contract to the contrary, the buyer is bound to disclose.")


def test_strip_footnotes_removes_only_footnote_lines():
    text = (
        "55. Rights and liabilities of buyer and seller.\n"
        "1. Subs. by Act 39 of 1948, s. 5, for s. 88.\n"
        "In the absence of a contract to the contrary, the buyer is bound to disclose."
    )
    cleaned = strip_footnotes(text)
    assert "Subs. by Act 39" not in cleaned
    assert "55. Rights and liabilities" in cleaned
    assert "bound to disclose" in cleaned


def test_split_into_sections_ignores_footnote_as_boundary():
    text = (
        "54. Sale defined.\n"
        "Sale is a transfer of ownership in exchange for a price paid.\n"
        "1. Subs. by Act 39 of 1948, s. 5, for s. 88.\n"
        "55. Rights and liabilities of buyer and seller.\n"
        "In the absence of a contract to the contrary, the buyer is bound to disclose."
    )
    sections = split_into_sections(text, "Transfer of Property Act")
    section_numbers = [s.section_number for s in sections if s.section_number]
    # Without footnote filtering, "1." would be misread as a section boundary
    # splitting section 54 apart from its own body text.
    assert "1" not in section_numbers
    assert section_numbers == ["54", "55"]
    sec_54 = next(s for s in sections if s.section_number == "54")
    assert "transfer of ownership" in sec_54.text
    assert "Subs. by Act" not in sec_54.text


def test_split_into_sections_extracts_verified_number_from_chunk_itself():
    text = "88. Some heading.\nBody text here."
    sections = split_into_sections(text, "Registration Act")
    assert len(sections) == 1
    assert sections[0].section_number == "88"
