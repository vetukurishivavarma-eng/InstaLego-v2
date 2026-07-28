# Negative-case fixtures

These are real, uploadable `.txt` fixture files that exercise two specific
cross-document verification checks in `src/landtitle/verification/checks.py`
end-to-end through the actual pipeline (text-intake -> extraction ->
verification) -- as opposed to `scripts/trap_test.py`, which proves the same
checks work but only by mutating cached Python objects in memory. Nothing in
this directory duplicates or replaces that script; it closes the "no real
fixture file" gap next to it.

Both files are plain `.txt` so that `src/landtitle/ocr/intake.py`'s
`extract_document()` passes them straight through as pre-extracted text (see
its docstring) -- no OCR, no PDF parsing, no dependency on Poppler/Tesseract
being installed.

## 1. `revenue_record_extent_mismatch.txt`

**Demonstrates:** a Revenue Record (Pahani) whose recorded land extent is
substantially different from the real cached Sale Deed's extent for the same
property.

**Check exercised:** `verification/checks.py::extent_match()`.

**Key deliberately-wrong value:** `Total Extent : 150.00 Sq. Yards` in this
fixture, versus the real cached Sale Deed's `329.61 square yards` (see
`scripts/_debug_opinion_cache.pkl`). The difference is ~54.5%, far exceeding
`_EXTENT_TOLERANCE_RATIO` (2%).

**Expected flag once run through the real pipeline:**
- `issue`: "Land extent mismatch between Sale Deed and Revenue Record"
- `severity`: `medium`
- `detail`: reports both extents (~1350 sq ft vs ~2966.5 sq ft) and that the
  difference exceeds the 2% tolerance.

All other fields (survey number `82/A/4`, boundaries, etc.) are modeled on
the real `sample_pahani_revenue_record.txt` reference for realism, but with
fictional party names (`Smt. Lakshmi Prasanna Devarakonda` etc.) -- none of
the real document's actual names are reused.

**Not independently structurally testable beyond intake passthrough**:
Revenue Record extraction (`extraction/revenue_record.py`) is flat-text LLM
extraction with no pure-code parsing step in front of it, so there is no
pure-code path to validate here beyond confirming the file exists, reads
correctly, and contains the intended value. See
`tests/test_negative_case_fixtures.py`.

## 2. `ec_unreleased_mortgage.txt`

**Demonstrates:** an Encumbrance Certificate containing a genuine
"Simple Mortgage" entry with no later release/discharge entry in the same
EC, alongside otherwise-clean sale deed entries.

**Check exercised:** `verification/checks.py::active_encumbrance_check()`.

**The mortgage entry:**
- Document No. `550/2015`, dated `10-08-2015`
- Nature of Document: `Simple Mortgage`
- Executant (mortgagor): `Fictional Ananya Devi Reddy` (the owner per the
  prior GPA sale deed, document 812/2013)
- Claimant (mortgagee): `Fictional Cooperative Bank Ltd. (Mortgagee)`
- Consideration: `Rs. 25,00,000/-`
- No later entry in the EC references release/discharge/satisfaction of this
  mortgage.

**Expected flag once run through the real pipeline:**
- `issue`: "Unreleased mortgage entry in Encumbrance Certificate"
- `severity`: `high`
- `detail`: "Document No. 550/2015 dated 10-08-2015 (Simple Mortgage) has no
  later release/discharge entry in the same EC."

A trailing NIL litigation placeholder row ("No court attachment or
litigation entries found...") is included to confirm it is correctly dropped
by `parse_fixed_width_text_table()`/`flatten_table()` rather than tripping a
spurious litigation flag via keyword substring match.

**This fixture is higher-risk** because EC extraction depends on exact
whitespace-column alignment (`extraction/ec_tables.py::parse_fixed_width_text_table()`),
not flat-text LLM extraction. It has been structurally validated: parsing it
with `parse_fixed_width_text_table()` + `flatten_table()` produces exactly 4
real rows (the 5th NIL row is dropped), and the mortgage row's
executant/claimant are correctly separated, not scrambled. Running those
parsed rows through the real, unmodified `active_encumbrance_check()`
produces exactly one flag: the expected "Unreleased mortgage entry" flag
above, with no spurious extras. See `tests/test_negative_case_fixtures.py`
for the automated version of this validation.

## Running a fixture through the real CLI

Once a live LLM endpoint is configured (`LLM_API_BASE_URL` etc. in `.env`),
either fixture can be run through the real pipeline, e.g.:

```
python -m landtitle.pipeline --sale-deed <a real/existing sale deed> --revenue-record fixtures/negative_cases/revenue_record_extent_mismatch.txt --output test_output.pdf
```

or, for the EC fixture:

```
python -m landtitle.pipeline --sale-deed <a real/existing sale deed> --ec fixtures/negative_cases/ec_unreleased_mortgage.txt --output test_output.pdf
```

(Flags confirmed against `src/landtitle/pipeline.py`'s current argparse
setup: `--sale-deed` is repeatable/chronological, `--revenue-record` and
`--ec` each take a single path, `--output` defaults to `opinion_draft.pdf`.
Run `python -m landtitle.pipeline --help` to reconfirm against your
checkout.)

## Validation status

Both fixtures have been structurally validated (whitespace/table parsing for
the EC fixture; file presence and content for the Revenue Record fixture) as
of this writing. **Neither has been run through a live LLM extraction call.**
That final end-to-end confirmation -- actually invoking
`extraction/revenue_record.py::extract_revenue_record()` and the EC
LLM-cleanup step against a configured model endpoint -- is a deliberate
separate step for after this work lands, not something performed here. No
live network call to any `LLM_API_BASE_URL` was made in producing or
validating these fixtures.
