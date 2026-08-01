# Land Title Due Diligence Engine

A standalone Python pipeline for Indian land-title due diligence: OCR intake →
structured extraction (Sale Deed / Encumbrance Certificate / Revenue Record) →
deterministic cross-document verification → RAG-grounded legal citations →
draft opinion generation with a mandatory human-review gate.

This is a **separate project**, not integrated with the InstaLego Spring Boot
app — it's built and run independently.

## Architecture

```
src/landtitle/
  config.py                  # model name, LLM_API_BASE_URL, thresholds, paths
  schemas.py                 # pydantic schemas for all extracted/derived data
  ocr/intake.py               # pypdf + pdf2image/pytesseract (eng+tel), per-page chunking
  llm/client.py                # QwenClient: HTTP client for an OpenAI-compatible /v1/chat/completions endpoint
  extraction/
    sale_deed.py              # flat-text LLM extraction
    revenue_record.py         # flat-text LLM extraction
    ec_tables.py               # table-structure extraction (pdfplumber / OpenCV grid + OCR)
    encumbrance.py             # orchestrates table-aware EC extraction
  verification/checks.py       # pure-code cross-document checks (NO LLM)
  legal/
    corpus_builder.py          # footnote-safe section chunking of the 3 Acts
    retrieval.py                # hybrid exact-section + FAISS semantic retrieval
  opinion/
    generator.py                # draft opinion generation + review gate
    pdf_export.py                # fpdf2 export, drafts are always watermarked
  pipeline.py                  # end-to-end orchestration + CLI
```

## Validated decisions (do not change without strong reason)

- **Model**: Qwen2.5-7B-Instruct, 4-bit quantized (nf4, double quant). Tested
  against 14B on the same tasks — identical errors, confirming 7B is
  sufficient and errors are prompting/architecture issues, not a model
  capability ceiling.
- **Approach**: pure RAG, no fine-tuning.
- **Embeddings**: BAAI/bge-small-en-v1.5. **Vector store**: FAISS
  (IndexFlatL2) — chosen over ChromaDB after a real dependency conflict
  (chromadb's opentelemetry deps broke on the target environment).

## How this connects to the LLM

`QwenClient` (`src/landtitle/llm/client.py`) does **not** load any model
weights in this process. It makes an HTTP POST to
`{LLM_API_BASE_URL}/v1/chat/completions` (note: `LLM_API_BASE_URL` itself
should be the bare origin, e.g. `https://your-id.ngrok-free.dev` — do NOT
include a trailing `/v1`, the client appends that path itself):

```jsonc
// Request body — the system prompt and user prompt passed to generate()
// are combined into a single "user" message; there is no "model" field.
{
  "messages": [
    {"role": "user", "content": "<system_prompt>\n\n<user_prompt>"}
  ],
  "temperature": 0.1,
  "max_tokens": 2048
}
```
```jsonc
// Expected response body
{
  "choices": [
    {"message": {"role": "assistant", "content": "..."}}
  ]
}
```

Headers sent: `Content-Type: application/json`, `ngrok-skip-browser-warning:
true` (harmless if your server isn't behind ngrok; prevents ngrok's free-tier
HTML interstitial page from being returned instead of JSON on the first hit
from a new client), and `Authorization: Bearer {LLM_API_KEY}` **only if**
`LLM_API_KEY` is set — omitted entirely otherwise.

Environment variables:
- `LLM_API_BASE_URL` — **required, no default.** `QwenClient()` raises
  `RuntimeError` immediately at construction if this is unset or empty —
  it will never silently fall back to localhost or to loading a model
  in-process (there is no in-process loading code left in this project).
- `LLM_API_KEY` — optional; sent as a Bearer token only if set.
- `LLM_API_TIMEOUT` — request timeout in seconds; defaults to `120`.

This is a small custom contract (single combined user message, no `model`
field), matched to a hand-written FastAPI server — not full OpenAI-API
compatibility. If you later swap to vLLM's or text-generation-webui's
built-in OpenAI-compatible server, you'd need to adjust `generate()` to send
separate `system`/`user` messages and a `model` field, since those expect
the standard OpenAI shape.

## Setup

```bash
pip install -r requirements.txt
```

`requirements.txt` no longer includes `torch`/`transformers`/`bitsandbytes`
for the LLM itself — those only matter on whichever machine is actually
serving the model (e.g. your Kaggle notebook). `sentence-transformers` (used
only for the legal-corpus embeddings, not the LLM) still pulls in `torch`
transitively.

For OCR you also need the Tesseract binary installed system-wide (with the
Telugu language pack) and Poppler (for `pdf2image`) — neither ships via pip:
- Windows: install Tesseract from the UB-Mannheim build (includes language
  packs), and Poppler for Windows; add both to PATH.
- The `eng+tel` language string requires `tel.traineddata` to be present in
  Tesseract's `tessdata` directory.

### Legal corpus — you must supply the source text yourself

`data/acts/` ships empty. **This project does not include or fabricate the
text of the Transfer of Property Act 1882, Registration Act 1908, or
Limitation Act 1963.** Download the official text of each from
indiacode.nic.in and save them as plain text at the paths in
`config.ACTS`:

```
data/acts/transfer_of_property_act_1882.txt
data/acts/registration_act_1908.txt
data/acts/limitation_act_1963.txt
```

Then build the FAISS index:

```python
from landtitle.config import ACTS, EMBEDDING_MODEL, FAISS_INDEX_PATH, CORPUS_METADATA_PATH
from landtitle.legal.corpus_builder import build_and_save_corpus
build_and_save_corpus(ACTS, EMBEDDING_MODEL, FAISS_INDEX_PATH, CORPUS_METADATA_PATH)
```

Fabricating statute text or citing a section number an LLM merely
*recalls* would be exactly the kind of confident fabrication this whole
project exists to prevent — see the citation-labeling logic in
`legal/retrieval.py`.

## Running it

```bash
python -m landtitle.pipeline \
  --sale-deed path/to/deed1.pdf --sale-deed path/to/deed2.pdf \
  --revenue-record path/to/pahani.pdf \
  --ec path/to/encumbrance_certificate.pdf \
  --output opinion_draft.pdf
```

Sale deeds should be passed in chronological order (oldest first) so the
chain-of-title check can compare consecutive owners correctly.

The output PDF is always stamped `DRAFT -- NOT REVIEWED -- NOT FOR
RELIANCE` unless you pass `--reviewed-by "Name"`, which calls
`finalize_opinion()` before export. **There is no other way to clear the
draft flag** — see `opinion/generator.py::finalize_opinion`. This mirrors
the product's non-negotiable requirement: this tool is a first-draft
accelerator for a licensed practitioner, never an autonomous opinion
generator.

## Logging and resilience

Set `LOG_LEVEL` (default `INFO`) for progress visibility during a run — each
extraction chunk, OCR page, verification result, legal retrieval, and
opinion-generation call logs a line, since a real run makes several slow,
sequential LLM calls against a personal Kaggle+ngrok endpoint with a history
of transient failures. `run_pipeline()` also logs which of its 4 stages
(extraction / verification / legal retrieval / opinion generation) it's in,
so a crash's traceback is preceded by *where* it happened.

`QwenClient.generate()` retries a transient 5xx server response (confirmed
live: the user's own FastAPI server returned one transient 500 while the
model was still loading, which succeeded on immediate retry) up to
`LLM_TRANSIENT_RETRY_ATTEMPTS` times (default 2) with a fixed backoff. A 4xx
is never retried — the request itself is wrong, not the server's state.

The CLI (`main()`) validates that all provided file paths exist before
starting the (slow, LLM-driven) pipeline, and missing Tesseract/Poppler
binaries now raise a clear `RuntimeError` pointing back to this README's
Setup section, instead of surfacing pytesseract's/pdf2image's raw exception.
`QwenClient.generate()` also retries a plain timeout the same way (not just
5xx) — confirmed live that a real call once timed out at exactly the
(previous, 120s) limit, likely from a second concurrent request queued
behind it on the user's single Kaggle server; `LLM_API_TIMEOUT` defaults to
240s now.

## Determinism

`LLM_TEMPERATURE` stays at **0.1 — do not lower it to 0.0.** Confirmed live
(2026-07-29) that the same document extracted twice can produce different
results (a prior-title-reference document number captured one run,
seemingly dropped the next) — undermining any check whose correctness
depends on that field being reliably extracted. Setting temperature to 0.0
(greedy decoding, the standard fix for exactly this) was tried and directly
confirmed, via curl probes with an otherwise-identical payload, to make the
user's real Kaggle/Qwen FastAPI server return a hard HTTP 500 on every
single request — their generation code almost certainly doesn't handle
true-zero-temperature sampling gracefully. 0.1 is the lowest value confirmed
safe against their actual server. **Do not lower this again without
re-probing the live server first with a raw curl call**, exactly like the
probe that caught this — a passing test suite would not have caught it,
since nothing here talks to a real server.

A `seed` field (`LLM_SEED`, default `42`) is now sent on every request as a
lower-risk partial mitigation for the same goal — confirmed live that the
server tolerates an unrecognized field without error, but whether it
*actually* increases determinism depends entirely on whether the server's
own generation code reads and applies it, which is outside this project's
control. Treat determinism as reduced, not solved: any check depending on
a single extraction call being accurate (e.g. `unevidenced_prior_reference_
check`) can still occasionally miss a real issue that a repeat run would
catch — this is a known, open risk, not a bug with a clean fix available
right now.

## Key confirmed findings baked into this code (see inline docstrings)

- **EC extraction must be table-aware.** Feeding flattened OCR/prose text of
  an Encumbrance Certificate to the LLM causes it to confuse Executants
  (sellers) and Claimants (buyers), especially on wrapped multi-line rows.
  `extraction/ec_tables.py` extracts table structure and pre-labels each
  field *before* the LLM ever sees it.
- **`represented_by_gpa` is a separate field** on `Party` — without it the
  model conflates a GPA holder's identity with the actual vendor.
- **`land_extent` and `built_up_area` are separate fields** — a combined
  field caused the model to return the wrong figure for the wrong unit.
- **Cross-document verification is pure code, never LLM judgment** — an LLM
  asked to freely identify "risk flags" repeatedly treated normal
  chain-of-title events (different parties at different times) as false
  positives. `verification/checks.py` implements narrow, deterministic
  checks instead, and the opinion-generation prompt is instructed to report
  *only* flags this layer actually found.
- **Resolved gap: a Sale Deed's own recital can narrate a prior acquisition
  (`prior_title_deed_references`) that nothing submitted actually backs
  up** — e.g. "Document No. 1123/1998" named in the recital but never
  itself supplied as a Sale Deed or EC entry. `verification/checks.py`'s
  `unevidenced_prior_reference_check()` now flags (low severity) any such
  reference that isn't matched by another submitted Sale Deed's
  `document_number` or an Encumbrance Certificate entry.
- **Partially mitigated, NOT fully solved: the opinion model invents
  transferor/transferee names for prior transactions it was never given
  party data for.** `prior_title_deed_references` is a bare document
  number/date self-reported by the current deed's own recital — no
  separate party extraction exists for whatever transaction it refers to.
  Confirmed live, twice, with two different fabricated names: the model
  still wrote a specific "X transferred to Y" sentence for such a
  reference — once reusing a relation-clause name (a seller's own father)
  for a transaction the source actually described as a family *partition*,
  not a purchase from a named vendor; once inventing an unrelated name
  entirely to paper over a genuine chain-of-title gap and make a broken
  chain read as continuous. `opinion/generator.py`'s
  `_ensure_prior_reference_parties_not_claimed()` unconditionally appends a
  disclaimer whenever `prior_title_deed_references` is non-empty, and the
  `SYSTEM_PROMPT` now has an explicit rule against this — but neither
  *removes* the model's own fabricated sentence from the narrative (parsing
  and stripping a specific false claim from arbitrary free text reliably is
  far more fragile than this project's established append-only pattern).
  The final opinion can still contain the model's misleading prose
  alongside the deterministic correction; a reviewer has everything needed
  to catch the problem, but the narrative itself isn't cleaned up. Treat
  this as an open, harder problem for future work, not resolved.
- **A second variant of the party-relationship confusion above: the model
  invents a family relationship between the deed's own transacting parties.**
  Confirmed live: given a Seller and Buyer whose own `relation` fields each
  name a *different*, non-transacting relative (i.e. genuinely unrelated to
  each other), the opinion model wrote that the Buyer acquired the property
  "from his father," misreading the Buyer's own relation clause (naming the
  Buyer's actual father, not a party to the deed) as describing the Seller.
  This happened despite an existing prompt rule against exactly the
  sibling issue (reusing a relation-clause name as if it were a party) —
  confirming again that a prompt rule alone doesn't reliably prevent a new
  shape of the same underlying mistake. Fixed with
  `_ensure_no_unverified_relationship_claims()` in `opinion/generator.py`
  (same append-only pattern as this module's other `_ensure_*` guards): for
  each deed with both a Seller and Buyer, if no party's own relation clause
  names the other transacting party, an unconditional disclaimer is
  appended stating no relationship is evidenced between them; if one
  genuinely does (a real family sale), the actual relationship is restated
  as verified ground truth instead. Live-verified against the project's
  standing `deed_A_2005_standalone.txt` fixture: the disclaimer fired
  correctly even on a run where the model's own prose happened not to
  fabricate the relationship — confirming the guard is deterministic and
  doesn't depend on re-observing the bug.
- **`owner_chain_continuity_check()` was rewritten -- its prior sort-by-
  registration_date + compare-adjacent-pairs design was order-dependent on
  submission order whenever two deeds shared a registration_date, which
  produced a wrong-pairing false positive.** Confirmed live with this
  project's own 3-deed synthetic test set (`deed_A_2005_standalone.txt` +
  `deed_B_2012_chainlink.txt`, both dated 15-07-2012 -- a real, expected
  same-day registration, not a data error): a tie made Python's stable
  sort fall back to input order, and depending on that order the check
  could compare Deed B's buyer against Deed A's seller -- the wrong pair
  entirely -- while the real, correct A->B link (which both deeds' own
  recitals agree on) went unchecked. Rewritten with two strategies per
  deed, tried in order: (1) if the deed's own `prior_title_deed_references`
  resolves by document number to another submitted deed, that's its
  self-declared predecessor -- compare that specific deed's buyers against
  this deed's sellers, sidestepping any date tie entirely; (2) otherwise,
  compare against the buyer(s) of the WHOLE SET of submitted deeds with a
  strictly earlier date (not one picked "predecessor"), so an internal tie
  within that set is harmless. A second real gap surfaced while live-
  testing this fix: a deed's own text can state an execution date without
  ever separately stating a registration date for that specific document
  number, so `registration_date` can legitimately be null even though the
  deed is real and correctly extracted -- strategy 2's date key now falls
  back to the year embedded in the deed's own `document_number` (the
  standard Indian "NNNN/YYYY" format) when `registration_date` doesn't
  parse. That fallback is deliberately compared at YEAR granularity only
  when mixed with an exact date (never naively as a full `datetime`, which
  would make a year-only estimate defaulting to January 1st look falsely
  "earlier" than a same-year deed with a real, later-in-the-year date) --
  confirmed live this exact naive version produced a new false positive
  before being caught and fixed the same session. Verified end-to-end
  against the real 3-deed set: exactly one correct high-severity flag,
  attributed to the genuinely broken deed, with the clean links quiet.
- **A Sale Deed's own recital narrating how its seller acquired title is
  often NOT a sale -- confirmed live it can be a Partition Deed, and the
  extraction schema had no guidance to preserve that.**
  `SaleDeed.prior_title_deed_references` had no field description and
  `extraction/sale_deed.py`'s prompt had no rule about instrument type, so
  a real recital reading "...acquired the same by way of family partition,
  evidenced by a registered Partition Deed bearing Document No. 331/1988"
  was extracted as a bare "Document No. 331/1988, dated 12-01-1988" --
  losing the instrument type -- and the opinion narrative then defaulted to
  describing it as a "Sale Deed" in at least one observed run. Fixed by
  adding an explicit field description and a new extraction prompt rule
  requiring the instrument type be captured verbatim alongside the number
  and date, never defaulted to "Sale Deed". The field description was
  deliberately written WITHOUT a realistic-looking example document number/
  date (describes the required *shape*, not a copyable value) -- this
  project already has one confirmed case of a small model copying a
  plausible example value from a field description verbatim (see
  `SaleDeed.land_extent` below); repeating that mistake here, using this
  exact real fixture's own numbers, was caught and corrected before
  landing. Verified live: the same fixture now correctly extracts
  "Partition Deed bearing Document No. 331/1988, dated 12-01-1988".
- **A second variant of the party-relationship confusion above, this time
  across a submitted MULTI-deed set: the narrative stated one deed's own
  document number alongside a DIFFERENT deed's date.** Confirmed live: the
  narrative said Document No. 3010/2012 "was registered on 16-07-2012" --
  but that deed's own text never states that date; 16-07-2012 is a
  different, unrelated deed's own (already independently incorrect) recital
  date. Mitigated (same append-only pattern, not a full fix -- consistent
  with this module's stance throughout) with
  `_ensure_verified_deed_registration_details_present()` in
  `opinion/generator.py`: guarantees each deed's own correct
  registration_date is present in the narrative somewhere, without editing
  the model's own (possibly wrong) sentence.
- **Footnote/amendment lines must be filtered before section-splitting** the
  source Acts, or they get misidentified as section boundaries and corrupt
  the corpus (`legal/corpus_builder.is_footnote_line`).
- **Section citations are injected as verified labels**, extracted
  programmatically from each retrieved chunk — never left to the LLM to
  recall from memory, which was confirmed to fabricate plausible-sounding
  wrong section numbers even against a clean corpus.
- **Citation *number* fabrication is prevented by design, but citation
  *relevance* is not verified separately — confirmed live with a real,
  non-fabricated but arguably inapposite citation.** An opinion cited
  "Transfer of Property Act, Section 30" ("Prior disposition not affected
  by invalidity of ulterior disposition") in support of a routine
  chain-of-title gap — checked against `data/acts/transfer_of_property_
  act_1882.txt` and the built corpus, the section number and text are both
  real (not invented), so the citation-label guarantee above held. But
  Section 30 is actually about the validity of a *conditional/contingent*
  disposition (e.g. "transfer to B for life, and if she doesn't do X, then
  to C") — a different legal concept from an undocumented prior title deed.
  Likely cause: `pipeline.select_citations()` uses each flag's `issue` text
  directly as a semantic search query, and the flag
  `"Prior title reference not independently evidenced"` shares enough
  surface wording with the section text ("**Prior** **disposition**...")
  to rank as a semantic match despite the legal concepts being unrelated.
  Not fixed — flagged here as a real, open retrieval-relevance gap distinct
  from the citation-fabrication problem this project already solved; a
  reviewer needs to sanity-check that cited sections are actually
  *applicable*, not just real.
  **Two more real citations checked the same way, on the live 3-deed run**:
  "Registration Act, Section 6" (real text: appointment of Registrars/
  Sub-Registrars -- purely administrative, essentially no relevance to a
  specific title opinion; the compliance narrative used it to support "the
  registration of the sale deeds," which Section 6 doesn't actually speak
  to) and "Transfer of Property Act, Section 3" (real text: the Act's
  interpretation/definitions clause -- plausibly citable generically when
  discussing terms like "immoveable property" or "registered," a weaker but
  not unreasonable use, consistent with the user's own earlier read of
  Section 3 as legitimate). Both are further evidence for the same
  systemic retrieval-relevance gap above, not new fabrication findings —
  no fix attempted this session beyond documenting the pattern.

## Web API (local)

`src/landtitle/api/` is a thin FastAPI wrapper around `run_pipeline()` +
`export_opinion_pdf()` above -- it does not change any pipeline logic, it
only calls into it. Because a real run makes several slow, sequential LLM
calls, submission is asynchronous: `POST` a job, then poll it, then
download the PDF once it's done.

Run it locally:

```bash
uvicorn landtitle.api.app:app --reload
```

Then open **http://127.0.0.1:8000/** in a browser -- FastAPI serves the
static frontend (`frontend/index.html`, plain HTML/JS, no build step) at
the app's own root path via a `StaticFiles` mount, so the page's `fetch()`
calls hit `/opinions` on the same origin with no separate server or CORS
setup needed.

### Exposing the backend to a Vercel-hosted frontend (ngrok)

If the frontend is instead deployed separately (e.g. on Vercel) while this
FastAPI backend stays local, expose port 8000 with ngrok in a second
terminal, alongside the `uvicorn` command above:

```bash
ngrok http 8000
```

ngrok prints a forwarding URL like `https://<random-id>.ngrok-free.app` --
use that as the frontend's backend base URL. (One-time setup on this
machine, not covered here: `ngrok config add-authtoken <your token>`.)

Because the frontend then calls this API cross-origin, CORS is enabled via
`fastapi.middleware.cors.CORSMiddleware`, configurable with the
`LANDTITLE_CORS_ORIGINS` env var (comma-separated origins; defaults to `*`
for now -- tighten this to the actual `https://*.vercel.app` origin once
known, since this default has no auth/session state to protect against but
is still worth narrowing).

Endpoints:
- `POST /opinions` -- multipart upload. Fields: `sale_deed` (one or more
  files, repeat the field for each; chronological order, oldest first),
  `revenue_record` (optional, single file), `ec` (optional, single file).
  Returns `{"job_id": ..., "status": "pending"}` immediately (HTTP 202).
- `GET /opinions/{job_id}` -- job status: `pending` / `running` / `done` /
  `failed` (with an `error` string on failure). Metadata only (file counts,
  booleans, timestamps) -- never document content.
- `GET /opinions/{job_id}/download` -- the generated PDF once `status` is
  `done` (409 before that, 410 if its retention window has already elapsed).
- `POST /generate-opinion` -- simpler synchronous alternative to the
  `/opinions` job flow: same multipart fields (`sale_deed` one-or-more
  required, `revenue_record` optional, `ec` optional), but this call blocks
  until the pipeline finishes (can take several minutes) and returns the
  generated opinion PDF directly (`application/pdf`) instead of a job id.
  On failure it returns HTTP 500 with `{"detail": "..."}` (a short, safe
  message -- never a raw traceback); the full exception is logged
  server-side only.

Required env vars are the same ones the CLI pipeline already needs (see
"Environment variables" above under "How this connects to the LLM") --
`LLM_API_BASE_URL` at minimum, set via `.env` at the repo root. If it's
unset, job creation still succeeds (the API doesn't require it up front),
but every submitted job will fail fast with `QwenClient`'s `RuntimeError` as
its recorded error -- this is expected, not a bug in the API layer.

Job/file handling: each job gets its own directory under the OS temp dir
(`tempfile.mkdtemp`, never inside this repo); uploaded PDFs are deleted the
moment the pipeline has consumed them (success or failure), and the
generated opinion PDF is deleted by a background reaper once
`LANDTITLE_JOB_RETENTION_SECONDS` (env var, default 1800 = 30 min) has
elapsed since the job finished. Jobs are tracked in an in-memory dict,
single-process -- restarting the server loses job history (by design, for
this first cut). Hosting/deployment/Docker is explicitly out of scope here;
this section is local-run only.

## What has and hasn't been run in this environment

137 unit tests pass (`python -m pytest tests/`), covering every pure-logic
layer: verification checks (including multi-deed chain continuity), footnote
filtering/section splitting, EC table flattening, party dedup/merge across
OCR-garbled chunks, opinion-generator flag/transaction-omission guards, PDF
export + review gate, and `QwenClient`'s HTTP contract against a mocked
`requests.post`.

Beyond unit tests, this has also been run against real user documents (a
real scanned 15-page Sale Deed, a real Pahani revenue record, and a real
Encumbrance Certificate) through a real Qwen2.5-7B endpoint: OCR, extraction,
verification, legal retrieval, and opinion generation have each been
exercised and had real bugs found and fixed this way (see inline docstrings
in `pipeline.py` and `opinion/generator.py` for the specific failure modes).

`scripts/trap_test.py` runs Layer 3 (`verification/checks.py`) against the
real cached extraction from that same real document set, deliberately
injecting one known inconsistency at a time (extent mismatch, a boundary
mismatch, an unreleased mortgage, a litigation entry, a genuine chain-of-title
gap) and confirming each is caught — plus confirming untouched real fields
and a released mortgage correctly stay quiet. 9/9 checks behaved as expected.

**Still not run** — these remain open before calling this production-ready:
- The scanned-EC OpenCV grid-detection path (real EC testing so far has only
  covered plain-text/hand-typed tables).
- Wider extraction testing across different registrars, eras, and scan
  qualities.
- GPU load-testing on whatever hardware actually serves the model — a
  15-page OCR'd document OOM'd on a 14.56GB T4 at full size in prior testing;
  this pipeline chunks via `ocr/intake.chunk_text` instead of truncating, but
  that strategy itself hasn't been load-tested.
- The LLM backend today is a Kaggle notebook tunneled via ngrok (URL changes
  every restart) — not viable as-is for a hosted deployment; a persistent
  inference endpoint needs to be decided before going live.
