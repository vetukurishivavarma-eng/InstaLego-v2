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
- **Footnote/amendment lines must be filtered before section-splitting** the
  source Acts, or they get misidentified as section boundaries and corrupt
  the corpus (`legal/corpus_builder.is_footnote_line`).
- **Section citations are injected as verified labels**, extracted
  programmatically from each retrieved chunk — never left to the LLM to
  recall from memory, which was confirmed to fabricate plausible-sounding
  wrong section numbers even against a clean corpus.

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

72 unit tests pass (`python -m pytest tests/`), covering every pure-logic
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
- **A full end-to-end run with 2+ real, chronologically-ordered Sale Deeds**,
  to exercise `owner_chain_continuity_check()` through the actual pipeline
  (OCR + extraction), not just its unit tests and `trap_test.py`'s synthetic
  second deed. Only one real historical deed has been sourced so far.
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
