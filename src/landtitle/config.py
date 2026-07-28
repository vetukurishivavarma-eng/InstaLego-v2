"""Central configuration for the Land Title Due Diligence Engine.

Values here reflect the decisions validated in Kaggle notebooks — see the
project README before changing MODEL_NAME or EMBEDDING_MODEL.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")  # no-op if the file doesn't exist
DATA_DIR = PROJECT_ROOT / "data"
ACTS_DIR = DATA_DIR / "acts"
CORPUS_DIR = DATA_DIR / "corpus"

# --- LLM (validated: Qwen2.5-7B beats no capability difference vs 14B on these tasks) ---
# The model itself is NOT loaded in-process — QwenClient (llm/client.py) POSTs
# to {LLM_API_BASE_URL}/v1/chat/completions. Point LLM_API_BASE_URL at wherever
# that model is actually being served (e.g. a Kaggle notebook running a FastAPI
# server, exposed via ngrok/localtunnel).
#
# No hardcoded default on purpose: if this is unset, QwenClient must fail
# loudly at construction time (see llm/client.py), not silently fall back to
# some guessed localhost URL or — worse — to loading a model in-process (this
# project no longer even has that code).
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LLM_API_BASE_URL = os.environ.get("LLM_API_BASE_URL")
LLM_API_KEY = os.environ.get("LLM_API_KEY")  # optional; only sent as a Bearer token if set
LLM_API_TIMEOUT = float(os.environ.get("LLM_API_TIMEOUT", "120"))
LLM_TEMPERATURE = 0.1  # low temp reduces confident fabrication (extraction + opinion gen)
LLM_MAX_NEW_TOKENS = 2048

# --- Embeddings / vector store (validated) ---
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
FAISS_INDEX_PATH = CORPUS_DIR / "legal_corpus.faiss"
CORPUS_METADATA_PATH = CORPUS_DIR / "legal_corpus_meta.json"

# --- OCR ---
OCR_DPI = 300
OCR_LANG = "eng+tel"

# --- Chunking for large OCR'd documents (avoid the OOM-by-truncation failure mode) ---
# A 15-page OCR'd document OOM'd at full size on a 14.56GB T4 in testing; production
# must page/chunk rather than silently truncate to N pages.
MAX_CHARS_PER_EXTRACTION_CHUNK = 6000
CHUNK_OVERLAP_CHARS = 500

# --- Legal citation budget for opinion generation ---
# Confirmed live against the real Kaggle T4 server: 12 citations (~7.2k
# prompt tokens) and even 6/5 citations (~5.5k tokens) triggered a real
# server-side "GPU out of memory" error on this call — the single largest in
# the pipeline, since it also carries the full facts JSON. Bisected live:
# 4 citations (~5.4k tokens) succeeded but only ~2% below the 5-citation
# failure point — too tight a margin to trust as a default given GPU memory
# isn't perfectly deterministic run-to-run. 3 citations (~4.0k tokens)
# succeeded with clear headroom below every observed failure; used instead
# of the riskier edge value.
MAX_CITATIONS_PER_OPINION = 3
OPINION_MAX_NEW_TOKENS = 768

# --- Name similarity threshold for cross-document fuzzy matching ---
NAME_SIMILARITY_THRESHOLD = 85  # rapidfuzz token_sort_ratio, 0-100

ACTS = {
    "Transfer of Property Act": ACTS_DIR / "transfer_of_property_act_1882.txt",
    "Registration Act": ACTS_DIR / "registration_act_1908.txt",
    "Limitation Act": ACTS_DIR / "limitation_act_1963.txt",
}
