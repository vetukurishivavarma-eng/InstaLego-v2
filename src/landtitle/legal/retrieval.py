"""Hybrid legal retrieval: exact section-number match (when the query names
a specific section) + semantic search (FAISS) as fallback/supplement.

This pattern carries over from the SARFAESI module and is proven.

CRITICAL: every retrieved chunk carries a verified `citation_label` derived
directly from the corpus's own leading section-number pattern (set at
corpus-build time — see corpus_builder.split_into_sections). This label,
not the LLM's own recollection, is what must be injected into opinion-
generation prompts, since an LLM asked to "cite the relevant section" will
sometimes fabricate a plausible-sounding but wrong section number even when
the retrieved content is correct.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from landtitle.config import CORPUS_METADATA_PATH, EMBEDDING_MODEL, FAISS_INDEX_PATH

_QUERY_SECTION_PATTERN = re.compile(r"section\s+(\d+[A-Z]?)", re.IGNORECASE)


@dataclass
class RetrievedChunk:
    act: str
    section_number: str | None
    text: str

    @property
    def citation_label(self) -> str:
        if self.section_number:
            return f"[{self.act}, Section {self.section_number}]"
        return f"[{self.act}]"


class LegalRetriever:
    def __init__(self, index_path=FAISS_INDEX_PATH, metadata_path=CORPUS_METADATA_PATH, embedding_model=EMBEDDING_MODEL):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embedding_model_name = embedding_model
        self._index = None
        self._sections: list[dict] = []
        self._embedder = None

    def _ensure_loaded(self) -> None:
        if self._index is not None:
            return
        import faiss

        self._index = faiss.read_index(str(self.index_path))
        self._sections = json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def _ensure_embedder(self) -> None:
        if self._embedder is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._embedder = SentenceTransformer(self.embedding_model_name)

    def exact_section_lookup(self, section_number: str, act: str | None = None) -> list[RetrievedChunk]:
        self._ensure_loaded()
        matches = []
        for s in self._sections:
            if s.get("section_number") == section_number and (act is None or s.get("act") == act):
                matches.append(RetrievedChunk(act=s["act"], section_number=s["section_number"], text=s["text"]))
        return matches

    def semantic_search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        self._ensure_loaded()
        self._ensure_embedder()
        query_vec = self._embedder.encode([query], convert_to_numpy=True)
        distances, indices = self._index.search(query_vec, k)
        results = []
        for idx in indices[0]:
            if idx < 0 or idx >= len(self._sections):
                continue
            s = self._sections[idx]
            results.append(RetrievedChunk(act=s["act"], section_number=s.get("section_number"), text=s["text"]))
        return results

    def hybrid_retrieve(self, query: str, act: str | None = None, k: int = 5) -> list[RetrievedChunk]:
        results: list[RetrievedChunk] = []
        section_match = _QUERY_SECTION_PATTERN.search(query)
        if section_match:
            results.extend(self.exact_section_lookup(section_match.group(1), act=act))

        semantic_results = self.semantic_search(query, k=k)
        seen = {(r.act, r.section_number, r.text) for r in results}
        for r in semantic_results:
            key = (r.act, r.section_number, r.text)
            if key not in seen:
                results.append(r)
                seen.add(key)
        return results


def build_citation_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks with their verified labels for prompt
    injection. The opinion-generation layer must instruct the model to use
    ONLY these exact labels — never a section number recalled from memory."""
    return "\n\n".join(f"{c.citation_label}\n{c.text}" for c in chunks)
