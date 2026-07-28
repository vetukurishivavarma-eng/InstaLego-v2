"""Legal corpus builder for Transfer of Property Act 1882, Registration Act
1908, and Limitation Act 1963.

IMPORTANT: this module does not embed the Acts' text. Statute text must be
downloaded by the user from indiacode.nic.in (public domain) and placed in
data/acts/ per config.ACTS — see the project README. Fabricating or
approximating statutory text here would be exactly the kind of confident
fabrication this whole project is designed to prevent.

CRITICAL corpus-building bug and fix (confirmed root cause of citation
fabrication downstream): naively section-chunking by splitting on
`\\n(\\d+[A-Z]?\\.\\s)` misidentifies footnote/amendment reference lines
(e.g. "1. Subs. by Act 39 of 1948, s. 5, for s. 88.") as section boundaries,
since they also start with a number-period pattern. This corrupts the
corpus: real section text gets split incorrectly and section-number
extraction becomes unreliable. Footnote lines must be filtered out BEFORE
section-splitting, using a broad keyword check, not just line-start shape.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

_AMENDMENT_KEYWORDS = r"(Subs\.|Ins\.|Omitted|Rep\.|Added|omitted by|substituted by|inserted by|repealed by|Act \d+ of \d{4})"
_SECTION_SPLIT_PATTERN = re.compile(r"\n(\d+[A-Z]?\.\s)")
_SECTION_NUMBER_PATTERN = re.compile(r"^(\d+[A-Z]?)\.\s")


def is_footnote_line(line: str) -> bool:
    line = line.strip()
    if re.match(r"^\d+\.?\s", line) and re.search(_AMENDMENT_KEYWORDS, line):
        return True
    return False


def strip_footnotes(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not is_footnote_line(line))


@dataclass
class Section:
    act: str
    section_number: str | None
    text: str


def split_into_sections(text: str, act_name: str) -> list[Section]:
    """Split cleaned (footnote-stripped) Act text into per-section chunks,
    extracting the verified section number directly from each chunk's own
    leading pattern rather than trusting any number generated later by an LLM."""
    cleaned = strip_footnotes(text)
    # Prepend a newline so a section number sitting at the very start of the
    # text (no preceding newline — e.g. the first section of a chunk) is
    # still recognized as a boundary by the \n-anchored split pattern below.
    parts = _SECTION_SPLIT_PATTERN.split("\n" + cleaned)

    sections: list[Section] = []
    # re.split with a capturing group interleaves: [preamble, marker, body, marker, body, ...]
    if len(parts) == 1:
        return [Section(act=act_name, section_number=None, text=cleaned.strip())]

    preamble = parts[0].strip()
    if preamble:
        sections.append(Section(act=act_name, section_number=None, text=preamble))

    for i in range(1, len(parts), 2):
        marker = parts[i]
        body = parts[i + 1] if i + 1 < len(parts) else ""
        chunk = (marker + body).strip()
        match = _SECTION_NUMBER_PATTERN.match(chunk)
        section_number = match.group(1) if match else None
        sections.append(Section(act=act_name, section_number=section_number, text=chunk))

    return sections


def build_sections_for_acts(acts: dict[str, Path]) -> list[Section]:
    all_sections: list[Section] = []
    for act_name, path in acts.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Act text not found at {path}. Download the official text of '{act_name}' "
                f"from indiacode.nic.in and place it there before building the corpus."
            )
        text = path.read_text(encoding="utf-8")
        all_sections.extend(split_into_sections(text, act_name))
    return all_sections


def build_and_save_corpus(acts: dict[str, Path], embedding_model: str, index_path: Path, metadata_path: Path) -> None:
    import faiss
    from sentence_transformers import SentenceTransformer

    sections = build_sections_for_acts(acts)
    model = SentenceTransformer(embedding_model)
    embeddings = model.encode([s.text for s in sections], convert_to_numpy=True, show_progress_bar=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    metadata_path.write_text(json.dumps([asdict(s) for s in sections], ensure_ascii=False, indent=2), encoding="utf-8")
