"""Table-aware pre-processing for Encumbrance Certificates.

CRITICAL ARCHITECTURAL FINDING (confirmed, reproduced on both 7B and 14B):
EC data is tabular (Sl.No, Document No., Date, Nature, Executants, Claimants,
Consideration). Feeding flat OCR'd/extracted prose text directly to the LLM
causes it to confuse which party is Executant (seller) vs Claimant (buyer),
especially on rows with multi-line wrapped text (e.g. a GPA reference
spanning two lines).

Fix: extract table structure BEFORE the LLM ever sees the text, and convert
each row into explicit pre-labeled fields (e.g. "Executants (Sellers): X").
This module never calls the LLM — it is pure structural extraction.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# --- Column recognition -----------------------------------------------------

_COLUMN_ALIASES: dict[str, list[str]] = {
    "document_number": ["document no", "doc no", "document number", "regd no", "registration no", "reg no"],
    "date": ["date"],
    "nature_of_document": ["nature"],
    "executants_sellers": ["executant", "vendor", "seller"],
    "claimants_buyers": ["claimant", "vendee", "buyer", "purchaser"],
    "consideration": ["consideration", "value", "amount"],
}

_COLUMN_LABELS: dict[str, str] = {
    "document_number": "Document Number",
    "date": "Date",
    "nature_of_document": "Nature of Document",
    "executants_sellers": "Executants (Sellers)",
    "claimants_buyers": "Claimants (Buyers)",
    "consideration": "Consideration",
}

# Fallback column order used only when no recognizable header row is found
# at all (e.g. a scanned table whose header row OCR'd too poorly to match).
_CANONICAL_FALLBACK_ORDER = [
    "sl_no",
    "document_number",
    "date",
    "nature_of_document",
    "executants_sellers",
    "claimants_buyers",
    "consideration",
]


def _match_column(header_cell: str) -> str | None:
    cell = (header_cell or "").strip().lower()
    if not cell:
        return None
    for key, aliases in _COLUMN_ALIASES.items():
        if any(alias in cell for alias in aliases):
            return key
    return None


def build_column_mapping(header_row: list[str]) -> dict[int, str]:
    """Map column index -> semantic key, based on a detected header row."""
    mapping: dict[int, str] = {}
    for idx, cell in enumerate(header_row):
        key = _match_column(cell or "")
        if key:
            mapping[idx] = key
    return mapping


def looks_like_header_row(row: list[str]) -> bool:
    matches = sum(1 for cell in row if _match_column(cell))
    return matches >= 2


@dataclass
class LabeledRow:
    row_index: int
    fields: dict[str, str] = field(default_factory=dict)

    def to_labeled_text(self) -> str:
        lines = []
        for key in ("document_number", "date", "nature_of_document", "executants_sellers", "claimants_buyers", "consideration"):
            if key in self.fields and self.fields[key].strip():
                lines.append(f"{_COLUMN_LABELS[key]}: {self.fields[key].strip()}")
        return "\n".join(lines)


def flatten_table(rows: list[list[str]]) -> list[LabeledRow]:
    """Given raw table rows (as returned by pdfplumber/camelot or the scanned
    grid detector), find the header, build a column mapping, and produce one
    pre-labeled row per data row — never a single flattened blob."""
    if not rows:
        return []

    header_idx = None
    mapping: dict[int, str] = {}
    for i, row in enumerate(rows[:3]):  # header is expected within the first few rows
        if looks_like_header_row(row):
            header_idx = i
            mapping = build_column_mapping(row)
            break

    if header_idx is None:
        # No recognizable header — fall back to canonical positional order.
        mapping = {i: key for i, key in enumerate(_CANONICAL_FALLBACK_ORDER)}
        data_rows = rows
    else:
        data_rows = rows[header_idx + 1:]

    labeled_rows = []
    for i, row in enumerate(data_rows):
        fields = {}
        for col_idx, cell in enumerate(row):
            key = mapping.get(col_idx)
            if key and key != "sl_no" and cell:
                fields[key] = str(cell).replace("\n", " ").strip()
        if not fields:
            continue
        # A row whose Document Number is "NIL"/"--" and has no party fields
        # is a placeholder remark (e.g. "No mortgage entries found for this
        # property...", "No court attachment..."), not a real registered
        # transaction. Skip it — otherwise its own negated wording would
        # trip the keyword-based mortgage/litigation flags downstream
        # (encumbrance._compute_flags does a plain substring match, and
        # "No mortgage entries found" still contains the word "mortgage").
        doc_num = fields.get("document_number", "").strip().lower()
        has_parties = bool(fields.get("executants_sellers", "").strip() or fields.get("claimants_buyers", "").strip())
        if doc_num in ("nil", "--", "n/a", "") and not has_parties:
            continue
        labeled_rows.append(LabeledRow(row_index=i, fields=fields))
    return labeled_rows


# --- Plain-text (non-PDF) table extraction ----------------------------------
#
# For hand-typed / whitespace-aligned EC text (e.g. a test fixture, or text
# already extracted some other way) rather than a real PDF table. Column
# alignment in hand-typed text is not always exact — in particular,
# continuation lines that wrap a cell's text (e.g. a GPA reference spanning
# two lines) may not land under the correct header column by character
# offset alone. Attribution order: (1) a GPA/POA keyword always means the
# continuation belongs to Executants — GPA holders act for the seller side,
# consistent with SaleDeed.represented_by_gpa elsewhere in this project;
# (2) otherwise, inherit whatever field the previous continuation line in
# the same row was attributed to; (3) otherwise, whichever wrappable column
# (Nature/Executants/Claimants) has a header offset closest to this line's
# indentation.

_TEXT_ROW_START_RE = re.compile(r"^\d+\.?\s")
_GPA_KEYWORDS = ("gpa", "general power of attorney", "power of attorney")
_SL_NO_HEADER_ALIASES = ("sl.no", "sl no", "sr.no", "sr no")


def _text_header_offsets(header_line: str) -> dict[str, int]:
    lower = header_line.lower()
    offsets: dict[str, int] = {}
    for alias in _SL_NO_HEADER_ALIASES:
        idx = lower.find(alias)
        if idx != -1:
            offsets["sl_no"] = idx
            break
    for key, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            idx = lower.find(alias)
            if idx != -1:
                offsets[key] = idx
                break
    return offsets


def parse_fixed_width_text_table(text: str) -> list[list[str]]:
    """Parse a whitespace-aligned EC table out of plain text into canonical-
    order rows (`_CANONICAL_FALLBACK_ORDER`), suitable to pass into
    `flatten_table()`. Returns [] if no recognizable header line is found."""
    lines = text.splitlines()

    header_idx = None
    for i, line in enumerate(lines):
        if looks_like_header_row(re.split(r"\s{2,}", line.strip())):
            header_idx = i
            break
    if header_idx is None:
        return []

    offsets = _text_header_offsets(lines[header_idx])
    if not offsets:
        return []
    ordered_keys = sorted(offsets, key=offsets.get)
    wrappable_fields = [k for k in ("nature_of_document", "executants_sellers", "claimants_buyers") if k in offsets]

    def slice_line(line: str) -> dict[str, str]:
        fields = {}
        for idx, key in enumerate(ordered_keys):
            start = offsets[key]
            end = offsets[ordered_keys[idx + 1]] if idx + 1 < len(ordered_keys) else None
            value = line[start:end] if end is not None else line[start:]
            fields[key] = value.strip(" ,;")
        return fields

    rows: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    last_continuation_field: str | None = None

    for line in lines[header_idx + 1:]:
        if not line.strip():
            current = None
            last_continuation_field = None
            continue
        if _TEXT_ROW_START_RE.match(line.strip()):
            candidate = slice_line(line)
            # A row whose Document Number is "NIL"/"--" is a placeholder
            # remark (e.g. "No mortgage entries found for this property...")
            # rather than a real transaction — its free-text sentence would
            # otherwise get chopped by column offsets and spill into the
            # Executants/Claimants slots, making it look like it has parties.
            # Drop it here, before that spillover happens.
            if candidate.get("document_number", "").strip().lower() in ("nil", "--", "n/a"):
                current = None
                last_continuation_field = None
                continue
            current = candidate
            rows.append(current)
            last_continuation_field = None
            continue
        if current is None:
            continue  # stray text outside any row — ignore

        stripped = line.strip()
        indent = len(line) - len(line.lstrip())

        if wrappable_fields and any(kw in stripped.lower() for kw in _GPA_KEYWORDS) and "executants_sellers" in offsets:
            target = "executants_sellers"
        elif last_continuation_field is not None:
            target = last_continuation_field
        elif wrappable_fields:
            target = min(wrappable_fields, key=lambda k: abs(offsets[k] - indent))
        else:
            continue

        current[target] = re.sub(r"\s+", " ", f"{current.get(target, '')} {stripped}").strip()
        last_continuation_field = target

    return [[row.get(key, "") for key in _CANONICAL_FALLBACK_ORDER] for row in rows]


# --- Text-based PDF table extraction (pdfplumber) ---------------------------


def extract_tables_pdfplumber(pdf_path: str) -> list[list[list[str]]]:
    import pdfplumber

    all_tables: list[list[list[str]]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                cleaned = [[(cell or "") for cell in row] for row in table]
                all_tables.append(cleaned)
    return all_tables


# --- Scanned PDF table extraction (OpenCV grid detection + per-cell OCR) ----


def _cluster_positions(positions: list[int], gap: int = 8) -> list[int]:
    """Merge nearby pixel indices (from a summed line mask) into single
    line centers, since a printed grid line is several pixels thick."""
    if not positions:
        return []
    positions = sorted(positions)
    clusters: list[list[int]] = [[positions[0]]]
    for p in positions[1:]:
        if p - clusters[-1][-1] <= gap:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [int(sum(c) / len(c)) for c in clusters]


def detect_table_grid(image) -> list[list[str]]:
    """Detect a table grid in a scanned page image via horizontal/vertical
    line morphology, then OCR each cell independently. Returns rows of cell
    text. Returns [] if no grid structure is found (caller should fall back
    to treating the page as non-tabular)."""
    import cv2
    import numpy as np
    import pytesseract

    from landtitle.config import OCR_LANG

    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(
        ~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2
    )

    horizontal_size = max(thresh.shape[1] // 30, 1)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.dilate(cv2.erode(thresh, horizontal_kernel), horizontal_kernel)

    vertical_size = max(thresh.shape[0] // 30, 1)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vertical = cv2.dilate(cv2.erode(thresh, vertical_kernel), vertical_kernel)

    row_positions = _cluster_positions(list(np.where(horizontal.sum(axis=1) > horizontal.shape[1] * 0.3 * 255)[0]))
    col_positions = _cluster_positions(list(np.where(vertical.sum(axis=0) > vertical.shape[0] * 0.3 * 255)[0]))

    if len(row_positions) < 2 or len(col_positions) < 2:
        return []

    rows: list[list[str]] = []
    for r in range(len(row_positions) - 1):
        y0, y1 = row_positions[r], row_positions[r + 1]
        row_cells = []
        for c in range(len(col_positions) - 1):
            x0, x1 = col_positions[c], col_positions[c + 1]
            cell_img = img[y0:y1, x0:x1]
            if cell_img.size == 0:
                row_cells.append("")
                continue
            cell_text = pytesseract.image_to_string(cell_img, lang=OCR_LANG).strip()
            row_cells.append(cell_text)
        rows.append(row_cells)
    return rows


def extract_tables_scanned(pdf_path: str, dpi: int | None = None) -> list[list[list[str]]]:
    from pdf2image import convert_from_path

    from landtitle.config import OCR_DPI

    images = convert_from_path(pdf_path, dpi=dpi or OCR_DPI)
    tables = []
    for image in images:
        grid = detect_table_grid(image)
        if grid:
            tables.append(grid)
    return tables
