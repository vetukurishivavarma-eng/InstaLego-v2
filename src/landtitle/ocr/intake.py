"""Document intake: text-based extraction with an OCR fallback per page.

Handles mixed English/Telugu Indian legal documents. Known issue (by design,
not a bug to fix): stamp-paper headers/watermarks with dense pink security
patterns OCR poorly. This is acceptable — core deed body text extracts
cleanly at 300 DPI even when headers are garbled, so we do not attempt
header/watermark cleanup here.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from landtitle.config import OCR_DPI, OCR_LANG

logger = logging.getLogger(__name__)

# A page with fewer than this many extractable characters is treated as
# image-only (scanned) rather than genuinely near-blank text.
MIN_CHARS_FOR_TEXT_PAGE = 40


@dataclass
class PageText:
    page_number: int  # 1-indexed
    text: str
    source: str  # "text" | "ocr"


def extract_pages_pypdf(pdf_path: str) -> list[PageText]:
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        pages.append(PageText(page_number=i, text=text, source="text"))
    return pages


def ocr_pages(pdf_path: str, page_numbers: list[int], dpi: int = OCR_DPI, lang: str = OCR_LANG) -> dict[int, str]:
    """OCR only the given 1-indexed page numbers (avoids re-rendering pages
    that already extracted cleanly as text)."""
    from pdf2image import convert_from_path
    from pdf2image.exceptions import PDFInfoNotInstalledError
    import pytesseract

    logger.info("OCR: %d page(s) need image-based extraction: %s", len(page_numbers), page_numbers)
    results: dict[int, str] = {}
    for page_number in page_numbers:
        try:
            images = convert_from_path(
                pdf_path, dpi=dpi, first_page=page_number, last_page=page_number
            )
        except PDFInfoNotInstalledError as exc:
            # Confirmed recurring setup friction: Poppler's pdfinfo/pdftoppm
            # binaries don't ship via pip and are easy to have installed but
            # not on PATH in a fresh shell -- see README "Setup" section.
            raise RuntimeError(
                "Poppler (pdftoppm/pdfinfo) is not on PATH -- required to rasterize scanned "
                "PDF pages before OCR. See README.md 'Setup' for install/PATH instructions."
            ) from exc
        if not images:
            results[page_number] = ""
            continue
        try:
            results[page_number] = pytesseract.image_to_string(images[0], lang=lang)
        except pytesseract.TesseractNotFoundError as exc:
            raise RuntimeError(
                "Tesseract OCR is not on PATH. See README.md 'Setup' for install/PATH "
                "instructions (including the eng+tel language pack)."
            ) from exc
        logger.debug("OCR: page %d -> %d chars", page_number, len(results[page_number]))
    return results


def extract_document(pdf_path: str) -> list[PageText]:
    """Extract every page as text where possible, OCR the rest.

    A document may be a mix of text-based and scanned pages (e.g. a scanned
    annexure attached to a typed deed) — this handles that per-page, rather
    than treating the whole document as one or the other.

    A `.txt` input is treated as already-extracted plain text (e.g. a test
    fixture, or text pulled out by some other tool) and returned as a single
    page — no PDF parsing or OCR involved.
    """
    if pdf_path.lower().endswith(".txt"):
        text = Path(pdf_path).read_text(encoding="utf-8")
        return [PageText(page_number=1, text=text, source="text")]

    pages = extract_pages_pypdf(pdf_path)
    scanned_page_numbers = [p.page_number for p in pages if len(p.text) < MIN_CHARS_FOR_TEXT_PAGE]

    if not scanned_page_numbers:
        logger.info("%s: %d page(s), all text-extractable", pdf_path, len(pages))
        return pages

    logger.info(
        "%s: %d page(s) total, %d need OCR", pdf_path, len(pages), len(scanned_page_numbers)
    )
    ocr_results = ocr_pages(pdf_path, scanned_page_numbers)
    final_pages = []
    for p in pages:
        if p.page_number in ocr_results:
            final_pages.append(PageText(page_number=p.page_number, text=ocr_results[p.page_number], source="ocr"))
        else:
            final_pages.append(p)
    return final_pages


def full_text(pages: list[PageText]) -> str:
    return "\n\n".join(p.text for p in pages if p.text)


def chunk_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks for pipelines/models with limited
    context, instead of silently truncating to the first N pages."""
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks
