"""Confirms missing Tesseract/Poppler binaries surface as a clear, actionable
RuntimeError pointing at README setup instructions, instead of the raw
pytesseract/pdf2image exception -- a real, repeated setup friction point on
this project (PATH not picked up in fresh shells)."""
from unittest.mock import patch

import pytest

from landtitle.ocr.intake import ocr_pages


@patch("pdf2image.convert_from_path")
def test_missing_poppler_raises_actionable_error(mock_convert):
    from pdf2image.exceptions import PDFInfoNotInstalledError

    mock_convert.side_effect = PDFInfoNotInstalledError("pdfinfo not found")
    with pytest.raises(RuntimeError, match="Poppler"):
        ocr_pages("fake.pdf", [1])


@patch("pytesseract.image_to_string")
@patch("pdf2image.convert_from_path")
def test_missing_tesseract_raises_actionable_error(mock_convert, mock_ocr):
    import pytesseract
    from PIL import Image

    mock_convert.return_value = [Image.new("RGB", (10, 10))]
    mock_ocr.side_effect = pytesseract.TesseractNotFoundError()
    with pytest.raises(RuntimeError, match="Tesseract"):
        ocr_pages("fake.pdf", [1])
