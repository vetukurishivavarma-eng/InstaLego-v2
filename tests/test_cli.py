import pytest

from landtitle.pipeline import _require_existing_file


def test_require_existing_file_passes_for_real_file(tmp_path):
    real_file = tmp_path / "deed.pdf"
    real_file.write_text("not a real pdf, just needs to exist")
    _require_existing_file(str(real_file), "Sale Deed")  # must not raise


def test_require_existing_file_fails_fast_on_missing_path(tmp_path):
    missing = tmp_path / "does_not_exist.pdf"
    with pytest.raises(SystemExit, match="Sale Deed not found"):
        _require_existing_file(str(missing), "Sale Deed")
