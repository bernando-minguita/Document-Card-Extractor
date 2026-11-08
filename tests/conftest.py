"""Shared pytest fixtures."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Qt must run headless in the test environment.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SAMPLE_PDF = PROJECT_ROOT / "sample.pdf"


@pytest.fixture(scope="session")
def sample_pdf() -> Path:
    if not SAMPLE_PDF.exists():
        pytest.skip("sample scan sample.pdf is not available")
    return SAMPLE_PDF


@pytest.fixture(scope="session")
def synthetic_pdf(tmp_path_factory) -> Path:
    """A small generated PDF (no sample.pdf needed, works on any machine)."""
    try:
        import pymupdf as _fitz
    except ImportError:  # pragma: no cover - older PyMuPDF releases
        import fitz as _fitz

    path = tmp_path_factory.mktemp("docs") / "synthetic.pdf"
    document = _fitz.open()
    page = document.new_page(width=500, height=350)
    page.draw_rect(_fitz.Rect(40, 40, 280, 200), color=(0, 0, 0), fill=(0.15, 0.2, 0.55))
    page.draw_rect(_fitz.Rect(300, 40, 460, 200), color=(0, 0, 0), fill=(0.55, 0.2, 0.2))
    document.save(str(path))
    document.close()
    return path


@pytest.fixture(scope="session")
def qapp():
    """A single QApplication for all GUI tests."""
    from PySide6.QtWidgets import QApplication

    application = QApplication.instance() or QApplication([])
    yield application
