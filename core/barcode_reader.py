"""QR / barcode reading (optional, via pyzbar).

Extracts structured data from 2D codes commonly found on ID cards
(Saudi National ID, driving licence QR, etc.) to populate identifiers
more reliably than OCR regexes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

try:
    from pyzbar import pyzbar
    ZBAR_AVAILABLE = True
except Exception:
    ZBAR_AVAILABLE = False


@dataclass
class BarcodeResult:
    """A single decoded barcode/QR code."""
    data: str
    type: str
    rect: tuple  # (x, y, w, h)
    polygon: List[tuple]


@dataclass
class BarcodeReadout:
    """Aggregated result of barcode scanning on one image."""
    barcodes: List[BarcodeResult]
    best_identifier: str = ""

    @property
    def has_data(self) -> bool:
        return len(self.barcodes) > 0


def refine_identifier(card, image: np.ndarray, reader: "BarcodeReader") -> bool:
    """Prefer an ID-like QR/barcode value over the OCR-derived identifier.

    A QR code on a Saudi ID carries the authoritative national ID number, which
    is more reliable than the OCR regex on the printed text.  Returns ``True``
    when *card.identifier* was set from the barcode.
    """
    if reader is None or not reader.is_available or image is None or image.size == 0:
        return False
    try:
        readout = reader.read(image)
    except Exception:
        return False
    best = (readout.best_identifier or "").strip()
    if not best or not BarcodeReader._looks_like_id(best):
        return False  # not a number/serial pattern - never hurt the filename
    card.identifier = best
    return True


class BarcodeReader:
    """Read QR codes and barcodes from document crops."""

    def __init__(self) -> None:
        self._available = ZBAR_AVAILABLE
        if not self._available:
            self._init_error = "pyzbar not available (install pyzbar and system zbar library)"
        else:
            self._init_error = ""

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def init_error(self) -> str:
        return self._init_error

    @classmethod
    def diagnostic_report(cls) -> str:
        lines = [
            f"pyzbar available     : {ZBAR_AVAILABLE}",
        ]
        if not ZBAR_AVAILABLE:
            lines.append(
                "Install: pip install pyzbar AND system package zbar (brew install zbar / apt install libzbar0)"
            )
        return "\n".join(lines)

    def read(self, image: np.ndarray) -> BarcodeReadout:
        """Scan *image* (BGR) for barcodes/QR codes."""
        if not self._available or image is None or image.size == 0:
            return BarcodeReadout(barcodes=[])

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

        try:
            decoded = pyzbar.decode(gray)
        except Exception:
            # Communicating the failure via a fake "identifier" would leak
            # "Error: ..." into filenames - callers just see an empty readout.
            return BarcodeReadout(barcodes=[])

        results: List[BarcodeResult] = []
        for d in decoded:
            data = d.data.decode("utf-8", errors="ignore")
            results.append(BarcodeResult(
                data=data,
                type=d.type,
                rect=d.rect,
                polygon=[(p.x, p.y) for p in d.polygon] if d.polygon else []
            ))

        # Heuristic: prefer data that looks like an ID number
        best = ""
        for r in results:
            txt = r.data.strip()
            if self._looks_like_id(txt) and not best:
                best = txt
            elif not best and txt:
                best = txt

        return BarcodeReadout(barcodes=results, best_identifier=best)

    @staticmethod
    def _looks_like_id(text: str) -> bool:
        t = text.replace(" ", "").replace("-", "")
        if t.isdigit() and 8 <= len(t) <= 15:
            return True
        if len(t) >= 6 and any(c.isdigit() for c in t) and any(c.isalpha() for c in t):
            return True
        return False