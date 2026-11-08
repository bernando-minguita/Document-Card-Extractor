"""Worker thread that runs card detection for a list of PDF pages."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional, Sequence

from PySide6.QtCore import QThread, Signal

from core.card_detector import CardDetector, DetectionConfig, candidate_to_card
from core.document_classifier import DocumentClassifier
from core.image_preprocessor import resize_max_dim
from core.models import DetectedCard
from core.ocr_engine import OcrEngine
from core.orientation_detector import OrientationDetector
from core.pdf_loader import PdfDocument
from core.perspective_corrector import make_thumbnail, rotate_image, warp_quad


def ocr_report(engine: Optional[OcrEngine], cards_read: int, total_cards: int) -> str:
    """Human readable summary of what OCR did during a run (for the status bar)."""
    if engine is None:
        return ""
    backend = engine.backend
    if not backend:
        return "enabled, but no engine is available"
    if total_cards <= 0:
        return backend
    return f"{backend} ({cards_read}/{total_cards} cards read)"


class DetectionWorker(QThread):
    """Detect, straighten and classify every card of the selected pages."""

    page_started = Signal(int)  # page index
    page_finished = Signal(int, int)  # page index, cards found
    card_found = Signal(object)  # DetectedCard
    progress = Signal(int, int, str)  # current, total, message
    failed = Signal(str)  # human readable error
    finished_ok = Signal(int)  # total cards
    cancelled = Signal(int)  # cards found until cancellation

    def __init__(
        self,
        pdf_path,
        page_indices: Sequence[int],
        settings,
        password: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._pdf_path = Path(pdf_path)
        self._page_indices = list(page_indices)
        # The GUI keeps editing the settings while detection runs (settings
        # page, Add Region presets, …) - the worker works on a frozen snapshot
        # so a mid-run change cannot tear the configuration.
        self._settings = copy.deepcopy(settings)
        self._password = password or ""
        self._cancel_requested = False
        self.ocr_report = ""  # filled while running, read when the worker finishes

    # ------------------------------------------------------------------ control
    def request_cancel(self) -> None:
        self._cancel_requested = True


    # --------------------------------------------------------------------- run
    def run(self) -> None:
        document: Optional[PdfDocument] = None
        total_cards = 0
        try:
            document = PdfDocument(self._pdf_path, password=self._password or None)
        except Exception as exc:
            self.failed.emit(str(exc))
            self.finished_ok.emit(0)
            return

        settings = self._settings
        try:
            detector = CardDetector(DetectionConfig.from_settings(settings))
            orientation = OrientationDetector()
            ocr_engine = OcrEngine(
                languages=(getattr(settings, "ocr_languages", "en") or "en").split(","),
                prefer=getattr(settings, "ocr_engine", "auto"),
            ) if bool(getattr(settings, "ocr_enabled", False)) else None
            classifier = (
                DocumentClassifier(ocr_engine=ocr_engine)
                if bool(getattr(settings, "auto_classify", False))
                else None
            )
            pages = self._page_indices or list(range(document.page_count))
            ocr_cards = 0

            for position, page_index in enumerate(pages):
                if self._cancel_requested:
                    self.cancelled.emit(total_cards)
                    return
                self.page_started.emit(page_index)
                self.progress.emit(
                    position, len(pages), f"Analysing page {page_index + 1} of {document.page_count}…"
                )
                page_cards = 0
                try:
                    page_image, working = self._render(document, page_index)
                    height, width = working.shape[:2]
                    for candidate in detector.detect(working):
                        if self._cancel_requested:
                            self.cancelled.emit(total_cards)
                            return
                        card = candidate_to_card(candidate, width, height, page_index, self._padding())
                        card, read = self._finish_card(
                            card, working, orientation, ocr_engine, classifier
                        )
                        if read:
                            ocr_cards += 1
                        total_cards += 1
                        page_cards += 1
                        self.card_found.emit(card)
                    del working, page_image
                except Exception as exc:  # a broken page must not stop the batch
                    self.failed.emit(f"Page {page_index + 1}: {exc}")
                self.page_finished.emit(page_index, page_cards)

            # A cancel that arrived during the very last page still counts as
            # cancelled (not as a successful run without cards).
            if self._cancel_requested:
                self.cancelled.emit(total_cards)
                return
            self.ocr_report = ocr_report(ocr_engine, ocr_cards, total_cards)
            self.progress.emit(len(pages), len(pages), "Detection finished")
            self.finished_ok.emit(total_cards)
        except Exception as exc:  # pragma: no cover - defensive
            self.failed.emit(str(exc))
            self.finished_ok.emit(total_cards)
        finally:
            if document is not None:
                document.close()

    # ------------------------------------------------------------------ helpers
    def _render(self, document: PdfDocument, page_index: int):
        settings = self._settings
        requested = int(getattr(settings, "render_dpi", 300))
        cap = bool(getattr(settings, "cap_dpi_to_native", True))
        dpi = document.effective_dpi(page_index, requested, cap_to_native=cap)
        page_image = document.render_page(page_index, dpi=dpi)
        working = resize_max_dim(page_image, int(getattr(settings, "detection_working_max_dim", 2200)))
        return page_image, working

    def _padding(self) -> float:
        return float(getattr(self._settings, "card_padding_ratio", 0.006))

    def _finish_card(
        self,
        card: DetectedCard,
        page_image,
        orientation: OrientationDetector,
        ocr_engine: Optional[OcrEngine],
        classifier: Optional[DocumentClassifier],
    ):
        """Warp the card, fix its orientation and (optionally) classify it.

        Returns ``(card, read_by_ocr)`` where *read_by_ocr* is ``True`` when the
        OCR engine returned text for this card.
        """
        settings = self._settings
        height, width = page_image.shape[:2]

        if self._cancel_requested:
            # A cancel arrived while warping this card - skip the slow stages
            # (orientation OCR, classification OCR).  Checking right above the
            # expensive OCR reads keeps the response time to a card boundary.
            return card, False

        quad = card.quad_pixels(width, height)
        crop = warp_quad(page_image, quad)

        if bool(getattr(settings, "auto_orientation", True)):
            use_ocr = bool(getattr(settings, "ocr_for_orientation", False)) and ocr_engine is not None
            if use_ocr:
                def reader(image):
                    if self._cancel_requested:
                        # OrientationDetector catches this and falls back to
                        # the heuristic instead of starting another OCR read.
                        raise RuntimeError("cancelled")
                    return ocr_engine.read(image)

                result = orientation.detect(crop, ocr_reader=reader, allow_ocr=True)
            else:
                result = orientation.detect(crop)
            card.rotation = int(result.rotation) % 360
            card.orientation_confident = bool(result.confident)

        upright = rotate_image(crop, card.rotation) if card.rotation else crop

        read_by_ocr = False
        if classifier is not None:
            try:
                if ocr_engine is None or self._cancel_requested:
                    text = ""
                else:
                    text = ocr_engine.read(upright).text
                read_by_ocr = bool(text.strip())
                card.ocr_text = text or ""
                result = classifier.classify(
                    image=None if text else upright,
                    ocr_text=text,
                    aspect_ratio=card.aspect_ratio(),
                    area_ratio=card.area_ratio(),
                )
                card.set_type(result.doc_type, result.confidence)
                card.identifier = result.identifier
                if result.reason:
                    card.notes = result.reason
            except Exception as exc:
                card.notes = f"Classification skipped: {exc}"

        card.thumbnail = make_thumbnail(upright, int(getattr(settings, "thumbnail_size", 420)))
        return card, read_by_ocr

