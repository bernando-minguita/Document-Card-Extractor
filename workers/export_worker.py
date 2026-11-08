"""Worker thread that writes the approved cards to disk."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from PySide6.QtCore import QThread, Signal

from core.image_exporter import ExportOptions, ExportSummary, ImageExporter, suggest_output_dir
from core.models import DetectedCard
from core.pdf_loader import PdfDocument


class ExportJob:
    """One PDF and the cards that should be exported from it."""

    def __init__(self, pdf_path, cards: Sequence[DetectedCard], password: str = "") -> None:
        self.pdf_path = Path(pdf_path)
        # Snapshot the cards: the GUI keeps mutating the originals (rotate,
        # include, type change, geometry) while the export thread reads them -
        # the thread must see a frozen state, not a half-rotated one.
        self.cards: List[DetectedCard] = [card.copy_card() for card in cards]
        # Password of an encrypted PDF (in memory only, not persisted).
        self.password = password or ""

    @property
    def included_cards(self) -> List[DetectedCard]:
        return [card for card in self.cards if card.included]


class ExportWorker(QThread):
    """Export every included card of one or more PDFs."""

    progress = Signal(int, int, str)  # current, total, message
    exported = Signal(str)  # file path
    failed = Signal(str)  # human readable error
    finished_ok = Signal(object)  # ExportSummary
    cancelled = Signal(object)  # ExportSummary

    def __init__(self, jobs: Sequence[ExportJob], options: ExportOptions, parent=None) -> None:
        super().__init__(parent)
        self._jobs = list(jobs)
        self._options = options
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    # --------------------------------------------------------------------- run
    def run(self) -> None:
        total_cards = sum(len(job.included_cards) for job in self._jobs)
        summary = ExportSummary(output_dir=Path(self._options.output_dir))
        processed = 0
        sequence = 1
        for job in self._jobs:
            if self._cancel_requested:
                self.cancelled.emit(summary)
                return
            cards = job.included_cards
            if not cards:
                continue
            output_dir = suggest_output_dir(
                job.pdf_path, self._options.output_dir, self._options.create_subfolder
            )
            self.progress.emit(processed, total_cards, f"Exporting {job.pdf_path.name}…")
            document: Optional[PdfDocument] = None
            try:
                document = PdfDocument(job.pdf_path, password=job.password or None)
                exporter = ImageExporter(self._options)
                job_summary = exporter.export_cards(
                    document,
                    cards,
                    output_dir,
                    start_sequence=sequence,
                    progress=self._forward_progress(processed, total_cards),
                    cancel_check=lambda: self._cancel_requested,
                )
                summary.files.extend(job_summary.files)
                summary.errors.extend(job_summary.errors)
                sequence += job_summary.success_count
                for item in job_summary.files:
                    if item.ok:
                        self.exported.emit(str(item.path))
                processed += len(cards)
            except Exception as exc:
                summary.errors.append(f"{job.pdf_path.name}: {exc}")
                self.failed.emit(f"{job.pdf_path.name}: {exc}")
            finally:
                if document is not None:
                    document.close()

        if self._cancel_requested:
            self.cancelled.emit(summary)
            return
        self.progress.emit(total_cards, total_cards, "Export finished")
        self.finished_ok.emit(summary)

    # ------------------------------------------------------------------ helpers
    def _forward_progress(self, offset: int, total: int):
        def _callback(current: int, _total: int, message: str) -> None:
            self.progress.emit(offset + current, max(total, 1), message)

        return _callback
