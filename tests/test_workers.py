"""Direct tests for the QThread workers (run synchronously, no event loop).

``run()`` is an ordinary method, so a worker that was never ``start()``-ed can
be executed synchronously and every ``emit`` calls the connected test code
directly.  That way thread behaviour (sequence numbering, cancellation, error
paths) can be checked without a GUI.
"""

from __future__ import annotations

from typing import List

from core.image_exporter import ExportOptions
from core.models import DetectedCard
from workers.detection_worker import DetectionWorker, ocr_report
from workers.export_worker import ExportJob, ExportWorker


def _card(page_index: int = 0, **kwargs) -> DetectedCard:
    quad = [[0.05, 0.05], [0.40, 0.05], [0.40, 0.40], [0.05, 0.40]]
    return DetectedCard(page_index=page_index, quad_norm=quad, **kwargs)


def test_export_worker_names_the_first_file_001(synthetic_pdf, tmp_path, qapp):
    """Regression: the first exported file must be numbered _001, not _002."""
    collector: List[object] = []
    worker = ExportWorker(
        [ExportJob(synthetic_pdf, [_card(source="a"), _card(source="b")])],
        ExportOptions(output_dir=str(tmp_path), image_format="png"),
    )
    worker.finished_ok.connect(collector.append)
    worker.run()

    summary = collector[0]
    names = [item.path.name for item in summary.files if item.ok]
    assert names[0].endswith("_P01_01.png")   # sequence 1, was _02 before the fix
    assert names[1].endswith("_P01_02.png")
    assert summary.errors == []


def test_export_worker_sequences_continuously_across_jobs(synthetic_pdf, tmp_path, qapp):
    """Two export jobs in one run must not restart or skip the numbering."""
    collector: List[object] = []
    job_one = ExportJob(synthetic_pdf, [_card()])
    job_two = ExportJob(synthetic_pdf, [_card(), _card()])
    worker = ExportWorker(
        [job_one, job_two], ExportOptions(output_dir=str(tmp_path), image_format="png")
    )
    worker.finished_ok.connect(collector.append)
    worker.run()

    summary = collector[0]
    names = [item.path.name for item in summary.files if item.ok]
    assert len(names) == 3
    assert [name.split("_")[-1] for name in names] == ["01.png", "02.png", "03.png"]


def test_export_worker_cancelled_before_run_writes_nothing(synthetic_pdf, tmp_path, qapp):
    cancelled: List[str] = []
    finished: List[object] = []
    worker = ExportWorker(
        [ExportJob(synthetic_pdf, [_card()])],
        ExportOptions(output_dir=str(tmp_path), image_format="png"),
    )
    worker.request_cancel()
    worker.cancelled.connect(lambda summary: cancelled.append("cancelled"))
    worker.finished_ok.connect(lambda summary: finished.append(summary))
    worker.run()

    assert cancelled == ["cancelled"]
    assert finished == []


def test_export_worker_ignores_excluded_cards(synthetic_pdf, tmp_path, qapp):
    excluded = _card()
    excluded.included = False
    collector: List[object] = []
    worker = ExportWorker(
        [ExportJob(synthetic_pdf, [excluded, _card()])],
        ExportOptions(output_dir=str(tmp_path), image_format="png"),
    )
    worker.finished_ok.connect(collector.append)
    worker.run()

    summary = collector[0]
    assert summary.success_count == 1


def test_export_worker_reports_a_skipped_face_blur_once(synthetic_pdf, tmp_path, qapp, monkeypatch):
    """An unavailable cascade must not fail the export, only warn (once per run)."""
    from core import image_exporter

    reason = "face blur needs OpenCV's Haar cascade"
    monkeypatch.setattr(image_exporter, "face_blur_support", lambda: (False, reason))

    collector: List[object] = []
    worker = ExportWorker(
        [ExportJob(synthetic_pdf, [_card(), _card()])],
        ExportOptions(output_dir=str(tmp_path), image_format="png", blur_faces=True),
    )
    worker.finished_ok.connect(collector.append)
    worker.run()

    summary = collector[0]
    assert summary.success_count == 2, "the images must still be written"
    assert summary.errors == []
    assert summary.warnings == [reason], "the skip must be reported exactly once"


def test_export_worker_reports_a_broken_file_as_an_error(tmp_path, qapp):
    unsupported = tmp_path / "not_a_scan.pdf"
    unsupported.write_bytes(b"this is not a real pdf")

    failures: List[str] = []
    finished: List[object] = []
    worker = ExportWorker(
        [ExportJob(unsupported, [_card()])],
        ExportOptions(output_dir=str(tmp_path / "out"), image_format="png"),
    )
    worker.failed.connect(failures.append)
    worker.finished_ok.connect(finished.append)
    worker.run()

    assert failures  # the open failure was reported
    assert len(finished) == 1  # the run itself still finished


def test_ocr_report_summary_states():
    class _Engine:
        backend = "tesseract"

    assert ocr_report(None, 0, 0) == ""
    assert ocr_report(_Engine(), 0, 0) == "tesseract"
    assert ocr_report(_Engine(), 3, 4) == "tesseract (3/4 cards read)"


def test_detection_worker_cancelled_before_run(synthetic_pdf, qapp):
    """A cancel requested before ``run()`` must report cancelled, not success."""
    cancelled: List[int] = []
    finished: List[int] = []
    cards: List[DetectedCard] = []

    settings = type("Settings", (), {})()
    worker = DetectionWorker(synthetic_pdf, [0], settings)
    worker.request_cancel()
    worker.cancelled.connect(cancelled.append)
    worker.finished_ok.connect(finished.append)
    worker.card_found.connect(cards.append)
    worker.run()

    assert len(cancelled) == 1
    assert finished == []
    assert cards == []


def test_detection_worker_snapshots_the_settings(qapp):
    """Changes to the live settings object must not leak into a started worker."""
    from app.settings import AppSettings

    settings = AppSettings()
    settings.render_dpi = 777
    worker = DetectionWorker("x.pdf", [0], settings)
    assert worker._settings is not settings
    assert worker._settings.render_dpi == 777
    settings.render_dpi = 999  # a mid-run change on the settings page
    assert worker._settings.render_dpi == 777


def test_export_job_snapshots_the_cards(qapp):
    """The GUI keeps editing the live cards while the export thread reads them."""
    card = _card()
    job = ExportJob("sample.pdf", [card])
    assert job.cards[0] is not card
    card.included = False  # a mid-run edit of the original
    assert len(job.included_cards) == 1


def test_detection_worker_runs_the_pipeline_without_ocr(synthetic_pdf, qapp):
    """A defaults-only run (no OCR settings) finishes cleanly, without errors."""
    finished: List[int] = []
    failures: List[str] = []
    cards: List[DetectedCard] = []

    settings = type("Settings", (), {})()  # every attribute falls back via getattr
    worker = DetectionWorker(synthetic_pdf, [0], settings)
    worker.finished_ok.connect(finished.append)
    worker.failed.connect(failures.append)
    worker.card_found.connect(cards.append)
    worker.run()

    assert len(finished) == 1
    assert failures == []
    assert finished[0] == len(cards)
    for card in cards:
        assert card.thumbnail is not None
