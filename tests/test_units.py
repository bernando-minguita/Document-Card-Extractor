"""Unit tests for geometry helpers, filename building and classification."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from core.document_classifier import DocumentClassifier, extract_identifier
from core.image_exporter import ExportOptions, ImageExporter, sanitize_filename, suggest_output_dir
from core.models import DetectedCard, DocumentType
from core.perspective_corrector import (
    expand_quad,
    make_thumbnail,
    order_quad_points,
    quad_iou,
    quad_size,
    rotate_image,
    warp_quad,
)


def _quad(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def test_order_quad_points_is_stable():
    shuffled = np.array([[100, 100], [0, 0], [0, 100], [100, 0]], dtype=np.float32)
    ordered = order_quad_points(shuffled.tolist())
    assert np.allclose(ordered[0], [0, 0])
    assert np.allclose(ordered[1], [100, 0])
    assert np.allclose(ordered[2], [100, 100])
    assert np.allclose(ordered[3], [0, 100])


def test_order_quad_points_keeps_rotation():
    rotated = _quad(10, 10, 10, 210)  # tall (90 degree rotated) card
    ordered = order_quad_points(rotated.tolist())
    assert np.allclose(ordered.sum(axis=0), rotated.sum(axis=0))


def test_expand_quad_pads_every_side_by_the_same_amount():
    """Regression: the diagonal cancelled out, so a wide card lost top/bottom padding."""
    quad = _quad(100, 100, 500, 200)  # 400 x 100 px, diagonal ~412.3
    diagonal = float(np.hypot(400.0, 100.0))
    expected = 0.02 * diagonal
    grown = expand_quad(quad, 0.02)
    assert 100.0 - float(grown[:, 0].min()) == pytest.approx(expected, rel=0.02)
    assert float(grown[:, 0].max()) - 500.0 == pytest.approx(expected, rel=0.02)
    assert 100.0 - float(grown[:, 1].min()) == pytest.approx(expected, rel=0.02)
    assert float(grown[:, 1].max()) - 200.0 == pytest.approx(expected, rel=0.02)


def test_page_range_parser_accepts_ranges_and_reports_typos():
    from core.page_range import parse_page_range

    assert parse_page_range("1,3,5-7", 10) == ([0, 2, 4, 5, 6], [])
    assert parse_page_range("", 3) == ([0, 1, 2], [])
    assert parse_page_range("0-2", 5) == ([0, 1], [])  # clamped to page 1
    assert parse_page_range("4-99", 5) == ([3, 4], [])  # clamped to page 5
    assert parse_page_range("1,x", 5) == ([0], ["x"])  # typo reported, rest kept
    assert parse_page_range("7-5", 5) == ([], ["7-5"])  # reversed range
    assert parse_page_range("1,x") == ([0], ["x"])  # validated before any file


def test_cli_reports_a_malformed_page_range_instead_of_crashing(sample_pdf, capsys):
    """Regression: ``--pages 1,x`` raised ValueError from int()."""
    import cli

    with pytest.raises(SystemExit) as excinfo:
        cli.main([str(sample_pdf), "--pages", "1,x"])
    assert excinfo.value.code == 2  # argparse usage error
    message = capsys.readouterr().err
    assert "--pages" in message and "x" in message


def test_cli_accepts_the_tif_extension():
    """The exporter writes .tif, so the command line must offer it too."""
    import cli

    assert cli.build_parser().parse_args(["scan.pdf", "--format", "tif"]).format == "tif"


def test_faint_text_meets_the_wcag_contrast_ratio():
    from app import styles

    def luminance(colour: str) -> float:
        channels = [int(colour[index : index + 2], 16) / 255.0 for index in (1, 3, 5)]

        def linear(value: float) -> float:
            return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4

        red, green, blue = (linear(value) for value in channels)
        return 0.2126 * red + 0.7152 * green + 0.0722 * blue

    def ratio(foreground: str, background: str) -> float:
        lighter = max(luminance(foreground), luminance(background))
        darker = min(luminance(foreground), luminance(background))
        return (lighter + 0.05) / (darker + 0.05)

    for background in (styles.SURFACE, styles.SURFACE_ALT, styles.BACKGROUND):
        for colour in (styles.TEXT, styles.TEXT_MUTED, styles.TEXT_FAINT):
            assert ratio(colour, background) >= 4.5, f"{colour} on {background} is too faint"



def test_quad_size_and_iou():
    a = _quad(0, 0, 200, 100)
    b = _quad(0, 0, 200, 100)
    assert quad_size(a) == (200, 100)
    assert quad_iou(a, b) > 0.98
    assert quad_iou(a, _quad(500, 500, 700, 600)) == 0.0


def test_warp_quad_produces_upright_crop():
    page = np.zeros((400, 500, 3), dtype=np.uint8)
    page[50:150, 60:260] = (30, 200, 30)
    crop = warp_quad(page, _quad(60, 50, 260, 150))
    assert crop.shape[:2] == (100, 200)
    assert crop.mean() > 40


def test_rotate_image_is_lossless():
    image = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    rotated = rotate_image(image, 90)
    assert rotated.shape == (4, 2, 3)
    assert np.array_equal(rotate_image(rotated, 270), image)
    assert np.array_equal(rotate_image(rotate_image(image, 180), 180), image)


def test_make_thumbnail_limits_size():
    image = np.zeros((400, 800, 3), dtype=np.uint8)
    thumb = make_thumbnail(image, 200)
    assert max(thumb.shape[:2]) == 200


def test_sanitize_filename_removes_illegal_characters():
    assert sanitize_filename('Iqama: 00*1?') == "Iqama_001"
    assert sanitize_filename("") == "document"
    assert sanitize_filename("  Passport  ") == "Passport"


def test_filename_template_and_uniqueness(tmp_path):
    exporter = ImageExporter(ExportOptions(filename_template="{type}_{identifier}_{page}_{sequence}"))
    card = DetectedCard(page_index=0, quad_norm=_quad(0, 0, 1, 1))
    card.set_type(DocumentType.DRIVING_LICENSE, 1.0)
    assert exporter.build_filename(card, 3, "scan", 2) == "DriversLicense_002_P01_03"

    used = set()
    first = exporter.unique_path(tmp_path, "Iqama_001_P01_01", used)
    second = exporter.unique_path(tmp_path, "Iqama_001_P01_01", used)
    assert first.name == "Iqama_001_P01_01.jpg"
    assert second.name == "Iqama_001_P01_01_2.jpg"


def test_export_writes_real_images(sample_pdf, tmp_path):
    from core.image_exporter import extract_card_image
    from core.pdf_loader import PdfDocument

    card = DetectedCard(page_index=0, quad_norm=_quad(0.1, 0.6, 0.3, 0.8))
    card.set_type(DocumentType.ISTIMARA, 1.0)
    with PdfDocument(sample_pdf) as document:
        options = ExportOptions(output_dir=str(tmp_path), image_format="png", render_dpi=200)
        summary = ImageExporter(options).export_cards(document, [card], tmp_path)
        assert summary.success_count == 1
        path = summary.files[0].path
        assert path.exists() and path.suffix == ".png"
        with Image.open(path) as saved:
            saved.verify()
        image = extract_card_image(document, card, dpi=200)
        assert image.shape[0] > 50 and image.shape[1] > 50


def test_face_detector_reports_no_backend_without_cascade_or_yunet(monkeypatch):
    """A build with neither detector must report why, and the blur is a no-op."""
    from types import SimpleNamespace

    from core import face_detector, image_exporter

    monkeypatch.setattr(face_detector, "cv2", SimpleNamespace(__version__="5.0.0"))
    monkeypatch.setattr(face_detector, "find_model", lambda: None)
    monkeypatch.setattr(face_detector, "_status_cache", None)

    backend, message = face_detector.status()
    assert backend == ""
    assert "5.0.0" in message
    assert face_detector.is_available() is False

    image = np.full((16, 16, 3), 200, dtype=np.uint8)
    original = image.copy()
    assert face_detector.detect_faces(image) == []
    assert np.array_equal(image_exporter.blur_faces(image), original)


def test_face_detector_prefers_yunet_when_the_model_is_present(monkeypatch, tmp_path):
    """OpenCV 5 path: the YuNet network is used once its ONNX model exists."""
    from core import face_detector

    model = tmp_path / face_detector.MODEL_FILENAME
    model.write_bytes(b"x" * (face_detector.MIN_MODEL_BYTES + 1))
    monkeypatch.setattr(face_detector, "model_candidates", lambda: [model])
    monkeypatch.setattr(face_detector, "_status_cache", None)

    backend, message = face_detector.status()
    assert backend == "yunet"
    assert face_detector.MODEL_FILENAME in message

    class _FakeDetector:
        def __init__(self) -> None:
            self.size = None

        def setInputSize(self, size):  # noqa: N802 - OpenCV naming
            self.size = size

        def detect(self, image):
            # x, y, w, h, 5 landmarks, score - the YuNet row layout
            faces = np.array(
                [
                    [10, 10, 40, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9],
                    [2, 2, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9],
                ],
                dtype=np.float32,
            )
            return len(faces), faces

    fake = _FakeDetector()
    monkeypatch.setattr(face_detector, "_get_detector", lambda: ("yunet", fake))

    image = np.zeros((60, 80, 3), dtype=np.uint8)
    boxes = face_detector.detect_faces(image, min_face_size=20)
    assert boxes == [(10, 10, 40, 40)], "tiny detections must be filtered out"
    assert fake.size == (80, 60), "the detector must be told the real image size"


def test_face_detector_finds_the_bundled_model(tmp_path, monkeypatch):
    """A clone (and a PyInstaller build) must find the shipped model offline."""
    import sys
    from pathlib import Path

    from core import face_detector

    bundled = (
        Path(face_detector.__file__).resolve().parents[1]
        / "assets"
        / "models"
        / face_detector.MODEL_FILENAME
    )
    assert any(Path(item) == bundled for item in face_detector.model_candidates())
    if bundled.is_file():
        assert face_detector.find_model() is not None

    # frozen layout: PyInstaller unpacks the payload into sys._MEIPASS
    payload = tmp_path / "frozen" / "assets" / "models" / face_detector.MODEL_FILENAME
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"x" * (face_detector.MIN_MODEL_BYTES + 1))
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "frozen"), raising=False)
    monkeypatch.setattr(face_detector, "model_folder", lambda: tmp_path / "empty")
    assert face_detector.find_model() == payload


def test_download_face_model_validates_and_stores_the_payload(tmp_path):
    from core import face_detector

    target = tmp_path / "models" / face_detector.MODEL_FILENAME
    payload = b"\x00" * (face_detector.MIN_MODEL_BYTES + 5)

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return payload

    written = face_detector.download_model(target, opener=lambda *a, **k: _Response())
    assert written == target and target.stat().st_size == len(payload)

    # an error page instead of the model must be rejected, not stored
    broken = tmp_path / "broken" / face_detector.MODEL_FILENAME

    class _ErrorResponse(_Response):
        def read(self):
            return b"<html>404</html>"

    with pytest.raises(RuntimeError):
        face_detector.download_model(broken, opener=lambda *a, **k: _ErrorResponse())
    assert not broken.exists()
    assert not list((tmp_path / "broken").glob("*.part"))



def test_export_skips_face_blur_when_the_cascade_is_missing(sample_pdf, tmp_path, monkeypatch):
    """Regression: an unavailable cascade aborted every file ("0 exported")."""
    from core import image_exporter
    from core.pdf_loader import PdfDocument

    reason = "face blur needs OpenCV's Haar cascade, which this build does not have"
    monkeypatch.setattr(image_exporter, "face_blur_support", lambda: (False, reason))

    def _must_not_run(*_args, **_kwargs):  # pragma: no cover - proof it is skipped
        raise AssertionError("blur_faces must not be called without a cascade")

    monkeypatch.setattr(image_exporter, "blur_faces", _must_not_run)

    card = DetectedCard(page_index=0, quad_norm=_quad(0.1, 0.6, 0.3, 0.8))
    with PdfDocument(sample_pdf) as document:
        options = ExportOptions(
            output_dir=str(tmp_path),
            image_format="png",
            render_dpi=150,
            blur_faces=True,
        )
        summary = image_exporter.ImageExporter(options).export_cards(document, [card], tmp_path)

    assert summary.errors == []
    assert summary.success_count == 1, "the export must still write the image"
    assert [item.path.name for item in summary.files if item.ok]
    assert summary.warnings == [reason]


def test_suggest_output_dir_creates_subfolder(tmp_path):
    folder = suggest_output_dir("C:/scans/trip 2026.pdf", str(tmp_path), True)

    assert folder.parent == tmp_path
    assert folder.name == "trip_2026"
    assert suggest_output_dir(None, str(tmp_path), True) == tmp_path


def test_document_type_helpers():
    assert DocumentType.from_value("passport") is DocumentType.PASSPORT
    assert DocumentType.from_value("Driver's License") is DocumentType.DRIVING_LICENSE
    assert DocumentType.from_value("nonsense") is DocumentType.UNKNOWN
    assert DocumentType.ISTIMARA.file_token == "VehicleRegistration"
    assert DocumentType.UNKNOWN.file_token == "Document"


def test_custom_document_type_registration():
    DocumentType.reset_custom_types()
    try:
        custom = DocumentType.register("Membership Card", "Membership Card", keywords=["MEMBERSHIP"])
        assert custom.token == "membership_card"
        assert custom.file_token == "MembershipCard"
        assert not custom.builtin
        assert custom.category == "custom"
        assert custom.is_known

        choices = DocumentType.choices()
        assert custom in choices
        assert choices[-1] is DocumentType.UNKNOWN  # Unknown always stays last
        assert DocumentType.from_value("membership_card") is custom
        assert DocumentType.from_value("Membership Card") is custom

        # built-in types can not be removed, custom ones can
        assert DocumentType.unregister("passport") is False
        assert DocumentType.unregister("membership_card") is True
        assert custom not in DocumentType.choices()
    finally:
        DocumentType.reset_custom_types()


def test_new_builtin_document_types_exist():
    expected = {
        "health_insurance": "Health Insurance Card",
        "bank_card": "Bank Card",
        "salary_card": "Salary Card",
        "national_id": "National ID",
    }
    builtins = {item.token: item for item in DocumentType.builtin_types()}
    for token, label in expected.items():
        assert token in builtins, f"{label} is not registered as a built-in type"
        assert builtins[token].display_name == label
        assert builtins[token].builtin is True
        assert builtins[token].category == "built-in"

    choices = DocumentType.choices()
    assert choices[-1] is DocumentType.UNKNOWN
    for token in expected:
        assert DocumentType.from_value(token) in choices


def test_builtin_types_cannot_be_removed_or_disabled_silently():
    assert DocumentType.unregister("bank_card") is False
    assert DocumentType.from_value("bank_card") is DocumentType.BANK_CARD
    assert DocumentType.UNKNOWN.enabled is True  # "Unknown" is never hideable


def test_types_can_be_disabled_and_enabled_again():
    DocumentType.set_disabled_tokens([])
    try:
        DocumentType.set_disabled_tokens(["bank_card", "national_id"])
        assert DocumentType.BANK_CARD.enabled is False
        assert DocumentType.NATIONAL_ID.enabled is False
        assert DocumentType.BANK_CARD not in DocumentType.choices()
        assert DocumentType.NATIONAL_ID not in DocumentType.choices()
        # still listed on the settings page and still resolvable
        assert DocumentType.BANK_CARD in DocumentType.all_choices()
        assert DocumentType.from_value("bank_card") is DocumentType.BANK_CARD
        assert DocumentType.PASSPORT in DocumentType.choices()
    finally:
        DocumentType.set_disabled_tokens([])
        assert DocumentType.BANK_CARD.enabled is True
        assert DocumentType.BANK_CARD in DocumentType.choices()


def test_settings_toggle_document_type():
    from app.settings import AppSettings

    DocumentType.set_disabled_tokens([])
    try:
        settings = AppSettings()
        assert settings.set_type_enabled("bank_card", False) is True
        assert settings.disabled_document_types == ["bank_card"]
        assert DocumentType.BANK_CARD.enabled is False
        assert settings.set_type_enabled("bank_card", False) is False  # already off
        assert settings.set_type_enabled("bank_card", True) is True
        assert settings.disabled_document_types == []
        assert DocumentType.BANK_CARD.enabled is True
        # "Unknown" can never be hidden
        assert settings.set_type_enabled("unknown", False) is False
        assert DocumentType.UNKNOWN.enabled is True

        # the state survives a settings round trip
        settings.set_type_enabled("salary_card", False)
        clone = AppSettings.from_dict(settings.to_dict())
        clone.apply_document_types()
        assert DocumentType.SALARY_CARD.enabled is False
    finally:
        DocumentType.set_disabled_tokens([])


def test_custom_types_avoid_builtin_tokens():
    DocumentType.reset_custom_types()
    try:
        entry = DocumentType.set_custom_types(
            [{"label": "Passport", "keywords": "X"}, {"label": "Visa", "keywords": "VISA, VISA NO"}]
        )
        tokens = [item.token for item in entry]
        assert tokens[0] != "passport"  # never shadows the built-in type
        assert DocumentType.PASSPORT.builtin is True
        assert DocumentType.from_value("visa") is entry[1]
        assert entry[1].keywords == ("VISA", "VISA NO")
    finally:
        DocumentType.reset_custom_types()


def test_settings_store_custom_document_types():
    from app.settings import AppSettings

    DocumentType.reset_custom_types()
    try:
        settings = AppSettings()
        entry = settings.add_custom_type("Work Permit", "WORK PERMIT, تصريح عمل")
        assert entry is not None
        assert DocumentType.from_value("work_permit").display_name == "Work Permit"

        clone = AppSettings.from_dict(settings.to_dict())
        DocumentType.reset_custom_types()
        clone.apply_custom_types()
        restored = DocumentType.from_value("work_permit")
        assert restored.display_name == "Work Permit"
        assert "WORK PERMIT" in restored.keywords

        assert clone.remove_custom_type("work_permit") is True
        assert DocumentType.from_value("work_permit") is DocumentType.UNKNOWN
    finally:
        DocumentType.reset_custom_types()


def test_custom_type_used_in_export_filename():
    DocumentType.reset_custom_types()
    try:
        DocumentType.set_custom_types([{"token": "membership_card", "label": "Membership Card"}])
        card = DetectedCard(page_index=0, quad_norm=_quad(0, 0, 1, 1))
        card.set_type(DocumentType.from_value("membership_card"), 1.0)
        exporter = ImageExporter(ExportOptions())
        assert exporter.build_filename(card, 1, "scan", 3).startswith("MembershipCard_")
    finally:
        DocumentType.reset_custom_types()


def test_classifier_uses_custom_keywords():
    DocumentType.reset_custom_types()
    try:
        DocumentType.set_custom_types(
            [{"token": "membership_card", "label": "Membership Card", "keywords": "MEMBERSHIP CARD, عضوية"}]
        )
        result = DocumentClassifier().classify(ocr_text="KINGDOM OF SAUDI ARABIA MEMBERSHIP CARD 1234567890")
        assert result.doc_type is DocumentType.from_value("membership_card")
        assert result.confidence >= 0.35
    finally:
        DocumentType.reset_custom_types()


def test_classifier_recognises_the_new_builtin_types():
    classifier = DocumentClassifier()
    assert classifier.classify(ocr_text="BUPA HEALTH INSURANCE CARD POLICY 123").doc_type is DocumentType.HEALTH_INSURANCE
    assert classifier.classify(ocr_text="AL RAJHI BANK CARD DEBIT").doc_type is DocumentType.BANK_CARD
    assert classifier.classify(ocr_text="SALARY CARD ACCOUNT 1234").doc_type is DocumentType.SALARY_CARD
    assert classifier.classify(ocr_text="KINGDOM OF SAUDI ARABIA NATIONAL ID 1023456789").doc_type is DocumentType.NATIONAL_ID


def test_disabled_type_is_not_assigned_by_the_classifier():
    classifier = DocumentClassifier()
    DocumentType.set_disabled_tokens([])
    try:
        assert classifier.classify(ocr_text="NATIONAL ID 1023456789").doc_type is DocumentType.NATIONAL_ID
        DocumentType.set_disabled_tokens(["national_id"])
        result = classifier.classify(ocr_text="NATIONAL ID 1023456789")
        assert result.doc_type is not DocumentType.NATIONAL_ID
    finally:
        DocumentType.set_disabled_tokens([])


def test_card_geometry_uses_page_size():
    card = DetectedCard(
        page_index=1,
        quad_norm=_quad(0.1, 0.1, 0.5, 0.25),
        page_size=(1000, 1400),
    )
    assert card.page_number == 2
    assert abs(card.aspect_ratio() - 400 / 210) < 0.05  # 400 x 210 px
    assert not card.is_portrait
    pixels = card.quad_pixels(1000, 1400)
    assert np.allclose(pixels[2], [500, 350])


def test_card_roundtrip_serialisation():
    card = DetectedCard(page_index=2, quad_norm=_quad(0.2, 0.2, 0.4, 0.3), confidence=0.7)
    card.set_type(DocumentType.IQAMA, 0.8)
    card.identifier = "1234567890"
    card.rotation = 90
    clone = DetectedCard.from_dict(card.to_dict())
    assert clone.doc_type is DocumentType.IQAMA
    assert clone.rotation == 90
    assert clone.identifier == "1234567890"
    assert np.allclose(clone.quad_norm, card.quad_norm)


def test_classifier_recognises_keywords():
    classifier = DocumentClassifier()
    assert classifier.classify(ocr_text="KINGDOM OF SAUDI ARABIA DRIVING LICENSE").doc_type is DocumentType.DRIVING_LICENSE
    assert classifier.classify(ocr_text="RESIDENT IDENTITY IQAMA").doc_type is DocumentType.IQAMA
    assert classifier.classify(ocr_text="VEHICLE REGISTRATION").doc_type is DocumentType.ISTIMARA
    assert classifier.classify(ocr_text="PASSPORT NO WE4110283").doc_type is DocumentType.PASSPORT
    unknown = classifier.classify(ocr_text="", aspect_ratio=1.5, area_ratio=0.05)
    assert unknown.doc_type is DocumentType.UNKNOWN


def test_classifier_uses_the_machine_readable_zone():
    """A MRZ line means passport even when no keyword is spelled correctly.

    The pattern looks for "<" characters, which the noise-tolerant text
    preparation removes - so the match has to happen on the raw text.
    """
    classifier = DocumentClassifier()
    mrz = "P<UTOSPECIMEN<<LUKA<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    assert classifier.classify(ocr_text=mrz).doc_type is DocumentType.PASSPORT
    # Some OCR engines read every "<" as a "«" glyph.
    angled = mrz.replace("<", "«")
    assert classifier.classify(ocr_text=angled).doc_type is DocumentType.PASSPORT


def test_mrz_without_text_preparation_matches_raw_characters():
    """Regression: the MRZ check must see the raw "<" characters."""
    classifier = DocumentClassifier()
    scores = classifier._keyword_scores("P UTO SPECIMEN", raw_text="P<UTO<<SPECIMEN<<<")
    assert scores.get(DocumentType.PASSPORT, 0.0) >= 1.2


def test_identifier_extraction():
    assert extract_identifier("PASSPORT WE4110283", DocumentType.PASSPORT) == "WE4110283"
    assert extract_identifier("ID 2567251562", DocumentType.IQAMA) == "2567251562"
    assert extract_identifier("", DocumentType.IQAMA) == ""


def test_ocr_engine_requires_pretrained_easyocr_models(tmp_path, monkeypatch):
    """The app must work offline: no models on disk means OCR is unavailable."""
    from core.ocr_engine import OcrEngine

    monkeypatch.setenv("EASYOCR_MODULE_PATH", str(tmp_path))
    assert OcrEngine.easyocr_model_dir() == tmp_path / "model"
    assert OcrEngine.easyocr_models_available(("en",)) is False

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "craft_mlt_25k.pth").write_bytes(b"stub")  # detector only
    assert OcrEngine.easyocr_models_available(("en",)) is False

    (model_dir / "english_g2.pth").write_bytes(b"stub")  # + recogniser
    assert OcrEngine.easyocr_models_available(("en",)) is True


def test_ocr_engine_explains_missing_easyocr_models(tmp_path, monkeypatch):
    from core.ocr_engine import OcrEngine

    if not OcrEngine.easyocr_available():
        import pytest

        pytest.skip("easyocr is not installed")

    monkeypatch.setenv("EASYOCR_MODULE_PATH", str(tmp_path))
    assert "model files missing" in OcrEngine.describe_availability()

    engine = OcrEngine(("en",), prefer="easyocr")
    if OcrEngine.tesseract_available():
        return  # a Tesseract installation legitimately takes over
    assert engine.backend == ""
    assert engine.is_available is False
    assert "never downloads" in engine.init_error


def test_ocr_engine_finds_real_models_when_present():
    from core.ocr_engine import OcrEngine

    if not OcrEngine.easyocr_available():
        import pytest

        pytest.skip("easyocr is not installed")
    if not OcrEngine.easyocr_models_available():
        import pytest

        pytest.skip("easyocr models are not downloaded on this machine")
    engine = OcrEngine(("en",))
    if not OcrEngine.tesseract_available():
        assert engine.backend == "easyocr"
        assert engine.is_available is True


def test_rotate_quad_keeps_centre_and_size():
    from core.perspective_corrector import rotate_quad

    quad = _quad(100, 100, 300, 200)
    rotated = rotate_quad(quad, 90)
    assert np.allclose(rotated.mean(axis=0), quad.mean(axis=0), atol=1e-3)
    sides = [float(np.linalg.norm(quad[i] - quad[(i + 1) % 4])) for i in range(4)]
    rotated_sides = [float(np.linalg.norm(rotated[i] - rotated[(i + 1) % 4])) for i in range(4)]
    assert np.allclose(sorted(sides), sorted(rotated_sides), atol=1e-2)


def test_fine_rotation_changes_the_effective_quad():
    card = DetectedCard(page_index=0, quad_norm=_quad(0.2, 0.2, 0.6, 0.4), page_size=(1000, 1400))
    assert np.allclose(card.effective_quad_norm(), card.quad_norm)

    card.fine_rotation = 5.0
    rotated = card.effective_quad_norm()
    assert not np.allclose(rotated, card.quad_norm)
    # a rotation keeps the centre and the diagonal length
    assert np.allclose(rotated.mean(axis=0), card.quad_norm.mean(axis=0), atol=1e-4)
    before = np.linalg.norm(card.quad_pixels(1000, 1400)[0] - card.quad_pixels(1000, 1400)[2])
    after = np.linalg.norm(
        card.effective_quad_pixels(1000, 1400)[0] - card.effective_quad_pixels(1000, 1400)[2]
    )
    assert abs(before - after) < 0.5

    card.bake_fine_rotation()
    assert card.fine_rotation == 0.0
    assert np.allclose(card.quad_norm, rotated, atol=1e-6)


def test_fine_rotation_is_clamped_and_serialised():
    card = DetectedCard(page_index=0, quad_norm=_quad(0.1, 0.1, 0.5, 0.3), fine_rotation=90)
    assert card.fine_rotation == 45.0
    card.fine_rotation = -3.5
    clone = DetectedCard.from_dict(card.to_dict())
    assert abs(clone.fine_rotation + 3.5) < 1e-6
    assert clone.copy_card().fine_rotation == card.fine_rotation


def test_resize_quad_scales_about_the_centre_along_its_axes():
    from core.perspective_corrector import expand_quad_axes, quad_frame, resize_quad

    quad = _quad(100, 100, 300, 200)  # 200 x 100
    centre, width_axis, height_axis = quad_frame(quad)
    assert np.allclose(centre, [200.0, 150.0], atol=1e-3)
    assert np.isclose(np.linalg.norm(width_axis), 1.0)
    assert np.isclose(np.linalg.norm(height_axis), 1.0)

    wider = resize_quad(quad, 1.5, 1.0)
    assert np.allclose(wider.mean(axis=0), centre, atol=1e-3)
    assert abs(np.linalg.norm(wider[1] - wider[0]) - 300.0) < 1e-2
    assert abs(np.linalg.norm(wider[3] - wider[0]) - 100.0) < 1e-2

    # delta is added on each side: 200 + 2*10 and 100 + 2*5
    grown = expand_quad_axes(quad, 10.0, 5.0)
    assert abs(np.linalg.norm(grown[1] - grown[0]) - 220.0) < 0.5
    assert abs(np.linalg.norm(grown[3] - grown[0]) - 110.0) < 0.5

    # negative deltas shrink it again
    shrunk = expand_quad_axes(quad, -20.0, -10.0)
    assert abs(np.linalg.norm(shrunk[1] - shrunk[0]) - 160.0) < 0.5
    assert abs(np.linalg.norm(shrunk[3] - shrunk[0]) - 80.0) < 0.5


def test_resize_quad_keeps_a_rotated_quad_rotated():
    from core.perspective_corrector import expand_quad_axes, rotate_quad

    quad = _quad(100, 100, 400, 250)
    rotated = rotate_quad(quad, 25.0)
    resized = expand_quad_axes(rotated, 12.0, 12.0)

    def direction(points):
        vector = points[1] - points[0]
        return vector / np.linalg.norm(vector)

    assert np.allclose(direction(resized), direction(rotated), atol=1e-4)
    assert np.allclose(resized.mean(axis=0), rotated.mean(axis=0), atol=1e-3)
    assert np.linalg.norm(resized[1] - resized[0]) > np.linalg.norm(rotated[1] - rotated[0])


def test_tesseract_language_string_maps_iso_codes():
    from core.ocr_engine import easyocr_language_list, tesseract_language_string

    assert tesseract_language_string(["en"]) == "eng"
    assert tesseract_language_string(["en", "ar"]) == "eng+ara"
    assert tesseract_language_string(["AR"]) == "ara"
    # duplicates collapse, order is kept
    assert tesseract_language_string(["en", "eng", "en"]) == "eng"
    # regional variants and Tesseract names pass through
    assert tesseract_language_string(["en_US", "de"]) == "eng+deu"
    assert tesseract_language_string([]) == "eng"
    assert tesseract_language_string(["xx"]) == "xx"

    assert easyocr_language_list(["EN", "ar", "en"]) == ["en", "ar"]
    assert easyocr_language_list([]) == ["en"]


def test_ocr_setup_hints_explain_how_to_install():
    from core.ocr_engine import OcrEngine

    tesseract_hint = OcrEngine.tesseract_setup_hint()
    assert "tesseract" in tesseract_hint.lower()
    assert "pip install pytesseract" in tesseract_hint
    assert "arabic" in tesseract_hint.lower()

    easy_hint = OcrEngine.easyocr_setup_hint()
    assert str(OcrEngine.easyocr_model_dir()) in easy_hint
    assert "easyocr.Reader" in easy_hint


def test_builtin_type_accepts_user_keywords():
    """Extra keywords must work for built-in types too (any language)."""
    from core.document_classifier import DocumentClassifier, builtin_keywords

    DocumentType.reset_extra_keywords()
    try:
        classifier = DocumentClassifier()
        # a made-up phrase that no built-in rule knows
        assert classifier.classify(ocr_text="FUHRERSCHEIN MUSTERMANN").doc_type is not DocumentType.DRIVING_LICENSE

        DocumentType.set_extra_keywords({"driving_license": ["FÜHRERSCHEIN", "FUHRERSCHEIN"]})
        result = classifier.classify(ocr_text="FUHRERSCHEIN MUSTERMANN 12345")
        assert result.doc_type is DocumentType.DRIVING_LICENSE
        assert result.confidence >= 0.35

        # the built-in keywords still work and are reported separately
        assert "DRIVING LICENSE" in builtin_keywords(DocumentType.DRIVING_LICENSE)
        assert DocumentType.DRIVING_LICENSE.user_keywords == ("FÜHRERSCHEIN", "FUHRERSCHEIN")
        assert DocumentType.DRIVING_LICENSE.all_keywords == ("FÜHRERSCHEIN", "FUHRERSCHEIN")
    finally:
        DocumentType.reset_extra_keywords()


def test_settings_store_keyword_overrides():
    from app.settings import AppSettings

    DocumentType.reset_extra_keywords()
    try:
        settings = AppSettings()
        assert settings.type_keywords("passport") == []

        assert settings.set_type_keywords("passport", "جواز, TRAVEL DOCUMENT") is True
        assert settings.type_keywords("passport") == ["جواز", "TRAVEL DOCUMENT"]
        assert DocumentType.PASSPORT.user_keywords == ("جواز", "TRAVEL DOCUMENT")

        # persisted through the JSON round trip and re-applied
        clone = AppSettings.from_dict(settings.to_dict())
        assert clone.type_keywords("passport") == ["جواز", "TRAVEL DOCUMENT"]

        # newlines work too, and an empty value removes the override
        assert settings.set_type_keywords("passport", "JOAZ\nPASSEPORT") is True
        assert settings.type_keywords("passport") == ["JOAZ", "PASSEPORT"]
        assert settings.set_type_keywords("passport", "") is True
        assert settings.type_keywords("passport") == []
        assert DocumentType.PASSPORT.user_keywords == ()
        # removing something that is not there is a no-op
        assert settings.set_type_keywords("passport", "") is False
    finally:
        DocumentType.reset_extra_keywords()


def test_ocr_text_is_stored_and_serialised():
    card = DetectedCard(page_index=0, quad_norm=_quad(0, 0, 1, 1))
    assert card.ocr_text == ""
    card.ocr_text = "KINGDOM OF SAUDI ARABIA PASSPORT WE4110283"
    clone = DetectedCard.from_dict(card.to_dict())
    assert clone.ocr_text == card.ocr_text
    assert clone.copy_card().ocr_text == card.ocr_text


def test_ocr_diagnostic_report_mentions_the_engine():
    from core.ocr_engine import OcrEngine

    report = OcrEngine.diagnostic_report(("en",), prefer="auto", smoke_test=False)
    assert "Tesseract found" in report
    assert "EasyOCR model folder" in report
    assert "Engine used" in report
    assert str(OcrEngine.easyocr_model_dir()) in report


def test_keyword_tables_have_no_duplicates():
    """Guard against copy/paste duplicates in the classifier keyword tables."""
    from core.document_classifier import (
        KEYWORD_RULES,
        _ARABIC_RULES,
        prepare_ocr_text,
    )

    for name, rules in (("KEYWORD_RULES", KEYWORD_RULES), ("_ARABIC_RULES", _ARABIC_RULES)):
        raw = [keyword for keyword, _type, _weight in rules]
        assert len(raw) == len(set(raw)), f"{name} contains a duplicate keyword: {raw}"
        normalised = [prepare_ocr_text(keyword) for keyword in raw]
        assert all(normalised), f"{name} contains a keyword that normalises to nothing"
        assert len(normalised) == len(set(normalised)), (
            f"{name} contains duplicates after OCR normalisation: {normalised}"
        )

    # no keyword is listed twice across the two tables
    english = {prepare_ocr_text(keyword) for keyword, _t, _w in KEYWORD_RULES}
    arabic = {prepare_ocr_text(keyword) for keyword, _t, _w in _ARABIC_RULES}
    assert not (english & arabic)

    # every rule points at a real type: a rule for "Unknown" would never be scored
    for name, rules in (("KEYWORD_RULES", KEYWORD_RULES), ("_ARABIC_RULES", _ARABIC_RULES)):
        for _keyword, doc_type, weight in rules:
            assert doc_type is not DocumentType.UNKNOWN, f"{name} has a dead 'Unknown' rule"
            assert weight > 0, f"{name} has a non-positive weight"


def test_keyword_overlaps_stay_within_one_document_type():
    """Keywords may be prefixes of each other, but never across two types."""
    from core.document_classifier import KEYWORD_RULES, _ARABIC_RULES, prepare_ocr_text

    rules = list(KEYWORD_RULES) + list(_ARABIC_RULES)
    for short, short_type, _w in rules:
        prepared_short = prepare_ocr_text(short)
        for long, long_type, _w2 in rules:
            if short == long:
                continue
            if prepared_short and prepared_short in prepare_ocr_text(long):
                assert short_type is long_type, (
                    f"'{short}' ({short_type.token}) is contained in '{long}' "
                    f"({long_type.token}) - a hit would score both types"
                )


def test_ocr_report_tells_which_engine_ran():
    from typing import cast

    from core.ocr_engine import OcrEngine
    from workers.detection_worker import ocr_report

    class FakeEngine:
        def __init__(self, backend: str) -> None:
            self.backend = backend

    fake_tesseract = cast(OcrEngine, FakeEngine("tesseract"))
    fake_easyocr = cast(OcrEngine, FakeEngine("easyocr"))
    fake_empty = cast(OcrEngine, FakeEngine(""))

    assert ocr_report(None, 0, 4) == ""
    assert ocr_report(fake_tesseract, 4, 4) == "tesseract (4/4 cards read)"
    assert ocr_report(fake_easyocr, 0, 3) == "easyocr (0/3 cards read)"
    assert ocr_report(fake_tesseract, 0, 0) == "tesseract"
    assert "no engine" in ocr_report(fake_empty, 0, 2)


def test_check_ocr_command_reports_a_missing_engine(monkeypatch, capsys, tmp_path):
    import cli
    from core.ocr_engine import OcrEngine

    monkeypatch.setenv("EASYOCR_MODULE_PATH", str(tmp_path))
    monkeypatch.setattr(OcrEngine, "tesseract_available", staticmethod(lambda: False))
    monkeypatch.setattr(OcrEngine, "easyocr_available", staticmethod(lambda: False))

    assert cli.main(["--check-ocr"]) == 1
    output = capsys.readouterr().out
    assert "OCR diagnostics" in output
    assert "Tesseract found      : False" in output
    assert "pip install pytesseract" in output
    assert str(OcrEngine.easyocr_model_dir()) in output


def test_check_ocr_command_accepts_no_pdf_without_the_flag():
    import cli

    parser = cli.build_parser()
    args = parser.parse_args(["--check-ocr", "--ocr-languages", "en,ar"])
    assert args.check_ocr is True
    assert args.ocr_languages == "en,ar"
    # without --check-ocr a PDF is required at runtime (not enforced by argparse,
    # so main() reports it instead of dying inside argparse)
    assert parser.parse_args([]).pdfs == []


def _write_card_image(path, size=(900, 600), dpi=300, cards=1):
    """Write a synthetic scan: a light page with one or more grey "cards"."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", size, (252, 252, 252))
    draw = ImageDraw.Draw(image)
    width, height = size
    for index in range(cards):
        left = int(width * 0.08) + index * int(width * 0.34)
        top = int(height * 0.2)
        draw.rectangle(
            [left, top, left + int(width * 0.28), top + int(height * 0.55)],
            fill=(205, 212, 224),
            outline=(120, 130, 145),
            width=3,
        )
    image.save(path, dpi=(dpi, dpi))
    return path


def test_image_file_is_loaded_as_a_single_page(tmp_path):
    from core.pdf_loader import PdfDocument

    path = _write_card_image(tmp_path / "scan.png", size=(900, 600), dpi=300)
    with PdfDocument(path) as document:
        assert document.is_image is True
        assert document.page_count == 1
        assert document.native_dpi(0) == 300
        page = document.page_info(0)
        assert round(page.width_pt) == 216   # 900 px at 300 dpi
        assert round(page.height_pt) == 144  # 600 px at 300 dpi
        full = document.render_page(0, dpi=300)
        assert full.shape == (600, 900, 3)
        # a lower dpi is honoured (preview), higher is capped to the native size
        assert document.render_page(0, dpi=150).shape == (300, 450, 3)
        assert document.render_page(0, dpi=600).shape == (600, 900, 3)
        grey = document.render_page(0, dpi=300, grayscale=True)
        assert grey.ndim == 2 and grey.shape == (600, 900)


def test_multipage_tiff_yields_one_page_per_frame(tmp_path):
    from PIL import Image

    from core.pdf_loader import PdfDocument

    pages = [
        Image.new("RGB", (800, 500), (255, 255, 255)),
        Image.new("RGB", (1000, 700), (238, 238, 238)),
    ]
    path = tmp_path / "scanned.tiff"
    pages[0].save(path, save_all=True, append_images=pages[1:], dpi=(200, 200))

    with PdfDocument(path) as document:
        assert document.page_count == 2
        assert document.native_dpi(0) == 200
        assert document.render_page(0).shape == (500, 800, 3)
        assert document.render_page(1).shape == (700, 1000, 3)


def test_unsupported_and_broken_files_are_rejected(tmp_path):
    from core.pdf_loader import PdfDocument, PdfError

    notes = tmp_path / "notes.txt"
    notes.write_text("not a scan", encoding="utf-8")
    with pytest.raises(PdfError):
        PdfDocument(notes)

    broken = tmp_path / "broken.png"
    broken.write_bytes(b"this is not a png")
    with pytest.raises(PdfError):
        PdfDocument(broken)

    with pytest.raises(PdfError):
        PdfDocument(tmp_path / "missing.pdf")


def test_supported_file_helper():
    from core.pdf_loader import is_supported_file, supported_file_filter

    for name in ("a.pdf", "a.PNG", "b.jpg", "c.jpeg", "d.tif", "e.tiff", "f.bmp", "g.webp"):
        assert is_supported_file(name), name
    for name in ("a.txt", "a.docx", "a", ""):
        assert not is_supported_file(name), name

    patterns = supported_file_filter()
    assert "*.pdf" in patterns and "*.png" in patterns and "*.tiff" in patterns


def test_settings_path_uses_the_application_folder(tmp_path, monkeypatch):
    """Settings live in <config>/DocumentCardExtractor/settings.json."""
    import json

    import app.settings as settings_module

    monkeypatch.setattr(
        settings_module.QStandardPaths,
        "writableLocation",
        staticmethod(lambda *_args, **_kwargs: str(tmp_path)),
    )

    path = settings_module.settings_path()
    assert settings_module.APP_FOLDER_NAME == "DocumentCardExtractor"
    assert path == tmp_path / "DocumentCardExtractor" / "settings.json"

    # saving writes exactly there and loading reads it back
    settings = settings_module.AppSettings()
    settings.render_dpi = 321
    settings.recent_files = ["C:/scans/a.pdf"]
    settings.save()
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["render_dpi"] == 321
    restored = settings_module.AppSettings.load()
    assert restored.render_dpi == 321
    assert restored.recent_files == ["C:/scans/a.pdf"]


def test_corrupt_settings_file_is_kept_for_diagnosis(tmp_path, monkeypatch):
    """A broken settings.json is backed up, not silently overwritten or lost."""
    import app.settings as settings_module

    monkeypatch.setattr(
        settings_module.QStandardPaths,
        "writableLocation",
        staticmethod(lambda *_args, **_kwargs: str(tmp_path)),
    )

    path = settings_module.settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ this is not json", encoding="utf-8")

    settings = settings_module.AppSettings.load()
    assert settings.render_dpi == 300  # defaults, the broken file was discarded
    backup = path.parent / "settings.invalid.json"
    assert backup.read_text(encoding="utf-8") == "{ this is not json"
    assert not path.exists()  # will be re-created cleanly on the next save


def test_settings_save_failure_keeps_the_previous_file(tmp_path, monkeypatch):
    """A failed write must not truncate the previous settings and is reported."""
    import json

    import app.settings as settings_module

    monkeypatch.setattr(
        settings_module.QStandardPaths,
        "writableLocation",
        staticmethod(lambda *_args, **_kwargs: str(tmp_path)),
    )

    settings = settings_module.AppSettings()
    settings.render_dpi = 321
    assert settings.save() is True
    path = settings_module.settings_path()

    def broken_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(settings_module.os, "replace", broken_replace)
    settings.render_dpi = 555
    assert settings.save() is False
    # the previous, valid settings file survives untouched
    assert json.loads(path.read_text(encoding="utf-8"))["render_dpi"] == 321


def test_history_add_reports_a_write_failure(tmp_path, monkeypatch):
    import app.history as history_module

    store = history_module.HistoryStore(tmp_path / "history.json")
    monkeypatch.setattr(history_module, "_write_json_atomic", lambda *_args, **_kwargs: False)
    assert store.add(history_module.HistoryEntry(document_count=3)) is False
    # the entry itself was still registered in memory
    assert store.totals()["documents"] == 3


def test_settings_path_collapses_a_duplicated_folder(tmp_path, monkeypatch):
    """Qt can return <config>/<App>/<App> when an organisation name is set."""
    import app.settings as settings_module

    doubled = tmp_path / settings_module.APP_FOLDER_NAME / settings_module.APP_FOLDER_NAME
    monkeypatch.setattr(
        settings_module.QStandardPaths,
        "writableLocation",
        staticmethod(lambda *_args, **_kwargs: str(doubled)),
    )

    path = settings_module.settings_path()
    assert path == tmp_path / settings_module.APP_FOLDER_NAME / "settings.json"


def test_orientation_detector_handles_rotations():
    from core.orientation_detector import OrientationDetector

    # A fake card made of horizontal text lines.
    card = np.full((240, 380, 3), 235, dtype=np.uint8)
    for row in range(30, 200, 26):
        card[row : row + 9, 30:340] = 40
        for gap in range(40, 330, 34):
            card[row : row + 9, gap : gap + 8] = 235

    detector = OrientationDetector()
    upright = detector.detect(card)
    assert upright.rotation == 0
    sideways = detector.detect(rotate_image(card, 90))
    assert sideways.rotation == 90
    assert sideways.confident is False  # the 0/180 flip stays undetermined


def test_effective_dpi_never_upscales_beyond_native():
    """`cap_to_native=True` must not floor a low-dpi source UP to 150 dpi."""
    from core.pdf_loader import PdfDocument

    document = PdfDocument.__new__(PdfDocument)  # no file behind it
    native = {"dpi": 96}
    document.native_dpi = lambda index: native["dpi"]

    # A 96 dpi screenshot renders at ~96 dpi, not at 150 dpi (= upscaling).
    assert document.effective_dpi(0, 300, cap_to_native=True) == 101
    assert document.effective_dpi(0, 150, cap_to_native=True) == 101
    # A normal scan stays at the requested dpi (small headroom is allowed).
    native["dpi"] = 300
    assert document.effective_dpi(0, 300, cap_to_native=True) == 300
    native["dpi"] = 400
    assert document.effective_dpi(0, 300, cap_to_native=True) == 300
    # Without the cap the request passes straight through.
    native["dpi"] = 72
    assert document.effective_dpi(0, 800, cap_to_native=False) == 800


def test_refine_identifier_uses_barcode_values():
    """QR/barcode contents override OCR regex identifiers when they look like IDs."""
    from core.barcode_reader import BarcodeReadout, BarcodeResult, refine_identifier

    class _FakeReader:
        def __init__(self, best=""):
            self.is_available = True
            self._best = best

        def read(self, image):
            results = []
            if self._best:
                results.append(
                    BarcodeResult(data=self._best, type="QRCODE", rect=(0, 0, 10, 10), polygon=[])
                )
            return BarcodeReadout(barcodes=results, best_identifier=self._best)

    card = DetectedCard(page_index=0, quad_norm=_quad(0.1, 0.1, 0.6, 0.6))
    card.set_type(DocumentType.IQAMA, 0.5)
    image = np.zeros((40, 40, 3), dtype=np.uint8)

    # No OCR identifier -> the QR fills the gap.
    reader = _FakeReader("2567251562")
    assert refine_identifier(card, image, reader) is True
    assert card.identifier == "2567251562"

    # A weak OCR identifier is replaced by the authoritative QR value.
    card.identifier = "12"
    assert refine_identifier(card, image, _FakeReader("2567251562")) is True
    assert card.identifier == "2567251562"

    # Text that is not an ID pattern is ignored - the OCR value survives.
    card.identifier = "GOOD123456"
    assert refine_identifier(card, image, _FakeReader("WELCOME")) is False
    assert card.identifier == "GOOD123456"

    # An unavailable reader must not touch the identifier.
    offline = _FakeReader("2567251562")
    offline.is_available = False
    assert refine_identifier(card, image, offline) is False
    assert card.identifier == "GOOD123456"

    # A successful QR read overrides a plain OCR regex hit (more reliable).
    assert refine_identifier(card, image, _FakeReader("2567251562")) is True
    assert card.identifier == "2567251562"

