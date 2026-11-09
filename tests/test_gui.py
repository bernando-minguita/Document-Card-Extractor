"""End to end GUI test: load the sample PDF, detect, edit and export (offscreen)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QRectF
from PySide6.QtWidgets import QMessageBox

from app.history import HistoryStore
from app.main_window import MainWindow
from app.settings import AppSettings
from core.models import DocumentType


@pytest.fixture
def window(qapp, tmp_path, monkeypatch):
    """A MainWindow whose dialogs never block and whose state stays in tmp_path."""
    for name in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok))
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))

    settings = AppSettings()
    settings.preview_dpi = 80
    settings.render_dpi = 200
    settings.output_dir = str(tmp_path / "out")
    settings.open_output_after_export = False
    settings.create_subfolder_per_pdf = True
    settings.save = lambda: True  # never touch the real user profile

    main = MainWindow(settings)
    main.history = HistoryStore(tmp_path / "history.json")
    yield main
    main.close()


def pump_until(predicate, timeout: float = 180.0) -> bool:
    """Run the Qt event loop until *predicate* is true (or the timeout passes)."""
    from PySide6.QtWidgets import QApplication

    deadline = time.time() + timeout
    while time.time() < deadline:
        QApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    return False


def flush_events(rounds: int = 10, delay: float = 0.01) -> None:
    """Let deferred (zero-timer) work, modal dialogs and repaints complete."""
    from PySide6.QtWidgets import QApplication

    for _ in range(rounds):
        QApplication.processEvents()
        time.sleep(delay)


def test_window_builds(window):
    from app.main_window import APP_NAME

    assert APP_NAME == "Document Card Extractor"
    assert window.windowTitle().startswith(APP_NAME)
    assert window._gallery is not None
    assert window._view is not None
    for index in range(4):
        window._switch_page(index)


def test_icons_stay_readable_on_coloured_buttons(qapp):
    """White icons on primary/danger buttons, dark strokes elsewhere."""

    from PySide6.QtGui import QIcon

    from app.icons import BUTTON_ICON_SIZE, button_icon, icon

    def brightness(qicon, mode: QIcon.Mode = QIcon.Mode.Normal) -> float:
        image = qicon.pixmap(BUTTON_ICON_SIZE, BUTTON_ICON_SIZE, mode).toImage()
        total = count = 0
        for y in range(image.height()):
            for x in range(image.width()):
                colour = image.pixelColor(x, y)
                if colour.alpha() > 40:
                    total += (colour.red() + colour.green() + colour.blue()) / 3.0
                    count += 1
        assert count > 10, "the icon rendered nothing"
        return total / count

    dark = brightness(icon("save"))
    assert dark < 90, "the default icon is too light for a white button"

    for variant in ("primary", "danger"):
        assert brightness(button_icon("save", variant)) > 200, (
            f"icons on {variant} buttons must be white"
        )

    # disabled buttons use a lighter stroke than normal ones
    plain = icon("save")
    assert brightness(plain, QIcon.Mode.Disabled) > dark + 40


def test_icons_are_cached(qapp):
    """Icon building renders 14 SVG pixmaps; equal requests must reuse it."""
    from app.icons import icon

    first = icon("save")
    assert icon("save") is first
    # a different colour/filled combination must not share the cache entry
    assert icon("save", color="#ffffff") is not first
    assert icon("save", filled=True) is not first


def test_load_image_file_in_the_window(window, tmp_path):
    """An image file behaves like a one-page document in the GUI."""
    from PIL import Image, ImageDraw

    path = tmp_path / "photo_pages.jpg"
    image = Image.new("RGB", (1200, 900), (248, 248, 248))
    draw = ImageDraw.Draw(image)
    draw.rectangle([100, 150, 520, 420], fill=(200, 208, 220), outline=(110, 120, 135), width=4)
    image.save(path, dpi=(300, 300))

    window.show()
    flush_events(2)
    assert window.load_pdf(str(path)) is True

    session = window.session
    assert session is not None
    assert session.path == path
    assert session.page_count == 1
    assert session.label().startswith(path.name)
    assert pump_until(lambda: window._page_image is not None)
    height, width = window._page_image.shape[:2]
    assert abs(width / height - 1200 / 900) < 0.02, "the image aspect ratio was not kept"
    # detection runs on the image exactly like on a PDF page
    window.start_detection()
    assert pump_until(lambda: not window._busy)
    assert window.session.cards, "nothing was detected on the image"
    assert all(card.page_index == 0 for card in window.session.cards)


def test_window_title_does_not_duplicate_the_app_name(window):
    """Qt appends the application display name to every window title."""
    from pathlib import Path

    from app.main_window import APP_NAME

    assert window.windowTitle().count(APP_NAME) == 1

    # main.py must not set a display name, otherwise the native title becomes
    # "Document Card Extractor 1.0.0 - Document Card Extractor"
    source = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
    assert "setApplicationDisplayName" not in source


def test_dashboard_recent_files_keep_their_label(window, sample_pdf):
    """Regression: selecting a recent file must not make its label invisible."""
    from app.styles import STYLESHEET

    window.settings.recent_files = [str(sample_pdf)]
    window._refresh_dashboard()

    assert window._recent_list.count() == 1
    item = window._recent_list.item(0)
    assert sample_pdf.name in item.text()

    window._recent_list.setCurrentRow(0)  # what a single click does
    assert sample_pdf.name in window._recent_list.item(0).text()

    # The theme must define a visible text colour for the selected state,
    # otherwise Qt falls back to the (white) highlight colour.
    assert "RecentList::item:selected" in STYLESHEET
    selected_block = STYLESHEET.split("RecentList::item:selected")[1][:160]
    assert "color:" in selected_block
    assert "white" not in selected_block.lower()


def test_full_workflow(window, sample_pdf, tmp_path):
    assert window.load_pdf(str(sample_pdf))
    session = window.session
    assert session is not None and session.page_count == 1
    assert window._page_spin.maximum() == 1
    assert window._view.page_size[0] > 0  # the page was rendered

    window.start_detection()
    assert pump_until(lambda: not window._busy and window._detection_worker is None)

    assert len(session.cards) == 4
    assert len(session.detected_pages) == 1
    assert window._gallery.count() == 4
    assert all(card.thumbnail is not None and card.thumbnail.size for card in session.cards)
    assert window._btn_export.isEnabled()


def test_card_editing_operations(window, sample_pdf):
    window.load_pdf(str(sample_pdf))
    window.start_detection()
    assert pump_until(lambda: not window._busy)
    session = window.session
    card = session.cards[0]

    window.rotate_card(card.card_id, 90)
    assert card.rotation == 90
    window.rotate_card(card.card_id, 270)
    assert card.rotation == 0

    window.set_card_type(card.card_id, DocumentType.PASSPORT)
    assert card.doc_type is DocumentType.PASSPORT

    window.toggle_card_included(card.card_id)
    assert card.included is False
    assert len(session.included_cards()) == 3
    window.toggle_card_included(card.card_id)
    assert card.included is True

    # A manual rectangle adds a card (the detector "missed" region).
    window._on_manual_rect(QRectF(0.05, 0.05, 0.2, 0.15))
    assert len(session.cards) == 5
    manual = [item for item in session.cards if item.source == "manual"]
    assert len(manual) == 1
    assert manual[0].quad_norm[2][0] > 0.2

    window.remove_card(manual[0].card_id)
    assert len(session.cards) == 4

    # Editing the crop quad of a detected card keeps it consistent.
    updated = card.quad_norm.copy()
    updated[2] -= 0.01
    window._on_quad_edited(card.card_id, updated)
    assert abs(float(card.quad_norm[2][0]) - float(updated[2][0])) < 1e-6


def test_export_creates_images_and_history(window, sample_pdf, tmp_path):
    window.load_pdf(str(sample_pdf))
    window.start_detection()
    assert pump_until(lambda: not window._busy)
    session = window.session
    assert len(session.cards) == 4

    window.export_all()
    assert pump_until(lambda: not window._busy)
    assert pump_until(lambda: window._export_worker is None)

    output = Path(window.settings.output_dir) / sample_pdf.stem
    files = sorted(output.glob("*.jpg"))
    assert len(files) == 4, f"expected 4 exported images in {output}, found {len(files)}"
    for path in files:
        assert path.stat().st_size > 5000

    entries = window.history.entries
    assert entries and entries[0].document_count == 4
    # page_count counts the *pages* that produced documents (one page here),
    # not the number of documents.
    assert entries[0].page_count == 1
    assert len(entries[0].files) == 4


def test_crop_editor_dialog_opens(window, sample_pdf):
    from PySide6.QtWidgets import QApplication

    from widgets.crop_editor import CropEditorDialog

    window.load_pdf(str(sample_pdf))
    window.start_detection()
    assert pump_until(lambda: not window._busy)
    card = window.session.cards[0]

    dialog = CropEditorDialog(sample_pdf, card, window)
    assert dialog.card.quad_norm.shape == (4, 2)
    dialog.show()
    QApplication.processEvents()

    label = dialog._preview
    source = label.source()
    assert not source.isNull()

    # The preview must always fit inside the label - never be clipped - and keep
    # the aspect ratio of the extracted card (High-DPI displays shrink the panel).
    for width, height in ((900, 620), (1240, 780), (1600, 900)):
        dialog.resize(width, height)
        QApplication.processEvents()
        shown = label.pixmap()
        assert shown is not None and not shown.isNull()
        assert shown.width() <= label.width(), f"preview clipped at {width}x{height}"
        assert shown.height() <= label.height(), f"preview clipped at {width}x{height}"
        source_ratio = source.width() / source.height()
        shown_ratio = shown.width() / shown.height()
        assert abs(source_ratio - shown_ratio) < 0.05, "preview aspect ratio distorted"

    # Rotating to a tall image must be fitted as well.
    dialog._rotate(90)
    QApplication.processEvents()
    shown = label.pixmap()
    assert shown.height() <= label.height() and shown.width() <= label.width()
    assert shown.height() > shown.width()  # portrait after the rotation
    assert dialog.card.rotation == 90
    dialog.close()



def test_settings_page_roundtrip(window):
    window._switch_page(3)
    window._set_render_dpi.setValue(350)
    window._set_sensitivity.setValue(7)
    window._set_min_area.setValue(1.5)
    window._save_settings_from_form()
    assert window.settings.render_dpi == 350
    assert window.settings.detection_sensitivity == 7
    assert abs(window.settings.min_document_area - 0.015) < 1e-6


def _detect(window, sample_pdf):
    window.load_pdf(str(sample_pdf))
    window.start_detection()
    assert pump_until(lambda: not window._busy)
    assert window.session is not None and window.session.cards
    return window.session


def test_type_selected_in_crop_editor_is_applied(window, sample_pdf):
    """Regression: the type chosen in the adjust dialog must reach the card."""
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from core.image_exporter import ExportOptions, ImageExporter
    from widgets.crop_editor import CropEditorDialog

    session = _detect(window, sample_pdf)
    # pick a card that does not already have the type we are going to set
    card = next(item for item in session.cards if item.doc_type is not DocumentType.IQAMA)

    def apply_type_in_dialog() -> None:
        dialogs = [
            widget
            for widget in QApplication.topLevelWidgets()
            if isinstance(widget, CropEditorDialog) and widget.isVisible()
        ]
        assert dialogs, "the crop editor did not open"
        combo = dialogs[0]._type_combo
        index = combo.findData(DocumentType.IQAMA.value)
        assert index >= 0, "Iqama is missing from the type combo"
        combo.setCurrentIndex(index)  # fires currentIndexChanged, like a user click
        dialogs[0].accept()

    QTimer.singleShot(250, apply_type_in_dialog)
    window.edit_crop(card.card_id)

    assert card.doc_type is DocumentType.IQAMA, "the type chosen in the dialog was not applied"
    assert card.type_confidence == 1.0
    exporter = ImageExporter(ExportOptions())
    assert exporter.build_filename(card, 1, sample_pdf.stem, 1).startswith("Iqama_")


def test_add_region_creates_a_document(window, sample_pdf):
    """Add Region must turn a drawn rubber band rectangle into a real card."""
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from widgets.pdf_viewer import PdfPageView

    window.show()
    QApplication.processEvents()
    window.load_pdf(str(sample_pdf))
    assert pump_until(lambda: window._view.page_size[0] > 0)
    session = window.session
    assert session is not None
    before = len(session.cards)

    # 1. entering "Add Region" mode
    window._btn_add_region.setChecked(True)
    QApplication.processEvents()
    assert window._view.mode == PdfPageView.MODE_ADD

    # 2. draw a rectangle over the lower right quarter of the page
    view = window._view
    width, height = view.page_size
    start = view.mapFromScene(QPointF(width * 0.55, height * 0.55))
    end = view.mapFromScene(QPointF(width * 0.80, height * 0.80))
    QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
    QTest.mouseMove(view.viewport(), end, 10)
    QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, end)
    QApplication.processEvents()

    # 3. a new manual card exists, with the drawn geometry
    assert len(session.cards) == before + 1, "Add Region did not create a document"
    card = session.cards[-1]
    assert card.source == "manual"
    assert card.doc_type is not None
    assert card.thumbnail is not None and card.thumbnail.size > 0
    xs = card.quad_norm[:, 0]
    ys = card.quad_norm[:, 1]
    assert abs(float(xs.min()) - 0.55) < 0.03
    assert abs(float(xs.max()) - 0.80) < 0.03
    assert abs(float(ys.min()) - 0.55) < 0.03
    assert abs(float(ys.max()) - 0.80) < 0.03

    # 4. the mode is left automatically and the card is selected + in the gallery
    assert not window._btn_add_region.isChecked()
    assert window._view.mode == PdfPageView.MODE_SELECT
    assert window._selected_card_id == card.card_id
    assert window._gallery.item_for(card.card_id) is not None


def test_dashboard_clear_recent_button(window, sample_pdf):
    """Issue: the dashboard needs a way to empty the recent files list."""
    window.settings.recent_files = [str(sample_pdf)]
    window._refresh_dashboard()
    assert window._recent_list.count() == 1
    assert window._recent_clear_btn.isEnabled()

    window._clear_recent_files()  # the fixture answers "Yes" to the confirmation

    assert window.settings.recent_files == []
    assert window._recent_list.count() == 1
    assert "No recent files" in window._recent_list.item(0).text()
    assert not window._recent_clear_btn.isEnabled()


def test_recent_files_are_populated_at_startup(qapp, tmp_path, monkeypatch, sample_pdf):
    """Issue: the list used to stay empty until the Dashboard button was clicked."""
    from app.history import HistoryStore
    from app.main_window import MainWindow
    from app.settings import AppSettings

    for name in ("information", "warning", "critical", "question"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok))

    settings = AppSettings()
    settings.preview_dpi = 80
    settings.recent_files = [str(sample_pdf)]
    settings.output_dir = str(tmp_path / "out")
    settings.save = lambda: True

    window = MainWindow(settings)
    window.history = HistoryStore(tmp_path / "history.json")
    try:
        assert window._pages.currentIndex() == 0  # the dashboard is the start page
        assert window._recent_list.count() == 1, "recent files were not loaded at startup"
        assert sample_pdf.name in window._recent_list.item(0).text()
        assert window._recent_clear_btn.isEnabled()
    finally:
        window.close()


def test_reopening_an_open_pdf_from_the_dashboard_shows_the_workspace(window, sample_pdf):
    """Issue: double-clicking an already open recent file did not switch pages."""
    window.load_pdf(str(sample_pdf))
    session_before = window.session
    assert window._pages.currentIndex() == 1

    window._switch_page(0)  # user goes back to the dashboard
    assert window._pages.currentIndex() == 0
    item = window._recent_list.item(0)
    assert item is not None and sample_pdf.name in item.text()

    window._open_recent_item(item)  # double-click on the same file

    assert window._pages.currentIndex() == 1, "the workspace was not shown"
    assert window.session is session_before, "the document must not be reloaded"
    assert window._nav_buttons[1].isChecked()


def test_document_type_list_shows_builtin_and_custom(window):
    """Every type must appear as a tile and be labelled built-in / custom."""
    from PySide6.QtWidgets import QInputDialog

    from core.models import DocumentType

    original = QInputDialog.getText
    answers = iter([("Membership Card", True), ("MEMBERSHIP", True)])

    def fake_get_text(*_args, **_kwargs):
        return next(answers)

    try:
        setattr(QInputDialog, "getText", staticmethod(fake_get_text))
        window._switch_page(3)
        window._add_document_type()
    finally:
        setattr(QInputDialog, "getText", original)

    try:
        cells = window._type_cells
        # every built-in type has a tile
        for doc_type in DocumentType.builtin_types():
            assert doc_type.token in cells, f"{doc_type.display_name} is missing from the grid"
            assert "built-in" in cells[doc_type.token]["info"].text()
        # the new built-ins requested by the user
        for token in ("health_insurance", "bank_card", "salary_card", "national_id"):
            assert token in cells
        # custom types are marked as custom and can be removed
        assert "custom" in cells["membership_card"]["info"].text()
        assert cells["membership_card"]["remove"] is not None

        # ticking is possible for everything except "Unknown"
        assert cells["bank_card"]["check"].isEnabled()
        assert cells["bank_card"]["check"].isChecked()
        assert not cells["unknown"]["check"].isEnabled()
        assert cells["unknown"]["check"].isChecked()
        # "Unknown" has no keyword button
        assert cells["unknown"]["keywords"] is None

        # delete guard: no remove button on built-ins, and the handler refuses
        assert cells["bank_card"]["remove"] is None, "built-in types must not be deletable"
        window._remove_document_type("bank_card")  # explains and keeps the type
        assert DocumentType.from_value("bank_card") is DocumentType.BANK_CARD
    finally:
        window.settings.remove_custom_type("membership_card")
        window._refresh_type_list()
        window._populate_type_combo()


def test_unticking_a_builtin_type_hides_it_from_the_pickers(window):
    from core.models import DocumentType

    window._switch_page(3)
    cell = window._type_cells["salary_card"]
    assert window._type_combo.findData("salary_card") >= 0
    assert DocumentType.SALARY_CARD.enabled is True

    cell["check"].setChecked(False)  # fires toggled

    assert "salary_card" in window.settings.disabled_document_types
    assert DocumentType.SALARY_CARD.enabled is False
    assert window._type_combo.findData("salary_card") < 0, "hidden type still offered in the picker"
    assert cell["cell"].property("enabledType") == "false"

    cell["check"].setChecked(True)

    assert "salary_card" not in window.settings.disabled_document_types
    assert DocumentType.SALARY_CARD.enabled is True
    assert window._type_combo.findData("salary_card") >= 0
    assert cell["cell"].property("enabledType") == "true"


def test_dragging_inside_a_card_moves_the_whole_region(window, sample_pdf):
    """All four corners must move together when the region is dragged."""
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    window.show()
    QApplication.processEvents()
    session = _detect(window, sample_pdf)
    view = window._view
    assert pump_until(lambda: view.page_size[0] > 0)
    card = session.cards[0]

    window._on_card_selected(card.card_id)  # selected -> the press starts a move
    before = card.quad_norm.copy()
    width, height = view.page_size
    centre = QPointF(float(before[:, 0].mean()) * width, float(before[:, 1].mean()) * height)
    start = view.mapFromScene(centre)
    end = view.mapFromScene(centre + QPointF(70, 45))

    QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
    QTest.mouseMove(view.viewport(), end, 10)
    QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, end)
    QApplication.processEvents()

    after = card.quad_norm
    assert not np.allclose(after, before), "the region did not move"
    # shape unchanged: same width and height
    assert abs((after[:, 0].max() - after[:, 0].min()) - (before[:, 0].max() - before[:, 0].min())) < 0.003
    assert abs((after[:, 1].max() - after[:, 1].min()) - (before[:, 1].max() - before[:, 1].min())) < 0.003
    # moved towards the drag direction (right/down)
    assert after[:, 0].mean() > before[:, 0].mean()
    assert after[:, 1].mean() > before[:, 1].mean()
    # it stays on the page
    assert after.min() >= -1e-6 and after.max() <= 1.0 + 1e-6


def test_add_region_uses_the_identity_card_size(window, sample_pdf):
    """With a preset selected, the drawn region takes that exact physical size."""
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    window.show()
    QApplication.processEvents()
    window.load_pdf(str(sample_pdf))
    assert pump_until(lambda: window._view.page_size[0] > 0)
    session = window.session
    before = len(session.cards)

    index = window._region_preset_combo.findText("ID card 85.6 × 54 mm")
    assert index > 0
    window._region_preset_combo.setCurrentIndex(index)  # id card preset
    QApplication.processEvents()
    assert window.settings.manual_region_fixed_size is True
    assert abs(window.settings.manual_region_width_mm - 85.6) < 0.01
    assert window._set_region_width.value() == pytest.approx(85.6)

    window._btn_add_region.setChecked(True)
    view = window._view
    width, height = view.page_size
    start = view.mapFromScene(QPointF(width * 0.20, height * 0.20))
    end = view.mapFromScene(QPointF(width * 0.80, height * 0.70))  # deliberately too big
    QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
    QTest.mouseMove(view.viewport(), end, 10)
    QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, end)
    QApplication.processEvents()

    assert len(session.cards) == before + 1
    card = session.cards[-1]
    assert card.source == "manual"

    px_per_mm = float(window._page_dpi) / 25.4
    expected_width = (85.6 * px_per_mm) / width
    expected_height = (53.98 * px_per_mm) / height
    assert abs((card.quad_norm[:, 0].max() - card.quad_norm[:, 0].min()) - expected_width) < 0.01
    assert abs((card.quad_norm[:, 1].max() - card.quad_norm[:, 1].min()) - expected_height) < 0.01
    # the centre of the drawn box is kept
    assert abs(float(card.quad_norm[:, 0].mean()) - 0.5) < 0.03
    assert abs(float(card.quad_norm[:, 1].mean()) - 0.45) < 0.03

    # back to free drawing for the following tests
    window._region_preset_combo.setCurrentIndex(0)
    assert window.settings.manual_region_fixed_size is False


def test_fine_rotation_from_the_dialog_reaches_the_main_form(window, sample_pdf):
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from widgets.crop_editor import CropEditorDialog

    window.show()
    QApplication.processEvents()
    session = _detect(window, sample_pdf)
    card = session.cards[0]
    before_quad = card.quad_norm.copy()

    def apply_fine_rotation() -> None:
        dialogs = [
            widget
            for widget in QApplication.topLevelWidgets()
            if isinstance(widget, CropEditorDialog) and widget.isVisible()
        ]
        assert dialogs, "the crop editor did not open"
        dialogs[0]._fine_spin.setValue(4.0)
        dialogs[0].accept()

    QTimer.singleShot(250, apply_fine_rotation)
    window.edit_crop(card.card_id)

    assert abs(card.fine_rotation - 4.0) < 1e-6
    # the overlay shows the deskewed quad
    width, height = window._view.page_size
    overlay = window._view._overlays[card.card_id]
    assert np.allclose(overlay.quad, card.effective_quad_pixels(width, height), atol=0.05)
    assert not np.allclose(overlay.quad, card.quad_pixels(width, height), atol=1.0)
    # the thumbnail was rebuilt and the geometry itself was not altered
    assert card.thumbnail is not None and card.thumbnail.size > 0
    assert np.allclose(card.quad_norm, before_quad, atol=1e-6)


def test_custom_region_preset_opens_the_size_dialog(window, sample_pdf):
    """Choosing "Custom…" must ask for the size and stay selected (no snap back)."""
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QDialogButtonBox

    from widgets.region_size_dialog import RegionSizeDialog

    window.show()
    flush_events(2)

    combo = window._region_preset_combo
    custom_index = combo.count() - 1
    assert combo.itemText(custom_index).startswith("Custom")

    def fill_dialog() -> None:
        dialogs = [
            widget
            for widget in QApplication.topLevelWidgets()
            if isinstance(widget, RegionSizeDialog) and widget.isVisible()
        ]
        assert dialogs, "the custom size dialog did not open"
        dialog = dialogs[0]
        dialog._width.setValue(120.0)
        dialog._height.setValue(80.0)
        dialog._swap()  # 80 x 120
        dialog._swap()  # back to 120 x 80
        buttons = dialog.findChild(QDialogButtonBox)
        assert buttons is not None
        buttons.button(QDialogButtonBox.StandardButton.Ok).click()

    QTimer.singleShot(150, fill_dialog)
    combo.setCurrentIndex(custom_index)
    flush_events()

    settings = window.settings
    assert settings.manual_region_preset == "custom"
    assert settings.manual_region_fixed_size is True
    assert settings.manual_region_width_mm == pytest.approx(120.0)
    assert settings.manual_region_height_mm == pytest.approx(80.0)
    # the selection must stick instead of jumping back to the ID card preset
    assert combo.currentIndex() == custom_index
    assert window._set_region_preset.currentIndex() == custom_index
    assert window._set_region_width.value() == pytest.approx(120.0)


def test_custom_region_preset_cancel_keeps_the_previous_selection(window, sample_pdf):
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QDialogButtonBox

    from widgets.region_size_dialog import RegionSizeDialog

    window.show()
    flush_events(2)
    combo = window._region_preset_combo
    combo.setCurrentIndex(2)  # business card
    flush_events(2)
    assert window.settings.manual_region_preset == "business_card"

    def cancel_dialog() -> None:
        dialogs = [
            widget
            for widget in QApplication.topLevelWidgets()
            if isinstance(widget, RegionSizeDialog) and widget.isVisible()
        ]
        assert dialogs
        dialog = dialogs[0]
        dialog._width.setValue(150.0)
        buttons = dialog.findChild(QDialogButtonBox)
        assert buttons is not None
        buttons.button(QDialogButtonBox.StandardButton.Cancel).click()

    QTimer.singleShot(150, cancel_dialog)
    combo.setCurrentIndex(combo.count() - 1)  # Custom…
    flush_events()

    settings = window.settings
    assert settings.manual_region_preset == "business_card"
    assert settings.manual_region_width_mm == pytest.approx(90.0)
    assert combo.currentIndex() == 2  # the combo was restored


def test_region_dimensions_typed_in_settings_become_custom(window, sample_pdf):
    from PySide6.QtWidgets import QApplication

    window.show()
    QApplication.processEvents()

    window._region_preset_combo.setCurrentIndex(1)  # ID card
    window._set_region_width.setValue(70.0)  # user edits the number
    window._set_region_height.setValue(45.0)

    assert window.settings.manual_region_preset == "custom"
    assert window.settings.manual_region_fixed_size is True
    assert window._region_preset_combo.currentIndex() == window._region_preset_combo.count() - 1


def test_rotate_grip_of_the_crop_editor_rotates_the_card(window, sample_pdf):
    """The round grip above the crop must rotate the document live."""
    import math

    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from widgets.crop_editor import CropEditorDialog

    window.show()
    QApplication.processEvents()
    session = _detect(window, sample_pdf)
    card = session.cards[0]

    dialog = CropEditorDialog(sample_pdf, card, window)
    dialog.show()
    QApplication.processEvents()

    view = dialog._view
    item = view._overlays[card.card_id]
    assert item.rotation_enabled is True
    handle = item.rotation_handle_point()
    centre = item.quad.mean(axis=0)

    # rotate the grip by ~18° around the centre of the card
    angle = math.radians(18.0)
    dx, dy = handle.x() - float(centre[0]), handle.y() - float(centre[1])
    target = QPointF(
        float(centre[0]) + dx * math.cos(angle) - dy * math.sin(angle),
        float(centre[1]) + dx * math.sin(angle) + dy * math.cos(angle),
    )

    QTest.mousePress(
        view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, view.mapFromScene(handle)
    )
    QTest.mouseMove(view.viewport(), view.mapFromScene(target), 20)
    QApplication.processEvents()
    QTest.mouseRelease(
        view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, view.mapFromScene(target)
    )
    QApplication.processEvents()

    assert dialog.card.fine_rotation == pytest.approx(18.0, abs=1.5)
    assert dialog._fine_spin.value() == pytest.approx(dialog.card.fine_rotation, abs=0.01)
    dialog.accept()


def test_region_size_controls_work_without_a_loaded_pdf(window):
    """The size controls must not be disabled just because no PDF is open."""
    assert window.session is None
    assert window._region_preset_combo.isEnabled()
    assert window._region_size_button.isEnabled()


def test_size_button_opens_the_dialog_even_when_custom_is_already_selected(window):
    """Regression: re-picking "Custom…" once it is current must still open the dialog."""
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QDialogButtonBox

    from widgets.region_size_dialog import RegionSizeDialog

    window.show()
    flush_events(2)

    # the button shows the current size once a fixed size is active
    window.settings.manual_region_preset = "custom"
    window.settings.manual_region_fixed_size = True
    window.settings.manual_region_width_mm = 85.6
    window.settings.manual_region_height_mm = 53.98
    window._sync_region_controls()
    assert "85.6" in window._region_size_button.text()

    def fill_dialog() -> None:
        dialogs = [
            widget
            for widget in QApplication.topLevelWidgets()
            if isinstance(widget, RegionSizeDialog) and widget.isVisible()
        ]
        assert dialogs, "the size dialog did not open from the Size… button"
        dialogs[0]._width.setValue(99.0)
        dialogs[0]._height.setValue(66.0)
        buttons = dialogs[0].findChild(QDialogButtonBox)
        assert buttons is not None
        buttons.button(QDialogButtonBox.StandardButton.Ok).click()

    QTimer.singleShot(150, fill_dialog)
    window._region_size_button.click()  # "<w> × <h> mm" button
    flush_events()

    assert window.settings.manual_region_preset == "custom"
    assert window.settings.manual_region_width_mm == pytest.approx(99.0)
    assert window.settings.manual_region_height_mm == pytest.approx(66.0)
    assert "99.0" in window._region_size_button.text()


def test_reactivating_custom_in_the_combo_opens_the_dialog_again(window):
    """A second click on the already-selected "Custom…" entry must re-open the dialog."""
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QDialogButtonBox

    from widgets.region_size_dialog import RegionSizeDialog

    window.show()
    flush_events(2)

    combo = window._region_preset_combo
    custom_index = combo.count() - 1

    def fill(width: float, height: float) -> None:
        def run() -> None:
            dialogs = [
                widget
                for widget in QApplication.topLevelWidgets()
                if isinstance(widget, RegionSizeDialog) and widget.isVisible()
            ]
            assert dialogs, "the size dialog did not open"
            dialogs[0]._width.setValue(width)
            dialogs[0]._height.setValue(height)
            buttons = dialogs[0].findChild(QDialogButtonBox)
            assert buttons is not None
            buttons.button(QDialogButtonBox.StandardButton.Ok).click()

        QTimer.singleShot(120, run)

    # first selection of "Custom…" (index change) -> deferred dialog
    fill(110.0, 70.0)
    combo.setCurrentIndex(custom_index)
    flush_events()
    assert window.settings.manual_region_width_mm == pytest.approx(110.0)

    # the entry is current now, so picking it again only emits "activated"
    fill(123.0, 45.0)
    window._on_region_preset_activated(custom_index)
    flush_events()
    assert window.settings.manual_region_width_mm == pytest.approx(123.0)
    assert window.settings.manual_region_height_mm == pytest.approx(45.0)


def _crop_sides(card, width, height):
    """Pixel width/height of the card's crop along its own axes."""
    quad = card.effective_quad_pixels(width, height)
    return (
        float(np.linalg.norm(quad[1] - quad[0])),
        float(np.linalg.norm(quad[3] - quad[0])),
    )


def test_arrow_keys_nudge_and_resize_the_crop(window, sample_pdf):
    """Arrows move the crop by 1 px, Shift+arrows resize it (no mouse needed)."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    from widgets.crop_editor import CropEditorDialog

    window.show()
    flush_events(2)
    session = _detect(window, sample_pdf)
    card = session.cards[0]

    dialog = CropEditorDialog(sample_pdf, card, window)
    dialog.show()
    flush_events(3)

    view = dialog._view
    width, height = view.page_size
    draft = dialog.card

    before = draft.quad_norm.copy()
    QTest.keyClick(view, Qt.Key.Key_Right)
    flush_events(2)
    after = draft.quad_norm
    # moved right by exactly one preview pixel
    assert after[:, 0].mean() - before[:, 0].mean() == pytest.approx(1.0 / width, rel=0.25)
    assert after[:, 1].mean() == pytest.approx(before[:, 1].mean(), abs=1e-6)

    QTest.keyClick(view, Qt.Key.Key_Down)
    flush_events(2)
    moved = draft.quad_norm
    assert moved[:, 1].mean() - after[:, 1].mean() == pytest.approx(1.0 / height, rel=0.25)

    # Shift + Right widens the crop by 1 px on each side
    width_before, height_before = _crop_sides(draft, width, height)
    QTest.keyClick(view, Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier)
    flush_events(2)
    width_after, height_after = _crop_sides(draft, width, height)
    assert width_after - width_before == pytest.approx(2.0, abs=0.5)
    assert height_after == pytest.approx(height_before, abs=0.5)

    # Shift + Up makes it taller
    QTest.keyClick(view, Qt.Key.Key_Up, Qt.KeyboardModifier.ShiftModifier)
    flush_events(2)
    _width_after2, height_after2 = _crop_sides(draft, width, height)
    assert height_after2 - height_after == pytest.approx(2.0, abs=0.5)

    # Ctrl makes the step ten times bigger
    before_ctrl = draft.quad_norm.copy()
    QTest.keyClick(view, Qt.Key.Key_Left, Qt.KeyboardModifier.ControlModifier)
    flush_events(2)
    assert before_ctrl[:, 0].mean() - draft.quad_norm[:, 0].mean() == pytest.approx(
        10.0 / width, rel=0.2
    )
    dialog.accept()


# --------------------------------------------------------------- review fixes
def _two_page_pdf(tmp_path) -> Path:
    """A generated 2 page PDF (no sample.pdf needed)."""
    try:
        import pymupdf as _fitz
    except ImportError:  # pragma: no cover - older PyMuPDF releases
        import fitz as _fitz

    path = tmp_path / "two_pages.pdf"
    document = _fitz.open()
    for _ in range(2):
        page = document.new_page(width=500, height=350)
        page.draw_rect(_fitz.Rect(40, 40, 280, 200), color=(0, 0, 0), fill=(0.2, 0.3, 0.6))
    document.save(str(path))
    document.close()
    return path


def _study_cards(window, pages: int):
    """Give the loaded session one synthetic card per page (no detection run)."""
    from core.models import DetectedCard

    session = window.session
    assert session is not None
    cards = []
    for page_index in range(pages):
        card = DetectedCard(
            page_index=page_index,
            quad_norm=np.array(
                [[0.1, 0.1], [0.6, 0.1], [0.6, 0.5], [0.1, 0.5]], dtype=np.float32
            ),
            confidence=0.9,
            page_size=(500, 350),
        )
        session.add_card(card)
        cards.append(card)
    session.detected_pages.update(range(pages))
    window._refresh_gallery()
    return cards


def test_gallery_page_filter_shows_the_page_it_names(window, tmp_path):
    """Regression: the combo stored 0-based pages, page_number is 1-based."""
    from PySide6.QtCore import Qt

    window.load_pdf(str(_two_page_pdf(tmp_path)))
    cards = _study_cards(window, 2)

    combo = window._gallery_page_filter
    assert combo.itemData(1) == 1, "Page 01 must filter on page_number 1"
    assert combo.itemData(2) == 2

    # "This page" (the default) follows the viewer; switch to "All pages" so the
    # page combo is in charge.
    window._gallery_filter.setCurrentIndex(1)
    flush_events(2)

    combo.setCurrentIndex(1)  # Page 01
    flush_events(2)
    shown = [
        window._gallery.item(row).data(Qt.ItemDataRole.UserRole)
        for row in range(window._gallery.count())
    ]
    assert [card.page_number for card in shown] == [1]
    assert shown[0].card_id == cards[0].card_id

    combo.setCurrentIndex(2)  # Page 02
    flush_events(2)
    shown = [
        window._gallery.item(row).data(Qt.ItemDataRole.UserRole)
        for row in range(window._gallery.count())
    ]
    assert [card.page_number for card in shown] == [2]
    assert shown[0].card_id == cards[1].card_id


def test_gallery_page_combo_follows_the_viewer(window, tmp_path):
    """The "This page / All pages" combo was connected but never read."""
    from PySide6.QtCore import Qt

    window.load_pdf(str(_two_page_pdf(tmp_path)))
    _study_cards(window, 2)

    window._gallery_filter.setCurrentIndex(0)  # This page
    window.select_page(0)
    flush_events(2)
    assert window._gallery.count() == 1
    assert not window._gallery_page_filter.isEnabled()

    window.select_page(1)
    flush_events(2)
    card = window._gallery.item(0).data(Qt.ItemDataRole.UserRole)
    assert card.page_number == 2

    window._gallery_filter.setCurrentIndex(1)  # All pages
    flush_events(2)
    assert window._gallery.count() == 2
    assert window._gallery_page_filter.isEnabled()


def test_export_card_writes_an_excluded_card(window, sample_pdf, tmp_path, monkeypatch):
    """Regression: "Export it anyway?" exported nothing at all."""
    from PySide6.QtWidgets import QFileDialog

    session = _detect(window, sample_pdf)
    card = session.cards[0]
    card.included = False

    target = tmp_path / "single"
    target.mkdir()
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(target))
    )

    window.export_card(card.card_id)
    assert pump_until(lambda: not window._busy)
    flush_events(3)

    assert list(target.glob("*")), "the confirmed export did not write a file"
    assert window.history.entries and window.history.entries[0].document_count == 1


def test_excluded_card_can_be_exported_by_the_job(window, sample_pdf):
    """The worker/exporter filter included cards; the snapshot must carry the flag."""
    from workers.export_worker import ExportJob

    session = _detect(window, sample_pdf)
    card = session.cards[0]
    card.included = False

    snapshot = card.copy_card()
    snapshot.included = True
    job = ExportJob(session.path, [snapshot])
    assert [item.card_id for item in job.included_cards] == [card.card_id]


def test_settings_form_round_trips_every_control(window, monkeypatch, tmp_path):
    """Regression: "Open the output folder…" was never loaded from the settings.

    A control that is not filled in shows its widget default and then *writes*
    that default back on "Save settings", silently changing the configuration.
    """
    settings = AppSettings()
    settings.render_dpi = 350
    settings.cap_dpi_to_native = False
    settings.preview_dpi = 111
    settings.detection_sensitivity = 9
    settings.min_document_area = 0.012
    settings.max_document_area = 0.8
    settings.detection_min_confidence = 0.5
    settings.refine_quads = False
    settings.detection_working_max_dim = 1800
    settings.manual_region_fixed_size = True
    settings.manual_region_preset = "id_card"
    settings.manual_region_width_mm = 85.6
    settings.manual_region_height_mm = 53.98
    settings.auto_orientation = False
    settings.auto_classify = False
    settings.ocr_enabled = True
    settings.ocr_for_orientation = True
    settings.ocr_engine = "tesseract"
    settings.ocr_languages = "en,ar"
    settings.export_format = "png"
    settings.jpeg_quality = 88
    settings.output_dir = str(tmp_path / "out")
    settings.create_subfolder_per_pdf = False
    settings.open_output_after_export = True
    settings.filename_template = "{type}-{page}"
    settings.blur_faces = True
    settings.face_blur_kernel = 41
    settings.min_face_size = 60
    monkeypatch.setattr(settings, "save", lambda: True)

    window.settings = settings
    window._switch_page(3)
    window._settings_page._load_settings_into_form()

    assert window._set_open_folder.isChecked(), "the checkbox was not loaded"
    window._settings_page._save_settings_from_form()

    saved = window.settings
    assert saved.render_dpi == 350
    assert saved.cap_dpi_to_native is False
    assert saved.preview_dpi == 111
    assert saved.detection_sensitivity == 9
    assert saved.detection_min_confidence == pytest.approx(0.5)
    assert saved.refine_quads is False
    assert saved.detection_working_max_dim == 1800
    assert saved.manual_region_preset == "id_card"
    assert saved.auto_orientation is False
    assert saved.auto_classify is False
    assert saved.ocr_enabled is True
    assert saved.ocr_for_orientation is True
    assert saved.ocr_engine == "tesseract"
    assert saved.ocr_languages == "en,ar"
    assert saved.export_format == "png"
    assert saved.jpeg_quality == 88
    assert saved.create_subfolder_per_pdf is False
    assert saved.open_output_after_export is True, "the save cleared it"
    assert saved.filename_template == "{type}-{page}"
    assert saved.blur_faces is True
    assert saved.face_blur_kernel == 41
    assert saved.min_face_size == 60


def test_saving_an_export_preset_writes_the_settings_file(window, monkeypatch, tmp_path):
    """Regression: presets were kept in memory only (lost on restart)."""
    from app import settings as settings_module

    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr(settings_module, "settings_path", lambda: settings_file)
    monkeypatch.setattr(
        "PySide6.QtWidgets.QInputDialog.getText",
        staticmethod(lambda *a, **k: ("Scans", True)),
    )
    # the window fixture stubs save() out so tests never touch the real profile;
    # this test is about that write, so restore the real one
    window.settings.__dict__.pop("save", None)

    window.settings.export_format = "png"
    window._switch_page(3)
    window._settings_page._save_preset()

    assert settings_file.exists(), "saving a preset did not write settings.json"
    reloaded = settings_module.AppSettings.load()
    assert reloaded.export_preset_names() == ["Scans"]
    assert reloaded.get_export_preset("Scans").image_format == "png"

    window._preset_combo.setCurrentIndex(window._preset_combo.findData("Scans"))
    window._settings_page._delete_preset()
    assert settings_module.AppSettings.load().export_preset_names() == []


def test_crop_editor_undo_shortcuts_and_dpi_hint(window, sample_pdf):
    """Regression: the tooltips promised Ctrl+Z / Ctrl+Y but nothing was bound."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeySequence
    from PySide6.QtTest import QTest

    from widgets.crop_editor import CropEditorDialog

    session = _detect(window, sample_pdf)
    card = session.cards[0]

    dialog = CropEditorDialog(sample_pdf, card, window, render_dpi=250)
    dialog.show()
    # the shortcuts are window-scoped, so the dialog must own the keyboard
    dialog.activateWindow()
    dialog.setFocus()
    flush_events(3)

    assert dialog._undo_shortcut.key() == QKeySequence(QKeySequence.StandardKey.Undo)
    assert dialog._redo_shortcut.key() == QKeySequence(QKeySequence.StandardKey.Redo)
    # the size estimate follows the configured DPI instead of a hard coded 300
    assert "at 250 dpi" in dialog._preview_info.text()

    before = dialog.card.quad_norm.copy()
    dialog._view.resize_selected(8.0, 8.0)
    flush_events(2)
    assert dialog._undo_stack.canUndo()

    QTest.keyClick(dialog, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    flush_events(2)
    assert np.allclose(dialog.card.quad_norm, before), "Ctrl+Z did not undo"
    assert dialog._undo_stack.canRedo()

    QTest.keyClick(dialog, Qt.Key.Key_Y, Qt.KeyboardModifier.ControlModifier)
    flush_events(2)
    assert not np.allclose(dialog.card.quad_norm, before), "Ctrl+Y did not redo"
    dialog.close()


def test_identifier_edit_is_a_single_undo_step(window, sample_pdf):
    """Regression: one undo command per keystroke; Reset kept the stack."""
    from widgets.crop_editor import CropEditorDialog

    session = _detect(window, sample_pdf)
    card = session.cards[0]
    original = card.identifier

    dialog = CropEditorDialog(sample_pdf, card, window)
    flush_events(2)

    for text in ("A", "AB", "AB1", "AB12"):
        dialog._identifier_edit.setText(text)  # one textChanged per keystroke
    assert not dialog._undo_stack.canUndo(), "typing must not push a command per key"

    dialog._identifier_edit.editingFinished.emit()  # the user left the field
    assert dialog._undo_stack.count() == 1
    dialog._undo_stack.undo()
    assert dialog.card.identifier == original

    # Reset starts over: earlier edits are no longer undoable
    dialog._identifier_edit.setText("ZZ99")
    dialog._identifier_edit.editingFinished.emit()
    assert dialog._undo_stack.canUndo()
    dialog._reset()
    assert not dialog._undo_stack.canUndo()
    assert dialog.card.identifier == original
    dialog.close()


def test_hiding_a_card_does_not_change_the_gallery_selection(window, tmp_path):
    """Regression: takeItem() moved the current row behind the user's back."""
    window.load_pdf(str(_two_page_pdf(tmp_path)))
    cards = _study_cards(window, 2)

    cards[1].included = False
    window._gallery_included_only.setChecked(True)
    flush_events(2)
    assert window._gallery.count() == 1

    window._gallery.select_card(cards[0].card_id)
    flush_events(2)
    announced = []
    window._gallery.cardSelected.connect(announced.append)

    cards[0].included = False  # now it no longer matches the filter
    window._gallery.update_card(cards[0])
    flush_events(2)

    assert window._gallery.count() == 0
    assert announced == [], "removing a card emitted a selection change"


def test_face_blur_settings_reflect_the_detector_backend(window, monkeypatch):
    """The settings must show which detector runs, or offer the model download."""
    from core import face_detector

    monkeypatch.setattr(face_detector, "_status_cache", None)
    monkeypatch.setattr(face_detector, "model_candidates", lambda: [])
    window._switch_page(3)
    window._settings_page._refresh_face_blur_state()

    if face_detector.can_download_model():
        # YuNet is compiled in but its model is missing: offer the download
        assert window._btn_download_model.isVisibleTo(window._settings_page)
        assert "unavailable" in window._blur_status.text().lower()
        assert not window._set_blur_faces.isEnabled()
    else:
        assert window._btn_download_model.isVisibleTo(window._settings_page) is False

    # a model file turns the option on and names the backend
    monkeypatch.setattr(
        face_detector,
        "status",
        lambda: ("yunet", "YuNet neural network (model.onnx)"),
    )
    monkeypatch.setattr(face_detector, "can_download_model", lambda: False)
    window._settings_page._refresh_face_blur_state()
    assert window._set_blur_faces.isEnabled()
    assert window._set_blur_kernel.isEnabled()
    assert "YuNet" in window._blur_status.text()
    assert not window._btn_download_model.isVisibleTo(window._settings_page)

    monkeypatch.setattr(face_detector, "status", lambda: ("", "no detector here"))
    window._settings_page._refresh_face_blur_state()
    assert not window._set_blur_faces.isEnabled()
    assert "no detector here" in window._blur_status.text()


def test_crop_editor_nudge_and_size_buttons(window, sample_pdf):
    from PySide6.QtWidgets import QApplication

    from widgets.crop_editor import CropEditorDialog

    window.show()
    flush_events(2)
    session = _detect(window, sample_pdf)
    card = session.cards[0]

    dialog = CropEditorDialog(sample_pdf, card, window)
    dialog.show()
    flush_events(3)
    view = dialog._view
    width, height = view.page_size
    draft = dialog.card

    before = draft.quad_norm.copy()
    dialog._nudge_buttons["right"].click()
    QApplication.processEvents()
    assert draft.quad_norm[:, 0].mean() - before[:, 0].mean() == pytest.approx(1.0 / width, rel=0.25)

    width_before, _ = _crop_sides(draft, width, height)
    dialog._size_buttons["wider"].click()
    QApplication.processEvents()
    width_after, _ = _crop_sides(draft, width, height)
    assert width_after - width_before == pytest.approx(2.0, abs=0.5)

    dialog._size_buttons["narrower"].click()
    QApplication.processEvents()
    width_back, _ = _crop_sides(draft, width, height)
    assert width_back == pytest.approx(width_before, abs=0.5)
    dialog.accept()


def test_margin_button_expands_the_crop_on_all_sides(window, sample_pdf):
    """The margin control is the easy way to keep the card's shadow."""
    from PySide6.QtWidgets import QApplication

    from widgets.crop_editor import CropEditorDialog

    window.show()
    flush_events(2)
    session = _detect(window, sample_pdf)
    card = session.cards[0]

    dialog = CropEditorDialog(sample_pdf, card, window)
    dialog.show()
    flush_events(3)
    view = dialog._view
    width, height = view.page_size
    draft = dialog.card

    width_before, height_before = _crop_sides(draft, width, height)
    centre_before = draft.quad_norm.mean(axis=0)

    dialog._margin_spin.setValue(2.0)  # 2 mm
    dialog._margin_button.click()
    QApplication.processEvents()

    width_after, height_after = _crop_sides(draft, width, height)
    expected = 2.0 * 2.0 * float(CropEditorDialog.PREVIEW_DPI) / 25.4  # both sides
    assert width_after - width_before == pytest.approx(expected, rel=0.05)
    assert height_after - height_before == pytest.approx(expected, rel=0.05)
    # symmetric growth: the centre stays where it was
    assert np.allclose(draft.quad_norm.mean(axis=0), centre_before, atol=1e-4)
    dialog.accept()


def test_settings_save_shows_a_confirmation(window, monkeypatch):
    """Saving must tell the user it worked."""
    seen = []
    monkeypatch.setattr(
        QMessageBox,
        "information",
        staticmethod(lambda parent, title, text, *a, **k: seen.append((title, text))),
    )
    window._switch_page(3)
    window._save_settings_from_form()

    assert seen, "no confirmation was shown"
    title, text = seen[-1]
    assert title == "Settings saved"
    assert "settings have been saved" in text.lower()


def test_ocr_check_button_reports_the_engine(window, monkeypatch):
    from core.ocr_engine import OcrEngine

    messages = []
    monkeypatch.setattr(
        QMessageBox,
        "information",
        staticmethod(lambda parent, title, text, *a, **k: messages.append((title, text))),
    )
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        staticmethod(lambda parent, title, text, *a, **k: messages.append((title, text))),
    )
    answers = iter(["\n".join(["Tesseract found      : True", "Engine used          : tesseract"]), ""])
    monkeypatch.setattr(
        OcrEngine,
        "diagnostic_report",
        classmethod(lambda cls, languages=("en",), prefer="auto", smoke_test=True: next(answers)),
    )
    monkeypatch.setattr(OcrEngine, "tesseract_available", staticmethod(lambda: True))
    monkeypatch.setattr(OcrEngine, "easyocr_available", staticmethod(lambda: False))
    monkeypatch.setattr(OcrEngine, "is_available", property(lambda self: True))

    window._switch_page(3)
    window._btn_check_ocr.click()

    assert messages, "the check produced no message box"
    title, text = messages[0]
    assert title == "OCR check"
    assert "Tesseract found" in text


def test_edit_keywords_dialog_adds_keywords_to_a_builtin_type(window, monkeypatch):
    """Built-in types must be extendable (new language) without touching the code."""
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QDialog, QPlainTextEdit

    from core.models import DocumentType

    DocumentType.reset_extra_keywords()
    try:
        window._switch_page(3)
        window._refresh_type_list()
        assert window._type_cells["passport"]["keywords"] is not None
        assert window._type_cells["passport"]["keywords"].isEnabled()

        def fill() -> None:
            dialogs = [
                w
                for w in QApplication.topLevelWidgets()
                if isinstance(w, QDialog) and w.isVisible() and "KEYWORDS" in w.windowTitle().upper()
            ]
            assert dialogs, "the keyword dialog did not open"
            editor = dialogs[0].findChild(QPlainTextEdit)
            assert editor is not None
            editor.setPlainText("PASSEPORT\nجواز سفر")
            dialogs[0].accept()

        QTimer.singleShot(150, fill)
        window._edit_keywords_for("passport")
        QApplication.processEvents()

        assert window.settings.type_keywords("passport") == ["PASSEPORT", "جواز سفر"]
        assert DocumentType.PASSPORT.user_keywords == ("PASSEPORT", "جواز سفر")
        # the tile summary mentions the extra keywords
        assert "PASSEPORT" in window._type_cells["passport"]["info"].text()
    finally:
        window.settings.set_type_keywords("passport", "")
        DocumentType.reset_extra_keywords()
        window._refresh_type_list()


def test_crop_editor_shows_the_ocr_text(window, sample_pdf):
    from widgets.crop_editor import CropEditorDialog

    session = _detect(window, sample_pdf)
    card = session.cards[0]
    card.ocr_text = "KINGDOM OF SAUDI ARABIA\nPASSPORT WE4110283"

    dialog = CropEditorDialog(sample_pdf, card, window)
    dialog.show()
    flush_events(2)
    try:
        assert "WE4110283" in dialog._ocr_text_view.toPlainText()
        assert dialog._copy_ocr_button.isEnabled()
    finally:
        dialog.accept()


def test_custom_type_added_in_settings_is_usable(window, sample_pdf):
    """A type added on the settings page appears in the combo and names the file."""
    from PySide6.QtWidgets import QInputDialog

    from core.models import DocumentType

    original_get_text = QInputDialog.getText
    answers = iter([("Work Permit", True), ("WORK PERMIT", True)])
    setattr(QInputDialog, "getText", staticmethod(lambda *args, **kwargs: next(answers)))
    try:
        window._switch_page(3)
        window._add_document_type()
    finally:
        QInputDialog.getText = original_get_text

    try:
        tokens_in_grid = list(window._type_cells)
        assert "work_permit" in tokens_in_grid, "the new type is missing from the settings grid"
        index = window._type_combo.findData("work_permit")
        assert index >= 0, "the custom type is missing from the gallery drop-down"

        session = _detect(window, sample_pdf)
        card = next(item for item in session.cards if item.doc_type is not DocumentType.UNKNOWN)
        window._on_card_selected(card.card_id)
        window._type_combo.setCurrentIndex(index)
        window._on_type_combo_changed(index)

        assert card.doc_type.token == "work_permit"
        assert card.doc_type.display_name == "Work Permit"

        from core.image_exporter import ExportOptions, ImageExporter

        assert ImageExporter(ExportOptions()).build_filename(card, 1, "scan", 1).startswith("WorkPermit_")
    finally:
        window.settings.remove_custom_type("work_permit")
        window._refresh_type_list()
        window._populate_type_combo()
        assert window._type_combo.findData("work_permit") < 0


def test_type_combo_in_gallery_is_applied(window, sample_pdf):
    """Regression: the type drop-down below the gallery must reach the card."""
    from PySide6.QtCore import Qt

    from core.models import DocumentType

    session = _detect(window, sample_pdf)
    card = next(item for item in session.cards if item.doc_type is not DocumentType.DRIVING_LICENSE)
    window._on_card_selected(card.card_id)

    index = window._type_combo.findData(DocumentType.DRIVING_LICENSE.value)
    assert index >= 0
    window._type_combo.setCurrentIndex(index)
    window._on_type_combo_changed(index)  # what the "activated" signal triggers

    assert card.doc_type is DocumentType.DRIVING_LICENSE
    item = window._gallery.item_for(card.card_id)
    assert item is not None
    assert item.data(Qt.ItemDataRole.UserRole).doc_type is DocumentType.DRIVING_LICENSE

