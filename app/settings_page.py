"""The settings page, extracted from ``app.main_window``.

``SettingsPage`` owns every widget of the settings page (rendering, detection,
Add Region size, OCR, export, and the document type manager) together with the
actions that read/write them.  It edits the *live* settings object held by
``MainWindow`` so a saved change is visible everywhere instantly.

The main window keeps the Add Region preset state machine (it couples the
settings widgets with the gallery toolbar) and the extract-page type combo -
the page calls back into it through ``self._mw``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.icons import button_icon, icon
from app.settings import AppSettings, settings_path
from core.image_exporter import ExportOptions
from core.models import DetectedCard, DocumentType
from core.ocr_engine import OcrEngine

# Physical sizes offered for "Add Region" (label, token, (width, height) in mm)
REGION_PRESETS: tuple = (
    ("Free (use the drawn box)", "free", None),
    ("ID card 85.6 × 54 mm", "id_card", (85.6, 53.98)),
    ("Business card 90 × 55 mm", "business_card", (90.0, 55.0)),
    ("Passport page 125 × 88 mm", "passport", (125.0, 88.0)),
    ("Custom…", "custom", None),
)
REGION_PRESET_TOKENS = tuple(token for _label, token, _size in REGION_PRESETS)


class SettingsPage(QWidget):
    """All the widgets and actions of the settings page."""

    #: Widget handles mirrored onto ``MainWindow`` after construction so the
    #: Add Region state machine and the tests can keep using the flat names.
    #: ``_type_cells`` is deliberately NOT listed - the page rebuilds that
    #: dict, so the window exposes it as a live property instead.
    PASSTHROUGH_HANDLES = (
        "_set_render_dpi",
        "_set_cap_native",
        "_set_preview_dpi",
        "_set_sensitivity",
        "_set_min_area",
        "_set_max_area",
        "_set_min_confidence",
        "_set_refine",
        "_set_work_dim",
        "_set_region_fixed",
        "_set_region_preset",
        "_set_region_width",
        "_set_region_height",
        "_set_auto_orientation",
        "_set_auto_classify",
        "_set_ocr_enabled",
        "_set_ocr_for_orientation",
        "_set_ocr_engine",
        "_set_ocr_languages",
        "_btn_check_ocr",
        "_ocr_hint",
        "_set_format",
        "_set_quality",
        "_set_output_dir",
        "_set_subfolder",
        "_set_open_folder",
        "_set_template",
        "_template_preview",
        "_type_grid_host",
        "_type_grid",
    )

    def __init__(self, main_window: "QMainWindow", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._mw = main_window
        self._build_settings_page()

    # ------------------------------------------------------------ construction
    def _build_settings_page(self) -> None:
        page = self
        outer = QVBoxLayout(page)
        outer.setContentsMargins(18, 16, 18, 16)
        outer.setSpacing(10)

        title = QLabel("Settings")
        title.setObjectName("PageTitle")
        subtitle = QLabel("Rendering, detection and export behaviour. Settings are stored in your user profile.")
        subtitle.setObjectName("PageSubtitle")
        outer.addWidget(title)
        outer.addWidget(subtitle)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        container = QWidget()
        form_layout = QVBoxLayout(container)
        form_layout.setContentsMargins(2, 2, 2, 2)
        form_layout.setSpacing(10)

        # ------------------------------------------------------------ rendering
        render_group = QGroupBox("PDF rendering")
        render_form = QGridLayout(render_group)
        render_form.setHorizontalSpacing(12)
        render_form.setVerticalSpacing(8)
        self._set_render_dpi = QSpinBox()
        self._set_render_dpi.setRange(150, 600)
        self._set_render_dpi.setSingleStep(25)
        self._set_render_dpi.setSuffix(" dpi")
        self._set_cap_native = QCheckBox("Never upscale scans beyond their native resolution")
        self._set_preview_dpi = QSpinBox()
        self._set_preview_dpi.setRange(80, 220)
        self._set_preview_dpi.setSuffix(" dpi")
        render_form.addWidget(QLabel("Export / detection DPI"), 0, 0)
        render_form.addWidget(self._set_render_dpi, 0, 1)
        render_form.addWidget(QLabel("Preview DPI"), 1, 0)
        render_form.addWidget(self._set_preview_dpi, 1, 1)
        render_form.addWidget(self._set_cap_native, 2, 0, 1, 2)
        form_layout.addWidget(render_group)

        # ------------------------------------------------------------ detection
        detect_group = QGroupBox("Card detection")
        detect_form = QGridLayout(detect_group)
        detect_form.setHorizontalSpacing(12)
        detect_form.setVerticalSpacing(8)
        self._set_sensitivity = QSpinBox()
        self._set_sensitivity.setRange(1, 10)
        self._set_sensitivity.setToolTip("Higher values find fainter document edges but may add false positives")
        self._set_min_area = QDoubleSpinBox()
        self._set_min_area.setRange(0.1, 20.0)
        self._set_min_area.setSingleStep(0.1)
        self._set_min_area.setSuffix(" %")
        self._set_max_area = QDoubleSpinBox()
        self._set_max_area.setRange(30.0, 100.0)
        self._set_max_area.setSingleStep(1.0)
        self._set_max_area.setSuffix(" %")
        self._set_min_confidence = QDoubleSpinBox()
        self._set_min_confidence.setRange(0.0, 0.95)
        self._set_min_confidence.setSingleStep(0.05)
        self._set_refine = QCheckBox("Snap detected edges onto the real image gradients")
        self._set_work_dim = QSpinBox()
        self._set_work_dim.setRange(1200, 4000)
        self._set_work_dim.setSingleStep(100)
        self._set_work_dim.setSuffix(" px")
        detect_form.addWidget(QLabel("Detection sensitivity"), 0, 0)
        detect_form.addWidget(self._set_sensitivity, 0, 1)
        detect_form.addWidget(QLabel("Minimum document area"), 1, 0)
        detect_form.addWidget(self._set_min_area, 1, 1)
        detect_form.addWidget(QLabel("Maximum document area"), 2, 0)
        detect_form.addWidget(self._set_max_area, 2, 1)
        detect_form.addWidget(QLabel("Minimum confidence"), 3, 0)
        detect_form.addWidget(self._set_min_confidence, 3, 1)
        detect_form.addWidget(QLabel("Working resolution"), 4, 0)
        detect_form.addWidget(self._set_work_dim, 4, 1)
        detect_form.addWidget(self._set_refine, 5, 0, 1, 2)
        form_layout.addWidget(detect_group)

        # ------------------------------------------------------- add region size
        region_group = QGroupBox("Add Region size")
        region_form = QGridLayout(region_group)
        region_form.setHorizontalSpacing(12)
        region_form.setVerticalSpacing(8)
        self._set_region_fixed = QCheckBox(
            "Squeeze a drawn region to a fixed physical size (identity card, passport page, …)"
        )
        # The Add Region state machine lives in the main window (it also uses
        # the gallery toolbar combo) - the widgets fire straight into it.
        self._set_region_fixed.toggled.connect(self._mw._on_region_fixed_toggled)
        self._set_region_preset = QComboBox()
        for label, token, _size in REGION_PRESETS:
            self._set_region_preset.addItem(label, token)
        self._set_region_preset.currentIndexChanged.connect(self._mw._on_region_preset_changed)
        self._set_region_preset.activated.connect(self._mw._on_region_preset_activated)
        self._set_region_width = QDoubleSpinBox()
        self._set_region_width.setRange(10.0, 400.0)
        self._set_region_width.setDecimals(1)
        self._set_region_width.setSuffix(" mm")
        self._set_region_width.setToolTip("Width of the region drawn with 'Add Region'")
        self._set_region_width.valueChanged.connect(self._mw._on_region_dimensions_changed)
        self._set_region_height = QDoubleSpinBox()
        self._set_region_height.setRange(10.0, 400.0)
        self._set_region_height.setDecimals(1)
        self._set_region_height.setSuffix(" mm")
        self._set_region_height.setToolTip("Height of the region drawn with 'Add Region'")
        self._set_region_height.valueChanged.connect(self._mw._on_region_dimensions_changed)
        region_note = QLabel(
            "The region keeps the centre of the box you draw and takes this exact size on the scanned page."
        )
        region_note.setObjectName("Faint")
        region_note.setWordWrap(True)
        region_form.addWidget(self._set_region_fixed, 0, 0, 1, 2)
        region_form.addWidget(QLabel("Preset"), 1, 0)
        region_form.addWidget(self._set_region_preset, 1, 1)
        region_form.addWidget(QLabel("Width"), 2, 0)
        region_form.addWidget(self._set_region_width, 2, 1)
        region_form.addWidget(QLabel("Height"), 3, 0)
        region_form.addWidget(self._set_region_height, 3, 1)
        region_form.addWidget(region_note, 4, 0, 1, 2)
        form_layout.addWidget(region_group)

        # ------------------------------------------------- orientation / OCR
        ocr_group = QGroupBox("Orientation and classification")
        ocr_form = QGridLayout(ocr_group)
        ocr_form.setHorizontalSpacing(12)
        ocr_form.setVerticalSpacing(8)
        self._set_auto_orientation = QCheckBox("Rotate documents automatically when the orientation is clear")
        self._set_auto_classify = QCheckBox("Classify documents (Passport / Iqama / Istimara / Driving licence)")
        self._set_ocr_enabled = QCheckBox("Enable local OCR (used for classification and identifiers)")
        self._set_ocr_for_orientation = QCheckBox("Use OCR to resolve 180° orientation (slow)")
        self._set_ocr_engine = QComboBox()
        self._set_ocr_engine.addItem("Automatic", "auto")
        self._set_ocr_engine.addItem("Tesseract", "tesseract")
        self._set_ocr_engine.addItem("EasyOCR", "easyocr")
        self._set_ocr_languages = QLineEdit()
        self._set_ocr_languages.setPlaceholderText("en")
        self._set_ocr_languages.setToolTip(
            "Comma separated ISO codes, e.g. 'en' or 'en,ar'.\n"
            "They are mapped to 'eng+ara' for Tesseract automatically."
        )
        self._btn_check_ocr = QPushButton(icon("search"), " Check")
        self._btn_check_ocr.setToolTip(
            "Test which OCR engine is installed and run a small recognition test"
        )
        self._btn_check_ocr.clicked.connect(self._check_ocr_engines)
        engine_row = QHBoxLayout()
        engine_row.setSpacing(8)
        engine_row.addWidget(self._set_ocr_engine, 1)
        engine_row.addWidget(self._btn_check_ocr)
        ocr_form.addWidget(self._set_auto_orientation, 0, 0, 1, 2)
        ocr_form.addWidget(self._set_auto_classify, 1, 0, 1, 2)
        ocr_form.addWidget(self._set_ocr_enabled, 2, 0, 1, 2)
        ocr_form.addWidget(self._set_ocr_for_orientation, 3, 0, 1, 2)
        ocr_form.addWidget(QLabel("OCR engine"), 4, 0)
        ocr_form.addLayout(engine_row, 4, 1)
        ocr_form.addWidget(QLabel("OCR languages"), 5, 0)
        ocr_form.addWidget(self._set_ocr_languages, 5, 1)
        self._ocr_hint = QLabel("")
        self._ocr_hint.setObjectName("Faint")
        self._ocr_hint.setWordWrap(True)
        ocr_form.addWidget(self._ocr_hint, 6, 0, 1, 2)
        form_layout.addWidget(ocr_group)

        # --------------------------------------------------------------- export
        export_group = QGroupBox("Export")
        export_form = QGridLayout(export_group)
        export_form.setHorizontalSpacing(12)
        export_form.setVerticalSpacing(8)
        self._set_format = QComboBox()
        for label, value in (("JPG", "jpg"), ("PNG", "png"), ("TIFF", "tiff")):
            self._set_format.addItem(label, value)
        self._set_quality = QSpinBox()
        self._set_quality.setRange(60, 100)
        self._set_quality.setSuffix(" %")
        self._set_output_dir = QLineEdit()
        output_browse = QPushButton(icon("folder"), " Browse…")
        output_browse.clicked.connect(self._browse_output_dir)
        output_row = QHBoxLayout()
        output_row.addWidget(self._set_output_dir, 1)
        output_row.addWidget(output_browse)
        self._set_subfolder = QCheckBox("Create a separate sub-folder for each file")
        self._set_open_folder = QCheckBox("Open the output folder when the export finishes")
        self._set_template = QLineEdit()
        self._set_template.textChanged.connect(self._update_filename_preview)
        self._template_preview = QLabel("")
        self._template_preview.setObjectName("Faint")

        export_form.addWidget(QLabel("Image format"), 0, 0)
        export_form.addWidget(self._set_format, 0, 1)
        export_form.addWidget(QLabel("JPEG quality"), 1, 0)
        export_form.addWidget(self._set_quality, 1, 1)
        export_form.addWidget(QLabel("Output folder"), 2, 0)
        export_form.addLayout(output_row, 2, 1)
        export_form.addWidget(self._set_subfolder, 3, 0, 1, 2)
        export_form.addWidget(self._set_open_folder, 4, 0, 1, 2)
        export_form.addWidget(QLabel("Filename format"), 5, 0)
        export_form.addWidget(self._set_template, 5, 1)
        export_form.addWidget(self._template_preview, 6, 1)
        form_layout.addWidget(export_group)

        # --------------------------------------------------------- document types
        types_group = QGroupBox("Document types")
        types_layout = QVBoxLayout(types_group)
        types_layout.setSpacing(8)
        types_hint = QLabel(
            "All built-in types and your own types are listed here. Untick a type to hide it from the "
            "document type pickers and from automatic classification — built-in types cannot be deleted. "
            "Keywords are matched against the OCR text (enable OCR to use them); use “Keywords…” on a "
            "tile to add words in any language."
        )
        types_hint.setObjectName("Faint")
        types_hint.setWordWrap(True)
        types_layout.addWidget(types_hint)

        self._type_grid_host = QWidget()
        self._type_grid = QGridLayout(self._type_grid_host)
        self._type_grid.setContentsMargins(0, 0, 0, 0)
        self._type_grid.setHorizontalSpacing(10)
        self._type_grid.setVerticalSpacing(8)
        self._type_cells: Dict[str, Dict[str, object]] = {}
        types_layout.addWidget(self._type_grid_host)

        type_buttons = QHBoxLayout()
        add_type_button = QPushButton(icon("plus"), " Add type…")
        add_type_button.clicked.connect(self._add_document_type)
        type_buttons.addWidget(add_type_button)
        type_buttons.addStretch(1)
        types_layout.addLayout(type_buttons)
        form_layout.addWidget(types_group)
        form_layout.addStretch(1)

        scroll.setWidget(container)
        outer.addWidget(scroll, 1)

        buttons = QHBoxLayout()
        save_button = QPushButton(button_icon("save", "primary"), " Save settings")
        save_button.setProperty("variant", "primary")
        save_button.clicked.connect(self._save_settings_from_form)
        defaults_button = QPushButton("Restore defaults")
        defaults_button.clicked.connect(self._restore_settings_defaults)
        buttons.addWidget(save_button)
        buttons.addWidget(defaults_button)
        buttons.addStretch(1)
        outer.addLayout(buttons)

    # ----------------------------------------------------------- form <-> settings
    def _load_settings_into_form(self) -> None:
        settings = self._mw.settings
        self._set_render_dpi.setValue(int(settings.render_dpi))
        self._set_cap_native.setChecked(bool(settings.cap_dpi_to_native))
        self._set_preview_dpi.setValue(int(settings.preview_dpi))
        self._set_sensitivity.setValue(int(settings.detection_sensitivity))
        self._set_min_area.setValue(float(settings.min_document_area) * 100.0)
        self._set_max_area.setValue(float(settings.max_document_area) * 100.0)
        self._set_min_confidence.setValue(float(settings.detection_min_confidence))
        self._set_refine.setChecked(bool(settings.refine_quads))
        self._set_work_dim.setValue(int(settings.detection_working_max_dim))
        self._set_auto_orientation.setChecked(bool(settings.auto_orientation))
        self._set_auto_classify.setChecked(bool(settings.auto_classify))
        self._set_ocr_enabled.setChecked(bool(settings.ocr_enabled))
        self._set_ocr_for_orientation.setChecked(bool(settings.ocr_for_orientation))
        index = self._set_ocr_engine.findData(settings.ocr_engine)
        self._set_ocr_engine.setCurrentIndex(max(0, index))
        self._set_ocr_languages.setText(settings.ocr_languages)
        format_index = self._set_format.findData(settings.export_format)
        self._set_format.setCurrentIndex(max(0, format_index))
        self._set_quality.setValue(int(settings.jpeg_quality))
        self._set_output_dir.setText(settings.output_dir)
        self._set_subfolder.setChecked(bool(settings.create_subfolder_per_pdf))
        self._set_open_folder.setChecked(bool(settings.open_output_after_export))
        self._set_template.setText(settings.filename_template)
        self._update_filename_preview()
        self._refresh_type_list()
        self._mw._normalise_region_preset()
        self._mw._sync_region_controls()
        self._refresh_ocr_hint()

    def _check_ocr_engines(self) -> None:
        """Test the configured engine(s) and show the diagnostic report."""
        languages = [
            item.strip()
            for item in (self._set_ocr_languages.text() or "en").split(",")
            if item.strip()
        ] or ["en"]
        prefer = str(self._set_ocr_engine.currentData() or "auto")

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self._mw._progress.set_message("Checking the OCR engine…")
        try:
            report = OcrEngine.diagnostic_report(tuple(languages), prefer=prefer)
        except Exception as exc:  # pragma: no cover - defensive
            report = f"The OCR check failed:\n{exc}"
        finally:
            QApplication.restoreOverrideCursor()

        usable = OcrEngine(tuple(languages), prefer=prefer).is_available
        self._mw._progress.set_message(
            "OCR engine is ready" if usable else "No usable OCR engine found"
        )
        self._refresh_ocr_hint()
        if usable:
            QMessageBox.information(self, "OCR check", report)
        else:
            QMessageBox.warning(self, "OCR check", report)

    def _refresh_ocr_hint(self) -> None:
        """Update the small hint line under the OCR settings."""
        engines = OcrEngine.describe_availability()
        if OcrEngine.easyocr_available() and not OcrEngine.easyocr_models_available():
            self._ocr_hint.setText(
                "EasyOCR is installed but its model files are missing, so OCR cannot run "
                f"(the app never downloads them). Copy the model files to "
                f"'{OcrEngine.easyocr_model_dir()}' - see docs/OCR_SETUP.md or press Check."
            )
        elif engines:
            self._ocr_hint.setText(
                f"Detected local OCR engine(s): {engines}. EasyOCR can be slow on older CPUs; "
                "leave the 180° option off unless you need it. Press Check to test the engine."
            )
        else:
            self._ocr_hint.setText(
                "No OCR engine found - classification falls back to layout heuristics and "
                "identifiers are not extracted. Install Tesseract (simplest, docs/OCR_SETUP.md) "
                "or EasyOCR, then press Check."
            )

    def _save_settings_from_form(self) -> None:
        settings = self._mw.settings
        settings.render_dpi = int(self._set_render_dpi.value())
        settings.cap_dpi_to_native = bool(self._set_cap_native.isChecked())
        settings.preview_dpi = int(self._set_preview_dpi.value())
        settings.detection_sensitivity = int(self._set_sensitivity.value())
        settings.min_document_area = float(self._set_min_area.value()) / 100.0
        settings.max_document_area = float(self._set_max_area.value()) / 100.0
        settings.detection_min_confidence = float(self._set_min_confidence.value())
        settings.refine_quads = bool(self._set_refine.isChecked())
        settings.detection_working_max_dim = int(self._set_work_dim.value())
        settings.manual_region_fixed_size = bool(self._set_region_fixed.isChecked())
        settings.manual_region_width_mm = float(self._set_region_width.value())
        settings.manual_region_height_mm = float(self._set_region_height.value())
        # the preset token is authoritative for "fixed size" and "custom"
        if not settings.manual_region_fixed_size:
            settings.manual_region_preset = "free"
        elif settings.manual_region_preset == "free":
            settings.manual_region_preset = "custom"
        settings.auto_orientation = bool(self._set_auto_orientation.isChecked())
        settings.auto_classify = bool(self._set_auto_classify.isChecked())
        settings.ocr_enabled = bool(self._set_ocr_enabled.isChecked())
        settings.ocr_for_orientation = bool(self._set_ocr_for_orientation.isChecked())
        settings.ocr_engine = str(self._set_ocr_engine.currentData() or "auto")
        settings.ocr_languages = self._set_ocr_languages.text().strip() or "en"
        settings.export_format = str(self._set_format.currentData() or "jpg")
        settings.jpeg_quality = int(self._set_quality.value())
        settings.output_dir = self._set_output_dir.text().strip() or str(Path("output"))
        settings.create_subfolder_per_pdf = bool(self._set_subfolder.isChecked())
        settings.open_output_after_export = bool(self._set_open_folder.isChecked())
        settings.filename_template = self._set_template.text().strip() or "{type}_{identifier}_{page}_{sequence}"
        if not settings.save():
            QMessageBox.warning(
                self,
                "Settings not saved",
                "The settings could not be written to:\n\n"
                f"{settings_path()}\n\n"
                "The options stay active for this session, but they will not "
                "survive a restart. Check the disk/permissions and try again.",
            )
            self._mw._progress.set_message("Settings could not be saved")
            return
        self._mw._progress.set_message("Settings saved")
        QMessageBox.information(
            self,
            "Settings saved",
            "Your settings have been saved to:\n\n"
            f"{settings_path()}\n\n"
            "They are used the next time you start detection or export.",
        )

    def _restore_settings_defaults(self) -> None:
        confirm = QMessageBox.question(
            self,
            "Restore defaults",
            "Reset every setting to its default value?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        current_recent = list(self._mw.settings.recent_files)
        self._mw.settings = AppSettings()
        self._mw.settings.recent_files = current_recent
        self._mw.settings.apply_document_types()
        self._load_settings_into_form()
        self._mw._populate_type_combo()

    def _browse_output_dir(self) -> None:
        start = self._set_output_dir.text().strip() or str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Select the output folder", start)
        if folder:
            self._set_output_dir.setText(folder)

    # --------------------------------------------------- custom document types
    def _type_grid_columns(self) -> int:
        """Two tiles per row, one on a narrow window."""
        width = self._type_grid_host.width()
        return 1 if 0 < width < 620 else 2

    def _clear_type_grid(self) -> None:
        while self._type_grid.count():
            item = self._type_grid.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._type_cells = {}

    def _type_summary(self, doc_type: DocumentType, known_words, user_words) -> str:
        parts = [doc_type.category]
        if known_words:
            label = "built-in keyword(s)" if doc_type.builtin else "keyword(s)"
            parts.append(f"{len(known_words)} {label}")
        if user_words:
            shown = ", ".join(user_words[:4]) + ("…" if len(user_words) > 4 else "")
            parts.append(f"your keywords: {shown}")
        if doc_type.is_known and not known_words and not user_words:
            parts.append("no keywords")
        return "   ·   ".join(parts)

    def _refresh_type_list(self) -> None:
        """Fill the settings grid with one tile per type (built-in + custom)."""
        if not hasattr(self, "_type_grid"):
            return
        from core.document_classifier import builtin_keywords

        self._clear_type_grid()
        columns = self._type_grid_columns()
        for index, doc_type in enumerate(DocumentType.all_choices()):
            user_words = list(doc_type.user_keywords)
            known_words = builtin_keywords(doc_type) if doc_type.is_known else []

            cell = QFrame()
            cell.setObjectName("TypeCell")
            cell.setProperty("enabledType", "true" if doc_type.enabled else "false")
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(10, 8, 10, 8)
            cell_layout.setSpacing(3)

            top = QHBoxLayout()
            top.setSpacing(6)
            check = QCheckBox(doc_type.display_name)
            check.setObjectName("TypeName")
            check.setChecked(doc_type.enabled)
            if not doc_type.is_known:
                check.setEnabled(False)  # "Unknown" is always available
            check.toggled.connect(
                lambda checked, token=doc_type.token: self._on_type_toggled(token, checked)
            )
            top.addWidget(check, 1)

            keywords_btn = None
            if doc_type.is_known:
                keywords_btn = QToolButton()
                keywords_btn.setObjectName("TypeAction")
                keywords_btn.setText("Keywords…")
                keywords_btn.setToolTip(
                    "Add OCR keywords to this type (any language) - works for built-in types too"
                )
                keywords_btn.clicked.connect(
                    lambda _checked=False, token=doc_type.token: self._edit_keywords_for(token)
                )
                top.addWidget(keywords_btn)
            cell_layout.addLayout(top)

            info = QLabel(self._type_summary(doc_type, known_words, user_words))
            info.setObjectName("Faint")
            cell_layout.addWidget(info)

            tooltip = (
                f"{doc_type.display_name} — {doc_type.category} type"
                + ("" if doc_type.builtin else " (you can remove it)")
                + "\nTick to offer it in the pickers, untick to hide it"
                + f"\nFilename token: {doc_type.file_token}"
            )
            if known_words:
                tooltip += f"\nBuilt-in OCR keywords: {', '.join(known_words)}"
            if user_words:
                tooltip += f"\nYour extra keywords: {', '.join(user_words)}"
            cell.setToolTip(tooltip)
            info.setToolTip(tooltip)
            check.setToolTip(tooltip)

            remove_btn = None
            if not doc_type.builtin:
                remove_btn = QToolButton()
                remove_btn.setObjectName("TypeAction")
                remove_btn.setText("Remove")
                remove_btn.setToolTip("Delete this document type")
                remove_btn.clicked.connect(
                    lambda _checked=False, token=doc_type.token: self._remove_document_type(token)
                )
                cell_layout.addWidget(remove_btn, 0, Qt.AlignmentFlag.AlignRight)

            row, column = divmod(index, columns)
            self._type_grid.addWidget(cell, row, column)
            self._type_cells[doc_type.token] = {
                "cell": cell,
                "check": check,
                "info": info,
                "keywords": keywords_btn,
                "remove": remove_btn,
                "builtin": doc_type.builtin,
            }
        for column in range(columns):
            self._type_grid.setColumnStretch(column, 1)

    def _set_type_tile_enabled_look(self, token: str, enabled: bool) -> None:
        """Grey out a tile that the user hid."""
        cell_info = self._type_cells.get(token)
        if not cell_info:
            return
        cell = cell_info.get("cell")
        if isinstance(cell, QFrame):
            cell.setProperty("enabledType", "true" if enabled else "false")
            style = cell.style()
            if style is not None:
                style.unpolish(cell)
                style.polish(cell)

    def _on_type_toggled(self, token: str, checked: bool) -> None:
        """The user ticked / unticked a document type tile."""
        if not token:
            return
        settings = self._mw.settings
        if not settings.set_type_enabled(token, bool(checked)):
            return
        settings.save()
        self._mw._populate_type_combo()
        self._set_type_tile_enabled_look(token, bool(checked))
        doc_type = DocumentType.from_value(token)
        verb = "enabled" if checked else "hidden"
        self._mw._progress.set_message(f"{doc_type.display_name} {verb}")

    def _edit_keywords_for(self, token: str) -> None:
        """Add extra OCR keywords to *token* (built-in types included)."""
        from core.document_classifier import builtin_keywords

        settings = self._mw.settings
        doc_type = DocumentType.from_value(token) if token else DocumentType.UNKNOWN
        if doc_type is DocumentType.UNKNOWN:
            QMessageBox.information(
                self, "Edit keywords", "This document type does not use OCR keywords."
            )
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(f"OCR keywords · {doc_type.display_name}")
        dialog.setMinimumWidth(520)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        known = builtin_keywords(doc_type)
        header = QLabel(
            "Additional words that appear on this document. One per line or separated by "
            "commas; they are matched in addition to the built-in keywords, so you can add "
            "another language here."
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        builtin_label = QLabel(
            "Built-in keywords: " + (", ".join(known) if known else "(none)")
        )
        builtin_label.setObjectName("Faint")
        builtin_label.setWordWrap(True)
        layout.addWidget(builtin_label)

        editor = QPlainTextEdit("\n".join(settings.type_keywords(token)))
        editor.setPlaceholderText("e.g. بطاقة الهوية\nNATIONAL IDENTITY CARD")
        editor.setMinimumHeight(120)
        layout.addWidget(editor, 1)

        if doc_type.builtin:
            note = QLabel(
                "Built-in types cannot be deleted - only extended. Clear the box to go back "
                "to the built-in keywords."
            )
        else:
            note = QLabel("Clear the box to remove all additional keywords of this type.")
        note.setObjectName("Faint")
        note.setWordWrap(True)
        layout.addWidget(note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        save_keywords = buttons.button(QDialogButtonBox.StandardButton.Ok)
        if save_keywords is not None:
            save_keywords.setText("Save keywords")
            save_keywords.setProperty("variant", "primary")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() == 0:
            return
        keywords = editor.toPlainText()
        if settings.set_type_keywords(token, keywords):
            settings.save()
            self._refresh_type_list()
            count = len(settings.type_keywords(token))
            self._mw._progress.set_message(
                f"{count} extra keyword(s) stored for {doc_type.display_name}"
                if count
                else f"{doc_type.display_name}: back to the built-in keywords"
            )

    def _add_document_type(self) -> None:
        label, accepted = QInputDialog.getText(
            self, "Add document type", "Name of the new document type (e.g. National ID):"
        )
        if not accepted or not label.strip():
            return
        keywords, keywords_accepted = QInputDialog.getText(
            self,
            "OCR keywords (optional)",
            "Words that appear on this document, separated by commas.\n"
            "Used for automatic classification when OCR is enabled:",
        )
        settings = self._mw.settings
        entry = settings.add_custom_type(label, keywords if keywords_accepted else "")
        if entry is None:
            QMessageBox.information(
                self,
                "Document type",
                f"'{label.strip()}' is already a known document type.\n"
                "Choose a different name (or edit the existing type).",
            )
            return
        settings.save()
        self._refresh_type_list()
        self._mw._populate_type_combo()
        self._mw._progress.set_message(f"Document type '{entry['label']}' added")

    def _remove_document_type(self, token: str = "") -> None:
        """Delete a custom type (built-ins explain why they cannot be removed)."""
        token = str(token or "").strip()
        if not token:
            QMessageBox.information(self, "Remove document type", "Choose a type first.")
            return
        doc_type = DocumentType.from_value(token)
        if doc_type is DocumentType.UNKNOWN:
            return
        if doc_type.builtin:
            QMessageBox.information(
                self,
                "Built-in document type",
                f"'{doc_type.display_name}' is a built-in type and cannot be removed.\n\n"
                "Untick it instead to hide it from the document type pickers.",
            )
            return
        confirm = QMessageBox.question(
            self,
            "Remove document type",
            f"Delete the document type '{doc_type.display_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        settings = self._mw.settings
        if not settings.remove_custom_type(token):
            return
        settings.save()
        self._refresh_type_list()
        self._mw._populate_type_combo()
        self._mw._progress.set_message(f"Document type '{doc_type.display_name}' removed")

    def _update_filename_preview(self) -> None:
        template = self._set_template.text().strip() or "{type}_{identifier}_{page}_{sequence}"
        options = ExportOptions(filename_template=template)
        from core.image_exporter import ImageExporter

        sample_quad = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        sample = DetectedCard(page_index=0, quad_norm=sample_quad)
        sample.set_type(DocumentType.IQAMA, 1.0)
        example = ImageExporter(options).build_filename(sample, 2, "scan", 1)
        self._template_preview.setText(
            f"Example: {example}{options.extension()}   ·   placeholders: "
            "{type} {identifier} {page} {sequence} {pdf}"
        )
