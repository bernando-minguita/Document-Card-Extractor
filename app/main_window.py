"""Main application window."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import zlib
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app import styles
from app.history import HistoryEntry, HistoryStore
from app.history_page import HistoryPage
from app.icons import app_icon, apply_icon_size, button_icon, icon
from app.session import DocumentSession
from app.settings import AppSettings
from app.settings_page import REGION_PRESET_TOKENS, REGION_PRESETS, SettingsPage
from core.image_exporter import ExportOptions, ExportSummary
from core.models import DetectedCard, DocumentType
from core.ocr_engine import OcrEngine
from core.pdf_loader import PdfDocument, PdfError, is_supported_file, supported_file_filter
from core.perspective_corrector import make_thumbnail, rotate_image, warp_quad
from widgets.crop_editor import CropEditorDialog
from widgets.document_gallery import DocumentGallery, GalleryFilter
from widgets.pdf_viewer import OverlaySpec, PdfPageView
from widgets.progress_widget import ProgressPanel
from widgets.qt_image import bgr_to_qimage
from widgets.region_size_dialog import RegionSizeDialog
from workers.detection_worker import DetectionWorker
from workers.export_worker import ExportJob, ExportWorker

APP_NAME = "Document Card Extractor"
APP_VERSION = "1.0.0"

TYPE_COLORS = {
    # Keyed by token (not by type object) so the mapping survives reloads of
    # the registry and stays stable while custom types come and go.
    "passport": "#2563eb",
    "iqama": "#0d9488",
    "istimara": "#f59e0b",
    "driving_license": "#7c3aed",
    "health_insurance": "#0284c7",
    "bank_card": "#be123c",
    "salary_card": "#15803d",
    "national_id": "#9333ea",
    "unknown": "#64748b",
}


class MainWindow(QMainWindow):
    """Everything the user interacts with."""

    def __init__(self, settings: Optional[AppSettings] = None) -> None:
        super().__init__()
        self.settings = settings or AppSettings.load()
        self.settings.apply_document_types()
        self.history = HistoryStore()

        self._sessions: List[DocumentSession] = []
        self._session_index = -1
        self._page_index = 0
        self._page_image: Optional[np.ndarray] = None
        self._page_dpi: float = 0.0
        self._region_pending_dialog: int = -1
        self._preview_document: Optional[PdfDocument] = None
        # A freshly probed document not yet attached to the preview (avoids
        # opening/decoding the same file twice in load_pdf).
        self._pending_preview_document: Optional[PdfDocument] = None
        self._selected_card_id = ""
        self._detection_worker: Optional[DetectionWorker] = None
        self._export_worker: Optional[ExportWorker] = None
        self._detection_errors: List[str] = []
        self._export_errors: List[str] = []
        self._active_export_jobs: List[ExportJob] = []
        self._busy = False
        # Thumbnail rebuilds are a full perspective warp; arrow-key nudges and
        # corner drags trigger them continuously, so they are coalesced.
        self._thumb_timer = QTimer(self)
        self._thumb_timer.setSingleShot(True)
        self._thumb_timer.setInterval(30)
        self._thumb_timer.timeout.connect(self._flush_thumbnails)
        self._thumb_pending: Optional[DetectedCard] = None

        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        self.setWindowIcon(app_icon())
        self.setAcceptDrops(True)
        self.resize(1440, 900)
        self.setMinimumSize(1100, 720)

        self._build_ui()
        self._create_shortcuts()
        # a slightly larger icon keeps the 1-px strokes of the SVG icons visible
        apply_icon_size(self)
        self._update_state()

    # ------------------------------------------------------------------ session
    @property
    def session(self) -> Optional[DocumentSession]:
        if 0 <= self._session_index < len(self._sessions):
            return self._sessions[self._session_index]
        return None

    # --------------------------------------------------------------------- ui
    def _build_ui(self) -> None:
        central = QWidget(self)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_sidebar())

        self._pages = QStackedWidget()
        self._pages.addWidget(self._build_dashboard_page())
        self._pages.addWidget(self._build_extract_page())
        self._pages.addWidget(self._build_history_page())
        self._pages.addWidget(self._build_settings_page())
        layout.addWidget(self._pages, 1)
        self.setCentralWidget(central)

        self._build_status_bar()
        self._load_settings_into_form()
        # The dashboard is the start page, so its counters and the recent file
        # list must be filled right away (not only when the nav button is used).
        self._refresh_dashboard()

    def _build_sidebar(self) -> QWidget:
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(236)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(14, 18, 14, 14)
        layout.setSpacing(6)

        brand_row = QHBoxLayout()
        brand_icon = QLabel()
        brand_icon.setPixmap(icon("extract", styles.PRIMARY).pixmap(28, 28))
        brand_row.addWidget(brand_icon)
        brand_text = QVBoxLayout()
        brand_text.setSpacing(0)
        title = QLabel("Card Extractor")
        title.setObjectName("BrandTitle")
        subtitle = QLabel("Scans → card images")
        subtitle.setObjectName("BrandSubtitle")
        brand_text.addWidget(title)
        brand_text.addWidget(subtitle)
        brand_row.addLayout(brand_text)
        brand_row.addStretch(1)
        layout.addLayout(brand_row)
        layout.addSpacing(18)

        self._nav_buttons: List[QPushButton] = []
        for index, (label, icon_name) in enumerate(
            (
                ("Dashboard", "dashboard"),
                ("Extract Documents", "extract"),
                ("Extraction History", "history"),
                ("Settings", "settings"),
            )
        ):
            button = QPushButton(icon(icon_name, styles.TEXT_MUTED), f"  {label}")
            button.setObjectName("NavButton")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setIconSize(QSize(18, 18))
            button.clicked.connect(lambda _checked=False, i=index: self._switch_page(i))
            layout.addWidget(button)
            self._nav_buttons.append(button)
        self._nav_buttons[0].setChecked(True)

        layout.addStretch(1)

        engine = OcrEngine.describe_availability()
        ocr_label = QLabel(f"OCR engine: {engine or 'not installed'}")
        ocr_label.setObjectName("Faint")
        ocr_label.setWordWrap(True)
        layout.addWidget(ocr_label)
        version = QLabel(f"Version {APP_VERSION} · offline")
        version.setObjectName("Faint")
        layout.addWidget(version)
        return sidebar

    def _build_status_bar(self) -> None:
        status = self.statusBar()
        status.setSizeGripEnabled(True)
        self._status_pages = QLabel("Pages: 0")
        self._status_detected = QLabel("Detected: 0")
        self._status_approved = QLabel("Approved: 0")
        for label in (self._status_pages, self._status_detected, self._status_approved):
            label.setContentsMargins(8, 0, 8, 0)
            status.addWidget(label)

        self._progress = ProgressPanel()
        self._progress.cancelRequested.connect(self._cancel_current_task)
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 8, 0)
        row.addWidget(self._progress)
        status.addPermanentWidget(container)

    def _create_shortcuts(self) -> None:
        open_action = QAction(self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_pdf_dialog)
        self.addAction(open_action)

        detect_action = QAction(self)
        detect_action.setShortcut(QKeySequence("Ctrl+R"))
        detect_action.triggered.connect(lambda: self.start_detection())
        self.addAction(detect_action)

        export_action = QAction(self)
        export_action.setShortcut(QKeySequence("Ctrl+E"))
        export_action.triggered.connect(self.export_all)
        self.addAction(export_action)

        escape_action = QAction(self)
        escape_action.setShortcut(QKeySequence("Esc"))
        escape_action.triggered.connect(self._leave_add_mode)
        self.addAction(escape_action)

    def _switch_page(self, index: int) -> None:
        self._pages.setCurrentIndex(index)
        for position, button in enumerate(self._nav_buttons):
            button.setChecked(position == index)
            color = styles.PRIMARY_DARK if position == index else styles.TEXT_MUTED
            button.setIcon(icon(("dashboard", "extract", "history", "settings")[position], color))
        if index == 0:
            self._refresh_dashboard()
        elif index == 2:
            self._refresh_history_table()

    # ------------------------------------------------------------ extract page
    def _build_extract_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 14, 18, 10)
        layout.setSpacing(10)

        header = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(2)
        title = QLabel("Extract Documents")
        title.setObjectName("PageTitle")
        subtitle = QLabel("Detect cards on a scanned page, review them, then export each one as an image.")
        subtitle.setObjectName("PageSubtitle")
        titles.addWidget(title)
        titles.addWidget(subtitle)
        header.addLayout(titles)
        header.addStretch(1)
        self._document_combo = QComboBox()
        self._document_combo.setMinimumWidth(300)
        self._document_combo.currentIndexChanged.connect(self._on_document_changed)
        header.addWidget(self._document_combo)
        layout.addLayout(header)

        toolbar = QFrame()
        toolbar.setObjectName("Toolbar")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(10, 8, 10, 8)
        toolbar_layout.setSpacing(8)

        self._btn_open = QPushButton(button_icon("folder", "primary"), " Open File")
        self._btn_open.setProperty("variant", "primary")
        self._btn_open.setToolTip("Open a scanned PDF or an image (JPG, PNG, TIFF, BMP, WEBP)")
        self._btn_open.clicked.connect(self.open_pdf_dialog)

        self._btn_add = QPushButton(icon("plus"), " Add File")
        self._btn_add.setToolTip("Add another PDF or image to work on side by side")
        self._btn_add.clicked.connect(self.add_pdf_dialog)

        self._btn_detect = QPushButton(icon("play"), " Start Detection")
        self._btn_detect.clicked.connect(lambda: self.start_detection())

        self._btn_redetect = QPushButton(icon("refresh"), " Re-run Page")
        self._btn_redetect.clicked.connect(lambda: self.start_detection([self._page_index]))

        self._btn_export = QPushButton(icon("export"), " Export All")
        self._btn_export.clicked.connect(self.export_all)

        for button in (self._btn_open, self._btn_add, self._btn_detect, self._btn_redetect, self._btn_export):
            button.setObjectName("ToolbarButton")
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            toolbar_layout.addWidget(button)
        toolbar_layout.addStretch(1)
        layout.addWidget(toolbar)

        # --------------------------------------------------------- page strip
        page_bar = QHBoxLayout()
        page_bar.setSpacing(6)
        self._btn_prev = QToolButton()
        self._btn_prev.setObjectName("IconButton")
        self._btn_prev.setIcon(icon("chevron_left"))
        self._btn_prev.setToolTip("Previous page")
        self._btn_prev.clicked.connect(lambda: self.select_page(self._page_index - 1))
        self._btn_next = QToolButton()
        self._btn_next.setObjectName("IconButton")
        self._btn_next.setIcon(icon("chevron_right"))
        self._btn_next.setToolTip("Next page")
        self._btn_next.clicked.connect(lambda: self.select_page(self._page_index + 1))
        self._page_spin = QSpinBox()
        self._page_spin.setRange(1, 1)
        self._page_spin.setFixedWidth(70)
        self._page_spin.valueChanged.connect(lambda value: self.select_page(value - 1))
        self._page_total_label = QLabel("of 0")
        self._page_total_label.setObjectName("Muted")

        page_bar.addWidget(self._btn_prev)
        page_bar.addWidget(self._page_spin)
        page_bar.addWidget(self._page_total_label)
        page_bar.addWidget(self._btn_next)
        page_bar.addSpacing(16)

        # Page range for detection (optional, empty = all pages)
        self._page_range_edit = QLineEdit()
        self._page_range_edit.setPlaceholderText("Pages (e.g. 1,3,5-7)")
        self._page_range_edit.setFixedWidth(160)
        self._page_range_edit.setToolTip("Pages to run detection on. Format: 1,3,5-7 (empty = all pages)")
        page_bar.addWidget(self._page_range_edit)

        self._btn_zoom_out = QToolButton()
        self._btn_zoom_out.setObjectName("IconButton")
        self._btn_zoom_out.setIcon(icon("zoom_out"))
        self._btn_zoom_out.clicked.connect(lambda: self._view.zoom_out())
        self._btn_zoom_in = QToolButton()
        self._btn_zoom_in.setObjectName("IconButton")
        self._btn_zoom_in.setIcon(icon("zoom_in"))
        self._btn_zoom_in.clicked.connect(lambda: self._view.zoom_in())
        self._btn_fit = QToolButton()
        self._btn_fit.setObjectName("IconButton")
        self._btn_fit.setIcon(icon("fit"))
        self._btn_fit.setToolTip("Fit page to window")
        self._btn_fit.clicked.connect(lambda: self._view.fit_to_window())
        for button in (self._btn_zoom_out, self._btn_zoom_in, self._btn_fit):
            page_bar.addWidget(button)

        page_bar.addSpacing(16)
        self._btn_add_region = QPushButton(icon("plus"), " Add Region")
        self._btn_add_region.setCheckable(True)
        self._btn_add_region.setToolTip("Draw a rectangle to add a document the detector missed (Esc to cancel)")
        self._btn_add_region.toggled.connect(self._toggle_add_mode)

        self._btn_delete = QPushButton(icon("trash"), " Remove")
        self._btn_delete.setToolTip("Remove the selected document from the list")
        self._btn_delete.clicked.connect(self._delete_selected_card)

        page_bar.addWidget(self._btn_add_region)
        self._region_preset_combo = QComboBox()
        self._region_preset_combo.setMinimumWidth(190)
        for label, token, _size in REGION_PRESETS:
            self._region_preset_combo.addItem(label, token)
        self._region_preset_combo.setToolTip(
            "Size of the region you draw with 'Add Region'\n"
            "e.g. ID card 85.6 × 54 mm so the crop matches the document exactly"
        )
        self._region_preset_combo.currentIndexChanged.connect(self._on_region_preset_changed)
        # re-selecting "Custom…" while it is already current does not change the
        # index, so the dedicated button gives the dialog an always-working path
        self._region_preset_combo.activated.connect(self._on_region_preset_activated)
        page_bar.addWidget(self._region_preset_combo)

        self._region_size_button = QToolButton()
        self._region_size_button.setText("Size…")
        self._region_size_button.setToolTip("Enter a custom width and height for the drawn region")
        self._region_size_button.clicked.connect(lambda: self._open_region_size_dialog())
        page_bar.addWidget(self._region_size_button)
        page_bar.addWidget(self._btn_delete)
        page_bar.addStretch(1)

        self._page_hint = QLabel("")
        self._page_hint.setObjectName("Faint")
        page_bar.addWidget(self._page_hint)
        layout.addLayout(page_bar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        self._view = PdfPageView()
        self._view.cardSelected.connect(self._on_card_selected)
        self._view.cardActivated.connect(self.edit_crop)
        self._view.quadEdited.connect(self._on_quad_edited)
        self._view.manualRectDrawn.connect(self._on_manual_rect)
        splitter.addWidget(self._view)
        splitter.addWidget(self._build_gallery_panel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([780, 560])
        layout.addWidget(splitter, 1)
        return page

    def _build_gallery_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("Card")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel("Extracted documents")
        title.setObjectName("SectionTitle")
        self._gallery_count = QLabel("0")
        self._gallery_count.setObjectName("Faint")
        header.addWidget(title)
        header.addWidget(self._gallery_count)
        header.addStretch(1)
        self._gallery_filter = QComboBox()
        self._gallery_filter.addItem("This page", "page")
        self._gallery_filter.addItem("All pages", "all")
        self._gallery_filter.currentIndexChanged.connect(lambda _index: self._refresh_gallery())
        header.addWidget(self._gallery_filter)
        layout.addLayout(header)

        # Filter bar
        filter_bar = QHBoxLayout()
        filter_bar.setSpacing(6)
        self._gallery_search = QLineEdit()
        self._gallery_search.setPlaceholderText("Search (type, identifier, OCR text, page…)")
        self._gallery_search.setClearButtonEnabled(True)
        self._gallery_search.textChanged.connect(lambda _t: self._refresh_gallery())
        self._gallery_type_filter = QComboBox()
        self._gallery_type_filter.addItem("All types", None)
        self._gallery_type_filter.setMinimumWidth(160)
        self._gallery_type_filter.currentIndexChanged.connect(lambda _i: self._refresh_gallery())
        self._gallery_page_filter = QComboBox()
        self._gallery_page_filter.addItem("All pages", None)
        self._gallery_page_filter.setMinimumWidth(100)
        self._gallery_page_filter.currentIndexChanged.connect(lambda _i: self._refresh_gallery())
        self._gallery_included_only = QCheckBox("Included only")
        self._gallery_included_only.toggled.connect(lambda _c: self._refresh_gallery())
        self._gallery_confident_only = QCheckBox("Orientation OK only")
        self._gallery_confident_only.toggled.connect(lambda _c: self._refresh_gallery())
        filter_bar.addWidget(self._gallery_search, 1)
        filter_bar.addWidget(self._gallery_type_filter)
        filter_bar.addWidget(self._gallery_page_filter)
        filter_bar.addWidget(self._gallery_included_only)
        filter_bar.addWidget(self._gallery_confident_only)
        layout.addLayout(filter_bar)

        self._gallery = DocumentGallery()
        self._gallery.cardSelected.connect(self._on_card_selected)
        self._gallery.cardActivated.connect(self.edit_crop)
        self._gallery.rotateRequested.connect(self.rotate_card)
        self._gallery.typeChangeRequested.connect(self.set_card_type)
        self._gallery.includeToggled.connect(self.toggle_card_included)
        self._gallery.cropEditRequested.connect(self.edit_crop)
        self._gallery.ocrTextRequested.connect(self.show_ocr_text)
        self._gallery.removeRequested.connect(self.remove_card)
        self._gallery.exportRequested.connect(self.export_card)
        layout.addWidget(self._gallery, 1)

        edit_row = QHBoxLayout()
        edit_row.setSpacing(6)
        self._btn_rot_left = QToolButton()
        self._btn_rot_left.setObjectName("IconButton")
        self._btn_rot_left.setIcon(icon("rotate_left"))
        self._btn_rot_left.setToolTip("Rotate 90° left")
        self._btn_rot_left.clicked.connect(lambda: self._rotate_selected(270))
        self._btn_rot_right = QToolButton()
        self._btn_rot_right.setObjectName("IconButton")
        self._btn_rot_right.setIcon(icon("rotate_right"))
        self._btn_rot_right.setToolTip("Rotate 90° right")
        self._btn_rot_right.clicked.connect(lambda: self._rotate_selected(90))
        self._btn_rot_180 = QToolButton()
        self._btn_rot_180.setObjectName("IconButton")
        self._btn_rot_180.setIcon(icon("flip"))
        self._btn_rot_180.setToolTip("Rotate 180°")
        self._btn_rot_180.clicked.connect(lambda: self._rotate_selected(180))
        self._btn_crop = QToolButton()
        self._btn_crop.setObjectName("IconButton")
        self._btn_crop.setIcon(icon("crop"))
        self._btn_crop.setToolTip("Adjust crop")
        self._btn_crop.clicked.connect(lambda: self.edit_crop(self._selected_card_id))

        self._type_combo = QComboBox()
        self._type_combo.setMinimumWidth(140)
        self._type_combo.activated.connect(self._on_type_combo_changed)
        self._populate_type_combo()

        self._include_check = QCheckBox("Include")
        self._include_check.toggled.connect(self._on_include_check_toggled)

        for widget in (self._btn_rot_left, self._btn_rot_right, self._btn_rot_180, self._btn_crop):
            edit_row.addWidget(widget)
        edit_row.addWidget(self._type_combo, 1)
        edit_row.addWidget(self._include_check)
        layout.addLayout(edit_row)
        return panel

    # ---------------------------------------------------------- dashboard page
    def _build_dashboard_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(18, 16, 18, 16)
        outer.setSpacing(14)

        title = QLabel("Dashboard")
        title.setObjectName("PageTitle")
        subtitle = QLabel(
            "Extract passport, Iqama, Istimara and driving licence cards from scanned "
            "PDF files or images."
        )
        subtitle.setObjectName("PageSubtitle")
        outer.addWidget(title)
        outer.addWidget(subtitle)

        stats_row = QHBoxLayout()
        stats_row.setSpacing(12)
        self._stat_cards: Dict[str, QLabel] = {}
        for key, label in (
            ("loaded", "Files loaded"),
            ("detected", "Documents detected"),
            ("approved", "Approved for export"),
            ("exported", "Documents exported"),
        ):
            card = QFrame()
            card.setObjectName("StatCard")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(16, 14, 16, 14)
            value = QLabel("0")
            value.setObjectName("StatValue")
            caption = QLabel(label)
            caption.setObjectName("StatLabel")
            card_layout.addWidget(value)
            card_layout.addWidget(caption)
            self._stat_cards[key] = value
            stats_row.addWidget(card)
        outer.addLayout(stats_row)

        actions = QFrame()
        actions.setObjectName("Card")
        actions_layout = QVBoxLayout(actions)
        actions_layout.setContentsMargins(16, 14, 16, 14)
        actions_layout.setSpacing(8)
        heading = QLabel("Quick start")
        heading.setObjectName("SectionTitle")
        actions_layout.addWidget(heading)
        hint = QLabel(
            "1. Open a scanned PDF or image (or drag & drop it into the window).\n"
            "2. Press “Start Detection” to find the documents automatically.\n"
            "3. Review the thumbnails, fix the rotation if necessary and set the document type.\n"
            "4. Press “Export All” to write one image file per document."
        )
        hint.setObjectName("Muted")
        hint.setWordWrap(True)
        actions_layout.addWidget(hint)
        button_row = QHBoxLayout()
        open_button = QPushButton(button_icon("folder", "primary"), " Open File")
        open_button.setProperty("variant", "primary")
        open_button.clicked.connect(self.open_pdf_dialog)
        extract_button = QPushButton(icon("extract"), " Go to extraction")
        extract_button.clicked.connect(lambda: self._switch_page(1))
        settings_button = QPushButton(icon("settings"), " Open settings")
        settings_button.clicked.connect(lambda: self._switch_page(3))
        button_row.addWidget(open_button)
        button_row.addWidget(extract_button)
        button_row.addWidget(settings_button)
        button_row.addStretch(1)
        actions_layout.addLayout(button_row)
        outer.addWidget(actions)

        recent = QFrame()
        recent.setObjectName("Card")
        recent_layout = QVBoxLayout(recent)
        recent_layout.setContentsMargins(16, 14, 16, 14)
        recent_layout.setSpacing(8)
        recent_title_row = QHBoxLayout()
        recent_title_row.setSpacing(8)
        recent_title = QLabel("Recent files")
        recent_title.setObjectName("SectionTitle")
        self._recent_clear_btn = QPushButton(icon("trash"), " Clear list")
        self._recent_clear_btn.setProperty("variant", "ghost")
        self._recent_clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._recent_clear_btn.setToolTip("Remove every entry from the recent files list")
        self._recent_clear_btn.clicked.connect(self._clear_recent_files)
        recent_title_row.addWidget(recent_title)
        recent_title_row.addStretch(1)
        recent_title_row.addWidget(self._recent_clear_btn)
        recent_layout.addLayout(recent_title_row)

        self._recent_list = QListWidget()
        self._recent_list.setObjectName("RecentList")
        self._recent_list.setMaximumHeight(190)
        self._recent_list.setCursor(Qt.CursorShape.PointingHandCursor)
        self._recent_list.itemDoubleClicked.connect(self._open_recent_item)
        recent_layout.addWidget(self._recent_list)
        outer.addWidget(recent)
        outer.addStretch(1)
        return page

    def _refresh_dashboard(self) -> None:
        detected = sum(len(session.cards) for session in self._sessions)
        approved = sum(len(session.included_cards()) for session in self._sessions)
        totals = self.history.totals()
        self._stat_cards["loaded"].setText(str(len(self._sessions)))
        self._stat_cards["detected"].setText(str(detected))
        self._stat_cards["approved"].setText(str(approved))
        self._stat_cards["exported"].setText(str(totals["documents"]))

        self._recent_list.clear()
        recent = [path for path in self.settings.recent_files if Path(path).exists()]
        if hasattr(self, "_recent_clear_btn"):
            self._recent_clear_btn.setEnabled(bool(recent))
        if not recent:
            item = QListWidgetItem("No recent files yet — open a PDF or image to get started.")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self._recent_list.addItem(item)
            return
        for path in recent:
            name = Path(path).name
            item = QListWidgetItem(icon("document"), f"{name}     {path}")
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setToolTip(f"{path}\n\nDouble-click to open this file")
            self._recent_list.addItem(item)

    def _clear_recent_files(self) -> None:
        """Empty the "recent files" list (the PDF files themselves are kept)."""
        if not self.settings.recent_files:
            return
        answer = QMessageBox.question(
            self,
            "Clear recent files",
            "Remove all entries from the recent files list?\n\nThe PDF files themselves are not deleted.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.settings.recent_files = []
        self.settings.save()
        self._refresh_dashboard()
        self._progress.set_message("Recent files list cleared")

    def _open_recent_item(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            self.load_pdf(str(path))

    # ------------------------------------------------------------ history page
    def _build_history_page(self) -> QWidget:
        """The history widgets/actions live in ``app.history_page``."""
        page = HistoryPage(self)
        self._history_table = page._history_table
        self._history_page = page
        return page

    def _refresh_history_table(self) -> None:
        self._history_page._refresh_history_table()

    def _open_history_folder(self) -> None:
        self._history_page._open_history_folder()

    def _clear_history(self) -> None:
        self._history_page._clear_history()

    # ----------------------------------------------------------- settings page
    def _build_settings_page(self) -> QWidget:
        """The settings page widgets/actions live in ``app.settings_page``.

        The widget handles are mirrored onto the window so the Add Region
        state machine below (and the tests) can keep using the flat names.
        """
        page = SettingsPage(self)
        for name in SettingsPage.PASSTHROUGH_HANDLES:
            setattr(self, name, getattr(page, name))
        self._settings_page = page
        return page

    @property
    def _type_cells(self) -> Dict[str, Dict[str, object]]:
        """Rebuilt by the settings page on every change - always read live."""
        return self._settings_page._type_cells

    def _load_settings_into_form(self) -> None:
        self._settings_page._load_settings_into_form()

    def _save_settings_from_form(self) -> None:
        self._settings_page._save_settings_from_form()

    def _restore_settings_defaults(self) -> None:
        self._settings_page._restore_settings_defaults()

    def _check_ocr_engines(self) -> None:
        self._settings_page._check_ocr_engines()

    def _refresh_ocr_hint(self) -> None:
        self._settings_page._refresh_ocr_hint()

    def _browse_output_dir(self) -> None:
        self._settings_page._browse_output_dir()

    def _refresh_type_list(self) -> None:
        self._settings_page._refresh_type_list()

    def _type_grid_columns(self) -> int:
        return self._settings_page._type_grid_columns()

    def _clear_type_grid(self) -> None:
        self._settings_page._clear_type_grid()

    def _type_summary(self, doc_type, known_words, user_words) -> str:
        return self._settings_page._type_summary(doc_type, known_words, user_words)

    def _set_type_tile_enabled_look(self, token: str, enabled: bool) -> None:
        self._settings_page._set_type_tile_enabled_look(token, enabled)

    def _on_type_toggled(self, token: str, checked: bool) -> None:
        self._settings_page._on_type_toggled(token, checked)

    def _edit_keywords_for(self, token: str) -> None:
        self._settings_page._edit_keywords_for(token)

    def _add_document_type(self) -> None:
        self._settings_page._add_document_type()

    def _remove_document_type(self, token: str = "") -> None:
        self._settings_page._remove_document_type(token)

    def _update_filename_preview(self) -> None:
        self._settings_page._update_filename_preview()

    # ------------------------------------------------------- document handling
    def open_pdf_dialog(self) -> None:
        path = self._ask_for_pdf("Open PDF or image")
        if path:
            self.load_pdf(path, add=False)

    def add_pdf_dialog(self) -> None:
        path = self._ask_for_pdf("Add PDF or image")
        if path:
            self.load_pdf(path, add=True)

    def _ask_for_pdf(self, caption: str) -> str:
        start = self.settings.last_input_dir or str(Path.home())
        filters = (
            f"Documents and scans ({supported_file_filter()});;"
            "PDF files (*.pdf);;"
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;"
            "All files (*)"
        )
        path, _filter = QFileDialog.getOpenFileName(self, caption, start, filters)
        return path

    def load_pdf(self, path: str, add: bool = False, password: str = "") -> bool:
        """Open *path* and make it the active document."""
        if self._busy:
            # A detection/export worker targets the session that was current
            # when it was started - switching files (or the page reset) mid-run
            # would deliver its results to the wrong document.
            self._progress.set_message("Busy - wait until the current task finishes")
            return False
        pdf_path = Path(path)
        for index, existing in enumerate(self._sessions):
            if existing.path == pdf_path:
                # Already open: switch to it and make sure the workspace is shown
                # (opening it from the dashboard used to leave the user there).
                if index != self._session_index:
                    self._session_index = index
                    self._sync_document_combo()
                    self._activate_session()
                else:
                    self._sync_document_combo()
                self._switch_page(1)
                return True
        try:
            document = PdfDocument(pdf_path, password=password or None)
            page_count = document.page_count
        except PdfError as exc:
            message = str(exc)
            if "password" in message.lower():
                # Prompt (or re-prompt) until the password is correct or the
                # user cancels - a wrong password should not force the user
                # through the file dialog all over again.
                if password:
                    message = f"That password was not accepted.\n{message}"
                entered, ok = QInputDialog.getText(
                    self, "Password required", message, QLineEdit.EchoMode.Password
                )
                if ok and entered:
                    return self.load_pdf(path, add=add, password=entered)
            QMessageBox.critical(self, "Cannot open file", message)
            return False
        except Exception as exc:  # pragma: no cover - unexpected failures
            QMessageBox.critical(self, "Cannot open file", str(exc))
            return False

        session = DocumentSession(path=pdf_path, page_count=page_count, password=password or "")
        # keep the already-open document for the preview instead of reopening
        self._pending_preview_document = document
        if add or not self._sessions:
            self._sessions.append(session)
            self._session_index = len(self._sessions) - 1
        else:
            self._sessions = [session]
            self._session_index = 0

        self.settings.remember_file(str(pdf_path))
        self.settings.save()
        self._sync_document_combo()
        self._activate_session()
        self._switch_page(1)
        self._refresh_dashboard()
        return True

    def _sync_document_combo(self) -> None:
        self._document_combo.blockSignals(True)
        self._document_combo.clear()
        for session in self._sessions:
            self._document_combo.addItem(icon("document"), session.label())
        if self._sessions:
            self._document_combo.setCurrentIndex(max(0, self._session_index))
        self._document_combo.blockSignals(False)

    def _on_document_changed(self, index: int) -> None:
        if index < 0 or index >= len(self._sessions) or index == self._session_index:
            return
        self._session_index = index
        self._activate_session()

    def _activate_session(self) -> None:
        self._close_preview_document()
        self._selected_card_id = ""
        self._page_index = 0
        session = self.session
        if session is not None:
            probe, self._pending_preview_document = self._pending_preview_document, None
            if probe is not None and probe.path == session.path:
                self._preview_document = probe
            else:
                try:
                    self._preview_document = PdfDocument(session.path, password=session.password or None)
                except Exception as exc:
                    QMessageBox.critical(self, "Cannot open file", str(exc))
                    self._preview_document = None
        self._page_spin.blockSignals(True)
        self._page_spin.setRange(1, max(1, session.page_count if session else 1))
        self._page_spin.setValue(1)
        self._page_spin.blockSignals(False)
        self._page_total_label.setText(f"of {session.page_count if session else 0}")

        # Update gallery page filter
        self._gallery_page_filter.blockSignals(True)
        self._gallery_page_filter.clear()
        self._gallery_page_filter.addItem("All pages", None)
        if session:
            for p in range(session.page_count):
                self._gallery_page_filter.addItem(f"Page {p+1:02d}", p)
        self._gallery_page_filter.blockSignals(False)

        self._show_current_page()
        self._refresh_gallery()
        self._update_state()

    def _close_preview_document(self) -> None:
        if self._preview_document is not None:
            try:
                self._preview_document.close()
            finally:
                self._preview_document = None

    def select_page(self, index: int) -> None:
        session = self.session
        if session is None or self._preview_document is None:
            return
        index = max(0, min(int(index), session.page_count - 1))
        if index == self._page_index:
            self._page_spin.blockSignals(True)
            self._page_spin.setValue(index + 1)
            self._page_spin.blockSignals(False)
            return
        self._page_index = index
        self._page_spin.blockSignals(True)
        self._page_spin.setValue(index + 1)
        self._page_spin.blockSignals(False)
        self._show_current_page()
        self._refresh_gallery()
        self._update_state()

    def _show_current_page(self) -> None:
        session = self.session
        if session is None or self._preview_document is None:
            self._page_image = None
            self._view.show_page(bgr_to_qimage(None))
            self._view.set_overlays([])
            return
        try:
            dpi = self._preview_document.effective_dpi(
                self._page_index, self.settings.preview_dpi, cap_to_native=self.settings.cap_dpi_to_native
            )
            self._page_image = self._preview_document.render_page(self._page_index, dpi=dpi)
            self._page_dpi = float(dpi)
        except Exception as exc:
            self._page_image = None
            self._page_dpi = 0.0
            self._progress.set_message(f"Page render failed: {exc}")
        image = bgr_to_qimage(self._page_image)
        self._view.show_page(image)
        self._refresh_overlays()

    # ------------------------------------------------------------- card display
    def _visible_cards(self) -> List[DetectedCard]:
        session = self.session
        if session is None:
            return []
        # Return all cards; the gallery will filter based on its filter settings
        cards = list(session.cards)
        return sorted(
            cards,
            key=lambda card: (card.page_index, float(card.quad_norm[:, 1].mean()), float(card.quad_norm[:, 0].mean())),
        )

    def _refresh_overlays(self) -> None:
        session = self.session
        if session is None:
            self._view.set_overlays([])
            return
        specs: List[OverlaySpec] = []
        for position, card in enumerate(session.cards_for_page(self._page_index), start=1):
            color = self._type_color(card.doc_type)
            type_label = "Document" if card.doc_type is DocumentType.UNKNOWN else card.doc_type.display_name
            suffix = f"  {card.rotation}°" if card.rotation else ""
            if abs(card.fine_rotation) >= 0.05:
                suffix += f" +{card.fine_rotation:.1f}°"
            specs.append(
                OverlaySpec(
                    card_id=card.card_id,
                    quad_norm=card.effective_quad_norm(),
                    color=color,
                    label=f"#{position} {type_label}{suffix}",
                    excluded=not card.included,
                )
            )
        self._view.set_overlays(specs, self._selected_card_id)

    def _refresh_gallery(self) -> None:
        cards = self._visible_cards()
        if self._selected_card_id and all(card.card_id != self._selected_card_id for card in cards):
            self._selected_card_id = ""

        # Build gallery filter from UI controls
        from core.models import DocumentType

        type_filter = None
        type_data = self._gallery_type_filter.currentData()
        if type_data is not None:
            type_filter = DocumentType.from_value(type_data)

        page_filter = None
        page_data = self._gallery_page_filter.currentData()
        if page_data is not None:
            page_filter = int(page_data)

        gallery_filter = GalleryFilter(
            text=self._gallery_search.text(),
            doc_type=type_filter,
            page=page_filter,
            included_only=self._gallery_included_only.isChecked(),
            orientation_confident_only=self._gallery_confident_only.isChecked(),
        )
        self._gallery.set_filter(gallery_filter, self._selected_card_id)
        self._gallery.set_cards(self._visible_cards(), self._selected_card_id)

        self._gallery_count.setText(f"({self._gallery.count()})")
        self._sync_card_controls()

    def _sync_card_controls(self) -> None:
        card = self._selected_card() if self.session else None
        enabled = card is not None
        for widget in (self._btn_rot_left, self._btn_rot_right, self._btn_rot_180, self._btn_crop, self._btn_delete):
            widget.setEnabled(enabled)
        self._type_combo.setEnabled(enabled)
        self._include_check.setEnabled(enabled)
        self._type_combo.blockSignals(True)
        self._include_check.blockSignals(True)
        if card is not None:
            index = self._type_combo.findData(card.doc_type.value)
            if index >= 0:
                self._type_combo.setCurrentIndex(index)
            self._include_check.setChecked(card.included)
        self._type_combo.blockSignals(False)
        self._include_check.blockSignals(False)

    def _selected_card(self) -> Optional[DetectedCard]:
        session = self.session
        if session is None or not self._selected_card_id:
            return None
        return session.find_card(self._selected_card_id)

    # ------------------------------------------------------------ document types
    def _populate_type_combo(self) -> None:
        """(Re)fill the gallery type drop-down, including user defined types."""
        current = self._type_combo.currentData()
        selected = self._selected_card()
        if selected is not None:
            current = selected.doc_type.value
        self._type_combo.blockSignals(True)
        self._type_combo.clear()
        for doc_type in DocumentType.choices():
            self._type_combo.addItem(doc_type.display_name, doc_type.value)
        if current:
            index = self._type_combo.findData(current)
            if index >= 0:
                self._type_combo.setCurrentIndex(index)
        self._type_combo.blockSignals(False)

        # Also populate the gallery type filter
        self._gallery_type_filter.blockSignals(True)
        self._gallery_type_filter.clear()
        self._gallery_type_filter.addItem("All types", None)
        for doc_type in DocumentType.choices():
            self._gallery_type_filter.addItem(doc_type.display_name, doc_type.value)
        self._gallery_type_filter.blockSignals(False)

    @staticmethod
    def _type_color(doc_type: DocumentType) -> str:
        """Stable colour for a card outline (built-in and custom types)."""
        known = TYPE_COLORS.get(doc_type.token)
        if known:
            return known
        if doc_type.builtin:
            return styles.TEXT_MUTED
        # Custom types: derived from the token via CRC32 so the colour is
        # stable across app restarts and does not shift when other custom
        # types are added or removed (a list index would reshuffle them).
        return styles.CARD_COLORS[zlib.crc32(doc_type.token.encode("utf-8")) % len(styles.CARD_COLORS)]

    def _on_card_selected(self, card_id: str) -> None:
        if not card_id:
            return
        self._selected_card_id = card_id
        card = self._selected_card()
        if card is not None and card.page_index != self._page_index:
            self.select_page(card.page_index)
        self._view.select_card(card_id)
        self._gallery.select_card(card_id)
        self._refresh_overlays()
        self._sync_card_controls()

    def _update_state(self) -> None:
        session = self.session
        pages = session.page_count if session else 0
        counts = session.counts() if session else {"detected": 0, "approved": 0}
        self._status_pages.setText(f"Pages: {pages}")
        self._status_detected.setText(f"Detected: {counts['detected']}")
        self._status_approved.setText(f"Approved: {counts['approved']}")

        has_session = session is not None
        busy = self._busy
        self._btn_open.setEnabled(not busy)
        self._btn_add.setEnabled(not busy)
        self._btn_detect.setEnabled(has_session and not busy)
        self._btn_redetect.setEnabled(has_session and not busy)
        self._btn_export.setEnabled(has_session and counts["approved"] > 0 and not busy)
        self._btn_prev.setEnabled(has_session and self._page_index > 0)
        self._btn_next.setEnabled(has_session and self._page_index < pages - 1)
        self._page_spin.setEnabled(has_session and pages > 0)
        self._document_combo.setEnabled(has_session and not busy and len(self._sessions) > 1)
        self._btn_add_region.setEnabled(has_session and not busy)
        self._gallery_filter.setEnabled(has_session)
        self._sync_card_controls()

        if not has_session:
            hint = "Open a scanned PDF or image to get started"
        elif busy:
            hint = "Working…"
        else:
            page_cards = len(session.cards_for_page(self._page_index))
            if page_cards == 0 and self._page_index not in session.detected_pages:
                hint = "No detection run on this page yet"
            elif page_cards == 0:
                hint = "No documents found on this page"
            else:
                hint = f"{page_cards} document(s) on this page"
        self._page_hint.setText(hint)

    # -------------------------------------------------------------- detection
    def start_detection(self, pages: Optional[List[int]] = None) -> None:
        session = self.session
        if session is None or self._busy:
            return
        if self._preview_document is None:
            QMessageBox.warning(self, "No document", "Open a PDF or image first.")
            return
        if pages is None:
            # A page range typed in the page bar ("1,3,5-7") narrows the run.
            range_text = self._page_range_edit.text().strip()
            if range_text:
                pages = self._parse_page_range(range_text, session.page_count)
            else:
                pages = list(range(session.page_count))
        pages = [page for page in pages if 0 <= page < session.page_count]
        if not pages:
            return

        existing = [card for card in session.cards if card.page_index in pages]
        if existing:
            answer = QMessageBox.question(
                self,
                "Re-run detection",
                f"{len(existing)} detected document(s) on the selected page(s) will be replaced. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        for page_index in pages:
            session.clear_page(page_index)
            session.detected_pages.discard(page_index)

        self._detection_errors: List[str] = []
        self._busy = True
        self._update_state()
        self._refresh_overlays()
        self._refresh_gallery()
        self._progress.start("Starting detection…")

        worker = DetectionWorker(session.path, pages, self.settings, password=session.password, parent=self)
        worker.progress.connect(self._on_detection_progress)
        worker.card_found.connect(self._on_card_found)
        worker.page_finished.connect(self._on_page_finished)
        worker.failed.connect(self._on_detection_failed)
        worker.finished_ok.connect(self._on_detection_finished)
        worker.cancelled.connect(self._on_detection_cancelled)
        # the reference must be cleared after the thread *really* returned
        # (finished_ok/cancelled arrive while the thread can still be running),
        # otherwise closeEvent would wait on nothing while the thread is alive
        worker.finished.connect(self._on_detection_thread_finished)
        worker.finished.connect(worker.deleteLater)
        self._detection_worker = worker
        worker.start()

    def _on_detection_progress(self, current: int, total: int, message: str) -> None:
        self._progress.set_progress(current, total, message)

    def _on_card_found(self, card: DetectedCard) -> None:
        session = self.session
        if session is not None:
            session.add_card(card)

    def _parse_page_range(self, text: str, page_count: int) -> List[int]:
        """Parse ``"1,3,5-7"`` into zero-based page indices."""
        result: List[int] = []
        for chunk in text.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "-" in chunk:
                start, _, end = chunk.partition("-")
                try:
                    first = max(1, int(start))
                    last = min(page_count, int(end))
                except ValueError:
                    continue
                result.extend(range(first - 1, last))
            else:
                try:
                    page = int(chunk)
                except ValueError:
                    continue
                if 1 <= page <= page_count:
                    result.append(page - 1)
        return sorted(set(result))

    def _on_page_finished(self, page_index: int, count: int) -> None:
        session = self.session
        if session is not None:
            session.detected_pages.add(page_index)
        if page_index == self._page_index:
            self._refresh_overlays()
            self._refresh_gallery()
        self._update_state()

    def _on_detection_failed(self, message: str) -> None:
        self._detection_errors.append(message)
        self._progress.set_message(message.splitlines()[0][:110])

    def _on_detection_thread_finished(self) -> None:
        """Thread cleanup of the detection worker (fires after ``run()`` returned)."""
        self._detection_worker = None

    def _on_detection_finished(self, total: int) -> None:
        worker = self._detection_worker
        ocr_note = getattr(worker, "ocr_report", "") if worker is not None else ""
        self._busy = False
        message = f"Detected {total} document(s)" if total else "No documents detected"
        if ocr_note:
            message += f"  ·  OCR: {ocr_note}"
        self._progress.finish(message)
        self._refresh_overlays()
        self._refresh_gallery()
        self._update_state()
        self._refresh_dashboard()
        if self._detection_errors:
            QMessageBox.warning(self, "Detection finished with warnings", "\n".join(self._detection_errors[:8]))
        if total == 0 and not self._detection_errors:
            # A run that produced errors already reported them (e.g. the file
            # could not be opened) - the "nothing detected, do this" advice
            # would be misleading on top of it.
            QMessageBox.information(
                self,
                "No documents found",
                "No document cards were detected on the selected page(s).\n\n"
                "Tips:\n"
                "• Increase the detection sensitivity in Settings.\n"
                "• Use “Add Region” to draw a card manually.\n"
                "• Make sure the scan is not blank or extremely low resolution.",
            )

    def _on_detection_cancelled(self, total: int) -> None:
        self._busy = False
        self._progress.finish(f"Detection cancelled ({total} document(s) kept)")
        self._refresh_overlays()
        self._refresh_gallery()
        self._update_state()

    # ----------------------------------------------------------------- export
    def export_all(self) -> None:
        if self._busy:
            return
        jobs = [
            ExportJob(session.path, session.included_cards(), password=session.password)
            for session in self._sessions
            if session.included_cards()
        ]
        if not jobs:
            QMessageBox.information(
                self, "Nothing to export", "Detect and approve at least one document before exporting."
            )
            return
        self._start_export(jobs, self.settings.export_options())

    def export_card(self, card_id: str) -> None:
        if self._busy:
            return
        session = self.session
        if session is None:
            return
        card = session.find_card(card_id)
        if card is None:
            return
        if not card.included:
            answer = QMessageBox.question(
                self,
                "Document excluded",
                "This document is excluded from export. Export it anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        folder = QFileDialog.getExistingDirectory(
            self, "Select the folder for this document", self.settings.output_dir or str(Path.home())
        )
        if not folder:
            return
        options = self.settings.export_options()
        options.output_dir = folder
        options.create_subfolder = False
        self._start_export([ExportJob(session.path, [card], password=session.password)], options)

    def _start_export(self, jobs: List[ExportJob], options: ExportOptions) -> None:
        self._active_export_jobs = jobs
        self._export_errors = []
        self._busy = True
        self._update_state()
        self._progress.start("Exporting documents…")

        worker = ExportWorker(jobs, options, self)
        worker.progress.connect(self._on_export_progress)
        worker.failed.connect(self._on_export_failed)
        worker.finished_ok.connect(self._on_export_finished)
        worker.cancelled.connect(self._on_export_cancelled)
        worker.finished.connect(self._on_export_thread_finished)
        worker.finished.connect(worker.deleteLater)
        self._export_worker = worker
        worker.start()

    def _on_export_thread_finished(self) -> None:
        """Thread cleanup of the export worker (fires after ``run()`` returned)."""
        self._export_worker = None

    def _on_export_progress(self, current: int, total: int, message: str) -> None:
        self._progress.set_progress(current, total, message)

    def _on_export_failed(self, message: str) -> None:
        self._export_errors.append(message)

    def _on_export_finished(self, summary: ExportSummary) -> None:
        self._finish_export(summary, cancelled=False)

    def _on_export_cancelled(self, summary: ExportSummary) -> None:
        self._finish_export(summary, cancelled=True)

    def _finish_export(self, summary: ExportSummary, cancelled: bool) -> None:
        self._busy = False
        self._update_state()
        exported = summary.success_count
        self._progress.finish(
            f"{'Export cancelled' if cancelled else 'Export finished'} · {exported} file(s) written"
        )

        jobs = getattr(self, "_active_export_jobs", [])
        entry = HistoryEntry(
            pdf_files=[str(job.pdf_path) for job in jobs],
            output_dir=str(summary.output_dir),
            document_count=exported,
            page_count=len(
                # Pages that actually contributed exported documents, not the
                # number of documents (which document_count already holds).
                {card.page_index for job in jobs for card in job.included_cards}
            ),
            files=[str(item.path) for item in summary.files if item.ok],
            errors=list(summary.errors) + list(getattr(self, "_export_errors", [])),
        )
        if not self.history.add(entry):
            self._progress.set_message("The history file could not be written")
        self._refresh_history_table()
        self._refresh_dashboard()

        if exported and self.settings.open_output_after_export:
            self._open_folder(summary.output_dir)

        message = f"{exported} document image(s) exported to:\n{summary.output_dir}"
        if summary.failure_count:
            message += f"\n\n{summary.failure_count} file(s) failed."
            details = "\n".join(summary.errors[:6]) or "\n".join(getattr(self, "_export_errors", [])[:6])
            if details:
                message += f"\n\n{details}"
            QMessageBox.warning(self, "Export finished with errors", message)
        elif cancelled:
            QMessageBox.information(self, "Export cancelled", message)
        else:
            QMessageBox.information(self, "Export finished", message)

    # -------------------------------------------------------- card operations
    def rotate_card(self, card_id: str, degrees: int) -> None:
        session = self.session
        card = session.find_card(card_id) if session else None
        if card is None:
            return
        card.rotation = (card.rotation + int(degrees)) % 360
        card.orientation_confident = True
        if card.thumbnail is not None:
            try:
                card.thumbnail = rotate_image(card.thumbnail, int(degrees))
            except Exception:
                pass
        self._selected_card_id = card_id
        self._gallery.update_card(card)
        self._refresh_overlays()
        self._sync_card_controls()
        self._update_state()

    def _rotate_selected(self, degrees: int) -> None:
        if self._selected_card_id:
            self.rotate_card(self._selected_card_id, degrees)

    def set_card_type(self, card_id: str, doc_type) -> None:
        session = self.session
        card = session.find_card(card_id) if session else None
        if card is None:
            return
        card.set_type(doc_type, 1.0)
        self._selected_card_id = card_id
        self._gallery.update_card(card)
        self._refresh_overlays()
        self._sync_card_controls()

    def toggle_card_included(self, card_id: str) -> None:
        session = self.session
        card = session.find_card(card_id) if session else None
        if card is None:
            return
        card.included = not card.included
        self._selected_card_id = card_id
        self._gallery.update_card(card)
        self._refresh_overlays()
        self._sync_card_controls()
        self._update_state()

    def remove_card(self, card_id: str) -> None:
        session = self.session
        if session is None:
            return
        if session.remove_card(card_id):
            if self._selected_card_id == card_id:
                self._selected_card_id = ""
            self._refresh_overlays()
            self._refresh_gallery()
            self._update_state()

    def _delete_selected_card(self) -> None:
        if self._selected_card_id:
            self.remove_card(self._selected_card_id)

    def edit_crop(self, card_id: str) -> None:
        session = self.session
        card = session.find_card(card_id) if session else None
        if card is None or session is None:
            return
        dialog = CropEditorDialog(session.path, card, parent=self, password=session.password)
        if dialog.exec() != CropEditorDialog.DialogCode.Accepted:
            return
        edited = dialog.card
        card.quad_norm = edited.quad_norm
        card.page_size = edited.page_size or card.page_size
        card.rotation = edited.rotation
        card.fine_rotation = edited.fine_rotation
        card.orientation_confident = edited.orientation_confident
        card.doc_type = edited.doc_type
        card.type_confidence = edited.type_confidence
        card.identifier = edited.identifier
        card.included = edited.included
        self._selected_card_id = card.card_id
        self._regenerate_thumbnail(card, immediate=True)
        self._refresh_overlays()
        self._gallery.update_card(card)
        self._sync_card_controls()
        self._update_state()

    def show_ocr_text(self, card_id: str) -> None:
        """Show the text the OCR engine found for one card (gallery context menu)."""
        session = self.session
        card = session.find_card(card_id) if session else None
        if card is None:
            return
        text = (card.ocr_text or "").strip()
        if not text:
            QMessageBox.information(
                self,
                "OCR text",
                "No text was recognised for this document.\n\n"
                "Enable “Enable local OCR” in the settings and run the detection again, "
                "or press Check next to the OCR engine to verify the engine works.",
            )
            return
        box = QMessageBox(self)
        box.setWindowTitle(f"OCR text · {card.doc_type.display_name}")
        box.setIcon(QMessageBox.Icon.Information)
        box.setText(
            "Recognised text (used for the classification keywords and the identifier):"
        )
        box.setDetailedText(text)
        box.setInformativeText(text[:600] + ("…" if len(text) > 600 else ""))
        copy_button = box.addButton("Copy text", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Close)
        box.exec()
        if box.clickedButton() is copy_button:
            from PySide6.QtWidgets import QApplication

            QApplication.clipboard().setText(text)
            self._progress.set_message("OCR text copied to the clipboard")

    def _regenerate_thumbnail(self, card: DetectedCard, immediate: bool = False) -> None:
        """Rebuild the thumbnail of *card* from the current page render.

        A rebuild warps the full-resolution page.  Continuous interactions
        (arrow-key nudges, mouse drags in the page view) call this on every
        step, so those rebuilds are coalesced to run ~30 ms after the last
        change; discrete actions pass ``immediate=True`` to get the new
        thumbnail right away.
        """
        if self._page_image is None or card.page_index != self._page_index:
            return
        if immediate:
            self._rebuild_thumbnail_now(card)
            self._gallery.update_card(card)
            return
        self._thumb_pending = card
        if not self._thumb_timer.isActive():
            self._thumb_timer.start()

    def _flush_thumbnails(self) -> None:
        card = self._thumb_pending
        self._thumb_pending = None
        if card is None or self._page_image is None or card.page_index != self._page_index:
            return
        self._rebuild_thumbnail_now(card)
        # repaint the gallery tile now that the thumbnail really changed
        self._gallery.update_card(card)

    def _rebuild_thumbnail_now(self, card: DetectedCard) -> None:
        try:
            height, width = self._page_image.shape[:2]
            quad = card.effective_quad_pixels(width, height)
            crop = warp_quad(self._page_image, quad)
            if card.rotation:
                crop = rotate_image(crop, card.rotation)
            card.thumbnail = make_thumbnail(crop, int(self.settings.thumbnail_size))
        except Exception:
            pass

    # --------------------------------------------------- add region (manual crop)
    def _normalise_region_preset(self) -> None:
        """Keep the preset token and the boolean flag consistent (also for old files)."""
        settings = self.settings
        token = str(getattr(settings, "manual_region_preset", "") or "")
        if token not in REGION_PRESET_TOKENS:
            # settings written before the token existed only had the boolean flag
            token = "custom" if settings.manual_region_fixed_size else "free"
        if token == "free" and settings.manual_region_fixed_size:
            token = "custom"
        settings.manual_region_preset = token
        settings.manual_region_fixed_size = token != "free"

    def _region_preset_index(self) -> int:
        """Index in REGION_PRESETS for the selected kind (custom stays custom)."""
        token = str(self.settings.manual_region_preset or "free")
        for index, (_label, preset_token, _size) in enumerate(REGION_PRESETS):
            if preset_token == token:
                return index
        return 0

    def _sync_region_controls(self) -> None:
        """Keep the toolbar combo and the settings widgets in sync."""
        settings = self.settings
        index = self._region_preset_index()
        if hasattr(self, "_region_preset_combo"):
            self._region_preset_combo.blockSignals(True)
            self._region_preset_combo.setCurrentIndex(index)
            self._region_preset_combo.blockSignals(False)
        if hasattr(self, "_set_region_preset"):
            self._set_region_preset.blockSignals(True)
            self._set_region_preset.setCurrentIndex(index)
            self._set_region_preset.blockSignals(False)
            self._set_region_fixed.blockSignals(True)
            self._set_region_fixed.setChecked(bool(settings.manual_region_fixed_size))
            self._set_region_fixed.blockSignals(False)
            self._set_region_width.blockSignals(True)
            self._set_region_width.setValue(float(settings.manual_region_width_mm))
            self._set_region_width.blockSignals(False)
            self._set_region_height.blockSignals(True)
            self._set_region_height.setValue(float(settings.manual_region_height_mm))
            self._set_region_height.blockSignals(False)
            enabled = bool(settings.manual_region_fixed_size)
            self._set_region_preset.setEnabled(True)
            self._set_region_width.setEnabled(enabled)
            self._set_region_height.setEnabled(enabled)
        if hasattr(self, "_region_size_button"):
            if settings.manual_region_fixed_size:
                self._region_size_button.setText(
                    f"{settings.manual_region_width_mm:.1f} × "
                    f"{settings.manual_region_height_mm:.1f} mm"
                )
                self._region_size_button.setToolTip(
                    "Change the size used by 'Add Region' (click to edit)"
                )
            else:
                self._region_size_button.setText("Size…")
                self._region_size_button.setToolTip(
                    "Give 'Add Region' a fixed physical size (e.g. ID card 85.6 × 54 mm)"
                )
        if hasattr(self, "_btn_add_region"):
            if settings.manual_region_fixed_size:
                self._btn_add_region.setToolTip(
                    "Draw a region; it is snapped to "
                    f"{settings.manual_region_width_mm:.1f} × {settings.manual_region_height_mm:.1f} mm "
                    "(Esc to cancel)"
                )
            else:
                self._btn_add_region.setToolTip(
                    "Draw a rectangle to add a document the detector missed (Esc to cancel)"
                )

    def _on_region_preset_changed(self, index: int) -> None:
        if index < 0 or index >= len(REGION_PRESETS):
            return
        _label, token, size = REGION_PRESETS[index]
        settings = self.settings
        if token == "custom":
            # ask for the dimensions instead of silently reusing a preset value.
            # The dialog is deferred so that the combo's "activated" signal (which
            # follows this one for the same user action) cannot open it twice.
            self._region_pending_dialog = index
            QTimer.singleShot(0, self._run_pending_region_dialog)
            return
        self._region_pending_dialog = -1
        if token == "free":
            settings.manual_region_preset = "free"
            settings.manual_region_fixed_size = False
        else:
            settings.manual_region_preset = token
            settings.manual_region_fixed_size = True
            settings.manual_region_width_mm, settings.manual_region_height_mm = size
        settings.save()
        self._sync_region_controls()

    def _run_pending_region_dialog(self) -> None:
        index = self._region_pending_dialog
        self._region_pending_dialog = -1
        if index < 0:
            return
        if not self._open_region_size_dialog():
            self._sync_region_controls()  # put the combo back where it was

    def _on_region_preset_activated(self, index: int) -> None:
        """The user picked an entry; re-picking the current one must still react."""
        if index < 0 or index >= len(REGION_PRESETS):
            return
        if self._region_pending_dialog == index:
            return  # this was a real change - the deferred call opens the dialog
        if REGION_PRESETS[index][1] != "custom":
            return  # plain presets need no dialog
        # "Custom…" was already the current entry: ask for the size again
        self._open_region_size_dialog()

    def _open_region_size_dialog(self) -> bool:
        """Ask for a custom width/height. Returns ``True`` when a size was applied."""
        settings = self.settings
        dialog = RegionSizeDialog(
            float(settings.manual_region_width_mm), float(settings.manual_region_height_mm), self
        )
        dialog.raise_()
        dialog.activateWindow()
        if not dialog.exec():
            return False
        settings.manual_region_width_mm, settings.manual_region_height_mm = dialog.size_mm()
        settings.manual_region_preset = "custom"
        settings.manual_region_fixed_size = True
        settings.save()
        self._sync_region_controls()
        return True

    def _on_region_fixed_toggled(self, checked: bool) -> None:
        settings = self.settings
        if bool(settings.manual_region_fixed_size) == bool(checked):
            return
        settings.manual_region_fixed_size = bool(checked)
        settings.manual_region_preset = "custom" if checked else "free"
        settings.save()
        self._sync_region_controls()

    def _on_region_dimensions_changed(self, _value: float) -> None:
        if not hasattr(self, "_set_region_width"):
            return
        width = float(self._set_region_width.value())
        height = float(self._set_region_height.value())
        settings = self.settings
        if abs(width - settings.manual_region_width_mm) < 1e-6 and abs(
            height - settings.manual_region_height_mm
        ) < 1e-6:
            return
        settings.manual_region_width_mm = width
        settings.manual_region_height_mm = height
        if settings.manual_region_preset not in ("free",):
            # typed numbers no longer match the preset that was selected
            settings.manual_region_preset = "custom"
        settings.save()
        self._sync_region_controls()

    def _fixed_region_size_norm(self):
        """Size of the fixed region in normalised page units (or ``None``)."""
        settings = self.settings
        if not settings.manual_region_fixed_size or self._page_image is None:
            return None
        height_px, width_px = self._page_image.shape[:2]
        if not (height_px and width_px):
            return None
        dpi = float(self._page_dpi or 0.0) or 300.0
        px_per_mm = dpi / 25.4
        width_norm = (float(settings.manual_region_width_mm) * px_per_mm) / float(width_px)
        height_norm = (float(settings.manual_region_height_mm) * px_per_mm) / float(height_px)
        if not (0.0 < width_norm <= 1.0 and 0.0 < height_norm <= 1.0):
            return None
        return width_norm, height_norm

    # ------------------------------------------------------ manual corrections
    def _toggle_add_mode(self, checked: bool) -> None:
        self._view.set_mode(PdfPageView.MODE_ADD if checked else PdfPageView.MODE_SELECT)
        if checked:
            self._progress.set_message("Draw a rectangle around the document you want to add")

    def _leave_add_mode(self) -> None:
        if self._btn_add_region.isChecked():
            self._btn_add_region.setChecked(False)
        self._view.set_mode(PdfPageView.MODE_SELECT)

    def _on_manual_rect(self, rect) -> None:
        self._leave_add_mode()
        session = self.session
        if session is None or rect is None:
            return

        centre_x = (rect.left() + rect.right()) / 2.0
        centre_y = (rect.top() + rect.bottom()) / 2.0
        fixed = self._fixed_region_size_norm()
        if fixed is not None:
            # snap to the configured physical size, keeping the drawn centre
            width_norm, height_norm = fixed
            x0 = min(max(centre_x - width_norm / 2.0, 0.0), 1.0 - width_norm)
            y0 = min(max(centre_y - height_norm / 2.0, 0.0), 1.0 - height_norm)
            quad = np.array(
                [
                    [x0, y0],
                    [x0 + width_norm, y0],
                    [x0 + width_norm, y0 + height_norm],
                    [x0, y0 + height_norm],
                ],
                dtype=np.float32,
            )
        else:
            quad = np.array(
                [
                    [rect.left(), rect.top()],
                    [rect.right(), rect.top()],
                    [rect.right(), rect.bottom()],
                    [rect.left(), rect.bottom()],
                ],
                dtype=np.float32,
            )
        page_size = (self._page_image.shape[1], self._page_image.shape[0]) if self._page_image is not None else (0, 0)
        card = DetectedCard(
            page_index=self._page_index,
            quad_norm=quad,
            confidence=1.0,
            source="manual",
            page_size=page_size,
        )
        card.set_type(DocumentType.UNKNOWN, 0.0)
        self._regenerate_thumbnail(card, immediate=True)
        session.add_card(card)
        self._selected_card_id = card.card_id
        self._refresh_overlays()
        self._refresh_gallery()
        self._gallery.select_card(card.card_id)
        self._sync_card_controls()
        self._update_state()
        if fixed is not None:
            self._progress.set_message(
                "Region added at "
                f"{self.settings.manual_region_width_mm:.1f} × "
                f"{self.settings.manual_region_height_mm:.1f} mm"
            )

    def _on_quad_edited(self, card_id: str, quad_norm) -> None:
        session = self.session
        card = session.find_card(card_id) if session else None
        if card is None:
            return
        # the view always reports the *displayed* quad, so a fine rotation is
        # baked into the geometry at this point
        card.quad_norm = np.asarray(quad_norm, dtype=np.float32).reshape(4, 2)
        card.fine_rotation = 0.0
        self._selected_card_id = card_id
        self._regenerate_thumbnail(card)
        self._refresh_overlays()
        self._gallery.update_card(card)
        self._sync_card_controls()

    def _on_type_combo_changed(self, index: int) -> None:
        card = self._selected_card()
        if card is None or index < 0:
            return
        raw = self._type_combo.itemData(index)
        if raw is None:
            return
        # Qt hands the DocumentType (a str enum) back as a plain string.
        self.set_card_type(card.card_id, DocumentType.from_value(raw))

    def _on_include_check_toggled(self, checked: bool) -> None:
        card = self._selected_card()
        if card is None or card.included == bool(checked):
            return
        self.toggle_card_included(card.card_id)

    # --------------------------------------------------------------- utilities
    def _cancel_current_task(self) -> None:
        if self._detection_worker is not None and self._detection_worker.isRunning():
            self._detection_worker.request_cancel()
            self._progress.set_message("Cancelling detection…")
        elif self._export_worker is not None and self._export_worker.isRunning():
            self._export_worker.request_cancel()
            self._progress.set_message("Cancelling export…")

    def _open_folder(self, folder) -> None:
        path = Path(folder)
        if not path.exists():
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception:
            pass

    # --------------------------------------------------------- drag and drop
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802 - Qt naming
        if self._event_has_pdf(event):
            event.acceptProposedAction()

    def dragMoveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._event_has_pdf(event):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802 - Qt naming
        paths = self._dropped_pdf_paths(event)
        if not paths:
            return
        event.acceptProposedAction()
        first = not bool(self._sessions)
        for index, path in enumerate(paths):
            # Open the first file (empty session: no notes yet), the rest are
            # added to the session list without touching the first one.
            self.load_pdf(path, add=not (index == 0 and first))

    @staticmethod
    def _dropped_pdf_paths(event) -> List[str]:
        mime = event.mimeData()
        if not mime.hasUrls():
            return []
        paths = []
        for url in mime.urls():
            local = url.toLocalFile()
            if local and is_supported_file(local) and Path(local).is_file():
                paths.append(local)
        return paths

    def _event_has_pdf(self, event) -> bool:
        return bool(self._dropped_pdf_paths(event))

    # ----------------------------------------------------------------- closing
    def _wait_for_workers(self) -> None:
        """Wait for running workers to finish after a cancel was requested.

        Destroying the window while a QThread is still running crashes Qt
        ("QThread: Destroyed while thread is still running").  Cancel is
        honoured at page/card granularity inside the workers and the OCR read
        of a *single* card can take several seconds, so the wait must not be a
        short timeout.  The event loop is pumped in between so the UI stays
        responsive; after a generous deadline ``terminate()`` is used as the
        last resort - engine crashes would be worse than a torn export.
        """
        for worker in (self._detection_worker, self._export_worker):
            if worker is None or not worker.isRunning():
                continue
            deadline = time.monotonic() + 60.0
            while worker.isRunning() and time.monotonic() < deadline:
                worker.wait(100)
                QApplication.processEvents()
            if worker.isRunning():
                worker.terminate()
                if not worker.wait(2000):
                    worker.terminate()
                    worker.wait(3000)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._busy:
            answer = QMessageBox.question(
                self,
                "Still working",
                "A task is still running. Cancel it and exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._cancel_current_task()
            self._wait_for_workers()
        if self._pending_preview_document is not None:
            self._pending_preview_document.close()
            self._pending_preview_document = None
        self._close_preview_document()
        self.settings.save()
        super().closeEvent(event)















