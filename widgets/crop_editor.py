"""Dialog that lets the user fine tune a crop, rotation and document type."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app import styles
from app.icons import apply_icon_size, icon
from core.models import DetectedCard, DocumentType
from core.pdf_loader import PdfDocument
from core.perspective_corrector import make_thumbnail, rotate_image, warp_quad
from widgets.pdf_viewer import OverlaySpec, PdfPageView
from widgets.qt_image import FitImageLabel, bgr_to_pixmap, bgr_to_qimage


class CropEditorDialog(QDialog):
    """Modal editor for a single detected card."""

    PREVIEW_DPI = 200

    def __init__(
        self,
        pdf_path: Path,
        card: DetectedCard,
        parent: Optional[QWidget] = None,
        password: str = "",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Adjust document · page {card.page_number}")
        self.resize(1240, 780)
        self.setMinimumSize(900, 620)
        self._pdf_path = Path(pdf_path)
        self._password = password or ""  # for encrypted PDFs (in memory only)
        self._draft = card.copy_card()
        self._reset_source = card.copy_card()
        self._page_image: Optional[np.ndarray] = None
        self._load_error = ""
        self._rotate_base = 0.0
        # Rendering the cropped preview is expensive (full-page perspective
        # warp + a 1000 px thumbnail).  During a live rotation drag or while
        # nudging with the arrow keys it fires per mouse-move, so rebuilds are
        # coalesced to at most ~30 per second with short-lived updates.
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(30)
        self._preview_timer.timeout.connect(self._flush_preview)
        self._preview_pending = False
        self._load_page()
        self._build_ui()
        apply_icon_size(self)
        self._refresh()

    # -------------------------------------------------------------------- setup
    def _load_page(self) -> None:
        document = None
        try:
            document = PdfDocument(self._pdf_path, password=self._password or None)
            self._page_image = document.render_page(self._draft.page_index, dpi=self.PREVIEW_DPI)
        except Exception as exc:  # pragma: no cover - surfaced in the UI
            self._page_image = None
            self._load_error = str(exc)
        finally:
            if document is not None:
                document.close()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        header = QLabel(
            "Drag the four handles, the round grip (rotate) or inside the crop "
            "(move). Arrow keys nudge the crop by 1 px, Shift+arrows resize it. "
            "Changes are applied to the document when you press Apply."
        )
        header.setObjectName("Muted")
        root.addWidget(header)
        if self._load_error:
            error = QLabel(f"Could not render the page preview: {self._load_error}")
            error.setStyleSheet(f"color: {styles.DANGER};")
            error.setWordWrap(True)
            root.addWidget(error)

        body = QHBoxLayout()
        body.setSpacing(14)
        root.addLayout(body, 1)

        self._view = PdfPageView(self)
        self._view.set_rotation_handle(True)
        if self._page_image is not None:
            self._view.show_page(bgr_to_qimage(self._page_image))
        self._view.quadEdited.connect(self._on_quad_edited)
        self._view.rotateStarted.connect(self._on_rotate_started)
        self._view.rotateDragged.connect(self._on_rotate_dragged)
        body.addWidget(self._view, 3)

        side = QFrame()
        side.setObjectName("Card")
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(14, 14, 14, 14)
        side_layout.setSpacing(10)
        body.addWidget(side, 2)

        title = QLabel("Preview")
        title.setObjectName("SectionTitle")
        side_layout.addWidget(title)

        self._preview = FitImageLabel()
        self._preview.setMinimumSize(320, 220)
        self._preview.setStyleSheet(
            f"background: {styles.SURFACE_ALT}; border: 1px solid {styles.BORDER}; border-radius: 8px;"
        )
        side_layout.addWidget(self._preview, 1)

        self._preview_info = QLabel("")
        self._preview_info.setObjectName("Faint")
        self._preview_info.setWordWrap(True)
        side_layout.addWidget(self._preview_info)

        rotate_row = QHBoxLayout()
        for label, degrees, icon_name in (
            ("90° left", 270, "rotate_left"),
            ("90° right", 90, "rotate_right"),
            ("180°", 180, "flip"),
        ):
            button = QPushButton(icon(icon_name), label)
            button.clicked.connect(lambda _checked=False, d=degrees: self._rotate(d))
            rotate_row.addWidget(button)
        side_layout.addLayout(rotate_row)

        fine_row = QHBoxLayout()
        fine_row.addWidget(QLabel("Fine rotation"))
        self._fine_spin = QDoubleSpinBox()
        self._fine_spin.setRange(-45.0, 45.0)
        self._fine_spin.setDecimals(1)
        self._fine_spin.setSingleStep(0.5)
        self._fine_spin.setSuffix("°")
        self._fine_spin.setToolTip(
            "Deskew the document: positive values rotate clockwise.\n"
            "Applies to the crop itself, so the export matches the preview."
        )
        self._fine_spin.valueChanged.connect(self._on_fine_rotation_changed)
        fine_reset = QToolButton()
        fine_reset.setText("0")
        fine_reset.setToolTip("Reset the fine rotation")
        fine_reset.clicked.connect(lambda: self._fine_spin.setValue(0.0))
        fine_row.addWidget(self._fine_spin, 1)
        fine_row.addWidget(fine_reset)
        side_layout.addLayout(fine_row)

        step_row = QHBoxLayout()
        step_row.setSpacing(4)
        step_row.addWidget(QLabel("Nudge"))
        self._nudge_buttons = {}
        for key, text, tooltip, dx, dy in (
            ("left", "\u25c0", "Move 1 px left\n(arrow key; Ctrl = 10 px, Alt = 20 px)", -1.0, 0.0),
            ("up", "\u25b2", "Move 1 px up\n(arrow key; Ctrl = 10 px, Alt = 20 px)", 0.0, -1.0),
            ("down", "\u25bc", "Move 1 px down\n(arrow key; Ctrl = 10 px, Alt = 20 px)", 0.0, 1.0),
            ("right", "\u25b6", "Move 1 px right\n(arrow key; Ctrl = 10 px, Alt = 20 px)", 1.0, 0.0),
        ):
            button = QToolButton()
            button.setText(text)
            button.setToolTip(tooltip)
            button.clicked.connect(
                lambda _checked=False, x=dx, y=dy: self._view.nudge_selected(x, y)
            )
            step_row.addWidget(button)
            self._nudge_buttons[key] = button
        step_row.addStretch(1)

        step_row.addWidget(QLabel("Size"))
        self._size_buttons = {}
        for key, text, tooltip, dw, dh in (
            ("narrower", "\u2212 W", "Narrower (Shift + \u2190)", -1.0, 0.0),
            ("wider", "+ W", "Wider (Shift + \u2192)", 1.0, 0.0),
            ("shorter", "\u2212 H", "Shorter (Shift + \u2193)", 0.0, -1.0),
            ("taller", "+ H", "Taller (Shift + \u2191)", 0.0, 1.0),
        ):
            button = QToolButton()
            button.setText(text)
            button.setToolTip(tooltip)
            button.clicked.connect(
                lambda _checked=False, w=dw, h=dh: self._view.resize_selected(w, h)
            )
            step_row.addWidget(button)
            self._size_buttons[key] = button
        side_layout.addLayout(step_row)

        margin_row = QHBoxLayout()
        margin_row.addWidget(QLabel("Margin"))
        self._margin_spin = QDoubleSpinBox()
        self._margin_spin.setRange(0.1, 25.0)
        self._margin_spin.setDecimals(1)
        self._margin_spin.setSingleStep(0.5)
        self._margin_spin.setValue(2.0)
        self._margin_spin.setSuffix(" mm")
        self._margin_spin.setToolTip(
            "Grow the crop on all four sides, e.g. to keep the card's shadow or "
            "its rounded corners inside the exported image"
        )
        margin_row.addWidget(self._margin_spin, 1)
        self._margin_button = QPushButton("Add to all sides")
        self._margin_button.setToolTip("Expand the crop by the margin above on every side")
        self._margin_button.clicked.connect(self._expand_margin)
        margin_row.addWidget(self._margin_button)
        side_layout.addLayout(margin_row)

        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Type"))
        self._type_combo = QComboBox()
        for doc_type in DocumentType.choices():
            self._type_combo.addItem(doc_type.display_name, doc_type.value)
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)
        type_row.addWidget(self._type_combo, 1)
        side_layout.addLayout(type_row)

        id_row = QHBoxLayout()
        id_row.addWidget(QLabel("Identifier"))
        self._identifier_edit = QLineEdit(self._draft.identifier)
        self._identifier_edit.setPlaceholderText("Optional, used in the filename")
        self._identifier_edit.textChanged.connect(self._on_identifier_changed)
        id_row.addWidget(self._identifier_edit, 1)
        side_layout.addLayout(id_row)

        self._ocr_text_view = QPlainTextEdit()
        self._ocr_text_view.setObjectName("OcrText")
        self._ocr_text_view.setReadOnly(True)
        self._ocr_text_view.setMaximumHeight(96)
        self._ocr_text_view.setPlaceholderText("No OCR text (OCR disabled or nothing recognised)")
        self._ocr_text_view.setToolTip(
            "Text the OCR engine found on this card.\n"
            "Use it to tune the keywords in Settings → Document types."
        )
        ocr_row = QHBoxLayout()
        ocr_row.addWidget(QLabel("OCR text"))
        self._copy_ocr_button = QToolButton()
        self._copy_ocr_button.setText("Copy")
        self._copy_ocr_button.setToolTip("Copy the recognised text to the clipboard")
        self._copy_ocr_button.clicked.connect(self._copy_ocr_text)
        ocr_row.addStretch(1)
        ocr_row.addWidget(self._copy_ocr_button)
        side_layout.addLayout(ocr_row)
        side_layout.addWidget(self._ocr_text_view)

        self._include_check = QCheckBox("Include in export")
        self._include_check.setChecked(self._draft.included)
        self._include_check.toggled.connect(self._on_include_toggled)
        side_layout.addWidget(self._include_check)

        hint = QLabel(
            "Keyboard: arrows move the crop by 1 px, Shift+arrows resize it "
            "(Ctrl = 10 px, Alt = 20 px). Use Margin to keep the card's shadow. "
            "The preview shows exactly what will be exported."
        )
        hint.setObjectName("Faint")
        hint.setWordWrap(True)
        side_layout.addWidget(hint)

        buttons = QDialogButtonBox()
        reset_button = buttons.addButton("Reset", QDialogButtonBox.ButtonRole.ResetRole)
        apply_button = buttons.addButton("Apply", QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton("Cancel", QDialogButtonBox.ButtonRole.RejectRole)
        apply_button.setProperty("variant", "primary")
        root.addWidget(buttons)
        reset_button.clicked.connect(self._reset)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        # arrow keys must reach the page view right away
        self._view.setFocus()

    # ------------------------------------------------------------------- state
    @property
    def card(self) -> DetectedCard:
        return self._draft

    def _overlays(self):
        return [
            OverlaySpec(
                card_id=self._draft.card_id,
                quad_norm=self._draft.effective_quad_norm(),
                color=styles.PRIMARY,
                label="Drag corners · grip rotates",
            )
        ]

    def _refresh(self) -> None:
        self._view.set_overlays(self._overlays(), self._draft.card_id)
        self._type_combo.blockSignals(True)
        index = self._type_combo.findData(self._draft.doc_type.value)
        self._type_combo.setCurrentIndex(max(0, index))
        self._type_combo.blockSignals(False)
        self._fine_spin.blockSignals(True)
        self._fine_spin.setValue(float(self._draft.fine_rotation))
        self._fine_spin.blockSignals(False)
        self._ocr_text_view.setPlainText(self._draft.ocr_text or "")
        self._copy_ocr_button.setEnabled(bool((self._draft.ocr_text or "").strip()))
        self._identifier_edit.blockSignals(True)
        self._identifier_edit.setText(self._draft.identifier)
        self._identifier_edit.blockSignals(False)
        self._include_check.blockSignals(True)
        self._include_check.setChecked(self._draft.included)
        self._include_check.blockSignals(False)
        self._update_preview()

    def _update_preview(self) -> None:
        if self._page_image is None:
            self._preview.clear_source()
            return
        height, width = self._page_image.shape[:2]
        quad = self._draft.effective_quad_pixels(width, height)
        try:
            cropped = warp_quad(self._page_image, quad)
        except Exception:
            return
        if self._draft.rotation:
            cropped = rotate_image(cropped, self._draft.rotation)
        # A large source keeps the scaled preview crisp; FitImageLabel fits it
        # into the available space without ever cutting it off.
        preview = make_thumbnail(cropped, 1000)
        self._preview.set_source(bgr_to_pixmap(preview))
        source_height, source_width = preview.shape[:2]
        export_scale = max(1.0, 300.0 / float(self.PREVIEW_DPI))
        rotation_text = f"{self._draft.rotation}°"
        if abs(self._draft.fine_rotation) >= 0.05:
            rotation_text += f" + {self._draft.fine_rotation:.1f}°"
        self._preview_info.setText(
            f"{self._draft.doc_type.display_name} · "
            f"{int(source_width * export_scale)} × {int(source_height * export_scale)} px at 300 dpi · "
            f"rotation {rotation_text}"
        )

    def _schedule_preview(self) -> None:
        """Rebuild the preview soon, at most ~30 times per second.

        ``_refresh`` (dialog open, reset) renders right away; the live-drag and
        arrow-key paths go through this coalescing timer instead.
        """
        self._preview_pending = True
        if not self._preview_timer.isActive():
            self._preview_timer.start()

    def _flush_preview(self) -> None:
        if self._preview_pending:
            self._preview_pending = False
            self._update_preview()

    # ------------------------------------------------------------------ actions
    def _on_quad_edited(self, card_id: str, quad_norm) -> None:
        if card_id != self._draft.card_id:
            return
        # the view reports the displayed (already deskewed) quad -> bake it
        self._draft.quad_norm = np.asarray(quad_norm, dtype=np.float32).reshape(4, 2)
        self._draft.fine_rotation = 0.0
        self._fine_spin.blockSignals(True)
        self._fine_spin.setValue(0.0)
        self._fine_spin.blockSignals(False)
        self._schedule_preview()

    def _on_fine_rotation_changed(self, value: float) -> None:
        self._draft.fine_rotation = float(value)
        self._view.set_overlays(self._overlays(), self._draft.card_id)
        self._schedule_preview()

    def _on_rotate_started(self, card_id: str) -> None:
        if card_id != self._draft.card_id:
            return
        self._rotate_base = float(self._draft.fine_rotation)

    def _on_rotate_dragged(self, card_id: str, total: float) -> None:
        """Apply the rotation of the grip above the crop (live, while dragging)."""
        if card_id != self._draft.card_id:
            return
        target = self._rotate_base + float(total)
        if abs(target) > 45.0:
            # fold the accumulated angle into the geometry and keep rotating
            self._draft.bake_fine_rotation()
            self._rotate_base = -float(total)
            target = 0.0
        self._apply_fine_rotation(target)

    def _apply_fine_rotation(self, angle: float) -> None:
        """Set the fine rotation (0.1° precision) and refresh overlay + preview."""
        rounded = round(max(-45.0, min(45.0, float(angle))), 1)
        if abs(self._fine_spin.value() - rounded) < 1e-9:
            # the spin box already shows this value - no valueChanged would fire
            self._on_fine_rotation_changed(rounded)
        else:
            self._fine_spin.setValue(rounded)

    def _copy_ocr_text(self) -> None:
        """Copy the recognised text to the clipboard."""
        from PySide6.QtWidgets import QApplication

        QApplication.clipboard().setText(self._draft.ocr_text or "")

    def _expand_margin(self) -> None:
        """Grow the crop on all four sides by the margin spin box (in mm)."""
        millimetres = float(self._margin_spin.value())
        pixels = millimetres * float(self.PREVIEW_DPI) / 25.4
        self._view.resize_selected(pixels, pixels)

    def _rotate(self, degrees: int) -> None:
        self._draft.rotation = (self._draft.rotation + int(degrees)) % 360
        self._draft.orientation_confident = True
        self._update_preview()

    def _on_type_changed(self, index: int) -> None:
        if index < 0:
            return
        raw = self._type_combo.itemData(index)
        if raw is None:
            return
        # Qt returns the DocumentType (a str enum) as a plain string, so it must
        # be converted back instead of being type-checked.
        doc_type = DocumentType.from_value(raw)
        if doc_type is not self._draft.doc_type:
            self._draft.set_type(doc_type, 1.0)

    def _on_identifier_changed(self, text: str) -> None:
        self._draft.identifier = text.strip()

    def _on_include_toggled(self, checked: bool) -> None:
        self._draft.included = bool(checked)

    def _reset(self) -> None:
        self._draft = self._reset_source.copy_card()
        self._refresh()


