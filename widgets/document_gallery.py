"""Thumbnail gallery with the extracted documents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from PySide6.QtCore import QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import QAction, QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
)

from app import styles
from app.icons import icon
from core.models import DetectedCard, DocumentType
from widgets.qt_image import bgr_to_pixmap

ITEM_SIZE = QSize(238, 250)
THUMB_HEIGHT = 140


@dataclass
class GalleryFilter:
    """Filter criteria for the gallery."""
    text: str = ""
    doc_type: Optional[DocumentType] = None
    page: Optional[int] = None
    included_only: bool = False
    orientation_confident_only: bool = False

    def matches(self, card: DetectedCard) -> bool:
        if self.text:
            t = self.text.lower()
            hay = " ".join([
                card.doc_type.display_name.lower(),
                card.identifier.lower() if card.identifier else "",
                f"page {card.page_number}",
                card.ocr_text.lower() if card.ocr_text else "",
            ])
            if t not in hay:
                return False
        if self.doc_type and card.doc_type != self.doc_type:
            return False
        if self.page is not None and card.page_number != self.page:
            return False
        if self.included_only and not card.included:
            return False
        if self.orientation_confident_only and not card.orientation_confident:
            return False
        return True


class CardItemDelegate(QStyledItemDelegate):
    """Paints a card thumbnail with type, page, confidence and badges."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pixmaps: Dict[str, QPixmap] = {}
        # Scaled thumbnails, keyed by (card id, target size).  The delegate
        # paints the same tiles on every repaint (selection, hover, scroll),
        # so repeating the smooth resize in paint() shows up as visible jank.
        self._scaled: Dict[tuple, QPixmap] = {}

    def invalidate(self, card_id: Optional[str] = None) -> None:
        if card_id is None:
            self._pixmaps.clear()
            self._scaled.clear()
        else:
            self._pixmaps.pop(card_id, None)
            self._scaled = {key: value for key, value in self._scaled.items() if key[0] != card_id}

    def sizeHint(self, option, index) -> QSize:  # noqa: N802 - Qt naming
        return ITEM_SIZE

    def _pixmap_for(self, card: DetectedCard) -> QPixmap:
        cached = self._pixmaps.get(card.card_id)
        if cached is not None and cached.cacheKey():
            return cached
        pixmap = bgr_to_pixmap(card.thumbnail)
        self._pixmaps[card.card_id] = pixmap
        return pixmap

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:  # noqa: N802
        card: Optional[DetectedCard] = index.data(Qt.ItemDataRole.UserRole)
        if not isinstance(card, DetectedCard):
            super().paint(painter, option, index)
            return
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        rect = option.rect.adjusted(5, 5, -5, -5)

        painter.setBrush(QColor(styles.PRIMARY_SOFT) if (card.included and selected) else QColor("#ffffff"))
        pen = QPen(QColor(styles.PRIMARY if selected else styles.BORDER))
        pen.setWidth(2 if selected else 1)
        if not card.included:
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setColor(QColor(styles.BORDER_STRONG))
        painter.setPen(pen)
        painter.drawRoundedRect(rect, 10, 10)

        # ---------------------------------------------------------- thumbnail
        thumb_rect = QRect(rect.left() + 10, rect.top() + 10, rect.width() - 20, THUMB_HEIGHT)
        painter.setBrush(QColor("#f1f5f9"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(thumb_rect, 6, 6)
        pixmap = self._pixmap_for(card)
        if not pixmap.isNull():
            key = (card.card_id, thumb_rect.width(), thumb_rect.height())
            scaled = self._scaled.get(key)
            if scaled is None or scaled.isNull():
                scaled = pixmap.scaled(
                    thumb_rect.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self._scaled[key] = scaled
            x = thumb_rect.left() + (thumb_rect.width() - scaled.width()) // 2
            y = thumb_rect.top() + (thumb_rect.height() - scaled.height()) // 2
            painter.setOpacity(0.45 if not card.included else 1.0)
            painter.drawPixmap(x, y, scaled)
            painter.setOpacity(1.0)

        # --------------------------------------------------------------- text
        text_left = rect.left() + 12
        text_width = rect.width() - 24
        title_rect = QRect(text_left, thumb_rect.bottom() + 8, text_width, 20)
        title_font = QFont(option.font)
        title_font.setPointSizeF(10.5)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(QColor(styles.TEXT if card.included else styles.TEXT_FAINT))
        type_text = "Unknown document" if card.doc_type is DocumentType.UNKNOWN else card.doc_type.display_name
        painter.drawText(
            title_rect,
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            painter.fontMetrics().elidedText(type_text, Qt.TextElideMode.ElideRight, text_width - 26),
        )

        meta_font = QFont(option.font)
        meta_font.setPointSizeF(9.0)
        painter.setFont(meta_font)
        painter.setPen(QColor(styles.TEXT_MUTED))
        meta_parts = [f"Page {card.page_number:02d}", card.confidence_label()]
        if card.rotation:
            meta_parts.append(f"{card.rotation}°")
        if card.identifier:
            meta_parts.append(card.identifier)
        meta_rect = QRect(text_left, title_rect.bottom() + 2, text_width, 18)
        painter.drawText(
            meta_rect,
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            painter.fontMetrics().elidedText("  ·  ".join(meta_parts), Qt.TextElideMode.ElideRight, text_width),
        )

        status_rect = QRect(text_left, meta_rect.bottom() + 2, text_width, 18)
        if not card.included:
            painter.setPen(QColor(styles.TEXT_FAINT))
            painter.drawText(status_rect, int(Qt.AlignmentFlag.AlignLeft), "Excluded from export")
        elif not card.orientation_confident:
            painter.setPen(QColor(styles.WARNING))
            painter.drawText(status_rect, int(Qt.AlignmentFlag.AlignLeft), "Check orientation")

        # -------------------------------------------------------------- badges
        if not card.orientation_confident:
            badge = QRect(rect.right() - 30, rect.top() + 18, 20, 20)
            painter.setBrush(QColor(styles.WARNING))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(badge)
            painter.setPen(QColor("#ffffff"))
            font = QFont(option.font)
            font.setBold(True)
            font.setPointSizeF(9.5)
            painter.setFont(font)
            painter.drawText(badge, int(Qt.AlignmentFlag.AlignCenter), "!")

        if selected:
            check = QRect(rect.right() - 30, rect.bottom() - 30, 20, 20)
            painter.setBrush(QColor(styles.PRIMARY))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(check)
            painter.setBrush(QColor("#ffffff"))
            painter.drawEllipse(check.adjusted(6, 6, -6, -6))
        painter.restore()


class DocumentGallery(QListWidget):
    """Grid of extracted card thumbnails with a per-card context menu."""

    cardSelected = Signal(str)
    cardActivated = Signal(str)
    rotateRequested = Signal(str, int)  # card id, clockwise degrees
    typeChangeRequested = Signal(str, object)  # card id, DocumentType
    includeToggled = Signal(str)
    cropEditRequested = Signal(str)
    ocrTextRequested = Signal(str)
    removeRequested = Signal(str)
    exportRequested = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._delegate = CardItemDelegate(self)
        self._cards: Dict[str, DetectedCard] = {}
        self._all_cards: List[DetectedCard] = []
        self._filter = GalleryFilter()
        self.setObjectName("DocumentGallery")
        self.setItemDelegate(self._delegate)
        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.setMovement(QListWidget.Movement.Static)
        self.setUniformItemSizes(True)
        self.setSpacing(4)
        self.setWordWrap(False)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self.itemSelectionChanged.connect(self._on_selection_changed)
        self.itemDoubleClicked.connect(self._on_item_activated)

    # ------------------------------------------------------------------- content
    def set_cards(self, cards: List[DetectedCard], selected_id: str = "") -> None:
        """Rebuild the gallery from *cards*."""
        self._all_cards = list(cards)
        self._apply_filter(selected_id)

    def _apply_filter(self, selected_id: str = "") -> None:
        """Rebuild visible items from _all_cards using current filter."""
        self.blockSignals(True)
        self.clear()
        self._cards.clear()
        self._delegate.invalidate()
        for card in self._all_cards:
            if not self._filter.matches(card):
                continue
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, card)
            item.setSizeHint(ITEM_SIZE)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            item.setToolTip(self._tooltip(card))
            self.addItem(item)
            self._cards[card.card_id] = card
            if selected_id and card.card_id == selected_id:
                item.setSelected(True)
        self.blockSignals(False)
        if selected_id:
            item = self.item_for(selected_id)
            if item is not None:
                self.scrollToItem(item)

    def set_filter(self, filter_: GalleryFilter, selected_id: str = "") -> None:
        """Update the filter and rebuild visible items."""
        self._filter = filter_
        self._apply_filter(selected_id)

    def filter(self) -> GalleryFilter:
        return self._filter

    def update_card(self, card: DetectedCard) -> None:
        """Refresh a single card (used after a rotation or type change)."""
        # Update in _all_cards
        for i, c in enumerate(self._all_cards):
            if c.card_id == card.card_id:
                self._all_cards[i] = card
                break
        # Update in visible if it passes filter
        if self._filter.matches(card):
            item = self.item_for(card.card_id)
            self._cards[card.card_id] = card
            self._delegate.invalidate(card.card_id)
            if item is not None:
                item.setData(Qt.ItemDataRole.UserRole, card)
                item.setToolTip(self._tooltip(card))
            self.viewport().update()
        else:
            # Card no longer matches filter - remove from visible
            item = self.item_for(card.card_id)
            if item is not None:
                row = self.row(item)
                self.takeItem(row)
                self._cards.pop(card.card_id, None)
                self._delegate.invalidate(card.card_id)

    def item_for(self, card_id: str) -> Optional[QListWidgetItem]:
        for row in range(self.count()):
            item = self.item(row)
            card = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(card, DetectedCard) and card.card_id == card_id:
                return item
        return None

    def selected_card_id(self) -> str:
        item = self.currentItem()
        if item is None:
            return ""
        card = item.data(Qt.ItemDataRole.UserRole)
        return card.card_id if isinstance(card, DetectedCard) else ""

    def select_card(self, card_id: str) -> None:
        item = self.item_for(card_id)
        if item is not None:
            self.blockSignals(True)
            self.setCurrentItem(item)
            item.setSelected(True)
            self.blockSignals(False)
            self.scrollToItem(item)


    @staticmethod
    def _tooltip(card: DetectedCard) -> str:
        lines = [
            card.doc_type.display_name,
            f"Page {card.page_number}  ·  confidence {card.confidence_label()}",
        ]
        if card.rotation:
            lines.append(f"Rotation {card.rotation}°")
        if card.identifier:
            lines.append(f"Identifier {card.identifier}")
        if card.notes:
            lines.append(card.notes)
        if card.ocr_text:
            snippet = " ".join(card.ocr_text.split())
            lines.append(f"OCR: {snippet[:150]}{'…' if len(snippet) > 150 else ''}")
        elif card.included:
            lines.append("No OCR text (OCR disabled or nothing recognised)")
        if not card.included:
            lines.append("Excluded from export")
        return "\n".join(lines)

    # -------------------------------------------------------------------- events
    def _on_selection_changed(self) -> None:
        card_id = self.selected_card_id()
        if card_id:
            self.cardSelected.emit(card_id)

    def _on_item_activated(self, item: QListWidgetItem) -> None:
        card = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(card, DetectedCard):
            self.cardActivated.emit(card.card_id)

    def _show_context_menu(self, position: QPoint) -> None:
        item = self.itemAt(position)
        if item is None:
            return
        card = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(card, DetectedCard):
            return
        self.select_card(card.card_id)

        menu = QMenu(self)
        rotate_left = QAction(icon("rotate_left"), "Rotate left 90°", menu)
        rotate_right = QAction(icon("rotate_right"), "Rotate right 90°", menu)
        flip = QAction(icon("flip"), "Rotate 180°", menu)
        crop = QAction(icon("crop"), "Adjust crop…", menu)
        ocr_text_action = QAction(icon("search"), "Show OCR text…", menu)
        ocr_text_action.setEnabled(bool((card.ocr_text or "").strip()))
        toggle_text = "Exclude from export" if card.included else "Include in export"
        toggle = QAction(icon("check"), toggle_text, menu)
        export_one = QAction(icon("save"), "Export this document…", menu)
        remove = QAction(icon("trash"), "Remove from list", menu)

        type_menu = menu.addMenu(icon("document"), "Document type")
        for doc_type in DocumentType.choices():
            action = QAction(doc_type.display_name, type_menu)
            action.setCheckable(True)
            action.setChecked(doc_type is card.doc_type)
            action.triggered.connect(
                lambda _checked=False, dt=doc_type, cid=card.card_id: self.typeChangeRequested.emit(cid, dt)
            )
            type_menu.addAction(action)

        menu.addAction(rotate_left)
        menu.addAction(rotate_right)
        menu.addAction(flip)
        menu.addSeparator()
        menu.addAction(crop)
        menu.addAction(ocr_text_action)
        menu.addAction(toggle)
        menu.addSeparator()
        menu.addAction(export_one)
        menu.addAction(remove)

        rotate_left.triggered.connect(lambda: self.rotateRequested.emit(card.card_id, 270))
        rotate_right.triggered.connect(lambda: self.rotateRequested.emit(card.card_id, 90))
        flip.triggered.connect(lambda: self.rotateRequested.emit(card.card_id, 180))
        crop.triggered.connect(lambda: self.cropEditRequested.emit(card.card_id))
        ocr_text_action.triggered.connect(
            lambda: self.ocrTextRequested.emit(card.card_id)
        )
        toggle.triggered.connect(lambda: self.includeToggled.emit(card.card_id))
        export_one.triggered.connect(lambda: self.exportRequested.emit(card.card_id))
        remove.triggered.connect(lambda: self.removeRequested.emit(card.card_id))
        menu.exec(self.viewport().mapToGlobal(position))


