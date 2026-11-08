"""Interactive PDF page view with card overlays and manual rectangle drawing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)

from app import styles
from core.perspective_corrector import expand_quad_axes


class CardOverlayItem(QGraphicsItem):
    """Draws one detected card as a polygon with draggable corner handles."""

    def __init__(self, card_id: str, quad: np.ndarray, color: str, label: str) -> None:
        super().__init__()
        self.card_id = card_id
        self.color = QColor(color)
        self.label = label
        self.selected = False
        self.editable = True
        self.excluded = False
        self.rotation_enabled = False
        self.handle_radius = 9.0
        self.setZValue(10)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.setAcceptHoverEvents(True)
        self._quad = np.asarray(quad, dtype=np.float64).reshape(4, 2).copy()
        self._hover_corner = -1

    # ------------------------------------------------------------------ geometry
    @property
    def quad(self) -> np.ndarray:
        return self._quad.copy()

    def set_quad(self, quad: np.ndarray) -> None:
        self.prepareGeometryChange()
        self._quad = np.asarray(quad, dtype=np.float64).reshape(4, 2).copy()
        self.update()

    def set_handle_radius(self, radius: float) -> None:
        if abs(radius - self.handle_radius) > 0.5:
            self.prepareGeometryChange()
            self.handle_radius = float(radius)
            self.update()

    def set_selected(self, selected: bool) -> None:
        if selected != self.selected:
            self.selected = bool(selected)
            self.update()

    def boundingRect(self) -> QRectF:  # noqa: N802 - Qt naming
        xs = self._quad[:, 0]
        ys = self._quad[:, 1]
        margin = self.handle_radius + 26
        if self.rotation_enabled:
            margin += self.rotation_handle_offset()
        return QRectF(
            float(xs.min()) - margin,
            float(ys.min()) - margin,
            float(xs.max() - xs.min()) + 2 * margin,
            float(ys.max() - ys.min()) + 2 * margin,
        )

    def polygon(self) -> QPolygonF:
        return QPolygonF([QPointF(float(x), float(y)) for x, y in self._quad])

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        path.addPolygon(self.polygon())
        return path

    def corner_at(self, point: QPointF, tolerance: float) -> int:
        """Index of the corner close to *point* or ``-1``."""
        for index, (x, y) in enumerate(self._quad):
            if abs(point.x() - x) <= tolerance and abs(point.y() - y) <= tolerance:
                return index
        return -1

    def move_corner(self, index: int, point: QPointF) -> None:
        if 0 <= index < 4:
            self.prepareGeometryChange()
            self._quad[index, 0] = point.x()
            self._quad[index, 1] = point.y()
            self.update()

    # ---------------------------------------------------------- rotation handle
    def rotation_handle_offset(self) -> float:
        """Distance of the rotation grip from the top edge, in scene units."""
        return self.handle_radius * 3.4

    def top_edge_center(self) -> QPointF:
        """Mid point of the edge between corner 0 (top-left) and corner 1."""
        top = (self._quad[0] + self._quad[1]) / 2.0
        return QPointF(float(top[0]), float(top[1]))

    def rotation_handle_point(self) -> QPointF:
        """Position of the rotation grip, outside the quad above the top edge."""
        top = self._quad[0:2].mean(axis=0)
        center = self._quad.mean(axis=0)
        direction = np.asarray(top, dtype=np.float64) - center
        length = float(np.linalg.norm(direction)) or 1.0
        offset = self.rotation_handle_offset()
        return QPointF(
            float(top[0] + direction[0] / length * offset),
            float(top[1] + direction[1] / length * offset),
        )

    def rotation_handle_at(self, point: QPointF, tolerance: float) -> bool:
        if not self.rotation_enabled:
            return False
        handle = self.rotation_handle_point()
        dx = point.x() - handle.x()
        dy = point.y() - handle.y()
        return (dx * dx + dy * dy) ** 0.5 <= tolerance

    # -------------------------------------------------------------------- paint
    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: ARG002
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        polygon = self.polygon()
        color = QColor(self.color)
        if self.excluded:
            color = QColor(styles.TEXT_FAINT)
        fill = QColor(color)
        fill.setAlpha(38 if self.selected else 20)
        pen = QPen(color, 2.4 if self.selected else 1.6)
        if self.excluded:
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setWidthF(1.8)
        painter.setPen(pen)
        painter.setBrush(QBrush(fill))
        painter.drawPolygon(polygon)

        if self.label:
            self._paint_label(painter)

        if self.selected and self.editable and not self.excluded:
            painter.setBrush(QBrush(QColor("#ffffff")))
            painter.setPen(QPen(color, 2.0))
            radius = self.handle_radius
            for x, y in self._quad:
                painter.drawEllipse(QPointF(float(x), float(y)), radius, radius)
            center = self._quad.mean(axis=0)
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(float(center[0]), float(center[1])), radius * 0.4, radius * 0.4)
            if self.rotation_enabled:
                self._paint_rotation_handle(painter, color)

    def _paint_rotation_handle(self, painter: QPainter, color: QColor) -> None:
        anchor = self.top_edge_center()
        grip = self.rotation_handle_point()
        radius = self.handle_radius * 0.92
        pen = QPen(color, 1.8, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawLine(anchor, grip)
        painter.setPen(QPen(color, 2.0))
        painter.setBrush(QBrush(QColor("#ffffff")))
        painter.drawEllipse(grip, radius, radius)
        # small curved arrow inside the grip
        painter.setBrush(QBrush(color))
        painter.setPen(QPen(color, 1.4))
        rect = QRectF(grip.x() - radius * 0.55, grip.y() - radius * 0.55, radius * 1.1, radius * 1.1)
        painter.drawArc(rect, 40 * 16, 260 * 16)

    def _paint_label(self, painter: QPainter) -> None:
        font = QFont(painter.font())
        font.setPointSizeF(max(7.0, self.handle_radius * 1.15))
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        text = self.label
        width = metrics.horizontalAdvance(text) + 16
        height = metrics.height() + 8
        anchor = self._quad[0]
        top = float(anchor[1]) - height - 6
        if top < 0:
            top = float(anchor[1]) + 6
        rect = QRectF(float(anchor[0]), top, width, height)
        background = QColor(self.color if not self.excluded else styles.TEXT_FAINT)
        background.setAlpha(238)
        painter.setBrush(QBrush(background))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(rect, 4, 4)
        painter.setPen(QPen(QColor("#ffffff")))
        painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), text)


@dataclass
class OverlaySpec:
    """Everything the view needs to draw one card."""

    card_id: str
    quad_norm: np.ndarray
    color: str = styles.PRIMARY
    label: str = ""
    excluded: bool = False


class PdfPageView(QGraphicsView):
    """Displays a rendered PDF page and its card overlays."""

    cardSelected = Signal(str)
    cardActivated = Signal(str)  # double click
    quadEdited = Signal(str, object)  # card id, quad in normalised coordinates
    rotateStarted = Signal(str)  # card id - the rotation grip was pressed
    rotateDragged = Signal(str, float)  # card id, total rotation of this drag (degrees)
    rotateFinished = Signal(str)  # card id - the grip was released
    manualRectDrawn = Signal(object)  # QRectF in normalised coordinates
    backgroundClicked = Signal()

    MODE_SELECT = "select"
    MODE_ADD = "add"

    # keyboard control of the selected region (scene pixels of the current page)
    NUDGE_STEP = 1.0
    NUDGE_STEP_FAST = 10.0
    MIN_QUAD_SIZE = 16.0

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
            | QPainter.RenderHint.TextAntialiasing
        )
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        self.setBackgroundBrush(QBrush(QColor(styles.SURFACE_ALT)))

        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._page_size = (0, 0)
        self._overlays: Dict[str, CardOverlayItem] = {}
        self._order: List[str] = []
        self._selected_id = ""
        self._mode = self.MODE_SELECT
        self._zoom = 1.0
        self._user_zoomed = False

        self._dragging_id = ""
        self._dragging_corner = -1
        self._moving_id = ""
        self._move_start = QPointF()
        self._move_origin: Optional[np.ndarray] = None
        self._rotation_enabled = False
        self._rotating_id = ""
        self._rotate_center = QPointF()
        self._rotate_last_angle = 0.0
        self._rotate_total = 0.0
        self._add_start: Optional[QPointF] = None
        self._rubber_item: Optional[QGraphicsRectItem] = None

    # -------------------------------------------------------------------- state
    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        self._mode = mode
        if mode == self.MODE_ADD:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
        self._cancel_rubber_band()

    @property
    def page_size(self) -> Tuple[int, int]:
        return self._page_size

    def set_rotation_handle(self, enabled: bool) -> None:
        """Show a rotation grip above the selected card (used by the crop editor)."""
        self._rotation_enabled = bool(enabled)
        for item in self._overlays.values():
            item.rotation_enabled = self._rotation_enabled
            item.update()

    # ------------------------------------------------- keyboard nudge / resize
    def selected_editable_item(self) -> Optional[CardOverlayItem]:
        """The selected overlay, if it may be edited."""
        item = self._overlays.get(self._selected_id)
        if item is None or not item.editable or item.excluded:
            return None
        return item

    def nudge_selected(self, dx: float, dy: float) -> bool:
        """Move the selected region by ``(dx, dy)`` scene pixels."""
        item = self.selected_editable_item()
        width, height = self._page_size
        if item is None or not (width and height):
            return False
        quad = item.quad
        # keep the region on the page
        delta_x = min(max(float(dx), -float(quad[:, 0].min())), float(width) - float(quad[:, 0].max()))
        delta_y = min(max(float(dy), -float(quad[:, 1].min())), float(height) - float(quad[:, 1].max()))
        moved = quad + np.array([delta_x, delta_y], dtype=np.float64)
        self._push_quad(item, moved)
        return True

    def resize_selected(self, delta_width: float, delta_height: float) -> bool:
        """Grow/shrink the selected region by *delta* on each side (own axes)."""
        item = self.selected_editable_item()
        page_width, page_height = self._page_size
        if item is None or not (page_width and page_height):
            return False
        quad = item.quad
        quad_width = float(np.linalg.norm(quad[1] - quad[0]))
        quad_height = float(np.linalg.norm(quad[3] - quad[0]))
        minimum = float(self.MIN_QUAD_SIZE)
        delta_width = max(float(delta_width), (minimum - quad_width) / 2.0)
        delta_height = max(float(delta_height), (minimum - quad_height) / 2.0)
        if abs(delta_width) < 1e-6 and abs(delta_height) < 1e-6:
            return False
        resized = expand_quad_axes(quad, delta_width, delta_height)
        self._push_quad(item, resized)
        return True

    def _push_quad(self, item: CardOverlayItem, quad: np.ndarray) -> None:
        """Apply *quad* (scene pixels) to the item and notify the owner."""
        width, height = self._page_size
        quad = np.asarray(quad, dtype=np.float64).reshape(4, 2)
        item.set_quad(quad)
        normalised = np.column_stack((quad[:, 0] / float(width), quad[:, 1] / float(height)))
        self.quadEdited.emit(item.card_id, normalised)

    def _handle_arrow_key(self, event) -> bool:
        """Arrows move the selected region, Shift+arrows resize it."""
        direction = {
            Qt.Key.Key_Left: (-1.0, 0.0),
            Qt.Key.Key_Right: (1.0, 0.0),
            Qt.Key.Key_Up: (0.0, -1.0),
            Qt.Key.Key_Down: (0.0, 1.0),
        }.get(event.key())
        if direction is None or self.selected_editable_item() is None:
            return False
        modifiers = event.modifiers()
        step = self.NUDGE_STEP
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            step = self.NUDGE_STEP_FAST
        elif modifiers & Qt.KeyboardModifier.AltModifier:
            step = self.NUDGE_STEP_FAST * 2.0
        delta_x, delta_y = direction
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            # Shift + Right/Left = wider/narrower, Shift + Up/Down = taller/shorter
            if delta_x:
                return self.resize_selected(delta_x * step, 0.0)
            return self.resize_selected(0.0, -delta_y * step)
        return self.nudge_selected(delta_x * step, delta_y * step)

    # -------------------------------------------------------------------- page
    def show_page(self, image: QImage, fit: bool = True) -> None:
        """Display *image* as the new page background."""
        self._scene.clear()
        self._overlays.clear()
        self._order.clear()
        self._selected_id = ""
        self._pixmap_item = None
        self._rubber_item = None
        self._dragging_id = ""
        self._dragging_corner = -1
        self._moving_id = ""
        self._move_origin = None
        if image.isNull():
            self._page_size = (0, 0)
            self._scene.setSceneRect(QRectF(0, 0, 1, 1))
            return
        pixmap = QPixmap.fromImage(image)
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._pixmap_item.setZValue(0)
        self._page_size = (pixmap.width(), pixmap.height())
        self._scene.setSceneRect(QRectF(0, 0, pixmap.width(), pixmap.height()))
        if fit:
            self.fit_to_window()
        else:
            self.set_zoom(self._zoom)

    def set_overlays(self, specs: Sequence[OverlaySpec], selected_id: str = "") -> None:
        """Replace all overlays with *specs*."""
        for item in self._overlays.values():
            self._scene.removeItem(item)
        self._overlays.clear()
        self._order.clear()
        width, height = self._page_size
        if width and height:
            handle = max(8.0, width * 0.0055)
            for spec in specs:
                quad = np.asarray(spec.quad_norm, dtype=np.float64).reshape(4, 2)
                quad_px = np.column_stack((quad[:, 0] * width, quad[:, 1] * height))
                item = CardOverlayItem(spec.card_id, quad_px, spec.color, spec.label)
                item.excluded = spec.excluded
                item.rotation_enabled = self._rotation_enabled
                item.set_handle_radius(handle)
                item.set_selected(bool(selected_id) and spec.card_id == selected_id)
                self._scene.addItem(item)
                self._overlays[spec.card_id] = item
                self._order.append(spec.card_id)
        self._selected_id = selected_id

    def select_card(self, card_id: str) -> None:
        self._selected_id = card_id
        for key, item in self._overlays.items():
            item.set_selected(key == card_id)


    # --------------------------------------------------------------------- zoom
    def fit_to_window(self) -> None:
        if self._pixmap_item is None:
            return
        self._user_zoomed = False
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom = self.transform().m11()
        self._update_handle_sizes()

    def set_zoom(self, factor: float) -> None:
        factor = max(0.05, min(8.0, float(factor)))
        self.resetTransform()
        self.scale(factor, factor)
        self._zoom = factor
        self._user_zoomed = True
        self._update_handle_sizes()

    def zoom_in(self) -> None:
        self.set_zoom(self.transform().m11() * 1.25)

    def zoom_out(self) -> None:
        self.set_zoom(self.transform().m11() / 1.25)


    def _update_handle_sizes(self) -> None:
        width = self._page_size[0] or 1
        scale = max(self.transform().m11(), 1e-3)
        # Keep the handles roughly constant in screen pixels.
        radius_scene = 9.0 / scale
        radius_scene = max(width * 0.0035, min(width * 0.02, radius_scene))
        for item in self._overlays.values():
            item.set_handle_radius(radius_scene)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        if not self._user_zoomed:
            self.fit_to_window()
        else:
            self._update_handle_sizes()

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta:
                self.set_zoom(self.transform().m11() * (1.15 if delta > 0 else 1 / 1.15))
            event.accept()
            return
        super().wheelEvent(event)

    # -------------------------------------------------------------------- input
    def _overlay_at(self, scene_pos: QPointF) -> Optional[CardOverlayItem]:
        candidates = self._scene.items(scene_pos)
        for item in candidates:
            if isinstance(item, CardOverlayItem):
                return item
        return None

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        scene_pos = self.mapToScene(event.position().toPoint())
        if self._mode == self.MODE_ADD and event.button() == Qt.MouseButton.LeftButton:
            # cancel the previous band first: it also clears the start point
            self._cancel_rubber_band()
            self._add_start = scene_pos
            self._rubber_item = self._scene.addRect(
                QRectF(scene_pos, scene_pos),
                QPen(QColor(styles.PRIMARY), 2, Qt.PenStyle.DashLine),
                QBrush(QColor(37, 99, 235, 40)),
            )
            self._rubber_item.setZValue(30)
            event.accept()
            return

        if event.button() == Qt.MouseButton.LeftButton:
            selected = self._overlays.get(self._selected_id)
            if selected is not None and selected.editable and not selected.excluded:
                if self._rotation_enabled and selected.rotation_handle_at(
                    scene_pos, selected.handle_radius * 1.9
                ):
                    self._start_rotation(selected, scene_pos)
                    event.accept()
                    return
                tolerance = selected.handle_radius * 1.8
                corner = selected.corner_at(scene_pos, tolerance)
                if corner >= 0:
                    self._dragging_id = selected.card_id
                    self._dragging_corner = corner
                    self.setDragMode(QGraphicsView.DragMode.NoDrag)
                    event.accept()
                    return

            item = self._overlay_at(scene_pos)
            if item is not None:
                was_selected = item.card_id == self._selected_id
                self.select_card(item.card_id)
                self.cardSelected.emit(item.card_id)
                if was_selected and item.editable and not item.excluded:
                    # second interaction with the selected card: move the whole
                    # region (all four corners together)
                    self._moving_id = item.card_id
                    self._move_start = scene_pos
                    self._move_origin = item.quad
                    self.setDragMode(QGraphicsView.DragMode.NoDrag)
                    self.viewport().setCursor(Qt.CursorShape.SizeAllCursor)
                event.accept()
                return
            if not self._selected_id:
                self.backgroundClicked.emit()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        scene_pos = self.mapToScene(event.position().toPoint())
        if self._rubber_item is not None and self._add_start is not None:
            rect = QRectF(self._add_start, scene_pos).normalized()
            self._rubber_item.setRect(rect)
            event.accept()
            return
        if self._dragging_id and self._dragging_corner >= 0:
            item = self._overlays.get(self._dragging_id)
            if item is not None:
                width, height = self._page_size
                point = QPointF(
                    min(max(scene_pos.x(), 0.0), float(width)),
                    min(max(scene_pos.y(), 0.0), float(height)),
                )
                item.move_corner(self._dragging_corner, point)
            event.accept()
            return
        if self._moving_id and self._move_origin is not None:
            item = self._overlays.get(self._moving_id)
            if item is not None:
                self._translate_item(item, scene_pos)
            event.accept()
            return
        if self._rotating_id:
            item = self._overlays.get(self._rotating_id)
            if item is not None:
                self._continue_rotation(scene_pos)
            event.accept()
            return
        super().mouseMoveEvent(event)

    # ---------------------------------------------------------------- rotation
    @staticmethod
    def _angle_at(point: QPointF, center: QPointF) -> float:
        return math.degrees(math.atan2(point.y() - center.y(), point.x() - center.x()))

    def _start_rotation(self, item: CardOverlayItem, scene_pos: QPointF) -> None:
        centre = item.quad.mean(axis=0)
        self._rotating_id = item.card_id
        self._rotate_center = QPointF(float(centre[0]), float(centre[1]))
        self._rotate_last_angle = self._angle_at(scene_pos, self._rotate_center)
        self._rotate_total = 0.0
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.viewport().setCursor(Qt.CursorShape.CrossCursor)
        self.rotateStarted.emit(item.card_id)

    def _continue_rotation(self, scene_pos: QPointF) -> None:
        angle = self._angle_at(scene_pos, self._rotate_center)
        delta = angle - self._rotate_last_angle
        while delta > 180.0:
            delta -= 360.0
        while delta < -180.0:
            delta += 360.0
        if abs(delta) < 0.02:
            return
        self._rotate_last_angle = angle
        self._rotate_total += delta
        # the owner (crop editor) applies the rotation and re-renders the overlays
        self.rotateDragged.emit(self._rotating_id, float(self._rotate_total))

    def _end_rotation(self) -> None:
        card_id = self._rotating_id
        self._rotating_id = ""
        self._rotate_total = 0.0
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
        if card_id:
            self.rotateFinished.emit(card_id)

    def _translate_item(self, item: CardOverlayItem, scene_pos: QPointF) -> None:
        """Move the whole quad by the drag delta, keeping it on the page."""
        origin = self._move_origin
        if origin is None:
            return
        delta_x = scene_pos.x() - self._move_start.x()
        delta_y = scene_pos.y() - self._move_start.y()
        width, height = self._page_size
        # clamp so no corner leaves the page
        min_x, max_x = float(origin[:, 0].min()), float(origin[:, 0].max())
        min_y, max_y = float(origin[:, 1].min()), float(origin[:, 1].max())
        delta_x = min(max(delta_x, -min_x), float(width) - max_x)
        delta_y = min(max(delta_y, -min_y), float(height) - max_y)
        moved = origin + np.array([delta_x, delta_y], dtype=np.float64)
        item.set_quad(moved)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._rubber_item is not None and self._add_start is not None:
            scene_pos = self.mapToScene(event.position().toPoint())
            rect = QRectF(self._add_start, scene_pos).normalized()
            self._cancel_rubber_band()
            width, height = self._page_size
            if width and height and rect.width() >= 12 and rect.height() >= 12:
                normalised = QRectF(
                    rect.x() / width, rect.y() / height, rect.width() / width, rect.height() / height
                )
                self.manualRectDrawn.emit(normalised)
            event.accept()
            return

        if self._dragging_id and self._dragging_corner >= 0:
            item = self._overlays.get(self._dragging_id)
            card_id = self._dragging_id
            self._dragging_id = ""
            self._dragging_corner = -1
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            if item is not None:
                width, height = self._page_size
                quad = item.quad
                quad[:, 0] = np.clip(quad[:, 0], 0, width)
                quad[:, 1] = np.clip(quad[:, 1], 0, height)
                normalised = np.column_stack((quad[:, 0] / width, quad[:, 1] / height))
                self.quadEdited.emit(card_id, normalised)
            event.accept()
            return

        if self._moving_id:
            item = self._overlays.get(self._moving_id)
            card_id = self._moving_id
            self._moving_id = ""
            self._move_origin = None
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
            if item is not None:
                width, height = self._page_size
                quad = item.quad
                normalised = np.column_stack((quad[:, 0] / width, quad[:, 1] / height))
                self.quadEdited.emit(card_id, normalised)
            event.accept()
            return

        if self._rotating_id:
            self._end_rotation()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt naming
        scene_pos = self.mapToScene(event.position().toPoint())
        item = self._overlay_at(scene_pos)
        if item is not None:
            self.cardActivated.emit(item.card_id)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.key() == Qt.Key.Key_Escape and self._mode == self.MODE_ADD:
            self._cancel_rubber_band()
            self.manualRectDrawn.emit(None)
            event.accept()
            return
        if self._handle_arrow_key(event):
            event.accept()
            return
        super().keyPressEvent(event)

    def _cancel_rubber_band(self) -> None:
        if self._rubber_item is not None:
            self._scene.removeItem(self._rubber_item)
        self._rubber_item = None
        self._add_start = None


