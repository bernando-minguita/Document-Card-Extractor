"""Conversion helpers between numpy images and Qt images."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy


def bgr_to_qimage(image: Optional[np.ndarray]) -> QImage:
    """Convert a BGR (or gray) ``numpy`` array into a detached :class:`QImage`."""
    if image is None or image.size == 0:
        return QImage()
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        height, width = array.shape
        buffer = np.ascontiguousarray(array)
        return QImage(buffer.data, width, height, width, QImage.Format.Format_Grayscale8).copy()
    if array.shape[2] == 4:
        rgba = np.ascontiguousarray(array[:, :, [2, 1, 0, 3]])
        height, width = rgba.shape[:2]
        return QImage(rgba.data, width, height, 4 * width, QImage.Format.Format_RGBA8888).copy()
    rgb = np.ascontiguousarray(array[:, :, ::-1])
    height, width = rgb.shape[:2]
    return QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()


def bgr_to_pixmap(image: Optional[np.ndarray]) -> QPixmap:
    """Convert a BGR ``numpy`` array into a :class:`QPixmap`."""
    qimage = bgr_to_qimage(image)
    if qimage.isNull():
        return QPixmap()
    return QPixmap.fromImage(qimage)


class FitImageLabel(QLabel):
    """Label that scales its image to fit and never clips it.

    A plain ``QLabel`` shows a pixmap at its natural size and simply cuts it off
    when the widget is smaller.  This label keeps the original pixmap, scales it
    with the aspect ratio preserved and re-scales whenever the widget is resized
    (window resize, High-DPI scaling, layout changes).
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._source = QPixmap()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setMinimumSize(240, 160)

    # ------------------------------------------------------------------ content
    def set_source(self, pixmap: Optional[QPixmap]) -> None:
        """Set the full size image that should be fitted into the label."""
        self._source = QPixmap() if pixmap is None else QPixmap(pixmap)
        self._apply_scale()

    def source(self) -> QPixmap:
        return self._source

    def clear_source(self) -> None:
        self._source = QPixmap()
        super().clear()

    # -------------------------------------------------------------------- paint
    def _apply_scale(self) -> None:
        if self._source.isNull() or self._source.width() <= 0 or self._source.height() <= 0:
            super().clear()
            return
        target = self.contentsRect().adjusted(2, 2, -2, -2).size()
        if target.width() <= 4 or target.height() <= 4:
            return
        scaled = self._source.scaled(
            target,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        super().setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._apply_scale()

