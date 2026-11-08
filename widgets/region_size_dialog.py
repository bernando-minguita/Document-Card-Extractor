"""Dialog used to type the physical size of a manually added region."""

from __future__ import annotations

from typing import Tuple

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)


def _make_spin(value: float, tooltip: str) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(10.0, 400.0)
    spin.setDecimals(1)
    spin.setSingleStep(0.5)
    spin.setSuffix(" mm")
    spin.setValue(float(value))
    spin.setToolTip(tooltip)
    spin.setMinimumWidth(140)
    return spin


class RegionSizeDialog(QDialog):
    """Width / height (millimetres) of the region drawn with *Add Region*."""

    def __init__(self, width_mm: float = 85.6, height_mm: float = 53.98, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Custom region size")
        self.setModal(True)
        self.setMinimumWidth(360)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        title = QLabel("Size of the region drawn with “Add Region”")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)

        self._width = _make_spin(width_mm, "Physical width of the region on the scanned page")
        self._height = _make_spin(height_mm, "Physical height of the region on the scanned page")

        form = QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)
        form.addRow("Width", self._width)
        form.addRow("Height", self._height)
        layout.addLayout(form)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Quick"))
        for label, width, height in (
            ("ID card", 85.6, 53.98),
            ("Business card", 90.0, 55.0),
            ("Passport page", 125.0, 88.0),
        ):
            button = QPushButton(label)
            button.clicked.connect(
                lambda _checked=False, w=width, h=height: self._apply(w, h)
            )
            preset_row.addWidget(button)
        preset_row.addStretch(1)
        layout.addLayout(preset_row)

        swap = QPushButton("⇄  Swap width and height")
        swap.setToolTip("Useful for documents that are stored in portrait")
        swap.clicked.connect(self._swap)
        layout.addWidget(swap)

        hint = QLabel(
            "The region keeps the centre of the box you draw and takes exactly this "
            "size on the page, so an identity card is cropped precisely."
        )
        hint.setObjectName("Faint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        apply_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        if apply_button is not None:
            apply_button.setText("Use this size")
            apply_button.setProperty("variant", "primary")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------ helpers
    def _apply(self, width: float, height: float) -> None:
        self._width.setValue(float(width))
        self._height.setValue(float(height))

    def _swap(self) -> None:
        width = self._width.value()
        self._width.setValue(self._height.value())
        self._height.setValue(width)

    def size_mm(self) -> Tuple[float, float]:
        """The chosen (width, height) in millimetres."""
        return float(self._width.value()), float(self._height.value())
