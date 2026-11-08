"""Progress reporting widget used in the status bar."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QPushButton, QWidget

from app import styles
from app.icons import icon


class ProgressPanel(QWidget):
    """Status label + progress bar + cancel button."""

    cancelRequested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self._label = QLabel("Ready")
        self._label.setStyleSheet(f"color: {styles.TEXT_MUTED};")
        self._label.setMinimumWidth(220)

        self._bar = QProgressBar()
        self._bar.setFixedWidth(220)
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)

        self._cancel = QPushButton(icon("close"), "Cancel")
        self._cancel.setProperty("variant", "ghost")
        self._cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        self._cancel.clicked.connect(self.cancelRequested.emit)
        self._cancel.hide()

        layout.addWidget(self._label)
        layout.addWidget(self._bar)
        layout.addWidget(self._cancel)
        self._bar.hide()

    # -------------------------------------------------------------------- state
    def start(self, message: str = "Working…") -> None:
        self._label.setText(message)
        self._bar.setRange(0, 0)  # busy indicator until a real range arrives
        self._bar.show()
        self._cancel.show()

    def set_progress(self, current: int, total: int, message: str = "") -> None:
        total = max(1, int(total))
        self._bar.setRange(0, total)
        self._bar.setValue(max(0, min(int(current), total)))
        if message:
            self._label.setText(message)

    def set_message(self, message: str) -> None:
        self._label.setText(message)

    def finish(self, message: str = "Ready") -> None:
        self._label.setText(message)
        self._bar.hide()
        self._cancel.hide()

