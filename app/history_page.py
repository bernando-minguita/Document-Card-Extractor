"""The extraction-history page, extracted from ``app.main_window``.

``HistoryPage`` owns the history table and its buttons; the main window keeps
the shared ``HistoryStore`` and refreshes the page through delegates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
)

from app.icons import button_icon, icon


class HistoryPage(QWidget):
    """The extraction history table plus its toolbar."""

    def __init__(self, main_window: "QMainWindow", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._mw = main_window
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        title = QLabel("Extraction History")
        title.setObjectName("PageTitle")
        subtitle = QLabel("Every export you ran, with the folder that received the images.")
        subtitle.setObjectName("PageSubtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        self._history_table = QTableWidget(0, 5)
        self._history_table.setHorizontalHeaderLabels(["Date", "File", "Documents", "Output folder", "Status"])
        self._history_table.verticalHeader().setVisible(False)
        self._history_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._history_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._history_table.setAlternatingRowColors(True)
        self._history_table.doubleClicked.connect(self._open_history_folder)
        header = self._history_table.horizontalHeader()
        header.setStretchLastSection(True)
        header.resizeSection(0, 170)
        header.resizeSection(1, 260)
        header.resizeSection(3, 320)
        layout.addWidget(self._history_table, 1)

        buttons = QHBoxLayout()
        open_folder = QPushButton(icon("folder"), " Open output folder")
        open_folder.clicked.connect(self._open_history_folder)
        refresh = QPushButton(icon("refresh"), " Refresh")
        refresh.clicked.connect(self._refresh_history_table)
        clear = QPushButton(button_icon("trash", "danger"), " Clear history")
        clear.setProperty("variant", "danger")
        clear.clicked.connect(self._clear_history)
        buttons.addWidget(open_folder)
        buttons.addWidget(refresh)
        buttons.addStretch(1)
        buttons.addWidget(clear)
        layout.addLayout(buttons)

    def _refresh_history_table(self) -> None:
        entries = self._mw.history.entries
        self._history_table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            values = [
                entry.display_time,
                entry.pdf_summary,
                str(entry.document_count),
                entry.output_dir,
                f"{entry.document_count} exported" if not entry.errors else f"{len(entry.errors)} error(s)",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, entry.output_dir)
                self._history_table.setItem(row, column, item)
        if not entries:
            self._history_table.setRowCount(0)

    def _open_history_folder(self) -> None:
        row = self._history_table.currentRow()
        if row < 0:
            return
        item = self._history_table.item(row, 0)
        if item is None:
            return
        folder = item.data(Qt.ItemDataRole.UserRole)
        if folder:
            self._mw._open_folder(Path(folder))

    def _clear_history(self) -> None:
        confirm = QMessageBox.question(
            self,
            "Clear history",
            "Remove all history entries? The exported image files stay on disk.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self._mw.history.clear()
            self._refresh_history_table()
            self._mw._refresh_dashboard()
