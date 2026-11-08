"""Extraction history persistence (used by the dashboard and history page)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from app.settings import _write_json_atomic, settings_path

HISTORY_FILENAME = "history.json"
MAX_ENTRIES = 200


@dataclass
class HistoryEntry:
    """One finished export run."""

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    pdf_files: List[str] = field(default_factory=list)
    output_dir: str = ""
    document_count: int = 0
    page_count: int = 0
    files: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def display_time(self) -> str:
        try:
            return datetime.fromisoformat(self.timestamp).strftime("%d %b %Y  %H:%M")
        except ValueError:
            return self.timestamp

    @property
    def pdf_summary(self) -> str:
        if not self.pdf_files:
            return "-"
        names = [Path(item).name for item in self.pdf_files]
        if len(names) == 1:
            return names[0]
        return f"{names[0]} (+{len(names) - 1} more)"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "pdf_files": list(self.pdf_files),
            "output_dir": self.output_dir,
            "document_count": int(self.document_count),
            "page_count": int(self.page_count),
            "files": list(self.files),
            "errors": list(self.errors),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HistoryEntry":
        return cls(
            str(data.get("timestamp", "")),
            [str(item) for item in data.get("pdf_files", [])],
            str(data.get("output_dir", "")),
            int(data.get("document_count", 0)),
            int(data.get("page_count", 0)),
            [str(item) for item in data.get("files", [])],
            [str(item) for item in data.get("errors", [])],
        )


class HistoryStore:
    """Small JSON backed list of past exports."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or (settings_path().parent / HISTORY_FILENAME)
        self._entries: List[HistoryEntry] = []
        self.load()

    @property
    def entries(self) -> List[HistoryEntry]:
        return list(self._entries)

    def load(self) -> None:
        self._entries = []
        try:
            if self._path.exists():
                with open(self._path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                self._entries = [HistoryEntry.from_dict(item) for item in data.get("entries", [])]
        except Exception:
            self._entries = []

    def save(self) -> bool:
        """Write the history atomically (see ``_write_json_atomic``)."""
        return _write_json_atomic(self._path, {"entries": [entry.to_dict() for entry in self._entries]})

    def add(self, entry: HistoryEntry) -> bool:
        """Insert *entry* at the top and trim the store; ``False`` on write failure."""
        self._entries.insert(0, entry)
        self._entries = self._entries[:MAX_ENTRIES]
        return self.save()

    def clear(self) -> None:
        self._entries = []
        self.save()

    def totals(self) -> Dict[str, int]:
        return {
            "runs": len(self._entries),
            "documents": sum(entry.document_count for entry in self._entries),
            "pdfs": len({item for entry in self._entries for item in entry.pdf_files}),
        }
