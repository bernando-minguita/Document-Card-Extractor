"""Application settings: dataclass, persistence and conversions."""

from __future__ import annotations

import json
import os
import re
import tempfile
import shutil
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, cast

from PySide6.QtCore import QStandardPaths

from core.image_exporter import ExportOptions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
SETTINGS_FILENAME = "settings.json"
# Folder name used in the user profile (per OS configuration folder).
APP_FOLDER_NAME = "DocumentCardExtractor"


def _config_folder() -> Path:
    """Absolute folder that holds the application's settings and history."""
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppConfigLocation)
    if not base:
        base = str(Path.home() / ".document_card_extractor")
    folder = Path(base)
    if folder.name.lower() != APP_FOLDER_NAME.lower():
        folder = folder / APP_FOLDER_NAME
    # Qt appends "<Organization>/<Application>" when both are set, which would
    # produce ".../DocumentCardExtractor/DocumentCardExtractor".  Collapse that.
    while folder.parent != folder and folder.parent.name.lower() == folder.name.lower():
        folder = folder.parent
    return folder


def settings_path() -> Path:
    """Location of the settings file (the user's configuration folder)."""
    return _config_folder() / SETTINGS_FILENAME


def _write_json_atomic(path: Path, data: Any) -> bool:
    """Write *data* to *path* without ever leaving a truncated JSON file.

    The content goes into a temporary file in the same folder first and then
    atomically replaces the target - a crash or power loss mid-write can
    therefore never truncate the previous settings (or history).
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temp_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=f"{path.stem}.", suffix=".tmp"
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as file_handle:
                json.dump(data, file_handle, indent=2)
            os.replace(temp_name, path)
        except BaseException:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise
    except Exception:
        return False
    return True


def _keep_corrupt_copy(path: Path) -> None:
    """Rename a broken JSON file so it is not silently overwritten."""
    try:
        if not path.exists():
            return
        backup = path.with_name(f"{path.stem}.invalid{path.suffix}")
        index = 2
        while backup.exists():
            backup = path.with_name(f"{path.stem}.invalid-{index}{path.suffix}")
            index += 1
        shutil.move(str(path), str(backup))
    except Exception:
        pass


@dataclass
class AppSettings:
    """All user configurable options of the application."""

    # ---------------------------------------------------------------- rendering
    render_dpi: int = 300
    cap_dpi_to_native: bool = True
    preview_dpi: int = 130
    detection_working_max_dim: int = 2200

    # ---------------------------------------------------------------- detection
    detection_sensitivity: int = 5
    min_document_area: float = 0.008
    max_document_area: float = 0.92
    detection_min_confidence: float = 0.42
    refine_quads: bool = True
    card_padding_ratio: float = 0.006
    # Size used by "Add Region" (millimetres, e.g. 85.6 x 53.98 for an ID card)
    manual_region_fixed_size: bool = False
    manual_region_preset: str = "free"  # free | id_card | business_card | passport | custom
    manual_region_width_mm: float = 85.6
    manual_region_height_mm: float = 53.98

    # ------------------------------------------------- orientation / classification
    auto_orientation: bool = True
    auto_classify: bool = True
    ocr_enabled: bool = False
    ocr_for_orientation: bool = False
    ocr_languages: str = "en"
    ocr_engine: str = "auto"

    # ------------------------------------------------------------------- export
    export_format: str = "jpg"
    jpeg_quality: int = 95
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    create_subfolder_per_pdf: bool = True
    filename_template: str = "{type}_{identifier}_{page}_{sequence}"
    open_output_after_export: bool = True

    # --------------------------------------------------------------------- misc
    thumbnail_size: int = 420
    last_input_dir: str = ""
    recent_files: List[str] = field(default_factory=list)
    # User defined document types: [{"token": ..., "label": ..., "keywords": "a, b"}]
    custom_document_types: List[Dict[str, Any]] = field(default_factory=list)
    # Extra OCR keywords per type token (works for built-in types too)
    type_keyword_overrides: Dict[str, List[str]] = field(default_factory=dict)
    # Tokens of types the user switched off on the settings page
    disabled_document_types: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------ mapping
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppSettings":
        known = {item.name for item in fields(cls)}
        filtered = {key: value for key, value in (data or {}).items() if key in known}
        settings = cls()
        for key, value in filtered.items():
            current = getattr(settings, key)
            try:
                if isinstance(current, bool):
                    value = bool(value)
                elif isinstance(current, int) and not isinstance(current, bool):
                    value = int(value)
                elif isinstance(current, float):
                    value = float(value)
                elif isinstance(current, list):
                    value = list(value)
            except (TypeError, ValueError):
                continue
            setattr(settings, key, value)
        return settings

    # -------------------------------------------------------------- persistence
    @classmethod
    def load(cls) -> "AppSettings":
        path = settings_path()
        try:
            if path.exists():
                with open(path, "r", encoding="utf-8") as handle:
                    settings = cls.from_dict(json.load(handle))
                    settings.apply_document_types()
                    return settings
        except Exception:
            # The file could not be parsed/converted.  Keep the broken file for
            # diagnosis instead of silently overwriting the user's data with
            # defaults on the next save.
            _keep_corrupt_copy(path)
        return cls()

    def save(self) -> bool:
        """Write the settings atomically; ``False`` (instead of a crash or a
        silently swallowed error) when the file could not be written."""
        path = settings_path()
        try:
            return _write_json_atomic(path, self.to_dict())
        except Exception:
            return False

    # ---------------------------------------------------------- document types
    def apply_document_types(self) -> None:
        """Register the custom types and apply the enable/disable switches."""
        from core.models import DocumentType

        DocumentType.set_custom_types(self.custom_document_types)
        DocumentType.set_disabled_tokens(self.disabled_document_types)
        DocumentType.set_extra_keywords(
            cast(Dict[str, Sequence[str]], self.type_keyword_overrides)
        )

    # kept for backwards compatibility
    def apply_custom_types(self) -> None:
        self.apply_document_types()

    def set_type_keywords(self, token: str, keywords) -> bool:
        """Store the extra OCR keywords for *token* (built-in or custom types).

        *keywords* may be a list or a comma/newline separated string.  An empty
        value removes the entry again (back to the built-in keywords).
        """
        from core.models import DocumentType

        token = str(token or "").strip()
        if not token:
            return False
        if isinstance(keywords, str):
            items = [
                part.strip()
                for part in keywords.replace("\n", ",").replace(";", ",").split(",")
            ]
        else:
            items = [str(item).strip() for item in (keywords or [])]
        items = [item for item in items if item]

        overrides = dict(self.type_keyword_overrides or {})
        if items:
            overrides[token] = items
        else:
            overrides.pop(token, None)
        if overrides == (self.type_keyword_overrides or {}):
            return False
        self.type_keyword_overrides = overrides
        DocumentType.set_extra_keywords(
            cast(Dict[str, Sequence[str]], self.type_keyword_overrides)
        )
        return True

    def type_keywords(self, token: str) -> list:
        """The extra keywords currently stored for *token*."""
        values = (self.type_keyword_overrides or {}).get(str(token or "").strip(), [])
        return [str(item) for item in values]

    def set_type_enabled(self, token: str, enabled: bool) -> bool:
        """Show or hide a document type in the pickers (works for built-ins too)."""
        from core.models import DocumentType

        token = str(token or "").strip()
        if not token:
            return False
        disabled = [item for item in self.disabled_document_types if str(item) != token]
        if not enabled:
            if DocumentType.from_value(token) is DocumentType.UNKNOWN:
                return False  # "Unknown" must stay available
            disabled.append(token)
        if disabled == list(self.disabled_document_types):
            return False
        self.disabled_document_types = disabled
        DocumentType.set_disabled_tokens(self.disabled_document_types)
        return True

    def add_custom_type(self, label: str, keywords: str = "") -> Optional[Dict[str, str]]:
        """Add a user defined document type and register it immediately.

        Returns ``None`` when the name is empty or already used by an existing
        type (built-in or custom).
        """
        from core.models import DocumentType

        label = (label or "").strip()
        if not label:
            return None
        if DocumentType.from_value(label) is not DocumentType.UNKNOWN:
            return None  # the label/token already exists
        token = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "custom"
        existing = {str(entry.get("token")) for entry in self.custom_document_types}
        base, counter = token, 2
        while token in existing or token in (entry.token for entry in DocumentType.builtin_types()):
            token = f"{base}_{counter}"
            counter += 1
        entry = {"token": token, "label": label, "keywords": (keywords or "").strip()}
        self.custom_document_types.append(entry)
        DocumentType.set_custom_types(self.custom_document_types)
        return entry

    def remove_custom_type(self, token: str) -> bool:
        """Remove a user defined document type."""
        from core.models import DocumentType

        remaining = [
            entry for entry in self.custom_document_types if str(entry.get("token")) != str(token)
        ]
        if len(remaining) == len(self.custom_document_types):
            return False
        self.custom_document_types = remaining
        DocumentType.set_custom_types(self.custom_document_types)
        return True

    # -------------------------------------------------------------- conversions
    def export_options(self) -> ExportOptions:
        return ExportOptions(
            output_dir=self.output_dir or str(DEFAULT_OUTPUT_DIR),
            image_format=self.export_format,
            jpeg_quality=self.jpeg_quality,
            render_dpi=self.render_dpi,
            filename_template=self.filename_template,
            create_subfolder=self.create_subfolder_per_pdf,
        )

    def remember_file(self, path: str) -> None:
        self.last_input_dir = str(Path(path).parent)
        recent = [item for item in self.recent_files if Path(item) != Path(path)]
        recent.insert(0, str(path))
        self.recent_files = recent[:12]
