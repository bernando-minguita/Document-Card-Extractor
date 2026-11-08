"""Shared data models used across the application layers."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Sequence

import numpy as np

from .perspective_corrector import rotate_quad

_TOKEN_PATTERN = re.compile(r"[^a-z0-9]+")


def _normalise_token(text: str) -> str:
    """Lower case, punctuation free token used for label comparisons."""
    return _TOKEN_PATTERN.sub("_", str(text).strip().lower()).strip("_")


def _file_token_from(label: str, explicit: Optional[str]) -> str:
    """Filename friendly token from a label (``Driver's License`` -> ``DriversLicense``)."""
    raw = str(explicit or label or "Document").strip()
    cleaned = re.sub(r"[^A-Za-z0-9]+", "", raw)
    return cleaned or "Document"


class DocumentType:
    """A kind of document.

    The built-in kinds (passport, Iqama, vehicle registration, driving licence,
    health insurance card, bank card, salary card, national ID) plus ``Unknown``
    are always present and can not be deleted - they can only be hidden
    (:meth:`set_disabled_tokens`).  Additional types can be registered at runtime
    from the settings page (:meth:`register`), optionally with OCR keywords so the
    classifier can recognise them automatically.

    Instances are created once and shared, therefore identity comparisons work
    everywhere::

        card.doc_type is DocumentType.IQAMA
    """

    _registry: Dict[str, "DocumentType"] = {}
    _custom_tokens: List[str] = []
    _aliases: Dict[str, "DocumentType"] = {}
    _disabled_tokens: set = set()
    _extra_keywords: Dict[str, tuple] = {}
    PASSPORT: ClassVar["DocumentType"]
    IQAMA: ClassVar["DocumentType"]
    ISTIMARA: ClassVar["DocumentType"]
    DRIVING_LICENSE: ClassVar["DocumentType"]
    HEALTH_INSURANCE: ClassVar["DocumentType"]
    BANK_CARD: ClassVar["DocumentType"]
    SALARY_CARD: ClassVar["DocumentType"]
    NATIONAL_ID: ClassVar["DocumentType"]
    UNKNOWN: ClassVar["DocumentType"]

    def __init__(
        self,
        token: str,
        label: str,
        file_token: str,
        keywords: Sequence[str] = (),
        builtin: bool = False,
    ) -> None:
        self.token = token
        self.label = label
        self._file_token = file_token
        self.keywords = tuple(keywords)
        self.builtin = builtin

    # ------------------------------------------------------------- str like API
    @property
    def value(self) -> str:
        """Token used for persistence (settings, history, JSON)."""
        return self.token

    @property
    def display_name(self) -> str:
        return self.label

    @property
    def file_token(self) -> str:
        """Token used when building export filenames (no spaces)."""
        return self._file_token

    @property
    def is_known(self) -> bool:
        return self is not DocumentType.UNKNOWN

    def __str__(self) -> str:
        return self.token

    def __repr__(self) -> str:
        return f"DocumentType({self.token!r})"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, DocumentType):
            return other is self
        if isinstance(other, str):
            return self.token == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.token)

    # ---------------------------------------------------------------- registry
    @classmethod
    def register(
        cls,
        token: str,
        label: str,
        file_token: Optional[str] = None,
        keywords: Sequence[str] = (),
        builtin: bool = False,
        replace: bool = False,
    ) -> "DocumentType":
        """Create (or return) the type identified by *token*.

        Built-in types can never be shadowed: a user defined type whose token is
        already taken gets a free token (``passport`` -> ``passport_2``).
        """
        base = _normalise_token(token) or "document"
        existing = cls._registry.get(base)

        if builtin:
            if existing is not None:
                return existing
            instance = cls._build(base, label, file_token, keywords, builtin=True)
            cls._registry[base] = instance
            return instance

        if existing is not None and not existing.builtin:
            if not replace:
                return existing
            cls._registry.pop(base, None)
            if base in cls._custom_tokens:
                cls._custom_tokens.remove(base)
            token = base
        elif existing is not None:  # token belongs to a built-in type
            counter = 2
            token = f"{base}_{counter}"
            while token in cls._registry:
                counter += 1
                token = f"{base}_{counter}"
        else:
            token = base

        instance = cls._build(token, label, file_token, keywords, builtin=False)
        cls._registry[token] = instance
        if token not in cls._custom_tokens:
            cls._custom_tokens.append(token)
        return instance

    @classmethod
    def _build(
        cls,
        token: str,
        label: str,
        file_token: Optional[str],
        keywords: Sequence[str],
        builtin: bool,
    ) -> "DocumentType":
        return cls(
            token=token,
            label=(label or token).strip(),
            file_token=_file_token_from(label or token, file_token),
            keywords=tuple(keyword for keyword in keywords if str(keyword).strip()),
            builtin=builtin,
        )

    @classmethod
    def unregister(cls, token: str) -> bool:
        """Remove a user defined type (built-in types cannot be removed)."""
        token = _normalise_token(token)
        entry = cls._registry.get(token)
        if entry is None or entry.builtin:
            return False
        cls._registry.pop(token, None)
        if token in cls._custom_tokens:
            cls._custom_tokens.remove(token)
        return True

    @classmethod
    def reset_custom_types(cls) -> None:
        for token in list(cls._custom_tokens):
            cls._registry.pop(token, None)
        cls._custom_tokens.clear()

    @classmethod
    def set_custom_types(cls, entries: Sequence[Dict[str, Any]]) -> List["DocumentType"]:
        """Replace the user defined types with *entries* (settings format)."""
        cls.reset_custom_types()
        created: List["DocumentType"] = []
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            label = str(entry.get("label") or entry.get("token") or "").strip()
            token = str(entry.get("token") or label).strip()
            if not token:
                continue
            keywords = entry.get("keywords") or ()
            if isinstance(keywords, str):
                keywords = [item.strip() for item in keywords.split(",")]
            created.append(
                cls.register(token, label or token, entry.get("file_token"), keywords=tuple(keywords))
            )
        return created

    @classmethod
    def custom_types(cls) -> List["DocumentType"]:
        return [cls._registry[token] for token in cls._custom_tokens if token in cls._registry]

    @classmethod
    def builtin_types(cls) -> List["DocumentType"]:
        return [item for item in cls._registry.values() if item.builtin and item.token != "unknown"]

    # ------------------------------------------------------ enable / disable
    @classmethod
    def set_disabled_tokens(cls, tokens) -> None:
        """Mark types as hidden: they are no longer offered or auto-assigned."""
        cls._disabled_tokens = {
            _normalise_token(token) for token in (tokens or []) if str(token).strip()
        }

    # -------------------------------------------------- extra OCR keywords
    @classmethod
    def set_extra_keywords(cls, mapping: Optional[Dict[str, Sequence[str]]]) -> None:
        """Keywords the user added to a type (built-in *and* custom ones).

        They are matched **in addition** to the built-in keyword tables, which is
        how another language or an alternative spelling can be supported without
        touching the code.
        """
        cleaned: Dict[str, tuple] = {}
        for token, values in (mapping or {}).items():
            key = _normalise_token(token)
            if not key:
                continue
            if isinstance(values, str):
                values = [item.strip() for item in values.split(",")]
            words = tuple(
                str(item).strip() for item in (values or []) if str(item).strip()
            )
            if words:
                cleaned[key] = words
        cls._extra_keywords = cleaned

    @classmethod
    def extra_keywords(cls, token: str) -> tuple:
        return tuple(cls._extra_keywords.get(_normalise_token(token), ()))

    @property
    def user_keywords(self) -> tuple:
        """Keywords the user added on the settings page."""
        return DocumentType.extra_keywords(self.token)

    @property
    def all_keywords(self) -> tuple:
        """Registered keywords plus the user's extras (no classifier built-ins)."""
        return tuple(dict.fromkeys(tuple(self.keywords) + self.user_keywords))

    @classmethod
    def reset_extra_keywords(cls) -> None:
        cls._extra_keywords = {}

    @classmethod
    def disabled_tokens(cls) -> List[str]:
        return sorted(cls._disabled_tokens)

    @property
    def enabled(self) -> bool:
        """``False`` when the user hid this type on the settings page."""
        if self.token == "unknown":
            return True  # "Unknown" is always available
        return self.token not in DocumentType._disabled_tokens

    @property
    def category(self) -> str:
        return "built-in" if self.builtin else "custom"

    @classmethod
    def all_choices(cls) -> List["DocumentType"]:
        """Every type, regardless of the enable/disable state (settings page)."""
        return cls.builtin_types() + cls.custom_types() + [cls.UNKNOWN]

    @classmethod
    def choices(cls) -> List["DocumentType"]:
        """Enabled types offered for assignment (Unknown always last)."""
        return [item for item in cls.builtin_types() + cls.custom_types() if item.enabled] + [
            cls.UNKNOWN
        ]

    @classmethod
    def from_value(cls, value: Any) -> "DocumentType":
        """Best effort conversion from a token or label (UI, settings, history)."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            text = _normalise_token(value)
            if not text:
                return cls.UNKNOWN
            if text in cls._registry:
                return cls._registry[text]
            for member in cls._registry.values():
                if text == _normalise_token(member.label):
                    return member
            if text in cls._aliases:
                return cls._aliases[text]
        return cls.UNKNOWN


DocumentType.PASSPORT = DocumentType.register("passport", "Passport", "Passport", builtin=True)
DocumentType.IQAMA = DocumentType.register("iqama", "Iqama", "Iqama", builtin=True)
DocumentType.ISTIMARA = DocumentType.register("vehicle_registration", "Vehicle Registration", "VehicleRegistration", builtin=True)
DocumentType.DRIVING_LICENSE = DocumentType.register(
    "driving_license", "Driver's License", "DriversLicense", builtin=True
)
DocumentType.HEALTH_INSURANCE = DocumentType.register(
    "health_insurance", "Health Insurance Card", "HealthInsuranceCard", builtin=True
)
DocumentType.BANK_CARD = DocumentType.register("bank_card", "Bank Card", "BankCard", builtin=True)
DocumentType.SALARY_CARD = DocumentType.register("salary_card", "Salary Card", "SalaryCard", builtin=True)
DocumentType.NATIONAL_ID = DocumentType.register(
    "national_id", "National ID", "NationalID", builtin=True
)
DocumentType.UNKNOWN = DocumentType.register("unknown", "Unknown", "Document", builtin=True)

DocumentType._aliases.update(
    {
        "driver_license": DocumentType.DRIVING_LICENSE,
        "drivers_license": DocumentType.DRIVING_LICENSE,
        "driving_licence": DocumentType.DRIVING_LICENSE,
        "license": DocumentType.DRIVING_LICENSE,
        "residence_permit": DocumentType.IQAMA,
        "residency": DocumentType.IQAMA,
        "vehicle_registration": DocumentType.ISTIMARA,
        "registration": DocumentType.ISTIMARA,
        "istimara": DocumentType.ISTIMARA,
        "insurance": DocumentType.HEALTH_INSURANCE,
        "medical_insurance": DocumentType.HEALTH_INSURANCE,
        "national_identity": DocumentType.NATIONAL_ID,
        "salary": DocumentType.SALARY_CARD,
        "bankcard": DocumentType.BANK_CARD,
    }
)


def _as_quad(value: Any) -> np.ndarray:
    """Convert *value* into a ``(4, 2)`` float32 numpy array."""
    return np.asarray(value, dtype=np.float32).reshape(4, 2)


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def _polygon_area(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


@dataclass
class DetectedCard:
    """A single document detected on a PDF page.

    The geometry is stored in **normalised page coordinates** (``0..1``).
    Keeping geometry normalised makes the card resolution independent: the
    preview can be drawn on a low resolution render while the export re-renders
    the page at full resolution and warps the same quad, so no quality is lost.
    """

    page_index: int
    quad_norm: np.ndarray
    confidence: float = 0.0
    source: str = "auto"
    doc_type: DocumentType = DocumentType.UNKNOWN
    type_confidence: float = 0.0
    identifier: str = ""
    ocr_text: str = ""  # raw text the OCR engine returned for this card
    rotation: int = 0
    fine_rotation: float = 0.0  # extra deskew in degrees (-45..45, clockwise)
    included: bool = True
    thumbnail: Optional[np.ndarray] = field(default=None, repr=False)
    notes: str = ""
    orientation_confident: bool = True
    page_size: tuple = (0, 0)  # pixel size of the render the quad was detected on
    card_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def __post_init__(self) -> None:
        self.quad_norm = _as_quad(self.quad_norm)
        self.rotation = int(self.rotation) % 360
        if self.rotation not in (0, 90, 180, 270):
            self.rotation = 0
        try:
            self.fine_rotation = max(-45.0, min(45.0, float(self.fine_rotation)))
        except (TypeError, ValueError):
            self.fine_rotation = 0.0

    # ------------------------------------------------------------------ geometry
    @property
    def page_number(self) -> int:
        """1-based page number."""
        return self.page_index + 1

    def quad_pixels(self, width: int, height: int) -> np.ndarray:
        """Quad expressed in pixels for a render of ``width`` x ``height``."""
        quad = self.quad_norm.copy()
        quad[:, 0] *= float(width)
        quad[:, 1] *= float(height)
        return quad

    def effective_quad_norm(self) -> np.ndarray:
        """Quad including the fine rotation (deskew) - what gets cropped/exported."""
        if abs(self.fine_rotation) < 1e-6:
            return self.quad_norm.copy()
        width, height = self.page_size if self.page_size else (0, 0)
        if not (width and height):
            return rotate_quad(self.quad_norm, self.fine_rotation)
        rotated = self.effective_quad_pixels(int(width), int(height))
        result = rotated.copy()
        result[:, 0] /= float(width)
        result[:, 1] /= float(height)
        return result

    def effective_quad_pixels(self, width: int, height: int) -> np.ndarray:
        """Pixel quad including the fine rotation for a render of *width* x *height*."""
        quad = self.quad_pixels(width, height)
        if abs(self.fine_rotation) < 1e-6:
            return quad
        return rotate_quad(quad, self.fine_rotation)

    def bake_fine_rotation(self) -> None:
        """Fold the fine rotation into ``quad_norm`` and reset the angle."""
        if abs(self.fine_rotation) < 1e-6:
            return
        self.quad_norm = self.effective_quad_norm()
        self.fine_rotation = 0.0


    def area_ratio(self) -> float:
        """Fraction of the page covered by the card."""
        return float(abs(_polygon_area(self.quad_norm)))

    def aspect_ratio(self) -> float:
        """Long edge / short edge of the card quad (always ``>= 1``)."""
        width, height = self.page_size if self.page_size else (0, 0)
        if width and height:
            quad = self.quad_pixels(int(width), int(height))
        else:
            quad = self.quad_norm
        widths = (_dist(quad[0], quad[1]), _dist(quad[3], quad[2]))
        heights = (_dist(quad[0], quad[3]), _dist(quad[1], quad[2]))
        long_edge = max(max(widths), max(heights))
        short_edge = min(min(widths), min(heights))
        if short_edge <= 1e-6:
            return 1.0
        return float(long_edge / short_edge)

    @property
    def is_portrait(self) -> bool:
        """``True`` when the card quad is taller than it is wide."""
        width, height = self.page_size if self.page_size else (0, 0)
        quad = self.quad_pixels(int(width), int(height)) if width and height else self.quad_norm
        horizontal = (_dist(quad[0], quad[1]) + _dist(quad[3], quad[2])) / 2.0
        vertical = (_dist(quad[0], quad[3]) + _dist(quad[1], quad[2])) / 2.0
        return vertical > horizontal

    # ------------------------------------------------------------------ helpers
    def set_type(self, doc_type: DocumentType, confidence: float = 0.0) -> None:
        self.doc_type = DocumentType.from_value(doc_type)
        self.type_confidence = float(confidence)

    def confidence_label(self) -> str:
        return f"{self.confidence * 100:.0f}%"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "card_id": self.card_id,
            "page_index": self.page_index,
            "quad_norm": self.quad_norm.tolist(),
            "confidence": self.confidence,
            "source": self.source,
            "doc_type": self.doc_type.value,
            "type_confidence": self.type_confidence,
            "identifier": self.identifier,
            "ocr_text": self.ocr_text,
            "rotation": self.rotation,
            "fine_rotation": self.fine_rotation,
            "included": self.included,
            "notes": self.notes,
            "orientation_confident": self.orientation_confident,
            "page_size": list(self.page_size) if self.page_size else [0, 0],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DetectedCard":
        page_size = data.get("page_size") or [0, 0]
        return cls(
            page_index=int(data.get("page_index", 0)),
            quad_norm=_as_quad(data.get("quad_norm", [[0, 0], [1, 0], [1, 1], [0, 1]])),
            confidence=float(data.get("confidence", 0.0)),
            source=str(data.get("source", "auto")),
            doc_type=DocumentType.from_value(data.get("doc_type", "unknown")),
            type_confidence=float(data.get("type_confidence", 0.0)),
            identifier=str(data.get("identifier", "")),
            ocr_text=str(data.get("ocr_text", "")),
            rotation=int(data.get("rotation", 0)),
            fine_rotation=float(data.get("fine_rotation", 0.0) or 0.0),
            included=bool(data.get("included", True)),
            notes=str(data.get("notes", "")),
            orientation_confident=bool(data.get("orientation_confident", True)),
            page_size=(int(page_size[0]), int(page_size[1])),
            card_id=str(data.get("card_id") or uuid.uuid4().hex[:8]),
        )

    def copy_card(self) -> "DetectedCard":
        clone = DetectedCard.from_dict(self.to_dict())
        clone.thumbnail = None if self.thumbnail is None else self.thumbnail.copy()
        return clone


@dataclass
class PageDetection:
    """Detection result for a single page."""

    page_index: int
    cards: List[DetectedCard] = field(default_factory=list)
    error: str = ""

    @property
    def page_number(self) -> int:
        return self.page_index + 1
