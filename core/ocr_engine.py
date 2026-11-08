"""Optional local OCR support.

Two engines are supported and detected at runtime:

* **Tesseract** via ``pytesseract`` (only when the Tesseract binary is installed);
* **EasyOCR** - pure local PyTorch models.  The engine is always created with
  ``download_enabled=False``, so the model files must already exist in
  ``~/.EasyOCR/model`` (or below ``EASYOCR_MODULE_PATH``).  They are never
  fetched by the application; :meth:`OcrEngine.easyocr_models_available` checks
  for them and :meth:`OcrEngine.easyocr_setup_hint` explains how to copy them
  from a machine with internet access.

The application works fine without any OCR engine: classification and
orientation detection then fall back to geometric heuristics.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np


@dataclass
class OcrWord:
    """A single recognised word."""

    text: str
    confidence: float = 0.0
    box: Optional[np.ndarray] = None


@dataclass
class OcrResult:
    """Aggregated OCR output for one image."""

    text: str = ""
    words: List[OcrWord] = field(default_factory=list)
    mean_confidence: float = 0.0
    engine: str = ""


# EasyOCR uses ISO 639-1 codes ("en", "ar") while Tesseract needs its own
# language data names ("eng", "ara").  Unknown codes are passed through, so
# "eng" or "deu" can be entered directly as well.
_TESSERACT_LANGUAGES = {
    "en": "eng",
    "ar": "ara",
    "fa": "fas",
    "fr": "fra",
    "de": "deu",
    "es": "spa",
    "it": "ita",
    "nl": "nld",
    "pt": "por",
    "ru": "rus",
    "tr": "tur",
    "ur": "urd",
    "hi": "hin",
    "bn": "ben",
    "zh": "chi_sim",
    "ja": "jpn",
    "ko": "kor",
    "th": "tha",
    "ta": "tam",
    "ml": "mal",
    "tl": "fil",
}


def tesseract_language_string(languages: Sequence[str]) -> str:
    """Map ``["en", "ar"]`` to the Tesseract notation ``"eng+ara"``."""
    mapped: List[str] = []
    for language in languages or ("en",):
        code = str(language).strip().lower().replace("_", "-").split("-")[0]
        if not code:
            continue
        mapped.append(_TESSERACT_LANGUAGES.get(code, code))
    unique = list(dict.fromkeys(mapped))
    return "+".join(unique) if unique else "eng"


def easyocr_language_list(languages: Sequence[str]) -> List[str]:
    """Normalise the configured languages for EasyOCR (``en``, ``ar``, …)."""
    result: List[str] = []
    for language in languages or ("en",):
        code = str(language).strip().lower().replace("_", "-").split("-")[0]
        if code and code not in result:
            result.append(code)
    return result or ["en"]


class OcrEngine:
    """Lazy, thread-safe façade over the available OCR backends."""

    _reader_cache: dict = {}
    _lock = threading.Lock()

    def __init__(self, languages: Sequence[str] = ("en",), prefer: str = "auto") -> None:
        self.languages = [str(lang) for lang in (languages or ("en",))]
        self.prefer = prefer
        self._backend: Optional[str] = None
        self._reader = None
        self._init_error = ""

    # ------------------------------------------------------------- availability
    @staticmethod
    def tesseract_available() -> bool:
        if shutil.which("tesseract"):
            return True
        try:
            import pytesseract

            for path in (
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            ):
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    return True
        except Exception:
            return False
        return False

    @staticmethod
    def easyocr_available() -> bool:
        try:
            import easyocr  # noqa: F401
        except Exception:
            return False
        return True

    # ------------------------------------------------------- easyocr model files
    @staticmethod
    def easyocr_model_dir() -> Path:
        """Folder that holds the EasyOCR ``*.pth`` model files."""
        override = os.environ.get("EASYOCR_MODULE_PATH") or os.environ.get("MODULE_PATH")
        base = Path(override) if override else Path.home() / ".EasyOCR"
        return base / "model"

    @classmethod
    def easyocr_models_available(cls, languages: Sequence[str] = ("en",)) -> bool:
        """``True`` when the detection + recognition models are already on disk.

        The application never downloads models (``download_enabled=False``), so a
        missing model file means OCR simply cannot run - better to detect that
        here than to fail silently on every card.
        """
        folder = cls.easyocr_model_dir()
        try:
            if not folder.is_dir():
                return False
            names = [item.name for item in folder.glob("*.pth")]
        except OSError:
            return False
        if not names:
            return False
        has_detector = any(name.startswith("craft_") for name in names)
        has_recogniser = any(not name.startswith("craft_") for name in names)
        return has_detector and has_recogniser

    @classmethod
    def easyocr_setup_hint(cls) -> str:
        folder = cls.easyocr_model_dir()
        return (
            "EasyOCR model files were not found in "
            f"'{folder}'.\n\n"
            "The application never downloads models (it must work offline). "
            "On a machine with internet access run once:\n"
            "    python -c \"import easyocr; easyocr.Reader(['en'])\"\n"
            "then copy the folder shown above to this computer "
            f"(default: %USERPROFILE%\\.EasyOCR\\model)."
        )

    @staticmethod
    def tesseract_setup_hint() -> str:
        """Explain how to install Tesseract (small and fast, no model download)."""
        return (
            "Tesseract was not found.\n\n"
            "1. Install the Windows build: https://github.com/UB-Mannheim/tesseract/wiki\n"
            "   Tick the language data you need (e.g. Arabic) in the installer.\n"
            "2. Install the Python wrapper:  pip install pytesseract\n"
            "3. Restart the application.  tesseract.exe is detected automatically in\n"
            "   C:\\Program Files\\Tesseract-OCR; otherwise add it to PATH."
        )

    @classmethod
    def diagnostic_report(
        cls,
        languages: Sequence[str] = ("en",),
        prefer: str = "auto",
        smoke_test: bool = True,
    ) -> str:
        """Multi-line report about the OCR engines (GUI *Check* button and CLI)."""
        wanted = easyocr_language_list(languages)
        lines = [
            f"Configured languages : {', '.join(wanted)}",
            f"Tesseract found      : {cls.tesseract_available()}",
            f"EasyOCR installed    : {cls.easyocr_available()}",
            f"EasyOCR model folder : {cls.easyocr_model_dir()}",
            f"Models for {', '.join(wanted):<10}: {cls.easyocr_models_available(wanted)}",
            f"Usable engines       : {cls.describe_availability() or 'none'}",
        ]
        engine = cls(languages=tuple(wanted), prefer=prefer)
        backend = engine.backend
        lines.append(f"Engine used          : {backend or 'none'} (setting: {prefer})")
        if backend and smoke_test:
            image = np.full((140, 460, 3), 255, dtype=np.uint8)
            cv2.putText(
                image,
                "PASSPORT WE4110283",
                (12, 88),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            started = time.perf_counter()
            result = engine.read(image)
            elapsed = time.perf_counter() - started
            text = (result.text or "").strip()
            lines.append(
                f"Smoke test           : engine={result.engine or 'none'} "
                f"time={elapsed:.2f}s text={text!r}"
            )
            if not text:
                lines.append(
                    "! The engine ran but recognised nothing - check the installed "
                    "language data (e.g. 'ara' for Arabic)."
                )
        if not backend:
            lines.append("")
            lines.append("No OCR engine is usable on this computer.")
            lines.append("")
            lines.extend(cls.tesseract_setup_hint().splitlines())
            lines.append("")
            lines.extend(cls.easyocr_setup_hint().splitlines())
        return "\n".join(lines)

    @classmethod
    def describe_availability(cls) -> str:
        engines = []
        if cls.tesseract_available():
            engines.append("Tesseract")
        if cls.easyocr_available():
            if cls.easyocr_models_available():
                engines.append("EasyOCR")
            else:
                engines.append("EasyOCR (model files missing)")
        return " + ".join(engines) if engines else ""

    # ------------------------------------------------------------------ backend
    def _select_backend(self) -> str:
        if self._backend is not None:
            return self._backend
        tesseract = self.tesseract_available()
        easyocr = self.easyocr_available()
        easyocr_ready = easyocr and self.easyocr_models_available(self.languages)

        if self.prefer == "tesseract" and tesseract:
            self._backend = "tesseract"
        elif self.prefer == "easyocr" and easyocr_ready:
            self._backend = "easyocr"
        elif tesseract:
            self._backend = "tesseract"
        elif easyocr_ready:
            self._backend = "easyocr"
        else:
            self._backend = ""
            if easyocr and not self._init_error:
                # installed but unusable offline: explain instead of failing silently
                self._init_error = self.easyocr_setup_hint()
        return self._backend

    @property
    def backend(self) -> str:
        return self._select_backend()

    @property
    def is_available(self) -> bool:
        return bool(self._select_backend())

    @property
    def init_error(self) -> str:
        return self._init_error

    # -------------------------------------------------------------- easyocr init
    def _get_easyocr_reader(self):
        languages = easyocr_language_list(self.languages)
        key = f"easyocr:{'-'.join(languages)}"
        with self._lock:
            if key in self._reader_cache:
                return self._reader_cache[key]

            if not self.easyocr_models_available(languages):
                self._init_error = self.easyocr_setup_hint()
                raise RuntimeError(self._init_error)

            import easyocr

            try:
                reader = easyocr.Reader(
                    languages, gpu=False, download_enabled=False, verbose=False
                )
            except TypeError:
                # Older easyocr builds do not know download_enabled.  The model
                # files were verified above, so nothing has to be downloaded;
                # if they were missing this would have raised already.
                reader = easyocr.Reader(languages, gpu=False, verbose=False)
            except Exception as exc:  # corrupt / unreadable model files
                self._init_error = f"EasyOCR could not load its models ({exc})."
                raise
            self._reader_cache[key] = reader
            return reader

    # ---------------------------------------------------------------------- read
    def read(self, image: np.ndarray, max_dimension: int = 1600) -> OcrResult:
        """Run OCR on *image* (BGR) and return the aggregated result."""
        if image is None or image.size == 0:
            return OcrResult(engine="none")
        prepared = _prepare(image, max_dimension)
        backend = self._select_backend()
        if backend == "tesseract":
            return self._read_tesseract(prepared)
        if backend == "easyocr":
            return self._read_easyocr(prepared)
        return OcrResult(engine="none")

    def _read_tesseract(self, image: np.ndarray) -> OcrResult:
        try:
            import pytesseract
            from pytesseract import Output

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            language = tesseract_language_string(self.languages)
            try:
                data = pytesseract.image_to_data(
                    rgb, output_type=Output.DICT, lang=language
                )
            except Exception:
                # the requested language data is not installed - fall back to the
                # Tesseract default instead of failing the whole card
                data = pytesseract.image_to_data(rgb, output_type=Output.DICT)
        except Exception as exc:  # pragma: no cover - machine dependent
            self._init_error = str(exc)
            return OcrResult(engine="tesseract")
        words: List[OcrWord] = []
        confidences: List[float] = []
        for index, text in enumerate(data.get("text", [])):
            text = (text or "").strip()
            if not text:
                continue
            try:
                confidence = float(data["conf"][index])
            except (KeyError, IndexError, ValueError):
                confidence = 0.0
            if confidence < 0:
                continue
            words.append(OcrWord(text=text, confidence=confidence / 100.0))
            confidences.append(confidence / 100.0)
        return _build_result(words, confidences, "tesseract")

    def _read_easyocr(self, image: np.ndarray) -> OcrResult:
        try:
            reader = self._get_easyocr_reader()
            raw = reader.readtext(image, detail=1, paragraph=False)
        except Exception as exc:
            self._init_error = str(exc)
            return OcrResult(engine="easyocr")
        words: List[OcrWord] = []
        confidences: List[float] = []
        for entry in raw:
            try:
                box, text, confidence = entry[0], entry[1], entry[2]
            except (IndexError, TypeError):
                continue
            text = (text or "").strip()
            if not text:
                continue
            confidence = float(confidence)
            words.append(OcrWord(text=text, confidence=confidence, box=np.asarray(box, dtype=np.float32)))
            confidences.append(confidence)
        return _build_result(words, confidences, "easyocr")


def _build_result(words: List[OcrWord], confidences: List[float], engine: str) -> OcrResult:
    text = " ".join(word.text for word in words)
    mean_confidence = float(np.mean(confidences)) if confidences else 0.0
    return OcrResult(text=text, words=words, mean_confidence=mean_confidence, engine=engine)


def _prepare(image: np.ndarray, max_dimension: int) -> np.ndarray:
    """Downscale very large crops so OCR stays fast, upscale very small ones."""
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest > max_dimension:
        scale = max_dimension / float(longest)
        return cv2.resize(
            image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA
        )
    if longest < 700:
        scale = 700.0 / float(max(1, longest))
        return cv2.resize(
            image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_CUBIC
        )
    return image

