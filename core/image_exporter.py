"""Rendering of extracted cards to image files."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import cv2
import numpy as np
from PIL import Image

from .models import DetectedCard, DocumentType
from .pdf_loader import PdfDocument
from .perspective_corrector import rotate_image, warp_quad

SUPPORTED_FORMATS = {
    "jpg": ".jpg",
    "jpeg": ".jpg",
    "png": ".png",
    "tif": ".tif",
    "tiff": ".tiff",
}

_ILLEGAL_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


# ------------------------------------------------------------------ face blur
_FACE_CASCADE = None


def _get_face_cascade():
    """Load OpenCV's built-in Haar cascade for frontal faces."""
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _FACE_CASCADE = cv2.CascadeClassifier(cascade_path)
        if _FACE_CASCADE.empty():
            raise RuntimeError(f"Failed to load face cascade from {cascade_path}")
    return _FACE_CASCADE


def blur_faces(image: np.ndarray, blur_kernel: int = 51, min_face_size: int = 40) -> np.ndarray:
    """Detect and blur faces in *image* (BGR). Returns a copy with faces blurred."""
    if image is None or image.size == 0:
        return image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    cascade = _get_face_cascade()
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(min_face_size, min_face_size),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if len(faces) == 0:
        return image
    result = image.copy()
    k = max(3, blur_kernel | 1)  # ensure odd
    for (x, y, w, h) in faces:
        roi = result[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        blurred = cv2.GaussianBlur(roi, (k, k), 0)
        result[y:y+h, x:x+w] = blurred
    return result


@dataclass
class ExportOptions:
    """User selectable export settings."""

    output_dir: str = "output"
    image_format: str = "jpg"
    jpeg_quality: int = 95
    render_dpi: int = 300
    filename_template: str = "{type}_{identifier}_{page}_{sequence}"
    create_subfolder: bool = True
    overwrite_existing: bool = False
    apply_rotation: bool = True
    padding_ratio: float = 0.0  # cards already carry their own padding
    blur_faces: bool = False
    face_blur_kernel: int = 51
    min_face_size: int = 40

    def extension(self) -> str:
        return SUPPORTED_FORMATS.get(self.image_format.lower().strip("."), ".jpg")

    def pil_format(self) -> str:
        ext = self.extension()
        if ext == ".jpg":
            return "JPEG"
        if ext == ".png":
            return "PNG"
        if ext in (".tif", ".tiff"):
            return "TIFF"
        return "PNG"

    def normalized_template(self) -> str:
        template = (self.filename_template or "").strip()
        return template or "{type}_{identifier}_{page}_{sequence}"


@dataclass
class ExportedFile:
    """One exported image."""

    path: Path
    card_id: str = ""
    doc_type: DocumentType = DocumentType.UNKNOWN
    page_number: int = 0
    width: int = 0
    height: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


@dataclass
class ExportSummary:
    """Result of an export run."""

    output_dir: Path
    files: List[ExportedFile] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        return sum(1 for item in self.files if item.ok)

    @property
    def failure_count(self) -> int:
        return len(self.files) - self.success_count


def sanitize_filename(name: str, max_length: int = 120) -> str:
    """Make *name* safe for the Windows file system."""
    cleaned = _ILLEGAL_FILENAME.sub("", str(name or ""))
    cleaned = re.sub(r"\s+", "_", cleaned).strip(" ._")
    if not cleaned:
        cleaned = "document"
    return cleaned[:max_length].rstrip(" .")


def suggest_output_dir(pdf_path, base_dir: str, create_subfolder: bool = True) -> Path:
    """Return the folder that should receive the exported images."""
    base = Path(base_dir).expanduser()
    if pdf_path is not None and create_subfolder:
        stem = sanitize_filename(Path(pdf_path).stem) or "documents"
        return base / stem
    return base


def extract_card_image(document: PdfDocument, card: DetectedCard, dpi: int, padding_ratio: float = 0.0) -> np.ndarray:
    """Render the page of *card* at *dpi* and warp the card out of it."""
    effective_dpi = document.effective_dpi(card.page_index, dpi)
    page_image = document.render_page(card.page_index, dpi=effective_dpi)
    height, width = page_image.shape[:2]
    quad = card.effective_quad_pixels(width, height)
    warped = warp_quad(page_image, quad, pad_ratio=padding_ratio)
    del page_image
    return warped


class ImageExporter:
    """Writes card images to disk while building readable filenames."""

    def __init__(self, options: ExportOptions) -> None:
        self.options = options

    # ------------------------------------------------------------------ naming
    def build_filename(
        self,
        card: DetectedCard,
        sequence: int,
        pdf_stem: str = "",
        identifier_fallback: int = 0,
    ) -> str:
        identifier = sanitize_filename(card.identifier) if card.identifier else f"{identifier_fallback:03d}"
        values: Dict[str, str] = {
            "type": card.doc_type.file_token,
            "doc_type": card.doc_type.file_token,
            "identifier": identifier,
            "id": identifier,
            "page": f"P{card.page_number:02d}",
            "page_number": f"{card.page_number:02d}",
            "sequence": f"{sequence:02d}",
            "seq": f"{sequence:02d}",
            "pdf": sanitize_filename(pdf_stem) if pdf_stem else "",
        }
        template = self.options.normalized_template()
        try:
            name = template.format(**values)
        except (KeyError, IndexError, ValueError):
            name = f"{values['type']}_{identifier}_P{card.page_number:02d}_{sequence:02d}"
        return sanitize_filename(name)

    def unique_path(self, folder: Path, stem: str, used: set) -> Path:
        extension = self.options.extension()
        candidate = folder / f"{stem}{extension}"
        counter = 2
        while (candidate.exists() and not self.options.overwrite_existing) or candidate.name.lower() in used:
            candidate = folder / f"{stem}_{counter}{extension}"
            counter += 1
        used.add(candidate.name.lower())
        return candidate

    # ------------------------------------------------------------------- files
    def save_image(self, image: np.ndarray, path: Path, dpi: int) -> None:
        """Write *image* (BGR) to *path* using the configured format/quality."""
        if image is None or image.size == 0:
            raise ValueError("The extracted image is empty.")
        path.parent.mkdir(parents=True, exist_ok=True)
        pil_image = Image.fromarray(image) if image.ndim == 2 else Image.fromarray(image[:, :, ::-1])
        pil_format = self.options.pil_format()
        save_kwargs: Dict[str, object] = {"dpi": (int(dpi), int(dpi))}
        if pil_format == "JPEG":
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            quality = int(min(100, max(40, self.options.jpeg_quality)))
            save_kwargs.update({"quality": quality, "subsampling": 0, "optimize": True})
        elif pil_format == "TIFF":
            save_kwargs.update({"compression": "tiff_lzw"})
        else:
            save_kwargs.update({"compress_level": 6})
        pil_image.save(path, format=pil_format, **save_kwargs)

    def export_cards(
        self,
        document: PdfDocument,
        cards: Sequence[DetectedCard],
        output_dir: Path,
        start_sequence: int = 1,
        progress: Optional[Callable[[int, int, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> ExportSummary:
        """Export every included card that belongs to *document*."""
        summary = ExportSummary(output_dir=output_dir)
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            summary.errors.append(f"Cannot create '{output_dir}': {exc}")
            return summary

        selected = [card for card in cards if card.included]
        total = len(selected)
        used: set = set()
        identifier_fallback = 0
        sequence = start_sequence

        for index, card in enumerate(selected):
            if cancel_check is not None and cancel_check():
                break
            identifier_fallback += 1
            if progress is not None:
                progress(index, total, f"Exporting page {card.page_number}…")
            try:
                image = extract_card_image(document, card, self.options.render_dpi, self.options.padding_ratio)
                if self.options.apply_rotation and card.rotation:
                    image = rotate_image(image, card.rotation)
                if self.options.blur_faces:
                    # min_face_size is tuned for the 300 dpi baseline - scale
                    # it with the actual render resolution, otherwise a
                    # sub-300 dpi export hunts face boxes that are larger than
                    # the whole (small) image.
                    dpi = document.effective_dpi(card.page_index, self.options.render_dpi)
                    min_face = max(16, int(round(self.options.min_face_size * dpi / 300.0)))
                    image = blur_faces(image, self.options.face_blur_kernel, min_face)
                stem = self.build_filename(card, sequence, document.stem, identifier_fallback)
                path = self.unique_path(output_dir, stem, used)
                self.save_image(image, path, self.options.render_dpi)
                height, width = image.shape[:2]
                summary.files.append(
                    ExportedFile(
                        path=path,
                        card_id=card.card_id,
                        doc_type=card.doc_type,
                        page_number=card.page_number,
                        width=width,
                        height=height,
                    )
                )
                sequence += 1
                del image
            except Exception as exc:  # report per-file errors and keep going
                summary.files.append(
                    ExportedFile(
                        path=output_dir,
                        card_id=card.card_id,
                        doc_type=card.doc_type,
                        page_number=card.page_number,
                        error=str(exc),
                    )
                )
                summary.errors.append(f"Page {card.page_number}: {exc}")
        if progress is not None:
            progress(total, total, "Export finished")
        return summary


