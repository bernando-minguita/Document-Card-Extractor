"""Document loading and page rendering (PDF *and* scanned images).

``PdfDocument`` is the single entry point used by the GUI, the workers and the
CLI.  It accepts PDF files (rendered with PyMuPDF) *and* raster images
(PNG/JPG/TIFF/BMP/WEBP - including multi-page TIFF), which are exposed as
"pages" of their own so every consumer keeps working unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:  # PyMuPDF >= 1.24 prefers the ``pymupdf`` module name
    import pymupdf as _fitz
except ImportError:  # pragma: no cover - older PyMuPDF releases
    import fitz as _fitz  # type: ignore[no-redef]


# Image formats accepted as input.  Multi-page TIFFs are supported (one page per
# frame), which is what most document scanners produce.
IMAGE_SUFFIXES = (
    ".png",
    ".jpg",
    ".jpeg",
    ".jpe",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".ppm",
    ".pgm",
)
SUPPORTED_SUFFIXES = (".pdf",) + IMAGE_SUFFIXES
# DPI assumed for an image that does not record its own resolution.
DEFAULT_IMAGE_DPI = 300


def is_supported_file(path: str | Path) -> bool:
    """``True`` when *path* (by extension) can be opened by :class:`PdfDocument`."""
    try:
        return Path(path).suffix.lower() in SUPPORTED_SUFFIXES
    except (TypeError, ValueError):
        return False


def supported_file_filter() -> str:
    """``QFileDialog`` / CLI style filter listing every accepted extension."""
    patterns = " ".join(f"*{suffix}" for suffix in SUPPORTED_SUFFIXES)
    return patterns


def _pil_modules():
    """Import Pillow lazily (only needed for image input)."""
    from PIL import Image, ImageOps

    return Image, ImageOps


class PdfError(RuntimeError):
    """Raised when a document cannot be opened, decrypted or rendered."""


@dataclass
class PageInfo:
    """Basic geometry information for a single page."""

    index: int
    width_pt: float
    height_pt: float
    rotation: int = 0


class PdfDocument:
    """Thin, thread friendly wrapper around a document.

    Accepts a **PDF** (rendered with PyMuPDF) or a **raster image**
    (PNG/JPG/TIFF/BMP/WEBP; every frame of a multi-page TIFF becomes a page).
    A worker thread must create its own :class:`PdfDocument` instance because
    the underlying MuPDF handles are not safe to share between threads.
    """

    def __init__(self, path: str | Path, password: Optional[str] = None) -> None:
        self._path = Path(path)
        self._doc = None
        self._password = password or ""
        self._page_cache: dict = {}
        self._image = None  # type: ignore[assignment]
        self._frames = 0
        self._image_dpi = 0
        self.open()

    # ------------------------------------------------------------------ lifetime
    def open(self) -> None:
        if not self._path.exists():
            raise PdfError(f"The file '{self._path.name}' could not be found.")
        suffix = self._path.suffix.lower()
        if suffix == ".pdf":
            self._open_pdf()
        elif suffix in IMAGE_SUFFIXES:
            self._open_image()
        else:
            raise PdfError(
                f"'{self._path.name}' is not a supported file.\n"
                "Supported types: PDF, PNG, JPG, TIFF (also multi-page), BMP, WEBP."
            )

    def _open_pdf(self) -> None:
        try:
            doc = _fitz.open(str(self._path))
        except Exception as exc:  # corrupted / unsupported file
            raise PdfError(f"'{self._path.name}' is not a readable PDF file.\n{exc}") from exc

        if getattr(doc, "is_encrypted", False) or doc.needs_pass:
            if not self._password or not doc.authenticate(self._password):
                doc.close()
                raise PdfError(
                    f"'{self._path.name}' is password protected.\n"
                    "Remove the password (or supply it) and try again."
                )
        if doc.page_count == 0:
            doc.close()
            raise PdfError(f"'{self._path.name}' does not contain any page.")
        self._doc = doc

    def _open_image(self) -> None:
        try:
            Image, _ImageOps = _pil_modules()
        except ImportError as exc:  # pragma: no cover - Pillow is a requirement
            raise PdfError(
                "Reading image files needs Pillow: pip install -r requirements.txt"
            ) from exc

        try:
            image = Image.open(self._path)
            image.load()  # fail early on a truncated/corrupt file
        except Exception as exc:
            raise PdfError(f"'{self._path.name}' is not a readable image.\n{exc}") from exc

        frames = int(getattr(image, "n_frames", 1) or 1)
        dpi_info = image.info.get("dpi") if hasattr(image, "info") else None
        dpi = 0.0
        try:
            if isinstance(dpi_info, (tuple, list)) and dpi_info:
                dpi = float(dpi_info[0])
            elif isinstance(dpi_info, (int, float)):
                dpi = float(dpi_info)
        except (TypeError, ValueError):
            dpi = 0.0
        self._image = image
        self._frames = max(1, frames)
        self._image_dpi = int(round(dpi)) if dpi >= 36 else DEFAULT_IMAGE_DPI

    def close(self) -> None:
        if self._doc is not None:
            try:
                self._doc.close()
            finally:
                self._doc = None
        if self._image is not None:
            try:
                self._image.close()
            finally:
                self._image = None
        self._page_cache.clear()

    def __enter__(self) -> "PdfDocument":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ metadata
    @property
    def path(self) -> Path:
        return self._path

    @property
    def filename(self) -> str:
        return self._path.name

    @property
    def stem(self) -> str:
        return self._path.stem

    @property
    def page_count(self) -> int:
        if self._doc is not None:
            return int(self._doc.page_count)
        return int(self._frames)

    @property
    def is_image(self) -> bool:
        """``True`` when the file was loaded as a raster image."""
        return self._image is not None

    @property
    def metadata(self) -> dict:
        if self._image is not None:
            info = dict(getattr(self._image, "info", {}) or {})
            info.update(
                {
                    "format": getattr(self._image, "format", "") or "",
                    "width": self._image.size[0] if hasattr(self._image, "size") else 0,
                    "height": self._image.size[1] if hasattr(self._image, "size") else 0,
                    "dpi": self._image_dpi,
                    "pages": self._frames,
                }
            )
            return info
        if self._doc is None:
            return {}
        try:
            return dict(self._doc.metadata or {})
        except Exception:
            return {}

    def _page(self, index: int):
        if self._doc is None:
            raise PdfError("The document is not open.")
        if index < 0 or index >= self.page_count:
            raise PdfError(f"Page {index + 1} does not exist (document has {self.page_count} pages).")
        return self._doc.load_page(index)

    def _image_frame(self, index: int):
        """Seek to frame *index* of an image and return it (EXIF rotation applied)."""
        if self._image is None:
            raise PdfError("The image is not open.")
        if index < 0 or index >= self.page_count:
            raise PdfError(
                f"Page {index + 1} does not exist (the file has {self.page_count} page(s))."
            )
        _Image, ImageOps = _pil_modules()
        if self._frames > 1:
            self._image.seek(index)
        return ImageOps.exif_transpose(self._image)

    def page_info(self, index: int) -> PageInfo:
        if self._image is not None:
            frame = self._image_frame(index)
            dpi = float(self._image_dpi or DEFAULT_IMAGE_DPI)
            width_px, height_px = frame.size
            return PageInfo(
                index=index,
                width_pt=float(width_px) * 72.0 / dpi,
                height_pt=float(height_px) * 72.0 / dpi,
                rotation=0,
            )
        page = self._page(index)
        rect = page.rect
        return PageInfo(index=index, width_pt=float(rect.width), height_pt=float(rect.height), rotation=int(page.rotation))


    # ------------------------------------------------------------------ rendering
    def render_page(self, index: int, dpi: int = 300, grayscale: bool = False) -> np.ndarray:
        """Render *index* as a BGR (or gray) ``numpy`` array at *dpi*."""
        dpi = max(36, min(int(dpi), 1200))
        if self._image is not None:
            return self._render_image(index, dpi, grayscale)

        page = self._page(index)
        zoom = dpi / 72.0
        matrix = _fitz.Matrix(zoom, zoom)
        colorspace = _fitz.csGRAY if grayscale else _fitz.csRGB
        try:
            pix = page.get_pixmap(matrix=matrix, colorspace=colorspace, alpha=False)
        except Exception as exc:
            raise PdfError(f"Page {index + 1} could not be rendered.\n{exc}") from exc

        buffer = np.frombuffer(pix.samples, dtype=np.uint8)
        if grayscale:
            return buffer.reshape(pix.height, pix.width).copy()
        image = buffer.reshape(pix.height, pix.width, pix.n)
        if pix.n == 3:
            return image[:, :, ::-1].copy()  # RGB -> BGR
        return image.copy()

    def _render_image(self, index: int, dpi: int, grayscale: bool) -> np.ndarray:
        """Return frame *index* of a raster image, downscaled to *dpi* if needed.

        A photo is never upscaled: asking for more DPI than the file contains
        returns the native pixels (that is all the detail there is).
        """
        try:
            Image, _ImageOps = _pil_modules()
            frame = self._image_frame(index)
            mode = "L" if grayscale else "RGB"
            if frame.mode != mode:
                frame = frame.convert(mode)
            scale = float(dpi) / float(self._image_dpi or DEFAULT_IMAGE_DPI)
            if scale < 0.999:
                size = (
                    max(1, int(round(frame.width * scale))),
                    max(1, int(round(frame.height * scale))),
                )
                resample = getattr(Image, "Resampling", Image).LANCZOS
                frame = frame.resize(size, resample)
            array = np.asarray(frame)
        except PdfError:
            raise
        except Exception as exc:
            raise PdfError(
                f"Page {index + 1} of '{self._path.name}' could not be decoded.\n{exc}"
            ) from exc
        if grayscale:
            return np.ascontiguousarray(array)
        return np.ascontiguousarray(array[:, :, ::-1])  # RGB -> BGR

    def native_dpi(self, index: int) -> int:
        """Estimate the resolution of the scan.

        For an image file this is the DPI stored in the file (``300`` when it
        records none); for a PDF page it is the highest effective DPI of the
        embedded images, or ``0`` when the page contains no raster image.
        """
        if self._image is not None:
            return int(self._image_dpi or DEFAULT_IMAGE_DPI)
        try:
            page = self._page(index)
            best = 0.0
            for info in page.get_images(full=True):
                xref = info[0]
                try:
                    rects = page.get_image_rects(xref)
                except Exception:
                    rects = []
                if not rects:
                    continue
                pixel_width = float(info[2] or 0)
                pixel_height = float(info[3] or 0)
                for rect in rects:
                    width_in = float(rect.width) / 72.0
                    height_in = float(rect.height) / 72.0
                    if width_in <= 0 or height_in <= 0:
                        continue
                    best = max(best, pixel_width / width_in, pixel_height / height_in)
            return int(round(best))
        except Exception:
            return 0

    def effective_dpi(self, index: int, requested_dpi: int, cap_to_native: bool = True) -> int:
        """Requested DPI clamped so we never upscale a scan beyond its detail."""
        requested_dpi = max(72, int(requested_dpi))
        if not cap_to_native:
            return requested_dpi
        native = self.native_dpi(index)
        if native <= 0:
            return requested_dpi
        # Allow a little headroom above the native resolution, then cap.  No
        # minimum-height floor here: a floor (e.g. max(150, ...)) would render a
        # sub-150-dpi source (screenshot, fax, phone photo) at 150 dpi, which IS
        # upscaling beyond its native detail - exactly what cap_to_native must
        # never do.
        return int(min(requested_dpi, math.ceil(native * 1.05)))
