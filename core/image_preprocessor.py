"""Image pre-processing helpers used by the document detector.

The goal of this module is to turn a scanned page into one or more binary masks
where the document cards stand out from the scanner background, regardless of
uneven lighting, colour casts or how light the cards themselves are.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class PreprocessConfig:
    """Tunable parameters for :func:`build_masks`."""

    sensitivity: int = 5  # 1 (conservative) .. 10 (aggressive)
    denoise_level: int = 1  # 0 = off, 1 = light, 2 = medium, 3 = strong
    use_color: bool = True  # add a saturation based mask for coloured cards

    def clamped_sensitivity(self) -> int:
        return int(min(10, max(1, self.sensitivity)))


@dataclass
class MaskBundle:
    """Result of the pre-processing stage."""

    gray: np.ndarray
    normalized: np.ndarray
    gradient: np.ndarray  # float32 gradient magnitude, normalised to 0..1
    masks: List[Tuple[str, np.ndarray, float]] = field(default_factory=list)


def to_gray(image: np.ndarray) -> np.ndarray:
    """Return a single channel ``uint8`` view of *image*."""
    if image is None:
        raise ValueError("No image supplied to to_gray().")
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return arr
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)


def _odd(value: float, minimum: int = 3, maximum: int = 999) -> int:
    value = int(round(value))
    value = max(minimum, min(maximum, value))
    if value % 2 == 0:
        value += 1
    return min(value, maximum if maximum % 2 == 1 else maximum - 1)


def denoise(gray: np.ndarray, level: int = 1) -> np.ndarray:
    """Reduce scanner noise while keeping document edges crisp."""
    level = int(max(0, min(3, level)))
    if level == 0:
        return gray
    if level == 1:
        return cv2.medianBlur(gray, 3)
    if level == 2:
        return cv2.bilateralFilter(gray, 7, 60, 60)
    return cv2.fastNlMeansDenoising(gray, None, h=9, templateWindowSize=7, searchWindowSize=21)


def normalize_illumination(gray: np.ndarray) -> np.ndarray:
    """Divide out the low frequency background (flat field correction).

    This is what makes the detector robust against uneven scanner lamps and
    shadows: after the division every surface has roughly the same brightness as
    the scanner platen, so a global/adaptive threshold behaves consistently
    across the whole page.
    """
    short_side = min(gray.shape[:2])
    kernel_size = _odd(short_side / 18.0, minimum=15, maximum=151)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    background = cv2.GaussianBlur(background, (0, 0), sigmaX=kernel_size / 4.0)
    background = np.where(background == 0, 1, background).astype(np.uint8)
    normalized = cv2.divide(gray, background, scale=255)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(normalized)


def gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    """Scharr gradient magnitude normalised to ``0..1``."""
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(gx, gy)
    peak = float(magnitude.max())
    if peak <= 0:
        return np.zeros_like(gray, dtype=np.float32)
    return np.clip(magnitude / (peak * 0.5), 0.0, 1.0).astype(np.float32)


def _clean(mask: np.ndarray, open_size: int, close_size: int) -> np.ndarray:
    """Remove speckles and close small gaps so a card becomes one solid blob."""
    mask = (mask > 0).astype(np.uint8) * 255
    if close_size >= 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (_odd(close_size), _odd(close_size)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if open_size >= 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (_odd(open_size), _odd(open_size)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed regions of *mask*.

    This is what turns a closed outline (from Canny) or a light card body into a
    solid, rectangle-filling blob: the background is flood filled from outside
    and everything that could not be reached becomes foreground.
    """
    if mask is None or mask.size == 0:
        return mask
    binary = (mask > 0).astype(np.uint8) * 255
    padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flooded = padded.copy()
    cv2.floodFill(flooded, None, (0, 0), 255)
    holes = cv2.bitwise_not(flooded)
    filled = cv2.bitwise_or(padded, holes)
    return filled[1:-1, 1:-1]


def adaptive_masks(normalized: np.ndarray, sensitivity: int) -> List[Tuple[str, np.ndarray, float]]:
    """Adaptive threshold masks (dark surroundings and light surroundings)."""
    short_side = min(normalized.shape[:2])
    block = _odd(short_side / 14.0, minimum=25, maximum=151)
    c_value = int(round(12 - sensitivity))  # higher sensitivity -> more foreground
    inverse = cv2.adaptiveThreshold(
        normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, max(1, c_value)
    )
    direct = cv2.adaptiveThreshold(
        normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, max(1, c_value)
    )
    open_size = _odd(short_side / 320.0, minimum=3, maximum=17)
    close_size = _odd(short_side / 220.0, minimum=5, maximum=25)
    return [
        ("adaptive_inv", _clean(inverse, open_size, close_size), 1.0),
        ("adaptive_dir", _clean(direct, open_size, close_size), 0.9),
    ]


def otsu_masks(normalized: np.ndarray) -> List[Tuple[str, np.ndarray, float]]:
    """Global Otsu threshold in both polarities (works on clean scans)."""
    short_side = min(normalized.shape[:2])
    blurred = cv2.GaussianBlur(normalized, (5, 5), 0)
    _, dark = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, light = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    open_size = _odd(short_side / 320.0, minimum=3, maximum=17)
    close_size = _odd(short_side / 200.0, minimum=5, maximum=25)
    return [
        ("otsu_dark", _clean(dark, open_size, close_size), 0.85),
        ("otsu_light", _clean(light, open_size, close_size), 0.8),
    ]


def edge_masks(normalized: np.ndarray, sensitivity: int) -> List[Tuple[str, np.ndarray, float]]:
    """Edge based card detection.

    The physical card outline is the most reliable cue for light cards on a light
    platen.  The outline is thickened slightly (so small gaps close), the enclosed
    area is flood filled and the thickening is undone - which turns every card
    outline into one solid, rectangle-filling blob without ever bridging the gap
    between two neighbouring cards.
    """
    short_side = min(normalized.shape[:2])
    blur = cv2.GaussianBlur(normalized, (5, 5), 0)
    low = int(round(16 + (10 - sensitivity) * 2.2))
    high = int(round(low * 3))
    kernel_size = _odd(short_side / 300.0, minimum=3, maximum=15)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    edges = cv2.Canny(blur, low, high)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = fill_holes(edges)
    edges = cv2.erode(edges, kernel, iterations=1)
    edges = _clean(edges, _odd(short_side / 420.0, minimum=3, maximum=11), 3)

    gradient = gradient_magnitude(normalized)
    threshold = max(0.08, 0.22 - sensitivity * 0.015)
    strong = (gradient > threshold).astype(np.uint8) * 255
    strong = cv2.dilate(strong, kernel, iterations=1)
    strong = fill_holes(strong)
    strong = cv2.erode(strong, kernel, iterations=1)
    strong = _clean(strong, _odd(short_side / 420.0, minimum=3, maximum=11), 3)
    return [("edge", edges, 1.0), ("gradient", strong, 0.85)]


def color_mask(image: np.ndarray) -> List[Tuple[str, np.ndarray, float]]:
    """Saturation mask that isolates coloured cards from a neutral platen."""
    if image is None or image.ndim != 3 or image.shape[2] < 3:
        return []
    short_side = min(image.shape[:2])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = cv2.GaussianBlur(hsv[:, :, 1], (5, 5), 0)
    _, saturated = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel_size = _odd(short_side / 260.0, minimum=3, maximum=17)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    saturated = cv2.dilate(saturated, kernel, iterations=1)
    saturated = fill_holes(saturated)
    saturated = cv2.erode(saturated, kernel, iterations=1)
    saturated = _clean(saturated, _odd(short_side / 420.0, minimum=3, maximum=11), 3)
    return [("color", saturated, 0.8)]


def resize_max_dim(image: np.ndarray, max_dimension: int) -> np.ndarray:
    """Downscale *image* so its longest side is at most *max_dimension*.

    Detection is resolution independent (quad geometries are normalised), so a
    downscaled working copy makes detection faster and less noise sensitive while
    the export still re-renders the page at full resolution.
    """
    if image is None or image.size == 0 or max_dimension <= 0:
        return image
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_dimension:
        return image
    scale = max_dimension / float(longest)
    size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def build_masks(image: np.ndarray, config: PreprocessConfig | None = None) -> MaskBundle:
    """Run the full pre-processing chain and return every candidate mask."""
    config = config or PreprocessConfig()
    sensitivity = config.clamped_sensitivity()
    gray_raw = to_gray(image)
    gray = denoise(gray_raw, config.denoise_level)
    normalized = normalize_illumination(gray)
    gradient = gradient_magnitude(normalized)

    masks: List[Tuple[str, np.ndarray, float]] = []
    masks.extend(adaptive_masks(normalized, sensitivity))
    masks.extend(otsu_masks(normalized))
    masks.extend(edge_masks(normalized, sensitivity))
    if config.use_color:
        masks.extend(color_mask(image))
    return MaskBundle(gray=gray_raw, normalized=normalized, gradient=gradient, masks=masks)

