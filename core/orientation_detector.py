"""Document orientation detection.

Two strategies are used:

* a fast heuristic based on the shape of word blobs.  Text blobs are wider than
  they are tall when the text is upright, and taller than wide when the document
  is rotated by 90 degrees, so comparing both measurements reliably separates
  ``0/180`` from ``90/270`` - offline and without OCR;
* an OCR assisted strategy which recognises the text at the two candidate
  rotations and keeps the one that is actually readable.  This resolves the
  ``0`` versus ``180`` ambiguity too, but it is slow, so it only runs when the
  user enables OCR.

When the module cannot be sure it reports ``confident=False`` instead of
guessing, and the user can rotate the document manually.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import cv2
import numpy as np

from .image_preprocessor import to_gray
from .perspective_corrector import rotate_image


@dataclass
class OrientationResult:
    """Outcome of an orientation analysis."""

    rotation: int = 0  # clockwise rotation to apply to make the text upright
    confident: bool = True  # False -> the user should verify the orientation
    method: str = "heuristic"
    score: float = 0.0
    details: Dict[str, float] = field(default_factory=dict)


class OrientationDetector:
    """Determine how a document must be rotated to be upright."""

    def __init__(self, min_margin: float = 0.12, analysis_size: int = 900) -> None:
        self.min_margin = float(min_margin)
        self.analysis_size = int(analysis_size)

    # ------------------------------------------------------------------ public
    def detect(
        self,
        image: np.ndarray,
        ocr_reader: Optional[Callable[[np.ndarray], object]] = None,
        allow_ocr: bool = True,
    ) -> OrientationResult:
        """Analyse *image* (BGR) and return the rotation that makes it upright."""
        if image is None or image.size == 0:
            return OrientationResult(0, False, "empty")

        heuristic = self._heuristic(image)
        if ocr_reader is not None and allow_ocr:
            ocr_result = self._ocr_refine(image, heuristic, ocr_reader)
            if ocr_result is not None:
                return ocr_result
        return heuristic

    # --------------------------------------------------------------- heuristic
    def _heuristic(self, image: np.ndarray) -> OrientationResult:
        horizontal = self._word_orientation_score(image)
        vertical = self._word_orientation_score(rotate_image(image, 90))
        if horizontal == 0.0 and vertical == 0.0:
            return OrientationResult(0, False, "heuristic", 0.0, {"no_text": 1.0})

        margin = horizontal - vertical
        if margin >= self.min_margin:
            rotation = 0
        elif margin <= -self.min_margin:
            rotation = 90
        else:
            return OrientationResult(
                0,
                False,
                "heuristic",
                float(margin),
                {"horizontal": float(horizontal), "vertical": float(vertical), "ambiguous": 1.0},
            )

        # The 0 vs 180 flip cannot be resolved from blob shapes alone: report the
        # canonical rotation and let the user (or OCR) resolve the flip.
        return OrientationResult(
            rotation=rotation,
            confident=False,
            method="heuristic",
            score=float(abs(margin)),
            details={"horizontal": float(horizontal), "vertical": float(vertical)},
        )

    def _word_orientation_score(self, image: np.ndarray) -> float:
        """Signed score: ``> 0`` horizontal text, ``< 0`` vertical text."""
        ink = self._ink_mask(image)
        if ink is None:
            return 0.0
        height, width = ink.shape[:2]
        half_width = max(3, int(round(width * 0.02)))
        if half_width % 2 == 0:
            half_width += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (half_width, 1))
        words = cv2.dilate(ink, kernel)
        count, _, stats, _ = cv2.connectedComponentsWithStats(words, connectivity=8)
        page_area = float(height * width)
        aspects = []
        weights = []
        for index in range(1, count):
            _, _, box_w, box_h, area = (int(value) for value in stats[index])
            if box_w < 5 or box_h < 5:
                continue
            if area < page_area * 1e-5:
                continue
            if box_h > 0.45 * height or box_w > 0.98 * width:
                continue  # single huge blob (photo, frame, sticker)
            aspect = box_w / float(box_h)
            if aspect > 25.0 or aspect < 1.0 / 25.0:
                continue  # straight line
            if area / float(box_w * box_h) < 0.12:
                continue  # hollow outline
            aspects.append(aspect)
            weights.append(float(area))
        if len(aspects) < 4:
            return 0.0
        aspect_array = np.asarray(aspects)
        weight_array = np.asarray(weights)
        weighted = float((aspect_array * weight_array).sum() / weight_array.sum())
        return math.log(max(weighted, 1e-6))

    def _ink_mask(self, image: np.ndarray) -> Optional[np.ndarray]:
        gray = to_gray(image)
        height, width = gray.shape[:2]
        longest = max(height, width)
        if longest > self.analysis_size:
            scale = self.analysis_size / float(longest)
            size = (max(1, int(width * scale)), max(1, int(height * scale)))
            gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
        if min(gray.shape[:2]) < 8:
            return None
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return ink


    # --------------------------------------------------------------------- OCR
    def _ocr_refine(
        self,
        image: np.ndarray,
        heuristic: OrientationResult,
        ocr_reader: Callable[[np.ndarray], object],
    ) -> Optional[OrientationResult]:
        base = heuristic.rotation % 180
        candidates = [base, (base + 180) % 360]
        scores: Dict[int, float] = {}
        try:
            for rotation in candidates:
                rotated = rotate_image(image, rotation)
                result = ocr_reader(rotated)
                if result is None:
                    continue
                text = str(getattr(result, "text", "") or "")
                confidence = float(getattr(result, "mean_confidence", 0.0) or 0.0)
                words = len([w for w in text.split() if len(w) > 1])
                if words == 0:
                    scores[rotation] = 0.0
                else:
                    scores[rotation] = confidence * min(1.0, words / 12.0) + 0.001 * min(60, words)
        except Exception:
            return None
        if not scores:
            return None
        best_rotation = max(scores, key=lambda rotation: scores[rotation])
        best_score = scores[best_rotation]
        other = max((value for key, value in scores.items() if key != best_rotation), default=0.0)
        if best_score <= 0.02:
            return None
        confident = best_score >= other * 1.25
        return OrientationResult(
            rotation=int(best_rotation) % 360,
            confident=bool(confident),
            method="ocr",
            score=float(best_score),
            details={f"score_{key}": float(value) for key, value in scores.items()},
        )

