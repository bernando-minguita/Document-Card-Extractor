"""Multi-strategy document card detector.

Several independent binarisation/edge strategies are run on every page.  Each
strategy produces rectangular candidates which are validated (area, aspect,
rectangularity, edge support), refined against the real image gradients and
finally merged, removing duplicates and nested rectangles (internal document
borders) using intersection-over-union and containment tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .image_preprocessor import MaskBundle, PreprocessConfig, build_masks
from .models import DetectedCard
from .perspective_corrector import (
    clip_quad,
    containment_ratio,
    expand_quad,
    order_quad_points,
    quad_iou,
    quad_size,
)


@dataclass
class DetectionConfig:
    """Configuration for :class:`CardDetector`."""

    sensitivity: int = 5
    min_area_ratio: float = 0.008
    max_area_ratio: float = 0.92
    min_aspect: float = 1.02
    max_aspect: float = 4.5
    min_confidence: float = 0.42
    min_rectangularity: float = 0.72
    min_extent: float = 0.55
    min_side_px: int = 90
    max_cards: int = 16
    merge_iou: float = 0.55
    containment_drop: float = 0.85
    padding_ratio: float = 0.006
    refine_quads: bool = True
    denoise_level: int = 1

    @classmethod
    def from_settings(cls, settings) -> "DetectionConfig":
        """Build a config from the application settings object."""
        return cls(
            sensitivity=int(getattr(settings, "detection_sensitivity", 5)),
            min_area_ratio=float(getattr(settings, "min_document_area", 0.008)),
            max_area_ratio=float(getattr(settings, "max_document_area", 0.92)),
            min_confidence=float(getattr(settings, "detection_min_confidence", 0.42)),
            refine_quads=bool(getattr(settings, "refine_quads", True)),
        )

    def preprocess_config(self) -> PreprocessConfig:
        return PreprocessConfig(sensitivity=self.sensitivity, denoise_level=self.denoise_level)


@dataclass
class Candidate:
    """A detected rectangular region in pixel coordinates."""

    quad: np.ndarray
    confidence: float
    source: str
    area_ratio: float = 0.0
    aspect_ratio: float = 1.0
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def center(self) -> Tuple[float, float]:
        return float(self.quad[:, 0].mean()), float(self.quad[:, 1].mean())

    @property
    def area(self) -> float:
        return abs(float(cv2.contourArea(self.quad.astype(np.float32))))


def candidate_to_card(
    candidate: Candidate,
    image_width: int,
    image_height: int,
    page_index: int,
    padding_ratio: float = 0.006,
) -> DetectedCard:
    """Convert a pixel-space :class:`Candidate` into a normalised card."""
    quad = candidate.quad
    if padding_ratio:
        quad = expand_quad(quad, padding_ratio)
    quad = clip_quad(quad, int(image_width), int(image_height))
    normalised = quad.copy()
    normalised[:, 0] = normalised[:, 0] / float(max(1, image_width))
    normalised[:, 1] = normalised[:, 1] / float(max(1, image_height))
    return DetectedCard(
        page_index=int(page_index),
        quad_norm=normalised,
        confidence=float(candidate.confidence),
        source=candidate.source,
        page_size=(int(image_width), int(image_height)),
    )


class CardDetector:
    """Detects document cards on a rendered PDF page."""

    def __init__(self, config: Optional[DetectionConfig] = None) -> None:
        self.config = config or DetectionConfig()

    # ------------------------------------------------------------------ public
    def detect(self, image: np.ndarray) -> List[Candidate]:
        """Return the validated, de-duplicated card candidates of *image*."""
        if image is None or image.size == 0:
            return []
        height, width = image.shape[:2]
        page_area = float(height * width)

        bundle = build_masks(image, self.config.preprocess_config())
        candidates: List[Candidate] = []
        for name, mask, weight in bundle.masks:
            candidates.extend(self._candidates_from_mask(name, mask, weight, bundle, page_area))

        kept = self._suppress(self._expand_merged(candidates))
        kept = [c for c in kept if c.confidence >= self.config.min_confidence]
        kept.sort(key=lambda c: c.confidence, reverse=True)
        kept = kept[: self.config.max_cards]
        if self.config.refine_quads:
            for candidate in kept:
                candidate.quad = self._refine_quad(bundle.gradient, candidate.quad)
        return self._sort_reading_order(kept)

    # ------------------------------------------------------------------ stages
    def _candidates_from_mask(
        self,
        name: str,
        mask: np.ndarray,
        weight: float,
        bundle: MaskBundle,
        page_area: float,
    ) -> List[Candidate]:
        height, width = mask.shape[:2]
        min_area = self.config.min_area_ratio * page_area
        max_area = self.config.max_area_ratio * page_area

        # A 2px border keeps contours of the page itself out of the results.
        work = mask.copy()
        work[:2, :] = 0
        work[-2:, :] = 0
        work[:, :2] = 0
        work[:, -2:] = 0

        contours, _ = cv2.findContours(work, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        results: List[Candidate] = []
        for contour in contours:
            area = abs(float(cv2.contourArea(contour)))
            if area < min_area or area > max_area:
                continue
            quad = self._quad_from_contour(contour)
            if quad is None:
                continue
            scored = self._score_candidate(quad, area, page_area, bundle, weight, name)
            if scored is not None:
                results.append(scored)
        return results

    def _quad_from_contour(self, contour: np.ndarray) -> Optional[np.ndarray]:
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            return None
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = approx.reshape(4, 2).astype(np.float32)
        else:
            rect = cv2.minAreaRect(contour)
            side_a, side_b = rect[1]
            if min(side_a, side_b) < self.config.min_side_px:
                return None
            quad = cv2.boxPoints(rect).astype(np.float32)
        if not _is_convex_quad(quad):
            return None
        return order_quad_points(quad)

    def _score_candidate(
        self,
        quad: np.ndarray,
        contour_area: float,
        page_area: float,
        bundle: MaskBundle,
        weight: float,
        source: str,
    ) -> Optional[Candidate]:
        """Validate a quad and compute its confidence score (``0..1``)."""
        height, width = bundle.gradient.shape[:2]
        quad_w, quad_h = quad_size(quad)
        if min(quad_w, quad_h) < self.config.min_side_px:
            return None

        quad_area = abs(float(cv2.contourArea(quad.astype(np.float32))))
        if quad_area <= 1:
            return None
        area_ratio = quad_area / page_area
        if area_ratio < self.config.min_area_ratio or area_ratio > self.config.max_area_ratio:
            return None

        long_side = max(quad_w, quad_h)
        short_side = max(1, min(quad_w, quad_h))
        aspect = long_side / short_side
        if aspect < self.config.min_aspect or aspect > self.config.max_aspect:
            return None

        rectangularity = min(1.0, contour_area / quad_area)
        if rectangularity < self.config.min_rectangularity:
            return None

        xs, ys = quad[:, 0], quad[:, 1]
        bbox_w = float(xs.max() - xs.min())
        bbox_h = float(ys.max() - ys.min())
        if bbox_w <= 1 or bbox_h <= 1:
            return None
        extent = min(1.0, contour_area / (bbox_w * bbox_h))
        if extent < self.config.min_extent:
            return None

        # Reject the page itself (or a frameless full-page scan).
        if bbox_w > 0.94 * width and bbox_h > 0.94 * height:
            return None
        if (
            xs.min() <= 0.01 * width
            and ys.min() <= 0.01 * height
            and xs.max() >= 0.99 * width
            and ys.max() >= 0.99 * height
        ):
            return None

        edge_support = _edge_support(bundle.gradient, quad)
        area_score = _area_plausibility(area_ratio)
        aspect_score = _aspect_plausibility(aspect)
        rectangularity_score = _ramp(rectangularity, self.config.min_rectangularity, 0.97)
        extent_score = _ramp(extent, self.config.min_extent, 0.96)

        score = (
            0.30 * edge_support
            + 0.20 * rectangularity_score
            + 0.14 * extent_score
            + 0.18 * area_score
            + 0.18 * aspect_score
        )
        score *= max(0.6, min(1.0, weight))

        if score < self.config.min_confidence:
            return None
        return Candidate(
            quad=quad,
            confidence=float(min(1.0, score)),
            source=source,
            area_ratio=area_ratio,
            aspect_ratio=aspect,
            metrics={
                "rectangularity": rectangularity,
                "extent": extent,
                "edge_support": edge_support,
                "aspect": aspect,
                "area_ratio": area_ratio,
            },
        )

    # ------------------------------------------------------------------ refine
    def _refine_quad(self, gradient: np.ndarray, quad: np.ndarray) -> np.ndarray:
        """Snap the quad sides onto the strongest nearby image gradients.

        Only accepted when the result stays convex and its area does not change
        much, so a bad refinement can never destroy an otherwise good detection.
        """
        height, width = gradient.shape[:2]
        diagonal = float(np.hypot(width, height))
        max_shift = max(2, int(diagonal * 0.012))
        lines: List[Tuple[np.ndarray, np.ndarray, float]] = []

        for index in range(4):
            p1 = quad[index].astype(np.float64)
            p2 = quad[(index + 1) % 4].astype(np.float64)
            edge = p2 - p1
            length = float(np.linalg.norm(edge))
            if length < 20:
                return quad
            direction = edge / length
            normal = np.array([-direction[1], direction[0]], dtype=np.float64)
            offsets = np.arange(-max_shift, max_shift + 0.5, 1.0)
            samples: List[np.ndarray] = []
            for t in np.linspace(0.12, 0.88, 24):
                base = p1 + direction * (t * length)
                best_value = 0.0
                best_offset = 0.0
                for offset in offsets:
                    point = base + normal * offset
                    x = int(round(point[0]))
                    y = int(round(point[1]))
                    if x < 1 or y < 1 or x >= width - 1 or y >= height - 1:
                        continue
                    value = float(gradient[y - 1 : y + 2, x - 1 : x + 2].max())
                    if value > best_value:
                        best_value = value
                        best_offset = float(offset)
                if best_value >= 0.30:
                    samples.append(base + normal * best_offset)
            if len(samples) < 8:
                return quad
            points = np.asarray(samples)
            line_offset = float(np.mean(points @ normal))
            lines.append((normal, direction, line_offset))

        refined = []
        for index in range(4):
            normal_a, _, offset_a = lines[index]
            normal_b, _, offset_b = lines[(index + 1) % 4]
            matrix = np.array([normal_a, normal_b], dtype=np.float64)
            if abs(np.linalg.det(matrix)) < 1e-6:
                return quad
            corner = np.linalg.solve(matrix, np.array([offset_a, offset_b], dtype=np.float64))
            refined.append(corner)
        refined_quad = np.asarray(refined, dtype=np.float32)

        if not _is_convex_quad(refined_quad):
            return quad
        original_area = abs(float(cv2.contourArea(quad.astype(np.float32))))
        refined_area = abs(float(cv2.contourArea(refined_quad.astype(np.float32))))
        if original_area <= 0 or abs(refined_area - original_area) / original_area > 0.15:
            return quad
        if np.max(np.linalg.norm(refined_quad - quad, axis=1)) > 3.0 * max_shift:
            return quad
        return order_quad_points(refined_quad)

    # ------------------------------------------------------------------ merging
    def _expand_merged(self, candidates: Sequence[Candidate]) -> List[Candidate]:
        """Replace a region that merged two neighbouring cards by the two cards.

        Shadows or a very small gap between two cards can join them into a single
        blob.  When two disjoint, individually valid detections together cover
        most of such a region, the region is replaced by them - that way a
        document is never lost because it touched its neighbour.
        """
        ordered = sorted(candidates, key=lambda c: c.area, reverse=True)
        removed: set = set()
        expanded: List[Candidate] = []
        for parent in ordered:
            if id(parent) in removed:
                continue
            parent_area = parent.area
            if parent_area <= 0:
                expanded.append(parent)
                continue
            children = [
                cand
                for cand in candidates
                if cand is not parent
                and cand.area < 0.75 * parent_area
                and cand.confidence >= self.config.min_confidence * 0.8
                and containment_ratio(cand.quad, parent.quad) >= 0.90
            ]
            best = None
            best_score = -1.0
            for index, first in enumerate(children):
                for second in children[index + 1 :]:
                    if quad_iou(first.quad, second.quad) >= 0.12:
                        continue
                    coverage = (first.area + second.area) / parent_area
                    if coverage < 0.68:
                        continue
                    score = first.confidence + second.confidence + coverage
                    if score > best_score:
                        best_score = score
                        best = (first, second)
            if best is None:
                expanded.append(parent)
            else:
                removed.add(id(parent))
                expanded.extend(best)
        return expanded

    def _suppress(self, candidates: Sequence[Candidate]) -> List[Candidate]:
        """Resolve duplicates and nested rectangles.

        A document typically produces several nested candidates: the physical
        card edge, a printed frame around the text and the photo block.  The
        outermost convincing rectangle wins, *unless* that rectangle is really
        two documents that touched (then most of its area is covered by two
        disjoint inner detections, and the inner ones are kept).
        """
        ordered = sorted(candidates, key=lambda c: c.confidence, reverse=True)
        kept: List[Candidate] = []
        for candidate in ordered:
            drop = False
            superseded: List[int] = []
            for index, existing in enumerate(kept):
                if quad_iou(candidate.quad, existing.quad) >= self.config.merge_iou:
                    drop = True
                    break
                inside = containment_ratio(candidate.quad, existing.quad)
                if inside >= self.config.containment_drop and candidate.area < existing.area * 0.95:
                    drop = True  # candidate is an internal frame / photo of `existing`
                    break
                contains = containment_ratio(existing.quad, candidate.quad)
                if contains >= self.config.containment_drop and existing.area < candidate.area * 0.95:
                    superseded.append(index)
            if not drop and superseded:
                if self._outer_looks_like_card(candidate, [kept[index] for index in superseded]):
                    for index in reversed(superseded):
                        kept.pop(index)
                else:
                    drop = True
            if not drop:
                kept.append(candidate)
        return kept

    def _outer_looks_like_card(self, outer: Candidate, inner: Sequence[Candidate]) -> bool:
        """Decide whether an outer rectangle is a card boundary or a merged group."""
        if outer.area_ratio > 0.55:
            return False  # a document never covers more than half of a page
        if outer.metrics.get("rectangularity", 0.0) < 0.88:
            return False
        if outer.metrics.get("extent", 0.0) < 0.85:
            return False
        # When two disjoint inner detections already fill most of the outer
        # rectangle, the outer rectangle is a merged group of documents.
        coverage = 0.0
        accepted: List[Candidate] = []
        for candidate in sorted(inner, key=lambda c: c.area, reverse=True):
            if candidate.area <= 0:
                continue
            overlapping = any(
                quad_iou(candidate.quad, other.quad) > 0.12
                or containment_ratio(candidate.quad, other.quad) > 0.90
                or containment_ratio(other.quad, candidate.quad) > 0.90
                for other in accepted
            )
            if overlapping:
                continue
            accepted.append(candidate)
            coverage += candidate.area
        outer_area = max(outer.area, 1.0)
        return (coverage / outer_area) < 0.68



    def _sort_reading_order(self, candidates: Sequence[Candidate]) -> List[Candidate]:
        """Sort top-to-bottom, left-to-right so filenames follow the page layout."""
        if not candidates:
            return []
        ordered = sorted(candidates, key=lambda c: c.center[1])
        heights = [max(1.0, float(np.ptp(c.quad[:, 1]))) for c in ordered]
        tolerance = 0.45 * float(np.median(heights))
        rows: List[List[Candidate]] = []
        for candidate in ordered:
            if rows and abs(candidate.center[1] - rows[-1][0].center[1]) <= tolerance:
                rows[-1].append(candidate)
            else:
                rows.append([candidate])
        result: List[Candidate] = []
        for row in rows:
            result.extend(sorted(row, key=lambda c: c.center[0]))
        return result


def _ramp(value: float, low: float, high: float) -> float:
    """Linear ramp: ``0`` at *low*, ``1`` at *high* (clamped)."""
    if high <= low:
        return 0.0
    return float(min(1.0, max(0.0, (value - low) / (high - low))))


_ASPECT_NODES = (1.00, 1.15, 1.30, 1.85, 2.60, 3.20, 4.50)
_ASPECT_VALUES = (0.45, 0.72, 1.00, 1.00, 0.80, 0.62, 0.45)


def _aspect_plausibility(aspect: float) -> float:
    """Soft preference for card-like aspect ratios (never a hard filter)."""
    return float(np.interp(aspect, _ASPECT_NODES, _ASPECT_VALUES))


def _area_plausibility(area_ratio: float) -> float:
    """Prefer cards that cover a sensible part of the page."""
    if area_ratio < 0.012:
        return _ramp(area_ratio, 0.004, 0.012) * 0.85
    if area_ratio > 0.62:
        return 1.0 - 0.55 * _ramp(area_ratio, 0.62, 0.92)
    return 1.0


def _is_convex_quad(points: np.ndarray) -> bool:
    pts = np.asarray(points, dtype=np.float64).reshape(4, 2)
    if not np.all(np.isfinite(pts)):
        return False
    signs = []
    for index in range(4):
        a = pts[index]
        b = pts[(index + 1) % 4]
        c = pts[(index + 2) % 4]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        signs.append(cross)
    signs_arr = np.asarray(signs)
    if np.any(np.abs(signs_arr) < 1e-6):
        return False
    return bool(np.all(signs_arr > 0) or np.all(signs_arr < 0))


def _edge_support(gradient: np.ndarray, quad: np.ndarray, samples_per_side: int = 28) -> float:
    """Fraction of the quad perimeter that sits on a strong image gradient."""
    if gradient is None or gradient.size == 0:
        return 0.0
    height, width = gradient.shape[:2]
    values: List[float] = []
    for index in range(4):
        p1 = np.asarray(quad[index], dtype=np.float64)
        p2 = np.asarray(quad[(index + 1) % 4], dtype=np.float64)
        for t in np.linspace(0.03, 0.97, samples_per_side):
            point = p1 + (p2 - p1) * t
            x = int(round(point[0]))
            y = int(round(point[1]))
            if x < 1 or y < 1 or x >= width - 1 or y >= height - 1:
                continue
            values.append(float(gradient[y - 1 : y + 2, x - 1 : x + 2].max()))
    if not values:
        return 0.0
    return float(min(1.0, np.mean(values) / 0.55))


