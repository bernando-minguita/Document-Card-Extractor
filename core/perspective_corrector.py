"""Quad ordering, perspective correction, rotation and thumbnail helpers."""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


def order_quad_points(points: Sequence[Sequence[float]]) -> np.ndarray:
    """Return the four points ordered as top-left, top-right, bottom-right, bottom-left.

    The ordering is derived from the angle around the centroid which keeps the
    winding (and therefore the output image, never mirrored) consistent even for
    cards rotated by 90 or 180 degrees.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    quad = pts[np.argsort(angles)]  # clockwise in image coordinates (y grows down)
    start = int(np.argmin(quad.sum(axis=1)))
    return np.roll(quad, -start, axis=0).astype(np.float32)


def quad_side_lengths(quad: np.ndarray) -> Tuple[float, float, float, float]:
    """Lengths of the top, right, bottom and left sides of *quad*."""
    quad = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    top = float(np.linalg.norm(quad[1] - quad[0]))
    right = float(np.linalg.norm(quad[2] - quad[1]))
    bottom = float(np.linalg.norm(quad[2] - quad[3]))
    left = float(np.linalg.norm(quad[3] - quad[0]))
    return top, right, bottom, left


def quad_size(quad: np.ndarray) -> Tuple[int, int]:
    """Natural output size ``(width, height)`` for a quad."""
    top, right, bottom, left = quad_side_lengths(quad)
    width = int(round(max(top, bottom)))
    height = int(round(max(left, right)))
    return max(width, 2), max(height, 2)


def rotate_quad(quad: np.ndarray, degrees: float, center: Optional[Sequence[float]] = None) -> np.ndarray:
    """Rotate the four corner points of *quad* around its centre (clockwise)."""
    points = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    angle_deg = float(degrees)
    if abs(angle_deg) < 1e-6:
        return points.copy()
    pivot = (
        np.asarray(center, dtype=np.float32).reshape(2)
        if center is not None
        else points.mean(axis=0)
    )
    angle = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    return ((points - pivot) @ rotation.T + pivot).astype(np.float32)


def expand_quad(quad: np.ndarray, ratio: float = 0.01) -> np.ndarray:
    """Grow *quad* around its centroid by *ratio* of its diagonal.

    A small outward padding guarantees that the physical document edge (which is
    a few pixels wide and often slightly darker/rounded) is never cut off.
    """
    quad = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    if ratio <= 0:
        return quad.copy()
    center = quad.mean(axis=0)
    diagonal = float(np.linalg.norm(quad[0] - quad[2]))
    if diagonal <= 1e-6:
        return quad.copy()
    scale = 1.0 + (2.0 * ratio * diagonal) / diagonal
    return ((quad - center) * scale + center).astype(np.float32)


def quad_frame(quad: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(centre, width_axis, height_axis)`` of *quad* as unit vectors.

    Corner 0 is the top-left and the corners run clockwise, so ``corner1 - corner0``
    is the width direction and ``corner3 - corner0`` the height direction.
    """
    points = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    centre = points.mean(axis=0)
    width_vector = points[1] - points[0]
    height_vector = points[3] - points[0]
    width_length = float(np.linalg.norm(width_vector)) or 1.0
    height_length = float(np.linalg.norm(height_vector)) or 1.0
    return centre, width_vector / width_length, height_vector / height_length


def quad_outward_normals(quad: np.ndarray) -> np.ndarray:
    """Unit normal of every edge, pointing away from the centroid.

    Edge ``i`` runs from corner ``i`` to corner ``i + 1``.
    """
    points = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    centre = points.mean(axis=0)
    normals = np.zeros((4, 2), dtype=np.float64)
    for index in range(4):
        start = points[index]
        end = points[(index + 1) % 4]
        edge = end - start
        length = float(np.linalg.norm(edge)) or 1.0
        normal = np.array([edge[1], -edge[0]], dtype=np.float64) / length
        if float(np.dot(normal, (start + end) / 2.0 - centre)) < 0.0:
            normal = -normal
        normals[index] = normal
    return normals


def offset_quad(
    quad: np.ndarray, delta_width: float = 0.0, delta_height: float = 0.0
) -> np.ndarray:
    """Move every side outward by the given amount (negative shrinks).

    ``delta_width`` applies to the left/right sides (edges 1 and 3) and
    ``delta_height`` to the top/bottom sides (edges 0 and 2), so
    ``delta_width=5`` makes the quad 10 units wider - exactly what "include the
    shadow of the card" needs. Corners are joined with a miter, which keeps the
    shape even for a skewed or rotated quad.
    """
    points = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    normals = quad_outward_normals(points)
    distances = np.array(
        [float(delta_height), float(delta_width), float(delta_height), float(delta_width)],
        dtype=np.float64,
    )
    result = points.copy()
    for index in range(4):
        previous = (index - 1) % 4  # edge that ends at this corner
        n_prev = normals[previous]
        n_next = normals[index]
        t_prev = distances[previous]
        t_next = distances[index]
        matrix = np.array([n_prev, n_next], dtype=np.float64)
        try:
            if abs(float(np.linalg.det(matrix))) < 1e-6:
                raise np.linalg.LinAlgError
            shift = np.linalg.solve(matrix, np.array([t_prev, t_next], dtype=np.float64))
        except np.linalg.LinAlgError:
            shift = 0.5 * (t_prev * n_prev + t_next * n_next)
        result[index] = points[index] + shift
    return result.astype(np.float32)


def resize_quad(quad: np.ndarray, scale_x: float = 1.0, scale_y: float = 1.0) -> np.ndarray:
    """Scale *quad* about its centre along its own width and height axes.

    Unlike a plain bounding-box scaling this keeps a rotated or skewed quad
    rotated/skewed, which is what "make the crop a little wider" should mean.
    """
    points = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    centre, width_axis, height_axis = quad_frame(points)
    relative = points - centre
    local_x = relative @ width_axis
    local_y = relative @ height_axis
    scaled = centre + np.outer(local_x * float(scale_x), width_axis) + np.outer(
        local_y * float(scale_y), height_axis
    )
    return scaled.astype(np.float32)


def expand_quad_axes(
    quad: np.ndarray, delta_width: float = 0.0, delta_height: float = 0.0
) -> np.ndarray:
    """Grow (or shrink) *quad* by *delta* on **each** side along its own axes.

    ``delta_width=5`` adds 5 units to the left and 5 to the right (10 in total),
    i.e. it is the natural way to include the shadow around a card.
    """
    points = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    width = float(np.linalg.norm(points[1] - points[0]))
    height = float(np.linalg.norm(points[3] - points[0]))
    minimum = 4.0
    if width + 2.0 * float(delta_width) < minimum:
        delta_width = (minimum - width) / 2.0
    if height + 2.0 * float(delta_height) < minimum:
        delta_height = (minimum - height) / 2.0
    return offset_quad(points, delta_width, delta_height)


def clip_quad(quad: np.ndarray, width: int, height: int) -> np.ndarray:
    quad = np.asarray(quad, dtype=np.float32).reshape(4, 2).copy()
    quad[:, 0] = np.clip(quad[:, 0], 0, width - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, height - 1)
    return quad


def warp_quad(
    image: np.ndarray,
    quad: np.ndarray,
    output_size: Optional[Tuple[int, int]] = None,
    pad_ratio: float = 0.0,
) -> np.ndarray:
    """Perspective-correct the region described by *quad*.

    ``pad_ratio`` grows the quad before sampling; the extra pixels are taken
    from the surrounding page (``BORDER_REPLICATE``) so no valid content is lost.
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot warp an empty image.")
    height, width = image.shape[:2]
    src = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    if pad_ratio:
        src = expand_quad(src, pad_ratio)
    src = clip_quad(src, width, height)

    out_w, out_h = output_size if output_size else quad_size(src)
    out_w = int(max(2, min(out_w, 20000)))
    out_h = int(max(2, min(out_h, 20000)))
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        image, matrix, (out_w, out_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def rotate_image(image: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate *image* clockwise by 0, 90, 180 or 270 degrees (lossless)."""
    degrees = int(degrees) % 360
    if degrees == 0:
        return image
    if degrees == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError("Rotation must be one of 0, 90, 180 or 270 degrees.")


def make_thumbnail(image: np.ndarray, max_size: int = 320) -> np.ndarray:
    """Downscale *image* so its longest side is at most *max_size* pixels."""
    if image is None or image.size == 0:
        raise ValueError("Cannot build a thumbnail of an empty image.")
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_size:
        return image.copy()
    scale = max_size / float(longest)
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def quad_iou(quad_a: np.ndarray, quad_b: np.ndarray) -> float:
    """Intersection over union of two convex quads (angle independent)."""
    a = np.asarray(quad_a, dtype=np.float32).reshape(-1, 1, 2)
    b = np.asarray(quad_b, dtype=np.float32).reshape(-1, 1, 2)
    area_a = abs(float(cv2.contourArea(a)))
    area_b = abs(float(cv2.contourArea(b)))
    if area_a <= 0 or area_b <= 0:
        return 0.0
    try:
        intersection, _ = cv2.intersectConvexConvex(a, b)
        intersection = abs(float(intersection))
    except cv2.error:
        return 0.0
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def containment_ratio(inner: np.ndarray, outer: np.ndarray) -> float:
    """How much of *inner* is covered by *outer* (``0..1``)."""
    inner_c = np.asarray(inner, dtype=np.float32).reshape(-1, 1, 2)
    outer_c = np.asarray(outer, dtype=np.float32).reshape(-1, 1, 2)
    area_inner = abs(float(cv2.contourArea(inner_c)))
    if area_inner <= 0:
        return 0.0
    try:
        intersection, _ = cv2.intersectConvexConvex(inner_c, outer_c)
    except cv2.error:
        return 0.0
    return float(min(1.0, abs(float(intersection)) / area_inner))

