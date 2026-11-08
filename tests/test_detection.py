"""Detection engine tests against the bundled sample scan."""

from __future__ import annotations

import cv2
import numpy as np

from core.card_detector import CardDetector, DetectionConfig, candidate_to_card
from core.image_preprocessor import resize_max_dim
from core.pdf_loader import PdfDocument
from core.perspective_corrector import warp_quad


def _detect_sample(sample_pdf, sensitivity: int = 5):
    with PdfDocument(sample_pdf) as document:
        dpi = document.effective_dpi(0, 300)
        page = document.render_page(0, dpi=dpi)
    working = resize_max_dim(page, 2200)
    detector = CardDetector(DetectionConfig(sensitivity=sensitivity))
    candidates = detector.detect(working)
    cards = [
        candidate_to_card(candidate, working.shape[1], working.shape[0], 0)
        for candidate in candidates
    ]
    return working, cards


def test_detects_documents_on_an_image_file(tmp_path):
    """Scans often arrive as images: detection must work on a PNG/TIFF too."""
    from PIL import Image, ImageDraw

    path = tmp_path / "scan.png"
    image = Image.new("RGB", (1600, 1100), (250, 250, 250))
    draw = ImageDraw.Draw(image)
    for left in (120, 860):
        draw.rectangle([left, 220, left + 620, 220 + 400], fill=(203, 210, 222), outline=(110, 120, 135), width=4)
        draw.rectangle([left + 30, 260, left + 200, 380], fill=(230, 234, 240))  # photo block
    image.save(path, dpi=(300, 300))

    with PdfDocument(path) as document:
        assert document.is_image is True and document.page_count == 1
        page = document.render_page(0, dpi=document.effective_dpi(0, 300))

    working = resize_max_dim(page, 2200)
    detector = CardDetector(DetectionConfig(sensitivity=5))
    cards = [
        candidate_to_card(candidate, working.shape[1], working.shape[0], 0)
        for candidate in detector.detect(working)
    ]
    assert len(cards) == 2, f"expected 2 documents on the image, detected {len(cards)}"
    for card in cards:
        assert 0.05 < card.quad_norm.min() and card.quad_norm.max() < 0.98


def test_sample_pdf_is_readable(sample_pdf):
    with PdfDocument(sample_pdf) as document:
        assert document.page_count >= 1
        info = document.page_info(0)
        assert info.width_pt > 100 and info.height_pt > 100
        image = document.render_page(0, dpi=100)
        assert image.ndim == 3 and image.shape[2] == 3
        assert image.mean() > 120  # a light scan, not a black page


def test_detects_all_four_documents(sample_pdf):
    _image, cards = _detect_sample(sample_pdf)
    assert len(cards) == 4, f"expected 4 documents, detected {len(cards)}"

    # The open passport is the large landscape spread at the top of the page.
    largest = max(cards, key=lambda card: card.area_ratio())
    assert 0.2 < largest.area_ratio() < 0.5
    assert 1.25 < largest.aspect_ratio() < 1.75
    assert largest.quad_norm[:, 1].mean() < 0.6  # upper part of the page

    # The three cards sit side by side in the lower half.
    lower = [card for card in cards if card is not largest]
    assert len(lower) == 3
    for card in lower:
        assert card.quad_norm[:, 1].mean() > 0.55
        assert 0.03 < card.area_ratio() < 0.2
        assert 1.3 < card.aspect_ratio() < 1.9
        assert card.confidence >= 0.4

    # Reading order: left to right within the bottom row.
    ordered = sorted(lower, key=lambda card: card.quad_norm[:, 0].mean())
    xs = [card.quad_norm[:, 0].mean() for card in ordered]
    assert xs == sorted(xs)
    assert xs[1] - xs[0] > 0.1 and xs[2] - xs[1] > 0.1


def test_detected_quads_stay_inside_the_page(sample_pdf):
    _image, cards = _detect_sample(sample_pdf)
    for card in cards:
        assert np.all(card.quad_norm >= -0.01)
        assert np.all(card.quad_norm <= 1.01)


def test_crops_are_not_clipped(sample_pdf):
    """Every crop must contain the whole card, i.e. reach the card's edge."""
    image, cards = _detect_sample(sample_pdf)
    for card in cards:
        height, width = image.shape[:2]
        quad = card.quad_pixels(width, height)
        crop = warp_quad(image, quad)
        assert crop.shape[0] > 100 and crop.shape[1] > 100
        # A clipped crop would be mostly platen in the outer ring; the true card
        # edge keeps the border of the crop close in brightness to the centre.
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        border = np.concatenate(
            [gray[:3, :].ravel(), gray[-3:, :].ravel(), gray[:, :3].ravel(), gray[:, -3:].ravel()]
        )
        centre = gray[gray.shape[0] // 3 : 2 * gray.shape[0] // 3, gray.shape[1] // 3 : 2 * gray.shape[1] // 3]
        assert abs(float(border.mean()) - float(centre.mean())) < 60


def test_no_duplicates_between_cards(sample_pdf):
    from core.perspective_corrector import quad_iou

    _image, cards = _detect_sample(sample_pdf)
    for index, first in enumerate(cards):
        for second in cards[index + 1 :]:
            quad_a = first.quad_pixels(1000, 1400)
            quad_b = second.quad_pixels(1000, 1400)
            assert quad_iou(quad_a, quad_b) < 0.1


def test_blank_page_yields_no_documents():
    blank = np.full((900, 700, 3), 235, dtype=np.uint8)
    detector = CardDetector()
    assert detector.detect(blank) == []


def test_synthetic_card_is_detected():
    page = np.full((1000, 800, 3), 200, dtype=np.uint8)
    page[200:420, 120:600] = (245, 245, 245)  # a light card
    cv2.rectangle(page, (120, 200), (599, 419), (60, 60, 60), 3)
    detector = CardDetector()
    candidates = detector.detect(page)
    assert len(candidates) == 1
    quad = candidates[0].quad
    xs, ys = quad[:, 0], quad[:, 1]
    assert abs(float(xs.min()) - 120) < 18
    assert abs(float(xs.max()) - 599) < 18
    assert abs(float(ys.min()) - 200) < 18
    assert abs(float(ys.max()) - 419) < 18
