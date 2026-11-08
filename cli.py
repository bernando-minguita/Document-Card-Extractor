"""Headless command line extraction (no GUI).

Examples
--------
Extract every document of a PDF or a scanned image into ``output/``::

    python cli.py scan.pdf
    python cli.py scan.png
    python cli.py scanned_pages.tiff

Only pages 1 and 3, PNG output::

    python cli.py scan.pdf --pages 1,3 --format png --dpi 400

Save debug overlays showing what was detected::

    python cli.py scan.pdf --debug-dir _debug

Check which local OCR engine is ready (no PDF needed)::

    python cli.py --check-ocr
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from core.card_detector import CardDetector, DetectionConfig, candidate_to_card
from core.document_classifier import DocumentClassifier
from core.image_exporter import ExportOptions, ImageExporter, suggest_output_dir
from core.image_preprocessor import resize_max_dim
from core.ocr_engine import OcrEngine
from core.orientation_detector import OrientationDetector
from core.pdf_loader import PdfDocument, PdfError
from core.perspective_corrector import make_thumbnail, rotate_image, warp_quad


def parse_pages(text: str, page_count: int) -> List[int]:
    """Parse ``"1,3,5-7"`` into zero based page indices."""
    if not text:
        return list(range(page_count))
    result: List[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, _, end = chunk.partition("-")
            first = max(1, int(start))
            last = min(page_count, int(end))
            result.extend(range(first - 1, last))
        else:
            page = int(chunk)
            if 1 <= page <= page_count:
                result.append(page - 1)
    return sorted(set(result))


def draw_debug(image: np.ndarray, cards, path: Path) -> None:
    """Write a page image with the detected quads drawn on top."""
    overlay = image.copy()
    colors = [(0, 0, 255), (0, 170, 0), (255, 0, 0), (0, 165, 255), (200, 0, 200), (120, 80, 0)]
    height, width = overlay.shape[:2]
    thickness = max(2, width // 400)
    for index, card in enumerate(cards):
        color = colors[index % len(colors)]
        quad = card.quad_pixels(width, height).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [quad], True, color, thickness)
        label = f"#{index + 1} {card.doc_type.display_name} {card.confidence * 100:.0f}%"
        x = int(card.quad_norm[:, 0].min() * width)
        y = int(card.quad_norm[:, 1].min() * height)
        cv2.putText(overlay, label, (x + 6, max(24, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.5, width / 2400), (0, 0, 0), thickness)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), make_thumbnail(overlay, 1400))


def process_document(path: Path, args) -> int:
    print(f"\n=== {path.name} ===")
    try:
        document = PdfDocument(path)
    except PdfError as exc:
        print(f"  ! {exc}")
        return 0
    except Exception as exc:  # pragma: no cover - unexpected failure
        print(f"  ! {path.name} could not be opened: {exc}")
        return 0

    detector = CardDetector(
        DetectionConfig(
            sensitivity=args.sensitivity,
            min_confidence=args.min_confidence,
            refine_quads=not args.no_refine,
        )
    )
    orientation = OrientationDetector()
    ocr_engine = OcrEngine(tuple(args.ocr_languages.split(","))) if args.ocr else None
    classifier = DocumentClassifier(ocr_engine=ocr_engine) if not args.no_classify else None
    exporter = ImageExporter(
        ExportOptions(
            output_dir=args.output,
            image_format=args.format,
            jpeg_quality=args.quality,
            render_dpi=args.dpi,
            create_subfolder=not args.flat,
        )
    )
    output_dir = suggest_output_dir(path, args.output, not args.flat)

    pages = parse_pages(args.pages, document.page_count)
    total_cards = 0
    if ocr_engine is not None:
        backend = ocr_engine.backend
        if backend:
            if backend == "tesseract":
                from core.ocr_engine import tesseract_language_string

                detail = tesseract_language_string(tuple(args.ocr_languages.split(",")))
            else:
                detail = args.ocr_languages
            print(f"  OCR: {backend} (languages: {detail})")
        else:
            print(
                "  ! OCR is enabled but no OCR engine is available - "
                "run 'python cli.py --check-ocr' for the setup instructions"
            )

    for page_index in pages:
        started = time.perf_counter()
        try:
            dpi = document.effective_dpi(page_index, args.dpi, cap_to_native=not args.no_cap)
            page_image = document.render_page(page_index, dpi=dpi)
        except Exception as exc:
            print(f"  ! page {page_index + 1}: {exc}")
            continue
        working = resize_max_dim(page_image, args.work_dim)
        height, width = working.shape[:2]
        cards = []
        for candidate in detector.detect(working):
            card = candidate_to_card(candidate, width, height, page_index)
            quad = card.quad_pixels(width, height)
            crop = warp_quad(working, quad)
            if not args.no_orient:
                result = orientation.detect(crop)
                card.rotation = result.rotation
                card.orientation_confident = result.confident
            upright = rotate_image(crop, card.rotation) if card.rotation else crop
            text = ""
            if classifier is not None:
                text = ocr_engine.read(upright).text if ocr_engine is not None else ""
                classification = classifier.classify(
                    image=None if text else upright,
                    ocr_text=text,
                    aspect_ratio=card.aspect_ratio(),
                    area_ratio=card.area_ratio(),
                )
                card.set_type(classification.doc_type, classification.confidence)
                card.identifier = classification.identifier
                card.ocr_text = text
            card.thumbnail = make_thumbnail(upright, 400)
            cards.append(card)
            check = "" if card.orientation_confident else "  (verify rotation)"
            ocr_note = ""
            if ocr_engine is not None:
                ocr_note = f" ocr={'text' if text.strip() else 'empty'}"
            print(
                f"  page {page_index + 1}: card #{len(cards)} "
                f"{card.doc_type.display_name:16s} conf={card.confidence:.2f} "
                f"rot={card.rotation:3d}°{check}{ocr_note}"
            )
            if getattr(args, "show_ocr_text", False):
                printed = " ".join((card.ocr_text or "").split())
                print(f"    OCR text: {printed[:400] or '(nothing recognised)'}")
        elapsed = time.perf_counter() - started
        print(f"  page {page_index + 1}: {len(cards)} document(s) in {elapsed:.2f}s")
        if args.debug_dir:
            draw_debug(working, cards, Path(args.debug_dir) / f"{path.stem}_p{page_index + 1:02d}.png")
        if args.preview_dir:
            preview_dir = Path(args.preview_dir)
            preview_dir.mkdir(parents=True, exist_ok=True)
            for index, card in enumerate(cards, start=1):
                if card.thumbnail is not None:
                    name = f"{path.stem}_p{page_index + 1:02d}_{index:02d}.png"
                    cv2.imwrite(str(preview_dir / name), card.thumbnail)

        if cards:
            summary = exporter.export_cards(document, cards, output_dir)
            for item in summary.files:
                if item.ok:
                    print(f"    -> {item.path}")
            for error in summary.errors:
                print(f"    ! {error}")
            total_cards += summary.success_count
        else:
            print("    (no documents detected on this page)")
        del working, page_image

    document.close()
    print(f"  {total_cards} image(s) written to {output_dir}")
    return total_cards


def check_ocr(languages: List[str]) -> int:
    """Print a diagnostic report about the local OCR engines (``--check-ocr``)."""
    wanted = languages or ["en"]
    print("OCR diagnostics")
    report = OcrEngine.diagnostic_report(tuple(wanted))
    for line in report.splitlines():
        print(f"  {line}" if line else "")
    return 0 if OcrEngine(tuple(wanted)).is_available else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract document cards from scanned PDF files and images."
    )
    parser.add_argument("pdfs", nargs="*", help="one or more PDF or image files")
    parser.add_argument(
        "--check-ocr",
        action="store_true",
        help="only report which local OCR engines/models are available, then exit",
    )
    parser.add_argument("--output", "-o", default="output", help="output folder (default: output)")
    parser.add_argument("--pages", default="", help='pages to process, e.g. "1,3,5-7" (default: all)')
    parser.add_argument("--format", "-f", default="jpg", choices=["jpg", "jpeg", "png", "tiff"],
                        help="image format (default: jpg)")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality 60-100 (default: 95)")
    parser.add_argument("--dpi", type=int, default=300, help="render resolution (default: 300)")
    parser.add_argument("--no-cap", action="store_true", help="allow rendering above the native scan resolution")
    parser.add_argument("--flat", action="store_true", help="write all images directly into the output folder")
    parser.add_argument("--sensitivity", type=int, default=5, help="detection sensitivity 1-10 (default: 5)")
    parser.add_argument("--min-confidence", type=float, default=0.42, help="minimum detection confidence")
    parser.add_argument("--work-dim", type=int, default=2200, help="working resolution for detection")
    parser.add_argument("--no-refine", action="store_true", help="disable quad edge refinement")
    parser.add_argument("--no-orient", action="store_true", help="disable automatic orientation detection")
    parser.add_argument("--no-classify", action="store_true", help="disable document classification")
    parser.add_argument("--ocr", action="store_true", help="enable OCR based classification and identifiers")
    parser.add_argument("--ocr-languages", default="en", help="comma separated OCR languages (default: en)")
    parser.add_argument(
        "--show-ocr-text",
        action="store_true",
        help="print the text the OCR engine found for every card (useful to tune keywords)",
    )
    parser.add_argument("--debug-dir", default="", help="write detection overlays into this folder")
    parser.add_argument("--preview-dir", default="", help="write card thumbnails into this folder")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    languages = [item.strip() for item in str(args.ocr_languages or "en").split(",") if item.strip()]
    if args.check_ocr:
        return check_ocr(languages)
    if not args.pdfs:
        build_parser().error("at least one PDF is required (or use --check-ocr)")
    total = 0
    for pdf in args.pdfs:
        path = Path(pdf)
        if not path.exists():
            print(f"! file not found: {path}")
            continue
        total += process_document(path, args)
    print(f"\nDone: {total} document image(s) extracted.")
    return 0 if total else 1


if __name__ == "__main__":
    raise SystemExit(main())


