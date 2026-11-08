"""Advisory document classification.

Classification combines OCR keywords (when OCR is enabled) with layout
heuristics (aspect ratio / relative size).  It is deliberately *advisory*: when
the evidence is weak the card is labelled ``Unknown`` and the user can pick the
type manually.  A card is never rejected because classification failed.
"""

from __future__ import annotations

import difflib
import itertools
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .models import DocumentType
from .ocr_engine import OcrEngine

# Keyword -> (type, weight).  Matching is done on an upper-cased, normalised text.
#
# Notes on this table (so it is not "cleaned up" by mistake):
# * Several keywords are deliberate prefixes of others ("PASSPORT" /
#   "PASSPORT NO", "SALARY" / "SALARY CARD", "REGISTRATION" /
#   "VEHICLE REGISTRATION", "RESIDENT" / "RESIDENT IDENTITY").  A hit is plain
#   substring containment, so the longer phrase also matches the shorter keyword
#   and both weights add up - for the *same* document type, which is intended
#   (more evidence = more confidence).
# * "DRIVING LICENSE" and "DRIVING LICENCE" are both listed on purpose: the exact
#   spelling scores 1.0 instead of relying on the fuzzy match (~0.86).
# * Every keyword must point at a real document type.  "Unknown" is the fallback
#   and a rule for it would never be scored, so generic phrases such as
#   "KINGDOM OF SAUDI ARABIA" or "MINISTRY OF INTERIOR" are deliberately absent
#   (they appear on almost every Saudi document and carry no information).
# * There are no exact duplicates - tests/test_units.py asserts that.
KEYWORD_RULES: Sequence[Tuple[str, DocumentType, float]] = (
    ("PASSPORT", DocumentType.PASSPORT, 1.0),
    ("PASSPORT NO", DocumentType.PASSPORT, 0.8),
    # ("REPUBLIC OF", DocumentType.PASSPORT, 0.5),
    # ("IMMIGRATION", DocumentType.PASSPORT, 0.6),
    ("RESIDENT IDENTITY", DocumentType.IQAMA, 1.0),
    ("RESIDENT PERMIT", DocumentType.IQAMA, 0.8),
    ("IQAMA", DocumentType.IQAMA, 0.9),
    ("RESIDENT", DocumentType.IQAMA, 0.4),
    ("DRIVING LICENSE", DocumentType.DRIVING_LICENSE, 1.0),
    ("DRIVING LICENCE", DocumentType.DRIVING_LICENSE, 1.0),
    ("DRIVER LICENSE", DocumentType.DRIVING_LICENSE, 0.9),
    ("VEHICLE REGISTRATION", DocumentType.ISTIMARA, 1.0),
    ("ISTIMARA", DocumentType.ISTIMARA, 0.9),
    ("REGISTRATION", DocumentType.ISTIMARA, 0.35),
    ("HEALTH INSURANCE", DocumentType.HEALTH_INSURANCE, 1.0),
    ("MEDICAL INSURANCE", DocumentType.HEALTH_INSURANCE, 0.9),
    ("INSURANCE CARD", DocumentType.HEALTH_INSURANCE, 0.8),
    ("INSURANCE POLICY", DocumentType.HEALTH_INSURANCE, 0.6),
    ("BANK CARD", DocumentType.BANK_CARD, 1.0),
    ("DEBIT CARD", DocumentType.BANK_CARD, 0.9),
    ("CREDIT CARD", DocumentType.BANK_CARD, 0.9),
    ("ATM CARD", DocumentType.BANK_CARD, 0.8),
    ("SALARY CARD", DocumentType.SALARY_CARD, 1.0),
    ("SALARY", DocumentType.SALARY_CARD, 0.4),
    ("NATIONAL ID", DocumentType.NATIONAL_ID, 1.0),
    ("NATIONAL IDENTITY", DocumentType.NATIONAL_ID, 0.9),
    ("IDENTITY CARD", DocumentType.NATIONAL_ID, 0.7),
)

# Arabic spellings of the same concepts.  Kept separate so the table above stays
# easy to compare with the English document texts.
_ARABIC_RULES: Sequence[Tuple[str, DocumentType, float]] = (
    ("هوية مقيم", DocumentType.IQAMA, 1.0),
    ("هوية وطنية", DocumentType.NATIONAL_ID, 1.0),
    ("رخصة قيادة", DocumentType.DRIVING_LICENSE, 1.0),
    ("رخصة سير", DocumentType.ISTIMARA, 1.0),
    ("استمارة", DocumentType.ISTIMARA, 0.8),
    ("جواز سفر", DocumentType.PASSPORT, 1.0),
    ("تأمين صحي", DocumentType.HEALTH_INSURANCE, 1.0),
    ("بطاقة تأمين", DocumentType.HEALTH_INSURANCE, 0.8),
    ("بطاقة بنكية", DocumentType.BANK_CARD, 1.0),
    ("بطاقة راتب", DocumentType.SALARY_CARD, 1.0),
)

MRZ_PATTERN = re.compile(r"P[<A-Z]{2}[A-Z<]{2,}")
SAUDI_ID_PATTERN = re.compile(r"(?<!\d)([12]\d{9})(?!\d)")
PASSPORT_NUMBER_PATTERN = re.compile(r"(?<![A-Z0-9])([A-Z]{1,2}\d{7,8})(?![A-Z0-9])")
GENERIC_NUMBER_PATTERN = re.compile(r"(?<!\d)(\d{6,12})(?!\d)")

# Characters OCR engines confuse with each other; normalising them before the
# keyword match makes the classification much more forgiving.
_OCR_CONFUSIONS = str.maketrans(
    {
        "0": "O",
        "1": "I",
        "2": "Z",
        "5": "S",
        "6": "G",
        "8": "B",
        "@": "A",
        "$": "S",
        "!": "I",
        "|": "I",
        "(": "C",
        ")": " ",
        "[": " ",
        "]": " ",
        "{": " ",
        "}": " ",
        "<": " ",
        ">": " ",
        "*": " ",
        "#": " ",
        '"': " ",
        "'": " ",
        "`": " ",
        "-": " ",
        "_": " ",
        ".": " ",
        ",": " ",
        ";": " ",
        ":": " ",
    }
)
_NON_WORD = re.compile(r"[^A-ZА-Я0-9\u0600-\u06FF]+")

FUZZY_THRESHOLD = 0.82


def prepare_ocr_text(text: str) -> str:
    """Upper-case, strip punctuation and fix typical OCR confusions."""
    if not text:
        return ""
    cleaned = str(text).upper().translate(_OCR_CONFUSIONS)
    cleaned = _NON_WORD.sub(" ", cleaned)
    return " ".join(cleaned.split())


def _similar(a: str, b: str) -> float:
    if a == b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def keyword_score(prepared_text: str, keyword: str, threshold: float = FUZZY_THRESHOLD) -> float:
    """How well *keyword* occurs in *prepared_text*.

    ``1.0`` for an exact phrase.  Otherwise the keyword is aligned against word
    windows of the text (allowing one stray token in between, because OCR often
    inserts or splits words) and the best similarity is returned; 0.0 when it
    stays below *threshold*.  This tolerates mistakes such as ``REBISTAATION``
    for ``REGISTRATION`` or a ``$`` read as a word between two words.
    """
    keyword_clean = prepare_ocr_text(keyword)
    if not keyword_clean or not prepared_text:
        return 0.0
    if keyword_clean in prepared_text:
        return 1.0

    text_tokens = prepared_text.split()
    keyword_tokens = keyword_clean.split()
    span = len(keyword_tokens)
    if span == 0 or len(text_tokens) < span:
        return 0.0

    best = 0.0
    # A window one token longer than the keyword lets us skip a stray token.
    window_size = min(span + 1, len(text_tokens))
    for start in range(len(text_tokens) - window_size + 1):
        window = text_tokens[start : start + window_size]
        for positions in itertools.combinations(range(window_size), span):
            score = sum(
                _similar(window[position], key) for position, key in zip(positions, keyword_tokens)
            ) / span
            if score > best:
                best = score
                if best >= 0.995:
                    break
        if best >= 0.995:
            break
    if span == len(text_tokens):  # very short texts: also try a direct alignment
        best = max(
            best,
            sum(_similar(word, key) for word, key in zip(text_tokens, keyword_tokens)) / span,
        )
    return best if best >= threshold else 0.0


def builtin_keywords(doc_type: DocumentType) -> List[str]:
    """Keywords the classifier knows for *doc_type* without user changes."""
    if doc_type is None or doc_type is DocumentType.UNKNOWN:
        return []  # "Unknown" is the fallback and has no keywords
    found: List[str] = []
    for keyword, rule_type, _weight in tuple(KEYWORD_RULES) + tuple(_ARABIC_RULES):
        if rule_type is doc_type and keyword not in found:
            found.append(keyword)
    for keyword in doc_type.keywords:
        text = str(keyword)
        if text not in found:
            found.append(text)
    return found


@dataclass
class ClassificationResult:
    """Result of classifying a single card."""

    doc_type: DocumentType = DocumentType.UNKNOWN
    confidence: float = 0.0
    reason: str = ""
    identifier: str = ""
    ocr_text: str = ""


class DocumentClassifier:
    """Classify extracted cards into one of the supported document types."""

    def __init__(self, ocr_engine: Optional[OcrEngine] = None, min_confidence: float = 0.35) -> None:
        self.ocr_engine = ocr_engine
        self.min_confidence = float(min_confidence)

    # ------------------------------------------------------------------ public
    def classify(
        self,
        image: Optional[np.ndarray] = None,
        ocr_text: str = "",
        aspect_ratio: float = 0.0,
        area_ratio: float = 0.0,
    ) -> ClassificationResult:
        """Classify a card.

        ``ocr_text`` may be supplied by the caller (already computed during
        orientation detection) to avoid running OCR twice.
        """
        text = (ocr_text or "").upper()
        if not text and image is not None and self.ocr_engine is not None:
            result = self.ocr_engine.read(image)
            text = (result.text or "").upper()
        prepared = prepare_ocr_text(text)
        scores = self._keyword_scores(prepared, raw_text=text)
        doc_type, confidence, reason = self._best_type(scores)
        if doc_type is DocumentType.UNKNOWN:
            hint_type, hint_confidence, hint_reason = self._geometry_hint(aspect_ratio, area_ratio)
            if hint_type is not DocumentType.UNKNOWN and hint_confidence > confidence:
                doc_type, confidence, reason = hint_type, hint_confidence, hint_reason
        identifier = extract_identifier(ocr_text or text, doc_type)
        return ClassificationResult(
            doc_type=doc_type,
            confidence=float(min(1.0, confidence)),
            reason=reason,
            identifier=identifier,
            ocr_text=ocr_text or "",
        )


    # ----------------------------------------------------------------- scoring
    def _keyword_scores(
        self, prepared_text: str, raw_text: str = ""
    ) -> Dict[DocumentType, float]:
        """Score each document type against the prepared OCR text.

        ``raw_text`` is the un-normalised upper-cased OCR text.  It is needed
        for the MRZ pattern, because the noise-tolerant preparation would have
        stripped the ``<`` characters that the pattern looks for.
        """
        scores: Dict[DocumentType, float] = {}
        if not prepared_text:
            return scores
        for keyword, doc_type, weight in KEYWORD_RULES:
            if doc_type is DocumentType.UNKNOWN or not doc_type.enabled:
                continue
            hit = keyword_score(prepared_text, keyword)
            if hit:
                scores[doc_type] = scores.get(doc_type, 0.0) + weight * (0.7 + 0.3 * hit)
        for keyword, doc_type, weight in _ARABIC_RULES:
            if not doc_type.enabled:
                continue
            hit = keyword_score(prepared_text, keyword)
            if hit:
                scores[doc_type] = scores.get(doc_type, 0.0) + weight * (0.7 + 0.3 * hit)
        # Keywords the user added on the settings page - for built-in types too,
        # which is how another language or an alternative spelling is supported.
        for doc_type in DocumentType.all_choices():
            if doc_type is DocumentType.UNKNOWN or not doc_type.enabled:
                continue
            for keyword in doc_type.user_keywords:
                hit = keyword_score(prepared_text, str(keyword))
                if hit:
                    scores[doc_type] = scores.get(doc_type, 0.0) + (0.7 + 0.3 * hit)
        # User defined document types bring their own keywords.
        for doc_type in DocumentType.custom_types():
            for keyword in doc_type.keywords:
                hit = keyword_score(prepared_text, str(keyword))
                if hit:
                    scores[doc_type] = scores.get(doc_type, 0.0) + (0.7 + 0.3 * hit)
        # The machine readable zone lives on "<" characters that the prepared
        # text no longer contains, so the pattern is matched against the raw
        # text instead.  "«" is what OCR most often reads instead of "<".
        raw = str(raw_text or "").upper().replace("«", "<")
        if MRZ_PATTERN.search(raw):
            scores[DocumentType.PASSPORT] = scores.get(DocumentType.PASSPORT, 0.0) + 1.2
        return scores

    def _best_type(self, scores: Dict[DocumentType, float]) -> Tuple[DocumentType, float, str]:
        if not scores:
            return DocumentType.UNKNOWN, 0.0, "No matching keywords found"
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        doc_type, best = ordered[0]
        second = ordered[1][1] if len(ordered) > 1 else 0.0
        confidence = min(1.0, 0.35 + 0.35 * best + 0.15 * max(0.0, best - second))
        reason = f"keyword score {best:.2f} (next {second:.2f})"
        if confidence < self.min_confidence:
            return DocumentType.UNKNOWN, confidence, reason
        return doc_type, confidence, reason

    def _geometry_hint(self, aspect_ratio: float, area_ratio: float) -> Tuple[DocumentType, float, str]:
        """Very weak geometric fall back. Never strong enough to be authoritative."""
        if aspect_ratio <= 0:
            return DocumentType.UNKNOWN, 0.0, ""
        # An open passport spread is clearly wider than a single card.
        if aspect_ratio >= 1.95:
            return DocumentType.PASSPORT, 0.30, "very wide layout (possible open passport)"
        if area_ratio >= 0.18 and aspect_ratio <= 1.55:
            return DocumentType.PASSPORT, 0.22, "large region (possible document page)"
        return DocumentType.UNKNOWN, 0.0, ""


def extract_identifier(text: str, doc_type: DocumentType) -> str:
    """Try to pull a document identifier out of the OCR text."""
    if not text:
        return ""
    cleaned = text.replace("\n", " ")
    upper = cleaned.upper()
    if doc_type is DocumentType.PASSPORT:
        match = PASSPORT_NUMBER_PATTERN.search(upper)
        if match:
            return match.group(1)
    if doc_type in (DocumentType.IQAMA, DocumentType.DRIVING_LICENSE):
        match = SAUDI_ID_PATTERN.search(cleaned)
        if match:
            return match.group(1)
    match = PASSPORT_NUMBER_PATTERN.search(upper) or GENERIC_NUMBER_PATTERN.search(cleaned)
    if match:
        return match.group(1)
    return ""

