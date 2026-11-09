"""Parsing of page range strings such as ``"1,3,5-7"``.

The GUI page bar and the command line share this parser so a typo behaves the
same way in both places: the GUI ignores the unparsable parts, the CLI prints a
usage error (a traceback is never acceptable).
"""

from __future__ import annotations

from typing import List, Optional, Tuple


def parse_page_range(
    text: str, page_count: Optional[int] = None
) -> Tuple[List[int], List[str]]:
    """Parse ``"1,3,5-7"`` into ``(zero-based indices, rejected chunks)``.

    * An empty *text* means "every page" (only when *page_count* is known).
    * Pages are clamped to ``1 … page_count``; a page outside that window is
      reported as rejected instead of being silently dropped.
    * ``page_count=None`` disables the upper bound, which lets a caller validate
      the user's input before any file (and therefore any page count) is known.
    * Reversed ranges (``"7-5"``) and non-numeric chunks (``"1,x"``) go into the
      second list so the caller can choose between ignoring them (GUI) and
      refusing to run (CLI).
    """
    if not str(text or "").strip():
        return (list(range(page_count)) if page_count is not None else []), []

    limit = None if page_count is None else max(1, int(page_count))
    indices: List[int] = []
    rejected: List[str] = []

    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_text, _, end_text = chunk.partition("-")
            first, last = _to_int(start_text), _to_int(end_text)
            if first is None or last is None or last < first:
                rejected.append(chunk)
                continue
            first = max(1, first)
            if limit is not None:
                last = min(limit, last)
            if last < first:
                rejected.append(chunk)
                continue
            indices.extend(range(first - 1, last))
        else:
            page = _to_int(chunk)
            if page is None or page < 1 or (limit is not None and page > limit):
                rejected.append(chunk)
                continue
            indices.append(page - 1)

    return sorted(set(indices)), rejected


def _to_int(value: str) -> Optional[int]:
    """``int(value)`` or ``None`` for anything that is not a plain number."""
    text = (value or "").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None
