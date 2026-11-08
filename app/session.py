"""Per-document state (loaded PDF + detected cards)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from core.models import DetectedCard


@dataclass
class DocumentSession:
    """A PDF or image opened in the application together with its extracted cards."""

    path: Path
    page_count: int
    cards: List[DetectedCard] = field(default_factory=list)
    detected_pages: set = field(default_factory=set)
    # Password of an encrypted PDF - kept in memory only (for reopening the
    # file during preview, detection, export and crop editing) and never
    # written to the settings or history files.
    password: str = ""

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def stem(self) -> str:
        return self.path.stem

    def label(self) -> str:
        pages = f"{self.page_count} page" + ("" if self.page_count == 1 else "s")
        return f"{self.name}  ·  {pages}"

    # -------------------------------------------------------------------- cards
    def cards_for_page(self, page_index: int) -> List[DetectedCard]:
        return [card for card in self.cards if card.page_index == page_index]

    def included_cards(self) -> List[DetectedCard]:
        return [card for card in self.cards if card.included]

    def add_card(self, card: DetectedCard) -> None:
        self.cards.append(card)

    def find_card(self, card_id: str) -> Optional[DetectedCard]:
        for card in self.cards:
            if card.card_id == card_id:
                return card
        return None

    def remove_card(self, card_id: str) -> bool:
        for index, card in enumerate(self.cards):
            if card.card_id == card_id:
                del self.cards[index]
                return True
        return False

    def clear_page(self, page_index: int) -> None:
        """Drop all cards of a page (used before re-running detection)."""
        self.cards = [card for card in self.cards if card.page_index != page_index]

    # ------------------------------------------------------------------- stats
    def counts(self) -> Dict[str, int]:
        detected = len(self.cards)
        approved = len(self.included_cards())
        return {"detected": detected, "approved": approved, "excluded": detected - approved}
