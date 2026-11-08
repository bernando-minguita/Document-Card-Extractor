"""Vector icons drawn from embedded SVG definitions (no binary assets needed)."""

from __future__ import annotations

from typing import Dict

from PySide6.QtCore import QByteArray, QRectF, QSize, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QAbstractButton, QWidget

# Slate-800: dark enough to stay readable on white / light-grey buttons.
DEFAULT_COLOR = "#1e293b"
# Lighter tone used automatically for disabled buttons.
DISABLED_COLOR = "#a3aebf"
# Icons on coloured (primary / danger) buttons must be white, not dark.
ON_COLOR = "#ffffff"
# Slightly larger than the Qt default (16 px) so the strokes stay visible.
BUTTON_ICON_SIZE = 18
# Rendered pixmap sizes; the ones we actually ask for are included so Qt never
# has to scale (scaling is what turns the strokes into a faint blur).
_PIXMAP_SIZES = (16, 18, 20, 22, 24, 32, 48)

# Feather style icons, 24x24 view box, stroke only.
_ICONS: Dict[str, str] = {    "dashboard": '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/>'
    '<rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
    "extract": '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="M3 10h18"/>'
    '<path d="M8 15h5"/>',
    "history": '<path d="M3 12a9 9 0 1 0 3-6.7"/><path d="M3 4v5h5"/><path d="M12 8v4l3 2"/>',
    "settings": '<path d="M4 6h10"/><path d="M18 6h2"/><circle cx="16" cy="6" r="2"/>'
    '<path d="M4 12h4"/><path d="M12 12h8"/><circle cx="10" cy="12" r="2"/>'
    '<path d="M4 18h10"/><path d="M18 18h2"/><circle cx="16" cy="18" r="2"/>',
    "folder": '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
    "plus": '<path d="M12 5v14"/><path d="M5 12h14"/>',
    "play": '<path d="M7 4.5l12 7.5-12 7.5z"/>',
    "export": '<path d="M12 4v10"/><path d="M8 10l4 4 4-4"/><path d="M4 17v2a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-2"/>',
    "save": '<path d="M5 4h11l3 3v13a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1z"/>'
    '<path d="M8 4v5h7V4"/><path d="M8 13h8"/>',
    "zoom_in": '<circle cx="11" cy="11" r="6"/><path d="M20 20l-4.2-4.2"/><path d="M11 8.5v5"/><path d="M8.5 11h5"/>',
    "zoom_out": '<circle cx="11" cy="11" r="6"/><path d="M20 20l-4.2-4.2"/><path d="M8.5 11h5"/>',
    "fit": '<path d="M4 9V5h4"/><path d="M20 9V5h-4"/><path d="M4 15v4h4"/><path d="M20 15v4h-4"/>',
    "rotate_left": '<path d="M4 10a8 8 0 1 1 3 7"/><path d="M4 4v6h6"/>',
    "rotate_right": '<path d="M20 10a8 8 0 1 0-3 7"/><path d="M20 4v6h-6"/>',
    "flip": '<path d="M12 3v18"/><path d="M8 7H4v10h4"/><path d="M16 7h4v10h-4"/>',
    "trash": '<path d="M4 7h16"/><path d="M9 7V5h6v2"/><path d="M6 7l1 13h10l1-13"/>',
    "edit": '<path d="M4 20h4l10-10-4-4L4 16z"/><path d="M14 6l4 4"/>',
    "crop": '<path d="M6 3v13a2 2 0 0 0 2 2h13"/><path d="M3 6h13a2 2 0 0 1 2 2v13"/>',
    "check": '<path d="M4 12.5l5 5L20 6.5"/>',
    "close": '<path d="M6 6l12 12"/><path d="M18 6L6 18"/>',
    "chevron_left": '<path d="M14 6l-6 6 6 6"/>',
    "chevron_right": '<path d="M10 6l6 6-6 6"/>',
    "image": '<rect x="3" y="5" width="18" height="14" rx="2"/><circle cx="8.5" cy="10" r="1.5"/>'
    '<path d="M5 17l4.5-4.5 3 3 3-2.5L20 17"/>',
    "alert": '<path d="M12 4l9 16H3z"/><path d="M12 10v4"/><path d="M12 17h.01"/>',
    "refresh": '<path d="M20 11a8 8 0 1 0-2.3 5.7"/><path d="M20 4v7h-7"/>',
    "search": '<circle cx="11" cy="11" r="6"/><path d="M20 20l-4.2-4.2"/>',
    "document": '<path d="M6 3h8l4 4v14a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1z"/>'
    '<path d="M14 3v4h4"/><path d="M8 13h8"/><path d="M8 17h5"/>',
    "drag": '<circle cx="9" cy="7" r="1"/><circle cx="15" cy="7" r="1"/><circle cx="9" cy="12" r="1"/>'
    '<circle cx="15" cy="12" r="1"/><circle cx="9" cy="17" r="1"/><circle cx="15" cy="17" r="1"/>',
}


def _svg_markup(name: str, color: str, stroke_width: float, filled: bool) -> str:
    body = _ICONS.get(name)
    if body is None:
        body = _ICONS["document"]
    fill = color if filled else "none"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
        f'fill="{fill}" stroke="{color}" stroke-width="{stroke_width}" '
        f'stroke-linecap="round" stroke-linejoin="round">{body}</svg>'
    )


def _stroke_width(size: int) -> float:
    """Stroke width for a render of *size* pixels.

    A 24 px view box stroke is scaled down with the pixmap, so an icon rendered
    at 16 px would end up with a ~1.2 px hairline that looks washed out.  Small
    sizes therefore get a proportionally thicker stroke.
    """
    return max(2.0, 1.7 * 24.0 / max(8, int(size)))


_ICON_CACHE: Dict[str, QIcon] = {}


def pixmap(
    name: str,
    size: int = 24,
    color: str = DEFAULT_COLOR,
    filled: bool = False,
    stroke: float | None = None,
) -> QPixmap:
    """Render *name* into a square pixmap of *size* pixels."""
    pixmap_size = max(8, int(size))
    stroke_width = _stroke_width(pixmap_size) if stroke is None else float(stroke)
    renderer = QSvgRenderer(
        QByteArray(_svg_markup(name, color, stroke_width, filled).encode("utf-8"))
    )
    canvas = QPixmap(pixmap_size, pixmap_size)
    canvas.fill(Qt.GlobalColor.transparent)
    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    renderer.render(painter, QRectF(0, 0, pixmap_size, pixmap_size))
    painter.end()
    return canvas


def icon(
    name: str,
    color: str = DEFAULT_COLOR,
    filled: bool = False,
    disabled_color: str = DISABLED_COLOR,
) -> QIcon:
    """Build a QIcon with normal and disabled variants.

    Icons are expensive to build (7 sizes × 2 modes rendered from SVG), and the
    widgets request the same (name, colour) combination constantly - e.g. the
    toolbar rebuilds its icons on every page switch - so results are cached.
    """
    key = f"{name}\x00{color}\x00{filled}\x00{disabled_color}"
    cached = _ICON_CACHE.get(key)
    if cached is not None:
        return cached
    result = QIcon()
    for size in _PIXMAP_SIZES:
        result.addPixmap(pixmap(name, size, color, filled), QIcon.Mode.Normal)
        result.addPixmap(
            pixmap(name, size, disabled_color, filled), QIcon.Mode.Disabled
        )
    _ICON_CACHE[key] = result
    return result


def button_icon(name: str, variant: str = "", filled: bool = False) -> QIcon:
    """Icon that stays readable on *variant* buttons.

    ``primary`` (blue) and ``danger`` (red) buttons get a white icon, everything
    else the dark default - a dark icon on a blue button is almost invisible.
    """
    color = ON_COLOR if str(variant) in ("primary", "danger") else DEFAULT_COLOR
    return icon(name, color, filled)


def apply_icon_size(widget: QWidget, size: int = BUTTON_ICON_SIZE) -> None:
    """Give every icon-bearing button inside *widget* a slightly larger icon."""
    for button in widget.findChildren(QAbstractButton):
        if not button.icon().isNull():
            button.setIconSize(QSize(int(size), int(size)))


def app_icon() -> QIcon:
    """Simple application icon: stacked document cards."""
    return icon("extract", "#2563eb")


