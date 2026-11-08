"""Light theme colours and the global Qt style sheet."""

from __future__ import annotations

# --------------------------------------------------------------------- palette
BACKGROUND = "#f4f6fa"
SURFACE = "#ffffff"
SURFACE_ALT = "#f8fafc"
BORDER = "#e2e8f0"
BORDER_STRONG = "#cbd5e1"
TEXT = "#0f172a"
TEXT_MUTED = "#64748b"
TEXT_FAINT = "#94a3b8"
PRIMARY = "#2563eb"
PRIMARY_DARK = "#1d4ed8"
PRIMARY_SOFT = "#dbeafe"
SUCCESS = "#15803d"
SUCCESS_SOFT = "#dcfce7"
WARNING = "#b45309"
WARNING_SOFT = "#fef3c7"
DANGER = "#b91c1c"
DANGER_SOFT = "#fee2e2"
CARD_COLORS = ("#2563eb", "#0ea5e9", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899")

STYLESHEET = f"""
QWidget {{
    color: {TEXT};
    font-family: "Segoe UI", "Inter", "Helvetica Neue", sans-serif;
    font-size: 13px;
}}
QMainWindow, QDialog {{
    background: {BACKGROUND};
}}
QToolTip {{
    background: {TEXT};
    color: #ffffff;
    border: none;
    padding: 6px 8px;
    border-radius: 4px;
}}

/* --------------------------------------------------------------- side bar */
QFrame#Sidebar {{
    background: {SURFACE};
    border-right: 1px solid {BORDER};
}}
QLabel#BrandTitle {{
    font-size: 15px;
    font-weight: 700;
}}
QLabel#BrandSubtitle {{
    font-size: 11px;
    color: {TEXT_MUTED};
}}
QPushButton#NavButton {{
    text-align: left;
    padding: 10px 12px;
    border: none;
    border-radius: 8px;
    color: {TEXT_MUTED};
    font-weight: 600;
    background: transparent;
}}
QPushButton#NavButton:hover {{
    background: {SURFACE_ALT};
    color: {TEXT};
}}
QPushButton#NavButton:checked {{
    background: {PRIMARY_SOFT};
    color: {PRIMARY_DARK};
}}

/* ---------------------------------------------------------------- headers */
QLabel#PageTitle {{
    font-size: 19px;
    font-weight: 700;
}}
QLabel#PageSubtitle {{
    font-size: 12px;
    color: {TEXT_MUTED};
}}
QLabel#SectionTitle {{
    font-size: 13px;
    font-weight: 700;
}}
QLabel#Muted {{
    color: {TEXT_MUTED};
}}
QLabel#Faint {{
    color: {TEXT_FAINT};
    font-size: 12px;
}}

/* ---------------------------------------------------------------- buttons */
QPushButton, QToolButton {{
    background: {SURFACE};
    border: 1px solid {BORDER_STRONG};
    border-radius: 8px;
    padding: 7px 14px;
    font-weight: 600;
    color: {TEXT};
}}
QPushButton:hover, QToolButton:hover {{
    background: {SURFACE_ALT};
    border-color: {TEXT_FAINT};
}}
QPushButton:pressed, QToolButton:pressed {{
    background: {BORDER};
}}
QPushButton:disabled, QToolButton:disabled {{
    color: {TEXT_FAINT};
    border-color: {BORDER};
    background: {SURFACE_ALT};
}}
QPushButton[variant="primary"] {{
    background: {PRIMARY};
    border-color: {PRIMARY};
    color: #ffffff;
}}
QPushButton[variant="primary"]:hover {{
    background: {PRIMARY_DARK};
    border-color: {PRIMARY_DARK};
}}
QPushButton[variant="primary"]:disabled {{
    background: #93b4f6;
    border-color: #93b4f6;
    color: #eff6ff;
}}
QPushButton[variant="danger"] {{
    background: {DANGER};
    border-color: {DANGER};
    color: #ffffff;
}}
QPushButton[variant="danger"]:hover {{
    background: #991b1b;
}}
QPushButton[variant="ghost"], QToolButton[variant="ghost"] {{
    background: transparent;
    border-color: transparent;
    color: {TEXT_MUTED};
}}
QPushButton[variant="ghost"]:hover, QToolButton[variant="ghost"]:hover {{
    background: {SURFACE_ALT};
    color: {TEXT};
}}
QPushButton#ToolbarButton {{
    padding: 8px 16px;
}}
QToolButton#IconButton {{
    padding: 6px;
    border-radius: 8px;
}}

/* ----------------------------------------------------------------- inputs */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {SURFACE};
    border: 1px solid {BORDER_STRONG};
    border-radius: 8px;
    padding: 6px 10px;
    min-height: 18px;
    selection-background-color: {PRIMARY};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {PRIMARY};
}}
QComboBox::drop-down {{
    border: none;
    width: 22px;
}}
QComboBox QAbstractItemView {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
    selection-background-color: {PRIMARY_SOFT};
    selection-color: {TEXT};
    padding: 4px;
}}
QCheckBox {{
    spacing: 8px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid {BORDER_STRONG};
    background: {SURFACE};
}}
QCheckBox::indicator:checked {{
    background: {PRIMARY};
    border-color: {PRIMARY};
}}
QSlider::groove:horizontal {{
    height: 4px;
    background: {BORDER};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    width: 16px;
    height: 16px;
    margin: -7px 0;
    border-radius: 8px;
    background: {PRIMARY};
}}

/* ------------------------------------------------------------ containers */
QFrame#Card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 12px;
}}
QFrame#Toolbar {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 12px;
}}
QFrame#StatCard {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 12px;
}}
QFrame#TypeCell {{
    background: {SURFACE_ALT};
    border: 1px solid {BORDER};
    border-radius: 10px;
}}
QFrame#TypeCell[enabledType="false"] {{
    background: {BACKGROUND};
}}
QCheckBox#TypeName {{
    font-weight: 600;
    font-size: 13px;
}}
QToolButton#TypeAction {{
    padding: 2px 8px;
    font-size: 11px;
    border-radius: 6px;
}}
QLabel#StatValue {{
    font-size: 26px;
    font-weight: 800;
    color: {PRIMARY};
}}
QLabel#StatLabel {{
    font-size: 12px;
    color: {TEXT_MUTED};
}}
QGroupBox {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 12px;
    margin-top: 14px;
    padding: 18px 14px 14px 14px;
    font-weight: 700;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    padding: 0 6px;
}}

/* -------------------------------------------------------------------- lists */
QListWidget, QListView {{
    background: transparent;
    border: none;
}}
QListWidget::item, QListView::item {{
    color: {TEXT};
}}
QListWidget::item:hover, QListView::item:hover {{
    background: {SURFACE_ALT};
}}
/* The selected colour must be set explicitly: with a transparent highlight the
   default (white) highlighted-text colour would make the label invisible. */
QListWidget::item:selected, QListView::item:selected {{
    background: transparent;
    color: {TEXT};
}}
QListWidget#RecentList::item {{
    padding: 5px 6px;
    border-radius: 6px;
}}
QListWidget#RecentList::item:hover {{
    background: {SURFACE_ALT};
}}
QListWidget#RecentList::item:selected {{
    background: {PRIMARY_SOFT};
    color: {PRIMARY_DARK};
}}
/* The gallery paints its own cards, so it must not get list-item backgrounds. */
QListWidget#DocumentGallery::item,
QListWidget#DocumentGallery::item:hover,
QListWidget#DocumentGallery::item:selected {{
    background: transparent;
    color: {TEXT};
}}
QTableWidget, QTableView {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    gridline-color: {BORDER};
    selection-background-color: {PRIMARY_SOFT};
    selection-color: {TEXT};
    alternate-background-color: {SURFACE_ALT};
}}
QHeaderView::section {{
    background: {SURFACE_ALT};
    color: {TEXT_MUTED};
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 8px;
    font-weight: 700;
}}

/* ------------------------------------------------------------------ graphics view */
QGraphicsView {{
    background: {SURFACE_ALT};
    border: 1px solid {BORDER};
    border-radius: 12px;
}}

/* --------------------------------------------------------------- scroll bars */
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER_STRONG};
    border-radius: 5px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {TEXT_FAINT};
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 2px;
}}
QScrollBar::handle:horizontal {{
    background: {BORDER_STRONG};
    border-radius: 5px;
    min-width: 30px;
}}
QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{
    background: none;
    border: none;
    height: 0;
    width: 0;
}}

/* ------------------------------------------------------------------ status */
QStatusBar {{
    background: {SURFACE};
    border-top: 1px solid {BORDER};
    color: {TEXT_MUTED};
}}
QStatusBar::item {{
    border: none;
}}
QProgressBar {{
    background: {BORDER};
    border: none;
    border-radius: 5px;
    height: 10px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background: {PRIMARY};
    border-radius: 5px;
}}
QSplitter::handle:horizontal {{
    background: transparent;
    width: 10px;
}}
QMenu {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 6px;
}}
QMenu::item {{
    padding: 7px 22px 7px 14px;
    border-radius: 6px;
}}
QMenu::item:selected {{
    background: {PRIMARY_SOFT};
}}
QMenu::separator {{
    height: 1px;
    background: {BORDER};
    margin: 4px 8px;
}}
QMessageBox {{
    background: {SURFACE};
}}
QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 10px;
    background: {SURFACE};
}}
QTabBar::tab {{
    background: transparent;
    padding: 8px 16px;
    color: {TEXT_MUTED};
    font-weight: 600;
}}
QTabBar::tab:selected {{
    color: {PRIMARY};
    border-bottom: 2px solid {PRIMARY};
}}
"""
