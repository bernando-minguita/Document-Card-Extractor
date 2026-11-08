"""Entry point of the Document Card Extractor.

Run with::

    python main.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the project importable even when the app is started from a different
# working directory (shortcut, "open with", pythonw, …).  Third party packages
# are expected in the interpreter's normal site-packages directory.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep Qt and OpenCV quiet and predictable on Windows HiDPI screens.
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")


def _friendly_import_error(exc: ImportError) -> str:
    missing = getattr(exc, "name", None) or str(exc)
    return (
        f"A required Python package is missing: {missing}\n\n"
        "Install the dependencies with:\n"
        "    pip install -r requirements.txt\n\n"
        f"Technical details: {exc}"
    )


def main() -> int:
    try:
        from PySide6.QtCore import Qt, QTimer
        from PySide6.QtGui import QFont
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:  # pragma: no cover - depends on the environment
        print(_friendly_import_error(exc), file=sys.stderr)
        return 1

    try:
        from app.icons import app_icon
        from app.main_window import MainWindow
        from app.settings import AppSettings
        from app.styles import STYLESHEET
    except ImportError as exc:  # pragma: no cover - depends on the environment
        print(_friendly_import_error(exc), file=sys.stderr)
        return 1

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    application = QApplication(sys.argv)
    application.setApplicationName("DocumentCardExtractor")
    # NOTE: QGuiApplication::applicationDisplayName is deliberately *not* set.
    # Qt appends it to every top level window title ("<title> - <display name>"),
    # which would duplicate the name that MainWindow already puts in its title.
    # The organization name is left empty on purpose as well: Qt would nest the
    # settings folder as "<Organization>/<Application>".
    application.setWindowIcon(app_icon())
    application.setStyleSheet(STYLESHEET)
    application.setFont(QFont("Segoe UI", 9))

    settings = AppSettings.load()
    window = MainWindow(settings)
    window.show()

    # A PDF **or image** supplied on the command line is opened right away (drag &
    # drop on the executable, "Open with", or `python main.py scan.png`).
    from core.pdf_loader import is_supported_file

    argument_files = [arg for arg in sys.argv[1:] if is_supported_file(arg)]
    if argument_files:
        first, rest = argument_files[0], argument_files[1:]

        def _open_start_files() -> None:
            # Deferred into the running event loop: load_pdf can show dialogs
            # (password prompt, "cannot open file") - modal dialogs before
            # application.exec() has spun the loop misbehave on some platforms
            # and would appear before the main window is painted.
            if window.load_pdf(first, add=False):
                for extra in rest:
                    window.load_pdf(extra, add=True)

        QTimer.singleShot(0, _open_start_files)

    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
