"""Face detection for the optional export-time blur.

Two backends, tried in this order:

``yunet``
    ``cv2.FaceDetectorYN`` (OpenCV 4.5.4+ **and** 5.x) with the 227 KB YuNet
    ONNX model.  This is what works on OpenCV 5, where the Haar cascade class
    was removed.
``haar``
    ``cv2.CascadeClassifier`` with OpenCV's bundled frontal-face cascade
    (OpenCV 4 only).

The model is looked up in ``$DCE_FACE_MODEL`` (explicit file), the app's
per-user data folder (``…/DocumentCardExtractor/models``), the copy shipped in
``assets/models`` and a plain ``models`` folder next to the sources.
:func:`download_model` fetches it once from the OpenCV model zoo - the only
network access in the application, and only when the user asks for it.
"""

from __future__ import annotations

import os
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"
MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
MODEL_FILE_ENV_VAR = "DCE_FACE_MODEL"
MODEL_DIR_ENV_VAR = "DCE_FACE_MODEL_DIR"
#: YuNet detection parameters (its own defaults, tuned for small ID photos).
SCORE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.3
TOP_K = 5000
#: A valid model is ~227 KB; anything much smaller is a broken download.
MIN_MODEL_BYTES = 10_000

_detector = None
_detector_path: Optional[Path] = None
_status_cache: Optional[Tuple[str, str]] = None


# ------------------------------------------------------------------ model files
def model_folder() -> Path:
    """Folder that holds a downloaded model (the app's per-user data folder)."""
    override = os.environ.get(MODEL_DIR_ENV_VAR)
    if override:
        return Path(override)
    try:
        from app.settings import data_folder  # lazy import: avoids an import cycle

        return data_folder() / "models"
    except Exception:  # pragma: no cover - core used without the GUI/settings
        base = os.environ.get("LOCALAPPDATA") or str(Path.home())
        return Path(base) / "DocumentCardExtractor" / "models"


def _application_roots() -> List[Path]:
    """Folders that may contain the bundled ``assets`` directory.

    ``sys._MEIPASS`` is where PyInstaller unpacks a one-file bundle (and where it
    keeps the read-only payload of a one-dir build, version 6+); the folder next
    to the executable covers the older one-dir layout.  Running from source falls
    back to the project folder.
    """
    roots: List[Path] = []
    meipass = getattr(sys, "_MEIPASS", "")
    if meipass:
        roots.append(Path(meipass))
    if getattr(sys, "frozen", False):
        roots.append(Path(sys.executable).resolve().parent)
    roots.append(Path(__file__).resolve().parents[1])
    unique: List[Path] = []
    for root in roots:
        if root not in unique:
            unique.append(root)
    return unique


def model_candidates() -> List[Path]:
    """Every place a model file is accepted from, most specific first."""
    candidates: List[Path] = []
    explicit = os.environ.get(MODEL_FILE_ENV_VAR)
    if explicit:
        candidates.append(Path(explicit))
    # per-user (what "Download model…" writes), then the copy that ships with
    # the application, then a plain "models" folder next to the sources
    candidates.append(model_folder() / MODEL_FILENAME)
    for root in _application_roots():
        candidates.append(root / "assets" / "models" / MODEL_FILENAME)
        candidates.append(root / "models" / MODEL_FILENAME)
    return candidates


def find_model() -> Optional[Path]:
    """The first usable model file, or ``None``."""
    for path in model_candidates():
        try:
            if path.is_file() and path.stat().st_size >= MIN_MODEL_BYTES:
                return path
        except OSError:
            continue
    return None


# --------------------------------------------------------------------- backends
def _yunet_available() -> bool:
    return hasattr(cv2, "FaceDetectorYN")


def _haar_available() -> bool:
    return hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data")


def can_download_model() -> bool:
    """True when a model download would enable face blur on this build."""
    return _yunet_available() and find_model() is None


def status() -> Tuple[str, str]:
    """Return ``(backend, message)``; *backend* is ``"yunet"``, ``"haar"`` or ``""``.

    The message is user facing: it is shown in the settings page and in the
    export summary when blurring had to be skipped.
    """
    global _status_cache
    if _status_cache is not None:
        return _status_cache
    version = getattr(cv2, "__version__", "?")
    model = find_model()
    if _yunet_available() and model is not None:
        _status_cache = ("yunet", f"YuNet neural network ({model.name})")
    elif _yunet_available():
        _status_cache = (
            "",
            f"OpenCV {version} needs the YuNet face model ({MODEL_FILENAME}); "
            "download it once in Settings → face blur",
        )
    elif _haar_available():
        _status_cache = ("haar", "OpenCV Haar cascade")
    else:
        _status_cache = (
            "",
            f"face blur needs OpenCV's Haar cascade or the YuNet model, which "
            f"OpenCV {version} does not provide - the images were exported "
            "without blurring",
        )
    return _status_cache


def is_available() -> bool:
    return bool(status()[0])


def clear_cache() -> None:
    """Forget the cached status/detector (tests, or after a model download)."""
    global _detector, _detector_path, _status_cache
    _detector = None
    _detector_path = None
    _status_cache = None


def _get_detector():
    """Return ``(backend, detector)``, building the detector once."""
    global _detector, _detector_path
    backend, _message = status()
    if not backend:
        return "", None
    model = find_model() if backend == "yunet" else None
    if backend == "yunet" and model is None:  # the model vanished under us
        clear_cache()
        return "", None
    if _detector is None or _detector_path != model:
        if backend == "yunet":
            _detector = _create_yunet(model)
        else:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            _detector = cv2.CascadeClassifier(cascade_path)
            if _detector.empty():
                raise RuntimeError(f"Failed to load face cascade from {cascade_path}")
        _detector_path = model
    return backend, _detector


def _create_yunet(model_path: Path):
    """Build the YuNet detector, without OpenCV's one-line DNN init warning.

    OpenCV 5 always prints "setPreferableTarget … not supported by the new graph
    engine" when the network is created; it is harmless but looks like an error
    in the console, so the log level is raised only for this call.
    """
    logging_api = getattr(getattr(cv2, "utils", None), "logging", None)
    previous = None
    if logging_api is not None:
        try:
            previous = logging_api.getLogLevel()
            logging_api.setLogLevel(getattr(logging_api, "LOG_LEVEL_SILENT", 0))
        except Exception:  # pragma: no cover - older/odd OpenCV builds
            previous = None
    try:
        return cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320), SCORE_THRESHOLD, NMS_THRESHOLD, TOP_K
        )
    finally:
        if previous is not None:
            try:
                logging_api.setLogLevel(previous)
            except Exception:  # pragma: no cover - best effort only
                pass


def detect_faces(
    image: np.ndarray, min_face_size: int = 40
) -> List[Tuple[int, int, int, int]]:
    """Return ``[(x, y, w, h), …]`` for every face found in *image* (BGR)."""
    if image is None or image.size == 0:
        return []
    backend, detector = _get_detector()
    if detector is None:
        return []
    minimum = max(1, int(min_face_size))
    if backend == "yunet":
        bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        height, width = bgr.shape[:2]
        detector.setInputSize((width, height))
        _count, faces = detector.detect(bgr)
        if faces is None:
            return []
        boxes: List[Tuple[int, int, int, int]] = []
        for row in faces:
            x, y, w, h = (int(round(float(value))) for value in row[:4])
            if w >= minimum and h >= minimum:
                boxes.append((x, y, w, h))
        return boxes

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    found = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(minimum, minimum)
    )
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in found]


# --------------------------------------------------------------------- download
def download_model(
    destination: Optional[Path] = None,
    timeout: float = 60.0,
    opener: Optional[Callable] = None,
) -> Path:
    """Fetch the YuNet model (once) and return its path.

    Raises :class:`RuntimeError` with a readable message on any failure; a
    partially written file is never kept.
    """
    target = Path(destination) if destination is not None else model_folder() / MODEL_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    fetch = opener or urllib.request.urlopen
    temp_name = ""
    try:
        with fetch(MODEL_URL, timeout=timeout) as response:
            payload = response.read()
        if len(payload) < MIN_MODEL_BYTES:
            raise RuntimeError(
                f"the downloaded file is only {len(payload)} bytes - the mirror "
                "probably returned an error page"
            )
        handle, temp_name = tempfile.mkstemp(dir=str(target.parent), suffix=".part")
        with os.fdopen(handle, "wb") as file_handle:
            file_handle.write(payload)
        os.replace(temp_name, target)
        temp_name = ""
    except Exception as exc:
        raise RuntimeError(f"could not download the face model: {exc}") from exc
    finally:
        if temp_name:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
    clear_cache()
    return target


