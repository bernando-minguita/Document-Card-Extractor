# Face detection model

`face_detection_yunet_2023mar.onnx` is the **YuNet** face detector from the
OpenCV model zoo, used by the optional "Blur faces in exported images" feature
through `cv2.FaceDetectorYN`.

| | |
| --- | --- |
| Source | <https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet> |
| Upstream | <https://github.com/ShiqiYu/libfacedetection> |
| License | Apache-2.0 (model and upstream code) |
| Size | 232 589 bytes |
| SHA-256 (first 16) | `8f2383e4dd3cfbb4` |
| Downloaded | 2026-11-09 |

Why it is here: OpenCV 5 removed `cv2.CascadeClassifier` (the Haar cascade the
feature used before), so the neural detector is the only offline option. Keeping
the file in the repository means a fresh clone can blur faces without any
network access.

`core/face_detector.py` looks for the model in this order:

1. `$DCE_FACE_MODEL` (explicit file path)
2. `%LOCALAPPDATA%\DocumentCardExtractor\models\` (what **Download model…** writes)
3. this folder (`assets/models/` - also the copy bundled by PyInstaller)
4. a `models/` folder next to the sources (or next to the `.exe` when frozen)

To refresh or replace it, press **Settings → Export → Download model…** (it
writes to location 2), or drop a newer file in one of the folders above.
