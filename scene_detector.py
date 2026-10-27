"""
scene_detector.py — On-device object detection using TFLite.

Primary model : EfficientDet-Lite0 INT8 quantised (~4.4 MB, 80 COCO classes)
Fallback model: MobileNet-SSD v2 INT8 if EfficientDet is unavailable

Model files are downloaded automatically on first run and cached next to
this file, mirroring the pattern used by detector.py.

Inference is intentionally kept on the calling thread — callers should
run this from a dedicated daemon thread (see scene_loop in main.py).
"""

from __future__ import annotations

import os
import urllib.request
import zipfile
from dataclasses import dataclass, field

import cv2
import numpy as np

# ── TFLite import (tflite-runtime preferred; fall back to full TF) ────────────
try:
    import tflite_runtime.interpreter as tflite
    _TFLITE_SOURCE = "tflite-runtime"
except ImportError:
    try:
        import tensorflow as tf
        tflite = tf.lite
        _TFLITE_SOURCE = "tensorflow"
    except ImportError:
        tflite = None
        _TFLITE_SOURCE = "none"

# ── COCO 80-class label map ───────────────────────────────────────────────────
COCO_LABELS: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# ── Model registry ────────────────────────────────────────────────────────────
_MODELS = {
    "efficientdet": {
        "url": (
            "https://storage.googleapis.com/tfhub-lite-models/tensorflow/"
            "lite-model/efficientdet/lite0/detection/metadata/1.tflite"
        ),
        "filename": "efficientdet_lite0_int8.tflite",
        "input_size": 320,
        # Output tensor indices for the metadata-embedded EfficientDet model:
        #   0 → boxes   [1, N, 4]   (ymin, xmin, ymax, xmax) normalised
        #   1 → classes [1, N]      (float, 0-indexed)
        #   2 → scores  [1, N]
        #   3 → count   [1]
        "output_order": ("boxes", "classes", "scores", "count"),
    },
    "mobilenet_ssd": {
        "url": (
            "https://storage.googleapis.com/download.tensorflow.org/models/"
            "tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip"
        ),
        "filename": "mobilenet_ssd_v2_int8.tflite",
        "input_size": 300,
        "output_order": ("boxes", "classes", "scores", "count"),
    },
}

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIDENCE_THRESHOLD = 0.55
_MIN_BOX_AREA = 0.002   # fraction of total frame area; smaller boxes are noise


# ── Detection result ──────────────────────────────────────────────────────────

@dataclass
class Detection:
    label: str
    confidence: float
    # Normalised coordinates (0-1)
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def cx(self) -> float:
        return (self.xmin + self.xmax) / 2

    @property
    def cy(self) -> float:
        return (self.ymin + self.ymax) / 2

    @property
    def area(self) -> float:
        return max(0.0, self.xmax - self.xmin) * max(0.0, self.ymax - self.ymin)


# ── Downloader ────────────────────────────────────────────────────────────────

def _download(url: str, dest: str) -> None:
    """Download url to dest with a progress print. Unzips if a zip archive."""
    print(f"[scene_detector] Downloading {os.path.basename(dest)} …")
    tmp_path = dest + ".tmp"
    try:
        urllib.request.urlretrieve(url, tmp_path)
        if zipfile.is_zipfile(tmp_path):
            with zipfile.ZipFile(tmp_path, "r") as z:
                tflite_names = [n for n in z.namelist() if n.endswith(".tflite")]
                if tflite_names:
                    extracted = z.extract(tflite_names[0], os.path.dirname(dest))
                    os.replace(extracted, dest)
                else:
                    raise RuntimeError("No .tflite found inside zip archive")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        else:
            os.replace(tmp_path, dest)
        print(f"[scene_detector] Downloaded {os.path.basename(dest)} OK.")
    except Exception as exc:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if os.path.exists(dest):
            os.remove(dest)
        raise RuntimeError(f"[scene_detector] Download failed: {exc}") from exc


def _ensure_model(key: str) -> str | None:
    """Return the local path of the model, downloading if needed. Returns None on failure."""
    spec = _MODELS[key]
    path = os.path.join(_MODEL_DIR, spec["filename"])
    if not os.path.exists(path):
        try:
            _download(spec["url"], path)
        except RuntimeError as exc:
            print(exc)
            return None
    return path


# ── SceneDetector ─────────────────────────────────────────────────────────────

class SceneDetector:
    """
    Wraps TFLite inference for real-time object detection (COCO classes).

    Usage
    -----
        detector = SceneDetector()
        if detector.available:
            detections = detector.detect(bgr_frame)
    """

    def __init__(self) -> None:
        self.available = False
        self._interpreter = None
        self._input_size = 300
        self._input_idx = 0
        self._output_map: dict[str, int] = {}

        if tflite is None:
            print(
                "[scene_detector] Neither tflite-runtime nor tensorflow found. "
                "Scene narration disabled. Install with: pip install tflite-runtime"
            )
            return

        # Prioritize MobileNet-SSD (standard 4-output post-processed COCO model)
        for model_key in ("mobilenet_ssd", "efficientdet"):
            path = _ensure_model(model_key)
            if path and self._load(path, model_key):
                print(f"[scene_detector] Loaded {model_key} from {os.path.basename(path)}")
                self.available = True
                break

        if not self.available:
            print("[scene_detector] Could not load any detection model. Scene narration disabled.")

    # ------------------------------------------------------------------
    def _load(self, path: str, model_key: str) -> bool:
        try:
            interp = tflite.Interpreter(model_path=path)
            interp.allocate_tensors()

            inp = interp.get_input_details()
            out = interp.get_output_details()

            order = _MODELS[model_key]["output_order"]
            if len(out) < len(order):
                return False

            self._interpreter = interp
            self._input_idx = inp[0]["index"]
            self._input_size = _MODELS[model_key]["input_size"]

            # Map logical output names to tensor indices
            self._output_map = {name: out[i]["index"] for i, name in enumerate(order)}
            return True
        except Exception as exc:
            print(f"[scene_detector] Failed to load {model_key}: {exc}")
            return False

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run inference on a BGR frame.

        Returns a deduplicated, filtered list of Detection objects.
        Returns [] if unavailable or frame is None.
        """
        if not self.available or frame is None:
            return []

        # ── Preprocess ────────────────────────────────────────────────
        sz = self._input_size
        resized = cv2.resize(frame, (sz, sz))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        inp_detail = self._interpreter.get_input_details()[0]
        if inp_detail["dtype"] == np.float32:
            tensor = rgb.astype(np.float32) / 255.0
        else:
            tensor = rgb.astype(np.uint8)

        self._interpreter.set_tensor(self._input_idx, tensor[np.newaxis])

        # ── Inference ─────────────────────────────────────────────────
        self._interpreter.invoke()

        # ── Extract outputs ───────────────────────────────────────────
        boxes   = self._interpreter.get_tensor(self._output_map["boxes"])[0]    # [N,4]
        classes = self._interpreter.get_tensor(self._output_map["classes"])[0]  # [N]
        scores  = self._interpreter.get_tensor(self._output_map["scores"])[0]   # [N]

        # ── Filter & build Detection objects ──────────────────────────
        raw_detections: list[Detection] = []
        for i in range(len(scores)):
            conf = float(scores[i])
            if conf < _CONFIDENCE_THRESHOLD:
                continue

            class_id = int(classes[i])
            if class_id < 0 or class_id >= len(COCO_LABELS):
                continue

            ymin, xmin, ymax, xmax = boxes[i]
            det = Detection(
                label=COCO_LABELS[class_id],
                confidence=conf,
                xmin=float(xmin),
                ymin=float(ymin),
                xmax=float(xmax),
                ymax=float(ymax),
            )
            if det.area < _MIN_BOX_AREA:
                continue
            raw_detections.append(det)

        return self._nms(raw_detections)

    # ------------------------------------------------------------------
    @staticmethod
    def _nms(detections: list[Detection], iou_threshold: float = 0.45) -> list[Detection]:
        """Non-maximum suppression using cv2.dnn.NMSBoxes."""
        if not detections:
            return []

        boxes_cv = [
            [int(d.xmin * 1000), int(d.ymin * 1000),
             int((d.xmax - d.xmin) * 1000), int((d.ymax - d.ymin) * 1000)]
            for d in detections
        ]
        scores_cv = [d.confidence for d in detections]

        indices = cv2.dnn.NMSBoxes(
            boxes_cv, scores_cv, _CONFIDENCE_THRESHOLD, iou_threshold
        )

        if len(indices) == 0:
            return []

        # cv2 returns either a list or a nested array depending on version
        flat = [int(i) for i in np.array(indices).flatten()]
        return [detections[i] for i in flat]
