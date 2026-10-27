"""
detector.py — MediaPipe HandLandmarker wrapper (Tasks API, mediapipe >= 0.10).

The legacy mp.solutions.hands API was removed in 0.10.x.
This module uses mediapipe.tasks.python.vision.HandLandmarker instead.

Model file: hand_landmarker.task is downloaded automatically on first run
and cached next to this file.
"""

import os
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult

# ---------------------------------------------------------------------------
# Landmark connection pairs (MediaPipe standard topology, 21 points)
# Used to draw the skeleton without mp.solutions.drawing_utils
# ---------------------------------------------------------------------------
_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (0, 9), (9, 10), (10, 11), (11, 12),     # middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (5, 9), (9, 13), (13, 17),               # palm knuckles
]

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")


def _ensure_model() -> str:
    """Download the model file if not already present."""
    if not os.path.exists(_MODEL_PATH):
        print("[detector] Downloading hand_landmarker.task (~8 MB) …")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("[detector] Model downloaded.")
    return _MODEL_PATH


class HandDetector:
    def __init__(self, max_hands: int = 1, min_detection_confidence: float = 0.7):
        model_path = _ensure_model()

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> list[list[dict]]:
        """
        Detect hands in a BGR frame.

        Returns a list of hands (up to max_hands).
        Each hand is a list of 21 landmark dicts: {"x": float, "y": float, "z": float}.
        Returns [] when no hands are found or frame is None.
        """
        if frame is None:
            return []

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result: HandLandmarkerResult = self._landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return []

        hands_out = []
        for hand in result.hand_landmarks:
            landmarks = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand]
            hands_out.append(landmarks)

        return hands_out

    # ------------------------------------------------------------------
    def draw_landmarks(self, frame: np.ndarray, hands: list[list[dict]]) -> np.ndarray:
        """
        Draw hand skeleton onto a copy of frame using plain OpenCV.
        Works without mp.solutions.drawing_utils.

        Parameters
        ----------
        frame : BGR numpy array
        hands : output of detect()

        Returns annotated BGR frame.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        for hand in hands:
            pts = [(int(lm["x"] * w), int(lm["y"] * h)) for lm in hand]

            # Draw connections
            for a, b in _CONNECTIONS:
                cv2.line(annotated, pts[a], pts[b], (0, 220, 120), 2, cv2.LINE_AA)

            # Draw landmark dots
            for px, py in pts:
                cv2.circle(annotated, (px, py), 4, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(annotated, (px, py), 4, (0, 150, 255), 1, cv2.LINE_AA)

        return annotated
