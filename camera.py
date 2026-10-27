"""
camera.py — Non-blocking webcam capture using OpenCV.
Runs a daemon thread that continuously grabs frames so the
caller always gets the latest one without blocking.
"""

import threading
import cv2
import numpy as np


class CameraCapture:
    def __init__(self, device_index: int = 0, width: int = 640, height: int = 480):
        self.device_index = device_index
        self.width = width
        self.height = height

        self._cap: cv2.VideoCapture | None = None
        self.frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Open the capture device and start the background reader thread."""
        self._cap = cv2.VideoCapture(self.device_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at device index {self.device_index}"
            )

        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True, name="CameraThread")
        self._thread.start()

    # ------------------------------------------------------------------
    def _reader(self) -> None:
        """Background loop: grab frames as fast as the camera allows."""
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self.frame = frame

    # ------------------------------------------------------------------
    def read(self) -> np.ndarray | None:
        """Return the latest captured frame (BGR numpy array) or None."""
        with self._lock:
            return self.frame.copy() if self.frame is not None else None

    # ------------------------------------------------------------------
    def stop(self) -> None:
        """Signal the reader thread to stop and release the device."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
        self._cap = None
