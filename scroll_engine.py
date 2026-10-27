"""
scroll_engine.py — Smooth, inertia-based scroll via pyautogui.

The engine runs a fixed-rate daemon thread.  Callers push a raw delta via
apply_delta(); the loop lerps toward it and applies exponential decay so
scrolling eases to a stop naturally.
"""

import threading
import time

import pyautogui


# Remove pyautogui's built-in inter-call pause so our own timing governs.
pyautogui.PAUSE = 0


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation: a + (b - a) * t"""
    return a + (b - a) * t


class ScrollEngine:
    def __init__(self, fps: int = 60, easing: float = 0.12, decay: float = 0.88):
        """
        Parameters
        ----------
        fps    : tick rate of the scroll loop
        easing : lerp factor toward target each tick (0 < easing < 1)
        decay  : multiplicative decay applied to target each tick
                 (0 < decay < 1) — produces natural inertia stop
        """
        self.fps = fps
        self.easing = easing
        self.decay = decay

        self._target_delta: float = 0.0
        self._current: float = 0.0
        self._running: bool = False
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the background scroll loop."""
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="ScrollThread"
        )
        self._thread.start()

    # ------------------------------------------------------------------
    def stop(self) -> None:
        """Stop the scroll loop gracefully."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    def apply_delta(self, delta: float) -> None:
        """
        Set the target scroll velocity.
        Positive delta → scroll down; negative → scroll up.
        """
        with self._lock:
            self._target_delta = delta

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        interval = 1.0 / self.fps
        while self._running:
            start = time.perf_counter()

            with self._lock:
                target = self._target_delta
                # Lerp current value toward target
                self._current = lerp(self._current, target, self.easing)
                # Apply decay to target so it drifts back to zero
                self._target_delta *= self.decay

            current_snap = self._current  # read outside lock for pyautogui call

            if abs(current_snap) > 0.5:
                # pyautogui.scroll positive = up, so negate our delta
                pyautogui.scroll(-int(current_snap))
            else:
                # Below threshold: clamp to zero to stop residual drift
                with self._lock:
                    self._current = 0.0

            elapsed = time.perf_counter() - start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
