"""
cursor.py — Smooth cursor control + left-click for the pointing gesture.

Cursor movement
───────────────
Uses an Exponential Moving Average (EMA) to suppress jitter from raw
MediaPipe landmark positions before calling pyautogui.moveTo().

The webcam image is horizontally mirrored (selfie view), so norm_x is
flipped before mapping to screen coordinates.

Left click
──────────
Detected by proximity of the thumb tip to the index PIP (proximal knuckle).
This is the 'air tap' motion: index stays extended while the thumb taps the
knuckle. Debounced: one click fires when the thumb enters the zone; no
re-fire until the thumb moves back out (prevents rapid-fire spam).
"""

import math
import pyautogui

pyautogui.PAUSE = 0


# Distance threshold (normalised coords) for thumb-to-index-PIP click
_CLICK_THRESHOLD = 0.06


class CursorController:
    def __init__(self, smoothing: float = 0.70, active_margin: float = 0.20):
        """
        Parameters
        ----------
        smoothing     : EMA factor α  (0 < α ≤ 1).
                        1.0 → raw / no smoothing (very jittery).
                        0.70 → snappy but with slight jitter suppression.
        active_margin : Float [0.0-0.4] describing the margin around the camera frame
                        that is ignored. This means you don't have to reach the
                        physical edge of the camera view to hit your screen edges.
        """
        self.smoothing = smoothing
        self.active_margin = active_margin
        self._screen_w, self._screen_h = pyautogui.size()

        # Smoothed screen-space position (None until first sample)
        self._sx: float | None = None
        self._sy: float | None = None

        # Click debounce: True while thumb is inside the click zone
        self._clicking: bool = False

    # ------------------------------------------------------------------
    def move(self, norm_x: float, norm_y: float) -> None:
        """
        Move the OS cursor to the position indicated by normalised coords.

        Parameters
        ----------
        norm_x : normalised x from MediaPipe  (0 = left of frame,  1 = right)
        norm_y : normalised y from MediaPipe  (0 = top  of frame,  1 = bottom)
        """
        # Map a central "active zone" of the camera to the full screen.
        # This means you don't have to stretch your hand to the very edges of the frame.
        margin = self.active_margin
        usable = 1.0 - (margin * 2.0)
        
        active_x = max(0.0, min(1.0, (norm_x - margin) / usable))
        active_y = max(0.0, min(1.0, (norm_y - margin) / usable))

        # Flip x: webcam is mirrored — moving hand right should move cursor right
        target_x = (1.0 - active_x) * self._screen_w
        target_y = active_y * self._screen_h

        if self._sx is None:
            # First sample: jump straight to position (no lag on initial lock-on)
            self._sx = target_x
            self._sy = target_y
        else:
            α = self.smoothing
            self._sx += α * (target_x - self._sx)
            self._sy += α * (target_y - self._sy)

        pyautogui.moveTo(int(self._sx), int(self._sy))

    # ------------------------------------------------------------------
    def update_click(self, thumb_tip: dict, index_pip: dict) -> bool:
        """
        Check if the thumb tip is tapping the index PIP joint and fire a
        left click if it just entered the zone (debounced).

        Parameters
        ----------
        thumb_tip  : landmark dict {x, y, z}  — landmark 4
        index_pip  : landmark dict {x, y, z}  — landmark 6

        Returns
        -------
        True if a click was fired this call, False otherwise.
        """
        dist = math.sqrt(
            (thumb_tip["x"] - index_pip["x"]) ** 2
            + (thumb_tip["y"] - index_pip["y"]) ** 2
        )
        in_zone = dist < _CLICK_THRESHOLD

        if in_zone and not self._clicking:
            # Thumb just entered the zone — fire one click
            pyautogui.click()
            self._clicking = True
            return True

        if not in_zone:
            # Thumb left the zone — arm for the next click
            self._clicking = False

        return False

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """
        Clear the smoothed position and click state.
        Call this when the pointing gesture ends so the next lock-on
        jumps instantly rather than sliding in from the last position.
        """
        self._sx = None
        self._sy = None
        self._clicking = False
