"""
classifier.py — Pure-geometry gesture classifier + circle-motion detector.
No ML model — pure geometry on landmarks.

MediaPipe landmark indices used:
  0  = wrist
  4  = thumb_tip
  8  = index_tip,  6  = index_pip
  12 = middle_tip, 10 = middle_pip
  16 = ring_tip,   14 = ring_pip
  20 = pinky_tip,  18 = pinky_pip

Circle gesture (temporal):
  CircleDetector accumulates index_tip positions over a rolling window and
  fires when the trajectory forms a closed loop (radius CV < 0.40, angular
  coverage >= 300 degrees).
"""

import collections
import math


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def euclidean(a: dict, b: dict) -> float:
    """2-D Euclidean distance using only the x,y components."""
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2)


def finger_extended(tip: dict, pip: dict, mcp: dict, wrist: dict) -> bool:
    """
    Check if a finger is extended regardless of hand rotation.
    A finger is considered extended if:
    1. Distance from tip to wrist is significantly larger than PIP to wrist.
    2. Distance from tip to MCP (base knuckle) is significantly larger than PIP to MCP.
    """
    d_tip_wrist = euclidean(tip, wrist)
    d_pip_wrist = euclidean(pip, wrist)
    d_tip_mcp = euclidean(tip, mcp)
    d_pip_mcp = euclidean(pip, mcp)
    return d_tip_wrist > d_pip_wrist and d_tip_mcp > d_pip_mcp


def thumb_extended(thumb_tip: dict, thumb_ip: dict, thumb_mcp: dict, wrist: dict) -> bool:
    """Check if thumb is extended."""
    return euclidean(thumb_tip, wrist) > euclidean(thumb_ip, wrist) and euclidean(thumb_tip, thumb_mcp) > euclidean(thumb_ip, thumb_mcp)


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

def classify_gesture(landmarks: list[dict]) -> str:
    """
    Classify a single hand's 21 landmarks into a gesture string.

    Priority order:
      1. "pinch"        — thumb tip and index tip are pinched together
      2. "pointing"     — only index finger extended
      3. "two_fingers"  — index + middle extended (V-sign)
      4. "three_fingers"— index + middle + ring extended
      5. "open_palm"    — all 4 fingers extended
      6. "closed_fist"  — all 4 fingers curled
      7. "none"         — fallback
    """
    if len(landmarks) < 21:
        return "none"

    wrist      = landmarks[0]
    thumb_tip  = landmarks[4];  thumb_ip   = landmarks[3]; thumb_mcp = landmarks[2]
    index_tip  = landmarks[8];  index_pip  = landmarks[6]; index_mcp  = landmarks[5]
    middle_tip = landmarks[12]; middle_pip = landmarks[10]; middle_mcp = landmarks[9]
    ring_tip   = landmarks[16]; ring_pip   = landmarks[14]; ring_mcp   = landmarks[13]
    pinky_tip  = landmarks[20]; pinky_pip  = landmarks[18]; pinky_mcp  = landmarks[17]

    # Hand scale reference: distance from wrist (0) to middle finger MCP (9)
    hand_size = max(0.04, euclidean(wrist, middle_mcp))

    # --- Evaluate finger extensions (orientation-invariant) ---------------
    index_ext  = finger_extended(index_tip,  index_pip,  index_mcp,  wrist)
    middle_ext = finger_extended(middle_tip, middle_pip, middle_mcp, wrist)
    ring_ext   = finger_extended(ring_tip,   ring_pip,   ring_mcp,   wrist)
    pinky_ext  = finger_extended(pinky_tip,  pinky_pip,  pinky_mcp,  wrist)

    # --- 1. Pinch --------------------------------------------------------
    # Pinch occurs when thumb tip and index tip touch/are very close,
    # but not when fingers are curled into a closed fist.
    pinch_dist = euclidean(thumb_tip, index_tip)
    if pinch_dist < (0.28 * hand_size):
        # Ensure it's an intentional pinch and other fingers aren't completely curled into a tight fist
        if middle_ext or ring_ext or pinky_ext or index_ext:
            return "pinch"

    # --- 2. Pointing (only index finger extended) ------------------------
    if index_ext and not middle_ext and not ring_ext and not pinky_ext:
        return "pointing"

    # --- 3. Two fingers / V-sign (index + middle extended) ---------------
    if index_ext and middle_ext and not ring_ext and not pinky_ext:
        return "two_fingers"

    # --- 4. Three fingers (index + middle + ring extended) ---------------
    if index_ext and middle_ext and ring_ext and not pinky_ext:
        return "three_fingers"

    # --- 5. Open palm (all 4 extended) -----------------------------------
    if index_ext and middle_ext and ring_ext and pinky_ext:
        return "open_palm"

    # --- 6. Closed fist (all fingers curled) -----------------------------
    if not index_ext and not middle_ext and not ring_ext and not pinky_ext:
        return "closed_fist"

    # --- 7. Fallback -----------------------------------------------------
    return "none"


# ---------------------------------------------------------------------------
# Circle gesture detector  (temporal — must persist across frames)
# ---------------------------------------------------------------------------

class CircleDetector:
    """
    Detects a circular hand motion using the index-finger tip trajectory.

    Usage — call once per processed frame:
        detected = circle_detector.update(index_tip["x"], index_tip["y"])

    Parameters
    ----------
    buffer_size     : frames to keep in history (~1.5 s at 30 fps)
    min_radius      : minimum mean radius in normalised coords (filters tiny wiggles)
    cv_threshold    : max coefficient of variation (std/mean) of the radius;
                      lower = stricter circularity
    coverage_deg    : minimum angular arc that must be covered (out of 360)
    cooldown_frames : frames to ignore after detection, to avoid re-firing
    """

    def __init__(
        self,
        buffer_size: int = 45,
        min_radius: float = 0.04,
        cv_threshold: float = 0.40,
        coverage_deg: float = 300.0,
        cooldown_frames: int = 60,
    ) -> None:
        self._buffer: collections.deque[tuple[float, float]] = collections.deque(
            maxlen=buffer_size
        )
        self.min_radius = min_radius
        self.cv_threshold = cv_threshold
        self.coverage_deg = coverage_deg
        self.cooldown_frames = cooldown_frames
        self._cooldown_remaining: int = 0

    # ------------------------------------------------------------------
    def update(self, x: float, y: float) -> bool:
        """
        Feed the current index-tip position.
        Returns True exactly once when a full circle has been traced.
        """
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            self._buffer.clear()
            return False

        self._buffer.append((x, y))

        # Need a full buffer before evaluating
        if len(self._buffer) < self._buffer.maxlen:
            return False

        if self._is_circle():
            self._cooldown_remaining = self.cooldown_frames
            self._buffer.clear()
            return True

        return False

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear accumulated history (call when hand leaves the frame)."""
        self._buffer.clear()

    # ------------------------------------------------------------------
    def _is_circle(self) -> bool:
        pts = list(self._buffer)
        n = len(pts)

        # Centroid
        cx = sum(p[0] for p in pts) / n
        cy = sum(p[1] for p in pts) / n

        # Radii from centroid
        dists = [math.sqrt((p[0] - cx) ** 2 + (p[1] - cy) ** 2) for p in pts]
        mean_r = sum(dists) / n

        # Filter out tiny movements
        if mean_r < self.min_radius:
            return False

        # Coefficient of variation — how circular the path is
        std_r = math.sqrt(sum((d - mean_r) ** 2 for d in dists) / n)
        cv = std_r / mean_r
        if cv > self.cv_threshold:
            return False

        # Angular coverage — how much of the 360 arc is traced
        angles_deg = [
            round(math.degrees(math.atan2(p[1] - cy, p[0] - cx)) % 360)
            for p in pts
        ]
        unique_angles = sorted(set(angles_deg))
        if len(unique_angles) < 8:   # need enough distinct angle samples
            return False

        # Largest gap between consecutive angles (wrap-around included)
        max_gap = max(
            (unique_angles[(i + 1) % len(unique_angles)] - unique_angles[i]) % 360
            for i in range(len(unique_angles))
        )
        coverage = 360.0 - max_gap
        return coverage >= self.coverage_deg
