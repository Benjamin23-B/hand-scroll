"""
scene_intelligence.py — Stage 4 of the scene narration pipeline.

Three components, all pure Python / numpy — no ML:

  SpatialMapper    — maps bounding box centre → position + depth labels
  SceneDeltaEngine — IoU-based change detection; suppresses stale objects
  PriorityQueue    — tier-based ordering of new detections

These run on the SceneThread, never on the UI thread.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from scene_detector import Detection


# ── Tier definitions ──────────────────────────────────────────────────────────
# Tier 1: hazard / navigation — announce immediately
_TIER_1 = {
    "person", "dog", "cat", "car", "truck", "bus", "motorcycle",
    "bicycle", "stairs", "door",
}
# Tier 2: furniture / obstacles — announce after 500 ms
_TIER_2 = {
    "chair", "couch", "bed", "dining table", "toilet", "bench",
    "refrigerator", "oven", "sink", "tv",
}
# Everything else is Tier 3 (announce on-demand / low priority)

_TIER_DELAY: dict[int, float] = {1: 0.0, 2: 0.5, 3: 999.0}

# Stale suppression — do not re-announce an object that hasn't moved for this long
_STALE_SECONDS = 3.0

# IoU threshold below which a detection is considered "moved" vs previous frame
_MOVE_IOU_THRESHOLD = 0.45


# ── Helpers ───────────────────────────────────────────────────────────────────

def _iou(a: Detection, b: Detection) -> float:
    """Intersection over union of two Detection bounding boxes."""
    ix1 = max(a.xmin, b.xmin)
    iy1 = max(a.ymin, b.ymin)
    ix2 = min(a.xmax, b.xmax)
    iy2 = min(a.ymax, b.ymax)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _tier(label: str) -> int:
    if label in _TIER_1:
        return 1
    if label in _TIER_2:
        return 2
    return 3


# ── SpatialMapper ─────────────────────────────────────────────────────────────

class SpatialMapper:
    """Maps a Detection's bounding box to human-readable position and depth."""

    # X-axis bins (normalised 0-1)
    _X_BINS = [
        (0.00, 0.20, "far left"),
        (0.20, 0.40, "left"),
        (0.40, 0.60, "centre"),
        (0.60, 0.80, "right"),
        (0.80, 1.00, "far right"),
    ]

    # Area-based depth (fraction of total frame area)
    _DEPTH_THRESHOLDS = [
        (0.25, "very close"),
        (0.08, "close"),
        (0.02, "mid distance"),
    ]

    @classmethod
    def position(cls, det: Detection) -> str:
        cx = det.cx
        for lo, hi, label in cls._X_BINS:
            if lo <= cx < hi:
                return label
        return "far right"

    @classmethod
    def depth(cls, det: Detection) -> str:
        area = det.area
        for threshold, label in cls._DEPTH_THRESHOLDS:
            if area >= threshold:
                return label
        return "far away"


# ── SceneDeltaEngine ──────────────────────────────────────────────────────────

@dataclass
class _TrackedObject:
    detection: Detection
    last_announced: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)


class SceneDeltaEngine:
    """
    Compares the current frame's detections against the previous frame.

    An object is NEW (worth announcing) if:
      - No same-label detection existed in the previous frame, OR
      - Its IoU with the closest previous same-label detection < _MOVE_IOU_THRESHOLD
        (it has moved significantly)

    An object is STALE (suppressed) if:
      - It has been continuously present and announced within _STALE_SECONDS
    """

    def __init__(self) -> None:
        # label → list of tracked objects currently in scene
        self._tracked: dict[str, list[_TrackedObject]] = {}

    # ------------------------------------------------------------------
    def update(self, detections: list[Detection]) -> list[Detection]:
        """
        Feed the current frame's detections.

        Returns the subset that are new or moved (not stale).
        """
        now = time.time()
        new_objects: list[Detection] = []

        # Group incoming by label
        by_label: dict[str, list[Detection]] = {}
        for det in detections:
            by_label.setdefault(det.label, []).append(det)

        # Process each label group
        next_tracked: dict[str, list[_TrackedObject]] = {}

        for label, dets in by_label.items():
            prev_list = self._tracked.get(label, [])
            next_list: list[_TrackedObject] = []

            for det in dets:
                matched_prev: _TrackedObject | None = None
                best_iou = 0.0

                # Find best IoU match in previous frame
                for prev in prev_list:
                    iou = _iou(det, prev.detection)
                    if iou > best_iou:
                        best_iou = iou
                        matched_prev = prev

                if matched_prev is None:
                    # Brand new object
                    new_objects.append(det)
                    tracked = _TrackedObject(detection=det)
                elif best_iou < _MOVE_IOU_THRESHOLD:
                    # Object moved significantly
                    new_objects.append(det)
                    tracked = _TrackedObject(detection=det)
                else:
                    # Same object, check staleness
                    tracked = _TrackedObject(
                        detection=det,
                        last_announced=matched_prev.last_announced,
                        last_seen=now,
                    )
                    if now - matched_prev.last_announced >= _STALE_SECONDS:
                        # Stale — re-announce
                        new_objects.append(det)
                        tracked.last_announced = now

                next_list.append(tracked)

            next_tracked[label] = next_list

        self._tracked = next_tracked
        return new_objects

    def mark_announced(self, det: Detection) -> None:
        """
        Call after a detection has been spoken so the stale timer resets.
        """
        now = time.time()
        for tracked in self._tracked.get(det.label, []):
            iou = _iou(det, tracked.detection)
            if iou > 0.3:
                tracked.last_announced = now
                break

    def reset(self) -> None:
        """Clear all tracked state (e.g. when scene narration is disabled)."""
        self._tracked.clear()


# ── PriorityQueue ─────────────────────────────────────────────────────────────

class ScenePriorityQueue:
    """
    Orders new detections by tier, then confidence.

    Tier 1 objects come first (immediate), Tier 2 after 500 ms delay,
    Tier 3 are suppressed unless there are fewer than 2 Tier-1/2 objects.
    """

    def ranked(self, detections: list[Detection]) -> list[tuple[Detection, float]]:
        """
        Returns a list of (Detection, delay_seconds) pairs, sorted by priority.
        The delay is the recommended wait before speaking the item.
        """
        result: list[tuple[Detection, float, int]] = []

        for det in detections:
            t = _tier(det.label)
            delay = _DELAY_ACCUMULATOR = _TIER_DELAY[t]
            result.append((det, delay, t))

        # Sort: tier ascending, then confidence descending
        result.sort(key=lambda x: (x[2], -x[0].confidence))

        # Accumulate sequential delays so objects don't overlap
        # Each subsequent announcement waits for the previous + 0.7 s gap
        out: list[tuple[Detection, float]] = []
        cumulative = 0.0
        for det, base_delay, tier in result:
            speak_at = max(cumulative, base_delay)
            out.append((det, speak_at))
            cumulative = speak_at + 0.8   # 800 ms between announcements

        return out
