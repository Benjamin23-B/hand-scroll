"""
nlg.py — Natural language generation for scene narration.

Stage 5: Template engine (primary, on-device, <1 ms).

Two sentence patterns:
  Simple  (1–5 new objects): "A {object} is {position}, {depth}."
  Complex (>5 new objects) : "You are in a scene with {enumerated list}."

No network calls — everything is pure string formatting.
"""

from __future__ import annotations

from scene_detector import Detection
from scene_intelligence import SpatialMapper


# ── Article helper ────────────────────────────────────────────────────────────

_VOWELS = set("aeiou")

def _article(word: str) -> str:
    return "an" if word and word[0].lower() in _VOWELS else "a"


# ── Simple template ───────────────────────────────────────────────────────────

def describe_single(det: Detection) -> str:
    """
    Generate a short sentence for a single detection.

    Example: "A person is on your left, close."
    """
    label    = det.label
    position = SpatialMapper.position(det)
    depth    = SpatialMapper.depth(det)
    article  = _article(label)

    # Refine position phrasing
    if position == "centre":
        pos_phrase = "ahead of you"
    else:
        pos_phrase = f"on your {position}"

    return f"{article.capitalize()} {label} is {pos_phrase}, {depth}."


# ── Complex template ──────────────────────────────────────────────────────────

def describe_scene(detections: list[Detection]) -> str:
    """
    Generate a richer summary sentence for a complex scene (>5 objects).

    Example: "You are in a busy scene — a person ahead, close, a chair on
    your left, mid distance, and a dog on your right, close."
    """
    if not detections:
        return ""

    parts: list[str] = []
    for det in detections:
        label    = det.label
        position = SpatialMapper.position(det)
        depth    = SpatialMapper.depth(det)
        article  = _article(label)

        if position == "centre":
            pos_phrase = "ahead"
        else:
            pos_phrase = f"on your {position}"

        parts.append(f"{article} {label} {pos_phrase}, {depth}")

    if len(parts) == 1:
        return f"You are in a scene with {parts[0]}."

    body = ", ".join(parts[:-1]) + f", and {parts[-1]}"
    return f"You are in a busy scene — {body}."


# ── Public entry point ────────────────────────────────────────────────────────

_COMPLEX_THRESHOLD = 5


def generate(detections: list[Detection]) -> list[str]:
    """
    Generate one or more sentences for the given list of new detections.

    Returns
    -------
    list[str]
        - For 1–N <= _COMPLEX_THRESHOLD detections: one sentence per detection.
        - For >_COMPLEX_THRESHOLD detections: a single complex summary sentence.
    """
    if not detections:
        return []

    if len(detections) > _COMPLEX_THRESHOLD:
        return [describe_scene(detections)]

    return [describe_single(det) for det in detections]
