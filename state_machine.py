"""
state_machine.py — Directional gesture state machine.

Each gesture is debounced (hold_ms) then mapped to an action based on the
direction of cumulative wrist movement since the gesture started.

  Discrete gestures : fire once per gesture arm, then wait for a gesture change.
  Continuous gestures: fire every update tick while held (e.g. scroll).
"""

from __future__ import annotations

# ── Direction detection ───────────────────────────────────────────────────────

# Minimum cumulative displacement (normalised 0-1 coords) to classify as directional
_DIR_THRESHOLD = 0.03


def _direction(cum_dx: float, cum_dy: float) -> str:
    """Map cumulative 2-D displacement to a cardinal direction string."""
    adx, ady = abs(cum_dx), abs(cum_dy)
    if adx < _DIR_THRESHOLD and ady < _DIR_THRESHOLD:
        return "none"
    if adx >= ady:
        return "right" if cum_dx > 0 else "left"
    return "down" if cum_dy > 0 else "up"


# ── Action lookup table ───────────────────────────────────────────────────────
# Keys   : (gesture_string, direction_string)
# Values : action descriptor dict
#   "action"     : str   — name of function in actions.py  (or "scroll"/"stop")
#   "continuous" : bool  — True → fires every tick; False → fires once per arm
#   "axis"       : "v"|"h"  — for scroll only, which axis
#   "factor"     : float    — for scroll only, multiplies frame delta

_ACTION_MAP: dict[tuple[str, str], dict] = {

    # ── Open palm ─────────────────────────────────────────────────────────
    # Still / vertical → continuous scroll
    ("open_palm", "none"):   {"action": "scroll",              "continuous": True,  "axis": "v", "factor":  800},
    ("open_palm", "up"):     {"action": "scroll",              "continuous": True,  "axis": "v", "factor": -800},
    ("open_palm", "down"):   {"action": "scroll",              "continuous": True,  "axis": "v", "factor":  800},
    # Horizontal → browser nav (discrete)
    ("open_palm", "left"):   {"action": "browser_back",        "continuous": False},
    ("open_palm", "right"):  {"action": "browser_forward",     "continuous": False},

    # ── Pinch ─────────────────────────────────────────────────────────────
    ("pinch",     "up"):     {"action": "zoom_in",             "continuous": False},
    ("pinch",     "down"):   {"action": "zoom_out",            "continuous": False},
    # Horizontal pinch → browser nav
    ("pinch",     "left"):   {"action": "browser_back",        "continuous": False},
    ("pinch",     "right"):  {"action": "browser_forward",     "continuous": False},

    # ── Two fingers (index + middle — V-sign) ─────────────────────────────
    ("two_fingers", "none"):  {"action": "right_click",         "continuous": False},
    ("two_fingers", "up"):    {"action": "task_view",           "continuous": False},
    ("two_fingers", "down"):  {"action": "show_desktop",        "continuous": False},
    ("two_fingers", "left"):  {"action": "switch_desktop_left", "continuous": False},
    ("two_fingers", "right"): {"action": "switch_desktop_right","continuous": False},

    # ── Three fingers (index + middle + ring) ─────────────────────────────
    ("three_fingers", "none"):  {"action": "action_center",     "continuous": False},
    ("three_fingers", "up"):    {"action": "task_view",         "continuous": False},
    ("three_fingers", "down"):  {"action": "minimize_all",      "continuous": False},
    ("three_fingers", "left"):  {"action": "app_switch_back",   "continuous": False},
    ("three_fingers", "right"): {"action": "app_switch_forward","continuous": False},

    # ── Closed fist — stop scrolling ──────────────────────────────────────
    ("closed_fist", "none"):  {"action": "stop", "continuous": True},
    ("closed_fist", "up"):    {"action": "stop", "continuous": True},
    ("closed_fist", "down"):  {"action": "stop", "continuous": True},
    ("closed_fist", "left"):  {"action": "stop", "continuous": True},
    ("closed_fist", "right"): {"action": "stop", "continuous": True},
}


# ── State machine ─────────────────────────────────────────────────────────────

class GestureStateMachine:
    def __init__(self, hold_ms: float = 200, cooldown_ms: float = 150):
        self.hold_ms = hold_ms
        self.cooldown_ms = cooldown_ms

        # Candidate (unconfirmed) gesture tracking
        self._candidate: str = "none"
        self._candidate_start_ms: float = 0.0
        self._candidate_start_pos: tuple[float, float] = (0.5, 0.5)

        # Per-frame position (for continuous scroll delta)
        self._prev_pos: tuple[float, float] = (0.5, 0.5)

        # Post-fire cooldown timestamp
        self._last_fire_ms: float = 0.0

        # Discrete guard: once fired, require gesture change before re-firing
        self._discrete_fired: bool = False

        # Direction lock: set on the first confirmed frame, cleared on gesture change
        self._locked_dir: str | None = None

    # ── Public API ───────────────────────────────────────────────────────────

    def update(
        self,
        gesture: str,
        palm_x: float,
        palm_y: float,
        timestamp_ms: float,
    ) -> dict | None:
        """
        Feed the latest classified gesture and wrist position.

        Parameters
        ----------
        gesture      : string from classify_gesture()
        palm_x       : normalised X of landmark[0] (wrist)
        palm_y       : normalised Y of landmark[0] (wrist)
        timestamp_ms : current time in milliseconds

        Returns
        -------
        dict  or  None
          Scroll:   {"action": "scroll",  "scroll_delta": float}
          Stop:     {"action": "stop",    "scroll_delta": 0.0}
          Discrete: {"action": "<name>"}  (name matches a function in actions.py)
          None: nothing to do this tick
        """
        now = timestamp_ms

        # ── Cooldown guard ────────────────────────────────────────────────
        if now - self._last_fire_ms < self.cooldown_ms:
            self._prev_pos = (palm_x, palm_y)
            return None

        # ── Gesture change → reset all candidate state ────────────────────
        if gesture != self._candidate:
            self._candidate            = gesture
            self._candidate_start_ms   = now
            self._candidate_start_pos  = (palm_x, palm_y)
            self._prev_pos             = (palm_x, palm_y)
            self._discrete_fired       = False
            self._locked_dir           = None
            return None

        # ── Hold check ────────────────────────────────────────────────────
        if (now - self._candidate_start_ms) < self.hold_ms:
            self._prev_pos = (palm_x, palm_y)
            return None

        # ── Direction (lock once on first confirmed frame) ─────────────────
        sx, sy = self._candidate_start_pos
        cum_dx = palm_x - sx
        cum_dy = palm_y - sy

        if self._locked_dir is None:
            self._locked_dir = _direction(cum_dx, cum_dy)

        locked = self._locked_dir

        # ── Action lookup ─────────────────────────────────────────────────
        spec = _ACTION_MAP.get((gesture, locked))
        if spec is None:
            # Gesture has no mapping for this direction — do nothing
            self._prev_pos = (palm_x, palm_y)
            return None

        # ── Discrete: fire only once per gesture arm ──────────────────────
        if not spec.get("continuous", False):
            if self._discrete_fired:
                self._prev_pos = (palm_x, palm_y)
                return None
            self._discrete_fired = True

        # ── Build & return result ─────────────────────────────────────────
        prev_x, prev_y = self._prev_pos
        frame_dx = palm_x - prev_x
        frame_dy = palm_y - prev_y

        self._last_fire_ms = now
        self._prev_pos = (palm_x, palm_y)
        return self._build_result(spec, frame_dx, frame_dy)

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    def _build_result(self, spec: dict, frame_dx: float, frame_dy: float) -> dict:
        action = spec["action"]

        if action == "scroll":
            axis   = spec.get("axis",   "v")
            factor = spec.get("factor", 800)
            raw    = frame_dy * factor if axis == "v" else frame_dx * factor
            return {"action": "scroll", "scroll_delta": self._clamp(raw, -80, 80)}

        if action == "stop":
            return {"action": "stop", "scroll_delta": 0.0}

        # All other actions are discrete one-shots; return the action name
        return {"action": action}
