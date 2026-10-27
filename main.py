"""
main.py — Entry point: wires all modules together.

Thread layout
─────────────
  main thread     → CameraCapture reader  (daemon)
                  → ScrollEngine loop     (daemon)
                  → Processing loop       (daemon)
                  → overlay.run() / Tkinter mainloop  ← blocking

NOTES
─────
• pyautogui.FAILSAFE = False  — prevents FailSafeException when moving
  the mouse to a corner at high speed.
• MediaPipe HandLandmarker is NOT thread-safe: all detect() calls happen
  only on the processing thread.
• Tkinter mainloop() MUST run on the main thread.
• Circle gesture launches Google Chrome via actions.open_chrome().
• All action names returned by GestureStateMachine map 1-to-1 to
  functions in actions.py (resolved with getattr).
"""

import argparse
import threading
import time

import cv2
import numpy as np
import pyautogui

import actions
from camera import CameraCapture
from classifier import CircleDetector, classify_gesture
from cursor import CursorController
from detector import HandDetector
from overlay import OverlayWindow
from scroll_engine import ScrollEngine
from state_machine import GestureStateMachine

# ── Scene narration modules ───────────────────────────────────────────────────
from scene_detector import SceneDetector
from scene_intelligence import SceneDeltaEngine, ScenePriorityQueue
from nlg import generate as nlg_generate
from speech_bridge import SpeechBridge

# Disable pyautogui fail-safe (rapid corner movement would raise otherwise)
pyautogui.FAILSAFE = False

# ── Human-readable labels for the overlay ─────────────────────────────────────
_ACTION_LABELS: dict[str, str] = {
    "scroll":              "↕ Scrolling",
    "stop":                "✊ Stopped",
    "browser_back":        "← Browser Back",
    "browser_forward":     "→ Browser Forward",
    "zoom_in":             "🔍 Zoom In",
    "zoom_out":            "🔎 Zoom Out",
    "right_click":         "🖱 Right Click",
    "task_view":           "🪟 Task View",
    "show_desktop":        "🖥 Show Desktop",
    "switch_desktop_left": "⟵ Desktop Left",
    "switch_desktop_right":"⟶ Desktop Right",
    "action_center":       "🔔 Action Centre",
    "minimize_all":        "⬇ Minimize All",
    "app_switch_forward":  "⇥ Next App",
    "app_switch_back":     "⇤ Prev App",
    "pointing_cursor":     "█ Moving Cursor",
}


def time_ms() -> float:
    return time.time() * 1000


# ── Processing loop ───────────────────────────────────────────────────────────

def processing_loop(
    camera: CameraCapture,
    detector: HandDetector,
    state_machine: GestureStateMachine,
    scroll_engine: ScrollEngine,
    circle_detector: CircleDetector,
    cursor: CursorController,
    overlay: OverlayWindow,
    stop_event: threading.Event,
    preview_enabled: threading.Event,
    scene_enabled: threading.Event,
    scene_detector: SceneDetector,
) -> None:
    """
    Main perception/action loop — runs at ~30 fps on its own daemon thread.
    MediaPipe detect() is called here and never from any other thread.
    """
    interval = 1 / 30

    def _render(annotated_frame: np.ndarray, label_text: str) -> None:
        overlay.update_frame(annotated_frame)
        overlay.update_label(label_text)

        if preview_enabled.is_set():
            disp = annotated_frame.copy()
            # Top status banner
            cv2.rectangle(disp, (0, 0), (disp.shape[1], 44), (20, 20, 36), -1)
            cv2.putText(
                disp,
                label_text,
                (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.78,
                (0, 255, 140),
                2,
                cv2.LINE_AA,
            )
            # Bottom shortcut bar
            cv2.rectangle(disp, (0, disp.shape[0] - 28), (disp.shape[1], disp.shape[0]), (20, 20, 36), -1)
            cv2.putText(
                disp,
                "[C] Toggle Preview  |  [S] Toggle Scene Voice  |  [Q] Quit",
                (14, disp.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 220),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow("Hand Scroll - Camera Preview", disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('c'), ord('C')):
                preview_enabled.clear()
                try:
                    cv2.destroyWindow("Hand Scroll - Camera Preview")
                except cv2.error:
                    pass
            elif key in (ord('s'), ord('S')):
                if scene_enabled.is_set():
                    scene_enabled.clear()
                    overlay.update_label("🔇 Scene narration OFF")
                elif scene_detector.available:
                    scene_enabled.set()
                    overlay.update_label("🔊 Scene narration ON")
            elif key in (ord('q'), ord('Q'), 27):
                stop_event.set()
        else:
            try:
                if cv2.getWindowProperty("Hand Scroll - Camera Preview", cv2.WND_PROP_VISIBLE) >= 1:
                    cv2.destroyWindow("Hand Scroll - Camera Preview")
            except cv2.error:
                pass

    try:
        while not stop_event.is_set():
            loop_start = time.perf_counter()

            # 1. Grab latest frame
            frame = camera.read()
            if frame is None:
                time.sleep(interval)
                continue

            # 2. Detect hands (MediaPipe — single-thread only)
            hands = detector.detect(frame)

            gesture = "none"
            palm_x  = 0.5
            palm_y  = 0.5

            # 3. Classify if a hand is present
            if hands:
                gesture = classify_gesture(hands[0])
                palm_x  = hands[0][0]["x"]   # wrist x
                palm_y  = hands[0][0]["y"]   # wrist y

                # 3a. Pointing gesture — move cursor in real time (bypasses state machine)
                if gesture == "pointing":
                    index_tip = hands[0][8]
                    cursor.move(index_tip["x"], index_tip["y"])

                    thumb_tip = hands[0][4]
                    index_pip = hands[0][6]
                    clicked = cursor.update_click(thumb_tip, index_pip)

                    annotated = detector.draw_landmarks(frame, hands)
                    click_msg = "☞ Moving Cursor (Click!)" if clicked else "☞ Moving Cursor"
                    _render(annotated, click_msg)

                    elapsed = time.perf_counter() - loop_start
                    if (interval - elapsed) > 0:
                        time.sleep(interval - elapsed)
                    continue
                else:
                    cursor.reset()   # clear EMA so next lock-on is instant

                # 3b. Circle detector — feed index tip every frame
                index_tip = hands[0][8]
                if circle_detector.update(index_tip["x"], index_tip["y"]):
                    actions.open_chrome()
                    frame_annotated = detector.draw_landmarks(frame, hands)
                    _render(frame_annotated, "🔵 Circle! Opening Chrome…")
                    time.sleep(0.8)   # hold the flash message briefly
                    elapsed = time.perf_counter() - loop_start
                    if (interval - elapsed) > 0:
                        time.sleep(interval - elapsed)
                    continue
            else:
                # No hand visible — clear trajectory and cursor EMA
                circle_detector.reset()
                cursor.reset()

            # 4. Feed state machine (returns action or None)
            result = state_machine.update(gesture, palm_x, palm_y, time_ms())

            # 5. Dispatch action
            overlay_label = f"Gesture: {gesture}"

            if result:
                action = result["action"]
                overlay_label = _ACTION_LABELS.get(action, action)

                if action == "scroll":
                    scroll_engine.apply_delta(result["scroll_delta"])

                elif action == "stop":
                    scroll_engine.apply_delta(0.0)

                else:
                    # Discrete action — look up and call the function in actions.py
                    fn = getattr(actions, action, None)
                    if fn is not None:
                        fn()
                    else:
                        print(f"[Hand Scroll] Warning: unknown action '{action}'")

            # 6. Annotate frame and push to overlay and preview window
            annotated = detector.draw_landmarks(frame, hands)
            _render(annotated, overlay_label)

            # 7. Throttle to ~30 fps
            elapsed = time.perf_counter() - loop_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


# ── Scene narration loop ──────────────────────────────────────────────────────

def scene_loop(
    camera: CameraCapture,
    scene_detector: SceneDetector,
    delta_engine: SceneDeltaEngine,
    priority_queue: ScenePriorityQueue,
    speech: SpeechBridge,
    enabled_flag: threading.Event,
    stop_event: threading.Event,
) -> None:
    """
    Scene narration loop — runs at 10 fps on its own daemon thread.

    Reads frames from the shared CameraCapture (thread-safe), runs
    EfficientDet-Lite0 inference, applies scene intelligence, generates
    natural-language sentences, and queues them for TTS.

    The loop respects enabled_flag: when cleared (S key pressed), inference
    is skipped entirely and the delta engine state is reset.
    """
    interval = 1 / 10  # 10 fps — enough for scene changes, cheap on CPU

    while not stop_event.is_set():
        loop_start = time.perf_counter()

        if not enabled_flag.is_set():
            # Scene narration disabled — reset delta state so the next
            # enable starts fresh (no stale suppression from old frames)
            delta_engine.reset()
            time.sleep(interval)
            continue

        if not scene_detector.available:
            time.sleep(interval)
            continue

        frame = camera.read()
        if frame is None:
            time.sleep(interval)
            continue

        # ── Inference ─────────────────────────────────────────────────
        detections = scene_detector.detect(frame)

        # ── Scene delta: keep only new / moved objects ─────────────────
        new_detections = delta_engine.update(detections)

        # ── Prioritise ────────────────────────────────────────────────
        ranked = priority_queue.ranked(new_detections)

        # ── Generate + speak (schedule with delays) ────────────────────
        for det, delay in ranked:
            sentences = nlg_generate([det])
            for sentence in sentences:
                if delay > 0:
                    time.sleep(min(delay, 1.5))   # cap wait to avoid long hangs
                speech.say(sentence)
                delta_engine.mark_announced(det)

        # ── Throttle to 10 fps ────────────────────────────────────────
        elapsed = time.perf_counter() - loop_start
        sleep_time = interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Hand Scroll - Gesture Screen Controller")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--no-preview", action="store_true", help="Disable the OpenCV camera preview window")
    args = parser.parse_args()

    # ── Instantiate components ────────────────────────────────────────
    camera        = CameraCapture(device_index=args.camera_index, width=640, height=480)
    detector      = HandDetector(max_hands=1, min_detection_confidence=0.7)
    state_machine = GestureStateMachine(hold_ms=200, cooldown_ms=150)
    scroll_engine = ScrollEngine(fps=60, easing=0.12, decay=0.88)
    circle_detector = CircleDetector(
        buffer_size=45,
        min_radius=0.04,
        cv_threshold=0.40,
        coverage_deg=300.0,
        cooldown_frames=60,
    )
    cursor  = CursorController(smoothing=0.70, active_margin=0.20)  # fast response, easier edges
    overlay = OverlayWindow()

    # ── Scene narration components ────────────────────────────────────
    scene_detector  = SceneDetector()
    delta_engine    = SceneDeltaEngine()
    priority_queue  = ScenePriorityQueue()
    speech          = SpeechBridge.instance()
    scene_enabled   = threading.Event()   # cleared = off, set = on
    preview_enabled = threading.Event()
    if not args.no_preview:
        preview_enabled.set()

    stop_event = threading.Event()

    # ── Start subsystems ──────────────────────────────────────────────
    camera.start()
    scroll_engine.start()
    overlay.start()   # creates window, does NOT block

    # ── Processing thread (daemon) ────────────────────────────────────
    proc_thread = threading.Thread(
        target=processing_loop,
        args=(camera, detector, state_machine, scroll_engine,
              circle_detector, cursor, overlay, stop_event,
              preview_enabled, scene_enabled, scene_detector),
        daemon=True,
        name="ProcessingThread",
    )
    proc_thread.start()

    # ── Scene narration thread (daemon) ───────────────────────────────
    scene_thread = threading.Thread(
        target=scene_loop,
        args=(camera, scene_detector, delta_engine, priority_queue,
              speech, scene_enabled, stop_event),
        daemon=True,
        name="SceneThread",
    )
    scene_thread.start()

    # ── Preview toggle (C key & video thumbnail click) ───────────────
    def _toggle_preview(event=None) -> None:  # noqa: ANN001
        if preview_enabled.is_set():
            preview_enabled.clear()
            overlay.update_label("📷 Camera preview OFF")
            print("[Hand Scroll] Camera preview window hidden.")
        else:
            preview_enabled.set()
            overlay.update_label("📷 Camera preview ON")
            print("[Hand Scroll] Camera preview window shown.")

    overlay.on_preview_toggle = _toggle_preview

    # ── S-key toggle for scene narration ─────────────────────────────
    def _toggle_scene(event=None) -> None:  # noqa: ANN001
        if scene_enabled.is_set():
            scene_enabled.clear()
            overlay.update_label("🔇 Scene narration OFF")
            print("[Hand Scroll] Scene narration disabled.")
        else:
            if not scene_detector.available:
                overlay.update_label("⚠ Scene model unavailable")
                print("[Hand Scroll] Scene model unavailable — cannot enable narration.")
                return
            scene_enabled.set()
            overlay.update_label("🔊 Scene narration ON")
            print("[Hand Scroll] Scene narration enabled.")

    # Bind shortcuts on the overlay Tkinter window
    try:
        overlay._root.bind("<KeyPress-c>", _toggle_preview)
        overlay._root.bind("<KeyPress-C>", _toggle_preview)
        overlay._root.bind("<KeyPress-s>", _toggle_scene)
        overlay._root.bind("<KeyPress-S>", _toggle_scene)
    except AttributeError:
        # overlay._root may not be exposed — silently skip key binding
        pass

    print("[Hand Scroll] Running — close the overlay window, press Q in preview, or Ctrl-C to quit.")
    print("[Hand Scroll] Camera feed:")
    print("  Live preview window : ON (press C or click thumbnail to toggle)")
    print("[Hand Scroll] Gesture map:")
    print("  Open palm  up/down  -> Scroll      left/right -> Browser back/forward")
    print("  Pinch      up/down  -> Zoom in/out")
    print("  V-sign     still    -> Right click  up -> Task View  down -> Desktop")
    print("  V-sign     left/right -> Virtual desktop switch")
    print("  3 fingers  still    -> Action Centre  up -> Task View  down -> Min all")
    print("  3 fingers  left/right -> App switch")
    print("  Fist                -> Stop scroll")
    print("  Circle (index tip)  -> Open Chrome")
    print()
    print("[Hand Scroll] Scene narration:")
    print("  Press S in overlay / preview to toggle scene narration ON/OFF")
    model_status = "ready" if scene_detector.available else "UNAVAILABLE (install tflite-runtime)"
    print(f"  Detection model : {model_status}")
    speech_status  = "ready" if speech.available else "UNAVAILABLE (install pyttsx3)"
    print(f"  TTS engine      : {speech_status}")

    try:
        # Blocks until the Tkinter window is closed
        overlay.run()
    except KeyboardInterrupt:
        print("\n[Hand Scroll] Interrupted — shutting down …")
    finally:
        stop_event.set()
        scene_enabled.clear()
        proc_thread.join(timeout=2.0)
        scene_thread.join(timeout=2.0)
        speech.stop()
        scroll_engine.stop()
        camera.stop()
        print("[Hand Scroll] Stopped.")


if __name__ == "__main__":
    main()
