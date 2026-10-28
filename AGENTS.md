# AGENTS.md

## What this is
Flat, single-app Python project: webcam -> MediaPipe hand landmarks -> gesture state machine -> PyAutoGUI OS actions, plus an optional on-device TFLite scene-detection/TTS pipeline. `main.py` is the only entrypoint. There is no package metadata, build step, test suite, linter config, or CI.

Windows-first: the local `venv/` is a Windows venv, `run.bat` calls `venv\Scripts\python.exe`, OS actions use Windows hotkeys, and `actions.open_chrome()` shells out to `cmd /c start`. Those behaviors do not work unchanged on Linux/macOS.

## Run and verify
- Windows PowerShell: `.\venv\Scripts\python.exe main.py` (or `.\run.bat`)
- WSL/Linux shell from repo root: `./venv/Scripts/python.exe main.py` (this is the Windows interpreter; `venv/bin/python` does not exist)
- Options: `--camera-index N` (default 0), `--no-preview`
- Fresh environment (Windows): `python -m venv venv`, then `.\venv\Scripts\python.exe -m pip install -r requirements.txt`. `venv/` is gitignored, so a fresh clone has none. Requirements are unpinned and Windows-centric (`pywin32`); avoid bulk-upgrading the existing venv.
- Smoke check (no automated tests exist): `./venv/Scripts/python.exe -m compileall -q -x 'venv|scratch|__pycache__' .`
- Real verification is manual: run with a webcam, confirm the Tk overlay opens and gestures fire. Scene narration is off by default; press `S` and check the startup log for `SceneDetector.available` / `SpeechBridge.available`.

Imports are flat absolute modules (`import actions`), so always run from the repo root. Do not restructure into a package unless asked.

## Threading invariants
- `main.py:main()` constructs all components and starts daemon threads; `overlay.run()` blocks the main thread.
- MediaPipe `HandLandmarker` is not thread-safe. Call `HandDetector.detect()` only from `processing_loop`.
- Tkinter stays on the main thread. Cross-thread UI updates go only through `OverlayWindow.update_frame()` / `update_label()`.
- `SceneDetector.detect()` and TTS run on their own threads. `SpeechBridge` is a process-wide singleton; call `stop()` once at shutdown.
- `CameraCapture.read()` is the only thread-safe frame accessor; do not read the capture object elsewhere.

## Gesture action wiring
Pipeline: `classifier.classify_gesture` -> `GestureStateMachine.update` (`_ACTION_MAP` in `state_machine.py`) -> `actions.<name>()` resolved by `getattr` in `main.py` -> `_ACTION_LABELS` in `main.py` for the overlay.

Adding or changing a discrete action requires all three (`scroll` and `stop` are special-cased in `main.py`):
1. `_ACTION_MAP` entry keyed by `(gesture, direction)` with `continuous` / `axis` / `factor`
2. an exactly matching function name in `actions.py`
3. a display label in `main.py:_ACTION_LABELS`

`classify_gesture` order is priority-sensitive (pinch before pointing before two/three/open/fist). `_locked_dir` is fixed on the first post-hold frame and only resets when the gesture changes. `CircleDetector` is fed the index tip on non-pointing hand frames before the state machine; pointing bypasses the state machine and drives `CursorController` directly.

## Scene narration
`scene_loop` (10 FPS, despite the README's 3–5) -> `SceneDetector` -> `SceneDeltaEngine` -> `ScenePriorityQueue` -> `nlg.generate` -> `SpeechBridge.say`.

- Only new, moved, or stale objects are spoken; static objects are suppressed for `_STALE_SECONDS` (3s).
- `scene_detector.py` actually tries `mobilenet_ssd` before `efficientdet`, despite the docs calling EfficientDet primary.
- Tier lists, IoU threshold (0.45), and stale timer live at the top of `scene_intelligence.py`; confidence (0.55) and min box area live in `scene_detector.py`.
- Model files are committed at repo root and auto-download if deleted. Do not rename them without updating `_MODELS` in `scene_detector.py` or `detector.py`.

## Gotchas
- Tuning is done via constructor args in `main.py:main()`; there is no config or env file.
- `pyautogui.FAILSAFE = False` (in `main.py`) and `pyautogui.PAUSE = 0` (in `actions.py`, `cursor.py`, `scroll_engine.py`) are deliberate; do not "fix" them.
- Running `main.py` takes over the real cursor, scroll wheel, and global hotkeys with the fail-safe disabled. Do not launch it for automated verification without the user's consent.
- `main.py` binds `C`/`S` through the private `overlay._root`; renaming it silently disables shortcuts because the `AttributeError` is swallowed.
- In the preview, `Q` and `Esc` set `stop_event` but do not end the Tk mainloop; closing the overlay window is what exits the process.
- On Linux, TTS needs `espeak`. The system `python3` is not the project environment.
- `scratch/` and `*.log` are gitignored experiments/logs, not production inputs.
- The README is partly stale (scene FPS, EfficientDet priority, `Esc` behavior, config examples). Trust the code over the README when they diverge.
