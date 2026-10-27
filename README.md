# ✋ Hand Scroll & Scene Intelligence

> Touchless OS navigation & intelligent real-time spatial scene narration powered by on-device computer vision.

Control your screen's cursor, scroll wheel, and window manager using intuitive hand gestures — and optionally enable on-device spatial scene narration that detects objects in your environment, prioritizes hazards, and announces them via text-to-speech without cloud dependencies.

---

## 📑 Table of Contents

- [Overview & Architecture](#-overview--architecture)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Gesture Reference](#-gesture-reference)
- [Scene Narration & Assistive Vision](#-scene-narration--assistive-vision)
- [Installation & Setup](#-installation--setup)
- [How to Use & Controls](#-how-to-use--controls)
- [Project Architecture & File Map](#-project-architecture--file-map)
- [Configuration & Tuning](#-configuration--tuning)
- [Troubleshooting](#-troubleshooting)
- [Platform Notes](#-platform-notes)

---

## 🔭 Overview & Architecture

**Hand Scroll** is an assistive, touchless computer interaction system that runs fully locally on your machine. It captures webcam video and operates two distinct pipelines in parallel without UI blocking:

1. **Hand Tracking & Gesture Pipeline (~30–60 FPS):** Extracts 21 landmark coordinates per hand via MediaPipe Tasks Vision, computes geometric spatial relationships, runs debounce & direction state machines, and executes operating system actions (scrolls, mouse clicks, window switching, zooming).
2. **Scene Intelligence & Narration Pipeline (~3–5 FPS):** Periodically inspects frames using an on-device INT8 quantized object detection model (EfficientDet-Lite0 / MobileNet-SSD), performs IoU tracking & spatial depth mapping, orders hazards via a priority queue, generates natural language descriptions, and vocalizes alerts via non-blocking TTS.

### Multi-Thread Layout

```
Main Thread
│
├── CameraCapture._reader   (Daemon)  ── Continuously polls camera buffer (drop-oldest queue)
├── ScrollEngine._loop      (Daemon)  ── 60 Hz smooth inertia physics (lerp + decay + pyautogui)
├── processing_loop         (Daemon)  ── 30 Hz Hand Landmark Detection → Classifier → State Machine
├── scene_loop              (Daemon)  ── 3 Hz TFLite Object Detection → Delta Tracking → Spatial NLG
├── SpeechBridge._worker    (Daemon)  ── Thread-safe pyttsx3 audio worker queue
└── Tkinter Mainloop        (Blocking)── Real-time floating overlay widget & OpenCV preview
```

---

## ✨ Key Features

- 🖐 **Full Touchpad Emulation:** Translates desktop precision touchpad gestures into natural airborne hand gestures.
- 🎯 **Pointing & Cursor Control:** Point your index finger to pilot your mouse cursor smoothly with Exponential Moving Average (EMA) jitter filtering. Tap thumb to trigger left clicks.
- ↕ **Smooth Inertia Scrolling:** Move your open palm up/down to scroll with custom lerp acceleration and friction decay.
- 🤏 **Pinch-to-Zoom:** Touch index and thumb tips together and move up/down to zoom in or out.
- ✌ **Multi-Finger Directional Gestures:** 2-finger and 3-finger poses trigger Task View, Virtual Desktop navigation, Show Desktop, Action Centre, and App Switching.
- ✊ **Fist Freeze:** Close your hand to instantly kill scroll momentum.
- 🔵 **Temporal Gesture Recognition:** Trace a circle in the air with your index finger to instantly launch Google Chrome.
- 👁 **On-Device Scene Intelligence:** Detects 80 COCO classes locally via TFLite INT8, categorizing objects into Tier 1 (immediate hazards like people, cars, stairs) and Tier 2 (furniture, obstacles), suppressing static stale objects.
- 🗣 **Local Audio Narration:** Synthesizes natural spoken spatial feedback (e.g., *"A person is ahead of you, close."*) with zero network latency or privacy leakage.
- 🪟 **Floating UI Overlay:** A sleek, dark-mode, always-on-top Tkinter status widget displaying real-time live video and action status.

---

## 🛠 Tech Stack

| Technology / Library | Purpose |
|---|---|
| **Python 3.10+** | Core runtime utilizing modern typing, match-case semantics, and multithreading. |
| **MediaPipe Tasks API (`>= 0.10.0`)** | State-of-the-art on-device hand landmark detection (`hand_landmarker.task`) with 21 3D keypoints. |
| **OpenCV (`opencv-python`)** | Camera frame capture, color conversion, geometric transforms, and HUD skeleton rendering. |
| **TensorFlow / `tflite-runtime`** | Edge inference engine running EfficientDet-Lite0 / MobileNet-SSD INT8 models. |
| **PyAutoGUI** | Low-level OS event synthesis (mouse move, click, scroll wheel, system hotkeys). |
| **Tkinter & Pillow (PIL)** | Always-on-top translucent desktop HUD overlay displaying live webcam thumbnail and status. |
| **pyttsx3** | Cross-platform, fully offline Text-to-Speech audio engine. |
| **NumPy** | High-performance coordinate vectorization, Euclidean distance calculation, and IoU matching. |

---

## 🖐 Gesture Reference

All gestures are performed with a single hand. Direction is calculated from the cumulative wrist displacement once the pose is held for **200 ms**.

### 1. Open Palm (All fingers extended)
- **Move Up / Down:** Smooth Scroll Up / Down
- **Move Left:** Browser Back (`Alt` + `←`)
- **Move Right:** Browser Forward (`Alt` + `→`)

### 2. Pointing (Only index finger extended)
- **Move Index Tip:** Move the OS cursor smoothly across your display.
- **Tap Thumb to Index Base:** Left Click at cursor position.

### 3. Pinch (Thumb tip touching Index tip)
- **Move Up:** Zoom In (`Ctrl` + Scroll Up)
- **Move Down:** Zoom Out (`Ctrl` + Scroll Down)

### 4. V-Sign / Two Fingers (Index + Middle extended)
- **Hold Still:** Right Click
- **Move Up:** Windows Task View (`Win` + `Tab`)
- **Move Down:** Show / Hide Desktop (`Win` + `D`)
- **Move Left:** Switch to Left Virtual Desktop (`Ctrl` + `Win` + `←`)
- **Move Right:** Switch to Right Virtual Desktop (`Ctrl` + `Win` + `→`)

### 5. Three Fingers (Index + Middle + Ring extended)
- **Hold Still:** Windows Action Centre (`Win` + `A`)
- **Move Up:** Task View (`Win` + `Tab`)
- **Move Down:** Minimize All Windows (`Win` + `M`)
- **Move Left:** Previous Application (`Alt` + `Shift` + `Tab`)
- **Move Right:** Next Application (`Alt` + `Tab`)

### 6. Closed Fist
- **Hold:** Instantly freeze / stop scrolling momentum.

### 7. Air Circle
- **Draw Loop with Index Tip:** Tracing ≥ 300° of an arc within 1.5 seconds automatically launches Google Chrome.

---

## 👁 Scene Narration & Assistive Vision

When Scene Narration is enabled (<kbd>S</kbd> key), an asynchronous pipeline scans the environment for obstacles and hazards:

- **Spatial Mapping:** Maps bounding boxes to horizontal positions (`left`, `centre`, `right`) and depth estimates (`close`, `mid distance`, `far`) based on area coverage.
- **Tier-Based Priority Queue:**
  - **Tier 1 (Immediate announcement):** Person, dog, cat, car, bicycle, stairs, door, bus.
  - **Tier 2 (500 ms debounce):** Chair, couch, desk, table, TV, appliance.
- **Stale Suppression:** Objects that remain stationary (IoU > 0.45 for > 3.0 seconds) are suppressed so you are not bombarded with repetitive alerts.
- **Natural Language Output:**
  - Single/Few objects: *"A chair is on your left, close."*
  - Busy scenes: *"You are in a scene with a person ahead of you, close, and a laptop on your right."*

---

## 🚀 Installation & Setup

### Prerequisites
- Windows 10/11, macOS, or Linux.
- Python 3.10, 3.11, or 3.12 installed and added to PATH.
- Webcam (built-in or USB).

### 1. Clone & Enter Project Directory
```powershell
git clone https://github.com/Benjamin23-B/hand-scroll.git
cd "hand-scroll"
```

### 2. Create and Activate Virtual Environment
```powershell
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Windows (Command Prompt)
.\venv\Scripts\activate.bat

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

> **Note on Models:** On first launch, the application will automatically download `hand_landmarker.task` (~8 MB) and `efficientdet_lite0_int8.tflite` (~4.4 MB) if they are not already cached locally.

---

## 🎮 How to Use & Controls

### Quick Launch
On Windows, you can double-click **`run.bat`** or run:
```powershell
.\venv\Scripts\python.exe main.py
```

### Command Line Arguments
```powershell
# Run with camera index 1 (external webcam)
python main.py --camera-index 1

# Launch in minimal background mode without the large OpenCV preview window
python main.py --no-preview
```

### Interactive Keyboard Shortcuts

| Key | Context | Action |
|---|---|---|
| <kbd>C</kbd> | Global / Preview | Toggle the OpenCV camera preview window on/off. |
| <kbd>S</kbd> | Global / Preview | Toggle Scene Narration voice alerts on/off. |
| <kbd>Q</kbd> | Camera Preview | Exit and terminate application cleanly. |
| <kbd>Esc</kbd> | Camera Preview | Close preview window. |

You can also click directly on the thumbnail in the floating bottom-right overlay widget to toggle the preview window.

---

## 📂 Project Architecture & File Map

```
hand-scroll/
├── main.py                  # System entrypoint, orchestrates threads, shortcuts & loops
├── camera.py                # Threaded non-blocking OpenCV webcam stream
├── detector.py              # MediaPipe Tasks HandLandmarker + plain OpenCV skeleton drawer
├── classifier.py            # Geometric rule-based gesture classifier & circle trajectory tracker
├── state_machine.py         # Temporal gesture hold confirmation, cooldown & direction resolver
├── cursor.py                # EMA-smoothed mouse pointer mapping and click controller
├── scroll_engine.py         # High-precision inertia scroll loop with velocity lerp and decay
├── overlay.py               # Translucent Tkinter corner HUD widget with live feedback
├── actions.py               # PyAutoGUI OS automation mappings (scrolls, hotkeys, apps)
│
├── scene_detector.py        # On-device TFLite object detection (EfficientDet-Lite0 INT8)
├── scene_intelligence.py    # Spatial mapper, IoU change detection, and priority queue
├── nlg.py                   # Lightweight natural language sentence generator
├── speech_bridge.py         # Thread-safe pyttsx3 text-to-speech speaker bridge
│
├── requirements.txt         # Project dependencies
├── run.bat                  # One-click Windows startup script
├── hand_landmarker.task     # MediaPipe HandLandmarker model asset
└── efficientdet_lite0_int8.tflite # Quantized object detection model
```

---

## ⚙ Configuration & Tuning

All parameters are exposed via constructor arguments for easy personalization:

### Adjusting Gesture Confirmation & Sensitivity
In [main.py](file:///c:/Users/murug/OneDrive/Documents/hand%20scroll/main.py):
```python
# Increase hold_ms to prevent accidental triggers; increase cooldown_ms to slow repeat rate
state_machine = GestureStateMachine(hold_ms=250, cooldown_ms=200)

# Adjust cursor smoothing (higher alpha = more responsive, lower alpha = smoother)
cursor = CursorController(screen_w=sw, screen_h=sh, alpha=0.35)
```

### Adjusting Scroll Physics
In [main.py](file:///c:/Users/murug/OneDrive/Documents/hand%20scroll/main.py):
```python
# Lower easing for softer acceleration; higher decay (e.g. 0.94) for longer glide
scroll_engine = ScrollEngine(fps=60, easing=0.10, decay=0.90)
```

---

## ❓ Troubleshooting

| Issue | Root Cause | Solution |
|---|---|---|
| `AttributeError: module 'mediapipe' has no attribute 'solutions'` | Old MediaPipe code run on `mediapipe >= 0.10.0`. | This project uses the modern MediaPipe Tasks Vision API in [detector.py](file:///c:/Users/murug/OneDrive/Documents/hand%20scroll/detector.py). Run inside `.\venv\`. |
| Black screen / Camera failed to open | Wrong camera index or camera in use by another app (e.g. Zoom/Teams). | Run with `python main.py --camera-index 1` or close conflicting software. |
| PyAutoGUI `FailSafeException` | Mouse hit screen edge rapidly. | Handled automatically (`pyautogui.FAILSAFE = False` is set in [main.py](file:///c:/Users/murug/OneDrive/Documents/hand%20scroll/main.py)). |
| TTS Voice not speaking | Scene narration disabled or pyttsx3 audio driver uninitialized. | Press <kbd>S</kbd> to enable scene narration. On Linux, ensure `espeak` is installed (`sudo apt install espeak`). |
| High CPU usage | TensorFlow full build running instead of TFLite runtime. | On Linux/macOS, install `tflite-runtime`. On Windows, the system throttles detection intervals to keep usage minimal. |

---

## 🌐 Platform Notes

- **Windows:** Fully supported out of the box (Windows 10/11 precision touchpad actions and hotkeys configured).
- **macOS:** Grant **Accessibility** and **Camera** permissions to Terminal / Python under *System Settings → Privacy & Security → Accessibility*.
- **Linux (X11 / Wayland):** Requires `xdotool` or `python3-xlib` for full mouse/scroll injection:
  ```bash
  sudo apt-get install python3-xlib espeak
  ```

---

## 📄 License

Distributed under the MIT License. Feel free to modify, extend, and adapt for your own assistive technology or touchless control applications.
