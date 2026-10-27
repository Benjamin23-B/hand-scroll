"""
speech_bridge.py — Non-blocking text-to-speech via pyttsx3.

Design
------
pyttsx3's engine.runAndWait() blocks the calling thread.  To keep the
SceneThread responsive, all TTS work is offloaded to a dedicated
SpeechThread that drains a queue.  The SceneThread only ever does a
non-blocking queue.put() — it is never stalled by audio playback.

Singleton pattern matches the project's shared-resource idiom.

Usage
-----
    bridge = SpeechBridge.instance()
    bridge.say("A person is ahead of you, close.")
    bridge.stop()   # call on shutdown
"""

from __future__ import annotations

import queue
import threading
from typing import ClassVar

# ── pyttsx3 import guard ──────────────────────────────────────────────────────
try:
    import pyttsx3 as _pyttsx3
    _PYTTSX3_AVAILABLE = True
except ImportError:
    _pyttsx3 = None
    _PYTTSX3_AVAILABLE = False


# ── Sentinel ──────────────────────────────────────────────────────────────────
_STOP = object()   # poison pill to terminate the speech thread


class SpeechBridge:
    """
    Thread-safe, non-blocking TTS bridge.

    The bridge starts automatically on first use.
    Call stop() once at application shutdown.
    """

    _instance: ClassVar[SpeechBridge | None] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    # ------------------------------------------------------------------
    @classmethod
    def instance(cls) -> "SpeechBridge":
        """Return the process-wide singleton, creating it on first call."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = SpeechBridge()
        return cls._instance

    # ------------------------------------------------------------------
    def __init__(self) -> None:
        self.available = _PYTTSX3_AVAILABLE
        self._queue: queue.Queue[str | object] = queue.Queue()
        self._thread: threading.Thread | None = None

        if not self.available:
            print(
                "[speech_bridge] pyttsx3 not found — TTS disabled. "
                "Install with: pip install pyttsx3"
            )
            return

        self._thread = threading.Thread(
            target=self._worker,
            daemon=True,
            name="SpeechThread",
        )
        self._thread.start()
        print("[speech_bridge] TTS ready (pyttsx3).")

    # ------------------------------------------------------------------
    def say(self, text: str) -> None:
        """
        Enqueue text for speech.  Returns immediately — never blocks.
        Silently ignored if TTS is unavailable.
        """
        if not self.available or not text.strip():
            return
        self._queue.put(text)

    # ------------------------------------------------------------------
    def stop(self) -> None:
        """
        Signal the speech thread to exit.  Call once at app shutdown.
        """
        if self._thread is not None and self._thread.is_alive():
            self._queue.put(_STOP)
            self._thread.join(timeout=3.0)

    # ------------------------------------------------------------------
    def _worker(self) -> None:
        """
        Speech thread: owns the pyttsx3 engine (not safe to share).
        Drains the queue, speaking each sentence in order.
        """
        engine = _pyttsx3.init()
        engine.setProperty("rate", 160)   # words per minute
        engine.setProperty("volume", 1.0)

        # Prefer a female voice if available (clearer for accessibility)
        voices = engine.getProperty("voices")
        for v in voices:
            if "female" in v.name.lower() or "zira" in v.name.lower():
                engine.setProperty("voice", v.id)
                break

        while True:
            item = self._queue.get()

            if item is _STOP:
                break

            text = str(item)
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as exc:
                print(f"[speech_bridge] TTS error: {exc}")

        try:
            engine.stop()
        except Exception:
            pass
