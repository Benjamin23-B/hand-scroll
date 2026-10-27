"""
overlay.py — Always-on-top Tkinter debug window.

Shows a live thumbnail of the annotated camera feed and the current
gesture label.  Uses after() callbacks so Tkinter's event loop stays
responsive without any blocking threads.
"""

import tkinter as tk
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk


class OverlayWindow:
    THUMB_W = 260
    THUMB_H = 160
    WIN_W   = 280
    WIN_H   = 220

    def __init__(self) -> None:
        self._root: Optional[tk.Tk] = None
        self._video_label: Optional[tk.Label] = None
        self._text_label: Optional[tk.Label] = None

        # Thread-safe queues for data coming from the processing thread
        self._pending_frame: Optional[np.ndarray] = None
        self._pending_text: str = "—"
        self._tk_image: Optional[ImageTk.PhotoImage] = None  # keep reference alive
        self.on_preview_toggle = None

    # ------------------------------------------------------------------
    def start(self) -> None:
        """
        Create the Tkinter root window.
        Call this before run() / mainloop().
        """
        self._root = tk.Tk()
        self._root.title("Hand Scroll")
        self._root.geometry(f"{self.WIN_W}x{self.WIN_H}")
        self._root.resizable(False, False)

        # Always on top, semi-transparent
        self._root.wm_attributes("-topmost", True)
        self._root.wm_attributes("-alpha", 0.85)
        # Position: bottom-right corner
        self._root.update_idletasks()
        screen_w = self._root.winfo_screenwidth()
        screen_h = self._root.winfo_screenheight()
        x = screen_w - self.WIN_W - 20
        y = screen_h - self.WIN_H - 60
        self._root.geometry(f"{self.WIN_W}x{self.WIN_H}+{x}+{y}")

        # Dark background
        self._root.configure(bg="#1a1a2e")

        # Video thumbnail label (use a blank PhotoImage so Tkinter sizes in pixels, not character units)
        blank = Image.new("RGB", (self.THUMB_W, self.THUMB_H), color=(26, 26, 46))
        self._tk_image = ImageTk.PhotoImage(blank)
        self._video_label = tk.Label(
            self._root,
            image=self._tk_image,
            bg="#1a1a2e",
            bd=1,
            relief="solid",
            cursor="hand2",
        )
        self._video_label.pack(pady=(8, 4))
        self._video_label.bind(
            "<Button-1>",
            lambda _e: self.on_preview_toggle() if callable(self.on_preview_toggle) else None,
        )

        # Gesture text label
        self._text_label = tk.Label(
            self._root,
            text="— no gesture —",
            font=("Segoe UI", 12, "bold"),
            fg="#e0e0ff",
            bg="#1a1a2e",
        )
        self._text_label.pack(pady=(0, 8))

        # Kick off the refresh loop
        self._root.after(30, self._refresh)

    # ------------------------------------------------------------------
    def update_frame(self, bgr_frame: np.ndarray) -> None:
        """Queue a new camera frame to display (called from any thread)."""
        if bgr_frame is not None:
            resized = cv2.resize(bgr_frame, (self.THUMB_W, self.THUMB_H))
            self._pending_frame = resized

    # ------------------------------------------------------------------
    def update_label(self, text: str) -> None:
        """Queue a gesture label update (called from any thread)."""
        self._pending_text = text

    # ------------------------------------------------------------------
    def _refresh(self) -> None:
        """Tkinter after() callback — runs on the main thread every ~30 ms."""
        if self._pending_frame is not None:
            rgb = cv2.cvtColor(self._pending_frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            self._tk_image = ImageTk.PhotoImage(pil_img)
            self._video_label.configure(image=self._tk_image)
            self._pending_frame = None

        self._text_label.configure(text=self._pending_text)

        # Schedule next refresh
        self._root.after(30, self._refresh)

    # ------------------------------------------------------------------
    def run(self) -> None:
        """
        Enter Tkinter's blocking event loop.
        MUST be called from the main thread.
        """
        if self._root is not None:
            self._root.mainloop()
