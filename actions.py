"""
actions.py — All OS-level actions triggered by hand gestures.

Every function here is a thin wrapper around pyautogui or subprocess.
They are called from the processing thread (not the Tkinter main thread),
which is safe for pyautogui calls.
"""

import subprocess

import pyautogui

# Remove inter-call delay so actions feel instant.
pyautogui.PAUSE = 0


# ── Scroll ────────────────────────────────────────────────────────────────────

def scroll(delta: float) -> None:
    """Vertical scroll.  Positive delta → scroll down, negative → scroll up."""
    if abs(delta) > 0.5:
        pyautogui.scroll(-int(delta))


# ── Zoom ─────────────────────────────────────────────────────────────────────

def zoom_in() -> None:
    """Zoom in: Ctrl + scroll up (3 clicks)."""
    pyautogui.keyDown("ctrl")
    pyautogui.scroll(3)
    pyautogui.keyUp("ctrl")


def zoom_out() -> None:
    """Zoom out: Ctrl + scroll down (3 clicks)."""
    pyautogui.keyDown("ctrl")
    pyautogui.scroll(-3)
    pyautogui.keyUp("ctrl")


# ── Mouse clicks ──────────────────────────────────────────────────────────────

def right_click() -> None:
    """Right-click at the current cursor position."""
    pyautogui.rightClick()


# ── Browser navigation ────────────────────────────────────────────────────────

def browser_back() -> None:
    """Navigate back in the active browser tab (Alt + Left)."""
    pyautogui.hotkey("alt", "left")


def browser_forward() -> None:
    """Navigate forward in the active browser tab (Alt + Right)."""
    pyautogui.hotkey("alt", "right")


# ── System / desktop gestures ─────────────────────────────────────────────────

def task_view() -> None:
    """Open Task View / timeline (Win + Tab)."""
    pyautogui.hotkey("win", "tab")


def show_desktop() -> None:
    """Toggle Show Desktop (Win + D)."""
    pyautogui.hotkey("win", "d")


def switch_desktop_left() -> None:
    """Switch to the virtual desktop on the left (Ctrl + Win + Left)."""
    pyautogui.hotkey("ctrl", "win", "left")


def switch_desktop_right() -> None:
    """Switch to the virtual desktop on the right (Ctrl + Win + Right)."""
    pyautogui.hotkey("ctrl", "win", "right")


def action_center() -> None:
    """Open the Windows Action Centre / Notification panel (Win + A)."""
    pyautogui.hotkey("win", "a")


def minimize_all() -> None:
    """Minimize all windows (Win + M)."""
    pyautogui.hotkey("win", "m")


# ── App switching ─────────────────────────────────────────────────────────────

def app_switch_forward() -> None:
    """Cycle to the next open app (Alt + Tab)."""
    pyautogui.hotkey("alt", "tab")


def app_switch_back() -> None:
    """Cycle to the previous open app (Alt + Shift + Tab)."""
    pyautogui.hotkey("alt", "shift", "tab")


# ── Launch Chrome ─────────────────────────────────────────────────────────────

def open_chrome() -> None:
    """
    Launch Google Chrome in the background (non-blocking).
    Falls back to the system default browser if Chrome isn't on PATH.
    """
    try:
        subprocess.Popen(
            ["cmd", "/c", "start", "chrome"],
            shell=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("[Hand Scroll] Circle gesture — opening Chrome.")
    except Exception as exc:
        print(f"[Hand Scroll] Could not open Chrome: {exc}")
