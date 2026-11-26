"""Win32 Window Handle Inspection, Extended Styles, and Focus Handback Utilities."""

import ctypes
import logging
import sys
import tkinter as tk
from typing import Any

logger = logging.getLogger("OnmyojiHelper")

GWL_EXSTYLE: int = -20
WS_EX_TRANSPARENT: int = 0x00000020
WS_EX_TOOLWINDOW: int = 0x00000080
WS_EX_NOACTIVATE: int = 0x08000000

win32con: Any = None
win32gui: Any = None
win32process: Any = None

if sys.platform == "win32":
    import win32con as _win32con
    import win32gui as _win32gui
    import win32process as _win32process

    win32con = _win32con
    win32gui = _win32gui
    win32process = _win32process
    _user32 = ctypes.windll.user32
    _IS_64BIT = ctypes.sizeof(ctypes.c_void_p) == 8
    _LONG_PTR = ctypes.c_ssize_t if _IS_64BIT else ctypes.c_long
    _get_window_long = getattr(_user32, "GetWindowLongPtrW", _user32.GetWindowLongW)
    _set_window_long = getattr(_user32, "SetWindowLongPtrW", _user32.SetWindowLongW)
    _get_window_long.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _get_window_long.restype = _LONG_PTR
    _set_window_long.argtypes = [ctypes.c_void_p, ctypes.c_int, _LONG_PTR]
    _set_window_long.restype = _LONG_PTR
else:
    _user32 = None
    _get_window_long = None
    _set_window_long = None


def get_root_hwnd(hwnd: int) -> int:
    """Returns the top-level GA_ROOT owner window handle for a given HWND."""
    if not hwnd:
        return 0
    try:
        root = win32gui.GetAncestor(hwnd, win32con.GA_ROOT)
        return int(root) if root else int(hwnd)
    except Exception:
        return int(hwnd)


def get_window_pid(hwnd: int) -> int:
    """Returns the Windows Process ID owning the given window handle."""
    if not hwnd:
        return 0
    try:
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        return int(pid)
    except Exception:
        return 0


def get_window_title(hwnd: int) -> str:
    """Safely retrieves a window's title text."""
    if not hwnd:
        return ""
    try:
        return str(win32gui.GetWindowText(hwnd) or "")
    except Exception:
        return ""


def get_tk_win32_hwnd(tk_window: tk.Misc) -> int:
    """Returns the native outer Win32 wrapper HWND for a Tkinter Toplevel or Tk widget."""
    child_hwnd = int(tk_window.winfo_id())
    if _user32 is None:
        return child_hwnd
    parent_hwnd = int(_user32.GetParent(child_hwnd))
    return parent_hwnd if parent_hwnd else child_hwnd


def apply_extended_window_style(hwnd: int, style_flags: int, topmost: bool = True) -> None:
    """Overlays Win32 extended window styles (64-bit safe) without activating the window."""
    if not hwnd or _get_window_long is None or _set_window_long is None:
        return
    current_style = int(_get_window_long(hwnd, GWL_EXSTYLE))
    new_style = current_style | int(style_flags)
    _set_window_long(hwnd, GWL_EXSTYLE, new_style)
    z_order = win32con.HWND_TOPMOST if topmost else win32con.HWND_NOTOPMOST
    win32gui.SetWindowPos(
        hwnd,
        z_order,
        0,
        0,
        0,
        0,
        win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE | win32con.SWP_FRAMECHANGED,
    )


def matches_game_foreground(fg_hwnd: int, game_hwnd: int, game_root: int, game_pid: int) -> bool:
    """Returns True if the foreground HWND matches the game HWND, its GA_ROOT, or its process PID."""
    if not fg_hwnd:
        return False
    if fg_hwnd == game_hwnd or fg_hwnd == game_root:
        return True
    if get_root_hwnd(fg_hwnd) == game_root:
        return True
    fg_pid = get_window_pid(fg_hwnd)
    return bool(fg_pid > 0 and game_pid > 0 and fg_pid == game_pid)


def focus_game_root_window(hwnd: int) -> bool:
    """Hands foreground focus back to the game's top-level GA_ROOT window.

    Windows forbids calling SetForegroundWindow directly on child render panes; this helper
    always resolves GA_ROOT first and uses AttachThreadInput fallback if needed.
    """
    try:
        root_hwnd = get_root_hwnd(hwnd)
        if (
            not root_hwnd
            or not win32gui.IsWindow(root_hwnd)
            or win32gui.IsIconic(root_hwnd)
            or not win32gui.IsWindowVisible(root_hwnd)
        ):
            return False

        win32gui.ShowWindow(root_hwnd, win32con.SW_SHOW)
        try:
            win32gui.SetForegroundWindow(root_hwnd)
            if get_root_hwnd(win32gui.GetForegroundWindow()) == root_hwnd:
                return True
        except Exception as exc:
            logger.debug(f"[Win32Window] Direct SetForegroundWindow deferred to AttachThreadInput: {exc}")

        fg_hwnd = win32gui.GetForegroundWindow()
        fg_tid = win32process.GetWindowThreadProcessId(fg_hwnd)[0] if fg_hwnd else 0
        target_tid = win32process.GetWindowThreadProcessId(root_hwnd)[0]
        attached = False
        try:
            if _user32 is not None and fg_tid and target_tid and fg_tid != target_tid:
                attached = bool(_user32.AttachThreadInput(fg_tid, target_tid, True))
            win32gui.BringWindowToTop(root_hwnd)
            win32gui.SetForegroundWindow(root_hwnd)
            return get_root_hwnd(win32gui.GetForegroundWindow()) == root_hwnd
        finally:
            if attached and _user32 is not None:
                try:
                    _user32.AttachThreadInput(fg_tid, target_tid, False)
                except Exception as exc:
                    logger.debug(f"[Win32Window] AttachThreadInput detach note: {exc}")
    except Exception as exc:
        logger.debug(f"⚠️ [Win32Window] 自動交還遊戲焦點失敗: {exc}")
        return False
