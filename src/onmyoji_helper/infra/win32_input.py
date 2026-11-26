"""Win32 64-bit SendInput, Per-Monitor DPI Cursor Motion, and Virtual Key Utilities."""

import ctypes
import logging
import math
import random
import time
from collections.abc import Callable

logger = logging.getLogger("OnmyojiHelper")

INPUT_MOUSE: int = 0
MOUSEEVENTF_MOVE: int = 0x0001
MOUSEEVENTF_LEFTDOWN: int = 0x0002
MOUSEEVENTF_LEFTUP: int = 0x0004
MOUSEEVENTF_VIRTUALDESK: int = 0x4000
MOUSEEVENTF_ABSOLUTE: int = 0x8000

SM_XVIRTUALSCREEN: int = 76
SM_YVIRTUALSCREEN: int = 77
SM_CXVIRTUALSCREEN: int = 78
SM_CYVIRTUALSCREEN: int = 79

VK_ESCAPE: int = 0x1B
VK_F12: int = 0x7B

ULONG_PTR = ctypes.c_uint64 if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_uint32


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_uint32),
        ("dwFlags", ctypes.c_uint32),
        ("time", ctypes.c_uint32),
        ("dwExtraInfo", ULONG_PTR),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", _MOUSEINPUT)]


class _INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("union", _INPUT_UNION),
    ]


class _POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def set_cursor_pos(x: int, y: int) -> bool:
    """Moves the cursor to exact Per-Monitor DPI-Aware physical screen coordinates."""
    try:
        return bool(ctypes.windll.user32.SetCursorPos(int(x), int(y)))
    except Exception:
        return False


def get_cursor_pos() -> tuple[int, int]:
    """Reads the current physical cursor position in screen coordinates."""
    pt = _POINT()
    if ctypes.windll.user32.GetCursorPos(ctypes.byref(pt)):
        return int(pt.x), int(pt.y)
    return 0, 0


def screen_to_absolute_mouse(x: int, y: int) -> tuple[int, int]:
    """Converts physical screen coordinates to 0..65535 SendInput virtual-desktop coordinates."""
    user32 = ctypes.windll.user32
    vx = int(user32.GetSystemMetrics(SM_XVIRTUALSCREEN))
    vy = int(user32.GetSystemMetrics(SM_YVIRTUALSCREEN))
    vw = max(1, int(user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)))
    vh = max(1, int(user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)))

    abs_x = int(round(((x - vx) * 65535.0) / float(max(1, vw - 1))))
    abs_y = int(round(((y - vy) * 65535.0) / float(max(1, vh - 1))))
    return max(0, min(65535, abs_x)), max(0, min(65535, abs_y))


def send_mouse_event(flags: int, abs_x: int = 0, abs_y: int = 0) -> bool:
    """Dispatches a single Win32 SendInput mouse event."""
    inp = _INPUT()
    inp.type = INPUT_MOUSE
    inp.union.mi = _MOUSEINPUT(
        dx=abs_x,
        dy=abs_y,
        mouseData=0,
        dwFlags=flags,
        time=0,
        dwExtraInfo=0,
    )
    sent = ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))
    return bool(sent == 1)


def is_vk_pressed(vk_code: int) -> bool:
    """Returns True if the given virtual key is currently held down."""
    try:
        return bool(ctypes.windll.user32.GetAsyncKeyState(vk_code) & 0x8000)
    except Exception:
        return False


def perform_humanized_click(
    target_x: int,
    target_y: int,
    mouse_takeover_px: float,
    abort_check_fn: Callable[[], bool],
    on_takeover_fn: Callable[[], None] | None = None,
) -> bool:
    """Interpolates cursor smoothly to `(target_x, target_y)`, stabilizes hover, and clicks.

    Tolerates minor accidental mouse bumps during the brief dispatch window (when
    `mouse_takeover_px <= 0`, mouse micro-movements never abort or downgrade Red Mode, and
    the cursor is re-anchored onto `(target_x, target_y)` right before `LEFTDOWN`), while
    checking `abort_check_fn` (`F12` / `ESC` emergency brake) at every step.
    """
    if abort_check_fn():
        return False

    start_x, start_y = get_cursor_pos()
    steps = random.randint(3, 5)
    dist = math.hypot(target_x - start_x, target_y - start_y)
    arc_amp = min(12.0, dist * 0.04) * random.choice((-1.0, 1.0))

    prev_actual_x, prev_actual_y = start_x, start_y
    for step_idx in range(1, steps + 1):
        if abort_check_fn():
            return False

        if mouse_takeover_px > 0.0 and step_idx > 1:
            cur_x, cur_y = get_cursor_pos()
            if math.hypot(cur_x - prev_actual_x, cur_y - prev_actual_y) > mouse_takeover_px:
                if callable(on_takeover_fn):
                    on_takeover_fn()
                return False

        t = step_idx / float(steps)
        ease = t * t * (3.0 - 2.0 * t)
        curve = math.sin(math.pi * t) * arc_amp
        use_x_curve = abs(target_y - start_y) > abs(target_x - start_x)
        nx = int(round(start_x + (target_x - start_x) * ease + (curve if use_x_curve else 0.0)))
        ny = int(round(start_y + (target_y - start_y) * ease + (0.0 if use_x_curve else curve)))

        if not set_cursor_pos(nx, ny):
            abs_x, abs_y = screen_to_absolute_mouse(nx, ny)
            send_mouse_event(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK, abs_x, abs_y)

        prev_actual_x, prev_actual_y = get_cursor_pos()
        time.sleep(random.uniform(0.007, 0.014))

    # Snap onto exact target pixel + hover wake + UI settle delay (0.035~0.050s)
    if not set_cursor_pos(target_x, target_y):
        abs_tx, abs_ty = screen_to_absolute_mouse(target_x, target_y)
        send_mouse_event(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK, abs_tx, abs_ty)
    send_mouse_event(MOUSEEVENTF_MOVE, 0, 0)
    time.sleep(random.uniform(0.035, 0.050))

    if abort_check_fn():
        return False

    # Re-anchor cursor if a slight wrist bump nudged the mouse during the 40ms hover settle
    cur_fx, cur_fy = get_cursor_pos()
    if math.hypot(cur_fx - target_x, cur_fy - target_y) > 3.0:
        if not set_cursor_pos(target_x, target_y):
            abs_tx, abs_ty = screen_to_absolute_mouse(target_x, target_y)
            send_mouse_event(
                MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK, abs_tx, abs_ty
            )
        send_mouse_event(MOUSEEVENTF_MOVE, 0, 0)

    down_ok = send_mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0)
    time.sleep(random.uniform(0.032, 0.050))
    up_ok = send_mouse_event(MOUSEEVENTF_LEFTUP, 0, 0)
    if not (down_ok and up_ok):
        logger.warning(
            f"⚠️ [Win32Input] SendInput 點擊未完整送出 (down={down_ok}, up={up_ok}, pos=({target_x},{target_y}))"
        )
        return False
    return True
