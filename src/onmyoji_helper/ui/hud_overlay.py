"""Transparent Click-Through HUD Overlay Window for Onmyoji Tactical Helper.

Renders real-time ally/enemy tracking rings, primary target lock crosshairs, and skill
availability boxes over the game client using physical pixels (`SetWindowPos`) and
`CropOffset` letterbox/pillarbox compensation.
"""

import logging
import tkinter as tk
from typing import Any

import numpy as np
import win32con
import win32gui

from onmyoji_helper.automation.coordinate_mapper import project_cropped_to_physical_client
from onmyoji_helper.domain.models import CropOffset
from onmyoji_helper.infra.win32_window import (
    WS_EX_NOACTIVATE,
    WS_EX_TOOLWINDOW,
    WS_EX_TRANSPARENT,
    apply_extended_window_style,
    get_tk_win32_hwnd,
)

logger = logging.getLogger("OnmyojiHelper")


class BattleOverlayUI:
    """Transparent click-through HUD overlay aligned with the game client window."""

    def __init__(self, parent_root: tk.Tk) -> None:
        """Initializes the transparent Toplevel Canvas and enables Win32 click-through."""
        self.window: tk.Toplevel = tk.Toplevel(parent_root)
        self.window.title("Onmyoji Target Overlay")
        self.window.overrideredirect(True)
        self.window.attributes("-topmost", True)

        self.bg_color: str = "#000001"
        self.window.configure(bg=self.bg_color)
        self.window.wm_attributes("-transparentcolor", self.bg_color)

        self.canvas: tk.Canvas = tk.Canvas(self.window, bg=self.bg_color, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self._last_rect: tuple[int, int, int, int] | None = None
        self._last_size: tuple[int, int] = (0, 0)

        # Complete normal Tkinter mapping before applying Win32 extended styles
        self.window.deiconify()
        self.window.update_idletasks()
        self.enable_click_through()

    def _get_win32_hwnd(self) -> int:
        return get_tk_win32_hwnd(self.window)

    def winfo_exists(self) -> bool:
        try:
            return bool(self.window.winfo_exists())
        except Exception:
            return False

    def enable_click_through(self) -> None:
        """Applies `WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE` (64-bit safe)."""
        try:
            hwnd = self._get_win32_hwnd()
            apply_extended_window_style(
                hwnd,
                WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE,
                topmost=True,
            )
        except Exception as exc:
            logger.warning(f"⚠️ [BattleOverlayUI] Failed to configure click-through style: {exc}")

    def sync_position(self, hwnd: int) -> None:
        """Synchronizes overlay physical position and size with the target game client area."""
        try:
            if not hwnd or not win32gui.IsWindow(hwnd):
                return

            left, top, right, bottom = win32gui.GetClientRect(hwnd)
            w, h = right - left, bottom - top
            if w <= 50 or h <= 50:
                return

            pt_x, pt_y = win32gui.ClientToScreen(hwnd, (0, 0))
            curr_rect = (pt_x, pt_y, w, h)

            if self.window.state() != "normal":
                self.window.deiconify()

            if curr_rect == self._last_rect and self.canvas.winfo_width() > 1 and self.canvas.winfo_height() > 1:
                return

            overlay_hwnd = self._get_win32_hwnd()
            win32gui.SetWindowPos(
                overlay_hwnd,
                win32con.HWND_TOPMOST,
                pt_x,
                pt_y,
                w,
                h,
                win32con.SWP_NOACTIVATE | win32con.SWP_SHOWWINDOW,
            )
            self._last_rect = curr_rect
            self._last_size = (w, h)
            self.window.update_idletasks()
        except Exception as exc:
            logger.debug(f"⚠️ [BattleOverlayUI] sync_position error: {exc}")

    def _get_canvas_size(self) -> tuple[int, int]:
        # Prefer the authoritative physical client dimensions set via SetWindowPos(w, h)
        # so early Tkinter frames never fall back to unconfigured logical widget dimensions.
        if self._last_size[0] > 50 and self._last_size[1] > 50:
            return self._last_size
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1:
            self.window.update_idletasks()
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()
        return canvas_w, canvas_h

    @staticmethod
    def _project_point(
        raw_x: float,
        raw_y: float,
        canvas_w: int,
        canvas_h: int,
        img_w: float,
        img_h: float,
        crop: CropOffset | None,
    ) -> tuple[float, float]:
        """Projects a cropped-image coordinate onto physical overlay canvas coordinates."""
        return project_cropped_to_physical_client(
            img_x=raw_x,
            img_y=raw_y,
            img_w=img_w,
            img_h=img_h,
            target_w=canvas_w,
            target_h=canvas_h,
            crop=crop,
        )

    def update_targets(
        self,
        status_dict: dict[str, dict[str, Any]],
        is_enemy: bool = False,
        img_w: float = 1.0,
        img_h: float = 1.0,
        target_slot: str | None = None,
        crop: CropOffset | None = None,
    ) -> None:
        """Draws unit tracking rings and golden primary target crosshairs on the overlay."""
        if not self.winfo_exists():
            return

        canvas_w, canvas_h = self._get_canvas_size()
        if canvas_w <= 1 or canvas_h <= 1:
            return

        for slot_key, info in status_dict.items():
            if info.get("alive", False) and info.get("pos", (0, 0)) != (0, 0):
                raw_cx, raw_cy = info["pos"]
                cx, cy = self._project_point(
                    float(raw_cx), float(raw_cy), canvas_w, canvas_h, img_w, img_h, crop
                )
                name = str(info.get("name", ""))
                is_ai_target = target_slot is not None and slot_key == target_slot

                if is_ai_target:
                    color = "#FFD700"
                    r = 30
                    self.canvas.create_oval(
                        cx - r - 4, cy - r - 4, cx + r + 4, cy + r + 4, outline="#FF4500", width=2
                    )
                    self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline=color, width=4)
                    self.canvas.create_line(cx - r - 10, cy, cx + r + 10, cy, fill=color, width=3)
                    self.canvas.create_line(cx, cy - r - 10, cx, cy + r + 10, fill=color, width=3)

                    target_prefix = "優先打擊" if is_enemy else "優先選擇"
                    label_text = f"🎯 [{target_prefix}] {name}"
                    self.canvas.create_rectangle(
                        cx - 55, cy - r - 28, cx + 55, cy - r - 4, fill="#2A1B00", outline=color, width=2
                    )
                    self.canvas.create_text(
                        cx,
                        cy - r - 16,
                        text=label_text,
                        fill="#FFD700",
                        font=("Microsoft JhengHei", 10, "bold"),
                    )
                else:
                    color = "#FF3333" if is_enemy else "#33CCFF"
                    r = 20
                    self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline=color, width=2)
                    self.canvas.create_line(cx - r - 6, cy, cx + r + 6, cy, fill=color, width=1)
                    self.canvas.create_line(cx, cy - r - 6, cx, cy + r + 6, fill=color, width=1)

                    label_text = f"[{slot_key}] {name}"
                    self.canvas.create_rectangle(
                        cx - 45, cy - r - 24, cx + 45, cy - r - 4, fill="#1E1E1E", outline=color, width=1
                    )
                    self.canvas.create_text(
                        cx,
                        cy - r - 14,
                        text=label_text,
                        fill="#FFFFFF",
                        font=("Microsoft JhengHei", 9, "bold"),
                    )

    def update_skills(
        self,
        detector: Any,
        is_my_turn: bool,
        skill_states: dict[str, bool],
        img_w: float = 1.0,
        img_h: float = 1.0,
        screen_color: np.ndarray | None = None,
        skill_mapping: dict[str, str | None] | None = None,
        crop: CropOffset | None = None,
    ) -> None:
        """Draws skill bar ROI boundary and per-button availability indicators."""
        if not self.winfo_exists():
            return

        canvas_w, canvas_h = self._get_canvas_size()
        if canvas_w <= 1 or canvas_h <= 1:
            return

        rx1, ry1, rx2, ry2 = detector.SKILL_BAR_ROI
        x1, y1 = self._project_point(img_w * rx1, img_h * ry1, canvas_w, canvas_h, img_w, img_h, crop)
        x2, y2 = self._project_point(img_w * rx2, img_h * ry2, canvas_w, canvas_h, img_w, img_h, crop)

        bar_color = "#FFFF00" if is_my_turn else "#888888"
        self.canvas.create_rectangle(int(x1), int(y1), int(x2), int(y2), outline=bar_color, width=2, dash=(4, 4))
        self.canvas.create_text(
            int(x1) + 5,
            int(y1) - 12,
            text=f"【Skill Bar】 我方回合: {is_my_turn}",
            fill=bar_color,
            anchor="w",
            font=("Microsoft JhengHei", 10, "bold"),
        )

        if skill_mapping is not None:
            mapping = skill_mapping
        elif screen_color is not None:
            mapping = detector.get_skill_mapping(screen_color)
        else:
            return

        colors = {"s1": "#FF9900", "s2": "#FF3333", "s3": "#33FF33", "s4": "#33CCFF"}
        for logical_name in ("s1", "s2", "s3", "s4"):
            physical_slot = mapping.get(logical_name)
            if physical_slot is None:
                continue

            s_rx1, s_ry1, s_rx2, s_ry2 = detector.PHYSICAL_SLOTS[physical_slot]
            sx1, sy1 = self._project_point(img_w * s_rx1, img_h * s_ry1, canvas_w, canvas_h, img_w, img_h, crop)
            sx2, sy2 = self._project_point(img_w * s_rx2, img_h * s_ry2, canvas_w, canvas_h, img_w, img_h, crop)

            is_active = bool(skill_states.get(logical_name, False))
            color = colors[logical_name] if is_active else "#666666"

            self.canvas.create_rectangle(int(sx1), int(sy1), int(sx2), int(sy2), outline=color, width=2)
            self.canvas.create_text(
                int(sx1) + 4,
                int(sy1) + 14,
                text=f"{logical_name.upper()} ({physical_slot}): {'可用' if is_active else '不可用'}",
                fill=color,
                anchor="w",
                font=("Microsoft JhengHei", 9, "bold"),
            )

    def clear(self) -> None:
        try:
            if self.winfo_exists():
                self.canvas.delete("all")
        except Exception:
            pass

    def close(self) -> None:
        self._last_rect = None
        self._last_size = (0, 0)
        try:
            if self.winfo_exists():
                self.canvas.delete("all")
                self.window.destroy()
        except Exception:
            pass
