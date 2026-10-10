# -*- coding: utf-8 -*-
"""Battle Overlay UI Module for Onmyoji Tactical Helper.

This module manages a transparent, click-through HUD overlay window positioned over the game client.
It renders real-time visual tracking rings, target selection indicators, and skill availability UI.
"""

import ctypes
from ctypes import wintypes
import tkinter as tk
from typing import Any, Dict, Optional

import numpy as np


class BattleOverlayUI:
    """Transparent HUD overlay displaying real-time battle targets and skill states."""

    def __init__(self, parent_root: tk.Tk) -> None:
        """Initializes the transparent top-level overlay window and enables click-through behavior.

        Args:
            parent_root: Reference to the parent Tkinter root window instance.
        """
        self.window: tk.Toplevel = tk.Toplevel(parent_root)
        self.window.title("Onmyoji Target Overlay")
        self.window.overrideredirect(True)
        self.window.attributes("-topmost", True)

        self.bg_color: str = "#000001"
        self.window.configure(bg=self.bg_color)
        self.window.wm_attributes("-transparentcolor", self.bg_color)

        self.canvas: tk.Canvas = tk.Canvas(self.window, bg=self.bg_color, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.enable_click_through()

    def enable_click_through(self) -> None:
        """Configures Windows API extended window styles to enable mouse click-through transparency."""
        try:
            hwnd = ctypes.windll.user32.GetParent(self.window.winfo_id())
            if hwnd == 0:
                hwnd = self.window.winfo_id()
            style = ctypes.windll.user32.GetWindowLongW(hwnd, -20)
            # Apply WS_EX_TRANSPARENT (0x20) and WS_EX_TOOLWINDOW (0x80)
            style |= 0x00000020 | 0x00000080
            ctypes.windll.user32.SetWindowLongW(hwnd, -20, style)
        except Exception as e:
            print(f"⚠️ Failed to configure window click-through style: {e}")

    def sync_position(self, hwnd: int) -> None:
        """Synchronizes overlay geometry with the client rect coordinates of the target game window.

        Args:
            hwnd: Window handle identifier of the game client window.
        """
        rect = wintypes.RECT()
        pt = wintypes.POINT(0, 0)

        ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(rect))
        ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(pt))

        x, y = pt.x, pt.y
        w, h = rect.right - rect.left, rect.bottom - rect.top
        self.window.geometry(f"{w}x{h}+{x}+{y}")

    def update_targets(
        self,
        status_dict: Dict[str, Dict[str, Any]],
        is_enemy: bool = False,
        img_w: float = 1.0,
        img_h: float = 1.0,
        target_slot: Optional[str] = None
    ) -> None:
        """Renders target tracking indicators and lock rings over character positions.

        Args:
            status_dict: Dictionary mapping character slots to status data.
            is_enemy: Boolean indicating if the targets belong to the enemy team.
            img_w: Width of the source captured image frame.
            img_h: Height of the source captured image frame.
            target_slot: Slot key identifier of the primary recommended target.
        """
        self.window.update_idletasks()
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        if canvas_w <= 1 or canvas_h <= 1:
            return

        scale_x = canvas_w / img_w if img_w > 0 else 1.0
        scale_y = canvas_h / img_h if img_h > 0 else 1.0

        for slot_key, info in status_dict.items():
            if info.get("alive", False) and info.get("pos", (0, 0)) != (0, 0):
                raw_cx, raw_cy = info["pos"]
                cx = raw_cx * scale_x
                cy = raw_cy * scale_y
                name = info.get("name", "")

                is_ai_target = (target_slot is not None and slot_key == target_slot)

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
                        cx, cy - r - 16, text=label_text, fill="#FFD700", font=("Microsoft JhengHei", 10, "bold")
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
                        cx, cy - r - 14, text=label_text, fill="#FFFFFF", font=("Microsoft JhengHei", 9, "bold")
                    )

    def update_skills(
        self,
        detector: Any,
        is_my_turn: bool,
        skill_states: Dict[str, bool],
        img_w: float = 1.0,
        img_h: float = 1.0,
        screen_color: Optional[np.ndarray] = None
    ) -> None:
        """Renders skill bar ROI bounding box and individual skill availability indicators.

        Args:
            detector: Reference to the TurnDetector instance.
            is_my_turn: Boolean flag indicating if it is currently the player's turn.
            skill_states: Dictionary mapping logical skills ('s1'-'s4') to availability booleans.
            img_w: Width of the source captured image frame.
            img_h: Height of the source captured image frame.
            screen_color: Optional BGR image array of the full screen.
        """
        if not self.window.winfo_exists():
            return

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1:
            return

        scale_x = canvas_w / img_w if img_w > 0 else 1.0
        scale_y = canvas_h / img_h if img_h > 0 else 1.0

        # 1. Render primary Skill Bar region bounding rectangle
        rx1, ry1, rx2, ry2 = detector.SKILL_BAR_ROI
        x1, y1 = int(img_w * rx1 * scale_x), int(img_h * ry1 * scale_y)
        x2, y2 = int(img_w * rx2 * scale_x), int(img_h * ry2 * scale_y)

        bar_color = "#FFFF00" if is_my_turn else "#888888"
        self.canvas.create_rectangle(x1, y1, x2, y2, outline=bar_color, width=2, dash=(4, 4))
        self.canvas.create_text(
            x1 + 5,
            y1 - 12,
            text=f"【Skill Bar】 我方回合: {is_my_turn}",
            fill=bar_color,
            anchor="w",
            font=("Microsoft JhengHei", 10, "bold")
        )

        if screen_color is None:
            return

        # 2. Query dynamic skill slot mapping and render slot indicators
        mapping = detector.get_skill_mapping(screen_color)
        colors = {"s1": "#FF9900", "s2": "#FF3333", "s3": "#33FF33", "s4": "#33CCFF"}

        for logical_name in ["s1", "s2", "s3", "s4"]:
            physical_slot = mapping.get(logical_name)
            if physical_slot is None:
                continue

            s_rx1, s_ry1, s_rx2, s_ry2 = detector.PHYSICAL_SLOTS[physical_slot]
            sx1, sy1 = int(img_w * s_rx1 * scale_x), int(img_h * s_ry1 * scale_y)
            sx2, sy2 = int(img_w * s_rx2 * scale_x), int(img_h * s_ry2 * scale_y)

            is_active = bool(skill_states.get(logical_name, False))
            color = colors[logical_name] if is_active else "#666666"

            self.canvas.create_rectangle(sx1, sy1, sx2, sy2, outline=color, width=2)
            self.canvas.create_text(
                sx1 + 4,
                sy1 + 14,
                text=f"{logical_name.upper()} ({physical_slot}): {'可用' if is_active else '不可用'}",
                fill=color,
                anchor="w",
                font=("Microsoft JhengHei", 9, "bold")
            )

    def clear(self) -> None:
        """Clears all drawn canvas elements from the overlay."""
        if self.window.winfo_exists():
            self.canvas.delete("all")

    def close(self) -> None:
        """Destroys the overlay top-level window."""
        if self.window.winfo_exists():
            self.window.destroy()