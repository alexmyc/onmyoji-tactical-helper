"""Primary Graphical User Interface Window for Onmyoji Tactical Helper.

Renders real-time Ban/Pick detection, XMind counter recommendations, opponent predictions,
and tri-state mode indicators (`Green / Yellow / Red`) in a dark-themed topmost window.
"""

import threading
import tkinter as tk
from collections.abc import Callable
from dataclasses import dataclass
from tkinter import font
from typing import Any

from onmyoji_helper.domain.models import OperatingMode


@dataclass(frozen=True)
class ThemePalette:
    """Dark-mode visual theme tokens for the primary helper window."""

    bg_color: str = "#121212"
    card_bg: str = "#1C1C1E"
    menu_bg: str = "#1C1C1E"
    text_primary: str = "#F5F5F7"
    text_dim: str = "#636366"
    ally_color: str = "#4DA8DA"
    opp_color: str = "#EE6C4D"
    highlight_color: str = "#E3B341"
    btn_bg: str = "#8B2626"
    btn_active_bg: str = "#A83232"


MODE_COLORS: dict[str, str] = {
    OperatingMode.GREEN.value: "#34C759",
    OperatingMode.YELLOW.value: "#FFCC00",
    OperatingMode.RED.value: "#FF3B30",
}

MODE_DESCRIPTIONS: dict[str, str] = {
    OperatingMode.GREEN.value: "綠色模式 (純助手主體)",
    OperatingMode.YELLOW.value: "黃色模式 (HUD 紅綠圈 + 戰術提示，不點擊)",
    OperatingMode.RED.value: "紅色模式 (全自動戰鬥執行 | F12/ESC 緊急煞車)",
}

MODE_CYCLE_ORDER: list[str] = [
    OperatingMode.GREEN.value,
    OperatingMode.YELLOW.value,
    OperatingMode.RED.value,
]


class OnmyojiHelperUI:
    """Primary Tkinter GUI window managing Ban/Pick drafting displays and mode toggles."""

    def __init__(
        self,
        root: tk.Tk,
        available_bans: list[str] | None,
        mode_change_callback: Callable[[str], Any] | None = None,
        sandbox_callback: Callable[[], None] | None = None,
    ) -> None:
        """Initializes GUI layout, typography, thread-safe state flags, and interactive widgets."""
        self.root: tk.Tk = root
        self.available_bans: list[str] = available_bans if available_bans else ["未載入"]
        self.mode_change_callback: Callable[[str], Any] | None = mode_change_callback
        self.sandbox_callback: Callable[[], None] | None = sandbox_callback
        self.theme: ThemePalette = ThemePalette()

        self.root.title("陰陽師鬥技助手")
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 1.0)

        window_width: int = 520
        window_height: int = 535 if self.sandbox_callback is not None else 500
        screen_width: int = self.root.winfo_screenwidth()
        pos_x: int = screen_width - window_width - 50
        pos_y: int = 100
        self.root.geometry(f"{window_width}x{window_height}+{pos_x}+{pos_y}")
        self.root.minsize(window_width, window_height)
        self.root.maxsize(window_width, window_height)
        self.root.resizable(False, False)

        self.current_mode: str = OperatingMode.GREEN.value
        self.mode_colors: dict[str, str] = dict(MODE_COLORS)
        self.mode_names: dict[str, str] = dict(MODE_DESCRIPTIONS)

        self.root.configure(bg=self.theme.bg_color)

        self.title_font: font.Font = font.Font(family="Microsoft JhengHei", size=11, weight="bold")
        self.normal_font: font.Font = font.Font(family="Microsoft JhengHei", size=10)
        self.large_font: font.Font = font.Font(family="Microsoft JhengHei", size=13, weight="bold")
        self.xl_font: font.Font = font.Font(family="Microsoft JhengHei", size=16, weight="bold")

        self._state_lock: threading.Lock = threading.Lock()
        self._current_selected_ban: str = self.available_bans[0]
        self._needs_reset: bool = False
        self.selected_ban: tk.StringVar = tk.StringVar(value=self.available_bans[0])

        self.create_widgets()

    def get_selected_system(self) -> str:
        """Thread-safe getter for the currently selected playbook dataset."""
        with self._state_lock:
            return self._current_selected_ban

    def check_and_clear_reset(self) -> bool:
        """Thread-safe check and atomic clear of the manual reset flag."""
        with self._state_lock:
            if self._needs_reset:
                self._needs_reset = False
                return True
            return False

    def create_widgets(self) -> None:
        """Constructs and arranges all UI cards and labels."""
        t = self.theme
        top_card = tk.Frame(self.root, bg=t.card_bg)
        top_card.pack(pady=(15, 10), fill=tk.X, padx=15, ipadx=10, ipady=8)

        self.mode_canvas = tk.Canvas(
            top_card, width=22, height=22, bg=t.card_bg, highlightthickness=0, cursor="hand2"
        )
        self.mode_canvas.pack(side=tk.LEFT, padx=(0, 12))
        self.draw_mode_dot()
        self.enable_mode_click()

        my_ban_frame = tk.Frame(top_card, bg=t.card_bg)
        my_ban_frame.pack(side=tk.LEFT)
        tk.Label(
            my_ban_frame, text="數據:", bg=t.card_bg, fg=t.text_dim, font=self.title_font
        ).pack(side=tk.LEFT)

        self.ban_menu = tk.OptionMenu(
            my_ban_frame, self.selected_ban, *self.available_bans, command=self.on_ban_changed
        )
        self.ban_menu.config(
            bg=t.menu_bg,
            fg=t.text_primary,
            activebackground="#3A3A3C",
            activeforeground=t.text_primary,
            highlightthickness=0,
            bd=0,
            relief="flat",
            indicatoron=False,
            cursor="hand2",
            font=self.title_font,
            width=14,
            anchor="w",
        )
        self.ban_menu["menu"].config(
            bg=t.menu_bg, fg=t.text_primary, bd=0, relief="flat", font=self.title_font
        )
        self.ban_menu.pack(side=tk.LEFT, padx=5)

        opp_ban_frame = tk.Frame(top_card, bg=t.card_bg)
        opp_ban_frame.pack(side=tk.RIGHT)
        tk.Label(
            opp_ban_frame, text="敵方禁用:", bg=t.card_bg, fg=t.text_dim, font=self.title_font
        ).pack(side=tk.LEFT)
        self.lbl_opp_ban_val = tk.Label(
            opp_ban_frame, text="— —", bg=t.card_bg, fg=t.text_dim, font=self.title_font
        )
        self.lbl_opp_ban_val.pack(side=tk.LEFT, padx=(5, 0))

        draft_card = tk.Frame(self.root, bg=t.bg_color)
        draft_card.pack(fill=tk.X, pady=5, padx=15)

        my_frame = tk.Frame(draft_card, bg=t.card_bg)
        my_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(0, 5), ipady=12)
        tk.Label(
            my_frame, text="⛨ 我方陣容", bg=t.card_bg, fg=t.ally_color, font=self.normal_font
        ).pack(pady=(5, 8))
        self.lbl_my_draft = tk.Label(
            my_frame, text="— — —", bg=t.card_bg, fg=t.text_dim, font=self.large_font, wraplength=200
        )
        self.lbl_my_draft.pack()

        opp_frame = tk.Frame(draft_card, bg=t.card_bg)
        opp_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=(5, 0), ipady=12)
        tk.Label(
            opp_frame, text="⚔ 敵方陣容", bg=t.card_bg, fg=t.opp_color, font=self.normal_font
        ).pack(pady=(5, 8))
        self.lbl_opp_draft = tk.Label(
            opp_frame, text="— — —", bg=t.card_bg, fg=t.text_dim, font=self.large_font, wraplength=200
        )
        self.lbl_opp_draft.pack()

        recs_card = tk.Frame(self.root, bg=t.card_bg)
        recs_card.pack(fill=tk.X, padx=15, pady=10, ipady=10)

        tk.Label(
            recs_card, text="✦ 推薦反制", bg=t.card_bg, fg=t.highlight_color, font=self.normal_font
        ).pack(pady=(8, 5))
        self.lbl_my_recs = tk.Label(
            recs_card, text="— — —", bg=t.card_bg, fg=t.text_dim, font=self.large_font
        )
        self.lbl_my_recs.pack(pady=(0, 10))

        tk.Label(
            recs_card, text="⚠ 預測對手", bg=t.card_bg, fg=t.opp_color, font=self.normal_font
        ).pack(pady=(0, 5))
        self.lbl_opp_preds = tk.Label(
            recs_card, text="— — —", bg=t.card_bg, fg=t.text_dim, font=self.large_font
        )
        self.lbl_opp_preds.pack()

        bottom_frame = tk.Frame(self.root, bg=t.bg_color)
        bottom_frame.pack(fill=tk.X, padx=15, pady=5)

        if self.sandbox_callback is not None:
            self.btn_sandbox = tk.Button(
                bottom_frame,
                text="🧪 Offline Demo",
                bg="#1F4E79",
                fg=t.text_primary,
                font=self.normal_font,
                activebackground="#2B6CB0",
                activeforeground=t.text_primary,
                command=self.sandbox_callback,
                relief=tk.FLAT,
                cursor="hand2",
            )
            self.btn_sandbox.pack(pady=(0, 6), fill=tk.X, padx=60)

        self.btn_reset = tk.Button(
            bottom_frame,
            text="↻ 重置對局",
            bg=t.btn_bg,
            fg=t.text_primary,
            font=self.normal_font,
            activebackground=t.btn_active_bg,
            activeforeground=t.text_primary,
            command=self.trigger_reset,
            relief=tk.FLAT,
            cursor="hand2",
        )
        self.btn_reset.pack(pady=(0, 8), fill=tk.X, padx=60)

        self.lbl_status = tk.Label(
            bottom_frame,
            text="系統就緒，等待畫面掃描...",
            bg=t.bg_color,
            fg=t.text_dim,
            font=self.normal_font,
            width=42,
            wraplength=440,
            justify="center",
        )
        self.lbl_status.pack()

    def disable_mode_click(self) -> None:
        self.mode_canvas.unbind("<Button-1>")
        self.mode_canvas.config(cursor="watch")

    def enable_mode_click(self) -> None:
        self.mode_canvas.bind("<Button-1>", self.cycle_mode)
        self.mode_canvas.config(cursor="hand2")

    def draw_mode_dot(self) -> None:
        self.mode_canvas.delete("all")
        color = self.mode_colors[self.current_mode]
        self.mode_canvas.create_oval(3, 3, 19, 19, fill=color, outline=color)

    def set_mode_ui(self, new_mode: str) -> None:
        mode_key = new_mode.value if isinstance(new_mode, OperatingMode) else str(new_mode)
        if mode_key in self.mode_colors:
            self.current_mode = mode_key
            self.draw_mode_dot()

    def cycle_mode(self, event: tk.Event | None = None) -> None:
        curr_index = MODE_CYCLE_ORDER.index(self.current_mode)
        next_mode = MODE_CYCLE_ORDER[(curr_index + 1) % len(MODE_CYCLE_ORDER)]
        if callable(self.mode_change_callback):
            self.mode_change_callback(next_mode)

    def on_ban_changed(self, selected: str) -> None:
        with self._state_lock:
            self._current_selected_ban = selected
        self.lbl_status.config(text=f"數據切換至： {selected}，等待新對局...")

    def trigger_reset(self) -> None:
        with self._state_lock:
            self._needs_reset = True
        self.lbl_status.config(text="正在重置數據...")

    def update_label(self, label: tk.Label, data_list: list[str] | None, active_color: str) -> None:
        if data_list:
            label.config(text=" ".join(data_list), fg=active_color)
        else:
            label.config(text="— — —", fg=self.theme.text_dim)

    def update_info(
        self,
        opp_ban: str | None,
        opp_draft: list[str],
        my_draft: list[str],
        my_recs: list[str],
        opp_preds: list[str],
        status: str,
    ) -> None:
        if opp_ban:
            self.lbl_opp_ban_val.config(text=opp_ban, font=self.xl_font, fg=self.theme.opp_color)
        else:
            self.lbl_opp_ban_val.config(text="— —", font=self.title_font, fg=self.theme.text_dim)

        self.update_label(self.lbl_opp_draft, opp_draft, self.theme.text_primary)
        self.update_label(self.lbl_my_draft, my_draft, self.theme.text_primary)
        self.update_label(self.lbl_my_recs, my_recs, self.theme.text_primary)
        self.update_label(self.lbl_opp_preds, opp_preds, self.theme.text_primary)
        self.lbl_status.config(text=status)
