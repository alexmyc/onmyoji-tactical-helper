# -*- coding: utf-8 -*-
"""Graphical User Interface Module for Onmyoji Tactical Helper.

This module provides a dark-themed Tkinter user interface for rendering
real-time ban/pick detection, tactical counter recommendations, mode indicators,
and interactive match controls.
"""

import tkinter as tk
from tkinter import font
from typing import Callable, Dict, List, Optional


class OnmyojiHelperUI:
    """Primary Tkinter GUI window managing match drafting state displays."""

    def __init__(
        self,
        root: tk.Tk,
        available_bans: Optional[List[str]],
        mode_change_callback: Optional[Callable[[str], None]] = None
    ) -> None:
        """Initializes GUI components, layout geometries, colors, and fonts.

        Args:
            root: Tkinter root window instance.
            available_bans: List of selectable ban dataset names.
            mode_change_callback: Callback function invoked when mode dot is toggled.
        """
        self.root: tk.Tk = root
        self.available_bans: List[str] = available_bans if available_bans else ["未載入"]
        self.mode_change_callback: Optional[Callable[[str], None]] = mode_change_callback

        self.root.title("陰陽師鬥技助手")
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 1.0)

        window_width: int = 520
        window_height: int = 500
        screen_width: int = self.root.winfo_screenwidth()
        pos_x: int = screen_width - window_width - 50
        pos_y: int = 100
        self.root.geometry(f"{window_width}x{window_height}+{pos_x}+{pos_y}")
        self.root.resizable(False, False)

        # Color Palette Configuration
        self.bg_color: str = "#121212"
        self.card_bg: str = "#1C1C1E"
        self.menu_bg: str = "#1C1C1E"
        self.text_primary: str = "#F5F5F7"
        self.text_dim: str = "#636366"

        self.ally_color: str = "#4DA8DA"
        self.opp_color: str = "#EE6C4D"
        self.highlight_color: str = "#E3B341"

        self.btn_bg: str = "#8B2626"
        self.btn_active_bg: str = "#A83232"

        # Mode Indicator Configurations
        self.current_mode: str = "green"
        self.mode_colors: Dict[str, str] = {
            "green": "#34C759",
            "yellow": "#FFCC00",
            "red": "#FF3B30"
        }
        self.mode_names: Dict[str, str] = {
            "green": "綠色模式 (純助手主體)",
            "yellow": "黃色模式 (助手 + LLM 戰術對策)",
            "red": "紅色模式 (未開放/預留功能)"
        }

        self.root.configure(bg=self.bg_color)

        # Font Configurations
        self.title_font: font.Font = font.Font(family="Microsoft JhengHei", size=11, weight="bold")
        self.normal_font: font.Font = font.Font(family="Microsoft JhengHei", size=10)
        self.large_font: font.Font = font.Font(family="Microsoft JhengHei", size=13, weight="bold")
        self.xl_font: font.Font = font.Font(family="Microsoft JhengHei", size=16, weight="bold")

        self.needs_reset: bool = False
        self.selected_ban: tk.StringVar = tk.StringVar(value=self.available_bans[0])

        self.create_widgets()

    def create_widgets(self) -> None:
        """Constructs and arranges all UI widgets and container frames."""
        top_card = tk.Frame(self.root, bg=self.card_bg)
        top_card.pack(pady=(15, 10), fill=tk.X, padx=15, ipadx=10, ipady=8)

        self.mode_canvas = tk.Canvas(
            top_card, width=22, height=22, bg=self.card_bg, highlightthickness=0, cursor="hand2"
        )
        self.mode_canvas.pack(side=tk.LEFT, padx=(0, 12))
        self.draw_mode_dot()
        self.enable_mode_click()

        my_ban_frame = tk.Frame(top_card, bg=self.card_bg)
        my_ban_frame.pack(side=tk.LEFT)
        tk.Label(
            my_ban_frame, text="數據:", bg=self.card_bg, fg=self.text_dim, font=self.title_font
        ).pack(side=tk.LEFT)

        self.ban_menu = tk.OptionMenu(
            my_ban_frame, self.selected_ban, *self.available_bans, command=self.on_ban_changed
        )
        self.ban_menu.config(
            bg=self.menu_bg,
            fg=self.text_primary,
            activebackground="#3A3A3C",
            activeforeground=self.text_primary,
            highlightthickness=0,
            bd=0,
            relief=tk.FLAT,
            indicatoron=0,
            cursor="hand2",
            font=self.title_font
        )
        self.ban_menu["menu"].config(
            bg=self.menu_bg, fg=self.text_primary, bd=0, relief=tk.FLAT, font=self.title_font
        )
        self.ban_menu.pack(side=tk.LEFT, padx=5)

        opp_ban_frame = tk.Frame(top_card, bg=self.card_bg)
        opp_ban_frame.pack(side=tk.RIGHT)
        tk.Label(
            opp_ban_frame, text="敵方禁用:", bg=self.card_bg, fg=self.text_dim, font=self.title_font
        ).pack(side=tk.LEFT)
        self.lbl_opp_ban_val = tk.Label(
            opp_ban_frame, text="— —", bg=self.card_bg, fg=self.text_dim, font=self.title_font
        )
        self.lbl_opp_ban_val.pack(side=tk.LEFT, padx=(5, 0))

        draft_card = tk.Frame(self.root, bg=self.bg_color)
        draft_card.pack(fill=tk.X, pady=5, padx=15)

        my_frame = tk.Frame(draft_card, bg=self.card_bg)
        my_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(0, 5), ipady=12)
        tk.Label(
            my_frame, text="⛨ 我方陣容", bg=self.card_bg, fg=self.ally_color, font=self.normal_font
        ).pack(pady=(5, 8))
        self.lbl_my_draft = tk.Label(
            my_frame, text="— — —", bg=self.card_bg, fg=self.text_dim, font=self.large_font, wraplength=200
        )
        self.lbl_my_draft.pack()

        opp_frame = tk.Frame(draft_card, bg=self.card_bg)
        opp_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=(5, 0), ipady=12)
        tk.Label(
            opp_frame, text="⚔ 敵方陣容", bg=self.card_bg, fg=self.opp_color, font=self.normal_font
        ).pack(pady=(5, 8))
        self.lbl_opp_draft = tk.Label(
            opp_frame, text="— — —", bg=self.card_bg, fg=self.text_dim, font=self.large_font, wraplength=200
        )
        self.lbl_opp_draft.pack()

        recs_card = tk.Frame(self.root, bg=self.card_bg)
        recs_card.pack(fill=tk.X, padx=15, pady=10, ipady=10)

        tk.Label(
            recs_card, text="✦ 推薦反制", bg=self.card_bg, fg=self.highlight_color, font=self.normal_font
        ).pack(pady=(8, 5))
        self.lbl_my_recs = tk.Label(
            recs_card, text="— — —", bg=self.card_bg, fg=self.text_dim, font=self.large_font
        )
        self.lbl_my_recs.pack(pady=(0, 10))

        tk.Label(
            recs_card, text="⚠ 預測對手", bg=self.card_bg, fg=self.opp_color, font=self.normal_font
        ).pack(pady=(0, 5))
        self.lbl_opp_preds = tk.Label(
            recs_card, text="— — —", bg=self.card_bg, fg=self.text_dim, font=self.large_font
        )
        self.lbl_opp_preds.pack()

        bottom_frame = tk.Frame(self.root, bg=self.bg_color)
        bottom_frame.pack(fill=tk.X, padx=15, pady=5)

        self.btn_reset = tk.Button(
            bottom_frame,
            text="↻ 重置對局",
            bg=self.btn_bg,
            fg=self.text_primary,
            font=self.normal_font,
            activebackground=self.btn_active_bg,
            activeforeground=self.text_primary,
            command=self.trigger_reset,
            relief=tk.FLAT,
            cursor="hand2"
        )
        self.btn_reset.pack(pady=(0, 8), fill=tk.X, padx=60)

        self.lbl_status = tk.Label(
            bottom_frame,
            text="系統就緒，等待畫面掃描...",
            bg=self.bg_color,
            fg=self.text_dim,
            font=self.normal_font,
            wraplength=450
        )
        self.lbl_status.pack()

    def disable_mode_click(self) -> None:
        """Unbinds mode toggle click events during background resource processing."""
        self.mode_canvas.unbind("<Button-1>")
        self.mode_canvas.config(cursor="watch")

    def enable_mode_click(self) -> None:
        """Binds mode toggle click events to the mode indicator canvas."""
        self.mode_canvas.bind("<Button-1>", self.cycle_mode)
        self.mode_canvas.config(cursor="hand2")

    def draw_mode_dot(self) -> None:
        """Renders the colored mode status circle on the mode canvas."""
        self.mode_canvas.delete("all")
        color = self.mode_colors[self.current_mode]
        self.mode_canvas.create_oval(3, 3, 19, 19, fill=color, outline=color)

    def set_mode_ui(self, new_mode: str) -> None:
        """Updates the visual mode indicator color on the UI.

        Args:
            new_mode: Target mode identifier ("green", "yellow", or "red").
        """
        if new_mode in self.mode_colors:
            self.current_mode = new_mode
            self.draw_mode_dot()

    def cycle_mode(self, event: Optional[tk.Event] = None) -> None:
        """Cycles to the next operational mode when the user clicks the mode dot.

        Args:
            event: Optional Tkinter click event object.
        """
        modes = ["green", "yellow", "red"]
        curr_index = modes.index(self.current_mode)
        next_mode = modes[(curr_index + 1) % len(modes)]

        if callable(self.mode_change_callback):
            self.mode_change_callback(next_mode)

    def get_current_mode(self) -> str:
        """Retrieves the active UI mode.

        Returns:
            Current mode string identifier.
        """
        return self.current_mode

    def on_ban_changed(self, selected: str) -> None:
        """Handles user dataset selection changes in the dropdown menu.

        Args:
            selected: Selected dataset identifier.
        """
        self.lbl_status.config(text=f"數據切換至： {selected}，等待新對局...")

    def trigger_reset(self) -> None:
        """Flags current match data for reset and updates the status label."""
        self.needs_reset = True
        self.lbl_status.config(text="正在重置數據...")

    def update_label(
        self,
        label: tk.Label,
        data_list: Optional[List[str]],
        active_color: str
    ) -> None:
        """Updates text and foreground color for a specified draft display label.

        Args:
            label: Reference to the target Tkinter Label widget.
            data_list: List of character names to display.
            active_color: Text color to apply when data list is non-empty.
        """
        if data_list:
            label.config(text=" ".join(data_list), fg=active_color)
        else:
            label.config(text="— — —", fg=self.text_dim)

    def update_info(
        self,
        opp_ban: Optional[str],
        opp_draft: List[str],
        my_draft: List[str],
        my_recs: List[str],
        opp_preds: List[str],
        status: str
    ) -> None:
        """Updates all match information displays on the user interface.

        Args:
            opp_ban: Detected opponent banned character name.
            opp_draft: List of detected opponent drafted characters.
            my_draft: List of detected player drafted characters.
            my_recs: Recommended counter picks.
            opp_preds: Predicted opponent choices.
            status: General status description text.
        """
        if opp_ban:
            self.lbl_opp_ban_val.config(text=opp_ban, font=self.xl_font, fg=self.opp_color)
        else:
            self.lbl_opp_ban_val.config(text="— —", font=self.title_font, fg=self.text_dim)

        self.update_label(self.lbl_opp_draft, opp_draft, self.text_primary)
        self.update_label(self.lbl_my_draft, my_draft, self.text_primary)
        self.update_label(self.lbl_my_recs, my_recs, self.text_primary)
        self.update_label(self.lbl_opp_preds, opp_preds, self.text_primary)

        self.lbl_status.config(text=status)