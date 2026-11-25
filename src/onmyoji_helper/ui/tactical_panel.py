"""AI Tactical Advice Floating Panel for Onmyoji Tactical Helper.

Displays real-time Gemma-4 or rule-based tactical advice inside a topmost non-activating
(`WS_EX_NOACTIVATE`) Tkinter Toplevel window without stealing keyboard/mouse focus from the game.
"""

import logging
import tkinter as tk
from collections.abc import Callable

from onmyoji_helper.infra.win32_window import (
    WS_EX_NOACTIVATE,
    apply_extended_window_style,
    get_tk_win32_hwnd,
)

logger = logging.getLogger("OnmyojiHelper")


class AITacticalUI:
    """Floating topmost window displaying real-time tactical recommendations."""

    def __init__(
        self,
        parent_root: tk.Tk,
        close_callback: Callable[[], None] | None = None,
    ) -> None:
        """Initializes the non-activating Toplevel window, white-bordered card, and advice label."""
        self.window: tk.Toplevel = tk.Toplevel(parent_root)
        self.window.title("AI 助手")
        self.window.geometry("520x320")
        self.window.resizable(False, False)
        self.window.attributes("-topmost", True)
        self.window.configure(bg="#121212")
        self.close_callback: Callable[[], None] | None = close_callback

        self.window.protocol("WM_DELETE_WINDOW", self._on_close)

        self.box_frame: tk.Frame = tk.Frame(
            self.window,
            bg="#121212",
            highlightbackground="#FFFFFF",
            highlightthickness=1,
        )
        self.box_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        self.lbl_advice: tk.Label = tk.Label(
            self.box_frame,
            text="",
            font=("Microsoft JhengHei", 10),
            fg="#FFFFFF",
            bg="#1C1C1E",
            wraplength=430,
            justify="left",
            anchor="nw",
        )
        self.lbl_advice.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # Complete normal Tkinter mapping first, then apply 64-bit safe WS_EX_NOACTIVATE
        self.window.deiconify()
        self.window.update_idletasks()
        try:
            hwnd = get_tk_win32_hwnd(self.window)
            apply_extended_window_style(hwnd, WS_EX_NOACTIVATE, topmost=True)
        except Exception as exc:
            logger.debug(f"⚠️ [AITacticalUI] Failed to apply WS_EX_NOACTIVATE: {exc}")

    def winfo_exists(self) -> bool:
        """Returns True if the underlying Tkinter Toplevel still exists."""
        try:
            return bool(self.window.winfo_exists())
        except Exception:
            return False

    def update_advice(self, advice_text: str) -> None:
        """Updates the tactical advice text inside the display box."""
        try:
            if self.winfo_exists():
                self.lbl_advice.config(text=advice_text)
        except Exception as exc:
            logger.debug(f"[AITacticalUI] Advice label update skipped: {exc}")

    def reset_to_blank(self) -> None:
        """Clears the tactical advice text."""
        try:
            if self.winfo_exists():
                self.lbl_advice.config(text="")
        except Exception as exc:
            logger.debug(f"[AITacticalUI] Reset to blank skipped: {exc}")

    def _on_close(self) -> None:
        if callable(self.close_callback):
            try:
                self.close_callback()
            except Exception as exc:
                logger.warning(f"⚠️ [AITacticalUI] Close callback exception: {exc}")
        self.destroy()

    def destroy(self) -> None:
        """Safely destroys the Toplevel window."""
        try:
            if self.winfo_exists():
                self.window.destroy()
        except Exception as exc:
            logger.debug(f"[AITacticalUI] Window destroy skipped: {exc}")
