# -*- coding: utf-8 -*-
"""AI Tactical UI Overlay Module for Onmyoji Tactical Helper.

This module provides a singleton Tkinter Toplevel window displaying real-time LLM tactical
recommendations inside a clean, dark-themed white-bordered display box.
"""

import tkinter as tk
from typing import Callable, Optional


class AITacticalUI:
    """Singleton Tkinter Toplevel window for displaying real-time tactical advice."""

    _instance: Optional["AITacticalUI"] = None

    def __new__(
        cls,
        parent_root: tk.Tk,
        close_callback: Optional[Callable[[], None]] = None
    ) -> "AITacticalUI":
        """Ensures singleton instantiation of the AI Tactical UI window.

        Args:
            parent_root: Reference to the parent Tkinter root instance.
            close_callback: Optional callback function triggered when the window is closed.

        Returns:
            Singleton instance of AITacticalUI.
        """
        if cls._instance is not None and cls._instance.window.winfo_exists():
            if close_callback:
                cls._instance.close_callback = close_callback
            return cls._instance
        instance = super(AITacticalUI, cls).__new__(cls)
        cls._instance = instance
        return instance

    def __init__(
        self,
        parent_root: tk.Tk,
        close_callback: Optional[Callable[[], None]] = None
    ) -> None:
        """Initializes window layout, border frame, and advice label widget.

        Args:
            parent_root: Reference to the parent Tkinter root instance.
            close_callback: Optional callback function triggered when the window is closed.
        """
        if getattr(self, '_initialized', False):
            return
        self._initialized: bool = True

        self.window: tk.Toplevel = tk.Toplevel(parent_root)
        self.window.title("AI 助手")
        self.window.geometry("520x320")
        self.window.resizable(False, False)
        self.window.attributes("-topmost", True)
        self.window.configure(bg="#121212")
        self.close_callback: Optional[Callable[[], None]] = close_callback

        # Listen for window WM_DELETE_WINDOW close event
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)

        # Minimal white-bordered box frame container
        self.box_frame: tk.Frame = tk.Frame(
            self.window,
            bg="#121212",
            highlightbackground="#FFFFFF",
            highlightthickness=1
        )
        self.box_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Tactical advice display label widget (blank by default)
        self.lbl_advice: tk.Label = tk.Label(
            self.box_frame,
            text="",
            font=("Microsoft JhengHei", 10),
            fg="#FFFFFF",
            bg="#1C1C1E",
            wraplength=430,
            justify="left",
            anchor="nw"
        )
        self.lbl_advice.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

    def update_advice(self, advice_text: str) -> None:
        """Updates the tactical advice text inside the UI display box.

        Args:
            advice_text: Formatted tactical recommendation string.
        """
        if self.window.winfo_exists():
            self.lbl_advice.config(text=advice_text)

    def reset_to_blank(self) -> None:
        """Clears tactical advice text, resetting the display box to a blank state."""
        if self.window.winfo_exists():
            self.lbl_advice.config(text="")

    def _on_close(self) -> None:
        """Handles WM_DELETE_WINDOW event, resets singleton state, and destroys window."""
        AITacticalUI._instance = None
        self._initialized = False
        if self.close_callback:
            try:
                self.close_callback()
            except Exception as e:
                print(f"⚠️ Close window callback exception: {e}")
        try:
            self.window.destroy()
        except Exception:
            pass

    def close(self) -> None:
        """Public wrapper method to trigger window closure and state cleanup."""
        self._on_close()