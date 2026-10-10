# -*- coding: utf-8 -*-
"""Mode Controller Module for Onmyoji Tactical Helper.

This module acts as a state manager governing system operation modes (Green, Yellow, Red)
and orchestrates tactical component lifecycles, UI controls, and GPU resource transitions.
"""

import threading
import time
import tkinter as tk
from typing import Any, List, Optional, Tuple


class ModeController:
    """Controls application execution modes and manages background AI analysis triggers."""

    MODE_GREEN: str = "green"   # 🟢 Core assistant mode (No LLM active)
    MODE_YELLOW: str = "yellow" # 🟡 Assistant mode + LLM tactical advice (Text-only)
    MODE_RED: str = "red"       # 🔴 Assistant mode + LLM tactical advice + Debug HUD Overlay

    def __init__(self, tactical_mgr: Optional[Any] = None) -> None:
        """Initializes the mode controller and registers self with the tactical manager.

        Args:
            tactical_mgr: Optional reference to the AITacticalManager instance.
        """
        self.current_mode: str = self.MODE_GREEN
        self.tactical_mgr: Optional[Any] = tactical_mgr

        # Automatically register self with TacticalManager for reverse UI synchronization
        if self.tactical_mgr and hasattr(self.tactical_mgr, "set_mode_controller"):
            self.tactical_mgr.set_mode_controller(self)

        self.saved_draft_data: Optional[Tuple[tk.Tk, List[str], List[str], Any, int]] = None
        self.is_busy: bool = False
        self.ui_app: Optional[Any] = None

    def set_ui_app(self, ui_app: Any) -> None:
        """Binds the primary UI application instance.

        Args:
            ui_app: Reference to the main OnmyojiHelperUI application instance.
        """
        self.ui_app = ui_app

    def save_and_handle_five_picks(self,
                                   root_window: tk.Tk,
                                   my_draft: List[str],
                                   opp_draft: List[str],
                                   screen_color: Any,
                                   hwnd: int) -> None:
        """Saves current draft selection parameters and initiates tactical analysis.

        Args:
            root_window: Reference to the main Tkinter root window.
            my_draft: List of player drafted character names.
            opp_draft: List of opponent drafted character names.
            screen_color: Screen frame image array.
            hwnd: Game window handle identifier.
        """
        self.saved_draft_data = (root_window, list(my_draft), list(opp_draft), screen_color, hwnd)
        self.handle_five_picks(root_window, my_draft, opp_draft, screen_color, hwnd)

    def clear_draft_data(self) -> None:
        """Clears stored draft selection data."""
        self.saved_draft_data = None

    def set_mode(self, new_mode: str) -> bool:
        """Requests a transition to a new execution mode asynchronously.

        Args:
            new_mode: Target operating mode ("green", "yellow", or "red").

        Returns:
            True if mode change request initiated successfully, False if busy or unchanged.
        """
        if self.is_busy or new_mode == self.current_mode:
            return False

        self.is_busy = True

        if self.ui_app:
            self.ui_app.disable_mode_click()
            self.ui_app.set_mode_ui(new_mode)
            if hasattr(self.ui_app, "lbl_status"):
                self.ui_app.lbl_status.config(text="⏳ 正在安全處理 GPU 資源，請稍候...")

        old_mode = self.current_mode
        print(f"🔄 [ModeController] Locking state and switching mode: {old_mode} -> {new_mode}")

        threading.Thread(
            target=self._async_switch_mode,
            args=(old_mode, new_mode),
            daemon=True
        ).start()
        return True

    def _async_switch_mode(self, old_mode: str, new_mode: str) -> None:
        """Asynchronously stops active models, updates mode state, and re-triggers AI analysis.

        Args:
            old_mode: Previous operating mode.
            new_mode: Target operating mode.
        """
        try:
            # 1. Teardown previous mode (Release CUDA memory if leaving Yellow or Red mode)
            if old_mode in [self.MODE_YELLOW, self.MODE_RED]:
                if self.tactical_mgr and hasattr(self.tactical_mgr, "close_ui_sync"):
                    self.tactical_mgr.close_ui_sync()
                time.sleep(0.5)

            self.current_mode = new_mode

            # 2. Setup new mode (Activate AI analysis for Yellow or Red mode)
            if new_mode in [self.MODE_YELLOW, self.MODE_RED]:
                if self.tactical_mgr:
                    if hasattr(self.tactical_mgr, "enable_mode"):
                        self.tactical_mgr.enable_mode(new_mode)
                    elif hasattr(self.tactical_mgr, "enable_yellow_mode"):
                        self.tactical_mgr.enable_yellow_mode()

                    if self.saved_draft_data:
                        self.handle_five_picks(*self.saved_draft_data, sync_wait=True)

        except Exception as e:
            print(f"⚠️ Exception during mode transition: {e}")
        finally:
            self.is_busy = False
            print(f"✅ [ModeController] Mode transition completed: {new_mode}. User interaction restored.")

            if self.ui_app:
                self.ui_app.root.after(0, self.ui_app.enable_mode_click)
                if hasattr(self.ui_app, "lbl_status") and hasattr(self.ui_app, "mode_names"):
                    mode_desc = self.ui_app.mode_names.get(new_mode, new_mode)
                    self.ui_app.root.after(
                        0, lambda: self.ui_app.lbl_status.config(text=f"已切換至：{mode_desc}")
                    )

    def get_mode(self) -> str:
        """Retrieves the current operating mode.

        Returns:
            Current mode identifier string.
        """
        return self.current_mode

    def handle_five_picks(self,
                          root_window: tk.Tk,
                          my_draft: List[str],
                          opp_draft: List[str],
                          screen_color: Any,
                          hwnd: int,
                          sync_wait: bool = False) -> None:
        """Triggers AI tactical analysis when in active AI modes (Yellow or Red).

        Args:
            root_window: Primary Tkinter root window instance.
            my_draft: List of player drafted character names.
            opp_draft: List of opponent drafted character names.
            screen_color: Screen frame image array.
            hwnd: Game window handle identifier.
            sync_wait: Whether to block until AI model loading completes.
        """
        if self.current_mode in [self.MODE_YELLOW, self.MODE_RED]:
            if self.tactical_mgr:
                self.tactical_mgr.trigger_analysis(
                    root_window, my_draft, opp_draft, screen_color, hwnd, sync_wait=sync_wait
                )