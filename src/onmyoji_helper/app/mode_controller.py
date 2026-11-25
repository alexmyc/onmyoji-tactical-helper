"""Thread-Safe Operational Mode Controller for Onmyoji Tactical Helper.

Manages transitions across Green, Yellow, and Red modes using an RLock and transition
epochs so an emergency F12/ESC downgrade is never overwritten by an in-flight switch thread,
and stores lightweight draft metadata without retaining raw 6MB image arrays.
"""

import logging
import threading
import time
import tkinter as tk
from typing import Any

from onmyoji_helper.domain.models import OperatingMode

logger = logging.getLogger("OnmyojiHelper")


class ModeController:
    """Synchronizes tri-state mode switches, GPU resource lifecycle, and emergency downgrades."""

    MODE_GREEN: str = OperatingMode.GREEN.value
    MODE_YELLOW: str = OperatingMode.YELLOW.value
    MODE_RED: str = OperatingMode.RED.value

    def __init__(self, tactical_mgr: Any | None = None) -> None:
        self._lock: threading.RLock = threading.RLock()
        self.current_mode: str = self.MODE_GREEN
        self.is_busy: bool = False
        self._transition_epoch: int = 0

        self.tactical_mgr: Any | None = tactical_mgr
        if self.tactical_mgr and hasattr(self.tactical_mgr, "set_mode_controller"):
            self.tactical_mgr.set_mode_controller(self)

        # Stores lightweight tuple: (root_window, my_draft, opp_draft, hwnd) — zero raw image retention
        self.saved_draft_data: tuple[tk.Tk, list[str], list[str], int] | None = None
        self.ui_app: Any | None = None

    def set_ui_app(self, ui_app: Any) -> None:
        self.ui_app = ui_app

    def save_and_handle_five_picks(
        self,
        root_window: tk.Tk,
        my_draft: list[str],
        opp_draft: list[str],
        screen_color: Any = None,
        hwnd: int = 0,
    ) -> None:
        """Persists lightweight 5-pick lineup metadata and triggers battle analysis if in AI mode."""
        with self._lock:
            self.saved_draft_data = (root_window, list(my_draft), list(opp_draft), int(hwnd))
        self.handle_five_picks(root_window, my_draft, opp_draft, screen_color, hwnd)

    def clear_draft_data(self) -> None:
        with self._lock:
            self.saved_draft_data = None

    def set_mode(self, new_mode: str) -> bool:
        """Initiates an asynchronous mode transition under `_lock`."""
        target = new_mode.value if isinstance(new_mode, OperatingMode) else str(new_mode)
        with self._lock:
            if self.is_busy or target == self.current_mode:
                return False
            self.is_busy = True
            self._transition_epoch += 1
            epoch = self._transition_epoch
            old_mode = self.current_mode

        if self.ui_app:
            self.ui_app.disable_mode_click()
            self.ui_app.set_mode_ui(target)
            if hasattr(self.ui_app, "lbl_status"):
                self.ui_app.lbl_status.config(text="⏳ 正在安全處理 GPU 資源，請稍候...")

        logger.info(f"🔄 [ModeController] 切換運作模式: {old_mode} -> {target}")
        threading.Thread(
            target=self._async_switch_mode,
            args=(old_mode, target, epoch),
            name="ModeSwitchWorker",
            daemon=True,
        ).start()
        return True

    def _async_switch_mode(self, old_mode: str, new_mode: str, epoch: int) -> None:
        try:
            if old_mode in (self.MODE_YELLOW, self.MODE_RED):
                if self.tactical_mgr and hasattr(self.tactical_mgr, "close_ui_sync"):
                    self.tactical_mgr.close_ui_sync()
                time.sleep(0.35)

            with self._lock:
                if epoch != self._transition_epoch:
                    return
                self.current_mode = new_mode
                saved = self.saved_draft_data

            if new_mode in (self.MODE_YELLOW, self.MODE_RED) and self.tactical_mgr:
                if hasattr(self.tactical_mgr, "enable_mode"):
                    self.tactical_mgr.enable_mode(new_mode)
                if saved is not None:
                    r_win, my_d, opp_d, saved_hwnd = saved
                    self.handle_five_picks(r_win, my_d, opp_d, None, saved_hwnd, sync_wait=True)
        except Exception as exc:
            logger.warning(f"⚠️ [ModeController] 模式切換異常: {exc}")
        finally:
            with self._lock:
                still_current_epoch = epoch == self._transition_epoch
                if still_current_epoch:
                    self.is_busy = False

            if still_current_epoch and self.ui_app:
                self.ui_app.root.after(0, self.ui_app.enable_mode_click)
                if hasattr(self.ui_app, "lbl_status") and hasattr(self.ui_app, "mode_names"):
                    mode_desc = self.ui_app.mode_names.get(new_mode, new_mode)
                    self.ui_app.root.after(
                        0, lambda desc=mode_desc: self.ui_app.lbl_status.config(text=f"已切換至：{desc}")
                    )

    def update_status_text(self, text: str) -> None:
        """Thread-safely updates the primary window's bottom status bar."""
        if self.ui_app and hasattr(self.ui_app, "root") and hasattr(self.ui_app, "lbl_status"):
            try:
                self.ui_app.root.after(0, lambda msg=text: self.ui_app.lbl_status.config(text=msg))
            except Exception:
                pass

    def emergency_downgrade_to_yellow(self, reason: str = "") -> None:
        """Instantly halts Red Mode auto-clicking and switches to Yellow Mode without unloading VRAM."""
        if self.tactical_mgr and hasattr(self.tactical_mgr, "disable_auto_battle"):
            self.tactical_mgr.disable_auto_battle(reason)

        with self._lock:
            self._transition_epoch += 1
            self.current_mode = self.MODE_YELLOW
            self.is_busy = False

        status_msg = (
            f"⚠️ 已手動緊急煞車，切回黃色模式 ({reason})"
            if reason
            else "⚠️ 已手動緊急煞車，切回黃色模式"
        )

        if self.ui_app and hasattr(self.ui_app, "root"):
            def _apply_ui() -> None:
                try:
                    self.ui_app.set_mode_ui(self.MODE_YELLOW)
                    self.ui_app.enable_mode_click()
                    if hasattr(self.ui_app, "lbl_status"):
                        self.ui_app.lbl_status.config(text=status_msg)
                except Exception:
                    pass

            try:
                self.ui_app.root.after(0, _apply_ui)
            except Exception:
                pass

    def handle_five_picks(
        self,
        root_window: tk.Tk,
        my_draft: list[str],
        opp_draft: list[str],
        screen_color: Any = None,
        hwnd: int = 0,
        sync_wait: bool = False,
    ) -> None:
        with self._lock:
            mode = self.current_mode
        if mode in (self.MODE_YELLOW, self.MODE_RED) and self.tactical_mgr:
            self.tactical_mgr.trigger_analysis(
                root_window, my_draft, opp_draft, screen_color, hwnd, sync_wait=sync_wait
            )
