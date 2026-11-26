"""Thread-Safe Operational Mode Controller for Onmyoji Tactical Helper.

Manages transitions across Green, Yellow, and Red modes under a strict Battle Lifecycle Gate:
- Before 5-pick draft lock (`saved_draft_data is None`), switching to Yellow or Red arms the
  mode and pre-warms GPU models in the background while keeping `AITacticalUI` and
  `BattleOverlayUI` strictly hidden.
- Once 5 picks lock and 5v5 battle entry is confirmed, `Yellow <-> Red` flips auto-execution
  in-place without tearing down `BattleOverlayUI`, `AITacticalUI`, or pre-warmed GPU models.
"""

import logging
import threading
import tkinter as tk
from typing import Any

from onmyoji_helper.domain.models import OperatingMode

logger = logging.getLogger("OnmyojiHelper")


class ModeController:
    """Synchronizes tri-state mode switches, battle-gated HUD lifecycle, and emergency downgrades."""

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

        # Stores lightweight tuple: (root_window, my_draft, opp_draft, hwnd) — None before 5-pick lock
        self.saved_draft_data: tuple[tk.Tk, list[str], list[str], int] | None = None
        self.ui_app: Any | None = None

    def set_ui_app(self, ui_app: Any) -> None:
        self.ui_app = ui_app
        if self.tactical_mgr and hasattr(self.ui_app, "root"):
            self.tactical_mgr.root_window = self.ui_app.root

    def save_and_handle_five_picks(
        self,
        root_window: tk.Tk,
        my_draft: list[str],
        opp_draft: list[str],
        screen_color: Any = None,
        hwnd: int = 0,
    ) -> None:
        """Persists lightweight 5-pick lineup metadata and launches battle tracking if in Yellow/Red."""
        with self._lock:
            self.saved_draft_data = (root_window, list(my_draft), list(opp_draft), int(hwnd))
        self.handle_five_picks(root_window, my_draft, opp_draft, screen_color, hwnd)

    def clear_draft_data(self) -> None:
        with self._lock:
            self.saved_draft_data = None

    def set_mode(self, new_mode: str) -> bool:
        """Initiates a mode transition under `_lock` without prematurely popping battle windows."""
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

        logger.info(f"🔄 [ModeController] 切換運作模式: {old_mode} -> {target}")
        threading.Thread(
            target=self._async_switch_mode,
            args=(old_mode, target, epoch),
            name="ModeSwitchWorker",
            daemon=True,
        ).start()
        return True

    def _async_switch_mode(self, old_mode: str, new_mode: str, epoch: int) -> None:
        saved: tuple[tk.Tk, list[str], list[str], int] | None = None
        try:
            # Case 1: Leaving Yellow/Red -> Green (close floating HUDs, keep models warm for next switch)
            if new_mode == self.MODE_GREEN and old_mode in (self.MODE_YELLOW, self.MODE_RED):
                if self.tactical_mgr and hasattr(self.tactical_mgr, "close_ui_sync"):
                    self.tactical_mgr.close_ui_sync(unload_models=False)

            with self._lock:
                if epoch != self._transition_epoch:
                    return
                self.current_mode = new_mode
                saved = self.saved_draft_data

            # Case 2: Entering or toggling between Yellow <-> Red
            if new_mode in (self.MODE_YELLOW, self.MODE_RED) and self.tactical_mgr:
                has_locked_draft = saved is not None
                if hasattr(self.tactical_mgr, "enable_mode"):
                    self.tactical_mgr.enable_mode(new_mode, has_locked_draft=has_locked_draft)

                if not has_locked_draft:
                    # Strict Battle Lifecycle Gate: never pop HUD/AI panels or run YOLO before 5-pick lock!
                    if hasattr(self.tactical_mgr, "hide_battle_windows"):
                        self.tactical_mgr.hide_battle_windows()
                else:
                    is_running = bool(
                        getattr(self.tactical_mgr, "is_active_session", False)
                        and getattr(self.tactical_mgr, "battle_thread", None) is not None
                        and self.tactical_mgr.battle_thread.is_alive()
                    )
                    if is_running:
                        if hasattr(self.tactical_mgr, "wake_loop"):
                            self.tactical_mgr.wake_loop()
                    else:
                        assert saved is not None
                        r_win, my_d, opp_d, saved_hwnd = saved
                        self.tactical_mgr.trigger_analysis(
                            r_win, my_d, opp_d, None, saved_hwnd, sync_wait=False
                        )
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
                    if new_mode in (self.MODE_YELLOW, self.MODE_RED) and saved is None:
                        status_txt = f"已切換至：{mode_desc}（待 5 手鎖陣進場後自動開啟戰術視窗）"
                    else:
                        status_txt = f"已切換至：{mode_desc}"
                    self.ui_app.root.after(
                        0, lambda txt=status_txt: self.ui_app.lbl_status.config(text=txt)
                    )

    def update_status_text(self, text: str) -> None:
        """Thread-safely updates the primary window's bottom status bar."""
        if self.ui_app and hasattr(self.ui_app, "root") and hasattr(self.ui_app, "lbl_status"):
            try:
                self.ui_app.root.after(0, lambda msg=text: self.ui_app.lbl_status.config(text=msg))
            except Exception as exc:
                logger.debug(f"[ModeController] Status bar update skipped: {exc}")

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
            ui_ref = self.ui_app

            def _apply_ui() -> None:
                try:
                    ui_ref.set_mode_ui(self.MODE_YELLOW)
                    ui_ref.enable_mode_click()
                    if hasattr(ui_ref, "lbl_status"):
                        ui_ref.lbl_status.config(text=status_msg)
                except Exception as exc:
                    logger.debug(f"[ModeController] Emergency UI widget update skipped: {exc}")

            try:
                ui_ref.root.after(0, _apply_ui)
            except Exception as exc:
                logger.debug(f"[ModeController] Emergency UI schedule skipped: {exc}")

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
            if hasattr(self.tactical_mgr, "enable_mode"):
                self.tactical_mgr.enable_mode(mode, has_locked_draft=True)
            self.tactical_mgr.trigger_analysis(
                root_window, my_draft, opp_draft, screen_color, hwnd, sync_wait=sync_wait
            )
