"""Thread-Safe Battle UI Presenter for Onmyoji Tactical Helper.

Marshals tactical text updates and HUD overlay rendering onto the Tkinter main thread
via `.after(0, ...)` with strict error isolation so UI glitches never disrupt CV or auto-clicks.
"""

import logging
import time
import tkinter as tk
from collections.abc import Callable
from typing import Any

from onmyoji_helper.domain.models import CropOffset
from onmyoji_helper.ui.hud_overlay import BattleOverlayUI
from onmyoji_helper.ui.tactical_panel import AITacticalUI

logger = logging.getLogger("OnmyojiHelper")


class BattlePresenter:
    """Manages lifecycle and main-thread updates for `AITacticalUI` and `BattleOverlayUI`."""

    def __init__(self) -> None:
        self.ai_ui: AITacticalUI | None = None
        self.overlay_ui: BattleOverlayUI | None = None
        self._last_warn_ts: float = 0.0

    def _warn_throttled(self, message: str) -> None:
        now = time.time()
        if now - self._last_warn_ts >= 5.0:
            self._last_warn_ts = now
            logger.warning(message)

    def ensure_windows(
        self,
        root_window: tk.Tk,
        show_overlay: bool,
        on_closed_by_user: Callable[[], None],
        is_cancelled_fn: Callable[[], bool],
    ) -> None:
        """Schedules creation or refresh of tactical and HUD windows on the main thread."""

        def _create() -> None:
            if is_cancelled_fn():
                return
            try:
                if self.ai_ui is None or not self.ai_ui.winfo_exists():
                    self.ai_ui = AITacticalUI(root_window, close_callback=on_closed_by_user)
                else:
                    self.ai_ui.update_advice("⏳ 正在分析新對局戰術...")
            except Exception as exc:
                self._warn_throttled(f"⚠️ [BattlePresenter] 建立 AI 戰術面板異常 (不影響主循環): {exc}")

            if show_overlay:
                try:
                    if self.overlay_ui is None or not self.overlay_ui.winfo_exists():
                        self.overlay_ui = BattleOverlayUI(root_window)
                except Exception as exc:
                    self._warn_throttled(f"⚠️ [BattlePresenter] 建立 Overlay 覆蓋層異常 (不影響主循環): {exc}")

        try:
            root_window.after(0, _create)
        except Exception as exc:
            self._warn_throttled(f"⚠️ [BattlePresenter] 排程 UI 初始化失敗: {exc}")

    def schedule_overlay_frame(
        self,
        root_window: tk.Tk,
        hwnd: int,
        my_status: dict[str, dict[str, Any]],
        opp_status: dict[str, dict[str, Any]],
        total_alive: int,
        img_w: int,
        img_h: int,
        is_my_turn: bool,
        target_slot: str | None,
        target_is_enemy: bool,
        turn_detector: Any,
        skill_states: dict[str, bool],
        skill_mapping: dict[str, str | None],
        crop: CropOffset | None,
        is_cancelled_fn: Callable[[], bool],
    ) -> None:
        """Snapshots frame state (without retaining the 6MB image array) and schedules HUD draw."""
        my_snap = {k: dict(v) for k, v in my_status.items()}
        opp_snap = {k: dict(v) for k, v in opp_status.items()}
        skills_snap = dict(skill_states)
        mapping_snap = dict(skill_mapping)

        def _draw() -> None:
            if is_cancelled_fn() or self.overlay_ui is None or not self.overlay_ui.winfo_exists():
                return
            try:
                self.overlay_ui.sync_position(hwnd)
                self.overlay_ui.clear()
                if total_alive > 0:
                    self.overlay_ui.update_targets(
                        my_snap,
                        is_enemy=False,
                        img_w=img_w,
                        img_h=img_h,
                        target_slot=(target_slot if not target_is_enemy else None),
                        crop=crop,
                    )
                    self.overlay_ui.update_targets(
                        opp_snap,
                        is_enemy=True,
                        img_w=img_w,
                        img_h=img_h,
                        target_slot=(target_slot if target_is_enemy else None),
                        crop=crop,
                    )
                self.overlay_ui.update_skills(
                    detector=turn_detector,
                    is_my_turn=is_my_turn,
                    skill_states=skills_snap,
                    img_w=img_w,
                    img_h=img_h,
                    skill_mapping=mapping_snap,
                    crop=crop,
                )
            except Exception as exc:
                self._warn_throttled(f"⚠️ [Overlay] 繪圖更新異常 (不影響辨識與自動戰鬥): {exc}")

        try:
            root_window.after(0, _draw)
        except Exception as exc:
            self._warn_throttled(f"⚠️ [Overlay] 排程繪圖異常: {exc}")

    def schedule_advice_text(
        self,
        root_window: tk.Tk | None,
        text: str,
        is_cancelled_fn: Callable[[], bool],
    ) -> None:
        """Schedules tactical advice text update on the main thread."""
        if root_window is None:
            return

        def _apply() -> None:
            if not is_cancelled_fn() and self.ai_ui is not None and self.ai_ui.winfo_exists():
                self.ai_ui.update_advice(text)

        try:
            root_window.after(0, _apply)
        except Exception as exc:
            logger.debug(f"[BattlePresenter] Advice text schedule skipped: {exc}")

    def reset_blank(self, root_window: tk.Tk | None) -> None:
        """Clears text and overlay canvas elements on match completion."""
        ui_ref = self.ai_ui
        ov_ref = self.overlay_ui
        if root_window is None:
            return

        def _clear() -> None:
            if ui_ref is not None and ui_ref.winfo_exists():
                ui_ref.reset_to_blank()
            if ov_ref is not None and ov_ref.winfo_exists():
                ov_ref.clear()

        try:
            root_window.after(0, _clear)
        except Exception as exc:
            logger.debug(f"[BattlePresenter] Blank reset schedule skipped: {exc}")

    def destroy_all(self, root_window: tk.Tk | None) -> None:
        """Destroys both floating windows safely on the main thread."""
        ui_to_destroy = self.ai_ui
        ov_to_destroy = self.overlay_ui
        self.ai_ui = None
        self.overlay_ui = None
        if root_window is None:
            return

        def _destroy() -> None:
            if ui_to_destroy is not None:
                ui_to_destroy.destroy()
            if ov_to_destroy is not None:
                ov_to_destroy.close()

        try:
            root_window.after(0, _destroy)
        except Exception as exc:
            logger.debug(f"[BattlePresenter] Window destroy schedule skipped: {exc}")
