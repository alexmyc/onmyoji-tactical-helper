"""Humanized Auto-Battle Executor for Onmyoji Tactical Helper (Red Mode).

Orchestrates 2-step Onmyoji turn execution (`[alive unit lock] -> [skill button click]`)
across single-target, enemy AoE (`敵方全體`), and ally AoE (`我方全體`) skills, with graceful
S1 normal-attack fallback, UIPI privilege checking, and `GA_ROOT` foreground gating.
"""

import logging
import os
import random
import time
from collections.abc import Callable
from typing import Any

import win32gui

from onmyoji_helper.automation.coordinate_mapper import CoordinateMapper, MappedScreenPoint
from onmyoji_helper.automation.safety_guard import SafetyGuard
from onmyoji_helper.domain.models import UNKNOWN_SHIKIGAMI, CropOffset
from onmyoji_helper.infra.win32_input import perform_humanized_click
from onmyoji_helper.infra.win32_privilege import is_uipi_blocked_for_hwnd
from onmyoji_helper.infra.win32_window import (
    focus_game_root_window,
    get_root_hwnd,
    get_window_pid,
    get_window_title,
    matches_game_foreground,
)

logger = logging.getLogger("OnmyojiHelper")


class AutoBattleExecutor:
    """Executes humanized target-lock and skill-cast mouse actions in Red Mode."""

    def __init__(
        self,
        min_turn_cooldown: float = 2.2,
        extra_turn_escape_sec: float = 5.5,
        target_jitter_px: float = 6.0,
        skill_jitter_px: float = 4.0,
        reaction_delay_range: tuple[float, float] = (0.25, 0.50),
        inter_click_delay_range: tuple[float, float] = (0.15, 0.28),
        mouse_takeover_px: float = 95.0,
        input_backend: Callable[[str, int, int], bool] | None = None,
        foreground_checker: Callable[[int], bool] | None = None,
        client_to_screen_fn: Callable[[int, int, int, int, int], MappedScreenPoint | None] | None = None,
        key_checker: Callable[[int], bool] | None = None,
        uipi_checker: Callable[[int], bool] | None = None,
    ) -> None:
        """Initializes the auto-battle executor, safety guard, and coordinate mapper."""
        self.target_jitter_px: float = float(target_jitter_px)
        self.skill_jitter_px: float = float(skill_jitter_px)
        self.reaction_delay_range: tuple[float, float] = reaction_delay_range
        self.inter_click_delay_range: tuple[float, float] = inter_click_delay_range
        self.mouse_takeover_px: float = float(mouse_takeover_px)

        self._input_backend = input_backend
        self._foreground_checker = foreground_checker
        self._uipi_checker = uipi_checker

        self.mapper: CoordinateMapper = CoordinateMapper(custom_mapper=client_to_screen_fn)
        self.safety: SafetyGuard = SafetyGuard(
            min_turn_cooldown=min_turn_cooldown,
            extra_turn_escape_sec=extra_turn_escape_sec,
            key_checker=key_checker,
        )

        self._logged_non_fg_key: tuple[int, int] | None = None
        self._uipi_cache: dict[int, bool] = {}
        self._logged_uipi_pid: int | None = None

    @property
    def emergency_stopped(self) -> bool:
        return self.safety.emergency_stopped

    def reset_for_match(self) -> None:
        self.safety.reset_for_match()

    def stop_immediately(self) -> None:
        self.safety.stop_immediately()

    def start_kill_switch_monitor(self, on_emergency_stop: Callable[[str], None] | None = None) -> None:
        self.safety.start_kill_switch_monitor(on_emergency_stop=on_emergency_stop)

    def stop_kill_switch_monitor(self) -> None:
        self.safety.stop_kill_switch_monitor()

    def trigger_emergency_stop(self, reason: str = "按下緊急煞車熱鍵") -> None:
        self.safety.trigger_emergency_stop(reason=reason)

    def poll_kill_switch_once(self) -> bool:
        return self.safety.poll_kill_switch_once()

    def observe_turn_state(self, is_my_turn: bool) -> None:
        self.safety.observe_turn_state(is_my_turn)

    def can_execute_this_turn(self, now: float | None = None) -> bool:
        return self.safety.can_execute_this_turn(now=now)

    def check_uipi_privilege_block(self, hwnd: int) -> bool:
        """Checks (with per-PID caching) if Windows UIPI blocks SendInput to an elevated game."""
        if self._uipi_checker is not None:
            return bool(self._uipi_checker(hwnd))
        game_pid = get_window_pid(hwnd)
        if game_pid <= 0:
            return False
        if game_pid not in self._uipi_cache:
            self._uipi_cache[game_pid] = is_uipi_blocked_for_hwnd(hwnd)
        blocked = self._uipi_cache[game_pid]
        if blocked and self._logged_uipi_pid != game_pid:
            self._logged_uipi_pid = game_pid
            logger.warning(
                f"⚠️ [AutoExecutor] UIPI 權限隔離偵測：遊戲進程 (pid={game_pid}) 以系統管理員運行，"
                "但目前助手為一般權限，SendInput 將被 Windows 攔截！請以系統管理員身分啟動助手。"
            )
        return blocked

    def is_game_window_foreground(self, hwnd: int, allow_reclaim_from_self: bool = False) -> bool:
        """Verifies game visibility & foreground state (auto-returning focus from self once)."""
        if self._foreground_checker is not None:
            return bool(self._foreground_checker(hwnd))
        try:
            if not hwnd or not win32gui.IsWindow(hwnd):
                return False
            game_root = get_root_hwnd(hwnd)
            if (
                win32gui.IsIconic(hwnd)
                or win32gui.IsIconic(game_root)
                or not win32gui.IsWindowVisible(game_root)
            ):
                return False

            game_pid = get_window_pid(game_root) or get_window_pid(hwnd)
            fg_hwnd = win32gui.GetForegroundWindow()

            if matches_game_foreground(fg_hwnd, hwnd, game_root, game_pid):
                self._logged_non_fg_key = None
                return True

            fg_pid = get_window_pid(fg_hwnd) if fg_hwnd else 0

            if allow_reclaim_from_self and fg_pid == os.getpid():
                logger.info(
                    f"🔄 [AutoExecutor] 前景為助手自身視窗 (pid={fg_pid})，自動將焦點交還給遊戲視窗 (hwnd={game_root})..."
                )
                if focus_game_root_window(game_root):
                    time.sleep(0.08)
                    fg_after = win32gui.GetForegroundWindow()
                    if matches_game_foreground(fg_after, hwnd, game_root, game_pid):
                        self._logged_non_fg_key = None
                        return True
                    fg_hwnd = fg_after
                    fg_pid = get_window_pid(fg_after) if fg_after else 0

            non_fg_key = (int(fg_hwnd or 0), int(fg_pid))
            if non_fg_key != self._logged_non_fg_key:
                self._logged_non_fg_key = non_fg_key
                fg_title = get_window_title(fg_hwnd) if fg_hwnd else "(無)"
                game_title = get_window_title(game_root) or get_window_title(hwnd)
                logger.info(
                    f"⏸️ [AutoExecutor] 遊戲非前景暫停 | "
                    f"前景='{fg_title}' (hwnd={fg_hwnd}, pid={fg_pid}) | "
                    f"遊戲='{game_title}' (hwnd={hwnd}, root={game_root}, pid={game_pid})"
                )
            return False
        except Exception:
            return False

    def map_image_to_screen(
        self,
        hwnd: int,
        img_x: float,
        img_y: float,
        img_w: int,
        img_h: int,
        crop: CropOffset | None = None,
    ) -> MappedScreenPoint | None:
        """Delegates coordinate transformation (with optional black-border crop offset)."""
        return self.mapper.map_image_to_screen(hwnd, img_x, img_y, img_w, img_h, crop=crop)

    def _interruptible_sleep(self, duration: float, hwnd: int) -> bool:
        """Sleeps in short slices while checking kill-switch, stop event, and foreground guard."""
        if duration <= 0.0:
            return (
                not self.poll_kill_switch_once()
                and not self.safety.stop_event.is_set()
                and self.is_game_window_foreground(hwnd)
            )
        end_ts = time.time() + duration
        while time.time() < end_ts:
            if self.safety.stop_event.is_set() or self.poll_kill_switch_once():
                return False
            if not self.is_game_window_foreground(hwnd):
                return False
            time.sleep(min(0.03, max(0.005, end_ts - time.time())))
        return True

    def _humanized_click(self, hwnd: int, target_x: int, target_y: int) -> bool:
        """Dispatches a humanized click via injectable test backend or Win32 SendInput."""
        if self.safety.stop_event.is_set() or self.poll_kill_switch_once():
            return False
        if not self.is_game_window_foreground(hwnd):
            return False

        if self._input_backend is not None:
            return bool(self._input_backend("click", target_x, target_y))

        return perform_humanized_click(
            target_x=target_x,
            target_y=target_y,
            mouse_takeover_px=self.mouse_takeover_px,
            abort_check_fn=lambda: (
                self.safety.stop_event.is_set()
                or self.poll_kill_switch_once()
                or not self.is_game_window_foreground(hwnd)
            ),
            on_takeover_fn=lambda: self.trigger_emergency_stop("偵測到手動滑鼠接管"),
        )

    @staticmethod
    def _pick_alive_slot(
        pool: dict[str, dict[str, Any]],
        preferred_slot: str | None = None,
    ) -> tuple[str | None, str, tuple[int, int]]:
        """Selects an alive unit with valid non-zero coordinates (`preferred_slot` -> `Slot 1..5`)."""
        if not pool:
            return None, "", (0, 0)

        ordered_keys = []
        if preferred_slot and preferred_slot in pool:
            ordered_keys.append(preferred_slot)
        for canonical in ("Slot 1", "Slot 2", "Slot 3", "Slot 4", "Slot 5"):
            if canonical in pool and canonical not in ordered_keys:
                ordered_keys.append(canonical)
        for extra_k in pool:
            if extra_k not in ordered_keys:
                ordered_keys.append(extra_k)

        for slot_k in ordered_keys:
            info = pool.get(slot_k, {})
            pos = info.get("pos", (0, 0))
            if info.get("alive", False) and pos and pos != (0, 0):
                return slot_k, str(info.get("name", slot_k)), (int(pos[0]), int(pos[1]))
        return None, "", (0, 0)

    @staticmethod
    def resolve_effective_skill(
        chosen_skill: str,
        available_skills: dict[str, bool],
        skill_mapping: dict[str, str | None],
    ) -> tuple[str, bool]:
        """Validates chosen skill and gracefully downgrades unavailable S2/S3/S4 to S1."""
        desired = str(chosen_skill or "S1").strip().lower()
        if desired not in ("s1", "s2", "s3", "s4"):
            desired = "s1"

        if skill_mapping.get(desired) is not None and bool(available_skills.get(desired, False)):
            return desired.upper(), False

        if desired != "s1" and skill_mapping.get("s1") is not None:
            return "S1", True

        for candidate in ("s1", "s2", "s3", "s4"):
            if skill_mapping.get(candidate) is not None and bool(available_skills.get(candidate, False)):
                return candidate.upper(), (candidate != desired)

        return "S1", (desired != "s1")

    def execute_turn_action(
        self,
        hwnd: int,
        img_w: int,
        img_h: int,
        active_shiki_name: str,
        chosen_skill: str,
        scope: str,
        target_slot: str | None,
        target_name: str,
        target_is_enemy: bool,
        available_skills: dict[str, bool],
        skill_mapping: dict[str, str | None],
        physical_slots: dict[str, tuple[float, float, float, float]],
        my_status: dict[str, dict[str, Any]],
        opp_status: dict[str, dict[str, Any]],
        crop: CropOffset | None = None,
    ) -> tuple[bool, str]:
        """Executes one complete humanized Onmyoji turn: `[unit lock] -> [skill button]`."""
        if not self.can_execute_this_turn():
            return False, ""

        if self.check_uipi_privilege_block(hwnd):
            return False, "⚠️ 遊戲以管理員運行，請以系統管理員身分啟動助手"

        if not self.is_game_window_foreground(hwnd, allow_reclaim_from_self=True):
            return False, "⏸️ [紅色全自動] 遊戲視窗非前景，暫停自動點擊"

        effective_skill, downgraded = self.resolve_effective_skill(
            chosen_skill=chosen_skill,
            available_skills=available_skills,
            skill_mapping=skill_mapping,
        )
        phys_slot = skill_mapping.get(effective_skill.lower())
        if not phys_slot or phys_slot not in physical_slots:
            return False, f"⚠️ [紅色全自動] 找不到技能 {effective_skill} 對應按鈕位置"

        effective_scope = str(scope or "").strip()
        effective_target_is_enemy = bool(target_is_enemy)
        effective_target_slot = target_slot
        effective_target_name = target_name

        if downgraded and effective_skill == "S1":
            effective_scope = "敵方單體"
            effective_target_is_enemy = True
            if not effective_target_slot or not target_is_enemy:
                picked_slot, picked_name, _ = self._pick_alive_slot(opp_status)
                effective_target_slot = picked_slot
                effective_target_name = picked_name

        is_aoe_skill = ("全體" in effective_scope) or ("群" in effective_scope)
        if is_aoe_skill:
            if "我方" in effective_scope or "友方" in effective_scope:
                effective_target_is_enemy = False
            elif "敵方" in effective_scope:
                effective_target_is_enemy = True

        rx1, ry1, rx2, ry2 = physical_slots[phys_slot]
        skill_img_x = ((rx1 + rx2) * 0.5) * img_w
        skill_img_y = ((ry1 + ry2) * 0.5) * img_h
        skill_mapped = self.map_image_to_screen(hwnd, skill_img_x, skill_img_y, img_w, img_h, crop=crop)
        if skill_mapped is None:
            return False, "⚠️ [紅色全自動] 技能按鈕座標轉換失敗"
        skill_sx, skill_sy, win_bounds = skill_mapped

        # Both single-target AND AoE (敵方全體 / 我方全體) perform 2-step [click alive unit -> click skill]
        needs_target_click = (effective_scope != "自己") and (is_aoe_skill or effective_target_slot is not None)

        target_mapped: MappedScreenPoint | None = None
        if needs_target_click:
            pool = opp_status if effective_target_is_enemy else my_status
            picked_slot, picked_name, pos = self._pick_alive_slot(pool, preferred_slot=effective_target_slot)
            if picked_slot is not None and pos != (0, 0):
                effective_target_slot = picked_slot
                if not is_aoe_skill or not effective_target_name or "全體" in effective_target_name:
                    effective_target_name = picked_name
                target_mapped = self.map_image_to_screen(hwnd, pos[0], pos[1], img_w, img_h, crop=crop)

        reaction_sec = random.uniform(*self.reaction_delay_range)
        if not self._interruptible_sleep(reaction_sec, hwnd):
            return False, ""

        if needs_target_click and target_mapped is not None:
            tx, ty, t_bounds = target_mapped
            jtx, jty = CoordinateMapper.apply_gaussian_jitter(tx, ty, self.target_jitter_px, t_bounds)
            if not self._humanized_click(hwnd, jtx, jty):
                return False, ""

            inter_sec = random.uniform(*self.inter_click_delay_range)
            if not self._interruptible_sleep(inter_sec, hwnd):
                return False, ""

        jsx, jsy = CoordinateMapper.apply_gaussian_jitter(skill_sx, skill_sy, self.skill_jitter_px, win_bounds)
        if not self._humanized_click(hwnd, jsx, jsy):
            return False, ""

        self.safety.mark_action_executed()

        actor_label = (
            active_shiki_name
            if (active_shiki_name and active_shiki_name != UNKNOWN_SHIKIGAMI)
            else "我方式神"
        )
        downgrade_tag = " (缺火/冷卻降級S1)" if downgraded else ""
        side_label = "敵方" if effective_target_is_enemy else "我方"

        if is_aoe_skill:
            aoe_label = "敵方全體" if effective_target_is_enemy else "我方全體"
            if effective_target_slot:
                action_msg = (
                    f"🤖 [紅色全自動] 已執行：{actor_label} {effective_skill}{downgrade_tag} "
                    f"({aoe_label}) → {side_label} {effective_target_slot}"
                )
            else:
                action_msg = f"🤖 [紅色全自動] 已執行：{actor_label} {effective_skill}{downgrade_tag} → {aoe_label}"
        elif needs_target_click and effective_target_slot:
            action_msg = (
                f"🤖 [紅色全自動] 已執行：{actor_label} {effective_skill}{downgrade_tag}"
                f" → {side_label} {effective_target_slot} ({effective_target_name})"
            )
        else:
            scope_label = effective_target_name or effective_scope or "自身"
            action_msg = f"🤖 [紅色全自動] 已執行：{actor_label} {effective_skill}{downgrade_tag} → {scope_label}"

        logger.info(action_msg)
        return True, action_msg
