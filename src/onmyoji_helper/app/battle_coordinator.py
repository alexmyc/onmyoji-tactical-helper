"""5v5 Real-Time Battle Coordinator for Onmyoji Tactical Helper.

Orchestrates screen capture, YOLOv8 spatial tracking, turn/skill geometric detection,
draft-avatar vs. action-bar template matching (`ActionBarDetector`), asynchronous Gemma-4
reasoning, click-through HUD presentation, and Red Mode skill-first auto-execution under
RLock-protected thread-safe session state.
"""

import ctypes
import logging
import random
import threading
import time
import tkinter as tk
from typing import Any

import numpy as np
import win32gui

from onmyoji_helper.automation.battle_executor import AutoBattleExecutor
from onmyoji_helper.decision.battle_context import BattleContextBuilder
from onmyoji_helper.domain.models import UNKNOWN_SHIKIGAMI, OperatingMode
from onmyoji_helper.domain.shikigami_repository import ShikigamiRepository
from onmyoji_helper.infra.capture_engine import CaptureEngine
from onmyoji_helper.infra.model_lifecycle import ModelLifecycleManager
from onmyoji_helper.ui.battle_presenter import BattlePresenter
from onmyoji_helper.vision.action_bar_detector import ActionBarDetector
from onmyoji_helper.vision.spatial_tracker import StateParser
from onmyoji_helper.vision.turn_detector import TurnDetector

logger = logging.getLogger("OnmyojiHelper")


class AITacticalManager:
    """Thread-safe coordinator binding CV detectors, LLM/rule engines, HUD UI, and auto-executor."""

    POLL_MY_TURN: float = 0.20
    POLL_OPP_TURN: float = 0.35
    POLL_ENTRY_SETTLE: float = 0.20
    POST_DECISION_BUFFER_RANGE: tuple[float, float] = (1.0, 1.5)
    DRAFT_FADE_GRACE_SEC: float = 2.5
    PROVISIONAL_OPENING_DELAY_SEC: float = 3.5
    STEALTH_FALLBACK_DELAY_SEC: float = 5.5
    MY_TURN_MEAN_DRIFT_RATIO: float = 0.025
    MY_TURN_MAX_DRIFT_RATIO: float = 0.045

    def __init__(self, repository: ShikigamiRepository | None = None) -> None:
        self.models: ModelLifecycleManager = ModelLifecycleManager()
        self.repository: ShikigamiRepository | None = repository
        self.context_builder: BattleContextBuilder | None = None
        self.parser: StateParser = StateParser()
        self.turn_detector: TurnDetector = TurnDetector()
        self.action_bar_detector: ActionBarDetector = ActionBarDetector()
        self.presenter: BattlePresenter = BattlePresenter()
        self.auto_executor: AutoBattleExecutor = AutoBattleExecutor()
        self.mode_controller: Any = None

        self.session_lock: threading.RLock = threading.RLock()
        self._state_lock: threading.RLock = threading.RLock()
        self.model_ready_event: threading.Event = threading.Event()
        self._wake_event: threading.Event = threading.Event()
        self.battle_thread: threading.Thread | None = None
        self.root_window: tk.Tk | None = None

        self.user_closed: bool = False
        self.stop_requested: bool = False
        self.is_active_session: bool = False
        self.battle_has_started: bool = False
        self.show_overlay: bool = False
        self.auto_battle_enabled: bool = False

        self.no_box_start_time: float = 0.0
        self.session_start_time: float = 0.0
        self.has_calibrated_on_my_turn: bool = False
        self.is_provisional_baseline: bool = False
        self._frame_seq: int = 0

        self.is_ai_reasoning: bool = False
        self.latest_advice_text: str = ""
        self.latest_auto_action_text: str = ""
        self.last_llm_time: float = 0.0
        self.my_draft_live: list[str] = []
        self.opp_draft_live: list[str] = []
        self.my_slot_map: dict[str, str] = {}
        self.last_turn_key: tuple[bool, str | None] | None = None

        self._decision_ready_ts: float = 0.0
        self._turn_settle_delay: float = 1.2
        self._llm_override_slot: str | None = None
        self._llm_override_skill: str | None = None

    def set_repository(self, repository: ShikigamiRepository) -> None:
        """Injects the shared `ShikigamiRepository` initialized at startup."""
        self.repository = repository
        if self.context_builder is not None:
            self.context_builder.repo = repository

    def set_mode_controller(self, mode_controller: Any) -> None:
        self.mode_controller = mode_controller

    def _is_cancelled(self) -> bool:
        return self.user_closed or self.stop_requested or not self.is_active_session

    def wake_loop(self) -> None:
        """Immediately wakes `_battle_loop` from poll sleep for zero-lag mode switches."""
        self._wake_event.set()

    def hide_battle_windows(self) -> None:
        """Hides and destroys floating tactical/HUD windows when outside an active battle."""
        self.presenter.destroy_all(self.root_window)

    def capture_draft_templates(
        self,
        draft_frame: np.ndarray | None,
        my_draft: list[str],
    ) -> bool:
        """Captures and locks the 5 ally portrait templates on the 5-pick draft screen."""
        if draft_frame is None or len(my_draft) != 5:
            return False
        with self.models.yolo_lock:
            self.action_bar_detector.reset()
            ok = self.action_bar_detector.capture_prep_templates(
                draft_frame,
                our_picks=my_draft,
                lock_on_success=True,
            )
            if ok:
                logger.info(
                    f"📸 [ActionBar] 已於 5 手翻牌階段擷取並鎖定我方頭像模板: {self.action_bar_detector.slot_to_shiki}"
                )
            return ok

    def update_live_draft(self, my_draft: list[str], opp_draft: list[str]) -> None:
        """Updates 5v5 draft names in-place under `_state_lock`."""
        with self._state_lock:
            self.my_draft_live = list(my_draft)
            self.opp_draft_live = list(opp_draft)
            self.my_slot_map = {f"Slot {i + 1}": name for i, name in enumerate(my_draft)}
        with self.models.yolo_lock:
            self.action_bar_detector.bind_picks(my_draft)
        self.wake_loop()

    def enable_mode(self, mode_name: str, has_locked_draft: bool = True) -> None:
        """Configures HUD overlay and Red Mode auto-executor flags on mode switch."""
        target = mode_name.value if isinstance(mode_name, OperatingMode) else str(mode_name)
        with self.session_lock, self._state_lock:
            self.user_closed = False
            self.stop_requested = False
            self.show_overlay = target in (OperatingMode.YELLOW.value, OperatingMode.RED.value)
            self.auto_battle_enabled = target == OperatingMode.RED.value
            self.latest_auto_action_text = ""
            self._decision_ready_ts = 0.0
            active_combat = self.is_active_session and self.battle_has_started

        if target in (OperatingMode.YELLOW.value, OperatingMode.RED.value):
            self.models.warmup_in_background()

        if target == OperatingMode.RED.value:
            self.auto_executor.reset_for_match()
            if has_locked_draft or active_combat:
                self.auto_executor.start_kill_switch_monitor(on_emergency_stop=self._handle_emergency_stop)
                try:
                    wins = CaptureEngine().find_windows()
                    if wins:
                        game_hwnd = int(wins[0]._hWnd)
                        if self.auto_executor.check_uipi_privilege_block(game_hwnd):
                            uipi_warn = "⚠️ 遊戲以管理員運行，請以系統管理員身分啟動助手"
                            with self._state_lock:
                                self.latest_auto_action_text = uipi_warn
                            if self.mode_controller and hasattr(self.mode_controller, "update_status_text"):
                                self.mode_controller.update_status_text(uipi_warn)
                        else:
                            self.auto_executor.is_game_window_foreground(
                                game_hwnd, allow_reclaim_from_self=True
                            )
                except Exception as exc:
                    logger.debug(f"[BattleCoordinator] Initial Red Mode window check skipped: {exc}")
        else:
            self.auto_executor.stop_immediately()

        self.wake_loop()

    def disable_auto_battle(self, reason: str = "") -> None:
        """Immediately stops Red Mode clicking while keeping Yellow Mode HUD and advice active."""
        self.auto_battle_enabled = False
        self.auto_executor.stop_immediately()
        msg = (
            f"⚠️ 已手動緊急煞車，切回黃色模式 ({reason})"
            if reason
            else "⚠️ 已手動緊急煞車，切回黃色模式"
        )
        with self._state_lock:
            self.latest_auto_action_text = msg
            adv = self.latest_advice_text
            display_txt = f"{msg}\n\n{adv}" if adv else msg

        self.presenter.schedule_advice_text(self.root_window, display_txt, self._is_cancelled)

    def _handle_emergency_stop(self, reason: str) -> None:
        self.disable_auto_battle(reason)
        if self.mode_controller and hasattr(self.mode_controller, "emergency_downgrade_to_yellow"):
            self.mode_controller.emergency_downgrade_to_yellow(reason)

    def close_ui_sync(self, unload_models: bool = False) -> None:
        """Synchronously halts active battle threads, resets tracker baselines, and destroys floating UIs (preserving locked match templates)."""
        self.auto_battle_enabled = False
        self.auto_executor.stop_immediately()
        self.wake_loop()

        with self.session_lock:
            self.user_closed = True
            self.stop_requested = True
            self.is_active_session = False
            self.battle_has_started = False
            self._reset_entry_gate_state()
            self.presenter.destroy_all(self.root_window)

        if self.battle_thread and self.battle_thread.is_alive():
            if self.battle_thread != threading.current_thread():
                self.battle_thread.join(timeout=1.5)

        with self.models.yolo_lock:
            self.parser.reset_baseline()
            self.turn_detector.reset_cache()

        if unload_models:
            retry = 0
            while retry < 20:
                with self._state_lock:
                    if not self.is_ai_reasoning:
                        break
                time.sleep(0.1)
                retry += 1
            self.models.unload_all()
        else:
            self.models.purge_cuda_cache()

    def close_ui(self) -> None:
        threading.Thread(target=self.close_ui_sync, daemon=True).start()

    def on_ui_closed_by_user(self) -> None:
        if self.mode_controller and hasattr(self.mode_controller, "set_mode"):
            if not self.mode_controller.set_mode(OperatingMode.GREEN.value):
                self.close_ui()
        else:
            self.close_ui()

    def _reset_entry_gate_state(self) -> None:
        """Clears turn-calibration and provisional baseline flags across session/mode resets."""
        self.has_calibrated_on_my_turn = False
        self.is_provisional_baseline = False

    @staticmethod
    def _extract_valid_combat_points(
        boxes_data: list[dict[str, Any]],
        screen_w: int,
        screen_h: int,
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """Extracts valid enemy and ally center points without mutating `StateParser` baselines."""
        opp_pts = sorted(
            [
                (float(b["cx"]), float(b["cy"]))
                for b in boxes_data
                if b.get("cls") == 0 and (screen_w * 0.02) <= float(b.get("cx", 0.0)) <= (screen_w * 0.98)
            ],
            key=lambda p: p[0],
        )
        my_pts = sorted(
            [
                (float(b["cx"]), float(b["cy"]))
                for b in boxes_data
                if b.get("cls") == 1
                and float(b.get("cy", 0.0)) > (screen_h * 0.40)
                and float(b.get("cx", 0.0)) < (screen_w * 0.87)
            ],
            key=lambda p: p[0],
        )
        return opp_pts, my_pts

    def _should_open_provisional_before_my_turn(
        self,
        raw_opp: list[tuple[float, float]],
        raw_my: list[tuple[float, float]],
        now: float,
        from_five_picks_lock: bool,
    ) -> bool:
        """Determines whether enemy-first provisional tracking may start while awaiting our first turn.

        Never triggers on short motion pauses during opening pre-emptive ('先機') animations:
        waits past `PROVISIONAL_OPENING_DELAY_SEC` (`3.5s`) after 5-pick lock with enemy units on screen
        (or `STEALTH_FALLBACK_DELAY_SEC` if only ally units are visible), and still forces a full
        `parser.reset_baseline()` on the very first `is_my_turn == True` skill-bar frame.
        """
        elapsed = max(0.0, now - self.session_start_time)
        required_delay = self.PROVISIONAL_OPENING_DELAY_SEC if from_five_picks_lock else 0.0
        if len(raw_opp) >= 1 and elapsed >= required_delay:
            return True
        stealth_delay = self.STEALTH_FALLBACK_DELAY_SEC if from_five_picks_lock else 1.5
        return len(raw_my) >= 1 and elapsed >= stealth_delay

    def _recalibrate_opp_if_drifted_on_my_turn(
        self,
        raw_opp: list[tuple[float, float]],
        screen_w: int,
    ) -> bool:
        """Self-heals enemy baseline on `is_my_turn` when all 5 distinct enemies are visible and drifted."""
        if screen_w <= 0 or len(raw_opp) != 5:
            return False
        if not all((raw_opp[i + 1][0] - raw_opp[i][0]) >= (screen_w * 0.03) for i in range(4)):
            return False

        drifts: list[float] = []
        missing_slot = False
        for idx, (slot_name, _) in enumerate(self.parser.OPP_SLOT_ANCHORS):
            tracked = self.parser.smooth_pos_opp.get(slot_name) or self.parser.baseline_opp.get(slot_name)
            if tracked is None:
                missing_slot = True
                break
            rx, ry = raw_opp[idx]
            drifts.append(float(np.hypot(rx - tracked[0], ry - tracked[1])))

        should_refresh = missing_slot or (
            bool(drifts)
            and (
                (sum(drifts) / float(len(drifts))) > (screen_w * self.MY_TURN_MEAN_DRIFT_RATIO)
                or max(drifts) > (screen_w * self.MY_TURN_MAX_DRIFT_RATIO)
            )
        )
        if should_refresh:
            with self.models.yolo_lock:
                self.parser.baseline_opp.clear()
                self.parser.smooth_pos_opp.clear()
                self.parser.last_seen_opp.clear()
                self.parser.is_opp_baseline_set = False
            logger.info("🔄 [BattleLoop] 我方回合 5 敵基準自檢發現偏移，已自動刷新敵方標準基準。")
            return True
        return False

    def reset_to_blank(self, root_window: tk.Tk | None) -> None:
        """Resets battle session state and closes HUD & advice panels on match end."""
        self.stop_requested = True
        self.battle_has_started = False
        self.no_box_start_time = 0.0
        self._reset_entry_gate_state()
        self.auto_executor.stop_immediately()
        self.wake_loop()

        with self._state_lock:
            self.latest_advice_text = ""
            self.latest_auto_action_text = ""
            self.last_llm_time = 0.0
            self.my_slot_map.clear()
            self.last_turn_key = None
            self._decision_ready_ts = 0.0
            self._llm_override_slot = None
            self._llm_override_skill = None

        with self.models.yolo_lock:
            self.action_bar_detector.reset()
            self.parser.reset_baseline()
            self.turn_detector.reset_cache()

        self.is_active_session = False
        self.presenter.destroy_all(root_window or self.root_window)
        self.models.purge_cuda_cache()

    def reset_for_next_match(self) -> None:
        """Hides floating windows and clears templates/baselines cleanly for a fresh match."""
        with self.session_lock, self._state_lock:
            self.stop_requested = True
            self.is_active_session = False
            self.battle_has_started = False
            self._reset_entry_gate_state()
            self.no_box_start_time = 0.0
            self.latest_advice_text = ""
            self.latest_auto_action_text = ""
            self.my_slot_map.clear()
            self.last_turn_key = None
            self._decision_ready_ts = 0.0
            self._llm_override_slot = None
            self._llm_override_skill = None
            self.user_closed = False
        self.auto_executor.stop_immediately()
        self.presenter.destroy_all(self.root_window)
        self.wake_loop()

        if self.battle_thread and self.battle_thread.is_alive():
            if self.battle_thread != threading.current_thread():
                self.battle_thread.join(timeout=1.0)

        with self.models.yolo_lock:
            self.action_bar_detector.reset()
            self.parser.reset_baseline()
            self.turn_detector.reset_cache()

    def _async_llm_reasoning(self, payload: dict[str, Any], turn_key: tuple[bool, str | None]) -> None:
        """Runs Gemma-4 / rule reasoning asynchronously and updates chosen skill + target slot."""
        try:
            if self._is_cancelled():
                return
            advice_dict = self.models.run_llm_advice(payload)
            if advice_dict is None or self._is_cancelled():
                return

            advice_msg = str(advice_dict.get("advice", ""))
            mode_desc = str(advice_dict.get("mode", ""))
            mode_tag = f" [{mode_desc}]" if mode_desc else ""
            llm_slot = advice_dict.get("target_slot")
            llm_skill = advice_dict.get("chosen_skill")

            with self._state_lock:
                slot_map_snapshot = dict(self.my_slot_map)

            for slot_key, shiki_name in slot_map_snapshot.items():
                if shiki_name:
                    target_str = f"我方 [{slot_key}]"
                    replacement_str = f"我方 [{slot_key}] {shiki_name}"
                    if target_str in advice_msg and replacement_str not in advice_msg:
                        advice_msg = advice_msg.replace(target_str, replacement_str)

            display_slot = llm_slot or payload.get("recommended_target_slot")
            target_name = str(payload.get("recommended_target_name", "目標單位"))

            if display_slot and target_name and "全體" not in target_name:
                final_text = f"🎯 建議優先處理：[{display_slot}] {target_name}\n\n💡 戰術對策{mode_tag}：\n{advice_msg}"
            else:
                final_text = f"💡 戰術對策{mode_tag}：\n{advice_msg}"

            with self._state_lock:
                if not self._is_cancelled() and self.last_turn_key == turn_key:
                    self.latest_advice_text = final_text
                    if isinstance(llm_slot, str) and llm_slot.startswith("Slot "):
                        self._llm_override_slot = llm_slot
                    if isinstance(llm_skill, str) and llm_skill.upper() in ("S1", "S2", "S3", "S4"):
                        self._llm_override_skill = llm_skill.upper()
                    if turn_key[0] and self._decision_ready_ts == 0.0:
                        self._decision_ready_ts = time.time()
            self.wake_loop()
        except Exception as exc:
            logger.debug(f"⚠️ [BattleCoordinator] Async LLM thread error: {exc}")
        finally:
            with self._state_lock:
                self.is_ai_reasoning = False
                if turn_key[0] and self._decision_ready_ts == 0.0:
                    self._decision_ready_ts = time.time()

    @staticmethod
    def _resolve_battle_hwnd(clicker: CaptureEngine, preferred_hwnd: int | None) -> int | None:
        if preferred_hwnd:
            try:
                if win32gui.IsWindow(preferred_hwnd) and win32gui.IsWindowVisible(preferred_hwnd):
                    return int(preferred_hwnd)
            except Exception as exc:
                logger.debug(f"[BattleCoordinator] Preferred HWND check skipped: {exc}")

        windows = clicker.find_windows()
        if not windows:
            return None

        try:
            fg_hwnd = win32gui.GetForegroundWindow()
            for w in windows:
                if w._hWnd == fg_hwnd:
                    return int(w._hWnd)
        except Exception as exc:
            logger.debug(f"[BattleCoordinator] Foreground HWND lookup skipped: {exc}")
        return int(windows[0]._hWnd)

    def _battle_loop(
        self,
        root_window: tk.Tk,
        my_draft: list[str],
        opp_draft: list[str],
        preferred_hwnd: int | None = None,
        from_five_picks_lock: bool = False,
    ) -> None:
        clicker = CaptureEngine()
        with self._state_lock:
            self.my_draft_live = list(my_draft)
            self.opp_draft_live = list(opp_draft)
            self.my_slot_map = {f"Slot {i + 1}": name for i, name in enumerate(my_draft)}
            self.last_turn_key = None
            self._decision_ready_ts = 0.0
            self._llm_override_slot = None
            self._llm_override_skill = None

        try:
            self.model_ready_event.clear()
            if not self.models.ensure_loaded(should_abort_fn=lambda: self.stop_requested or self.user_closed):
                self.is_active_session = False
                return

            if self.context_builder is None:
                self.context_builder = BattleContextBuilder(repo=self.repository)

            self.model_ready_event.set()
            windows_ensured = False
            logged_battle_entry = False

            while not self._is_cancelled():
                hwnd = self._resolve_battle_hwnd(clicker, preferred_hwnd)
                if not hwnd:
                    self._wake_event.wait(timeout=0.4)
                    self._wake_event.clear()
                    continue

                if ctypes.windll.user32.IsIconic(hwnd):
                    self._wake_event.wait(timeout=0.8)
                    self._wake_event.clear()
                    continue

                frame = clicker.capture_frame(hwnd, need_gray=False)
                if frame is None:
                    self._wake_event.wait(timeout=0.25)
                    self._wake_event.clear()
                    continue

                self._frame_seq += 1
                seq = self._frame_seq
                screen_color = frame.bgr
                screen_w, screen_h = frame.width, frame.height

                with self._state_lock:
                    cur_my_draft = list(self.my_draft_live)
                    cur_opp_draft = list(self.opp_draft_live)

                is_my_turn = self.turn_detector.is_my_turn(screen_color, frame_seq=seq)
                skill_states = self.turn_detector.detect_available_skills(screen_color, frame_seq=seq)

                if self._is_cancelled():
                    break

                current_time = time.time()

                # Fallback template capture ONLY right after 5-pick lock while still on the draft screen
                if (
                    from_five_picks_lock
                    and not self.battle_has_started
                    and not self.action_bar_detector.has_locked_templates
                ):
                    if (current_time - self.session_start_time) <= 2.2 and not is_my_turn:
                        if self.action_bar_detector.capture_prep_templates(
                            screen_color, our_picks=cur_my_draft, lock_on_success=True
                        ):
                            logger.info(
                                f"📸 [ActionBar] 已擷取並鎖定我方 5 隻式神頭像模板: {self.action_bar_detector.slot_to_shiki}"
                            )

                results = self.models.predict_yolo(screen_color)
                if results is None or self._is_cancelled():
                    break

                boxes_data = []
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    xyxy = box.xyxy[0].cpu().numpy()
                    boxes_data.append(
                        {
                            "cls": cls_id,
                            "cx": float((xyxy[0] + xyxy[2]) / 2.0),
                            "cy": float((xyxy[1] + xyxy[3]) / 2.0),
                        }
                    )

                current_time = time.time()

                # Brief fade-out guard ONLY right after 5th pick so Ban/Pick card exit doesn't flash early
                if (
                    from_five_picks_lock
                    and not self.battle_has_started
                    and not is_my_turn
                    and (current_time - self.session_start_time) < self.DRAFT_FADE_GRACE_SEC
                ):
                    self._wake_event.wait(timeout=self.POLL_ENTRY_SETTLE)
                    self._wake_event.clear()
                    continue

                raw_opp_pts, raw_my_pts = self._extract_valid_combat_points(boxes_data, screen_w, screen_h)

                # Deterministic Turn-UI Baseline Gate:
                # 1) Never lock on static pauses during VS / opening '先機' animations.
                # 2) Primary trigger: `is_my_turn == True` (skill bar rendered -> canonical combat camera!).
                # 3) Enemy-first fallback: provisional entry after opening buffer, followed by a MANDATORY
                #    `parser.reset_baseline()` hard recalibration on the very first `is_my_turn == True` frame!
                if not self.battle_has_started:
                    can_enter_provisional = (
                        not is_my_turn
                        and self._should_open_provisional_before_my_turn(
                            raw_opp_pts, raw_my_pts, current_time, from_five_picks_lock
                        )
                    )
                    if not is_my_turn and not can_enter_provisional:
                        if (current_time - self.session_start_time) >= 90.0:
                            self.reset_to_blank(root_window)
                            break
                        self._wake_event.wait(timeout=self.POLL_ENTRY_SETTLE)
                        self._wake_event.clear()
                        continue

                    with self.models.yolo_lock:
                        self.parser.reset_baseline()
                    self.battle_has_started = True
                    self.has_calibrated_on_my_turn = bool(is_my_turn)
                    self.is_provisional_baseline = not bool(is_my_turn)
                    self.no_box_start_time = 0.0
                    if self.is_provisional_baseline:
                        logger.info(
                            "⏳ [BattleLoop] 敵方先手行動中，啟用臨時追蹤基準（待首個我方回合技能欄就緒後強制重校準）。"
                        )
                    if self.auto_battle_enabled:
                        self.auto_executor.reset_for_match()
                        self.auto_executor.start_kill_switch_monitor(
                            on_emergency_stop=self._handle_emergency_stop
                        )
                        self.auto_executor.is_game_window_foreground(
                            hwnd, allow_reclaim_from_self=True
                        )
                elif is_my_turn and not self.has_calibrated_on_my_turn:
                    # Mandatory hard recalibration on the FIRST ally skill-bar turn after enemy-first provisional entry!
                    with self.models.yolo_lock:
                        self.parser.reset_baseline()
                    self.has_calibrated_on_my_turn = True
                    self.is_provisional_baseline = False
                    logger.info(
                        "🎯 [BattleLoop] 首個我方回合技能欄已出現，已強制執行 reset_baseline() 重新鎖定標準敵我基準！"
                    )
                elif is_my_turn and self.has_calibrated_on_my_turn:
                    self._recalibrate_opp_if_drifted_on_my_turn(raw_opp_pts, screen_w)

                my_status, opp_status = self.parser.parse_frame(
                    cur_my_draft, cur_opp_draft, boxes_data, screen_w, screen_height=screen_h
                )

                my_alive_count = sum(1 for info in my_status.values() if info.get("alive", False))
                opp_alive_count = sum(1 for info in opp_status.values() if info.get("alive", False))
                total_alive = my_alive_count + opp_alive_count
                current_time = time.time()

                need_ai_win = self.presenter.ai_ui is None or not self.presenter.ai_ui.winfo_exists()
                need_ov_win = self.show_overlay and (
                    self.presenter.overlay_ui is None or not self.presenter.overlay_ui.winfo_exists()
                )
                if not windows_ensured or need_ai_win or need_ov_win:
                    self.presenter.ensure_windows(
                        root_window=root_window,
                        show_overlay=self.show_overlay,
                        on_closed_by_user=self.on_ui_closed_by_user,
                        is_cancelled_fn=self._is_cancelled,
                    )
                    windows_ensured = True

                if not logged_battle_entry:
                    logged_battle_entry = True
                    logger.info(
                        f"⚔️ [BattleLoop] 已進入實戰追蹤 | hwnd={hwnd} | 畫面={screen_w}x{screen_h} "
                        f"| 我方存活={my_alive_count} 敵方存活={opp_alive_count} "
                        f"| 頭像模板={len(self.action_bar_detector.prep_templates)}/5 "
                        f"| overlay={self.show_overlay} auto_red={self.auto_battle_enabled}"
                    )

                active_slot: str | None = None
                active_shiki: str = UNKNOWN_SHIKIGAMI
                match_score: float = 0.0
                if is_my_turn:
                    res_slot, matched_name, match_score = self.action_bar_detector.identify_active_turn(
                        screen_color
                    )
                    if res_slot:
                        active_slot = res_slot
                        with self._state_lock:
                            active_shiki = self.my_slot_map.get(active_slot, matched_name)

                current_turn_key = (is_my_turn, active_slot)
                with self._state_lock:
                    if current_turn_key != self.last_turn_key:
                        self.last_turn_key = current_turn_key
                        self.latest_advice_text = ""
                        self.last_llm_time = 0.0
                        self._llm_override_slot = None
                        self._llm_override_skill = None
                        if is_my_turn:
                            if active_slot:
                                logger.info(
                                    f"🎯 [ActionBar] 行動條頭像比對成功：當前行動式神 = [{active_slot}] {active_shiki} "
                                    f"(相似度={match_score:.3f})"
                                )
                            else:
                                logger.info(
                                    f"⚠️ [ActionBar] 行動條頭像未匹配高信心式神 (最高相似度={match_score:.3f}, "
                                    f"已存模板={len(self.action_bar_detector.prep_templates)}/5)"
                                )
                            self._turn_settle_delay = random.uniform(*self.POST_DECISION_BUFFER_RANGE)
                            self._decision_ready_ts = (
                                0.0 if self.models.has_active_gpu_llm() else current_time
                            )
                        else:
                            self._decision_ready_ts = 0.0
                            self.latest_auto_action_text = ""

                    locked_llm_slot = self._llm_override_slot
                    locked_llm_skill = self._llm_override_skill

                payload = self.context_builder.build_payload(
                    my_status,
                    opp_status,
                    is_my_turn=is_my_turn,
                    skill_states=skill_states,
                    locked_target_slot=locked_llm_slot,
                    active_slot=active_slot,
                    active_shiki_name=active_shiki,
                )

                rec_slot = payload.get("recommended_target_slot")
                rec_name = str(payload.get("recommended_target_name", ""))
                target_is_enemy = bool(payload.get("target_is_enemy", True))
                chosen_skill = str(payload.get("chosen_skill", "S1"))
                skill_scope = str(payload.get("skill_scope", "敵方單體"))

                # Apply LLM-selected skill override (when valid & available) and align its target scope
                if is_my_turn and locked_llm_skill and bool(skill_states.get(locked_llm_skill.lower(), False)):
                    chosen_skill = locked_llm_skill
                    shiki_skills = payload.get("active_shikigami_skills") or {}
                    sk_info = shiki_skills.get(chosen_skill, {}) or shiki_skills.get(chosen_skill.lower(), {})
                    override_scope = str(sk_info.get("scope", "")).strip()
                    if override_scope:
                        skill_scope = override_scope
                        if "全體" in skill_scope or "群" in skill_scope:
                            is_ally_aoe = any(
                                kw in skill_scope for kw in ("我方", "友方", "群輔", "治療", "拉條", "回火")
                            )
                            target_is_enemy = ("敵方" in skill_scope) or (not is_ally_aoe)
                        elif skill_scope == "自己":
                            target_is_enemy = False
                            rec_slot = active_slot
                        elif "我方" in skill_scope:
                            target_is_enemy = False
                        else:
                            target_is_enemy = True

                self.auto_executor.observe_turn_state(is_my_turn)
                skill_mapping_snapshot = dict(self.turn_detector.get_skill_mapping(screen_color))

                if self.show_overlay:
                    self.presenter.schedule_overlay_frame(
                        root_window=root_window,
                        hwnd=hwnd,
                        my_status=my_status,
                        opp_status=opp_status,
                        total_alive=total_alive,
                        img_w=screen_w,
                        img_h=screen_h,
                        is_my_turn=is_my_turn,
                        target_slot=rec_slot,
                        target_is_enemy=target_is_enemy,
                        turn_detector=self.turn_detector,
                        skill_states=skill_states,
                        skill_mapping=skill_mapping_snapshot,
                        crop=frame.crop,
                        is_cancelled_fn=self._is_cancelled,
                    )

                if total_alive >= 1 or is_my_turn:
                    self.no_box_start_time = 0.0

                    fallback_msg = str(payload.get("fallback_advice", ""))
                    with self._state_lock:
                        if not self.latest_advice_text:
                            if rec_slot and rec_name and "全體" not in rec_name:
                                self.latest_advice_text = (
                                    f"🎯 建議優先處理：[{rec_slot}] {rec_name}\n\n💡 戰術對策：\n{fallback_msg}"
                                )
                            else:
                                self.latest_advice_text = f"💡 戰術對策：\n{fallback_msg}"

                    should_spawn_llm = False
                    with self._state_lock:
                        if not self.is_ai_reasoning and (
                            current_time - self.last_llm_time >= 3.0 or self.last_llm_time == 0.0
                        ):
                            self.is_ai_reasoning = True
                            self.last_llm_time = current_time
                            should_spawn_llm = True

                    if should_spawn_llm:
                        threading.Thread(
                            target=self._async_llm_reasoning,
                            args=(payload, current_turn_key),
                            name="AsyncGemmaWorker",
                            daemon=True,
                        ).start()

                    # Cap GPU LLM wait to 0.75s so slow prefill never delays turn decision indefinitely
                    with self._state_lock:
                        if (
                            is_my_turn
                            and self._decision_ready_ts == 0.0
                            and self.last_llm_time > 0.0
                            and (current_time - self.last_llm_time) >= 0.75
                        ):
                            self._decision_ready_ts = current_time
                        ready_ts = self._decision_ready_ts
                        settle_req = self._turn_settle_delay

                    if (
                        self.auto_battle_enabled
                        and is_my_turn
                        and not self._is_cancelled()
                        and not self.auto_executor.emergency_stopped
                    ):
                        attempts_done = self.auto_executor.attempts_in_current_turn
                        settle_elapsed = (current_time - ready_ts) if ready_ts > 0.0 else 0.0

                        if attempts_done == 0 and (ready_ts == 0.0 or settle_elapsed < settle_req):
                            rem = max(0.1, settle_req - settle_elapsed) if ready_ts > 0.0 else settle_req
                            wait_msg = (
                                f"⏳ [紅色全自動] 已決策 [{chosen_skill}] → {rec_name}，"
                                f"等待鏡頭與技能欄穩定 ({rem:.1f}s)..."
                            )
                            with self._state_lock:
                                self.latest_auto_action_text = wait_msg
                        elif self.auto_executor.can_execute_this_turn(now=current_time):
                            try:
                                _, action_msg = self.auto_executor.execute_turn_action(
                                    hwnd=hwnd,
                                    img_w=screen_w,
                                    img_h=screen_h,
                                    active_shiki_name=active_shiki,
                                    chosen_skill=chosen_skill,
                                    scope=skill_scope,
                                    target_slot=rec_slot,
                                    target_name=rec_name,
                                    target_is_enemy=target_is_enemy,
                                    available_skills=skill_states,
                                    skill_mapping=skill_mapping_snapshot,
                                    physical_slots=self.turn_detector.PHYSICAL_SLOTS,
                                    my_status=my_status,
                                    opp_status=opp_status,
                                    crop=frame.crop,
                                    skip_initial_reaction_sleep=True,
                                )
                                if action_msg:
                                    with self._state_lock:
                                        self.latest_auto_action_text = action_msg
                                    if self.mode_controller and hasattr(
                                        self.mode_controller, "update_status_text"
                                    ):
                                        self.mode_controller.update_status_text(action_msg)
                            except Exception as exec_err:
                                logger.warning(
                                    f"⚠️ [AutoExecutor] 回合執行異常 (已隔離恢復): {exec_err}", exc_info=True
                                )
                else:
                    if self.no_box_start_time == 0.0:
                        self.no_box_start_time = current_time
                    elif (current_time - self.no_box_start_time) >= 8.0:
                        logger.info("🏁 [BattleLoop] 實戰畫面已結束，自動隱藏戰術面板與 HUD。")
                        self.reset_to_blank(root_window)
                        break

                with self._state_lock:
                    auto_txt = self.latest_auto_action_text
                    adv_txt = self.latest_advice_text
                    if auto_txt and adv_txt:
                        advice_to_show = f"{auto_txt}\n\n{adv_txt}"
                    else:
                        advice_to_show = auto_txt or adv_txt

                if advice_to_show and not self._is_cancelled():
                    self.presenter.schedule_advice_text(root_window, advice_to_show, self._is_cancelled)

                poll_sec = self.POLL_MY_TURN if is_my_turn else self.POLL_OPP_TURN
                self._wake_event.wait(timeout=poll_sec)
                self._wake_event.clear()

        except Exception as loop_exc:
            logger.exception(f"❌ [BattleLoop] 主循環發生未預期異常: {loop_exc}")
        finally:
            self.auto_executor.stop_immediately()
            self.is_active_session = False
            self.model_ready_event.set()

    def trigger_analysis(
        self,
        root_window: tk.Tk,
        my_draft: list[str],
        opp_draft: list[str],
        screen_color: Any = None,
        hwnd: int = 0,
        sync_wait: bool = False,
        from_five_picks_lock: bool = False,
    ) -> None:
        current_time = time.time()
        self.root_window = root_window

        with self.session_lock:
            if self.user_closed:
                return

            if self.battle_thread and self.battle_thread.is_alive():
                self.stop_requested = True
                self.wake_loop()
                self.battle_thread.join(timeout=1.0)

            self.battle_has_started = False
            self._reset_entry_gate_state()
            self.no_box_start_time = 0.0

            with self._state_lock:
                self.latest_advice_text = ""
                self.latest_auto_action_text = ""
                self.my_slot_map.clear()
                self.last_turn_key = None
                self._decision_ready_ts = 0.0
                self._llm_override_slot = None
                self._llm_override_skill = None

            if self.auto_battle_enabled:
                self.auto_executor.reset_for_match()
                self.auto_executor.start_kill_switch_monitor(on_emergency_stop=self._handle_emergency_stop)
            else:
                self.auto_executor.stop_immediately()

            with self.models.yolo_lock:
                # CRITICAL: Preserve locked 5-pick draft portrait templates across battle entry & mode toggles!
                if not self.action_bar_detector.has_locked_templates:
                    self.action_bar_detector.reset()
                    if isinstance(screen_color, np.ndarray) and len(my_draft) == 5:
                        self.action_bar_detector.capture_prep_templates(
                            screen_color, our_picks=my_draft, lock_on_success=True
                        )
                else:
                    self.action_bar_detector.bind_picks(my_draft)
                self.parser.reset_baseline()
                self.turn_detector.reset_cache()

            self.is_active_session = True
            self.stop_requested = False
            self.session_start_time = current_time
            self._wake_event.clear()

            self.battle_thread = threading.Thread(
                target=self._battle_loop,
                args=(root_window, list(my_draft), list(opp_draft), int(hwnd), bool(from_five_picks_lock)),
                name="BattleCoordinatorLoop",
                daemon=True,
            )
            self.battle_thread.start()

        if sync_wait:
            self.model_ready_event.wait(timeout=3.0)
