"""5v5 Real-Time Battle Coordinator for Onmyoji Tactical Helper.

Orchestrates screen capture, YOLOv8 spatial tracking, turn/skill geometric detection,
asynchronous Gemma-4 reasoning, instant click-through HUD presentation, and Red Mode
skill-first auto-execution (with `1.0 ~ 1.5s` post-decision camera-settle buffer and
`1.2 ~ 1.8s` unfinished-turn retry loop) under RLock-protected thread-safe session state.
"""

import ctypes
import logging
import random
import threading
import time
import tkinter as tk
from typing import Any

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
    POST_DECISION_BUFFER_RANGE: tuple[float, float] = (1.0, 1.5)

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
        self.consecutive_alive_frames: int = 0
        self._frame_seq: int = 0

        self.is_ai_reasoning: bool = False
        self.latest_advice_text: str = ""
        self.latest_auto_action_text: str = ""
        self.last_llm_time: float = 0.0
        self.my_draft_live: list[str] = []
        self.opp_draft_live: list[str] = []
        self.my_slot_map: dict[str, str] = {}
        self.last_turn_key: tuple[bool, str | None] | None = None

        # Non-blocking 1.0~1.5s post-decision settle state (keeps HUD refreshing smoothly while waiting)
        self._decision_ready_ts: float = 0.0
        self._turn_settle_delay: float = 1.2
        self._llm_override_slot: str | None = None

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

    def show_windows_immediately(self, root_window: tk.Tk, show_overlay: bool = True) -> None:
        """Creates `AITacticalUI` and `BattleOverlayUI` immediately on mode toggle."""
        self.root_window = root_window
        self.presenter.ensure_windows(
            root_window=root_window,
            show_overlay=show_overlay,
            on_closed_by_user=self.on_ui_closed_by_user,
            is_cancelled_fn=lambda: self.user_closed,
        )

    def update_live_draft(self, my_draft: list[str], opp_draft: list[str]) -> None:
        """Hot-updates 5v5 draft names in-place if the session was started before 5 picks locked."""
        with self._state_lock:
            self.my_draft_live = list(my_draft)
            self.opp_draft_live = list(opp_draft)
            self.my_slot_map = {f"Slot {i + 1}": name for i, name in enumerate(my_draft)}
        self.wake_loop()

    def enable_mode(self, mode_name: str) -> None:
        """Configures HUD overlay and Red Mode auto-executor flags on mode switch."""
        target = mode_name.value if isinstance(mode_name, OperatingMode) else str(mode_name)
        with self.session_lock, self._state_lock:
            self.user_closed = False
            self.stop_requested = False
            self.show_overlay = target in (OperatingMode.YELLOW.value, OperatingMode.RED.value)
            self.auto_battle_enabled = target == OperatingMode.RED.value
            self.latest_auto_action_text = ""
            self._decision_ready_ts = 0.0

        if target == OperatingMode.RED.value:
            self.auto_executor.reset_for_match()
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
                        self.auto_executor.is_game_window_foreground(game_hwnd, allow_reclaim_from_self=True)
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
        """Synchronously halts active battle threads and destroys floating UIs (keeping models warm by default)."""
        self.auto_battle_enabled = False
        self.auto_executor.stop_immediately()
        self.wake_loop()

        with self.session_lock:
            self.user_closed = True
            self.stop_requested = True
            self.is_active_session = False
            self.presenter.destroy_all(self.root_window)

        if self.battle_thread and self.battle_thread.is_alive():
            if self.battle_thread != threading.current_thread():
                self.battle_thread.join(timeout=1.5)

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

    def reset_to_blank(self, root_window: tk.Tk | None) -> None:
        """Resets battle session state and clears HUD/advice panels on match completion."""
        self.stop_requested = True
        self.battle_has_started = False
        self.no_box_start_time = 0.0
        self.consecutive_alive_frames = 0
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

        with self.models.yolo_lock:
            self.action_bar_detector.reset()
            self.parser.reset_baseline()
            self.turn_detector.reset_cache()

        self.is_active_session = False
        self.presenter.reset_blank(root_window)
        self.models.purge_cuda_cache()

    def reset_for_next_match(self) -> None:
        """Prepares coordinator for a fresh match session."""
        with self.session_lock, self._state_lock:
            self.stop_requested = True
            self.battle_has_started = False
            self.consecutive_alive_frames = 0
            self.no_box_start_time = 0.0
            self.latest_advice_text = ""
            self.latest_auto_action_text = ""
            self.my_slot_map.clear()
            self.last_turn_key = None
            self._decision_ready_ts = 0.0
            self._llm_override_slot = None
            self.user_closed = False
        self.auto_executor.stop_immediately()
        self.wake_loop()

        if self.battle_thread and self.battle_thread.is_alive():
            if self.battle_thread != threading.current_thread():
                self.battle_thread.join(timeout=1.0)

    def _async_llm_reasoning(self, payload: dict[str, Any], turn_key: tuple[bool, str | None]) -> None:
        """Runs Gemma-4 / rule reasoning asynchronously and marks post-decision buffer ready."""
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
    ) -> None:
        clicker = CaptureEngine()
        with self._state_lock:
            self.my_draft_live = list(my_draft)
            self.opp_draft_live = list(opp_draft)
            self.my_slot_map = {f"Slot {i + 1}": name for i, name in enumerate(my_draft)}
            self.last_turn_key = None
            self._decision_ready_ts = 0.0
            self._llm_override_slot = None

        try:
            # Show HUD Overlay & AI Panel BEFORE model check so UI appears immediately!
            self.presenter.ensure_windows(
                root_window=root_window,
                show_overlay=self.show_overlay,
                on_closed_by_user=self.on_ui_closed_by_user,
                is_cancelled_fn=lambda: self.user_closed or self.stop_requested,
            )

            self.model_ready_event.clear()
            if not self.models.ensure_loaded(should_abort_fn=lambda: self.stop_requested or self.user_closed):
                self.is_active_session = False
                return

            if self.context_builder is None:
                self.context_builder = BattleContextBuilder(repo=self.repository)

            self.model_ready_event.set()

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

                if not self.battle_has_started and len(self.action_bar_detector.prep_templates) < 5:
                    self.action_bar_detector.capture_prep_templates(screen_color, our_picks=cur_my_draft)

                active_slot: str | None = None
                active_shiki: str = UNKNOWN_SHIKIGAMI
                if is_my_turn:
                    res_slot, _, _ = self.action_bar_detector.identify_active_turn(screen_color)
                    if res_slot:
                        active_slot = res_slot
                        with self._state_lock:
                            active_shiki = self.my_slot_map.get(active_slot, UNKNOWN_SHIKIGAMI)

                current_time = time.time()
                current_turn_key = (is_my_turn, active_slot)
                with self._state_lock:
                    if current_turn_key != self.last_turn_key:
                        self.last_turn_key = current_turn_key
                        self.latest_advice_text = ""
                        self.last_llm_time = 0.0
                        self._llm_override_slot = None
                        if is_my_turn:
                            self._turn_settle_delay = random.uniform(*self.POST_DECISION_BUFFER_RANGE)
                            # In 0-VRAM Rule Mode, decision is ready on this exact frame; in GPU LLM mode, wait for LLM completion (bounded below)
                            self._decision_ready_ts = (
                                0.0 if self.models.has_active_gpu_llm() else current_time
                            )
                        else:
                            self._decision_ready_ts = 0.0
                            self.latest_auto_action_text = ""

                if self._is_cancelled():
                    break

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

                my_status, opp_status = self.parser.parse_frame(
                    cur_my_draft, cur_opp_draft, boxes_data, screen_w, screen_height=screen_h
                )

                my_alive_count = sum(1 for info in my_status.values() if info.get("alive", False))
                opp_alive_count = sum(1 for info in opp_status.values() if info.get("alive", False))
                total_alive = my_alive_count + opp_alive_count
                current_time = time.time()

                with self._state_lock:
                    locked_llm_slot = self._llm_override_slot

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

                self.auto_executor.observe_turn_state(is_my_turn)
                skill_mapping_snapshot = dict(self.turn_detector.get_skill_mapping(screen_color))

                # Render HUD Overlay immediately on Frame 1 (never wait for LLM reasoning!)
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

                if total_alive >= 4 or is_my_turn:
                    self.consecutive_alive_frames += 1
                    if not self.battle_has_started:
                        self.battle_has_started = True
                        logger.info(
                            f"⚔️ [BattleLoop] 已進入實戰追蹤 | hwnd={hwnd} | 畫面={screen_w}x{screen_h} "
                            f"| 我方存活={my_alive_count} 敵方存活={opp_alive_count} "
                            f"| overlay={self.show_overlay} auto_red={self.auto_battle_enabled}"
                        )
                    self.no_box_start_time = 0.0
                else:
                    self.consecutive_alive_frames = 0
                    if self.battle_has_started:
                        if self.no_box_start_time == 0.0:
                            self.no_box_start_time = current_time
                        if (current_time - self.no_box_start_time) >= 8.0:
                            self.reset_to_blank(root_window)
                            break
                    elif (current_time - self.session_start_time) >= 60.0:
                        self.reset_to_blank(root_window)
                        break

                if total_alive >= 1 or is_my_turn:
                    # Populate immediate rule-based advice on Frame 1 before spawning async LLM
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

                        # On initial attempt of the turn, wait 1.0~1.5s post-decision while keeping HUD live!
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
            self.consecutive_alive_frames = 0
            self.no_box_start_time = 0.0

            with self._state_lock:
                self.latest_advice_text = ""
                self.latest_auto_action_text = ""
                self.my_slot_map.clear()
                self.last_turn_key = None
                self._decision_ready_ts = 0.0
                self._llm_override_slot = None

            if self.auto_battle_enabled:
                self.auto_executor.reset_for_match()
                self.auto_executor.start_kill_switch_monitor(on_emergency_stop=self._handle_emergency_stop)
            else:
                self.auto_executor.stop_immediately()

            with self.models.yolo_lock:
                self.action_bar_detector.reset()
                self.parser.reset_baseline()
                self.turn_detector.reset_cache()

            self.is_active_session = True
            self.stop_requested = False
            self.session_start_time = current_time
            self._wake_event.clear()

            self.battle_thread = threading.Thread(
                target=self._battle_loop,
                args=(root_window, list(my_draft), list(opp_draft), int(hwnd)),
                name="BattleCoordinatorLoop",
                daemon=True,
            )
            self.battle_thread.start()

        if sync_wait:
            self.model_ready_event.wait(timeout=3.0)
