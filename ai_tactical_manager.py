# -*- coding: utf-8 -*-
"""AI Tactical Manager for Onmyoji Tactical Helper.

This module orchestrates real-time battle state tracking, YOLO unit detection,
background LLM tactical reasoning, CUDA memory lifecycle management, and HUD UI overlays.
"""

import gc
import os
import sys
import threading
import time
import tkinter as tk
from typing import Any, Dict, List, Optional, Tuple

from ultralytics import YOLO

from action_bar_detector import ActionBarDetector
from ai_tactical_ui import AITacticalUI
from battle_context import BattleContextBuilder
from battle_monitor import StateParser
from capture_engine import CaptureEngine
from llm_engine import TacticalEngine
from overlay_module import BattleOverlayUI
from turn_detector import TurnDetector


def get_base_dir() -> str:
    """Retrieves the absolute base directory path for local assets and executables.

    Returns:
        Absolute directory path as a string.
    """
    if getattr(sys, 'frozen', False) or "__compiled__" in globals():
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


BASE_DIR: str = get_base_dir()
MODEL_PATH: str = os.path.join(BASE_DIR, "best.pt")


class AITacticalManager:
    """Manages AI inference loops, YOLO detection, LLM reasoning, and UI overlays."""

    def __init__(self) -> None:
        """Initializes the tactical manager with thread locks, state detectors, and UI handlers."""
        self.yolo_model: Optional[YOLO] = None
        self.tactical_engine: Optional[TacticalEngine] = None
        self.context_builder: Optional[BattleContextBuilder] = None
        self.parser: StateParser = StateParser()
        self.turn_detector: TurnDetector = TurnDetector()
        self.action_bar_detector: ActionBarDetector = ActionBarDetector()

        self.ai_ui_instance: Optional[AITacticalUI] = None
        self.overlay_ui_instance: Optional[BattleOverlayUI] = None
        self.mode_controller: Any = None  # Reference to ModeController instance

        self.user_closed: bool = False
        self.stop_requested: bool = False
        self.is_active_session: bool = False
        self.match_finished_lock: bool = False
        self.battle_has_started: bool = False

        self.last_main_call_time: float = 0.0
        self.match_finished_time: float = 0.0
        self.no_box_start_time: float = 0.0
        self.session_start_time: float = 0.0
        self.consecutive_alive_frames: int = 0

        self.is_ai_reasoning: bool = False
        self.latest_advice_text: str = ""
        self.llm_target_slot: Optional[str] = None
        self.last_llm_time: float = 0.0
        self.my_slot_map: Dict[str, str] = {}
        self.last_turn_key: Optional[Tuple[bool, Optional[str]]] = None

        # Thread synchronization locks and synchronization events
        self.model_lock: threading.Lock = threading.Lock()
        self.session_lock: threading.Lock = threading.Lock()
        self.model_ready_event: threading.Event = threading.Event()
        self.battle_thread: Optional[threading.Thread] = None
        self.root_window: Optional[tk.Tk] = None

        self.show_overlay: bool = False  # Debug HUD Overlay feature flag

    def set_mode_controller(self, mode_controller: Any) -> None:
        """Binds the mode controller to enable bidirectional UI state synchronization.

        Args:
            mode_controller: Instance of the ModeController.
        """
        self.mode_controller = mode_controller

    def enable_mode(self, mode_name: str) -> None:
        """Configures feature flags and UI overlays based on the specified operating mode.

        Args:
            mode_name: Name of the mode ("green", "yellow", or "red").
        """
        with self.session_lock:
            self.user_closed = False
            self.stop_requested = False
            self.show_overlay = (mode_name == "red")

    def enable_yellow_mode(self) -> None:
        """Backwards compatibility handler forwarding to enable_mode("yellow")."""
        self.enable_mode("yellow")

    def close_ui_sync(self) -> None:
        """Synchronously stops analysis threads and cleans up GPU memory.

        Called by ModeController during mode switches.
        """
        with self.session_lock:
            self.user_closed = True
            self.stop_requested = True
            self.is_active_session = False

            ui_to_destroy = self.ai_ui_instance
            overlay_to_destroy = self.overlay_ui_instance
            self.ai_ui_instance = None
            self.overlay_ui_instance = None

            if self.root_window:
                self.root_window.after(
                    0, lambda: self._destroy_uis_safe(ui_to_destroy, overlay_to_destroy)
                )

        # 1. Wait for battle analysis loop thread to exit
        if self.battle_thread and self.battle_thread.is_alive():
            if self.battle_thread != threading.current_thread():
                self.battle_thread.join(timeout=3.0)

        # 2. Wait for asynchronous LLM reasoning thread to complete
        retry = 0
        while self.is_ai_reasoning and retry < 25:
            time.sleep(0.1)
            retry += 1

        # 3. Release AI models and purge CUDA VRAM cache
        self.unload_models()

    def close_ui(self) -> None:
        """Asynchronously triggers UI closing and resource release."""
        threading.Thread(target=self.close_ui_sync, daemon=True).start()

    def _destroy_uis_safe(
        self,
        ui_inst: Optional[AITacticalUI],
        overlay_inst: Optional[BattleOverlayUI]
    ) -> None:
        """Safely destroys UI windows on the main GUI thread.

        Args:
            ui_inst: Tactical advice UI instance to destroy.
            overlay_inst: HUD overlay UI instance to close.
        """
        if ui_inst:
            try:
                if hasattr(ui_inst, 'window') and ui_inst.window.winfo_exists():
                    ui_inst.window.destroy()
            except Exception:
                pass
        if overlay_inst:
            try:
                overlay_inst.close()
            except Exception:
                pass

    def on_ui_closed_by_user(self) -> None:
        """Handles window close event triggered by clicking the close button ('X').

        Switches the system mode back to Green mode.
        """
        if self.mode_controller and hasattr(self.mode_controller, "set_mode"):
            success = self.mode_controller.set_mode("green")
            if not success:
                self.close_ui()
        else:
            self.close_ui()

    def unload_models(self) -> None:
        """Safely unloads YOLO and LLM engines, releasing CUDA VRAM resources."""
        with self.model_lock:
            self.model_ready_event.clear()
            if self.tactical_engine is not None:
                try:
                    self.tactical_engine.release()
                except Exception:
                    pass
                self.tactical_engine = None

            if self.yolo_model is not None:
                try:
                    if hasattr(self.yolo_model, 'model'):
                        del self.yolo_model.model
                    del self.yolo_model
                except Exception:
                    pass
                self.yolo_model = None

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                pass

            gc.collect()
            gc.collect()
            time.sleep(0.3)

    def reset_to_blank(self, root_window: tk.Tk) -> None:
        """Resets session state and clears UI windows on battle end or timeout.

        Args:
            root_window: Reference to the main Tkinter root window.
        """
        self.stop_requested = True
        self.battle_has_started = False
        self.no_box_start_time = 0.0
        self.consecutive_alive_frames = 0
        self.latest_advice_text = ""
        self.llm_target_slot = None
        self.last_llm_time = 0.0
        self.my_slot_map.clear()
        self.last_turn_key = None

        with self.model_lock:
            self.action_bar_detector.prep_templates.clear()
            if hasattr(self.action_bar_detector, 'slot_to_shiki'):
                self.action_bar_detector.slot_to_shiki.clear()
            self.parser.reset_baseline()

        self.match_finished_lock = True
        self.match_finished_time = time.time()
        self.is_active_session = False

        ui_to_reset = self.ai_ui_instance
        overlay_to_reset = self.overlay_ui_instance

        def _reset_uis() -> None:
            if ui_to_reset and hasattr(ui_to_reset, 'window') and ui_to_reset.window.winfo_exists():
                ui_to_reset.reset_to_blank()
            if overlay_to_reset and hasattr(overlay_to_reset, 'window') and overlay_to_reset.window.winfo_exists():
                overlay_to_reset.clear()

        root_window.after(0, _reset_uis)

    def _async_llm_reasoning(self, payload: Dict[str, Any]) -> None:
        """Executes LLM tactical reasoning in a background thread.

        Args:
            payload: Structured battle context data payload.
        """
        try:
            with self.model_lock:
                if (
                    self.tactical_engine is None
                    or self.user_closed
                    or self.stop_requested
                    or not self.is_active_session
                ):
                    return
                advice_dict = self.tactical_engine.get_tactical_advice(payload, mode="LLM")

            advice_msg = advice_dict.get("advice", "")

            for slot_key, shiki_name in self.my_slot_map.items():
                if shiki_name:
                    target_str = f"我方 [{slot_key}]"
                    replacement_str = f"我方 [{slot_key}] {shiki_name}"
                    if target_str in advice_msg and replacement_str not in advice_msg:
                        advice_msg = advice_msg.replace(target_str, replacement_str)

            display_slot = payload.get("recommended_target_slot")
            target_name = payload.get("recommended_target_name", "目標單位")

            if display_slot and target_name and "全體" not in target_name:
                final_text = f"🎯 建議優先處理：[{display_slot}] {target_name}\n\n💡 戰術對策：\n{advice_msg}"
            else:
                final_text = f"💡 戰術對策：\n{advice_msg}"

            self.latest_advice_text = final_text
        except Exception:
            pass
        finally:
            self.is_ai_reasoning = False

    def _battle_loop(self, root_window: tk.Tk, my_draft: List[str], opp_draft: List[str]) -> None:
        """Main loop performing screen capture, YOLO detection, and turn tracking.

        Args:
            root_window: Reference to the Tkinter root window.
            my_draft: List of drafted character names for player side.
            opp_draft: List of drafted character names for opponent side.
        """
        clicker = CaptureEngine()

        self.my_slot_map = {f"Slot {i+1}": name for i, name in enumerate(my_draft)}
        self.last_turn_key = None

        try:
            self.model_ready_event.clear()

            # 1. Atomically load YOLO and Tactical Engine models
            if self.stop_requested or self.user_closed:
                return

            with self.model_lock:
                if self.stop_requested or self.user_closed:
                    return
                if self.yolo_model is None:
                    if not os.path.exists(MODEL_PATH):
                        self.is_active_session = False
                        return
                    self.yolo_model = YOLO(MODEL_PATH)

                if self.context_builder is None:
                    self.context_builder = BattleContextBuilder()

                if self.tactical_engine is None:
                    if not self.stop_requested and not self.user_closed:
                        self.tactical_engine = TacticalEngine()

            # Models successfully loaded, signal ready event
            self.model_ready_event.set()

            def _ensure_ui() -> None:
                if not self.user_closed and not self.stop_requested:
                    # 1. Tactical Advice Panel (Enabled in Yellow and Red modes)
                    if (
                        self.ai_ui_instance is None
                        or not hasattr(self.ai_ui_instance, 'window')
                        or not self.ai_ui_instance.window.winfo_exists()
                    ):
                        self.ai_ui_instance = AITacticalUI(
                            root_window, close_callback=self.on_ui_closed_by_user
                        )

                    # 2. Debug HUD Overlay (Enabled strictly in Red mode)
                    if self.show_overlay:
                        if (
                            self.overlay_ui_instance is None
                            or not hasattr(self.overlay_ui_instance, 'window')
                            or not self.overlay_ui_instance.window.winfo_exists()
                        ):
                            self.overlay_ui_instance = BattleOverlayUI(root_window)

            root_window.after(0, _ensure_ui)

            # 2. Main battle analysis loop
            while not self.user_closed and not self.stop_requested and self.is_active_session:
                windows = clicker.find_windows()
                if not windows:
                    time.sleep(0.5)
                    continue

                hwnd = windows[0]._hWnd
                screen_gray, screen_color = clicker.capture_window_bitblt(hwnd)

                if screen_color is None:
                    time.sleep(0.3)
                    continue

                screen_h, screen_w = screen_color.shape[:2]

                is_my_turn = self.turn_detector.is_my_turn(screen_color)
                skill_states = self.turn_detector.detect_available_skills(screen_color)

                if self.user_closed or self.stop_requested:
                    break

                if not self.battle_has_started and len(self.action_bar_detector.prep_templates) < 5:
                    time.sleep(0.5)
                    if self.user_closed or self.stop_requested:
                        break
                    _, fresh_screen = clicker.capture_window_bitblt(hwnd)
                    target_frame = fresh_screen if fresh_screen is not None else screen_color
                    self.action_bar_detector.capture_prep_templates(
                        target_frame, our_picks=my_draft, delay=0
                    )

                active_slot = None
                active_shiki = "未知"
                if is_my_turn:
                    res_slot, _, score = self.action_bar_detector.identify_active_turn(screen_color)
                    if res_slot:
                        active_slot = res_slot
                        active_shiki = self.my_slot_map.get(active_slot, "未知")

                current_turn_key = (is_my_turn, active_slot)
                if current_turn_key != self.last_turn_key:
                    self.last_turn_key = current_turn_key
                    self.llm_target_slot = None
                    self.latest_advice_text = ""
                    self.last_llm_time = 0.0

                results = None
                with self.model_lock:
                    if (
                        self.yolo_model is not None
                        and not self.user_closed
                        and not self.stop_requested
                        and self.is_active_session
                    ):
                        results = self.yolo_model.predict(
                            screen_color, conf=0.30, iou=0.45, verbose=False
                        )

                if results is None or self.user_closed or self.stop_requested or not self.is_active_session:
                    break

                boxes_data = []
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    xyxy = box.xyxy[0].cpu().numpy()
                    cx = (xyxy[0] + xyxy[2]) / 2.0
                    cy = (xyxy[1] + xyxy[3]) / 2.0
                    boxes_data.append({"cls": cls_id, "cx": cx, "cy": cy})

                my_status, opp_status = self.parser.parse_frame(
                    my_draft, opp_draft, boxes_data, screen_w
                )

                my_alive_count = sum(1 for info in my_status.values() if info.get("alive", False))
                opp_alive_count = sum(1 for info in opp_status.values() if info.get("alive", False))
                total_alive = my_alive_count + opp_alive_count

                current_time = time.time()

                payload = self.context_builder.build_payload(
                    my_status,
                    opp_status,
                    is_my_turn=is_my_turn,
                    skill_states=skill_states,
                    locked_target_slot=None,
                    active_slot=active_slot,
                    active_shiki_name=active_shiki
                )

                rec_slot = payload.get("recommended_target_slot")
                rec_name = payload.get("recommended_target_name", "")
                target_is_enemy = payload.get("target_is_enemy", True)
                final_target_slot = rec_slot

                # Execute HUD Overlay drawing strictly when Red Mode feature flag is active
                if self.show_overlay:
                    def _draw_overlay() -> None:
                        if (
                            not self.user_closed
                            and not self.stop_requested
                            and self.overlay_ui_instance
                            and hasattr(self.overlay_ui_instance, 'window')
                            and self.overlay_ui_instance.window.winfo_exists()
                        ):
                            try:
                                self.overlay_ui_instance.sync_position(hwnd)
                                self.overlay_ui_instance.clear()
                                if total_alive > 0:
                                    self.overlay_ui_instance.update_targets(
                                        my_status,
                                        is_enemy=False,
                                        img_w=screen_w,
                                        img_h=screen_h,
                                        target_slot=(final_target_slot if not target_is_enemy else None)
                                    )
                                    self.overlay_ui_instance.update_targets(
                                        opp_status,
                                        is_enemy=True,
                                        img_w=screen_w,
                                        img_h=screen_h,
                                        target_slot=(final_target_slot if target_is_enemy else None)
                                    )
                                self.overlay_ui_instance.update_skills(
                                    detector=self.turn_detector,
                                    is_my_turn=is_my_turn,
                                    skill_states=skill_states,
                                    img_w=screen_w,
                                    img_h=screen_h,
                                    screen_color=screen_color
                                )
                            except Exception:
                                pass

                    root_window.after(0, _draw_overlay)

                if total_alive >= 5:
                    self.consecutive_alive_frames += 1
                    if self.consecutive_alive_frames >= 3 and not self.battle_has_started:
                        self.battle_has_started = True

                    self.no_box_start_time = 0.0
                else:
                    self.consecutive_alive_frames = 0
                    if self.battle_has_started:
                        if self.no_box_start_time == 0.0:
                            self.no_box_start_time = current_time

                        elapsed = current_time - self.no_box_start_time
                        if elapsed >= 5.0:
                            self.reset_to_blank(root_window)
                            break
                    else:
                        wait_time = current_time - self.session_start_time
                        if wait_time >= 30.0:
                            self.reset_to_blank(root_window)
                            break

                if self.battle_has_started and total_alive >= 1:
                    fallback_msg = payload.get("fallback_advice", "")

                    if not self.latest_advice_text:
                        if final_target_slot and rec_name and "全體" not in rec_name:
                            self.latest_advice_text = f"🎯 建議優先處理：[{final_target_slot}] {rec_name}\n\n💡 戰術對策：\n{fallback_msg}"
                        else:
                            self.latest_advice_text = f"💡 戰術對策：\n{fallback_msg}"

                    if not self.is_ai_reasoning and (
                        current_time - self.last_llm_time >= 3.0 or self.last_llm_time == 0.0
                    ):
                        self.is_ai_reasoning = True
                        self.last_llm_time = current_time
                        threading.Thread(
                            target=self._async_llm_reasoning, args=(payload,), daemon=True
                        ).start()

                if not self.user_closed and not self.stop_requested and self.latest_advice_text:
                    advice_to_show = self.latest_advice_text
                    root_window.after(0, lambda: self._update_ui_text(advice_to_show))

                time.sleep(0.3)

        finally:
            self.is_active_session = False
            self.model_ready_event.set()

    def _update_ui_text(self, advice_text: str) -> None:
        """Updates the displayed tactical advice text on the UI.

        Args:
            advice_text: Formatted tactical advice string.
        """
        if (
            not self.user_closed
            and not self.stop_requested
            and self.ai_ui_instance
            and hasattr(self.ai_ui_instance, 'window')
            and self.ai_ui_instance.window.winfo_exists()
        ):
            self.ai_ui_instance.update_advice(advice_text)

    def trigger_analysis(
        self,
        root_window: tk.Tk,
        my_draft: List[str],
        opp_draft: List[str],
        screen_color: Any,
        hwnd: int,
        sync_wait: bool = False
    ) -> None:
        """Triggers the asynchronous battle analysis loop thread.

        Args:
            root_window: Reference to the primary Tkinter root window.
            my_draft: Drafted character list for the player.
            opp_draft: Drafted character list for the opponent.
            screen_color: Current screen frame array.
            hwnd: Window handle identifier for the game client.
            sync_wait: Whether to synchronously block until model initialization completes.
        """
        current_time = time.time()
        self.root_window = root_window

        with self.session_lock:
            if self.user_closed or self.stop_requested:
                return

            if self.battle_thread and self.battle_thread.is_alive():
                return

            if self.match_finished_lock and not self.is_active_session:
                time_since_last_call = current_time - self.last_main_call_time
                time_since_finished = current_time - self.match_finished_time

                if self.last_main_call_time > 0 and time_since_last_call > 5.0 and time_since_finished > 5.0:
                    self.match_finished_lock = False
                    self.battle_has_started = False
                    self.consecutive_alive_frames = 0
                    self.no_box_start_time = 0.0
                    self.llm_target_slot = None

            self.last_main_call_time = current_time

            if self.match_finished_lock:
                return

            self.is_active_session = True
            self.stop_requested = False
            self.session_start_time = current_time
            my_draft_copy = list(my_draft)
            opp_draft_copy = list(opp_draft)

            self.battle_thread = threading.Thread(
                target=self._battle_loop,
                args=(root_window, my_draft_copy, opp_draft_copy),
                daemon=True
            )
            self.battle_thread.start()

        if sync_wait:
            self.model_ready_event.wait(timeout=10.0)