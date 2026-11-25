"""Safety Guard & Turn Cooldown Lock Module for Onmyoji Tactical Helper (Red Mode).

Manages the global `F12` / `ESC` emergency brake monitor thread, manual takeover stops,
and the single-turn anti-spam cooldown state machine (`>= 2.2s` + `True -> False -> True`).
"""

import logging
import threading
import time
from collections.abc import Callable

from onmyoji_helper.infra.win32_input import VK_ESCAPE, VK_F12, is_vk_pressed

logger = logging.getLogger("OnmyojiHelper")


class SafetyGuard:
    """Enforces Red Mode emergency brakes and single-action-per-turn cooldown locks."""

    def __init__(
        self,
        min_turn_cooldown: float = 2.2,
        extra_turn_escape_sec: float = 5.5,
        key_checker: Callable[[int], bool] | None = None,
    ) -> None:
        """Initializes cooldown thresholds, thread locks, and optional injectable key checker."""
        self.min_turn_cooldown: float = max(2.2, float(min_turn_cooldown))
        self.extra_turn_escape_sec: float = float(extra_turn_escape_sec)
        self._key_checker: Callable[[int], bool] = key_checker or is_vk_pressed

        self._lock: threading.Lock = threading.Lock()
        self._last_action_time: float = 0.0
        self._waiting_for_turn_cycle: bool = False
        self._last_observed_my_turn: bool = False
        self._emergency_stopped: bool = False
        self._emergency_reason: str = ""
        self.stop_event: threading.Event = threading.Event()

        self._kill_monitor_thread: threading.Thread | None = None
        self._kill_monitor_running: bool = False
        self._on_emergency_callback: Callable[[str], None] | None = None

    @property
    def emergency_stopped(self) -> bool:
        """Returns True if an emergency brake has been triggered in the current session."""
        return self._emergency_stopped

    def reset_for_match(self) -> None:
        """Resets all per-turn cooldown locks and emergency stop flags for a new session."""
        with self._lock:
            self._last_action_time = 0.0
            self._waiting_for_turn_cycle = False
            self._last_observed_my_turn = False
            self._emergency_stopped = False
            self._emergency_reason = ""
            self.stop_event.clear()

    def stop_immediately(self) -> None:
        """Halts pending execution sleeps and stops the background hotkey watcher thread."""
        self.stop_event.set()
        self.stop_kill_switch_monitor()
        with self._lock:
            self._waiting_for_turn_cycle = False
            self._last_action_time = 0.0

    def start_kill_switch_monitor(
        self,
        on_emergency_stop: Callable[[str], None] | None = None,
    ) -> None:
        """Starts the low-CPU global F12/ESC emergency kill-switch daemon thread."""
        self._on_emergency_callback = on_emergency_stop
        self._emergency_stopped = False
        self._emergency_reason = ""
        self.stop_event.clear()

        if self._kill_monitor_thread and self._kill_monitor_thread.is_alive():
            return

        self._kill_monitor_running = True
        self._kill_monitor_thread = threading.Thread(
            target=self._kill_switch_loop,
            name="RedModeKillSwitch",
            daemon=True,
        )
        self._kill_monitor_thread.start()

    def stop_kill_switch_monitor(self) -> None:
        """Signals the kill-switch monitor loop to exit."""
        self._kill_monitor_running = False

    def trigger_emergency_stop(self, reason: str = "按下緊急煞車熱鍵") -> None:
        """Engages emergency brake immediately and invokes the downgrade callback once."""
        should_notify = False
        with self._lock:
            if not self._emergency_stopped:
                self._emergency_stopped = True
                self._emergency_reason = reason
                self.stop_event.set()
                should_notify = True

        self._kill_monitor_running = False
        if should_notify:
            logger.warning(f"🛑 [AutoExecutor] 緊急煞車觸發：{reason}")
            cb = self._on_emergency_callback
            if callable(cb):
                try:
                    cb(reason)
                except Exception as exc:
                    logger.warning(f"⚠️ [AutoExecutor] 緊急煞車回呼異常: {exc}")

    def poll_kill_switch_once(self) -> bool:
        """Polls F12 and ESC state once; triggers emergency stop if either is held."""
        if self._emergency_stopped:
            return True
        if self._key_checker(VK_F12):
            self.trigger_emergency_stop("偵測到 F12 緊急煞車")
            return True
        if self._key_checker(VK_ESCAPE):
            self.trigger_emergency_stop("偵測到 ESC 緊急煞車")
            return True
        return False

    def _kill_switch_loop(self) -> None:
        while self._kill_monitor_running and not self.stop_event.is_set():
            if self.poll_kill_switch_once():
                break
            time.sleep(0.05)

    def observe_turn_state(self, is_my_turn: bool) -> None:
        """Unlocks the turn-cycle latch once `is_my_turn` transitions `True -> False`."""
        with self._lock:
            if not is_my_turn:
                self._waiting_for_turn_cycle = False
            self._last_observed_my_turn = bool(is_my_turn)

    def can_execute_this_turn(self, now: float | None = None) -> bool:
        """Checks whether a new turn action is permitted under cooldown and turn-cycle locks."""
        now_ts = time.time() if now is None else float(now)
        if self._emergency_stopped or self.stop_event.is_set():
            return False

        with self._lock:
            elapsed = now_ts - self._last_action_time
            if self._last_action_time > 0.0 and elapsed < self.min_turn_cooldown:
                return False
            if self._waiting_for_turn_cycle and elapsed < self.extra_turn_escape_sec:
                return False
            return True

    def mark_action_executed(self, now: float | None = None) -> None:
        """Latches the turn cooldown lock right after a skill click completes."""
        now_ts = time.time() if now is None else float(now)
        with self._lock:
            self._last_action_time = now_ts
            self._waiting_for_turn_cycle = True
