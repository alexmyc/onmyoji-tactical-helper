"""Deterministic Ban/Pick Draft Finite-State Machine for Onmyoji Tactical Helper.

Decouples pure drafting transition rules (Ban hard-gate, 1..5 sequential pick locking,
rematch/settlement resets, and idle sleep timeouts) from Win32 screen capture and OCR,
enabling 100% offline unit testing.
"""

import logging
import time
from collections.abc import Sequence
from typing import Any

from onmyoji_helper.domain.models import DraftPhase, DraftSnapshot

logger = logging.getLogger("OnmyojiHelper")

BAN_KEYWORDS: tuple[str, ...] = ("禁", "禁選", "禁选", "禁用")
# Explicit multi-char Ban headers only (never bare '禁', which appears in combat debuffs like '禁療')
BAN_RESET_KEYWORDS: tuple[str, ...] = ("禁選", "禁选", "禁用", "對方禁", "对方禁", "我方禁")
QUEUE_KEYWORDS: tuple[str, ...] = ("配對中", "配对中", "匹配成功")
LOBBY_KEYWORDS: tuple[str, ...] = ("名士", "陰陽師")
SETTLE_KEYWORDS: tuple[str, ...] = ("勝利", "失敗", "胜利", "失败")


class DraftStateMachine:
    """Manages Ban/Pick draft state transitions and enforces the strict Ban Hard Gate."""

    IDLE_SLEEP_TIMEOUT: float = 25.0
    PARTIAL_DRAFT_TIMEOUT: float = 90.0
    POST_FIVE_PICK_LOBBY_GUARD_SEC: float = 45.0
    POST_FIVE_PICK_BAN_GUARD_SEC: float = 12.0

    def __init__(self) -> None:
        """Initializes an empty draft state machine in `SLEEP` phase."""
        self.phase: DraftPhase = DraftPhase.SLEEP
        self.remembered_opp_ban: str | None = None
        self.opp_draft: list[str] = []
        self.my_draft: list[str] = []
        self.has_triggered_five_picks: bool = False
        self.five_picks_locked_time: float = 0.0
        self.in_ban_waiting: bool = False
        self.logged_ban_enter: bool = False
        self.last_wakeup_time: float = time.time()
        self.last_draft_change_time: float = time.time()

    def reset(self, target_phase: DraftPhase = DraftPhase.SLEEP) -> None:
        """Clears all remembered Ban/Pick state and resets phase."""
        self.phase = target_phase
        self.remembered_opp_ban = None
        self.opp_draft.clear()
        self.my_draft.clear()
        self.has_triggered_five_picks = False
        self.five_picks_locked_time = 0.0
        self.in_ban_waiting = (target_phase == DraftPhase.BAN_WAITING)
        self.logged_ban_enter = False
        now = time.time()
        self.last_wakeup_time = now
        self.last_draft_change_time = now

    def get_recommended_poll_interval(
        self,
        sleep_sec: float = 2.5,
        ban_wait_sec: float = 0.9,
        active_sec: float = 1.3,
        in_battle_sec: float = 4.5,
    ) -> float:
        """Returns the adaptive OCR polling interval for the current FSM phase."""
        if self.has_triggered_five_picks:
            return in_battle_sec
        if self.in_ban_waiting and not self.remembered_opp_ban:
            return ban_wait_sec
        return sleep_sec if self.phase == DraftPhase.SLEEP else active_sec

    @property
    def is_active_phase(self) -> bool:
        """Returns True when actively scanning Ban or 1..5 picks."""
        return self.phase in (DraftPhase.BAN_WAITING, DraftPhase.ACTIVE_DRAFT)

    def step(
        self,
        ocr_opp: Sequence[Any],
        ocr_my: Sequence[Any],
        sorted_full_to_short: Sequence[tuple[str, str]],
        my_ban: str,
        now: float | None = None,
        is_combat_active: bool = False,
    ) -> tuple[DraftSnapshot, bool]:
        """Advances the draft FSM given OCR token results from opponent and ally columns.

        Args:
            ocr_opp: RapidOCR result tuples `(box, text, score)` from the opponent/header ROI.
            ocr_my: RapidOCR result tuples `(box, text, score)` from the ally ROI.
            sorted_full_to_short: Length-descending `(full_name, short_name)` alias pairs.
            my_ban: Player's configured Ban short name for the active system.
            now: Optional injectable timestamp for deterministic testing.
            is_combat_active: True while 5v5 YOLOv8 combat tracking is actively running,
                preventing VS-banner / rank-header keywords (`'名士'`, `'陰陽師'`) from resetting
                an ongoing battle.

        Returns:
            Tuple `(DraftSnapshot, skip_ui_refresh)` where `skip_ui_refresh=True` when a
            transient `"不能上"` tooltip occluded the draft column.
        """
        now_ts = time.time() if now is None else float(now)
        should_reset_battle = False
        just_completed_five = False

        opp_texts = [str(res[1]) for res in ocr_opp] if ocr_opp else []
        my_texts = [str(res[1]) for res in ocr_my] if ocr_my else []
        full_text_opp = "".join(opp_texts)

        if full_text_opp:
            has_ban_kw = any(kw in full_text_opp for kw in BAN_KEYWORDS)
            has_explicit_ban_reset = any(kw in full_text_opp for kw in BAN_RESET_KEYWORDS)

            # 1. Explicit queue / rematch screens: reset any leftover partial/full draft
            if any(kw in full_text_opp for kw in QUEUE_KEYWORDS):
                self.last_wakeup_time = now_ts
                had_state = bool(
                    self.remembered_opp_ban
                    or self.opp_draft
                    or self.my_draft
                    or self.has_triggered_five_picks
                )
                if had_state:
                    logger.info(f"🔄 [DraftFSM] 偵測到配對畫面，重置對局狀態 (OCR='{full_text_opp[:36]}')")
                    self.reset(
                        DraftPhase.BAN_WAITING if "匹配成功" in full_text_opp else DraftPhase.ACTIVE_DRAFT
                    )
                    should_reset_battle = True
                else:
                    self.phase = (
                        DraftPhase.BAN_WAITING if "匹配成功" in full_text_opp else DraftPhase.ACTIVE_DRAFT
                    )
                    if "匹配成功" in full_text_opp:
                        self.in_ban_waiting = True

            # 2. Settlement screen (victory / defeat / early surrender) — check BEFORE lobby labels!
            elif any(kw in full_text_opp for kw in SETTLE_KEYWORDS):
                had_state = bool(
                    self.remembered_opp_ban
                    or self.opp_draft
                    or self.my_draft
                    or self.has_triggered_five_picks
                )
                if had_state:
                    logger.info(f"🏁 [DraftFSM] 偵測到對局結算畫面，重置狀態 (OCR='{full_text_opp[:36]}')")
                self.reset(DraftPhase.SLEEP)
                if had_state:
                    should_reset_battle = True

            # 3. Lobby / rank labels: NEVER reset on VS banner or while 5v5 combat is actively tracked!
            elif any(kw in full_text_opp for kw in LOBBY_KEYWORDS):
                self.last_wakeup_time = now_ts
                completed_five = (
                    self.has_triggered_five_picks
                    or len(self.opp_draft) >= 5
                    or len(self.my_draft) >= 5
                )
                post_lock_elapsed = (
                    (now_ts - self.five_picks_locked_time)
                    if self.five_picks_locked_time > 0.0
                    else float("inf")
                )
                if completed_five:
                    if not is_combat_active and post_lock_elapsed >= self.POST_FIVE_PICK_LOBBY_GUARD_SEC:
                        logger.info(
                            f"🏠 [DraftFSM] 對局已結束並返回鬥技大廳，重置狀態 (OCR='{full_text_opp[:36]}')"
                        )
                        self.reset(DraftPhase.ACTIVE_DRAFT)
                        should_reset_battle = True
                else:
                    self.phase = DraftPhase.ACTIVE_DRAFT
                    if has_ban_kw and not self.remembered_opp_ban:
                        self.in_ban_waiting = True
                        self.phase = DraftPhase.BAN_WAITING

            # 4. Ban phase header detected
            elif has_ban_kw:
                self.last_wakeup_time = now_ts
                if not self.remembered_opp_ban:
                    self.in_ban_waiting = True
                    self.phase = DraftPhase.BAN_WAITING
                    if not self.logged_ban_enter:
                        self.logged_ban_enter = True
                        logger.info(f"⏳ 已進入禁選階段，等待對手鎖定 Ban 位 (OCR='{full_text_opp[:40]}')")

            # Idle / stale partial-draft sleep protection
            if self.is_active_phase:
                if not self.remembered_opp_ban and not self.opp_draft and not self.my_draft:
                    if (now_ts - self.last_wakeup_time) > self.IDLE_SLEEP_TIMEOUT:
                        self.phase = DraftPhase.SLEEP
                        self.in_ban_waiting = False
                        self.logged_ban_enter = False
                elif 0 < len(self.opp_draft) < 5:
                    if (now_ts - self.last_draft_change_time) > self.PARTIAL_DRAFT_TIMEOUT:
                        self.phase = DraftPhase.SLEEP

            if "不能上" in full_text_opp:
                return self._make_snapshot(False, should_reset_battle), True

            # New Ban phase starting right after an earlier match without lobby screen:
            # Require explicit multi-char Ban header (NOT bare '禁' from '禁療') and post-5-pick guard!
            if has_explicit_ban_reset and (
                self.opp_draft or self.my_draft or self.has_triggered_five_picks
            ):
                allow_ban_reset = True
                if self.has_triggered_five_picks and self.five_picks_locked_time > 0.0:
                    allow_ban_reset = (
                        now_ts - self.five_picks_locked_time
                    ) >= self.POST_FIVE_PICK_BAN_GUARD_SEC
                if allow_ban_reset:
                    logger.info(
                        f"🔄 [DraftFSM] 偵測到新對局禁選階段，重置上一場陣容 (OCR='{full_text_opp[:36]}')"
                    )
                    self.reset(DraftPhase.BAN_WAITING)
                    self.last_wakeup_time = now_ts
                    should_reset_battle = True

            # Parse opponent Ban & 1..5 picks under the strict Ban Hard Gate
            for text in opp_texts:
                if not self.remembered_opp_ban:
                    if has_ban_kw:
                        for full_name, short_name in sorted_full_to_short:
                            if full_name in text and short_name:
                                self.remembered_opp_ban = short_name
                                self.in_ban_waiting = False
                                self.phase = DraftPhase.ACTIVE_DRAFT
                                self.last_draft_change_time = now_ts
                                logger.info(f"🚫 偵測到敵方 Ban 位: {short_name} (OCR原文='{text}')")
                                if short_name in self.opp_draft:
                                    self.opp_draft.remove(short_name)
                                break

                if self.remembered_opp_ban and len(self.opp_draft) < 5:
                    for full_name, short_name in sorted_full_to_short:
                        if full_name in text and short_name:
                            if not any(bkw in full_text_opp for bkw in ("禁", "禁選", "禁选")):
                                if (
                                    short_name != self.remembered_opp_ban
                                    and short_name != my_ban
                                    and short_name not in self.opp_draft
                                ):
                                    self.opp_draft.append(short_name)
                                    self.last_draft_change_time = now_ts
                                    logger.info(
                                        f"⚔️ 敵方上陣 ({len(self.opp_draft)}/5): {self.opp_draft}"
                                    )
                            break

        # Parse ally picks bounded by revealed opponent pick count
        if self.remembered_opp_ban and len(self.my_draft) < len(self.opp_draft):
            for text in my_texts:
                for full_name, short_name in sorted_full_to_short:
                    if full_name in text and short_name:
                        if (
                            short_name != self.remembered_opp_ban
                            and short_name != my_ban
                            and short_name not in self.my_draft
                        ):
                            self.my_draft.append(short_name)
                            self.last_draft_change_time = now_ts
                            logger.info(f"⛨ 我方上陣 ({len(self.my_draft)}/5): {self.my_draft}")
                        break

        if len(self.opp_draft) == 5:
            self.phase = DraftPhase.IN_BATTLE
            if not self.has_triggered_five_picks:
                self.has_triggered_five_picks = True
                self.five_picks_locked_time = now_ts
                just_completed_five = True

        return self._make_snapshot(just_completed_five, should_reset_battle), False

    def _make_snapshot(
        self,
        just_completed_five_picks: bool,
        should_reset_battle: bool,
    ) -> DraftSnapshot:
        return DraftSnapshot(
            phase=self.phase,
            opp_ban=self.remembered_opp_ban,
            opp_draft=list(self.opp_draft),
            my_draft=list(self.my_draft),
            just_completed_five_picks=just_completed_five_picks,
            should_reset_battle=should_reset_battle,
        )
