"""Unit tests for the pure deterministic Ban/Pick DraftStateMachine."""

from onmyoji_helper.decision.draft_fsm import DraftStateMachine
from onmyoji_helper.domain.models import DraftPhase

SORTED_DICT = [
    ("封陽君", "封"),
    ("不知火", "離"),
    ("荒骷髏", "骷"),
    ("大夜摩天閻魔", "SP閻"),
    ("鈴鹿御前", "06"),
    ("須佐之男", "須"),
    ("八岐大蛇", "蛇"),
]


def _tok(*texts: str) -> list[tuple[list[list[int]], str, float]]:
    return [([[0, 0], [10, 0], [10, 10], [0, 10]], t, 0.98) for t in texts]


def test_ban_hard_gate_blocks_picks_before_ban_locked() -> None:
    fsm = DraftStateMachine()
    # Character names appearing before any Ban keyword must NOT be recorded as picks
    snap, skip = fsm.step(_tok("不知火鈴鹿御前"), _tok("須佐之男"), SORTED_DICT, my_ban="禪", now=100.0)
    assert not skip
    assert snap.opp_ban is None
    assert snap.opp_draft == []
    assert snap.my_draft == []

    # Entering Ban countdown switches to BAN_WAITING (0.9s fast interval)
    snap2, _ = fsm.step(_tok("禁選式神15對方禁中"), [], SORTED_DICT, my_ban="禪", now=101.0)
    assert snap2.phase == DraftPhase.BAN_WAITING
    assert fsm.get_recommended_poll_interval() == 0.9


def test_full_five_pick_progression_and_settlement_reset() -> None:
    fsm = DraftStateMachine()
    # 1. Lock opponent Ban
    snap, _ = fsm.step(_tok("禁選封陽君"), [], SORTED_DICT, my_ban="禪", now=10.0)
    assert snap.opp_ban == "封"
    assert snap.phase == DraftPhase.ACTIVE_DRAFT

    # 2. Tooltip '不能上' skips UI refresh without corrupting picks
    _, skip = fsm.step(_tok("不能上不知火"), [], SORTED_DICT, my_ban="禪", now=11.0)
    assert skip is True
    assert fsm.opp_draft == []

    # 3. Draft 1..5 picks (banned '封' and my_ban '禪' are strictly ignored)
    picks_opp = ["不知火", "鈴鹿御前", "須佐之男", "大夜摩天閻魔", "八岐大蛇"]
    picks_my = ["須佐之男", "不知火", "荒骷髏", "鈴鹿御前", "八岐大蛇"]
    for idx in range(5):
        snap, skip = fsm.step(
            _tok(*picks_opp[: idx + 1]),
            _tok(*picks_my[: idx + 1]),
            SORTED_DICT,
            my_ban="禪",
            now=12.0 + idx,
        )
        assert not skip
        if idx < 4:
            assert not snap.just_completed_five_picks
        else:
            assert snap.just_completed_five_picks is True
            assert snap.phase == DraftPhase.IN_BATTLE
            assert len(snap.opp_draft) == 5
            assert len(snap.my_draft) == 5

    # 4. Victory settlement screen cleanly resets draft & requests battle coordinator reset
    end_snap, _ = fsm.step(_tok("戰鬥勝利"), [], SORTED_DICT, my_ban="禪", now=60.0)
    assert end_snap.phase == DraftPhase.SLEEP
    assert end_snap.opp_ban is None
    assert end_snap.opp_draft == []
    assert end_snap.should_reset_battle is True
