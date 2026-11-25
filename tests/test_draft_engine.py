"""Unit tests for the XMind tree-structured draft recommendation & opponent prediction engine."""

from onmyoji_helper.decision.draft_engine import (
    build_database,
    get_recommendation,
    parse_draft_string,
)

SAMPLE_MULTI_SHORTS = ["SP06", "酒吞", "百羽", "SP熊", "SP吸"]

SAMPLE_RAW_PLAYBOOK = """
Ban封
離神葛連荒 神SP06石閻SP熊
離百羽酒吞SP吸貓 神SP06石連SP熊
Ban須
千連SP06吞荒 離吞千鐮鈴
"""


def test_parse_draft_string_multi_char_tokens() -> None:
    tokens = parse_draft_string("離百羽酒吞SP06SP熊", SAMPLE_MULTI_SHORTS)
    assert tokens == ["離", "百羽", "酒吞", "SP06", "SP熊"]


def test_cold_start_turn_one_excludes_banned_cards() -> None:
    db = build_database(SAMPLE_RAW_PLAYBOOK, SAMPLE_MULTI_SHORTS)
    my_recs, opp_preds, status = get_recommendation(
        db=db,
        current_ban="Ban封",
        opp_draft=[],
        my_draft=[],
        my_ban="禪",
    )
    assert my_recs[0] == "離"
    assert "封" not in my_recs and "禪" not in my_recs
    assert "封" not in opp_preds and "禪" not in opp_preds
    assert "Ban封" in status


def test_ordered_backbone_and_unordered_opponent_set_matching() -> None:
    db = build_database(SAMPLE_RAW_PLAYBOOK, SAMPLE_MULTI_SHORTS)
    # Opponent picks revealed in reversed order ('石', 'SP06', '神') must match identically to ('神', 'SP06', '石')
    recs_a, preds_a, _ = get_recommendation(
        db=db,
        current_ban="Ban封",
        opp_draft=["神", "SP06", "石"],
        my_draft=["離", "神"],
        my_ban="禪",
    )
    recs_b, preds_b, _ = get_recommendation(
        db=db,
        current_ban="Ban封",
        opp_draft=["石", "SP06", "神"],
        my_draft=["離", "神"],
        my_ban="禪",
    )
    assert recs_a == recs_b
    assert preds_a == preds_b
    # Strict availability mask: already picked 'SP06' & '神' cannot be recommended again; next in chain is '葛'
    assert recs_a[0] == "葛"
    for taken in ("封", "禪", "離", "神", "SP06", "石"):
        assert taken not in recs_a
        assert taken not in preds_a


def test_dead_branch_penalty_pivots_when_core_card_is_stolen() -> None:
    raw = """
Ban封
離葛連吞荒 離石閻熊貓
離百羽酒吞SP06連 離石閻熊葛
"""
    db = build_database(raw, SAMPLE_MULTI_SHORTS)
    # Opponent steals '葛' and '吞', crippling branch 1 while branch 2 remains 100% viable
    my_recs, _, _ = get_recommendation(
        db=db,
        current_ban="Ban封",
        opp_draft=["離", "葛", "吞"],
        my_draft=["離"],
        my_ban="禪",
    )
    assert my_recs[0] == "百羽"


def test_mirror_copycat_detection_against_our_own_backbone() -> None:
    db = build_database(SAMPLE_RAW_PLAYBOOK, SAMPLE_MULTI_SHORTS)
    # Opponent copies our own signature backbone ['離', '百羽', '酒吞']
    _, opp_preds, status = get_recommendation(
        db=db,
        current_ban="Ban封",
        opp_draft=["離", "百羽", "酒吞"],
        my_draft=["千"],
        my_ban="封",
    )
    assert "SP吸" in opp_preds
    assert "鏡像/內戰" in status
