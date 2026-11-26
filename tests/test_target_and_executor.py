"""Unit tests for TargetSelector, Skill-First AutoBattleExecutor, Unfinished-Turn Retries, and SafetyGuard."""

import time

from onmyoji_helper.automation.battle_executor import AutoBattleExecutor
from onmyoji_helper.decision.target_selector import TargetSelector
from onmyoji_helper.domain.shikigami_repository import ShikigamiRepository
from onmyoji_helper.infra.win32_input import VK_F12
from onmyoji_helper.vision.turn_detector import TurnDetector

MOCK_TAGS = {
    "神啟荒": {
        "role": "輸出",
        "priority": 2,
        "suggested_skill": "S3",
        "skill_details": {"S3": {"scope": "敵方全體"}},
    },
    "追月神": {
        "role": "供火",
        "priority": 3,
        "suggested_skill": "S3",
        "skill_details": {"S3": {"scope": "我方全體"}},
    },
    "帝釋天": {
        "role": "控制",
        "priority": 1,
        "suggested_skill": "S2",
        "skill_details": {"S2": {"scope": "敵方單體", "target_role_pref": ["控制", "輸出"]}},
    },
}


def _expected_skill_px(td: TurnDetector, slot_key: str, w: int = 1136, h: int = 640) -> tuple[int, int]:
    rx1, ry1, rx2, ry2 = td.PHYSICAL_SLOTS[slot_key]
    return int(round(((rx1 + rx2) * 0.5) * w)), int(round(((ry1 + ry2) * 0.5) * h))


def _build_test_executor(
    clicks: list[tuple[str, int, int]],
    uipi_blocked: bool = False,
    pressed_vk: int = 0,
    min_turn_cooldown: float = 1.2,
    retry_cooldown_range: tuple[float, float] | None = (1.2, 1.8),
) -> AutoBattleExecutor:
    return AutoBattleExecutor(
        min_turn_cooldown=min_turn_cooldown,
        retry_cooldown_range=retry_cooldown_range,
        reaction_delay_range=(0.0, 0.002),
        inter_click_delay_range=(0.0, 0.002),
        target_jitter_px=0.0,
        skill_jitter_px=0.0,
        input_backend=lambda act, x, y: (clicks.append((act, x, y)) or True),
        foreground_checker=lambda _h: True,
        client_to_screen_fn=lambda _h, ix, iy, w, hg: (ix, iy, (0, 0, w, hg)),
        key_checker=lambda vk: (vk == pressed_vk),
        uipi_checker=lambda _h: uipi_blocked,
    )


def test_skill_first_target_second_order_across_single_and_aoe_scopes() -> None:
    td = TurnDetector()
    skill_map = {"s1": "slot_2", "s2": "slot_3", "s3": "slot_4", "s4": None}
    avail = {"s1": True, "s2": True, "s3": True, "s4": False}
    s2_xy = _expected_skill_px(td, "slot_3")
    s3_xy = _expected_skill_px(td, "slot_4")

    my_status = {
        "Slot 1": {"name": "追月神", "alive": True, "pos": (350, 500)},
        "Slot 2": {"name": "神啟荒", "alive": True, "pos": (550, 500)},
    }
    opp_status = {
        "Slot 1": {"name": "帝釋天", "alive": False, "pos": (0, 0)},
        "Slot 2": {"name": "神啟荒", "alive": True, "pos": (740, 260)},
    }

    # 1. Single-Target: MUST click S2 button FIRST, then Enemy Slot 2 SECOND
    st_clicks: list[tuple[str, int, int]] = []
    ex_st = _build_test_executor(st_clicks)
    ok_st, _ = ex_st.execute_turn_action(
        hwnd=999,
        img_w=1136,
        img_h=640,
        active_shiki_name="帝釋天",
        chosen_skill="S2",
        scope="敵方單體",
        target_slot="Slot 2",
        target_name="神啟荒",
        target_is_enemy=True,
        available_skills=avail,
        skill_mapping=skill_map,
        physical_slots=td.PHYSICAL_SLOTS,
        my_status=my_status,
        opp_status=opp_status,
    )
    assert ok_st is True
    assert st_clicks == [("click", s2_xy[0], s2_xy[1]), ("click", 740, 260)]

    # 2. Enemy AoE (Slot 1 dead -> resolves alive Enemy Slot 2): clicks S3 FIRST, then Enemy Slot 2 SECOND
    en_clicks: list[tuple[str, int, int]] = []
    ex_en = _build_test_executor(en_clicks)
    ok_en, msg_en = ex_en.execute_turn_action(
        hwnd=999,
        img_w=1136,
        img_h=640,
        active_shiki_name="神啟荒",
        chosen_skill="S3",
        scope="敵方全體",
        target_slot=None,
        target_name="敵方全體",
        target_is_enemy=True,
        available_skills=avail,
        skill_mapping=skill_map,
        physical_slots=td.PHYSICAL_SLOTS,
        my_status=my_status,
        opp_status=opp_status,
    )
    assert ok_en is True
    assert en_clicks == [("click", s3_xy[0], s3_xy[1]), ("click", 740, 260)]
    assert "敵方全體" in msg_en and "敵方 Slot 2" in msg_en

    # 3. Ally AoE: clicks S3 FIRST, then alive Ally Slot 1 SECOND
    al_clicks: list[tuple[str, int, int]] = []
    ex_al = _build_test_executor(al_clicks)
    ok_al, msg_al = ex_al.execute_turn_action(
        hwnd=999,
        img_w=1136,
        img_h=640,
        active_shiki_name="追月神",
        chosen_skill="S3",
        scope="我方全體",
        target_slot=None,
        target_name="我方全體",
        target_is_enemy=False,
        available_skills=avail,
        skill_mapping=skill_map,
        physical_slots=td.PHYSICAL_SLOTS,
        my_status=my_status,
        opp_status=opp_status,
    )
    assert ok_al is True
    assert al_clicks == [("click", s3_xy[0], s3_xy[1]), ("click", 350, 500)]
    assert "我方全體" in msg_al and "我方 Slot 1" in msg_al


def test_orb_shortage_s1_downgrade_and_unfinished_turn_retry_loop() -> None:
    td = TurnDetector()
    skill_map = {"s1": "slot_2", "s2": "slot_3", "s3": "slot_4", "s4": None}
    s1_xy = _expected_skill_px(td, "slot_2")
    no_orb_avail = {"s1": True, "s2": False, "s3": False, "s4": False}
    my_status = {"Slot 1": {"name": "追月神", "alive": True, "pos": (350, 500)}}
    opp_status = {"Slot 1": {"name": "帝釋天", "alive": True, "pos": (850, 260)}}

    clicks: list[tuple[str, int, int]] = []
    ex = _build_test_executor(clicks, min_turn_cooldown=0.05, retry_cooldown_range=(0.05, 0.06))
    ok, msg = ex.execute_turn_action(
        hwnd=999,
        img_w=1136,
        img_h=640,
        active_shiki_name="追月神",
        chosen_skill="S3",
        scope="我方全體",
        target_slot=None,
        target_name="我方全體",
        target_is_enemy=False,
        available_skills=no_orb_avail,
        skill_mapping=skill_map,
        physical_slots=td.PHYSICAL_SLOTS,
        my_status=my_status,
        opp_status=opp_status,
    )
    assert ok is True
    # Downgraded from Ally AoE S3 -> Enemy Single-Target S1: S1 clicked FIRST, Enemy Slot 1 SECOND!
    assert clicks == [("click", s1_xy[0], s1_xy[1]), ("click", 850, 260)]
    assert "缺火/冷卻降級S1" in msg
    assert ex.attempts_in_current_turn == 1

    # Immediate repeat call before retry window expires is blocked
    ok_immediate, _ = ex.execute_turn_action(
        hwnd=999,
        img_w=1136,
        img_h=640,
        active_shiki_name="追月神",
        chosen_skill="S1",
        scope="敵方單體",
        target_slot="Slot 1",
        target_name="帝釋天",
        target_is_enemy=True,
        available_skills=no_orb_avail,
        skill_mapping=skill_map,
        physical_slots=td.PHYSICAL_SLOTS,
        my_status=my_status,
        opp_status=opp_status,
    )
    assert ok_immediate is False
    assert len(clicks) == 2

    # Once retry interval elapses and TurnDetector still reports is_my_turn=True, executor retries automatically!
    time.sleep(0.07)
    ok_retry, retry_msg = ex.execute_turn_action(
        hwnd=999,
        img_w=1136,
        img_h=640,
        active_shiki_name="追月神",
        chosen_skill="S1",
        scope="敵方單體",
        target_slot="Slot 1",
        target_name="帝釋天",
        target_is_enemy=True,
        available_skills=no_orb_avail,
        skill_mapping=skill_map,
        physical_slots=td.PHYSICAL_SLOTS,
        my_status=my_status,
        opp_status=opp_status,
    )
    assert ok_retry is True
    assert len(clicks) == 4
    assert "重試#1" in retry_msg
    assert ex.attempts_in_current_turn == 2

    # Observing turn end (is_my_turn=False) resets retry counter back to 0
    ex.observe_turn_state(False)
    assert ex.attempts_in_current_turn == 0


def test_f12_kill_switch_and_uipi_warning_and_repository_lookup() -> None:
    repo = ShikigamiRepository(tags_db=MOCK_TAGS)
    selector = TargetSelector(repo=repo)
    res = selector.select_target(
        active_shiki_name="帝釋天",
        active_slot="Slot 1",
        skill_states={"s1": True, "s2": True, "s3": False, "s4": False},
        my_status={"Slot 1": {"name": "帝釋天", "alive": True, "pos": (300, 500)}},
        opp_status={
            "Slot 1": {"name": "神啟荒", "alive": True, "pos": (850, 260)},
            "Slot 2": {"name": "帝釋天", "alive": True, "pos": (660, 260)},
        },
        tags_db=repo,
        is_my_turn=True,
    )
    assert res["chosen_skill"] == "S2"
    assert res["target_slot"] == "Slot 2"

    uipi_ex = _build_test_executor([], uipi_blocked=True)
    ok_u, u_msg = uipi_ex.execute_turn_action(
        hwnd=999,
        img_w=1136,
        img_h=640,
        active_shiki_name="帝釋天",
        chosen_skill="S2",
        scope="敵方單體",
        target_slot="Slot 2",
        target_name="帝釋天",
        target_is_enemy=True,
        available_skills={"s1": True, "s2": True},
        skill_mapping={"s1": "slot_2", "s2": "slot_3"},
        physical_slots=TurnDetector().PHYSICAL_SLOTS,
        my_status={},
        opp_status={},
    )
    assert not ok_u and "系統管理員" in u_msg

    reasons: list[str] = []
    f12_ex = _build_test_executor([], pressed_vk=VK_F12)
    f12_ex.start_kill_switch_monitor(on_emergency_stop=reasons.append)
    if f12_ex.safety._kill_monitor_thread is not None:
        f12_ex.safety._kill_monitor_thread.join(timeout=0.5)
    assert f12_ex.poll_kill_switch_once() is True
    assert f12_ex.emergency_stopped is True
    assert reasons and "F12" in reasons[0]
    f12_ex.stop_immediately()
