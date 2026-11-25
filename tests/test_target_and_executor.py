"""Unit tests for TargetSelector, AutoBattleExecutor (single-target + AoE 2-step clicks), and SafetyGuard."""

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


def _build_test_executor(
    clicks: list[tuple[str, int, int]],
    uipi_blocked: bool = False,
    pressed_vk: int = 0,
) -> AutoBattleExecutor:
    return AutoBattleExecutor(
        min_turn_cooldown=2.2,
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


def test_enemy_aoe_and_ally_aoe_execute_two_step_clicks() -> None:
    td = TurnDetector()
    skill_map = {"s1": "slot_2", "s2": "slot_3", "s3": "slot_4", "s4": None}
    avail = {"s1": True, "s2": True, "s3": True, "s4": False}
    my_status = {
        "Slot 1": {"name": "追月神", "alive": True, "pos": (350, 500)},
        "Slot 2": {"name": "神啟荒", "alive": True, "pos": (550, 500)},
    }
    opp_status = {
        "Slot 1": {"name": "帝釋天", "alive": False, "pos": (0, 0)},
        "Slot 2": {"name": "神啟荒", "alive": True, "pos": (740, 260)},
    }

    # 1. Enemy AoE with Slot 1 dead -> automatically picks alive Enemy Slot 2, then clicks S3
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
    assert len(en_clicks) == 2
    assert en_clicks[0] == ("click", 740, 260)
    assert "敵方全體" in msg_en and "敵方 Slot 2" in msg_en

    # 2. Ally AoE -> clicks alive Ally Slot 1, then clicks S3
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
    assert len(al_clicks) == 2
    assert al_clicks[0] == ("click", 350, 500)
    assert "我方全體" in msg_al and "我方 Slot 1" in msg_al


def test_orb_shortage_downgrades_to_s1_and_engages_cooldown_lock() -> None:
    td = TurnDetector()
    skill_map = {"s1": "slot_2", "s2": "slot_3", "s3": "slot_4", "s4": None}
    no_orb_avail = {"s1": True, "s2": False, "s3": False, "s4": False}
    my_status = {"Slot 1": {"name": "追月神", "alive": True, "pos": (350, 500)}}
    opp_status = {"Slot 1": {"name": "帝釋天", "alive": True, "pos": (850, 260)}}

    clicks: list[tuple[str, int, int]] = []
    ex = _build_test_executor(clicks)
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
    # Downgraded from Ally AoE S3 -> Enemy Single-Target S1!
    assert clicks[0] == ("click", 850, 260)
    assert "缺火/冷卻降級S1" in msg

    # Immediate repeat call within 2.2s cooldown must be blocked
    ok_repeat, _ = ex.execute_turn_action(
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
    assert ok_repeat is False
    assert len(clicks) == 2


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
    assert res["target_slot"] == "Slot 2"  # Control priority 1 beats DPS priority 2

    # UIPI warning check
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

    # F12 emergency brake check (waits for daemon monitor thread callback completion)
    reasons: list[str] = []
    f12_ex = _build_test_executor([], pressed_vk=VK_F12)
    f12_ex.start_kill_switch_monitor(on_emergency_stop=reasons.append)
    if f12_ex.safety._kill_monitor_thread is not None:
        f12_ex.safety._kill_monitor_thread.join(timeout=0.5)
    assert f12_ex.poll_kill_switch_once() is True
    assert f12_ex.emergency_stopped is True
    assert reasons and "F12" in reasons[0]
    f12_ex.stop_immediately()
