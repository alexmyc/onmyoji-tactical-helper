"""Regression tests for pre-draft popup guard, ActionBar portrait-to-LLM skill decision, and Red Mode loop."""

import time
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
import pytest

import onmyoji_helper.app.battle_coordinator as coord_mod
from onmyoji_helper.app.battle_coordinator import AITacticalManager
from onmyoji_helper.app.mode_controller import ModeController
from onmyoji_helper.automation.battle_executor import AutoBattleExecutor
from onmyoji_helper.domain.models import CaptureFrame, CropOffset
from onmyoji_helper.domain.public_demo_data import get_public_demo_bundle
from onmyoji_helper.domain.shikigami_repository import ShikigamiRepository
from onmyoji_helper.vision.action_bar_detector import ActionBarDetector
from onmyoji_helper.vision.turn_detector import TurnDetector


class _SyncRootStub:
    def after(self, _ms: int, fn: Callable[..., Any], *args: Any) -> str:
        fn(*args)
        return "ok"


class _StatusLabelStub:
    def __init__(self) -> None:
        self.text: str = ""

    def config(self, **kwargs: Any) -> None:
        if "text" in kwargs:
            self.text = str(kwargs["text"])


class _MainUIStub:
    def __init__(self) -> None:
        self.root = _SyncRootStub()
        self.lbl_status = _StatusLabelStub()
        self.mode_names = {"green": "綠色", "yellow": "黃色", "red": "紅色"}
        self.current_ui_mode = "green"

    def disable_mode_click(self) -> None:
        pass

    def enable_mode_click(self) -> None:
        pass

    def set_mode_ui(self, mode: str) -> None:
        self.current_ui_mode = mode


class _WindowStub:
    def __init__(self) -> None:
        self.exists = True
        self.text = ""

    def winfo_exists(self) -> bool:
        return self.exists

    def update_advice(self, text: str) -> None:
        self.text = text

    def reset_to_blank(self) -> None:
        self.text = ""

    def destroy(self) -> None:
        self.exists = False

    def close(self) -> None:
        self.exists = False

    def sync_position(self, _hwnd: int) -> None:
        pass

    def clear(self) -> None:
        pass

    def update_targets(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def update_skills(self, *_args: Any, **_kwargs: Any) -> None:
        pass


class _FakeBox:
    def __init__(self, cls_id: int, cx: float, cy: float) -> None:
        self.cls = [cls_id]
        arr = np.array([cx - 15.0, cy - 25.0, cx + 15.0, cy + 25.0], dtype=np.float32)

        class _TensorStub:
            def cpu(self) -> "_TensorStub":
                return self

            def numpy(self) -> np.ndarray:
                return arr

        self.xyxy = [_TensorStub()]


class _FakeYoloResult:
    def __init__(self, boxes: list[_FakeBox]) -> None:
        self.boxes = boxes


def _paint_portrait_pattern(canvas: np.ndarray, x: int, y: int, w: int, h: int, idx: int) -> None:
    tile = np.full((h, w, 3), 28 + idx * 14, dtype=np.uint8)
    if idx == 0:
        cv2.circle(tile, (w // 2, h // 2), min(w, h) // 3, (240, 80, 40), -1)
    elif idx == 1:
        cv2.rectangle(tile, (8, 8), (w - 8, h - 8), (40, 220, 100), 4)
        cv2.circle(tile, (w // 2, h // 2), min(w, h) // 5, (250, 250, 250), -1)
    elif idx == 2:
        pts = np.array([[w // 2, 6], [6, h - 6], [w - 6, h - 6]], dtype=np.int32)
        cv2.fillPoly(tile, [pts], (60, 125, 245))
    elif idx == 3:
        cv2.line(tile, (0, h // 2), (w, h // 2), (255, 220, 40), 6)
        cv2.line(tile, (w // 2, 0), (w // 2, h), (255, 60, 200), 6)
    else:
        for step in range(4, w, 11):
            cv2.line(tile, (step, 0), (step, h), (240, 240, 120), 3)
    canvas[y : y + h, x : x + w] = tile


def _build_draft_and_combat_frames(
    td: TurnDetector,
    abd: ActionBarDetector,
    w: int = 1182,
    h: int = 666,
) -> tuple[np.ndarray, np.ndarray]:
    # 1. Draft screen with 5 ally flip portraits at top-left (i=0 -> Slot 5 ... i=4 -> Slot 1)
    draft_frame = np.full((h, w, 3), 24, dtype=np.uint8)
    box_w, box_h = int(w * abd.pw), int(h * abd.ph)
    y_top, gap = int(h * abd.py), int(w * abd.pgap)
    for i in range(5):
        x = int(w * abd.px) + i * gap
        _paint_portrait_pattern(draft_frame, x, y_top, box_w, box_h, i)

    # 2. Combat frame: top-left has dragon statue noise (NOT portraits!), bottom-right has Slot 4 ('閻', i=1)
    combat_frame = np.full((h, w, 3), 20, dtype=np.uint8)
    cv2.rectangle(combat_frame, (int(w * 0.05), int(h * 0.01)), (int(w * 0.36), int(h * 0.12)), (180, 160, 90), -1)

    rx, ry = int(w * abd.rx), int(h * abd.ry)
    rw, rh = int(w * abd.rw), int(h * abd.rh)
    _paint_portrait_pattern(combat_frame, rx, ry, rw, rh, 1)  # i=1 corresponds to Slot 4 ('閻')

    for slot_k in ("slot_1", "slot_2", "slot_3", "slot_4"):
        rx1, ry1, rx2, ry2 = td.PHYSICAL_SLOTS[slot_k]
        cx = int(round(w * (rx1 + rx2) * 0.5))
        cy = int(round(h * (ry1 + ry2) * 0.5))
        cv2.circle(combat_frame, (cx, cy), 25, (40, 180, 245), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(combat_frame, (cx, cy), 26, (235, 245, 255), thickness=2, lineType=cv2.LINE_AA)
    return draft_frame, combat_frame


def test_pre_draft_mode_switch_does_not_pop_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    _, dict_map, multi_shorts, tags_db = get_public_demo_bundle()
    repo = ShikigamiRepository(tags_db=tags_db, full_to_short_map=dict_map, multi_char_shorts=multi_shorts)
    mgr = AITacticalManager(repository=repo)
    monkeypatch.setattr(mgr.models, "warmup_in_background", lambda: None)

    created_windows: list[str] = []
    monkeypatch.setattr(mgr.presenter, "ensure_windows", lambda **_kw: created_windows.append("opened"))

    ctrl = ModeController(mgr)
    ui_stub = _MainUIStub()
    ctrl.set_ui_app(ui_stub)

    assert ctrl.set_mode("yellow") is True
    while ctrl.is_busy:
        time.sleep(0.01)
    assert ctrl.set_mode("red") is True
    while ctrl.is_busy:
        time.sleep(0.01)

    assert created_windows == []
    assert mgr.is_active_session is False


def test_draft_portrait_lock_action_bar_to_llm_decision_and_red_mode_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, dict_map, multi_shorts, tags_db = get_public_demo_bundle()
    repo = ShikigamiRepository(tags_db=tags_db, full_to_short_map=dict_map, multi_char_shorts=multi_shorts)
    mgr = AITacticalManager(repository=repo)
    mgr.POLL_MY_TURN = 0.02
    mgr.POLL_OPP_TURN = 0.02
    mgr.POST_DECISION_BUFFER_RANGE = (0.03, 0.04)
    mgr.DRAFT_FADE_GRACE_SEC = 0.02

    clicks: list[tuple[str, int, int]] = []
    mgr.auto_executor = AutoBattleExecutor(
        min_turn_cooldown=0.06,
        retry_cooldown_range=(0.06, 0.07),
        reaction_delay_range=(0.0, 0.005),
        inter_click_delay_range=(0.0, 0.005),
        target_jitter_px=0.0,
        skill_jitter_px=0.0,
        input_backend=lambda act, x, y: (clicks.append((act, x, y)) or True),
        foreground_checker=lambda _h: True,
        client_to_screen_fn=lambda _h, ix, iy, iw, ih: (ix, iy, (0, 0, iw, ih)),
        key_checker=lambda _vk: False,
        uipi_checker=lambda _h: False,
    )

    llm_received_payloads: list[dict[str, Any]] = []

    def _fake_llm_advice(payload: dict[str, Any]) -> dict[str, Any]:
        llm_received_payloads.append(dict(payload))
        return {
            "mode": "Gemma-4",
            "chosen_skill": "S3",
            "target_slot": "Slot 1",
            "advice": "我方 [Slot 4] 閻 回合！發動 [S3大招] 壓制敵方 [Slot 1] 玲。",
        }

    monkeypatch.setattr(mgr.models, "warmup_in_background", lambda: None)
    monkeypatch.setattr(mgr.models, "ensure_loaded", lambda should_abort_fn=None: True)
    monkeypatch.setattr(mgr.models, "has_active_gpu_llm", lambda: True)
    monkeypatch.setattr(mgr.models, "run_llm_advice", _fake_llm_advice)
    monkeypatch.setattr(coord_mod.ctypes.windll.user32, "IsIconic", lambda _h: 0)
    monkeypatch.setattr(mgr, "_resolve_battle_hwnd", staticmethod(lambda _c, _p: 8888))

    w, h = 1182, 666
    draft_frame, combat_frame = _build_draft_and_combat_frames(mgr.turn_detector, mgr.action_bar_detector, w, h)

    class _CaptureStub:
        def find_windows(self) -> list[Any]:
            return []

        def capture_frame(self, hwnd: int, need_gray: bool = False, window_title: str = "") -> CaptureFrame:
            return CaptureFrame(
                bgr=combat_frame,
                gray=None,
                hwnd=hwnd,
                window_title=window_title,
                crop=CropOffset(x0=0, y0=0, client_w=w, client_h=h),
            )

    monkeypatch.setattr(coord_mod, "CaptureEngine", _CaptureStub)

    # Even with only 1 enemy + 1 ally visible (partial visibility), windows MUST pop immediately!
    monkeypatch.setattr(
        mgr.models,
        "predict_yolo",
        lambda _img: [
            _FakeYoloResult(
                [_FakeBox(0, w * 0.75, h * 0.30), _FakeBox(1, w * 0.130, h * 0.62)]
            )
        ],
    )
    mgr.turn_detector.ACTIVATION_THRESHOLD = 0.01

    def _stub_ensure_windows(**_kw: Any) -> None:
        if mgr.presenter.ai_ui is None:
            mgr.presenter.ai_ui = _WindowStub()  # type: ignore[assignment]
        if mgr.presenter.overlay_ui is None:
            mgr.presenter.overlay_ui = _WindowStub()  # type: ignore[assignment]

    monkeypatch.setattr(mgr.presenter, "ensure_windows", _stub_ensure_windows)

    my_picks = ["葛", "帝", "骷", "閻", "連"]
    opp_picks = ["玲", "SP桃", "葛", "酒吞", "季"]

    # 1. Capture & lock 5 draft portraits at 5-pick flip while still in Green Mode
    assert mgr.action_bar_detector.CONF_THRESHOLD == 0.15
    assert mgr.capture_draft_templates(draft_frame, my_picks) is True
    assert mgr.action_bar_detector.has_locked_templates is True

    ctrl = ModeController(mgr)
    ui_stub = _MainUIStub()
    ctrl.set_ui_app(ui_stub)
    ctrl.save_and_handle_five_picks(ui_stub.root, my_picks, opp_picks, draft_frame, 8888)  # type: ignore[arg-type]

    # 2. Switch Green -> Red mid-battle: locked draft portrait templates MUST NOT be wiped!
    ctrl.set_mode("red")
    while ctrl.is_busy:
        time.sleep(0.01)
    assert mgr.action_bar_detector.has_locked_templates is True

    # 3. Wait for ActionBarDetector -> LLM decision -> Skill-First + Target-Second execution + retry
    deadline = time.time() + 2.5
    while len(clicks) < 4 and time.time() < deadline:
        time.sleep(0.02)

    assert mgr.battle_has_started is True
    assert mgr.presenter.ai_ui is not None and mgr.presenter.overlay_ui is not None
    assert llm_received_payloads
    # Verify ActionBarDetector accurately matched '閻' (Slot 4) and passed it + skills to LLM!
    assert any(
        p.get("active_shikigami_slot") == "Slot 4" and p.get("active_shikigami_name") == "閻"
        for p in llm_received_payloads
    )
    assert len(clicks) >= 4
    # S3 button (slot_3 at 4-skill mode) clicked FIRST, Enemy Slot 1 clicked SECOND
    assert clicks[0][2] > int(h * 0.85) and clicks[1][2] < int(h * 0.45)

    mgr.reset_for_next_match()
    assert mgr.action_bar_detector.has_locked_templates is False
    assert mgr.presenter.ai_ui is None and mgr.presenter.overlay_ui is None


def test_opening_preemptive_pause_never_locks_until_skill_bar_my_turn_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scenario 1: Pre-armed Red/Yellow Mode ignores stationary tilted '先機' animation pauses
    (`is_my_turn=False`) and cleanly initializes `StateParser` baselines at the exact moment
    the skill bar first becomes available (`is_my_turn=True`).
    """
    _, dict_map, multi_shorts, tags_db = get_public_demo_bundle()
    repo = ShikigamiRepository(tags_db=tags_db, full_to_short_map=dict_map, multi_char_shorts=multi_shorts)
    mgr = AITacticalManager(repository=repo)
    mgr.POLL_MY_TURN = 0.015
    mgr.POLL_OPP_TURN = 0.015
    mgr.POLL_ENTRY_SETTLE = 0.015
    mgr.DRAFT_FADE_GRACE_SEC = 0.01
    mgr.PROVISIONAL_OPENING_DELAY_SEC = 3.5

    monkeypatch.setattr(mgr.models, "warmup_in_background", lambda: None)
    monkeypatch.setattr(mgr.models, "ensure_loaded", lambda should_abort_fn=None: True)
    monkeypatch.setattr(mgr.models, "has_active_gpu_llm", lambda: False)
    monkeypatch.setattr(coord_mod.ctypes.windll.user32, "IsIconic", lambda _h: 0)
    monkeypatch.setattr(mgr, "_resolve_battle_hwnd", staticmethod(lambda _c, _p: 9999))
    mgr.auto_executor = AutoBattleExecutor(
        input_backend=lambda _a, _x, _y: True,
        foreground_checker=lambda _h: True,
        client_to_screen_fn=lambda _h, ix, iy, iw, ih: (ix, iy, (0, 0, iw, ih)),
        key_checker=lambda _vk: False,
        uipi_checker=lambda _h: False,
    )
    mgr.turn_detector.ACTIVATION_THRESHOLD = 0.01

    w, h = 1280, 720
    draft_frame, active_turn_frame = _build_draft_and_combat_frames(mgr.turn_detector, mgr.action_bar_detector, w, h)
    preemptive_anim_frame = np.full((h, w, 3), 22, dtype=np.uint8)  # No skill bar during 先機 realm!

    # Canonical combat coords when skill bar opens vs. tilted 先機 realm pause (+185 px skewed!)
    canonical_opp = [(w * r, h * 0.31) for r in (0.25, 0.35, 0.45, 0.55, 0.65)]
    canonical_my = [(w * r, h * 0.63) for r in (0.130, 0.315, 0.495, 0.665, 0.835)]
    tilted_preemptive_boxes: list[_FakeBox] = [
        _FakeBox(0, x + 185.0, y - 50.0) for x, y in canonical_opp
    ] + [_FakeBox(1, x - 120.0, y + 40.0) for x, y in canonical_my]

    frame_idx_holder = [0]
    pre_turn_states: list[tuple[int, bool, bool]] = []
    overlay_snapshots: list[dict[str, dict[str, Any]]] = []

    class _CaptureStub:
        def find_windows(self) -> list[Any]:
            return []

        def capture_frame(self, hwnd: int, need_gray: bool = False, window_title: str = "") -> CaptureFrame:
            # Frames 0..3: stationary tilted 先機 animation (0 px/frame motion -> used to fool 2-frame velocity gate!)
            # Frames 4+: skill bar appears (`active_turn_frame` -> `is_my_turn == True`)!
            bgr = preemptive_anim_frame if frame_idx_holder[0] < 4 else active_turn_frame
            return CaptureFrame(
                bgr=bgr,
                gray=None,
                hwnd=hwnd,
                window_title=window_title,
                crop=CropOffset(x0=0, y0=0, client_w=w, client_h=h),
            )

    monkeypatch.setattr(coord_mod, "CaptureEngine", _CaptureStub)

    def _scripted_yolo(_img: np.ndarray) -> list[_FakeYoloResult]:
        idx = frame_idx_holder[0]
        pre_turn_states.append((idx, mgr.battle_has_started, mgr.parser.is_opp_baseline_set))
        if idx < 4:
            boxes = tilted_preemptive_boxes
        elif idx == 4 or idx == 5:
            boxes = [_FakeBox(0, x, y) for x, y in canonical_opp] + [_FakeBox(1, x, y) for x, y in canonical_my]
        else:
            # Frame 6+: occlude Enemy Slot 1 (index 3, x=0.55w) to verify locked baseline accuracy
            boxes = [_FakeBox(0, x, y) for i, (x, y) in enumerate(canonical_opp) if i != 3] + [
                _FakeBox(1, x, y) for x, y in canonical_my
            ]
        frame_idx_holder[0] += 1
        return [_FakeYoloResult(boxes)]

    monkeypatch.setattr(mgr.models, "predict_yolo", _scripted_yolo)

    def _stub_ensure_windows(**_kw: Any) -> None:
        if mgr.presenter.ai_ui is None:
            mgr.presenter.ai_ui = _WindowStub()  # type: ignore[assignment]
        if mgr.presenter.overlay_ui is None:
            mgr.presenter.overlay_ui = _WindowStub()  # type: ignore[assignment]

    monkeypatch.setattr(mgr.presenter, "ensure_windows", _stub_ensure_windows)
    orig_sched_ov = mgr.presenter.schedule_overlay_frame

    def _spy_overlay(**kwargs: Any) -> None:
        overlay_snapshots.append({k: dict(v) for k, v in kwargs["opp_status"].items()})
        orig_sched_ov(**kwargs)

    monkeypatch.setattr(mgr.presenter, "schedule_overlay_frame", _spy_overlay)

    my_picks = ["葛", "帝", "骷", "閻", "連"]
    opp_picks = ["玲", "SP桃", "葛", "酒吞", "季"]

    ctrl = ModeController(mgr)
    ui_stub = _MainUIStub()
    ctrl.set_ui_app(ui_stub)
    assert ctrl.set_mode("yellow") is True
    while ctrl.is_busy:
        time.sleep(0.01)

    assert mgr.capture_draft_templates(draft_frame, my_picks) is True
    ctrl.save_and_handle_five_picks(ui_stub.root, my_picks, opp_picks, draft_frame, 9999)  # type: ignore[arg-type]

    deadline = time.time() + 2.5
    while (frame_idx_holder[0] < 8 or len(overlay_snapshots) < 2) and time.time() < deadline:
        time.sleep(0.015)

    # Even across 4 completely motionless tilted 先機 frames (idx 0..3) + TurnDetector's 2-frame debounce (idx 4),
    # battle MUST NOT start and baselines MUST remain untouched until `is_my_turn` is confirmed (idx 5)!
    for f_idx, started_before, base_set_before in pre_turn_states:
        if f_idx <= 5:
            assert started_before is False, f"Locked prematurely during 先機 animation (f_idx={f_idx})"
            assert base_set_before is False, f"Baseline polluted during 先機 animation (f_idx={f_idx})"

    assert mgr.battle_has_started is True
    assert mgr.has_calibrated_on_my_turn is True
    assert mgr.is_provisional_baseline is False
    assert len(mgr.parser.baseline_opp) == 5
    assert abs(mgr.parser.baseline_opp["Slot 5"][0] - (w * 0.25)) <= 2.0
    assert abs(mgr.parser.baseline_opp["Slot 1"][0] - (w * 0.55)) <= 2.0
    assert abs(mgr.parser.baseline_opp["Slot 4"][0] - (w * 0.65)) <= 2.0
    assert abs(overlay_snapshots[-1]["Slot 1"]["pos"][0] - (w * 0.55)) <= 3.0

    mgr.reset_for_next_match()


def test_enemy_first_provisional_entry_forces_hard_recalibration_on_first_my_turn_and_drift_self_heal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scenario 2: When the enemy acts first (`is_my_turn=False` past opening buffer), provisional
    tracking starts, AND on the FIRST ally turn (`is_my_turn=True`) `parser.reset_baseline()` is
    forced automatically—plus subsequent `is_my_turn` 5-enemy drift self-check heals stale baselines.
    """
    _, dict_map, multi_shorts, tags_db = get_public_demo_bundle()
    repo = ShikigamiRepository(tags_db=tags_db, full_to_short_map=dict_map, multi_char_shorts=multi_shorts)
    mgr = AITacticalManager(repository=repo)
    mgr.POLL_MY_TURN = 0.015
    mgr.POLL_OPP_TURN = 0.015
    mgr.POLL_ENTRY_SETTLE = 0.015
    mgr.DRAFT_FADE_GRACE_SEC = 0.01
    mgr.PROVISIONAL_OPENING_DELAY_SEC = 0.05  # Compressed 3.5s enemy-first buffer for fast deterministic test

    monkeypatch.setattr(mgr.models, "warmup_in_background", lambda: None)
    monkeypatch.setattr(mgr.models, "ensure_loaded", lambda should_abort_fn=None: True)
    monkeypatch.setattr(mgr.models, "has_active_gpu_llm", lambda: False)
    monkeypatch.setattr(coord_mod.ctypes.windll.user32, "IsIconic", lambda _h: 0)
    monkeypatch.setattr(mgr, "_resolve_battle_hwnd", staticmethod(lambda _c, _p: 7777))
    mgr.turn_detector.ACTIVATION_THRESHOLD = 0.01

    w, h = 1280, 720
    draft_frame, active_turn_frame = _build_draft_and_combat_frames(mgr.turn_detector, mgr.action_bar_detector, w, h)
    enemy_turn_frame = np.full((h, w, 3), 20, dtype=np.uint8)

    skewed_enemy_turn_opp = [(w * r + 150.0, h * 0.27) for r in (0.25, 0.35, 0.45, 0.55, 0.65)]
    canonical_opp = [(w * r, h * 0.31) for r in (0.25, 0.35, 0.45, 0.55, 0.65)]
    canonical_my = [(w * r, h * 0.63) for r in (0.130, 0.315, 0.495, 0.665, 0.835)]

    phase = ["enemy_first"]

    class _CaptureStub:
        def find_windows(self) -> list[Any]:
            return []

        def capture_frame(self, hwnd: int, need_gray: bool = False, window_title: str = "") -> CaptureFrame:
            bgr = enemy_turn_frame if phase[0] == "enemy_first" else active_turn_frame
            return CaptureFrame(
                bgr=bgr,
                gray=None,
                hwnd=hwnd,
                window_title=window_title,
                crop=CropOffset(x0=0, y0=0, client_w=w, client_h=h),
            )

    monkeypatch.setattr(coord_mod, "CaptureEngine", _CaptureStub)

    def _scripted_yolo(_img: np.ndarray) -> list[_FakeYoloResult]:
        opp_src = skewed_enemy_turn_opp if phase[0] == "enemy_first" else canonical_opp
        boxes = [_FakeBox(0, x, y) for x, y in opp_src] + [_FakeBox(1, x, y) for x, y in canonical_my]
        return [_FakeYoloResult(boxes)]

    monkeypatch.setattr(mgr.models, "predict_yolo", _scripted_yolo)
    monkeypatch.setattr(mgr.presenter, "ensure_windows", lambda **_kw: None)

    my_picks = ["葛", "帝", "骷", "閻", "連"]
    opp_picks = ["玲", "SP桃", "葛", "酒吞", "季"]
    ctrl = ModeController(mgr)
    ui_stub = _MainUIStub()
    ctrl.set_ui_app(ui_stub)
    ctrl.set_mode("yellow")
    while ctrl.is_busy:
        time.sleep(0.01)

    ctrl.save_and_handle_five_picks(ui_stub.root, my_picks, opp_picks, draft_frame, 7777)  # type: ignore[arg-type]

    # 1. Wait for provisional enemy-first entry (has_calibrated_on_my_turn == False, skewed +150 px)
    deadline = time.time() + 1.5
    while not mgr.battle_has_started and time.time() < deadline:
        time.sleep(0.015)
    assert mgr.battle_has_started is True
    assert mgr.is_provisional_baseline is True
    assert mgr.has_calibrated_on_my_turn is False
    assert abs(mgr.parser.baseline_opp["Slot 5"][0] - (w * 0.25 + 150.0)) <= 2.0

    # 2. Switch to our first skill-bar turn (`is_my_turn=True`) -> MUST force `reset_baseline()` to canonical!
    phase[0] = "my_first_turn"
    deadline = time.time() + 1.5
    while not mgr.has_calibrated_on_my_turn and time.time() < deadline:
        time.sleep(0.015)
    time.sleep(0.04)  # Let parse_frame complete on the calibrated frame

    assert mgr.has_calibrated_on_my_turn is True
    assert mgr.is_provisional_baseline is False
    assert abs(mgr.parser.baseline_opp["Slot 5"][0] - (w * 0.25)) <= 2.0
    assert abs(mgr.parser.smooth_pos_opp["Slot 5"][0] - (w * 0.25)) <= 2.0

    # 3. Verify `is_my_turn` 5-enemy zero-drift self-check heals any injected smooth_pos_opp drift
    for s_key in mgr.parser.smooth_pos_opp:
        ox, oy = mgr.parser.smooth_pos_opp[s_key]
        mgr.parser.smooth_pos_opp[s_key] = (ox + 90.0, oy)
    assert mgr._recalibrate_opp_if_drifted_on_my_turn(canonical_opp, w) is True
    assert mgr.parser.is_opp_baseline_set is False

    mgr.reset_for_next_match()


def test_red_mode_max_turn_retries_circuit_breaker_downgrades_to_yellow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies `MAX_TURN_RETRIES = 5`: if the client freezes on `is_my_turn=True` across 1 initial cast
    + 5 retries (6 total dispatches), Red Mode halts clicking and downgrades cleanly to Yellow Mode.
    """
    _, dict_map, multi_shorts, tags_db = get_public_demo_bundle()
    repo = ShikigamiRepository(tags_db=tags_db, full_to_short_map=dict_map, multi_char_shorts=multi_shorts)
    mgr = AITacticalManager(repository=repo)
    assert mgr.MAX_TURN_RETRIES == 5

    mgr.POLL_MY_TURN = 0.01
    mgr.POLL_OPP_TURN = 0.01
    mgr.POST_DECISION_BUFFER_RANGE = (0.01, 0.015)

    clicks: list[tuple[str, int, int]] = []
    mgr.auto_executor = AutoBattleExecutor(
        min_turn_cooldown=0.02,
        retry_cooldown_range=(0.02, 0.025),
        reaction_delay_range=(0.0, 0.002),
        inter_click_delay_range=(0.0, 0.002),
        target_jitter_px=0.0,
        skill_jitter_px=0.0,
        input_backend=lambda act, x, y: (clicks.append((act, x, y)) or True),
        foreground_checker=lambda _h: True,
        client_to_screen_fn=lambda _h, ix, iy, iw, ih: (ix, iy, (0, 0, iw, ih)),
        key_checker=lambda _vk: False,
        uipi_checker=lambda _h: False,
    )

    monkeypatch.setattr(mgr.models, "warmup_in_background", lambda: None)
    monkeypatch.setattr(mgr.models, "ensure_loaded", lambda should_abort_fn=None: True)
    monkeypatch.setattr(mgr.models, "has_active_gpu_llm", lambda: False)
    monkeypatch.setattr(coord_mod.ctypes.windll.user32, "IsIconic", lambda _h: 0)
    monkeypatch.setattr(mgr, "_resolve_battle_hwnd", staticmethod(lambda _c, _p: 6666))
    mgr.turn_detector.ACTIVATION_THRESHOLD = 0.005

    w, h = 1182, 666
    draft_frame, frozen_turn_frame = _build_draft_and_combat_frames(
        mgr.turn_detector, mgr.action_bar_detector, w, h
    )

    class _CaptureStub:
        def find_windows(self) -> list[Any]:
            return []

        def capture_frame(self, hwnd: int, need_gray: bool = False, window_title: str = "") -> CaptureFrame:
            return CaptureFrame(
                bgr=frozen_turn_frame,
                gray=None,
                hwnd=hwnd,
                window_title=window_title,
                crop=CropOffset(x0=0, y0=0, client_w=w, client_h=h),
            )

    monkeypatch.setattr(coord_mod, "CaptureEngine", _CaptureStub)
    monkeypatch.setattr(
        mgr.models,
        "predict_yolo",
        lambda _img: [_FakeYoloResult([_FakeBox(0, w * 0.55, h * 0.30), _FakeBox(1, w * 0.315, h * 0.63)])],
    )
    monkeypatch.setattr(mgr.presenter, "ensure_windows", lambda **_kw: None)

    my_picks = ["葛", "帝", "骷", "閻", "連"]
    opp_picks = ["玲", "SP桃", "葛", "酒吞", "季"]
    mgr.capture_draft_templates(draft_frame, my_picks)

    ctrl = ModeController(mgr)
    ui_stub = _MainUIStub()
    ctrl.set_ui_app(ui_stub)
    ctrl.save_and_handle_five_picks(ui_stub.root, my_picks, opp_picks, draft_frame, 6666)  # type: ignore[arg-type]
    ctrl.set_mode("red")
    while ctrl.is_busy:
        time.sleep(0.01)

    deadline = time.time() + 2.5
    while ctrl.current_mode == "red" and time.time() < deadline:
        time.sleep(0.015)

    # 1 initial cast + 5 retries = 6 dispatches * 2 clicks ([skill, target]) = 12 clicks max, then auto-downgrade!
    assert ctrl.current_mode == "yellow"
    assert mgr.auto_battle_enabled is False
    assert len(clicks) == (1 + mgr.MAX_TURN_RETRIES) * 2
    assert "重試達上限" in ui_stub.lbl_status.text

    mgr.reset_for_next_match()
