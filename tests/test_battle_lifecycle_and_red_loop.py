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


def test_pre_armed_red_mode_waits_for_camera_settle_without_baseline_pollution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end verification of the Settled Battlefield Entry Gate:
    Pre-arming Red/Yellow Mode during Ban/Pick -> surviving VS banner + high-speed camera swoop
    frames -> locking clean `StateParser` baselines ONLY after units land and stabilize.
    """
    _, dict_map, multi_shorts, tags_db = get_public_demo_bundle()
    repo = ShikigamiRepository(tags_db=tags_db, full_to_short_map=dict_map, multi_char_shorts=multi_shorts)
    mgr = AITacticalManager(repository=repo)
    mgr.POLL_MY_TURN = 0.015
    mgr.POLL_OPP_TURN = 0.015
    mgr.POLL_ENTRY_SETTLE = 0.015
    mgr.DRAFT_FADE_GRACE_SEC = 0.01

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

    w, h = 1280, 720
    draft_frame, _ = _build_draft_and_combat_frames(mgr.turn_detector, mgr.action_bar_detector, w, h)
    passive_combat_frame = np.full((h, w, 3), 22, dtype=np.uint8)

    class _CaptureStub:
        def find_windows(self) -> list[Any]:
            return []

        def capture_frame(self, hwnd: int, need_gray: bool = False, window_title: str = "") -> CaptureFrame:
            return CaptureFrame(
                bgr=passive_combat_frame,
                gray=None,
                hwnd=hwnd,
                window_title=window_title,
                crop=CropOffset(x0=0, y0=0, client_w=w, client_h=h),
            )

    monkeypatch.setattr(coord_mod, "CaptureEngine", _CaptureStub)

    # True landed combat formation (e.g. shifted combat camera x = 0.25..0.65 for enemies)
    landed_opp = [(w * r, h * 0.31) for r in (0.25, 0.35, 0.45, 0.55, 0.65)]
    landed_my = [(w * r, h * 0.63) for r in (0.130, 0.315, 0.495, 0.665, 0.835)]

    # Scripted sequence of YOLO frames:
    # Frame 0: VS banner with 1 spurious silhouette at (0.88w, 0.18h)
    # Frame 1: High-speed camera swoop (+190 px offset!)
    # Frame 2: Mid-swoop (+95 px offset -> 95 px inter-frame jump > 0.018*w!)
    # Frame 3: Units touch down at true landed coords (95 px jump from Frame 2 -> streak=1, NOT opened yet!)
    # Frame 4: Units settled at true landed coords (±2 px idle breathing -> streak=2 -> LOCK & OPEN!)
    # Frame 5+: Enemy Slot 1 (index 3, x=0.55w) occluded by opening VFX -> must predict landed x=0.55w, NOT swoop!
    scripted_frames: list[list[_FakeBox]] = [
        [_FakeBox(0, w * 0.88, h * 0.18)],
        [_FakeBox(0, x + 190.0, y - 55.0) for x, y in landed_opp]
        + [_FakeBox(1, x - 140.0, y + 45.0) for x, y in landed_my],
        [_FakeBox(0, x + 95.0, y - 28.0) for x, y in landed_opp]
        + [_FakeBox(1, x - 70.0, y + 22.0) for x, y in landed_my],
        [_FakeBox(0, x, y) for x, y in landed_opp] + [_FakeBox(1, x, y) for x, y in landed_my],
        [_FakeBox(0, x + 2.0, y - 1.0) for x, y in landed_opp]
        + [_FakeBox(1, x - 1.5, y + 1.0) for x, y in landed_my],
        [_FakeBox(0, x, y) for idx, (x, y) in enumerate(landed_opp) if idx != 3]
        + [_FakeBox(1, x, y) for x, y in landed_my],
    ]

    frame_cursor = [0]
    gate_snapshots: list[tuple[int, bool, bool]] = []
    overlay_snapshots: list[dict[str, dict[str, Any]]] = []

    def _scripted_yolo(_img: np.ndarray) -> list[_FakeYoloResult]:
        idx = frame_cursor[0]
        gate_snapshots.append((idx, mgr.battle_has_started, mgr.parser.is_opp_baseline_set))
        boxes = scripted_frames[idx] if idx < len(scripted_frames) else scripted_frames[-1]
        frame_cursor[0] += 1
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

    # 1. User pre-arms Red Mode BEFORE 5-pick lock (during Ban/Pick)
    ctrl = ModeController(mgr)
    ui_stub = _MainUIStub()
    ctrl.set_ui_app(ui_stub)
    assert ctrl.set_mode("red") is True
    while ctrl.is_busy:
        time.sleep(0.01)
    assert mgr.battle_has_started is False

    # 2. Pick 5 locks -> enters transition (VS banner -> swoop -> landing)
    assert mgr.capture_draft_templates(draft_frame, my_picks) is True
    ctrl.save_and_handle_five_picks(ui_stub.root, my_picks, opp_picks, draft_frame, 9999)  # type: ignore[arg-type]

    deadline = time.time() + 2.5
    while (frame_cursor[0] < 7 or len(overlay_snapshots) < 2) and time.time() < deadline:
        time.sleep(0.015)

    # Verify Frames 0..4 (before Frame 4 finished processing) NEVER started battle or polluted baselines!
    for f_idx, started_before_frame, base_set_before_frame in gate_snapshots:
        if f_idx <= 4:
            assert started_before_frame is False, f"Premature battle start before settled frame (f_idx={f_idx})"
            assert base_set_before_frame is False, f"Baseline polluted during VS/swoop frame (f_idx={f_idx})"

    assert mgr.battle_has_started is True
    assert mgr.parser.is_opp_baseline_set is True and mgr.parser.is_my_baseline_set is True

    # Baseline for Enemy Slot 5 (leftmost, 0.25w) and Slot 1 (0.55w) MUST match landed coords, NOT +190px swoop!
    assert abs(mgr.parser.baseline_opp["Slot 5"][0] - (w * 0.25)) <= 4.0
    assert abs(mgr.parser.baseline_opp["Slot 1"][0] - (w * 0.55)) <= 4.0
    # Even when Slot 1 is occluded on Frame 5+, affine prediction holds exact landed x=0.55w!
    latest_opp = overlay_snapshots[-1]
    assert latest_opp["Slot 1"]["alive"] is True
    assert abs(latest_opp["Slot 1"]["pos"][0] - (w * 0.55)) <= 5.0

    # 3. Switching Red -> Green clears SpatialTracker baselines cleanly while keeping locked draft templates
    assert ctrl.set_mode("green") is True
    while ctrl.is_busy:
        time.sleep(0.01)
    assert mgr.battle_has_started is False
    assert mgr.parser.is_opp_baseline_set is False and mgr.parser.baseline_opp == {}
    assert mgr.action_bar_detector.has_locked_templates is True

    mgr.reset_for_next_match()


def test_settled_gate_stealth_fallback_prevents_deadlock() -> None:
    """Verifies anti-deadlock fallback (`>= 4.5s`) when only one camp is visible on a settled screen."""
    mgr = AITacticalManager()
    w = 1280
    only_my = [(w * 0.315, 450.0), (w * 0.495, 460.0)]

    # Frame 1 & Frame 2 at t=1.0s, 1.3s (< 4.5s fallback): single-camp only -> waits for enemy camp
    assert mgr._check_battlefield_settled([], only_my, is_my_turn=False, screen_w=w, now=1.0) is False
    assert mgr._check_battlefield_settled([], only_my, is_my_turn=False, screen_w=w, now=1.3) is False

    # Once >= 4.5s has elapsed since first unit seen (t=5.6s) and frames remain motion-settled -> enters!
    assert mgr._check_battlefield_settled([], only_my, is_my_turn=False, screen_w=w, now=5.6) is True
