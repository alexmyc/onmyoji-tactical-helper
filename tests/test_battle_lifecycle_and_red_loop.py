"""Regression tests for strict battle lifecycle visibility gating, dark-frame viewport lock, and Red Mode loop."""

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


def _build_turn_frame(td: TurnDetector, w: int = 1182, h: int = 666, lit_bar: bool = True) -> np.ndarray:
    frame = np.full((h, w, 3), (20, 22, 26), dtype=np.uint8)
    if not lit_bar:
        return frame
    for slot_k in ("slot_1", "slot_2", "slot_3", "slot_4"):
        rx1, ry1, rx2, ry2 = td.PHYSICAL_SLOTS[slot_k]
        cx = int(round(w * (rx1 + rx2) * 0.5))
        cy = int(round(h * (ry1 + ry2) * 0.5))
        cv2.circle(frame, (cx, cy), 25, (40, 180, 245), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 26, (235, 245, 255), thickness=2, lineType=cv2.LINE_AA)
    return frame


def test_strict_visibility_gate_keeps_windows_hidden_before_five_pick_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, dict_map, multi_shorts, tags_db = get_public_demo_bundle()
    repo = ShikigamiRepository(tags_db=tags_db, full_to_short_map=dict_map, multi_char_shorts=multi_shorts)
    mgr = AITacticalManager(repository=repo)
    monkeypatch.setattr(mgr.models, "warmup_in_background", lambda: None)

    created_windows: list[str] = []
    monkeypatch.setattr(
        mgr.presenter,
        "ensure_windows",
        lambda **_kw: created_windows.append("opened"),
    )

    ctrl = ModeController(mgr)
    ui_stub = _MainUIStub()
    ctrl.set_ui_app(ui_stub)

    # Toggle Green -> Yellow -> Red BEFORE any 5-pick draft lock (saved_draft_data is None)
    assert ctrl.set_mode("yellow") is True
    deadline = time.time() + 1.0
    while ctrl.is_busy and time.time() < deadline:
        time.sleep(0.01)

    assert ctrl.set_mode("red") is True
    deadline = time.time() + 1.0
    while ctrl.is_busy and time.time() < deadline:
        time.sleep(0.01)

    # Strict regression assertions: ZERO battle windows popped, ZERO premature battle thread!
    assert created_windows == []
    assert mgr.is_active_session is False
    assert mgr.battle_has_started is False
    assert mgr.parser.is_opp_baseline_set is False
    assert "待 5 手鎖陣" in ui_stub.lbl_status.text


def test_red_mode_survives_vs_banner_and_executes_skill_first_with_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, dict_map, multi_shorts, tags_db = get_public_demo_bundle()
    repo = ShikigamiRepository(tags_db=tags_db, full_to_short_map=dict_map, multi_char_shorts=multi_shorts)
    mgr = AITacticalManager(repository=repo)
    mgr.POLL_MY_TURN = 0.02
    mgr.POLL_OPP_TURN = 0.02
    mgr.POST_DECISION_BUFFER_RANGE = (0.04, 0.05)

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

    monkeypatch.setattr(mgr.models, "warmup_in_background", lambda: None)
    monkeypatch.setattr(mgr.models, "ensure_loaded", lambda should_abort_fn=None: True)
    monkeypatch.setattr(mgr.models, "has_active_gpu_llm", lambda: False)
    monkeypatch.setattr(
        mgr.models,
        "run_llm_advice",
        lambda _p: {"advice": "鎖定核心目標施放技能", "mode": "Rule", "target_slot": "Slot 1"},
    )
    monkeypatch.setattr(coord_mod.ctypes.windll.user32, "IsIconic", lambda _h: 0)
    monkeypatch.setattr(mgr, "_resolve_battle_hwnd", staticmethod(lambda _c, _p: 8888))

    w, h = 1182, 666
    vs_frame = _build_turn_frame(mgr.turn_detector, w, h, lit_bar=False)
    combat_frame = _build_turn_frame(mgr.turn_detector, w, h, lit_bar=True)

    # Phase 0..2: VS Banner (0 allies, 5 enemy-class boxes — exact repro of log line 176!)
    # Phase 3+: Real 5v5 arena (5 allies + 5 enemies + active turn skill bar)
    frame_idx = [0]

    class _CaptureStub:
        def find_windows(self) -> list[Any]:
            return []

        def capture_frame(self, hwnd: int, need_gray: bool = False, window_title: str = "") -> CaptureFrame:
            idx = frame_idx[0]
            frame_idx[0] += 1
            bgr = vs_frame if idx < 3 else combat_frame
            return CaptureFrame(
                bgr=bgr,
                gray=None,
                hwnd=hwnd,
                window_title=window_title,
                crop=CropOffset(x0=0, y0=0, client_w=w, client_h=h),
            )

    monkeypatch.setattr(coord_mod, "CaptureEngine", _CaptureStub)

    def _fake_yolo(_img: np.ndarray) -> list[_FakeYoloResult]:
        if frame_idx[0] <= 3:
            # VS Banner: 5 enemy boxes, ZERO ally boxes -> MUST NOT start battle or pollute baseline!
            return [_FakeYoloResult([_FakeBox(0, w * r, h * 0.30) for r in (0.48, 0.57, 0.66, 0.75, 0.84)])]
        boxes = [_FakeBox(0, w * r, h * 0.30) for r in (0.48, 0.57, 0.66, 0.75, 0.84)] + [
            _FakeBox(1, w * r, h * 0.62) for r in (0.130, 0.315, 0.495, 0.665, 0.835)
        ]
        return [_FakeYoloResult(boxes)]

    monkeypatch.setattr(mgr.models, "predict_yolo", _fake_yolo)
    mgr.turn_detector.ACTIVATION_THRESHOLD = 0.01

    def _stub_ensure_windows(**_kw: Any) -> None:
        if mgr.presenter.ai_ui is None:
            mgr.presenter.ai_ui = _WindowStub()  # type: ignore[assignment]
        if mgr.presenter.overlay_ui is None:
            mgr.presenter.overlay_ui = _WindowStub()  # type: ignore[assignment]

    monkeypatch.setattr(mgr.presenter, "ensure_windows", _stub_ensure_windows)

    ctrl = ModeController(mgr)
    ui_stub = _MainUIStub()
    ctrl.set_ui_app(ui_stub)
    ctrl.set_mode("red")
    while ctrl.is_busy:
        time.sleep(0.01)

    # Lock 5 picks -> triggers battle loop
    ctrl.save_and_handle_five_picks(
        ui_stub.root,  # type: ignore[arg-type]
        ["葛", "帝", "骷", "閻", "連"],
        ["玲", "SP桃", "葛", "酒吞", "季"],
        None,
        8888,
    )

    # Wait until initial cast AND at least 1 automatic unfinished-turn retry (4 clicks total = 2 casts)
    deadline = time.time() + 2.5
    while len(clicks) < 4 and time.time() < deadline:
        time.sleep(0.02)

    assert mgr.battle_has_started is True
    assert mgr.presenter.ai_ui is not None and mgr.presenter.overlay_ui is not None
    assert len(clicks) >= 4

    # Verify Skill-First -> Target-Second click geometry (Skill Y > 0.87*h, Enemy Target Y ~ 0.30*h)
    first_skill_y = clicks[0][2]
    first_target_y = clicks[1][2]
    retry_skill_y = clicks[2][2]
    retry_target_y = clicks[3][2]
    assert first_skill_y > int(h * 0.85) and first_target_y < int(h * 0.45)
    assert retry_skill_y > int(h * 0.85) and retry_target_y < int(h * 0.45)

    # Reset for next match MUST immediately destroy/hide both floating windows
    mgr.reset_for_next_match()
    assert mgr.presenter.ai_ui is None
    assert mgr.presenter.overlay_ui is None


def test_capture_engine_viewport_lock_prevents_dark_transition_shrinkage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import onmyoji_helper.infra.capture_engine as cap_mod
    from onmyoji_helper.infra.capture_engine import CaptureEngine

    # Simulate 1420x800 client rect at 125% DPI where DWM draws 1136x640 at (0, 0)
    client_w, client_h = 1420, 800
    logical_w, logical_h = 1136, 640

    active_buffer: list[np.ndarray] = []

    class _BmpStub:
        def CreateCompatibleBitmap(self, _dc: Any, _w: int, _h: int) -> None:
            pass

        def GetBitmapBits(self, _as_str: bool) -> bytes:
            return active_buffer[0].tobytes()

        def GetHandle(self) -> int:
            return 1

    class _DcStub:
        def CreateCompatibleDC(self) -> "_DcStub":
            return self

        def SelectObject(self, _bmp: Any) -> None:
            pass

        def BitBlt(self, *_args: Any) -> None:
            pass

        def DeleteDC(self) -> None:
            pass

    monkeypatch.setattr(cap_mod.win32gui, "IsWindow", lambda _h: True)
    monkeypatch.setattr(cap_mod.win32gui, "GetClientRect", lambda _h: (0, 0, client_w, client_h))
    monkeypatch.setattr(cap_mod.win32gui, "GetDC", lambda _h: 101)
    monkeypatch.setattr(cap_mod.win32gui, "ReleaseDC", lambda _h, _dc: 1)
    monkeypatch.setattr(cap_mod.win32gui, "DeleteObject", lambda _o: None)
    monkeypatch.setattr(cap_mod.win32ui, "CreateDCFromHandle", lambda _dc: _DcStub())
    monkeypatch.setattr(cap_mod.win32ui, "CreateBitmap", lambda: _BmpStub())

    engine = CaptureEngine()

    # Frame 1: Normal 125% DPI frame (non-black 1136x640 at top-left, DWM black margin right/bottom)
    f1 = np.zeros((client_h, client_w, 4), dtype=np.uint8)
    f1[0:logical_h, 0:logical_w, :3] = 60
    active_buffer[:] = [f1]
    cap1 = engine.capture_frame(hwnd=7777, need_gray=False)
    assert cap1 is not None
    assert (cap1.width, cap1.height) == (logical_w, logical_h)
    assert cap1.crop.x0 == 0 and cap1.crop.y0 == 0

    # Frame 2: Dark ink transition / ultimate cutscene where only a 705x440 center patch is non-black
    # (Reproduces runtime log line 159: '尺寸=705x440' -> must stay locked at 1136x640!)
    f2 = np.zeros((client_h, client_w, 4), dtype=np.uint8)
    f2[100:540, 215:920, :3] = 90
    active_buffer[:] = [f2]
    cap2 = engine.capture_frame(hwnd=7777, need_gray=False)
    assert cap2 is not None
    assert (cap2.width, cap2.height) == (logical_w, logical_h)
    assert cap2.crop.x0 == 0 and cap2.crop.y0 == 0

