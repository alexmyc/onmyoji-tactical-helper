"""End-to-End Combat Pipeline Integration Test.

Verifies the complete dataflow without hardware or display dependencies:
`Synthetic Frame -> StateParser + TurnDetector -> BattleContextBuilder -> TacticalEngine (Rule)
 -> BattlePresenter + CoordinateMapper (125% Mixed-DPI Physical Projection)`.
"""

from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
import pytest

import onmyoji_helper.vision.turn_detector as turn_mod
from onmyoji_helper.automation.coordinate_mapper import (
    CoordinateMapper,
    project_cropped_to_physical_client,
)
from onmyoji_helper.decision.battle_context import BattleContextBuilder
from onmyoji_helper.decision.llm_engine import TacticalEngine
from onmyoji_helper.domain.models import CropOffset
from onmyoji_helper.domain.public_demo_data import get_public_demo_bundle
from onmyoji_helper.domain.shikigami_repository import ShikigamiRepository
from onmyoji_helper.ui.battle_presenter import BattlePresenter
from onmyoji_helper.vision.spatial_tracker import StateParser
from onmyoji_helper.vision.turn_detector import TurnDetector


class _StubRootWindow:
    """Deterministic synchronous Tk `.after(0, fn)` stub for headless CI execution."""

    def __init__(self) -> None:
        self.scheduled_callbacks: int = 0

    def after(self, _delay_ms: int, callback: Callable[..., Any], *args: Any) -> str:
        self.scheduled_callbacks += 1
        callback(*args)
        return f"after#{self.scheduled_callbacks}"


class _StubAdvicePanel:
    def __init__(self) -> None:
        self.last_text: str = ""

    def winfo_exists(self) -> bool:
        return True

    def update_advice(self, text: str) -> None:
        self.last_text = text


class _StubOverlayPanel:
    def __init__(self) -> None:
        self.synced_hwnd: int = 0
        self.target_calls: list[tuple[bool, str | None]] = []
        self.skills_called: bool = False

    def winfo_exists(self) -> bool:
        return True

    def sync_position(self, hwnd: int) -> None:
        self.synced_hwnd = hwnd

    def clear(self) -> None:
        pass

    def update_targets(
        self,
        status_dict: dict[str, dict[str, Any]],
        is_enemy: bool = False,
        img_w: float = 1.0,
        img_h: float = 1.0,
        target_slot: str | None = None,
        crop: CropOffset | None = None,
    ) -> None:
        assert len(status_dict) == 5
        self.target_calls.append((is_enemy, target_slot))

    def update_skills(
        self,
        detector: Any,
        is_my_turn: bool,
        skill_states: dict[str, bool],
        img_w: float = 1.0,
        img_h: float = 1.0,
        skill_mapping: dict[str, str | None] | None = None,
        crop: CropOffset | None = None,
    ) -> None:
        assert is_my_turn is True
        assert skill_states["s3"] is True
        self.skills_called = True


def _build_125dpi_logical_combat_frame(detector: TurnDetector, w: int = 1144, h: int = 648) -> np.ndarray:
    """Builds a logical 1144x648 frame (captured from a 1430x810 @ 125% DPI client window)."""
    frame = np.full((h, w, 3), (18, 20, 24), dtype=np.uint8)
    radius = 25
    slot_states = {"slot_1": True, "slot_2": False, "slot_3": True, "slot_4": True}
    for slot_key, lit in slot_states.items():
        rx1, ry1, rx2, ry2 = detector.PHYSICAL_SLOTS[slot_key]
        cx = int(round(w * (rx1 + rx2) * 0.5))
        cy = int(round(h * (ry1 + ry2) * 0.5))
        if lit:
            cv2.circle(frame, (cx, cy), radius - 1, (38, 178, 245), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), radius, (235, 245, 255), thickness=2, lineType=cv2.LINE_AA)
        else:
            cv2.circle(frame, (cx, cy), radius - 1, (54, 54, 54), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), radius, (158, 158, 158), thickness=2, lineType=cv2.LINE_AA)
    return frame


def test_end_to_end_synthetic_combat_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    sim_clock = [500.0]
    monkeypatch.setattr(turn_mod.time, "time", lambda: sim_clock[0])

    _, dict_map, multi_shorts, tags_db = get_public_demo_bundle()
    repo = ShikigamiRepository(tags_db=tags_db, full_to_short_map=dict_map, multi_char_shorts=multi_shorts)

    turn_detector = TurnDetector()
    tracker = StateParser()
    ctx_builder = BattleContextBuilder(repo=repo)
    engine = TacticalEngine(force_rule_only=True)

    img_w, img_h = 1144, 648
    phys_w, phys_h = 1430, 810  # 125% DWM DPI physical client window
    crop = CropOffset(x0=0, y0=0, client_w=phys_w, client_h=phys_h)
    frame = _build_125dpi_logical_combat_frame(turn_detector, img_w, img_h)

    # 1. TurnDetector geometric & HSV pass (0.15s + 2-frame debounce)
    assert turn_detector.is_my_turn(frame, frame_seq=1) is False
    sim_clock[0] = 500.16
    is_my_turn = turn_detector.is_my_turn(frame, frame_seq=2)
    assert is_my_turn is True
    skill_mapping = turn_detector.get_skill_mapping(frame)
    skill_states = turn_detector.detect_available_skills(frame, frame_seq=2)
    assert skill_states == {"s1": True, "s2": False, "s3": True, "s4": True}

    # 2. SpatialTracker 5v5 slot assignment
    my_draft = ["葛", "帝", "骷", "閻", "連"]
    opp_draft = ["玲", "SP桃", "葛", "酒吞", "季"]
    yolo_boxes: list[dict[str, float | int]] = [
        {"cls": 0, "cx": img_w * r, "cy": img_h * 0.30} for r in (0.48, 0.57, 0.66, 0.75, 0.84)
    ] + [
        {"cls": 1, "cx": img_w * r, "cy": img_h * 0.62} for r in (0.130, 0.315, 0.495, 0.665, 0.835)
    ]
    my_status, opp_status = tracker.parse_frame(
        my_draft, opp_draft, yolo_boxes, screen_width=img_w, screen_height=img_h, now=500.45
    )

    # 3. BattleContextBuilder + Rule TacticalEngine decision
    payload = ctx_builder.build_payload(
        my_status=my_status,
        opp_status=opp_status,
        is_my_turn=is_my_turn,
        skill_states=skill_states,
        active_slot="Slot 4",
        active_shiki_name="閻",
    )
    assert payload["chosen_skill"] == "S3"
    assert payload["recommended_target_slot"] == "Slot 1"
    assert payload["recommended_target_name"] == "玲"
    assert payload["target_is_enemy"] is True

    decision = engine.get_tactical_advice(payload, mode="RULE")
    assert decision["target_slot"] == "Slot 1"
    assert "閻" in decision["advice"] and "玲" in decision["advice"]

    # 4. BattlePresenter main-thread marshaling
    stub_root = _StubRootWindow()
    presenter = BattlePresenter()
    presenter.ai_ui = _StubAdvicePanel()  # type: ignore[assignment]
    presenter.overlay_ui = _StubOverlayPanel()  # type: ignore[assignment]

    presenter.schedule_overlay_frame(
        root_window=stub_root,  # type: ignore[arg-type]
        hwnd=99901,
        my_status=my_status,
        opp_status=opp_status,
        total_alive=10,
        img_w=img_w,
        img_h=img_h,
        is_my_turn=is_my_turn,
        target_slot=payload["recommended_target_slot"],
        target_is_enemy=payload["target_is_enemy"],
        turn_detector=turn_detector,
        skill_states=skill_states,
        skill_mapping=skill_mapping,
        crop=crop,
        is_cancelled_fn=lambda: False,
    )
    presenter.schedule_advice_text(
        root_window=stub_root,  # type: ignore[arg-type]
        text=decision["advice"],
        is_cancelled_fn=lambda: False,
    )
    assert presenter.overlay_ui.synced_hwnd == 99901  # type: ignore[attr-defined]
    assert presenter.overlay_ui.skills_called is True  # type: ignore[attr-defined]
    assert presenter.overlay_ui.target_calls == [(False, None), (True, "Slot 1")]  # type: ignore[attr-defined]
    assert "閻" in presenter.ai_ui.last_text  # type: ignore[attr-defined]

    # 5. CoordinateMapper 125% DWM DPI physical projection
    target_img_x, target_img_y = opp_status["Slot 1"]["pos"]
    phys_x, phys_y = project_cropped_to_physical_client(
        target_img_x, target_img_y, img_w, img_h, phys_w, phys_h, crop=crop
    )
    assert abs(phys_x - target_img_x * 1.25) <= 1.0
    assert abs(phys_y - target_img_y * 1.25) <= 1.0

    origin_x, origin_y = 200, 120
    mapper = CoordinateMapper(
        custom_mapper=lambda _h, ix, iy, iw, ih: (
            origin_x + int(round(ix * (phys_w / iw))),
            origin_y + int(round(iy * (phys_h / ih))),
            (origin_x, origin_y, origin_x + phys_w, origin_y + phys_h),
        )
    )
    mapped = mapper.map_image_to_screen(99901, target_img_x, target_img_y, img_w, img_h, crop=crop)
    assert mapped is not None
    sx, sy, bounds = mapped
    assert bounds[0] <= sx <= bounds[2] and bounds[1] <= sy <= bounds[3]
