"""Synthetic Multi-Frame Vision Replay Regression Suite.

Offline deterministic verification of:
1. `TurnDetector`: CLAHE + HoughCircles + 16-point circumferential Sobel ring gradient + HSV
   skill-availability classification across 3-skill/4-skill layouts, temporal debouncing, and
   full-screen VFX flashbang false-positive rejection.
2. `StateParser` (`SpatialTracker`): Continuous multi-frame camera pan/zoom (`estimateAffinePartial2D`)
   and symmetric `3.5s` ultimate-VFX occlusion coordinate continuity.
"""

import math

import cv2
import numpy as np
import pytest

import onmyoji_helper.vision.turn_detector as turn_mod
from onmyoji_helper.vision.spatial_tracker import StateParser
from onmyoji_helper.vision.turn_detector import TurnDetector

FRAME_W: int = 1280
FRAME_H: int = 720
MY_LINEUP: list[str] = ["葛", "帝", "骷", "閻", "連"]
OPP_LINEUP: list[str] = ["玲", "SP桃", "葛", "酒吞", "季"]


def _slot_center_px(detector: TurnDetector, slot_key: str, w: int = FRAME_W, h: int = FRAME_H) -> tuple[int, int]:
    rx1, ry1, rx2, ry2 = detector.PHYSICAL_SLOTS[slot_key]
    cx = int(round(w * (rx1 + rx2) * 0.5))
    cy = int(round(h * (ry1 + ry2) * 0.5))
    return cx, cy


def _render_synthetic_combat_frame(
    detector: TurnDetector,
    slot_configs: dict[str, bool],
    w: int = FRAME_W,
    h: int = FRAME_H,
) -> np.ndarray:
    """Synthesizes a 1280x720 BGR combat frame with dark arena background and circular skill buttons.

    Args:
        detector: `TurnDetector` instance providing normalized physical slot bounds.
        slot_configs: Mapping of `'slot_1'..'slot_4'` -> `True` (lit/available) or `False` (cooldown/0-orb).
    """
    frame = np.full((h, w, 3), (18, 20, 24), dtype=np.uint8)
    radius = 28

    for slot_key, is_available in slot_configs.items():
        cx, cy = _slot_center_px(detector, slot_key, w, h)
        if is_available:
            # Vibrant golden-amber active skill core + bright metallic ring edge
            cv2.circle(frame, (cx, cy), radius - 1, (38, 178, 245), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), radius - 9, (65, 205, 255), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), radius, (235, 245, 255), thickness=2, lineType=cv2.LINE_AA)
        else:
            # Desaturated dim cooldown/0-orb skill core + visible button bezel ring
            cv2.circle(frame, (cx, cy), radius - 1, (54, 54, 54), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), radius, (158, 158, 158), thickness=2, lineType=cv2.LINE_AA)

    return frame


def test_turn_detector_3skill_and_4skill_ring_sobel_and_hsv_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = TurnDetector()
    sim_clock = [100.0]
    monkeypatch.setattr(turn_mod.time, "time", lambda: sim_clock[0])

    # Replay A: 3-skill unit turn (slot_1 absent; slot_2=S1 lit, slot_3=S2 greyed out, slot_4=S3 lit)
    frame_3skill = _render_synthetic_combat_frame(
        detector,
        {"slot_2": True, "slot_3": False, "slot_4": True},
    )

    # t=100.00s: First detection starts debounce timer (< 0.40s threshold -> not yet confirmed)
    assert detector.is_my_turn(frame_3skill, frame_seq=1) is False

    # t=100.45s: Exceeds 0.40s activation threshold -> turn confirmed!
    sim_clock[0] = 100.45
    assert detector.is_my_turn(frame_3skill, frame_seq=2) is True

    mapping_3 = detector.get_skill_mapping(frame_3skill)
    assert mapping_3 == {"s1": "slot_2", "s2": "slot_3", "s3": "slot_4", "s4": None}

    skills_3 = detector.detect_available_skills(frame_3skill, frame_seq=2)
    assert skills_3 == {"s1": True, "s2": False, "s3": True, "s4": False}

    # End turn and advance past 0.15s deactivation grace + 0.50s post-turn lockout
    dark_frame = _render_synthetic_combat_frame(detector, {})
    sim_clock[0] = 100.50
    detector.is_my_turn(dark_frame, frame_seq=3)
    sim_clock[0] = 100.70
    assert detector.is_my_turn(dark_frame, frame_seq=4) is False

    # Replay B: 4-skill unit turn (slot_1=S1 lit, slot_2=S2 false, slot_3=S3 false, slot_4=S4 lit)
    detector.reset_cache()
    frame_4skill = _render_synthetic_combat_frame(
        detector,
        {"slot_1": True, "slot_2": False, "slot_3": False, "slot_4": True},
    )
    sim_clock[0] = 102.00
    assert detector.is_my_turn(frame_4skill, frame_seq=10) is False
    sim_clock[0] = 102.45
    assert detector.is_my_turn(frame_4skill, frame_seq=11) is True

    mapping_4 = detector.get_skill_mapping(frame_4skill)
    assert mapping_4 == {"s1": "slot_1", "s2": "slot_2", "s3": "slot_3", "s4": "slot_4"}

    skills_4 = detector.detect_available_skills(frame_4skill, frame_seq=11)
    assert skills_4 == {"s1": True, "s2": False, "s3": False, "s4": True}


def test_turn_detector_rejects_vfx_flashbang_and_enforces_temporal_debounce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = TurnDetector()
    sim_clock = [200.0]
    monkeypatch.setattr(turn_mod.time, "time", lambda: sim_clock[0])

    # Synthesize bright VFX explosion / linear slash streaks across the skill-bar ROI (no circular rings)
    vfx_flash = np.full((FRAME_H, FRAME_W, 3), (225, 240, 255), dtype=np.uint8)
    for offset in range(860, 1260, 35):
        cv2.line(vfx_flash, (offset, 610), (offset + 40, 718), (40, 90, 180), thickness=4)

    assert detector.is_my_turn(vfx_flash, frame_seq=1) is False
    sim_clock[0] = 200.60
    assert detector.is_my_turn(vfx_flash, frame_seq=2) is False
    assert detector.detect_available_skills(vfx_flash, frame_seq=2) == {
        "s1": False,
        "s2": False,
        "s3": False,
        "s4": False,
    }

    # Confirm active turn, then verify a transient 100ms (< 150ms grace) occlusion does NOT drop the turn
    detector.reset_cache()
    active_frame = _render_synthetic_combat_frame(
        detector,
        {"slot_2": True, "slot_3": True, "slot_4": True},
    )
    sim_clock[0] = 202.00
    detector.is_my_turn(active_frame, frame_seq=3)
    sim_clock[0] = 202.45
    assert detector.is_my_turn(active_frame, frame_seq=4) is True

    # Transient 100ms particle flash occluding buttons (< 0.15s DEACTIVATION_GRACE)
    sim_clock[0] = 202.50
    assert detector.is_my_turn(vfx_flash, frame_seq=5) is True
    sim_clock[0] = 202.60
    assert detector.is_my_turn(vfx_flash, frame_seq=6) is True


def test_spatial_tracker_multiframe_camera_motion_and_3_5s_occlusion_continuity() -> None:
    """Replays 12 frames (3.3s) of camera pan/zoom while Enemy Slot 1 & Ally Slot 2 are occluded."""
    parser = StateParser()
    w, h = 1144, 640

    opp_base = {
        "Slot 5": (w * 0.48, h * 0.30),
        "Slot 3": (w * 0.57, h * 0.29),
        "Slot 2": (w * 0.66, h * 0.28),
        "Slot 1": (w * 0.75, h * 0.30),
        "Slot 4": (w * 0.84, h * 0.32),
    }
    my_base = {
        "Slot 4": (w * 0.130, h * 0.61),
        "Slot 1": (w * 0.315, h * 0.63),
        "Slot 2": (w * 0.495, h * 0.65),
        "Slot 3": (w * 0.665, h * 0.63),
        "Slot 5": (w * 0.835, h * 0.60),
    }

    def transform_pt(pt: tuple[float, float], scale: float, dx: float, dy: float) -> tuple[float, float]:
        cx0, cy0 = w * 0.5, h * 0.5
        return (
            cx0 + (pt[0] - cx0) * scale + dx,
            cy0 + (pt[1] - cy0) * scale + dy,
        )

    prev_opp_slot1: tuple[int, int] | None = None
    prev_my_slot2: tuple[int, int] | None = None

    # Replay t = 0.0s .. 3.3s in 0.3s steps (frames 2..11 have Enemy Slot 1 & Ally Slot 2 occluded for 3.0s)
    for frame_idx in range(12):
        t_sec = 50.0 + frame_idx * 0.30
        is_occluded = frame_idx >= 2  # Occluded from t=50.6s through t=53.3s (delta = 3.0s < 3.5s grace)

        phase = frame_idx * 0.45
        scale = 1.0 + 0.02 * math.sin(phase) if frame_idx > 0 else 1.0
        dx = 16.0 * math.sin(phase) if frame_idx > 0 else 0.0
        dy = -9.0 * math.sin(phase) if frame_idx > 0 else 0.0

        yolo_boxes: list[dict[str, float | int]] = []
        gt_opp_slot1 = transform_pt(opp_base["Slot 1"], scale, dx, dy)
        gt_my_slot2 = transform_pt(my_base["Slot 2"], scale, dx, dy)

        for slot_name, b_pt in opp_base.items():
            if is_occluded and slot_name == "Slot 1":
                continue
            tx, ty = transform_pt(b_pt, scale, dx, dy)
            yolo_boxes.append({"cls": 0, "cx": tx, "cy": ty})

        for slot_name, b_pt in my_base.items():
            if is_occluded and slot_name == "Slot 2":
                continue
            tx, ty = transform_pt(b_pt, scale, dx, dy)
            yolo_boxes.append({"cls": 1, "cx": tx, "cy": ty})

        my_status, opp_status = parser.parse_frame(
            MY_LINEUP, OPP_LINEUP, yolo_boxes, screen_width=w, screen_height=h, now=t_sec
        )

        # Every single unit on both teams must remain alive with valid non-zero coordinates
        assert all(u["alive"] and u["pos"] != (0, 0) for u in opp_status.values())
        assert all(u["alive"] and u["pos"] != (0, 0) for u in my_status.values())

        est_opp1 = opp_status["Slot 1"]["pos"]
        est_my2 = my_status["Slot 2"]["pos"]

        if is_occluded:
            # 2D partial affine solver tracks camera motion within <= 4 px of ground truth!
            assert math.hypot(est_opp1[0] - gt_opp_slot1[0], est_opp1[1] - gt_opp_slot1[1]) <= 4.0
            assert math.hypot(est_my2[0] - gt_my_slot2[0], est_my2[1] - gt_my_slot2[1]) <= 4.0

        if prev_opp_slot1 is not None and prev_my_slot2 is not None:
            # Verify trajectory smoothness (no sudden coordinate teleportation)
            assert math.hypot(est_opp1[0] - prev_opp_slot1[0], est_opp1[1] - prev_opp_slot1[1]) <= 18.0
            assert math.hypot(est_my2[0] - prev_my_slot2[0], est_my2[1] - prev_my_slot2[1]) <= 18.0

        prev_opp_slot1 = est_opp1
        prev_my_slot2 = est_my2


def test_spatial_tracker_grace_expiry_marks_defeated_unit_cleanly() -> None:
    """Verifies that absence >= 3.5s marks a defeated unit `alive=False` while re-appearing allies re-lock."""
    parser = StateParser()
    w, h = 1144, 640
    full_boxes: list[dict[str, float | int]] = [
        {"cls": 0, "cx": w * r, "cy": h * 0.30} for r in (0.48, 0.57, 0.66, 0.75, 0.84)
    ] + [
        {"cls": 1, "cx": w * r, "cy": h * 0.62} for r in (0.130, 0.315, 0.495, 0.665, 0.835)
    ]

    parser.parse_frame(MY_LINEUP, OPP_LINEUP, full_boxes, w, h, now=10.0)

    # Remove Enemy Slot 1 (0.75 * w) permanently; keep all others visible
    post_ko_boxes = [
        b for b in full_boxes if not (b["cls"] == 0 and abs(float(b["cx"]) - w * 0.75) < 5.0)
    ]

    # At t = 13.4s (delta = 3.4s < 3.5s): still protected by occlusion grace period
    _, opp_grace = parser.parse_frame(MY_LINEUP, OPP_LINEUP, post_ko_boxes, w, h, now=13.4)
    assert opp_grace["Slot 1"]["alive"] is True
    assert opp_grace["Slot 1"]["pos"] != (0, 0)

    # At t = 13.6s (delta = 3.6s > 3.5s): grace expires -> unit marked defeated cleanly
    _, opp_ko = parser.parse_frame(MY_LINEUP, OPP_LINEUP, post_ko_boxes, w, h, now=13.6)
    assert opp_ko["Slot 1"]["alive"] is False
    assert opp_ko["Slot 1"]["pos"] == (0, 0)
    assert sum(1 for u in opp_ko.values() if u["alive"]) == 4
