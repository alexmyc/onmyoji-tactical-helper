"""Synthetic Vision Regression Suite (Noisy Tracking, Occlusion, Sobel Turn & OCR Confidence).

Offline deterministic verification of:
1. `TurnDetector`: CLAHE + HoughCircles + 16-point circumferential Sobel ring gradient + HSV
   skill-availability classification across 3-skill/4-skill layouts, temporal debouncing, and
   full-screen VFX flashbang false-positive rejection.
2. `StateParser` (`SpatialTracker`): Multi-frame camera pan/zoom (`estimateAffinePartial2D`)
   under realistic `±4 px` Gaussian detection jitter, random single-frame dropouts, and
   continuous `3.0s` (`< 3.5s` grace) ultimate-VFX occlusion.
3. `DraftOcrScanner`: Configurable `min_confidence=0.65` filtering of low-confidence background
   OCR noise without dropping genuine *Onmyoji* Ban/Pick text.
"""

import math
import random
from typing import Any

import cv2
import numpy as np
import pytest

import onmyoji_helper.vision.turn_detector as turn_mod
from onmyoji_helper.decision.draft_fsm import DraftStateMachine
from onmyoji_helper.domain.public_demo_data import get_public_demo_bundle
from onmyoji_helper.domain.shikigami_repository import ShikigamiRepository
from onmyoji_helper.vision.draft_ocr_scanner import DraftOcrScanner
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
    """Synthesizes a 1280x720 BGR combat frame with dark arena background and circular skill buttons."""
    frame = np.full((h, w, 3), (18, 20, 24), dtype=np.uint8)
    radius = 28

    for slot_key, is_available in slot_configs.items():
        cx, cy = _slot_center_px(detector, slot_key, w, h)
        if is_available:
            cv2.circle(frame, (cx, cy), radius - 1, (38, 178, 245), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), radius - 9, (65, 205, 255), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), radius, (235, 245, 255), thickness=2, lineType=cv2.LINE_AA)
        else:
            cv2.circle(frame, (cx, cy), radius - 1, (54, 54, 54), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), radius, (158, 158, 158), thickness=2, lineType=cv2.LINE_AA)

    return frame


def test_turn_detector_3skill_and_4skill_ring_sobel_and_hsv_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = TurnDetector()
    sim_clock = [100.0]
    monkeypatch.setattr(turn_mod.time, "time", lambda: sim_clock[0])

    frame_3skill = _render_synthetic_combat_frame(
        detector,
        {"slot_2": True, "slot_3": False, "slot_4": True},
    )

    assert detector.is_my_turn(frame_3skill, frame_seq=1) is False
    sim_clock[0] = 100.45
    assert detector.is_my_turn(frame_3skill, frame_seq=2) is True

    mapping_3 = detector.get_skill_mapping(frame_3skill)
    assert mapping_3 == {"s1": "slot_2", "s2": "slot_3", "s3": "slot_4", "s4": None}

    skills_3 = detector.detect_available_skills(frame_3skill, frame_seq=2)
    assert skills_3 == {"s1": True, "s2": False, "s3": True, "s4": False}

    dark_frame = _render_synthetic_combat_frame(detector, {})
    sim_clock[0] = 100.50
    detector.is_my_turn(dark_frame, frame_seq=3)
    sim_clock[0] = 100.70
    assert detector.is_my_turn(dark_frame, frame_seq=4) is False

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

    detector.reset_cache()
    active_frame = _render_synthetic_combat_frame(
        detector,
        {"slot_2": True, "slot_3": True, "slot_4": True},
    )
    sim_clock[0] = 202.00
    detector.is_my_turn(active_frame, frame_seq=3)
    sim_clock[0] = 202.45
    assert detector.is_my_turn(active_frame, frame_seq=4) is True

    sim_clock[0] = 202.50
    assert detector.is_my_turn(vfx_flash, frame_seq=5) is True
    sim_clock[0] = 202.60
    assert detector.is_my_turn(vfx_flash, frame_seq=6) is True


def test_spatial_tracker_noisy_gaussian_jitter_and_random_single_frame_dropouts() -> None:
    """Verifies EMA jitter suppression and affine occlusion recovery under ±4 px noise & random dropouts."""
    rng = random.Random(20261125)
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
        return cx0 + (pt[0] - cx0) * scale + dx, cy0 + (pt[1] - cy0) * scale + dy

    def bounded_jitter() -> float:
        return max(-4.0, min(4.0, rng.gauss(0.0, 2.6)))

    raw_step_deltas: list[float] = []
    ema_step_deltas: list[float] = []
    occlusion_errors_px: list[float] = []
    prev_raw_my1: tuple[float, float] | None = None
    prev_ema_my1: tuple[int, int] | None = None

    # 24 frames (t = 50.0s .. 56.9s at 0.30s cadence):
    # - Frame 0: clean 5v5 baseline lock
    # - Frames 1..10 (3.0s < 3.5s grace): Enemy Slot 1 continuously occluded + ±4 px noise + random single-frame dropouts
    # - Frames 11..23: Static camera with ±4 px detection jitter to measure EMA variance reduction
    for frame_idx in range(24):
        t_sec = 50.0 + frame_idx * 0.30
        in_camera_pan = 1 <= frame_idx <= 10
        occlude_opp_slot1 = 1 <= frame_idx <= 10

        phase = frame_idx * 0.40
        scale = 1.0 + (0.02 * math.sin(phase) if in_camera_pan else 0.0)
        dx = 14.0 * math.sin(phase) if in_camera_pan else 0.0
        dy = -8.0 * math.sin(phase) if in_camera_pan else 0.0

        yolo_boxes: list[dict[str, float | int]] = []
        gt_opp_slot1 = transform_pt(opp_base["Slot 1"], scale, dx, dy)

        # Deterministically drop at most 1 extra non-Slot-1 unit on alternating frames to simulate detector flicker
        dropped_opp_extra = "Slot 4" if (in_camera_pan and frame_idx % 3 == 2) else None
        dropped_my_extra = "Slot 5" if (in_camera_pan and frame_idx % 4 == 1) else None

        for slot_name, b_pt in opp_base.items():
            if (occlude_opp_slot1 and slot_name == "Slot 1") or slot_name == dropped_opp_extra:
                continue
            tx, ty = transform_pt(b_pt, scale, dx, dy)
            jx = tx + (bounded_jitter() if frame_idx > 0 else 0.0)
            jy = ty + (bounded_jitter() if frame_idx > 0 else 0.0)
            yolo_boxes.append({"cls": 0, "cx": jx, "cy": jy})

        curr_raw_my1: tuple[float, float] | None = None
        for slot_name, b_pt in my_base.items():
            if slot_name == dropped_my_extra:
                continue
            tx, ty = transform_pt(b_pt, scale, dx, dy)
            jx = tx + (bounded_jitter() if frame_idx > 0 else 0.0)
            jy = ty + (bounded_jitter() if frame_idx > 0 else 0.0)
            yolo_boxes.append({"cls": 1, "cx": jx, "cy": jy})
            if slot_name == "Slot 1":
                curr_raw_my1 = (jx, jy)

        my_status, opp_status = parser.parse_frame(
            MY_LINEUP, OPP_LINEUP, yolo_boxes, screen_width=w, screen_height=h, now=t_sec
        )

        # 100% slot retention across both continuous 3.0s occlusion and random single-frame dropouts
        assert all(u["alive"] and u["pos"] != (0, 0) for u in opp_status.values())
        assert all(u["alive"] and u["pos"] != (0, 0) for u in my_status.values())

        if occlude_opp_slot1:
            est_opp1 = opp_status["Slot 1"]["pos"]
            err_px = math.hypot(est_opp1[0] - gt_opp_slot1[0], est_opp1[1] - gt_opp_slot1[1])
            occlusion_errors_px.append(err_px)
            # Even under worst-case cantilever extrapolation (2 rightmost anchors dropped + ±4 px noise), stays well inside ~50 px unit hitbox
            assert err_px <= 22.0, f"Noisy affine occlusion prediction drifted by {err_px:.2f} px"

        # Measure frame-to-frame jitter suppression on static-camera noisy frames (12..23)
        if frame_idx >= 12 and curr_raw_my1 is not None:
            curr_ema_my1 = my_status["Slot 1"]["pos"]
            if prev_raw_my1 is not None and prev_ema_my1 is not None:
                raw_step_deltas.append(
                    math.hypot(curr_raw_my1[0] - prev_raw_my1[0], curr_raw_my1[1] - prev_raw_my1[1])
                )
                ema_step_deltas.append(
                    math.hypot(curr_ema_my1[0] - prev_ema_my1[0], curr_ema_my1[1] - prev_ema_my1[1])
                )
            prev_raw_my1 = curr_raw_my1
            prev_ema_my1 = curr_ema_my1

    assert (sum(occlusion_errors_px) / len(occlusion_errors_px)) <= 10.0
    # EMA smoothing must measurably reduce frame-to-frame jitter compared to raw noisy detections
    mean_raw_jitter = sum(raw_step_deltas) / len(raw_step_deltas)
    mean_ema_jitter = sum(ema_step_deltas) / len(ema_step_deltas)
    assert mean_ema_jitter < mean_raw_jitter * 0.65


def test_spatial_tracker_grace_expiry_marks_defeated_unit_cleanly() -> None:
    """Verifies that absence >= 3.5s marks a defeated unit `alive=False`."""
    parser = StateParser()
    w, h = 1144, 640
    full_boxes: list[dict[str, float | int]] = [
        {"cls": 0, "cx": w * r, "cy": h * 0.30} for r in (0.48, 0.57, 0.66, 0.75, 0.84)
    ] + [
        {"cls": 1, "cx": w * r, "cy": h * 0.62} for r in (0.130, 0.315, 0.495, 0.665, 0.835)
    ]

    parser.parse_frame(MY_LINEUP, OPP_LINEUP, full_boxes, w, h, now=10.0)
    post_ko_boxes = [
        b for b in full_boxes if not (b["cls"] == 0 and abs(float(b["cx"]) - w * 0.75) < 5.0)
    ]

    _, opp_grace = parser.parse_frame(MY_LINEUP, OPP_LINEUP, post_ko_boxes, w, h, now=13.4)
    assert opp_grace["Slot 1"]["alive"] is True
    assert opp_grace["Slot 1"]["pos"] != (0, 0)

    _, opp_ko = parser.parse_frame(MY_LINEUP, OPP_LINEUP, post_ko_boxes, w, h, now=13.6)
    assert opp_ko["Slot 1"]["alive"] is False
    assert opp_ko["Slot 1"]["pos"] == (0, 0)
    assert sum(1 for u in opp_ko.values() if u["alive"]) == 4


def test_draft_ocr_scanner_min_confidence_filters_noise_without_dropping_picks() -> None:
    """Verifies 2-frame temporal consistency: filters 1-frame 0.52 noise, confirms 2-frame 0.55 '季'."""
    _, dict_map, multi_shorts, tags_db = get_public_demo_bundle()
    repo = ShikigamiRepository(tags_db=tags_db, full_to_short_map=dict_map, multi_char_shorts=multi_shorts)

    call_idx = [0]

    def fake_rapidocr(stitched_img: np.ndarray) -> tuple[list[Any], None]:
        call_idx[0] += 1
        _h, total_w = stitched_img.shape[:2]
        opp_x = float(total_w - 150)

        if call_idx[0] == 1:
            return (
                [
                    # High-confidence (>= 0.65) -> passes immediately on Frame 1
                    ([[20.0, 40.0], [80.0, 40.0], [80.0, 70.0], [20.0, 70.0]], "不知火", 0.96),
                    ([[opp_x, 20.0], [opp_x + 90, 20.0], [opp_x + 90, 50.0], [opp_x, 50.0]], "禁用封陽君", 0.94),
                    # Single-frame transient background spark hallucination ('火' @ 0.52) -> held on Frame 1, disappears on Frame 2 -> FILTERED
                    ([[25.0, 90.0], [55.0, 90.0], [55.0, 118.0], [25.0, 118.0]], "火", 0.52),
                    # Genuine single-char shikigami '季' @ 0.55 -> Frame 1 streak=1 (held), Frame 2 streak=2 -> CONFIRMED
                    ([[opp_x, 95.0], [opp_x + 40.0, 95.0], [opp_x + 40.0, 125.0], [opp_x, 125.0]], "季", 0.55),
                    # Non-whitelisted garbled noise (< 0.65) -> immediately dropped
                    ([[25.0, 140.0], [55.0, 140.0], [55.0, 165.0], [25.0, 165.0]], "※亂碼#", 0.42),
                ],
                None,
            )
        return (
            [
                ([[20.0, 40.0], [80.0, 40.0], [80.0, 70.0], [20.0, 70.0]], "不知火", 0.95),
                # '季' persists on 2nd consecutive frame at 0.55 -> reaches min_consecutive_frames=2 and passes!
                ([[opp_x, 95.0], [opp_x + 40.0, 95.0], [opp_x + 40.0, 125.0], [opp_x, 125.0]], "季", 0.55),
                # New single-frame flicker '白' @ 0.52 on Frame 2 -> streak=1 < 2 -> FILTERED
                ([[opp_x, 150.0], [opp_x + 40.0, 150.0], [opp_x + 40.0, 180.0], [opp_x, 180.0]], "白", 0.52),
            ],
            None,
        )

    scanner = DraftOcrScanner(
        ocr_callable=fake_rapidocr,
        min_confidence=0.65,
        known_token_min_confidence=0.35,
        min_consecutive_frames=2,
        known_tokens=repo.known_ocr_tokens,
    )
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    fsm = DraftStateMachine()

    # Frame 1: High-confidence '不知火' & '禁用封陽君' pass immediately; 1-frame '火' (0.52) and '季' (0.55, 1st frame) are held
    ocr_opp_1, ocr_my_1, invoked_1 = scanner.scan_draft_rois(
        screen_color=dummy_frame,
        screen_gray=None,
        is_active_state=True,
        has_locked_opp_ban=False,
        in_battle_five_picks=False,
        now=10.0,
    )
    assert invoked_1 is True
    assert [r[1] for r in ocr_my_1] == ["不知火"]
    assert [r[1] for r in ocr_opp_1] == ["禁用封陽君"]

    snap1, _ = fsm.step(ocr_opp_1, ocr_my_1, repo.sorted_full_to_short, my_ban="禪", now=10.0)
    assert snap1.opp_ban == "封"

    # Frame 2 (even on a static dummy_frame, pending '季' streak bypasses motion-diff skip!):
    # Transient '火' (0.52) vanished and never leaked; persistent '季' (0.55) is confirmed on Frame 2!
    ocr_opp_2, ocr_my_2, invoked_2 = scanner.scan_draft_rois(
        screen_color=dummy_frame,
        screen_gray=None,
        is_active_state=True,
        has_locked_opp_ban=True,
        in_battle_five_picks=False,
        now=11.0,
    )
    assert invoked_2 is True
    assert [r[1] for r in ocr_my_2] == ["不知火"]
    assert [r[1] for r in ocr_opp_2] == ["季"]

    snap2, _ = fsm.step(ocr_opp_2, ocr_my_2, repo.sorted_full_to_short, my_ban="禪", now=11.0)
    assert snap2.opp_draft == ["季"]
    assert snap2.my_draft == ["離"]
