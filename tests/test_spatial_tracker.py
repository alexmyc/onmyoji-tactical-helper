"""Unit tests for 5v5 YOLOv8 spatial tracking, affine camera-shake compensation, and VFX occlusion recovery."""

from onmyoji_helper.vision.spatial_tracker import StateParser

MY_DRAFT = ["離", "百羽", "酒吞", "SP吸", "SP貓"]
OPP_DRAFT = ["神", "SP06", "石", "連", "SP熊"]


def _make_initial_boxes(w: int = 1136, h: int = 640) -> list[dict[str, float | int]]:
    opp_xs = [0.48, 0.57, 0.66, 0.75, 0.84]
    my_xs = [0.130, 0.315, 0.495, 0.665, 0.835]
    boxes: list[dict[str, float | int]] = []
    for rx in opp_xs:
        boxes.append({"cls": 0, "cx": w * rx, "cy": h * 0.30})
    for rx in my_xs:
        boxes.append({"cls": 1, "cx": w * rx, "cy": h * 0.62})
    return boxes


def test_enemy_and_ally_occlusion_grace_preserves_nonzero_coordinates() -> None:
    parser = StateParser()
    init_boxes = _make_initial_boxes(1136, 640)

    my_s, opp_s = parser.parse_frame(MY_DRAFT, OPP_DRAFT, init_boxes, 1136, 640, now=100.0)
    assert all(v["alive"] and v["pos"] != (0, 0) for v in opp_s.values())
    assert all(v["alive"] and v["pos"] != (0, 0) for v in my_s.values())
    slot2_initial_pos = opp_s["Slot 2"]["pos"]

    # Simulate ultimate VFX completely occluding Enemy Slot 2 (0.66 * 1136) for 1.8s (< 3.5s grace)
    occluded_boxes = [b for b in init_boxes if not (b["cls"] == 0 and abs(float(b["cx"]) - 1136 * 0.66) < 5)]
    _, opp_during_vfx = parser.parse_frame(MY_DRAFT, OPP_DRAFT, occluded_boxes, 1136, 640, now=101.8)

    assert opp_during_vfx["Slot 2"]["alive"] is True
    # Regression assertion: occluded enemy unit MUST retain valid non-zero coordinates!
    assert opp_during_vfx["Slot 2"]["pos"] != (0, 0)
    assert abs(opp_during_vfx["Slot 2"]["pos"][0] - slot2_initial_pos[0]) <= 3

    # After 4.0s (> 3.5s grace), genuinely defeated unit transitions to alive=False
    _, opp_after_death = parser.parse_frame(MY_DRAFT, OPP_DRAFT, occluded_boxes, 1136, 640, now=104.2)
    assert opp_after_death["Slot 2"]["alive"] is False


def test_affine_camera_pan_and_live_window_resize_rescaling() -> None:
    parser = StateParser()
    base_boxes = _make_initial_boxes(1136, 640)
    parser.parse_frame(MY_DRAFT, OPP_DRAFT, base_boxes, 1136, 640, now=10.0)

    # Shift 4 visible enemy boxes by +22px X, -12px Y while Slot 1 is occluded -> affine predicts Slot 1!
    panned_boxes = []
    for b in base_boxes:
        if b["cls"] == 0 and abs(float(b["cx"]) - 1136 * 0.75) < 5:
            continue  # Occlude Enemy Slot 1
        dx = 22.0 if b["cls"] == 0 else 0.0
        dy = -12.0 if b["cls"] == 0 else 0.0
        panned_boxes.append({"cls": b["cls"], "cx": float(b["cx"]) + dx, "cy": float(b["cy"]) + dy})

    _, opp_panned = parser.parse_frame(MY_DRAFT, OPP_DRAFT, panned_boxes, 1136, 640, now=10.5)
    expected_x = int(round(1136 * 0.75 + 22.0))
    assert opp_panned["Slot 1"]["alive"] is True
    assert abs(opp_panned["Slot 1"]["pos"][0] - expected_x) <= 4

    # Live window resize 1.5x (1704x960) proportionally rescales stored baselines
    resized_boxes = _make_initial_boxes(1704, 960)
    my_res, opp_res = parser.parse_frame(MY_DRAFT, OPP_DRAFT, resized_boxes, 1704, 960, now=11.0)
    assert parser.baseline_w == 1704 and parser.baseline_h == 960
    assert all(v["alive"] for v in opp_res.values())
    assert all(v["alive"] for v in my_res.values())


def test_enemy_edge_slots_perspective_zoom_and_dynamic_missing_edge_backfill() -> None:
    parser = StateParser()
    w, h = 1280, 720

    # 1. Battle entry frame only detects middle 3 enemies (Slot 3, Slot 2, Slot 1); outer Slot 5 & Slot 4 occluded!
    partial_boxes: list[dict[str, float | int]] = [
        {"cls": 0, "cx": w * r, "cy": h * 0.30} for r in (0.57, 0.66, 0.75)
    ]
    _, opp_partial = parser.parse_frame(MY_DRAFT, OPP_DRAFT, partial_boxes, w, h, now=1.0)
    assert opp_partial["Slot 3"]["alive"] and opp_partial["Slot 2"]["alive"] and opp_partial["Slot 1"]["alive"]
    assert not opp_partial["Slot 5"]["alive"] and not opp_partial["Slot 4"]["alive"]

    # 2a. Next frame: only 4 boxes visible — far-left Slot 5 appears at x=0.25*w while Slot 4 is still occluded
    backfill_left_boxes: list[dict[str, float | int]] = [
        {"cls": 0, "cx": w * r, "cy": h * 0.30} for r in (0.25, 0.57, 0.66, 0.75)
    ]
    _, opp_left_filled = parser.parse_frame(MY_DRAFT, OPP_DRAFT, backfill_left_boxes, w, h, now=1.2)
    assert opp_left_filled["Slot 5"]["alive"] is True and opp_left_filled["Slot 5"]["pos"] != (0, 0)
    assert not opp_left_filled["Slot 4"]["alive"]

    # 2b. Following frame: Slot 2 briefly occluded while far-right Slot 4 appears at x=0.95*w (near action bar)
    backfill_right_boxes: list[dict[str, float | int]] = [
        {"cls": 0, "cx": w * r, "cy": h * 0.30} for r in (0.25, 0.57, 0.75, 0.95)
    ]
    _, opp_full = parser.parse_frame(MY_DRAFT, OPP_DRAFT, backfill_right_boxes, w, h, now=1.4)
    assert all(v["alive"] and v["pos"] != (0, 0) for v in opp_full.values())
    assert parser.is_opp_baseline_set is True

    # 3. Camera push-in causes wider perspective spread on outer edge slots (-95px left, +20px right > old 0.06*w=76px!)
    #    even when 1 middle unit (Slot 2) is past grace at t=6.0s -> outer edge slots MUST NOT drop!
    zoomed_four_boxes: list[dict[str, float | int]] = [
        {"cls": 0, "cx": w * 0.25 - 95.0, "cy": h * 0.29},
        {"cls": 0, "cx": w * 0.57 - 18.0, "cy": h * 0.30},
        {"cls": 0, "cx": w * 0.75 + 15.0, "cy": h * 0.30},
        {"cls": 0, "cx": w * 0.95 + 20.0, "cy": h * 0.31},
    ]
    _, opp_zoomed = parser.parse_frame(MY_DRAFT, OPP_DRAFT, zoomed_four_boxes, w, h, now=6.0)
    assert opp_zoomed["Slot 5"]["alive"] is True and opp_zoomed["Slot 5"]["pos"] != (0, 0)
    assert opp_zoomed["Slot 4"]["alive"] is True and opp_zoomed["Slot 4"]["pos"] != (0, 0)
