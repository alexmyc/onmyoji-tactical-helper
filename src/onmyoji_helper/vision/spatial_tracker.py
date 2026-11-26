"""Spatial Unit Tracker Module for Onmyoji Tactical Helper.

Maps YOLOv8 bounding boxes to 5v5 ally and enemy formation slots using partial 2D affine
camera-shake compensation (`cv2.estimateAffinePartial2D`), Exponential Moving Average (EMA)
coordinate smoothing, and symmetric 3.5-second skill-animation occlusion grace periods for
both ally and enemy teams (eliminating `(0, 0)` target ring flicker during ultimate VFX).
"""

import time
from typing import Any

import cv2
import numpy as np


class StateParser:
    """Tracks real-time 5v5 ally and enemy slot coordinates across camera motion and VFX occlusions."""

    OPP_GRACE_PERIOD: float = 3.5
    MY_GRACE_PERIOD: float = 3.5

    def __init__(self) -> None:
        """Initializes slot anchor ratios, 2D affine baselines, and EMA coordinate buffers."""
        self.MY_SLOT_ANCHORS: list[tuple[str, float]] = [
            ("Slot 4", 0.130),
            ("Slot 1", 0.315),
            ("Slot 2", 0.495),
            ("Slot 3", 0.665),
            ("Slot 5", 0.835),
        ]

        self.OPP_SLOT_ANCHORS: list[tuple[str, float]] = [
            ("Slot 5", 0.48),
            ("Slot 3", 0.57),
            ("Slot 2", 0.66),
            ("Slot 1", 0.75),
            ("Slot 4", 0.84),
        ]

        self.baseline_opp: dict[str, tuple[float, float]] = {}
        self.is_opp_baseline_set: bool = False

        self.baseline_my: dict[str, tuple[float, float]] = {}
        self.is_my_baseline_set: bool = False

        self.baseline_w: int = 0
        self.baseline_h: int = 0

        self.smooth_pos_my: dict[str, tuple[float, float]] = {}
        self.smooth_pos_opp: dict[str, tuple[float, float]] = {}
        self.last_seen_opp: dict[str, float] = {}
        self.last_seen_my: dict[str, float] = {}

    def reset_baseline(self) -> None:
        """Clears stored baseline positions, EMA buffers, and occlusion timers."""
        self.baseline_opp.clear()
        self.is_opp_baseline_set = False
        self.baseline_my.clear()
        self.is_my_baseline_set = False
        self.baseline_w = 0
        self.baseline_h = 0
        self.smooth_pos_my.clear()
        self.smooth_pos_opp.clear()
        self.last_seen_opp.clear()
        self.last_seen_my.clear()

    def _rescale_baselines_if_resized(self, screen_width: int, screen_height: int) -> None:
        """Proportionally rescales baselines and both EMA coordinate maps on live window resize."""
        if (
            screen_width <= 0
            or screen_height <= 0
            or self.baseline_w <= 0
            or self.baseline_h <= 0
        ):
            return
        if screen_width == self.baseline_w and screen_height == self.baseline_h:
            return

        sx = screen_width / float(self.baseline_w)
        sy = screen_height / float(self.baseline_h)

        for k, (bx, by) in list(self.baseline_opp.items()):
            self.baseline_opp[k] = (bx * sx, by * sy)
        for k, (bx, by) in list(self.baseline_my.items()):
            self.baseline_my[k] = (bx * sx, by * sy)
        for k, (mx, my) in list(self.smooth_pos_my.items()):
            self.smooth_pos_my[k] = (mx * sx, my * sy)
        for k, (ox, oy) in list(self.smooth_pos_opp.items()):
            self.smooth_pos_opp[k] = (ox * sx, oy * sy)

        self.baseline_w = screen_width
        self.baseline_h = screen_height

    @staticmethod
    def _estimate_affine_predictions(
        boxes: list[dict[str, Any]],
        baseline: dict[str, tuple[float, float]],
        match_radius: float,
    ) -> dict[str, tuple[float, float]]:
        """Estimates 2D partial affine transform (translation + scale + rotation) from matched anchors."""
        predicted: dict[str, tuple[float, float]] = {}
        if len(boxes) < 2 or not baseline:
            return predicted

        src_pts: list[tuple[float, float]] = []
        dst_pts: list[tuple[float, float]] = []
        for b in boxes:
            best_slot: str | None = None
            min_d = float("inf")
            for slot_key, base_pos in baseline.items():
                d = float(np.hypot(b["cx"] - base_pos[0], b["cy"] - base_pos[1]))
                if d < min_d:
                    min_d = d
                    best_slot = slot_key
            if min_d < match_radius and best_slot:
                src_pts.append(baseline[best_slot])
                dst_pts.append((float(b["cx"]), float(b["cy"])))

        if len(src_pts) >= 3:
            matrix_h, _ = cv2.estimateAffinePartial2D(
                np.array(src_pts, dtype=np.float32),
                np.array(dst_pts, dtype=np.float32),
            )
            if matrix_h is not None:
                for slot_key, base_pos in baseline.items():
                    pt = np.array([base_pos[0], base_pos[1], 1.0])
                    pred_x, pred_y = np.dot(matrix_h, pt)
                    predicted[slot_key] = (float(pred_x), float(pred_y))
        return predicted

    def parse_frame(
        self,
        my_draft: list[str],
        opp_draft: list[str],
        yolo_boxes: list[dict[str, Any]],
        screen_width: int,
        screen_height: int | None = None,
        now: float | None = None,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        """Maps single-frame YOLO detections to ally and enemy slot state dictionaries.

        Args:
            my_draft: Ordered list of 5 ally shikigami names.
            opp_draft: Ordered list of 5 enemy shikigami names.
            yolo_boxes: List of `{cls, cx, cy}` bounding-box center dicts (`0` = enemy, `1` = ally).
            screen_width: Source image width in pixels.
            screen_height: Optional source image height in pixels (defaults to 16:9).
            now: Optional injectable monotonic timestamp for deterministic testing.

        Returns:
            Tuple `(my_status, opp_status)` mapping `'Slot 1'..'Slot 5'` to `{name, alive, pos}`.
        """
        current_time = time.time() if now is None else float(now)
        actual_h = (
            screen_height
            if (screen_height is not None and screen_height > 0)
            else (int(screen_width * 0.5625) if screen_width > 0 else 0)
        )

        if screen_width > 0 and actual_h > 0:
            self._rescale_baselines_if_resized(screen_width, actual_h)

        enemy_boxes = sorted(
            [b for b in yolo_boxes if b.get("cls") == 0],
            key=lambda b: b.get("cx", 0),
        )
        ally_boxes = sorted(
            [
                b
                for b in yolo_boxes
                if b.get("cls") == 1
                and b.get("cy", 0) > (actual_h * 0.40)
                and b.get("cx", 0) < (screen_width * 0.87)
            ],
            key=lambda b: b.get("cx", 0),
        )

        my_status: dict[str, dict[str, Any]] = {
            f"Slot {i}": {
                "name": my_draft[i - 1] if i - 1 < len(my_draft) else f"我方{i}號",
                "alive": False,
                "pos": (0, 0),
            }
            for i in range(1, 6)
        }
        opp_status: dict[str, dict[str, Any]] = {
            f"Slot {i}": {
                "name": opp_draft[i - 1] if i - 1 < len(opp_draft) else f"敵方{i}號",
                "alive": False,
                "pos": (0, 0),
            }
            for i in range(1, 6)
        }

        if screen_width <= 0:
            return my_status, opp_status

        if len(enemy_boxes) == 5 and not self.is_opp_baseline_set:
            if self.baseline_w <= 0:
                self.baseline_w, self.baseline_h = screen_width, actual_h
            for i, box in enumerate(enemy_boxes[:5]):
                slot_name = self.OPP_SLOT_ANCHORS[i][0]
                self.baseline_opp[slot_name] = (float(box["cx"]), float(box["cy"]))
            self.is_opp_baseline_set = True

        if len(ally_boxes) == 5 and not self.is_my_baseline_set:
            if self.baseline_w <= 0:
                self.baseline_w, self.baseline_h = screen_width, actual_h
            for i, box in enumerate(ally_boxes[:5]):
                slot_name = self.MY_SLOT_ANCHORS[i][0]
                self.baseline_my[slot_name] = (float(box["cx"]), float(box["cy"]))
            self.is_my_baseline_set = True

        predicted_opp_pos = (
            self._estimate_affine_predictions(enemy_boxes, self.baseline_opp, screen_width * 0.06)
            if self.is_opp_baseline_set
            else {}
        )
        predicted_my_pos = (
            self._estimate_affine_predictions(ally_boxes, self.baseline_my, screen_width * 0.08)
            if self.is_my_baseline_set
            else {}
        )

        # Enemy slot assignment + EMA smoothing
        strict_radius_opp = screen_width * 0.06
        opp_candidates: list[tuple[float, str, int]] = []
        for slot_name, anchor_ratio in self.OPP_SLOT_ANCHORS:
            target_pos = predicted_opp_pos.get(slot_name) or self.baseline_opp.get(slot_name)
            tx = target_pos[0] if target_pos else (anchor_ratio * screen_width)
            ty = target_pos[1] if target_pos else 0.0
            for b_idx, b in enumerate(enemy_boxes):
                dist = float(np.hypot(b["cx"] - tx, b["cy"] - ty)) if target_pos else abs(float(b["cx"]) - tx)
                if dist < strict_radius_opp:
                    opp_candidates.append((dist, slot_name, b_idx))

        opp_candidates.sort(key=lambda x: x[0])
        assigned_opp_slots = set()
        assigned_opp_boxes = set()
        for _, slot_name, b_idx in opp_candidates:
            if slot_name not in assigned_opp_slots and b_idx not in assigned_opp_boxes:
                assigned_opp_slots.add(slot_name)
                assigned_opp_boxes.add(b_idx)
                box = enemy_boxes[b_idx]
                raw_ox, raw_oy = float(box["cx"]), float(box["cy"])
                if slot_name in self.smooth_pos_opp:
                    old_ox, old_oy = self.smooth_pos_opp[slot_name]
                    sm_ox = 0.45 * raw_ox + 0.55 * old_ox
                    sm_oy = 0.45 * raw_oy + 0.55 * old_oy
                else:
                    sm_ox, sm_oy = raw_ox, raw_oy
                self.smooth_pos_opp[slot_name] = (sm_ox, sm_oy)
                opp_status[slot_name]["alive"] = True
                opp_status[slot_name]["pos"] = (int(round(sm_ox)), int(round(sm_oy)))
                self.last_seen_opp[slot_name] = current_time

        # Ally slot assignment + EMA smoothing
        strict_radius_ally = screen_width * 0.08
        ally_candidates: list[tuple[float, str, int]] = []
        for slot_name, anchor_ratio in self.MY_SLOT_ANCHORS:
            target_pos = predicted_my_pos.get(slot_name) or self.baseline_my.get(slot_name)
            tx = target_pos[0] if target_pos else (anchor_ratio * screen_width)
            ty = target_pos[1] if target_pos else 0.0
            for b_idx, b in enumerate(ally_boxes):
                dist = float(np.hypot(b["cx"] - tx, b["cy"] - ty)) if target_pos else abs(float(b["cx"]) - tx)
                if dist < strict_radius_ally:
                    ally_candidates.append((dist, slot_name, b_idx))

        ally_candidates.sort(key=lambda x: x[0])
        assigned_my_slots = set()
        assigned_my_boxes = set()
        for _, slot_name, b_idx in ally_candidates:
            if slot_name not in assigned_my_slots and b_idx not in assigned_my_boxes:
                assigned_my_slots.add(slot_name)
                assigned_my_boxes.add(b_idx)
                box = ally_boxes[b_idx]
                raw_cx, raw_cy = float(box["cx"]), float(box["cy"])
                if slot_name in self.smooth_pos_my:
                    old_cx, old_cy = self.smooth_pos_my[slot_name]
                    sm_cx = 0.30 * raw_cx + 0.70 * old_cx
                    sm_cy = 0.30 * raw_cy + 0.70 * old_cy
                else:
                    sm_cx, sm_cy = raw_cx, raw_cy
                self.smooth_pos_my[slot_name] = (sm_cx, sm_cy)
                my_status[slot_name]["alive"] = True
                my_status[slot_name]["pos"] = (int(round(sm_cx)), int(round(sm_cy)))
                self.last_seen_my[slot_name] = current_time

        # Symmetric temporal grace period recovery for BOTH enemy and ally units during VFX occlusions
        for slot_name, info in opp_status.items():
            if not info["alive"]:
                last_time = self.last_seen_opp.get(slot_name, 0.0)
                if last_time > 0.0 and (current_time - last_time < self.OPP_GRACE_PERIOD):
                    info["alive"] = True
                    fallback_pos = (
                        predicted_opp_pos.get(slot_name)
                        or self.smooth_pos_opp.get(slot_name)
                        or self.baseline_opp.get(slot_name)
                    )
                    if fallback_pos is not None:
                        info["pos"] = (int(round(fallback_pos[0])), int(round(fallback_pos[1])))

        for slot_name, info in my_status.items():
            if not info["alive"]:
                last_time = self.last_seen_my.get(slot_name, 0.0)
                if last_time > 0.0 and (current_time - last_time < self.MY_GRACE_PERIOD):
                    info["alive"] = True
                    fallback_pos = (
                        predicted_my_pos.get(slot_name)
                        or self.smooth_pos_my.get(slot_name)
                        or self.baseline_my.get(slot_name)
                    )
                    if fallback_pos is not None:
                        info["pos"] = (int(round(fallback_pos[0])), int(round(fallback_pos[1])))

        return my_status, opp_status
