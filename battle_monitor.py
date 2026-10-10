# -*- coding: utf-8 -*-
"""Battle Monitor Module for Onmyoji Tactical Helper.

This module provides spatial state parsing for battle scenes. It uses YOLO detection boxes,
affine transformation matrices (cv2.estimateAffinePartial2D), Exponential Moving Average (EMA)
position smoothing, and temporal grace periods to track ally and enemy character positions.
"""

import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


class StateParser:
    """Parses battle frame YOLO bounding boxes and maps character locations to team slot anchors."""

    def __init__(self) -> None:
        """Initializes slot anchor coordinates, baseline position containers, and tracking states."""
        # Ally slot relative X-axis anchors
        self.MY_SLOT_ANCHORS: List[Tuple[str, float]] = [
            ("Slot 4", 0.130),  # Far left (13.0%)
            ("Slot 1", 0.315),  # Left 2 (31.5%)
            ("Slot 2", 0.495),  # Center (49.5%)
            ("Slot 3", 0.665),  # Right 2 (66.5%)
            ("Slot 5", 0.835)   # Far right (83.5%)
        ]

        # Enemy slot relative X-axis anchors
        self.OPP_SLOT_ANCHORS: List[Tuple[str, float]] = [
            ("Slot 5", 0.48),
            ("Slot 3", 0.57),
            ("Slot 2", 0.66),
            ("Slot 1", 0.75),
            ("Slot 4", 0.84)
        ]

        self.baseline_opp: Dict[str, Tuple[float, float]] = {}
        self.is_opp_baseline_set: bool = False

        self.baseline_my: Dict[str, Tuple[float, float]] = {}
        self.is_my_baseline_set: bool = False

        self.smooth_pos_my: Dict[str, Tuple[float, float]] = {}
        self.last_seen_opp: Dict[str, float] = {}
        self.last_seen_my: Dict[str, float] = {}

    def reset_baseline(self) -> None:
        """Clears stored baseline coordinate positions and active tracking timers."""
        self.baseline_opp.clear()
        self.is_opp_baseline_set = False
        self.baseline_my.clear()
        self.is_my_baseline_set = False
        self.smooth_pos_my.clear()
        self.last_seen_opp.clear()
        self.last_seen_my.clear()

    def parse_frame(
        self,
        my_draft: List[str],
        opp_draft: List[str],
        yolo_boxes: List[Dict[str, Any]],
        screen_width: int
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """Parses a single battle frame to map detected bounding boxes to slot positions.

        Args:
            my_draft: List of player drafted character names.
            opp_draft: List of opponent drafted character names.
            yolo_boxes: List of bounding box dictionaries detected by the YOLO model.
            screen_width: Source image frame width in pixels.

        Returns:
            Tuple containing (my_status, opp_status) dictionaries mapped to slot keys.
        """
        current_time = time.time()
        screen_height = int(screen_width * 0.5625) if screen_width > 0 else 0

        OPP_GRACE_PERIOD: float = 3.5
        MY_GRACE_PERIOD: float = 3.5

        # 1. Filter and sort enemy boxes (class 0) by X coordinate
        enemy_boxes = [b for b in yolo_boxes if b.get("cls") == 0]
        enemy_boxes.sort(key=lambda b: b.get("cx", 0))

        # 2. Filter and sort ally boxes (class 1) in lower half screen excluding Kagura
        ally_boxes = [
            b for b in yolo_boxes
            if b.get("cls") == 1
            and b.get("cy", 0) > (screen_height * 0.40)
            and b.get("cx", 0) < (screen_width * 0.87)
        ]
        ally_boxes.sort(key=lambda b: b.get("cx", 0))

        my_status: Dict[str, Dict[str, Any]] = {
            f"Slot {i}": {
                "name": my_draft[i - 1] if i - 1 < len(my_draft) else f"我方{i}號",
                "alive": False,
                "pos": (0, 0)
            }
            for i in range(1, 6)
        }
        opp_status: Dict[str, Dict[str, Any]] = {
            f"Slot {i}": {
                "name": opp_draft[i - 1] if i - 1 < len(opp_draft) else f"敵方{i}號",
                "alive": False,
                "pos": (0, 0)
            }
            for i in range(1, 6)
        }

        if screen_width <= 0:
            return my_status, opp_status

        # 3. Establish 2D baseline anchors when 5 units are detected
        if len(enemy_boxes) == 5:
            for i, box in enumerate(enemy_boxes[:5]):
                slot_name = self.OPP_SLOT_ANCHORS[i][0]
                self.baseline_opp[slot_name] = (box["cx"], box["cy"])
            self.is_opp_baseline_set = True

        if len(ally_boxes) == 5 and not self.is_my_baseline_set:
            for i, box in enumerate(ally_boxes[:5]):
                slot_name = self.MY_SLOT_ANCHORS[i][0]
                self.baseline_my[slot_name] = (box["cx"], box["cy"])
            self.is_my_baseline_set = True

        predicted_opp_pos: Dict[str, Tuple[float, float]] = {}
        predicted_my_pos: Dict[str, Tuple[float, float]] = {}

        # 4. Estimate enemy partial 2D affine transformation matrix
        if self.is_opp_baseline_set and len(enemy_boxes) >= 2:
            src_pts = []
            dst_pts = []
            for b in enemy_boxes:
                best_slot = None
                min_d = float('inf')
                for slot_key, base_pos in self.baseline_opp.items():
                    d = np.hypot(b["cx"] - base_pos[0], b["cy"] - base_pos[1])
                    if d < min_d:
                        min_d = d
                        best_slot = slot_key
                if min_d < screen_width * 0.06 and best_slot:
                    src_pts.append(self.baseline_opp[best_slot])
                    dst_pts.append((b["cx"], b["cy"]))

            if len(src_pts) >= 3:
                H, _ = cv2.estimateAffinePartial2D(
                    np.array(src_pts, dtype=np.float32),
                    np.array(dst_pts, dtype=np.float32)
                )
                if H is not None:
                    for slot_key, base_pos in self.baseline_opp.items():
                        pt = np.array([base_pos[0], base_pos[1], 1.0])
                        pred_x, pred_y = np.dot(H, pt)
                        predicted_opp_pos[slot_key] = (pred_x, pred_y)

        # 5. Estimate ally partial 2D affine transformation matrix
        if self.is_my_baseline_set and len(ally_boxes) >= 2:
            src_pts_my = []
            dst_pts_my = []
            for b in ally_boxes:
                best_slot = None
                min_d = float('inf')
                for slot_key, base_pos in self.baseline_my.items():
                    d = np.hypot(b["cx"] - base_pos[0], b["cy"] - base_pos[1])
                    if d < min_d:
                        min_d = d
                        best_slot = slot_key
                if min_d < screen_width * 0.08 and best_slot:
                    src_pts_my.append(self.baseline_my[best_slot])
                    dst_pts_my.append((b["cx"], b["cy"]))

            if len(src_pts_my) >= 3:
                H_my, _ = cv2.estimateAffinePartial2D(
                    np.array(src_pts_my, dtype=np.float32),
                    np.array(dst_pts_my, dtype=np.float32)
                )
                if H_my is not None:
                    for slot_key, base_pos in self.baseline_my.items():
                        pt = np.array([base_pos[0], base_pos[1], 1.0])
                        pred_x, pred_y = np.dot(H_my, pt)
                        predicted_my_pos[slot_key] = (pred_x, pred_y)

        # 6. Map enemy slots using distance optimization
        strict_radius_opp = screen_width * 0.06
        opp_candidates = []
        for slot_name, anchor_ratio in self.OPP_SLOT_ANCHORS:
            target_pos = predicted_opp_pos.get(slot_name) or self.baseline_opp.get(slot_name)
            tx = target_pos[0] if target_pos else (anchor_ratio * screen_width)
            ty = target_pos[1] if target_pos else 0

            for b_idx, b in enumerate(enemy_boxes):
                dist = np.hypot(b["cx"] - tx, b["cy"] - ty) if target_pos else abs(b["cx"] - tx)
                if dist < strict_radius_opp:
                    opp_candidates.append((dist, slot_name, b_idx))

        opp_candidates.sort(key=lambda x: x[0])
        assigned_opp_slots = set()
        assigned_opp_boxes = set()

        for dist, slot_name, b_idx in opp_candidates:
            if slot_name not in assigned_opp_slots and b_idx not in assigned_opp_boxes:
                assigned_opp_slots.add(slot_name)
                assigned_opp_boxes.add(b_idx)
                box = enemy_boxes[b_idx]
                opp_status[slot_name]["alive"] = True
                opp_status[slot_name]["pos"] = (int(box["cx"]), int(box["cy"]))
                self.last_seen_opp[slot_name] = current_time

        # 7. Map ally slots with Exponential Moving Average (EMA) smoothing
        strict_radius_ally = screen_width * 0.08
        ally_candidates = []

        for slot_name, anchor_ratio in self.MY_SLOT_ANCHORS:
            target_pos = predicted_my_pos.get(slot_name) or self.baseline_my.get(slot_name)
            tx = target_pos[0] if target_pos else (anchor_ratio * screen_width)
            ty = target_pos[1] if target_pos else 0

            for b_idx, b in enumerate(ally_boxes):
                dist = np.hypot(b["cx"] - tx, b["cy"] - ty) if target_pos else abs(b["cx"] - tx)
                if dist < strict_radius_ally:
                    ally_candidates.append((dist, slot_name, b_idx))

        ally_candidates.sort(key=lambda x: x[0])
        assigned_my_slots = set()
        assigned_my_boxes = set()

        for dist, slot_name, b_idx in ally_candidates:
            if slot_name not in assigned_my_slots and b_idx not in assigned_my_boxes:
                assigned_my_slots.add(slot_name)
                assigned_my_boxes.add(b_idx)
                box = ally_boxes[b_idx]

                raw_cx, raw_cy = box["cx"], box["cy"]

                # EMA Position Smoothing (alpha = 0.30)
                if slot_name in self.smooth_pos_my:
                    old_cx, old_cy = self.smooth_pos_my[slot_name]
                    sm_cx = 0.30 * raw_cx + 0.70 * old_cx
                    sm_cy = 0.30 * raw_cy + 0.70 * old_cy
                else:
                    sm_cx, sm_cy = raw_cx, raw_cy

                self.smooth_pos_my[slot_name] = (sm_cx, sm_cy)
                my_status[slot_name]["alive"] = True
                my_status[slot_name]["pos"] = (int(sm_cx), int(sm_cy))
                self.last_seen_my[slot_name] = current_time

        # 8. Apply temporal grace period lock for skill animation occlusions
        for slot_name, info in opp_status.items():
            if not info["alive"]:
                last_time = self.last_seen_opp.get(slot_name, 0)
                if last_time > 0 and (current_time - last_time < OPP_GRACE_PERIOD):
                    info["alive"] = True

        for slot_name, info in my_status.items():
            if not info["alive"]:
                last_time = self.last_seen_my.get(slot_name, 0)
                if last_time > 0 and (current_time - last_time < MY_GRACE_PERIOD):
                    info["alive"] = True
                    if slot_name in self.smooth_pos_my:
                        info["pos"] = (
                            int(self.smooth_pos_my[slot_name][0]),
                            int(self.smooth_pos_my[slot_name][1])
                        )

        return my_status, opp_status