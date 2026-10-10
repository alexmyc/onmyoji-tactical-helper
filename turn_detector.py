# -*- coding: utf-8 -*-
"""Turn and Skill Bar Detector Module for Onmyoji Tactical Helper.

This module provides computer vision algorithms (Hough Circles, Sobel gradient analysis,
CLAHE contrast enhancement, and HSV color analysis) to detect active player turns,
dynamically map 3/4 skill button configurations, and evaluate skill availability.
"""

import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


class TurnDetector:
    """Detects active player turns and skill button availability using computer vision."""

    def __init__(self) -> None:
        """Initializes ROI bounding coordinates, resolution norms, and debouncing timers."""
        # Primary Skill Bar Region of Interest (ROI) bounding box (normalized coordinates)
        self.SKILL_BAR_ROI: Tuple[float, float, float, float] = (0.665, 0.85, 0.998, 0.995)

        # Fixed physical slot relative coordinates (slot_1 to slot_4, left-to-right)
        self.PHYSICAL_SLOTS: Dict[str, Tuple[float, float, float, float]] = {
            "slot_1": (0.693, 0.875, 0.759, 0.990),
            "slot_2": (0.771, 0.875, 0.837, 0.990),
            "slot_3": (0.849, 0.875, 0.915, 0.990),
            "slot_4": (0.927, 0.875, 0.993, 0.990)
        }

        self.norm_size: Tuple[int, int] = (64, 64)

        # Timeline debouncing and cooldown lockout buffers
        self.turn_start_time: float = 0.0
        self.turn_loss_time: float = 0.0
        self.is_turn_confirmed: bool = False
        self.lockout_end_time: float = 0.0  # Mandatory cooldown lockout after skill cast

        # Lock skill count mode within a single turn (3-skill or 4-skill mode) to prevent flicker
        self.locked_skill_mode: Optional[int] = None

        self.ACTIVATION_THRESHOLD: float = 0.40  # Sustained duration (seconds) to confirm turn
        self.DEACTIVATION_GRACE: float = 0.15   # Grace period (seconds) before declaring turn loss

    def _verify_circle_ring_edge(self, gray_img: np.ndarray, cx: int, cy: int, r: int) -> bool:
        """Verifies circumferential gradient magnitude density for candidate circles.

        Prevents false positive detections caused by background specular highlights or arcs.

        Args:
            gray_img: Grayscale image array of the region.
            cx: Center X coordinate of the candidate circle.
            cy: Center Y coordinate of the candidate circle.
            r: Radius of the candidate circle.

        Returns:
            True if circumferential edge gradient density satisfies the threshold, False otherwise.
        """
        h, w = gray_img.shape[:2]
        sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)

        angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        hit_count = 0

        for angle in angles:
            px = int(cx + r * np.cos(angle))
            py = int(cy + r * np.sin(angle))
            if 0 <= px < w and 0 <= py < h:
                if grad_mag[py, px] >= 32.0:
                    hit_count += 1

        return hit_count >= 7

    def _eval_single_frame_turn(self, screen_color: Optional[np.ndarray]) -> bool:
        """Evaluates a single frame for active skill bar buttons using Hough Circles.

        Args:
            screen_color: Input BGR image array of the full screen.

        Returns:
            True if valid skill button geometric structures are identified, False otherwise.
        """
        if screen_color is None:
            return False

        h, w = screen_color.shape[:2]
        rx1, ry1, rx2, ry2 = self.SKILL_BAR_ROI
        x1, y1 = int(w * rx1), int(h * ry1)
        x2, y2 = int(w * rx2), int(h * ry2)

        bar_roi = screen_color[y1:y2, x1:x2]
        if bar_roi.size == 0:
            return False

        roi_h, roi_w = bar_roi.shape[:2]

        gray = cv2.cvtColor(bar_roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blur = cv2.GaussianBlur(enhanced, (5, 5), 0)

        min_r = int(roi_h * 0.20)
        max_r = int(roi_h * 0.42)

        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=int(roi_w * 0.15),
            param1=120,
            param2=42,
            minRadius=min_r,
            maxRadius=max_r
        )

        if circles is None:
            return False

        circles = np.uint16(np.around(circles[0]))
        valid_circles = []

        for c in circles:
            cx, cy, cr = int(c[0]), int(c[1]), int(c[2])
            if int(roi_h * 0.20) <= cy <= int(roi_h * 0.80):
                if self._verify_circle_ring_edge(gray, cx, cy, cr):
                    valid_circles.append((cx, cy, cr))

        if len(valid_circles) < 2:
            return False

        valid_structure = False
        for i in range(len(valid_circles)):
            for j in range(i + 1, len(valid_circles)):
                c1, c2 = valid_circles[i], valid_circles[j]
                r_diff = abs(c1[2] - c2[2]) / float(max(c1[2], c2[2]))
                y_diff = abs(c1[1] - c2[1]) / float(roi_h)

                if r_diff <= 0.12 and y_diff <= 0.08:
                    valid_structure = True
                    break

        return valid_structure

    def is_my_turn(self, screen_color: Optional[np.ndarray]) -> bool:
        """Determines if it is currently the player's turn with debouncing and cooldown lockout.

        Args:
            screen_color: Input BGR image array of the full screen.

        Returns:
            True if active player turn is confirmed, False otherwise.
        """
        current_time = time.time()

        # 1. Enforce mandatory cooldown lockout period (0.5s) post skill execution
        if current_time < self.lockout_end_time:
            self.turn_start_time = 0.0
            self.is_turn_confirmed = False
            self.locked_skill_mode = None
            return False

        raw_turn = self._eval_single_frame_turn(screen_color)

        if raw_turn:
            self.turn_loss_time = 0.0
            if self.turn_start_time == 0.0:
                self.turn_start_time = current_time

            if (current_time - self.turn_start_time) >= self.ACTIVATION_THRESHOLD:
                self.is_turn_confirmed = True
        else:
            self.turn_start_time = 0.0
            if self.turn_loss_time == 0.0:
                self.turn_loss_time = current_time

            # 2. Trigger 0.5s lockout upon turn completion (skill cast detected)
            if (
                self.is_turn_confirmed
                and (current_time - self.turn_loss_time) >= self.DEACTIVATION_GRACE
            ):
                self.is_turn_confirmed = False
                self.locked_skill_mode = None
                self.lockout_end_time = current_time + 0.50

        return self.is_turn_confirmed

    def _check_button_exists(self, roi_color: Optional[np.ndarray]) -> bool:
        """Checks for the presence of a valid skill button within a specific slot ROI.

        Args:
            roi_color: BGR image array of the slot ROI.

        Returns:
            True if a valid skill button circle is detected, False otherwise.
        """
        if roi_color is None or roi_color.size == 0:
            return False

        norm = cv2.resize(roi_color, self.norm_size, interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=20,
            param1=100,
            param2=28,
            minRadius=12,
            maxRadius=28
        )

        if circles is None:
            return False

        circles = np.uint16(np.around(circles[0]))
        cx_target, cy_target = 32, 30

        for c in circles:
            cx, cy, cr = int(c[0]), int(c[1]), int(c[2])
            if abs(cx - cx_target) <= 10 and abs(cy - cy_target) <= 10:
                if self._verify_circle_ring_edge(gray, cx, cy, cr):
                    return True

        return False

    def get_skill_mapping(self, screen_color: Optional[np.ndarray]) -> Dict[str, Optional[str]]:
        """Determines physical slot mappings for logical skills (3-skill or 4-skill mode).

        Args:
            screen_color: Input BGR image array of the full screen.

        Returns:
            Dictionary mapping logical skill names ('s1'-'s4') to physical slot keys.
        """
        default_4_mapping: Dict[str, Optional[str]] = {
            "s1": "slot_1",
            "s2": "slot_2",
            "s3": "slot_3",
            "s4": "slot_4"
        }
        mapping_3: Dict[str, Optional[str]] = {
            "s1": "slot_2",
            "s2": "slot_3",
            "s3": "slot_4",
            "s4": None
        }

        if not self.is_turn_confirmed or screen_color is None:
            return default_4_mapping

        # Return locked mode mapping if mode was previously established for the active turn
        if self.locked_skill_mode == 4:
            return default_4_mapping
        elif self.locked_skill_mode == 3:
            return mapping_3

        # Evaluate Slot 1 once at turn start to establish skill mode
        h, w = screen_color.shape[:2]
        rx1, ry1, rx2, ry2 = self.PHYSICAL_SLOTS["slot_1"]
        x1, y1 = int(w * rx1), int(h * ry1)
        x2, y2 = int(w * rx2), int(h * ry2)
        slot1_roi = screen_color[y1:y2, x1:x2]

        has_slot1_button = False
        if slot1_roi.size > 0:
            has_slot1_button = self._check_button_exists(slot1_roi)

        if has_slot1_button:
            self.locked_skill_mode = 4
            return default_4_mapping
        else:
            self.locked_skill_mode = 3
            return mapping_3

    def detect_available_skills(self, screen_color: Optional[np.ndarray]) -> Dict[str, bool]:
        """Evaluates skill availability (active vs. grayed-out) using normalized HSV channel metrics.

        Args:
            screen_color: Input BGR image array of the full screen.

        Returns:
            Dictionary mapping logical skill keys ('s1'-'s4') to boolean availability status.
        """
        if not self.is_turn_confirmed or not self._eval_single_frame_turn(screen_color):
            return {"s1": False, "s2": False, "s3": False, "s4": False}

        h, w = screen_color.shape[:2]
        mapping = self.get_skill_mapping(screen_color)
        skill_states: Dict[str, bool] = {}

        for logical_name in ["s1", "s2", "s3", "s4"]:
            physical_slot = mapping[logical_name]

            if physical_slot is None:
                skill_states[logical_name] = False
                continue

            rx1, ry1, rx2, ry2 = self.PHYSICAL_SLOTS[physical_slot]
            x1, y1 = int(w * rx1), int(h * ry1)
            x2, y2 = int(w * rx2), int(h * ry2)
            roi = screen_color[y1:y2, x1:x2]

            if roi.size == 0:
                skill_states[logical_name] = False
                continue

            norm_roi = cv2.resize(roi, self.norm_size, interpolation=cv2.INTER_LINEAR)
            core = norm_roi[12:52, 12:52]

            hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)
            s_channel = hsv[:, :, 1]
            v_channel = hsv[:, :, 2]

            v_mean = np.mean(v_channel)
            v_90 = np.percentile(v_channel, 90)
            s_mean = np.mean(s_channel)
            s_85 = np.percentile(s_channel, 85)

            is_active = (v_mean >= 75.0) and (v_90 >= 110.0) and ((s_mean >= 30.0) or (s_85 >= 45.0))
            skill_states[logical_name] = is_active

        return skill_states