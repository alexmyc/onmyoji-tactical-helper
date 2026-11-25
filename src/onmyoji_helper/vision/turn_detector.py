"""Turn and Skill Bar Geometric Detector Module for Onmyoji Tactical Helper.

Uses CLAHE contrast enhancement, Hough Circle detection, circumferential Sobel gradient
verification, and HSV color analysis to detect player turns, map 3/4-skill button modes,
and evaluate skill availability without unsafe Python `id()` pointer caching.
"""

import time

import cv2
import numpy as np


class TurnDetector:
    """Detects active player turns and skill button availability via geometric CV."""

    def __init__(self) -> None:
        """Initializes normalized skill-bar ROIs, debouncing thresholds, and Sobel ring LUTs."""
        self.SKILL_BAR_ROI: tuple[float, float, float, float] = (0.665, 0.85, 0.998, 0.995)
        self.PHYSICAL_SLOTS: dict[str, tuple[float, float, float, float]] = {
            "slot_1": (0.693, 0.875, 0.759, 0.990),
            "slot_2": (0.771, 0.875, 0.837, 0.990),
            "slot_3": (0.849, 0.875, 0.915, 0.990),
            "slot_4": (0.927, 0.875, 0.993, 0.990),
        }

        self.norm_size: tuple[int, int] = (64, 64)

        self.turn_start_time: float = 0.0
        self.turn_loss_time: float = 0.0
        self.is_turn_confirmed: bool = False
        self.lockout_end_time: float = 0.0
        self.locked_skill_mode: int | None = None

        self.ACTIVATION_THRESHOLD: float = 0.40
        self.DEACTIVATION_GRACE: float = 0.15

        self._clahe: cv2.CLAHE = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        self._ring_cos: np.ndarray = np.cos(angles)
        self._ring_sin: np.ndarray = np.sin(angles)

        self._prev_bar_small: np.ndarray | None = None
        self._prev_bar_shape: tuple[int, int] | None = None
        self._last_raw_turn: bool = False
        self._last_eval_time: float = 0.0
        self._last_frame_seq: int | None = None
        self.ROI_STATIC_DIFF_THRESH: float = 2.0
        self.ROI_CACHE_MAX_AGE: float = 1.5

    def reset_cache(self) -> None:
        """Clears cached skill-bar ROI diff state on match reset or window resize."""
        self._prev_bar_small = None
        self._prev_bar_shape = None
        self._last_raw_turn = False
        self._last_eval_time = 0.0
        self._last_frame_seq = None

    def _verify_circle_ring_edge(
        self,
        gray_img: np.ndarray,
        cx: int,
        cy: int,
        r: int,
        grad_mag: np.ndarray | None = None,
    ) -> bool:
        """Verifies circumferential Sobel gradient density along a candidate circle perimeter."""
        h, w = gray_img.shape[:2]
        if grad_mag is None:
            sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.hypot(sobelx, sobely)

        pxs = (cx + r * self._ring_cos).astype(np.int32)
        pys = (cy + r * self._ring_sin).astype(np.int32)
        valid_mask = (pxs >= 0) & (pxs < w) & (pys >= 0) & (pys < h)
        if not np.any(valid_mask):
            return False

        hit_count = int(np.count_nonzero(grad_mag[pys[valid_mask], pxs[valid_mask]] >= 32.0))
        return hit_count >= 7

    def _eval_single_frame_turn(
        self,
        screen_color: np.ndarray | None,
        frame_seq: int | None = None,
    ) -> bool:
        """Evaluates a single frame for active skill bar buttons using HoughCircles + pixel-diff gating."""
        if screen_color is None:
            return False

        now = time.time()
        if frame_seq is not None and frame_seq == self._last_frame_seq:
            return self._last_raw_turn

        h, w = screen_color.shape[:2]
        rx1, ry1, rx2, ry2 = self.SKILL_BAR_ROI
        x1, y1 = int(w * rx1), int(h * ry1)
        x2, y2 = int(w * rx2), int(h * ry2)

        bar_roi = screen_color[y1:y2, x1:x2]
        if bar_roi.size == 0:
            return False

        roi_h, roi_w = bar_roi.shape[:2]
        gray = cv2.cvtColor(bar_roi, cv2.COLOR_BGR2GRAY)

        curr_small = cv2.resize(gray, (64, 32), interpolation=cv2.INTER_NEAREST)
        if (
            self._prev_bar_small is not None
            and self._prev_bar_shape == (roi_h, roi_w)
            and (now - self._last_eval_time) < self.ROI_CACHE_MAX_AGE
        ):
            diff = cv2.mean(cv2.absdiff(curr_small, self._prev_bar_small))[0]
            if diff < self.ROI_STATIC_DIFF_THRESH:
                if frame_seq is not None:
                    self._last_frame_seq = frame_seq
                return self._last_raw_turn

        self._prev_bar_small = curr_small
        self._prev_bar_shape = (roi_h, roi_w)
        self._last_eval_time = now
        if frame_seq is not None:
            self._last_frame_seq = frame_seq

        enhanced = self._clahe.apply(gray)
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
            maxRadius=max_r,
        )

        if circles is None:
            self._last_raw_turn = False
            return False

        circles_arr = np.uint16(np.around(circles[0]))
        valid_circles = []

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.hypot(sobelx, sobely)

        for c in circles_arr:
            cx, cy, cr = int(c[0]), int(c[1]), int(c[2])
            if int(roi_h * 0.20) <= cy <= int(roi_h * 0.80):
                if self._verify_circle_ring_edge(gray, cx, cy, cr, grad_mag=grad_mag):
                    valid_circles.append((cx, cy, cr))

        if len(valid_circles) < 2:
            self._last_raw_turn = False
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
            if valid_structure:
                break

        self._last_raw_turn = valid_structure
        return valid_structure

    def is_my_turn(
        self,
        screen_color: np.ndarray | None,
        frame_seq: int | None = None,
    ) -> bool:
        """Determines if it is currently the player's turn with temporal debouncing & lockout."""
        current_time = time.time()

        if current_time < self.lockout_end_time:
            self.turn_start_time = 0.0
            self.is_turn_confirmed = False
            self.locked_skill_mode = None
            return False

        raw_turn = self._eval_single_frame_turn(screen_color, frame_seq=frame_seq)

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
            if (
                self.is_turn_confirmed
                and (current_time - self.turn_loss_time) >= self.DEACTIVATION_GRACE
            ):
                self.is_turn_confirmed = False
                self.locked_skill_mode = None
                self.lockout_end_time = current_time + 0.50

        return self.is_turn_confirmed

    def _check_button_exists(self, roi_color: np.ndarray | None) -> bool:
        """Checks for a valid circular skill button within a physical slot ROI."""
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
            maxRadius=28,
        )
        if circles is None:
            return False

        circles_arr = np.uint16(np.around(circles[0]))
        cx_target, cy_target = 32, 30
        for c in circles_arr:
            cx, cy, cr = int(c[0]), int(c[1]), int(c[2])
            if abs(cx - cx_target) <= 10 and abs(cy - cy_target) <= 10:
                if self._verify_circle_ring_edge(gray, cx, cy, cr):
                    return True
        return False

    def get_skill_mapping(self, screen_color: np.ndarray | None) -> dict[str, str | None]:
        """Maps logical skills ('s1'..'s4') to physical slots ('slot_1'..'slot_4') for 3/4-skill units."""
        default_4_mapping: dict[str, str | None] = {
            "s1": "slot_1",
            "s2": "slot_2",
            "s3": "slot_3",
            "s4": "slot_4",
        }
        mapping_3: dict[str, str | None] = {
            "s1": "slot_2",
            "s2": "slot_3",
            "s3": "slot_4",
            "s4": None,
        }

        if not self.is_turn_confirmed or screen_color is None:
            return default_4_mapping
        if self.locked_skill_mode == 4:
            return default_4_mapping
        if self.locked_skill_mode == 3:
            return mapping_3

        h, w = screen_color.shape[:2]
        rx1, ry1, rx2, ry2 = self.PHYSICAL_SLOTS["slot_1"]
        x1, y1 = int(w * rx1), int(h * ry1)
        x2, y2 = int(w * rx2), int(h * ry2)
        slot1_roi = screen_color[y1:y2, x1:x2]

        has_slot1_button = self._check_button_exists(slot1_roi) if slot1_roi.size > 0 else False
        if has_slot1_button:
            self.locked_skill_mode = 4
            return default_4_mapping
        self.locked_skill_mode = 3
        return mapping_3

    def detect_available_skills(
        self,
        screen_color: np.ndarray | None,
        frame_seq: int | None = None,
    ) -> dict[str, bool]:
        """Evaluates skill availability (lit vs. grayed-out/cooldown) via normalized HSV metrics."""
        if not self.is_turn_confirmed or not self._eval_single_frame_turn(screen_color, frame_seq=frame_seq):
            return {"s1": False, "s2": False, "s3": False, "s4": False}

        h, w = screen_color.shape[:2]
        mapping = self.get_skill_mapping(screen_color)
        skill_states: dict[str, bool] = {}

        for logical_name in ("s1", "s2", "s3", "s4"):
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

            v_mean = float(np.mean(v_channel))
            v_90 = float(np.percentile(v_channel, 90))
            s_mean = float(np.mean(s_channel))
            s_85 = float(np.percentile(s_channel, 85))

            skill_states[logical_name] = (
                (v_mean >= 75.0) and (v_90 >= 110.0) and ((s_mean >= 30.0) or (s_85 >= 45.0))
            )

        return skill_states
