# -*- coding: utf-8 -*-
"""Action Bar Detector Module for Onmyoji Tactical Helper.

This module provides computer vision template-matching mechanisms to capture clean
shikigami avatar templates during the match preparation phase and identify the active
acting shikigami on the turn action bar in real-time.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class ActionBarDetector:
    """Detects active acting shikigami on the action bar using normalized template matching."""

    def __init__(self) -> None:
        """Initializes relative bounding box coordinates, template sizes, and memory dicts."""
        # Relative coordinates for top avatar slots (Slot 1 to Slot 5)
        self.px: float = 0.0720
        self.py: float = 0.0180
        self.pw: float = 0.0530
        self.ph: float = 0.0950
        self.pgap: float = 0.0570

        # Relative coordinates for bottom-right active turn avatar frame
        self.rx: float = 0.9410
        self.ry: float = 0.6120
        self.rw: float = 0.0510
        self.rh: float = 0.0900

        # Normalized search and template sizes (140x140 search window vs 110x110 template)
        self.search_size: Tuple[int, int] = (140, 140)
        self.template_size: Tuple[int, int] = (110, 110)

        # Stores normalized match start fingerprint templates (110x110 arrays)
        self.prep_templates: Dict[str, np.ndarray] = {}
        # Mapping table linking slot keys to shikigami short names (e.g., 'Slot 1' -> 'SP鹿')
        self.slot_to_shiki: Dict[str, str] = {}

    def capture_prep_templates(
        self,
        frame: Optional[np.ndarray],
        our_picks: Optional[List[str]] = None,
        delay: int = 0
    ) -> bool:
        """Captures avatar fingerprint templates during preparation phase and binds slot names.

        Args:
            frame: Input BGR image array of the game screen.
            our_picks: Optional list of player drafted character names (length 5).
            delay: Unused legacy delay parameter retained for backward compatibility.

        Returns:
            True if all 5 slot templates are successfully captured and normalized, False otherwise.
        """
        if frame is None:
            return False

        h, w = frame.shape[:2]
        self.prep_templates.clear()
        self.slot_to_shiki.clear()

        # Bind Slot 1-5 to player drafted character names (Slot 1 = Pick 1, Slot 5 = Pick 5)
        if our_picks and len(our_picks) == 5:
            for idx, name in enumerate(our_picks):
                self.slot_to_shiki[f"Slot {idx + 1}"] = name

        box_w = int(w * self.pw)
        box_h = int(h * self.ph)
        y_top = int(h * self.py)
        gap = int(w * self.pgap)

        for i in range(5):
            slot_num = 5 - i
            slot_name = f"Slot {slot_num}"

            x = int(w * self.px) + (i * gap)
            crop = frame[y_top:y_top + box_h, x:x + box_w]

            if crop.size > 0:
                norm_tmpl = cv2.resize(crop, self.template_size, interpolation=cv2.INTER_LINEAR)
                self.prep_templates[slot_name] = norm_tmpl

        return len(self.prep_templates) == 5

    def identify_active_turn(
        self,
        frame: Optional[np.ndarray]
    ) -> Tuple[Optional[str], str, float]:
        """Identifies the active acting shikigami on the current turn using template matching.

        Args:
            frame: Input BGR image array of the full screen.

        Returns:
            Tuple containing:
            - Target slot key string (e.g., 'Slot 3'), or None if confidence is below threshold.
            - Character name string, or '未知' if unassigned.
            - Matching confidence score float.
        """
        if not self.prep_templates or frame is None:
            return None, "未知", 0.0

        h, w = frame.shape[:2]

        # Expand sliding search ROI by 15% to absorb resolution scaling offsets
        pad_w = int(w * self.rw * 0.15)
        pad_h = int(h * self.rh * 0.15)

        ax = max(0, int(w * self.rx) - pad_w)
        ay = max(0, int(h * self.ry) - pad_h)
        aw = int(w * self.rw) + (pad_w * 2)
        ah = int(h * self.rh) + (pad_h * 2)

        current_crop = frame[ay:ay + ah, ax:ax + aw]
        if current_crop.size == 0:
            return None, "未知", 0.0

        # Normalize cropped region to 140x140 search array
        search_img = cv2.resize(current_crop, self.search_size, interpolation=cv2.INTER_LINEAR)
        method_scores: Dict[str, float] = {}

        # Perform template matching against 110x110 fingerprint templates
        for slot_name, tmpl_processed in self.prep_templates.items():
            res = cv2.matchTemplate(search_img, tmpl_processed, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            method_scores[slot_name] = float(max_val)

        sorted_scores = sorted(method_scores.items(), key=lambda x: x[1], reverse=True)
        best_slot, best_score = sorted_scores[0][0], sorted_scores[0][1]

        # Retrieve bound character short name
        shiki_name = self.slot_to_shiki.get(best_slot, "未知式神")

        if best_score >= 0.15:
            return best_slot, shiki_name, best_score

        return None, shiki_name if shiki_name != "未知式神" else "未知", best_score