"""Action Bar Avatar Detector Module for Onmyoji Tactical Helper.

Captures contrast-validated avatar fingerprint templates during the 5-pick draft reveal /
match preparation phase and identifies the active acting shikigami on the bottom-right
action-bar turn indicator via normalized template matching (`cv2.TM_CCOEFF_NORMED`).
"""

import cv2
import numpy as np

from onmyoji_helper.domain.models import UNKNOWN_SHIKIGAMI


class ActionBarDetector:
    """Detects the active acting shikigami on the action bar using normalized template matching."""

    # Calibrated threshold matching original battle-proven `daf2cea` (`0.15`), accounting for
    # rectangular draft card portraits vs. circular butterfly-framed action-bar turn indicators.
    CONF_THRESHOLD: float = 0.15

    def __init__(self) -> None:
        """Initializes relative avatar ROI ratios, normalized sizes, and template buffers."""
        self.px: float = 0.0720
        self.py: float = 0.0180
        self.pw: float = 0.0530
        self.ph: float = 0.0950
        self.pgap: float = 0.0570

        self.rx: float = 0.9410
        self.ry: float = 0.6120
        self.rw: float = 0.0510
        self.rh: float = 0.0900

        self.search_size: tuple[int, int] = (140, 140)
        self.template_size: tuple[int, int] = (110, 110)

        self.prep_templates: dict[str, np.ndarray] = {}
        self.slot_to_shiki: dict[str, str] = {}
        self.templates_locked: bool = False

    @property
    def has_locked_templates(self) -> bool:
        """Returns True if 5 valid draft portrait templates are locked for the current match."""
        return self.templates_locked and len(self.prep_templates) == 5

    def reset(self) -> None:
        """Clears captured avatar fingerprint templates and slot-name bindings on match end."""
        self.prep_templates.clear()
        self.slot_to_shiki.clear()
        self.templates_locked = False

    def bind_picks(self, our_picks: list[str] | None) -> None:
        """Updates `Slot 1..5 -> shikigami` bindings without discarding captured portrait images."""
        if our_picks and len(our_picks) == 5:
            self.slot_to_shiki = {f"Slot {idx + 1}": name for idx, name in enumerate(our_picks)}

    def capture_prep_templates(
        self,
        frame: np.ndarray | None,
        our_picks: list[str] | None = None,
        lock_on_success: bool = False,
    ) -> bool:
        """Captures 5 ally avatar fingerprint templates and optionally locks them for the match."""
        if self.has_locked_templates:
            self.bind_picks(our_picks)
            return True

        if frame is None:
            return False

        h, w = frame.shape[:2]
        if h <= 100 or w <= 100:
            return False

        box_w = int(w * self.pw)
        box_h = int(h * self.ph)
        y_top = int(h * self.py)
        gap = int(w * self.pgap)

        temp_crops: dict[str, np.ndarray] = {}
        for i in range(5):
            slot_num = 5 - i
            slot_name = f"Slot {slot_num}"

            x = int(w * self.px) + (i * gap)
            crop = frame[y_top : y_top + box_h, x : x + box_w]
            if crop.size == 0:
                return False

            mean_val, std_val = cv2.meanStdDev(crop)
            brightness = float(np.mean(mean_val))
            std_dev = float(np.mean(std_val))
            if std_dev < 12.0 or brightness < 15.0:
                return False

            temp_crops[slot_name] = cv2.resize(crop, self.template_size, interpolation=cv2.INTER_LINEAR)

        if len(temp_crops) == 5:
            self.prep_templates = temp_crops
            self.slot_to_shiki.clear()
            self.bind_picks(our_picks)
            if lock_on_success:
                self.templates_locked = True
            return True
        return False

    def identify_active_turn(
        self,
        frame: np.ndarray | None,
    ) -> tuple[str | None, str, float]:
        """Identifies the active acting shikigami slot and name via `cv2.TM_CCOEFF_NORMED`."""
        if not self.prep_templates or frame is None:
            return None, UNKNOWN_SHIKIGAMI, 0.0

        h, w = frame.shape[:2]
        pad_w = int(w * self.rw * 0.15)
        pad_h = int(h * self.rh * 0.15)

        ax = max(0, int(w * self.rx) - pad_w)
        ay = max(0, int(h * self.ry) - pad_h)
        aw = int(w * self.rw) + (pad_w * 2)
        ah = int(h * self.rh) + (pad_h * 2)

        current_crop = frame[ay : ay + ah, ax : ax + aw]
        if current_crop.size == 0:
            return None, UNKNOWN_SHIKIGAMI, 0.0

        search_img = cv2.resize(current_crop, self.search_size, interpolation=cv2.INTER_LINEAR)
        method_scores: dict[str, float] = {}
        for slot_name, tmpl_processed in self.prep_templates.items():
            res = cv2.matchTemplate(search_img, tmpl_processed, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            method_scores[slot_name] = float(max_val)

        sorted_scores = sorted(method_scores.items(), key=lambda x: x[1], reverse=True)
        best_slot, best_score = sorted_scores[0][0], sorted_scores[0][1]
        shiki_name = self.slot_to_shiki.get(best_slot, UNKNOWN_SHIKIGAMI)

        if best_score >= self.CONF_THRESHOLD:
            return best_slot, shiki_name, best_score
        return None, shiki_name, best_score
