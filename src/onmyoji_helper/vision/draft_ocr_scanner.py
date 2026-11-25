"""Draft Screen OCR Scanner Module for Onmyoji Tactical Helper.

Encapsulates RapidOCR lazy initialization (DirectML/CPU memory-isolated, `use_cls=True`,
2 intra-op threads), Ban-phase widened top-center ROI (`x >= 0.38`), dual-column horizontal
stitching, localized motion-diff throttling, and two-tier + multi-frame temporal consistency
filtering using domain tokens injected from `ShikigamiRepository`.
"""

import logging
import time
from collections.abc import Callable, Iterable
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger("OnmyojiHelper")

MIN_OPP_ROI_WIDTH: int = 500
OCR_FALLBACK_INTERVAL: float = 3.0
IN_BATTLE_DIFF_INTERVAL: float = 4.5
DEFAULT_MIN_OCR_CONFIDENCE: float = 0.65
DEFAULT_KNOWN_TOKEN_MIN_CONFIDENCE: float = 0.35
DEFAULT_MIN_CONSECUTIVE_FRAMES: int = 2

# UI-structural Ban-header keywords only (domain shikigami tokens are injected via `known_tokens`)
_DRAFT_HEADER_KEYWORDS: frozenset[str] = frozenset({"禁", "禁選", "禁选", "禁用"})


def _normalize_matched_known_token(text: str, known_whitelist: frozenset[str]) -> str | None:
    """Returns a canonical token key if `text` matches injected `known_tokens` or Ban headers."""
    cleaned = text.strip()
    if not cleaned:
        return None
    if cleaned in known_whitelist or cleaned in _DRAFT_HEADER_KEYWORDS:
        return cleaned
    if len(cleaned) <= 6 and any(kw in cleaned for kw in ("禁用", "禁選", "禁选")):
        remainder = (
            cleaned.replace("禁用", "").replace("禁選", "").replace("禁选", "").strip()
        )
        if not remainder or remainder in known_whitelist:
            return cleaned
    return None


def get_downsampled_blur(img_gray: np.ndarray | None) -> np.ndarray | None:
    """Generates a 120x80 Gaussian-blurred thumbnail for localized ROI motion-diff gating."""
    if img_gray is None or img_gray.size == 0:
        return None
    small = cv2.resize(img_gray, (120, 80), interpolation=cv2.INTER_NEAREST)
    return cv2.GaussianBlur(small, (5, 5), 0)


class DraftOcrScanner:
    """Performs motion-throttled dual-column RapidOCR scanning with temporal token verification."""

    def __init__(
        self,
        ocr_callable: Callable[[np.ndarray], tuple[Any, Any]] | None = None,
        min_confidence: float = DEFAULT_MIN_OCR_CONFIDENCE,
        known_token_min_confidence: float = DEFAULT_KNOWN_TOKEN_MIN_CONFIDENCE,
        min_consecutive_frames: int = DEFAULT_MIN_CONSECUTIVE_FRAMES,
        known_tokens: Iterable[str] | None = None,
    ) -> None:
        """Initializes motion-diff buffers, injected domain whitelist, and temporal streak state."""
        self._ocr_engine: Callable[[np.ndarray], tuple[Any, Any]] | None = ocr_callable
        self.min_confidence: float = float(min_confidence)
        self.known_token_min_confidence: float = float(known_token_min_confidence)
        self.min_consecutive_frames: int = max(1, int(min_consecutive_frames))
        self.known_whitelist: frozenset[str] = frozenset(
            t.strip() for t in (known_tokens or ()) if t and t.strip()
        )

        self.prev_opp_small_blur: np.ndarray | None = None
        self.prev_my_small_blur: np.ndarray | None = None
        self.last_ocr_time: float = 0.0
        self.last_window_size: tuple[int, int] = (0, 0)
        self._token_streaks: dict[str, dict[str, int]] = {}

    def _ensure_ocr_engine(self) -> Callable[[np.ndarray], tuple[Any, Any]]:
        """Lazily initializes RapidOCR on first scan so importing this module has zero overhead."""
        if self._ocr_engine is None:
            from rapidocr_onnxruntime import RapidOCR

            self._ocr_engine = RapidOCR(
                providers=["DmlExecutionProvider", "CPUExecutionProvider"],
                intra_op_num_threads=2,
                inter_op_num_threads=1,
                use_cls=True,
                det_box_thresh=0.3,
                det_unclip_ratio=1.8,
                det_thresh=0.3,
            )
        return self._ocr_engine

    def reset_motion_cache(self) -> None:
        """Resets ROI blur buffers and multi-frame OCR streak counters on match reset or resize."""
        self.prev_opp_small_blur = None
        self.prev_my_small_blur = None
        self._token_streaks.clear()

    def _has_pending_verifications(self) -> bool:
        """Returns True if any medium-confidence known token is awaiting next-frame confirmation."""
        target = self.min_consecutive_frames
        if target <= 1:
            return False
        return any(
            0 < streak < target
            for col_streaks in self._token_streaks.values()
            for streak in col_streaks.values()
        )

    def _filter_column_temporal(
        self,
        column_key: str,
        raw_items: list[Any],
        min_conf: float,
        known_conf: float,
    ) -> tuple[list[Any], int]:
        """Applies two-tier + cross-frame consecutive verification to a single ROI column.

        - `score >= min_conf` (0.65): immediate single-frame pass.
        - `known_conf <= score < min_conf` (0.35..0.65) AND in `known_whitelist`: requires
          appearing in `min_consecutive_frames` (2) consecutive scans of `column_key`.
        - Otherwise: dropped immediately.
        """
        if min_conf <= 0.0:
            return list(raw_items), 0

        prev_streaks = self._token_streaks.get(column_key, {})
        next_streaks: dict[str, int] = {}
        kept: list[Any] = []
        dropped = 0
        required_frames = self.min_consecutive_frames

        for res in raw_items:
            if not isinstance(res, (list, tuple)) or len(res) < 3 or res[2] is None:
                kept.append(res)
                continue
            try:
                score = float(res[2])
            except (TypeError, ValueError):
                kept.append(res)
                continue

            raw_text = str(res[1]) if len(res) >= 2 and res[1] is not None else ""
            matched_tok = _normalize_matched_known_token(raw_text, self.known_whitelist)

            if score >= min_conf:
                if matched_tok is not None:
                    next_streaks[matched_tok] = max(
                        next_streaks.get(matched_tok, 0),
                        prev_streaks.get(matched_tok, 0) + 1,
                        required_frames,
                    )
                kept.append(res)
            elif score >= known_conf and matched_tok is not None:
                # Increment strictly once per frame relative to `prev_streaks`
                new_streak = max(next_streaks.get(matched_tok, 0), prev_streaks.get(matched_tok, 0) + 1)
                next_streaks[matched_tok] = new_streak
                if new_streak >= required_frames:
                    kept.append(res)
                else:
                    dropped += 1
            else:
                dropped += 1

        self._token_streaks[column_key] = next_streaks
        return kept, dropped

    def scan_draft_rois(
        self,
        screen_color: np.ndarray,
        screen_gray: np.ndarray | None,
        is_active_state: bool,
        has_locked_opp_ban: bool,
        in_battle_five_picks: bool,
        now: float | None = None,
        min_confidence: float | None = None,
        known_token_min_confidence: float | None = None,
    ) -> tuple[list[Any], list[Any], bool]:
        """Scans ally and opponent draft columns (or lightweight settlement ROI in battle)."""
        now_ts = time.time() if now is None else float(now)
        h, w = screen_color.shape[:2]
        if (w, h) != self.last_window_size:
            if self.last_window_size != (0, 0):
                self.reset_motion_cache()
            self.last_window_size = (w, h)

        if screen_gray is None:
            screen_gray = cv2.cvtColor(screen_color, cv2.COLOR_BGR2GRAY)

        # Extend left boundary to 0.38 before Ban lock so top-center "禁選" badge is never clipped
        opp_x_start = int(w * 0.38) if not has_locked_opp_ban else int(w * 0.45)
        roi_opp_color = screen_color[0 : int(h * 0.52), opp_x_start : int(w * 0.92)]
        roi_opp_motion_gray = screen_gray[0 : int(h * 0.48), int(w * 0.35) : int(w * 0.90)]

        curr_opp_small_blur = get_downsampled_blur(roi_opp_motion_gray)
        curr_my_small_blur: np.ndarray | None = None
        roi_my_color: np.ndarray | None = None
        run_ocr = True

        if in_battle_five_picks:
            if self.prev_opp_small_blur is not None and curr_opp_small_blur is not None:
                diff_opp = cv2.mean(cv2.absdiff(curr_opp_small_blur, self.prev_opp_small_blur))[0]
                if diff_opp < 6.0 and (now_ts - self.last_ocr_time) < IN_BATTLE_DIFF_INTERVAL:
                    run_ocr = False
        else:
            roi_my_color = screen_color[int(h * 0.10) : int(h * 0.52), int(w * 0.10) : int(w * 0.20)]
            roi_my_motion_gray = screen_gray[int(h * 0.10) : int(h * 0.48), int(w * 0.10) : int(w * 0.20)]
            curr_my_small_blur = get_downsampled_blur(roi_my_motion_gray)

            # Never skip OCR before Ban lock or while a medium-confidence token awaits 2nd-frame confirmation
            if (
                is_active_state
                and has_locked_opp_ban
                and not self._has_pending_verifications()
                and self.prev_opp_small_blur is not None
                and curr_opp_small_blur is not None
                and self.prev_my_small_blur is not None
                and curr_my_small_blur is not None
            ):
                diff_opp = cv2.mean(cv2.absdiff(curr_opp_small_blur, self.prev_opp_small_blur))[0]
                diff_my = cv2.mean(cv2.absdiff(curr_my_small_blur, self.prev_my_small_blur))[0]
                if (
                    diff_opp < 8.0
                    and diff_my < 8.0
                    and (now_ts - self.last_ocr_time) < OCR_FALLBACK_INTERVAL
                ):
                    run_ocr = False

        if not run_ocr:
            return [], [], False

        ocr_fn = self._ensure_ocr_engine()
        self.prev_opp_small_blur = curr_opp_small_blur
        self.last_ocr_time = now_ts

        conf_thresh = self.min_confidence if min_confidence is None else float(min_confidence)
        known_thresh = (
            self.known_token_min_confidence
            if known_token_min_confidence is None
            else float(known_token_min_confidence)
        )
        ocr_opp: list[Any] = []
        ocr_my: list[Any] = []
        h_opp, w_opp = roi_opp_color.shape[:2] if roi_opp_color.ndim >= 2 else (0, 0)

        if in_battle_five_picks:
            if h_opp > 0 and w_opp > 0:
                if w_opp > 640:
                    scale = 640.0 / float(w_opp)
                    settle_img = cv2.resize(
                        roi_opp_color,
                        (640, max(1, int(h_opp * scale))),
                        interpolation=cv2.INTER_LINEAR,
                    )
                else:
                    settle_img = roi_opp_color
                ocr_settle, _ = ocr_fn(settle_img)
                raw_settle = list(ocr_settle) if ocr_settle else []
                ocr_opp, dropped = self._filter_column_temporal(
                    "settle", raw_settle, conf_thresh, known_thresh
                )
                if dropped > 0:
                    logger.debug(
                        f"[DraftOcrScanner] Filtered {dropped} unverified/low-confidence settlement OCR boxes."
                    )
            return ocr_opp, ocr_my, True

        self.prev_my_small_blur = curr_my_small_blur
        h_my, w_my = (
            roi_my_color.shape[:2]
            if roi_my_color is not None and roi_my_color.ndim >= 2
            else (0, 0)
        )
        if h_opp > 0 and w_opp > 0 and h_my > 0 and w_my > 0 and roi_my_color is not None:
            if w_opp < MIN_OPP_ROI_WIDTH:
                target_w_opp = 640
                target_h_opp = max(1, int(h_opp * (target_w_opp / float(w_opp))))
            else:
                target_w_opp = max(1, int(w_opp * 1.30))
                target_h_opp = max(1, int(h_opp * 1.30))

            img_opp_resized = cv2.resize(
                roi_opp_color, (target_w_opp, target_h_opp), interpolation=cv2.INTER_LINEAR
            )
            target_h_my = target_h_opp
            target_w_my = max(1, int(w_my * (target_h_my / float(h_my))))
            img_my_resized = cv2.resize(
                roi_my_color, (target_w_my, target_h_my), interpolation=cv2.INTER_LINEAR
            )

            combined_roi = np.hstack((img_my_resized, img_opp_resized))
            x_border = target_w_my

            ocr_combined, _ = ocr_fn(combined_roi)
            raw_my: list[Any] = []
            raw_opp: list[Any] = []
            if ocr_combined:
                for res in ocr_combined:
                    box = res[0]
                    x_center = sum(pt[0] for pt in box) / 4.0
                    if x_center < x_border:
                        raw_my.append(res)
                    else:
                        raw_opp.append(res)

            ocr_my, dropped_my = self._filter_column_temporal("my", raw_my, conf_thresh, known_thresh)
            ocr_opp, dropped_opp = self._filter_column_temporal("opp", raw_opp, conf_thresh, known_thresh)
            total_dropped = dropped_my + dropped_opp
            if total_dropped > 0:
                logger.debug(
                    f"[DraftOcrScanner] Held/filtered {total_dropped} draft OCR boxes pending confidence/temporal gate."
                )

        return ocr_opp, ocr_my, True
