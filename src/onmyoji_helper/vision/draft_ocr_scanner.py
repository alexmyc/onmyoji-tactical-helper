"""Draft Screen OCR Scanner Module for Onmyoji Tactical Helper.

Encapsulates RapidOCR lazy initialization (DirectML/CPU memory-isolated, `use_cls=True`,
2 intra-op threads), Ban-phase widened top-center ROI (`x >= 0.38`), dual-column horizontal
stitching, localized motion-diff throttling, and two-tier whitelist confidence filtering
(preserving genuine single-character shikigami such as `季`, `瀧`, `離`, `神` while dropping
low-confidence background OCR noise).
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

# Canonical single-character & short-name Onmyoji shikigami and draft-phase keywords whose
# isolated CJK glyphs frequently score 0.35..0.65 in RapidOCR.
KNOWN_SHORT_SHIKIGAMI_WHITELIST: frozenset[str] = frozenset(
    {
        "季",
        "瀧",
        "荒",
        "離",
        "神",
        "連",
        "石",
        "封",
        "禪",
        "須",
        "千",
        "帝",
        "葛",
        "骷",
        "閻",
        "火",
        "言",
        "玲",
        "吞",
        "鏡",
        "薰",
        "鴆",
        "覺",
        "鯉",
        "白",
        "切",
        "花",
        "燈",
        "蓮",
        "嵐",
        "岳",
        "蛇",
        "鹿",
        "緣",
        "熊",
        "草",
        "葉",
        "桃",
        "追月",
        "百羽",
        "酒吞",
        "神荒",
        "不知火",
        "天照",
        "葛葉",
        "千姬",
        "言靈",
        "閻魔",
        "禁",
        "禁選",
        "禁选",
        "禁用",
    }
)


def _is_whitelisted_draft_token(text: str, whitelist: frozenset[str]) -> bool:
    """Returns True if recognized text matches a known single-char or short shikigami/draft token."""
    cleaned = text.strip()
    if not cleaned:
        return False
    if cleaned in whitelist:
        return True
    # Support short Ban badges attached directly to a single-char shikigami (e.g. "禁用季", "禁選瀧")
    if len(cleaned) <= 4 and any(
        kw in cleaned for kw in ("禁用", "禁選", "禁选")
    ):
        remainder = (
            cleaned.replace("禁用", "").replace("禁選", "").replace("禁选", "").strip()
        )
        if not remainder or remainder in whitelist:
            return True
    return False


def _passes_confidence(
    res: Any,
    min_confidence: float,
    known_token_min_confidence: float,
    whitelist: frozenset[str],
) -> bool:
    """Evaluates two-tier OCR confidence: `min_confidence` (0.65) general, `0.35` for known tokens."""
    if min_confidence <= 0.0:
        return True
    if not isinstance(res, (list, tuple)) or len(res) < 3 or res[2] is None:
        return True
    try:
        score = float(res[2])
    except (TypeError, ValueError):
        return True

    if score >= min_confidence:
        return True
    if score < known_token_min_confidence:
        return False

    raw_text = str(res[1]) if len(res) >= 2 and res[1] is not None else ""
    return _is_whitelisted_draft_token(raw_text, whitelist)


def get_downsampled_blur(img_gray: np.ndarray | None) -> np.ndarray | None:
    """Generates a 120x80 Gaussian-blurred thumbnail for localized ROI motion-diff gating."""
    if img_gray is None or img_gray.size == 0:
        return None
    small = cv2.resize(img_gray, (120, 80), interpolation=cv2.INTER_NEAREST)
    return cv2.GaussianBlur(small, (5, 5), 0)


class DraftOcrScanner:
    """Performs motion-throttled dual-column RapidOCR scanning for Ban/Pick & settlement screens."""

    def __init__(
        self,
        ocr_callable: Callable[[np.ndarray], tuple[Any, Any]] | None = None,
        min_confidence: float = DEFAULT_MIN_OCR_CONFIDENCE,
        known_token_min_confidence: float = DEFAULT_KNOWN_TOKEN_MIN_CONFIDENCE,
        known_tokens: Iterable[str] | None = None,
    ) -> None:
        """Initializes motion-diff buffers, two-tier confidence gates, and optional OCR backend."""
        self._ocr_engine: Callable[[np.ndarray], tuple[Any, Any]] | None = ocr_callable
        self.min_confidence: float = float(min_confidence)
        self.known_token_min_confidence: float = float(known_token_min_confidence)
        extra_tokens = {t.strip() for t in (known_tokens or ()) if t and t.strip()}
        self.known_whitelist: frozenset[str] = KNOWN_SHORT_SHIKIGAMI_WHITELIST | extra_tokens
        self.prev_opp_small_blur: np.ndarray | None = None
        self.prev_my_small_blur: np.ndarray | None = None
        self.last_ocr_time: float = 0.0
        self.last_window_size: tuple[int, int] = (0, 0)

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
        """Resets previous ROI blur buffers on manual match reset or window resize."""
        self.prev_opp_small_blur = None
        self.prev_my_small_blur = None

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

            # Never skip OCR before Ban is locked; apply ROI motion-diff throttling only during Draft 1~5
            if (
                is_active_state
                and has_locked_opp_ban
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
                if ocr_settle:
                    ocr_opp = [
                        r
                        for r in ocr_settle
                        if _passes_confidence(r, conf_thresh, known_thresh, self.known_whitelist)
                    ]
                    dropped = len(ocr_settle) - len(ocr_opp)
                    if dropped > 0:
                        logger.debug(
                            f"[DraftOcrScanner] Filtered {dropped} low-confidence settlement OCR boxes."
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
            if ocr_combined:
                dropped = 0
                for res in ocr_combined:
                    if not _passes_confidence(res, conf_thresh, known_thresh, self.known_whitelist):
                        dropped += 1
                        continue
                    box = res[0]
                    x_center = sum(pt[0] for pt in box) / 4.0
                    if x_center < x_border:
                        ocr_my.append(res)
                    else:
                        ocr_opp.append(res)
                if dropped > 0:
                    logger.debug(
                        f"[DraftOcrScanner] Filtered {dropped} low-confidence draft OCR boxes."
                    )

        return ocr_opp, ocr_my, True
