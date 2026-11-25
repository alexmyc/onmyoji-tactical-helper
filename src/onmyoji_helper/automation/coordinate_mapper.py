"""DPI-Aware Physical Coordinate Mapper for Onmyoji Tactical Helper.

Provides the single authoritative projection from cropped BitBlt image coordinates
`(img_x, img_y)` onto physical client/screen pixels, shared identically by both
`BattleOverlayUI` and `AutoBattleExecutor`.

Why DWM DPI-virtualized windows require logical-buffer scaling:
    When a DPI-unaware game window (e.g. logical `1144x648`) runs on a 125% / 150% DPI
    display while this helper is Per-Monitor DPI Aware (`SetProcessDpiAwareness(2)`),
    `GetClientRect(hwnd)` returns physical pixels (`1430x810` at 125%), whereas `GetDC(hwnd)` +
    `BitBlt` captures the unscaled `1144x648` DWM surface at `(0, 0)` with black right/bottom
    margins. Scaling by `target_w / (img_w + 2 * pad_x)` (`1430 / 1144 = 1.25`) restores exact
    1:1 physical alignment across 100%, 125%, 150%, and 200% DPI as well as 16:10 letterboxes.
"""

import logging
import random
from collections.abc import Callable

import win32gui

from onmyoji_helper.domain.models import CropOffset
from onmyoji_helper.infra.win32_window import get_root_hwnd

logger = logging.getLogger("OnmyojiHelper")

ScreenBounds = tuple[int, int, int, int]
MappedScreenPoint = tuple[int, int, ScreenBounds]


def project_cropped_to_physical_client(
    img_x: float,
    img_y: float,
    img_w: float,
    img_h: float,
    target_w: float,
    target_h: float,
    crop: CropOffset | None = None,
) -> tuple[float, float]:
    """Projects `(img_x, img_y)` from cropped BitBlt image space to physical client pixels.

    Handles both DWM DPI scaling (`100% / 125% / 150% / 200%`, where `img_w = target_w / dpi`)
    and symmetric fullscreen letterbox/pillarbox padding (`pad_x = crop.x0`, `pad_y = crop.y0`).
    """
    if img_w <= 0 or img_h <= 0:
        return float(img_x), float(img_y)

    pad_x = float(crop.x0) if (crop is not None and crop.x0 > 0) else 0.0
    pad_y = float(crop.y0) if (crop is not None and crop.y0 > 0) else 0.0

    logical_full_w = float(img_w) + (2.0 * pad_x)
    logical_full_h = float(img_h) + (2.0 * pad_y)

    scale_x = float(target_w) / logical_full_w if logical_full_w > 0 else 1.0
    scale_y = float(target_h) / logical_full_h if logical_full_h > 0 else 1.0

    return (pad_x + float(img_x)) * scale_x, (pad_y + float(img_y)) * scale_y


class CoordinateMapper:
    """Converts cropped image coordinates into Per-Monitor DPI-Aware physical screen coordinates."""

    def __init__(
        self,
        custom_mapper: Callable[[int, int, int, int, int], MappedScreenPoint | None] | None = None,
    ) -> None:
        """Initializes the coordinate mapper with an optional injectable test stub."""
        self._custom_mapper = custom_mapper

    def map_image_to_screen(
        self,
        hwnd: int,
        img_x: float,
        img_y: float,
        img_w: int,
        img_h: int,
        crop: CropOffset | None = None,
    ) -> MappedScreenPoint | None:
        """Maps a cropped-frame `(img_x, img_y)` coordinate to physical screen pixels."""
        if img_w <= 0 or img_h <= 0:
            return None
        if img_x <= 0 or img_y <= 0 or img_x >= img_w or img_y >= img_h:
            return None

        if self._custom_mapper is not None:
            return self._custom_mapper(hwnd, int(round(img_x)), int(round(img_y)), img_w, img_h)

        try:
            if not hwnd or not win32gui.IsWindow(hwnd):
                return None

            left, top, right, bottom = win32gui.GetClientRect(hwnd)
            client_w, client_h = right - left, bottom - top
            if client_w <= 50 or client_h <= 50:
                return None

            origin_x, origin_y = win32gui.ClientToScreen(hwnd, (0, 0))
            bound_left, bound_top = origin_x, origin_y
            bound_right, bound_bottom = origin_x + client_w, origin_y + client_h

            root_hwnd = get_root_hwnd(hwnd)
            if root_hwnd and win32gui.IsWindow(root_hwnd):
                r_left, r_top, r_right, r_bottom = win32gui.GetWindowRect(root_hwnd)
                if r_right > r_left and r_bottom > r_top:
                    inter_left = max(bound_left, r_left)
                    inter_top = max(bound_top, r_top)
                    inter_right = min(bound_right, r_right)
                    inter_bottom = min(bound_bottom, r_bottom)
                    if (inter_right - inter_left) > 50 and (inter_bottom - inter_top) > 50:
                        bound_left, bound_top, bound_right, bound_bottom = (
                            inter_left,
                            inter_top,
                            inter_right,
                            inter_bottom,
                        )

            rel_px, rel_py = project_cropped_to_physical_client(
                img_x=img_x,
                img_y=img_y,
                img_w=img_w,
                img_h=img_h,
                target_w=client_w,
                target_h=client_h,
                crop=crop,
            )
            sx = origin_x + int(round(rel_px))
            sy = origin_y + int(round(rel_py))

            if not (bound_left <= sx <= bound_right and bound_top <= sy <= bound_bottom):
                logger.warning(
                    f"⚠️ [CoordinateMapper] 轉換後座標超出視窗範圍: ({sx},{sy}) not in "
                    f"({bound_left},{bound_top},{bound_right},{bound_bottom})"
                )
                return None

            sx = max(bound_left + 2, min(bound_right - 2, sx))
            sy = max(bound_top + 2, min(bound_bottom - 2, sy))
            return sx, sy, (bound_left, bound_top, bound_right, bound_bottom)
        except Exception as exc:
            logger.debug(f"⚠️ [CoordinateMapper] 座標轉換失敗: {exc}")
            return None

    @staticmethod
    def apply_gaussian_jitter(
        x: int,
        y: int,
        max_offset: float,
        bounds: ScreenBounds,
    ) -> tuple[int, int]:
        """Applies clamped Gaussian coordinate jitter strictly within `bounds`."""
        if max_offset <= 0.0:
            bx1, by1, bx2, by2 = bounds
            return max(bx1, min(bx2, int(x))), max(by1, min(by2, int(y)))

        sigma = max(0.5, max_offset * 0.42)
        dx = max(-max_offset, min(max_offset, random.gauss(0.0, sigma)))
        dy = max(-max_offset, min(max_offset, random.gauss(0.0, sigma)))
        bx1, by1, bx2, by2 = bounds
        nx = int(round(max(bx1 + 4, min(bx2 - 4, x + dx))))
        ny = int(round(max(by1 + 4, min(by2 - 4, y + dy))))
        return nx, ny
