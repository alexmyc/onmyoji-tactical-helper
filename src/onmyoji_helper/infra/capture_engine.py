"""GDI BitBlt Screen Capture Engine for Onmyoji Tactical Helper.

Provides low-latency Windows GDI BitBlt client-area capture, dynamic black letterbox/pillarbox
cropping with exact offset tracking (`CropOffset`), and guaranteed GDI handle release.
"""

import logging
import time

import cv2
import numpy as np
import pygetwindow as gw
import win32con
import win32gui
import win32ui

from onmyoji_helper.domain.models import CaptureFrame, CropOffset

logger = logging.getLogger("OnmyojiHelper")

EXCLUDED_WINDOW_KEYWORDS: tuple[str, ...] = (
    "Google Chrome",
    "Microsoft​ Edge",
    "Microsoft Edge",
    "Mozilla Firefox",
    "File Explorer",
    "檔案總管",
    "Visual Studio Code",
    "Notepad",
    "Discord",
)


class CaptureEngine:
    """High-performance GDI BitBlt window capture engine with dynamic border-crop metadata."""

    def __init__(self, target_title: str = "陰陽師Onmyoji") -> None:
        """Initializes the capture engine with a target window title filter.

        Args:
            target_title: Substring filter for identifying the target game client window.
        """
        self.target_title: str = target_title
        self._last_err_log_time: float = 0.0
        # Per-HWND viewport cache: hwnd -> (client_w, client_h, logical_w, logical_h)
        # Prevents dark skill VFX or ink transitions from shrinking captured frame dimensions.
        self._viewport_cache: dict[int, tuple[int, int, int, int]] = {}

    def find_windows(self) -> list[gw.Window]:
        """Searches desktop windows matching the target game title while excluding browsers/editors."""
        all_windows = gw.getAllWindows()
        found: list[gw.Window] = []
        for win in all_windows:
            title = win.title or ""
            if self.target_title not in title:
                continue
            if any(excluded in title for excluded in EXCLUDED_WINDOW_KEYWORDS):
                continue
            found.append(win)
        return found

    def capture_frame(
        self,
        hwnd: int,
        need_gray: bool = True,
        window_title: str = "",
    ) -> CaptureFrame | None:
        """Captures a window frame via GDI BitBlt and preserves exact black-border crop offsets.

        Args:
            hwnd: Windows handle identifier of the target client window.
            need_gray: Whether to retain and return the aligned grayscale array.
            window_title: Optional human-readable window title for diagnostics.

        Returns:
            `CaptureFrame` containing BGR/gray arrays and `CropOffset`, or `None` on failure.
        """
        hwnd_dc = None
        mfc_dc = None
        save_dc = None
        save_bitmap = None

        try:
            if not hwnd or not win32gui.IsWindow(hwnd):
                return None

            left, top, right, bottom = win32gui.GetClientRect(hwnd)
            width, height = right - left, bottom - top
            if width <= 0 or height <= 0:
                return None

            hwnd_dc = win32gui.GetDC(hwnd)
            if not hwnd_dc:
                return None

            mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
            save_dc = mfc_dc.CreateCompatibleDC()

            save_bitmap = win32ui.CreateBitmap()
            save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
            save_dc.SelectObject(save_bitmap)

            save_dc.BitBlt((0, 0), (width, height), mfc_dc, (0, 0), win32con.SRCCOPY)

            signed_ints_array = save_bitmap.GetBitmapBits(True)
            img = np.frombuffer(signed_ints_array, dtype="uint8")
            img.shape = (height, width, 4)

            img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            temp_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            # Viewport-stabilized border & DWM-DPI margin crop:
            # Dark cutscenes / ink transitions must NEVER shrink the established logical viewport
            # (e.g. 1136x640 -> 705x440) or shift origin (x0, y0) asymmetrically.
            rows_nonblack = np.any(temp_gray > 0, axis=1)
            cols_nonblack = np.any(temp_gray > 0, axis=0)

            cached = self._viewport_cache.get(int(hwnd))
            if cached is not None and cached[0] == width and cached[1] == height:
                logical_w, logical_h = cached[2], cached[3]
            else:
                logical_w, logical_h = 0, 0

            sym_pad_x, sym_pad_y = 0, 0
            if rows_nonblack.any() and cols_nonblack.any():
                row_indices = np.flatnonzero(rows_nonblack)
                col_indices = np.flatnonzero(cols_nonblack)
                raw_y0, raw_y1 = int(row_indices[0]), int(row_indices[-1]) + 1
                raw_x0, raw_x1 = int(col_indices[0]), int(col_indices[-1]) + 1

                # DWM DPI scale is isotropic: (raw_x0 + raw_x1) / width == (raw_y0 + raw_y1) / height
                est_lw = min(width, raw_x0 + raw_x1) if raw_x0 <= 64 else raw_x1
                est_lh = min(height, raw_y0 + raw_y1) if raw_y0 <= 96 else raw_y1
                if (
                    est_lw >= int(width * 0.45)
                    and est_lh >= int(height * 0.45)
                    and abs((width / float(est_lw)) - (height / float(est_lh))) <= 0.08
                ):
                    logical_w = max(logical_w, est_lw)
                    logical_h = max(logical_h, est_lh)
                    self._viewport_cache[int(hwnd)] = (width, height, logical_w, logical_h)
                elif logical_w > 0 and logical_h > 0:
                    if raw_x1 > logical_w and raw_y1 > logical_h:
                        logical_w = min(width, raw_x1)
                        logical_h = min(height, raw_y1)
                        self._viewport_cache[int(hwnd)] = (width, height, logical_w, logical_h)

                eff_lw = logical_w if logical_w > 0 else width
                eff_lh = logical_h if logical_h > 0 else height

                # Only crop inner letterbox/pillarbox when TWO opposite edges are symmetric while
                # the perpendicular axis spans the full logical viewport (true 16:10 / 21:9 bars).
                crop_x0, crop_x1 = 0, eff_lw
                crop_y0, crop_y1 = 0, eff_lh
                if (
                    raw_x0 <= 4
                    and abs(eff_lw - raw_x1) <= 4
                    and raw_y0 >= 12
                    and abs((eff_lh - raw_y1) - raw_y0) <= 8
                ):
                    crop_y0, crop_y1 = raw_y0, raw_y1
                    sym_pad_y = raw_y0
                elif (
                    raw_y0 <= 4
                    and abs(eff_lh - raw_y1) <= 4
                    and raw_x0 >= 12
                    and abs((eff_lw - raw_x1) - raw_x0) <= 8
                ):
                    crop_x0, crop_x1 = raw_x0, raw_x1
                    sym_pad_x = raw_x0

                img_bgr = img_bgr[crop_y0:crop_y1, crop_x0:crop_x1]
                img_gray = temp_gray[crop_y0:crop_y1, crop_x0:crop_x1] if need_gray else None
            else:
                eff_lw = logical_w if logical_w > 0 else width
                eff_lh = logical_h if logical_h > 0 else height
                img_bgr = img_bgr[0:eff_lh, 0:eff_lw]
                img_gray = temp_gray[0:eff_lh, 0:eff_lw] if need_gray else None

            return CaptureFrame(
                bgr=img_bgr,
                gray=img_gray,
                hwnd=int(hwnd),
                window_title=window_title,
                crop=CropOffset(x0=sym_pad_x, y0=sym_pad_y, client_w=width, client_h=height),
            )

        except Exception as exc:
            now = time.time()
            if now - self._last_err_log_time >= 5.0:
                self._last_err_log_time = now
                logger.warning(f"⚠️ [CaptureEngine] BitBlt 截圖異常 (hwnd={hwnd}): {exc}")
            return None

        finally:
            if save_bitmap is not None:
                try:
                    win32gui.DeleteObject(save_bitmap.GetHandle())
                except Exception as exc:
                    logger.debug(f"[CaptureEngine] DeleteObject cleanup note: {exc}")
            if save_dc is not None:
                try:
                    save_dc.DeleteDC()
                except Exception as exc:
                    logger.debug(f"[CaptureEngine] save_dc.DeleteDC cleanup note: {exc}")
            if mfc_dc is not None:
                try:
                    mfc_dc.DeleteDC()
                except Exception as exc:
                    logger.debug(f"[CaptureEngine] mfc_dc.DeleteDC cleanup note: {exc}")
            if hwnd_dc is not None and hwnd:
                try:
                    win32gui.ReleaseDC(hwnd, hwnd_dc)
                except Exception as exc:
                    logger.debug(f"[CaptureEngine] ReleaseDC cleanup note: {exc}")

    def capture_window_bitblt(
        self,
        hwnd: int,
        need_gray: bool = True,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Backwards-compatible tuple wrapper returning `(img_gray, img_bgr)`."""
        frame = self.capture_frame(hwnd=hwnd, need_gray=need_gray)
        if frame is None:
            return None, None
        return frame.gray, frame.bgr
