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

            # Dynamic per-frame black padding crop (vectorized O(H+W), ~0.2ms, zero stale-crop risk)
            rows_nonblack = np.any(temp_gray > 0, axis=1)
            cols_nonblack = np.any(temp_gray > 0, axis=0)

            x0, y0 = 0, 0
            sym_pad_x, sym_pad_y = 0, 0
            if rows_nonblack.any() and cols_nonblack.any():
                row_indices = np.flatnonzero(rows_nonblack)
                col_indices = np.flatnonzero(cols_nonblack)
                y0, y1 = int(row_indices[0]), int(row_indices[-1]) + 1
                x0, x1 = int(col_indices[0]), int(col_indices[-1]) + 1
                img_bgr = img_bgr[y0:y1, x0:x1]
                img_gray = temp_gray[y0:y1, x0:x1] if need_gray else None

                # Distinguish symmetric fullscreen letterbox/pillarbox bars (top & bottom / left & right)
                # from right/bottom-only DWM DPI scaling margins (where x0=0, y0=0 and x1 = width / dpi_scale).
                right_black = width - x1
                bottom_black = height - y1
                if x0 >= 12 and right_black >= (x0 - 4):
                    sym_pad_x = x0
                if y0 >= 12 and bottom_black >= (y0 - 4):
                    sym_pad_y = y0
            else:
                img_gray = temp_gray if need_gray else None

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
                except Exception:
                    pass
            if save_dc is not None:
                try:
                    save_dc.DeleteDC()
                except Exception:
                    pass
            if mfc_dc is not None:
                try:
                    mfc_dc.DeleteDC()
                except Exception:
                    pass
            if hwnd_dc is not None and hwnd:
                try:
                    win32gui.ReleaseDC(hwnd, hwnd_dc)
                except Exception:
                    pass

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
