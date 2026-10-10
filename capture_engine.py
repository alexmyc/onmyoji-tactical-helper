# -*- coding: utf-8 -*-
"""Capture Engine Module for Onmyoji Tactical Helper.

This module provides high-speed Windows GDI BitBlt screen capture capabilities
targeting the game client window, automatically cropping black padding borders
and outputting normalized BGR and grayscale image arrays for computer vision pipelines.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np
import pygetwindow as gw
import win32con
import win32gui
import win32ui


class CaptureEngine:
    """High-performance GDI BitBlt window capture engine."""

    def __init__(self, target_title: str = "陰陽師Onmyoji") -> None:
        """Initializes the capture engine with a target window title filter.

        Args:
            target_title: Substring filter for identifying the target game client window.
        """
        self.target_title: str = target_title

    def find_windows(self) -> List[gw.Window]:
        """Searches active desktop windows matching the target title.

        Returns:
            List of matching PyGetWindow window instances excluding Chrome and File Explorer.
        """
        all_windows = gw.getAllWindows()
        found = [
            w for w in all_windows
            if self.target_title in w.title
            and "Google Chrome" not in w.title
            and "File Explorer" not in w.title
        ]
        return found

    def capture_window_bitblt(
        self,
        hwnd: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Captures the client area of the specified window using GDI BitBlt.

        Args:
            hwnd: Windows handle identifier of the target client window.

        Returns:
            Tuple of (img_gray, img_bgr) image arrays, or (None, None) if capture fails.
        """
        try:
            # Retrieve client rect dimensions (excluding window frame borders)
            left, top, right, bottom = win32gui.GetClientRect(hwnd)
            width, height = right - left, bottom - top

            if width <= 0 or height <= 0:
                return None, None

            # Acquire GDI Device Context for client area
            hwnd_dc = win32gui.GetDC(hwnd)
            mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
            save_dc = mfc_dc.CreateCompatibleDC()

            save_bitmap = win32ui.CreateBitmap()
            save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
            save_dc.SelectObject(save_bitmap)

            # Perform high-speed BitBlt capture
            save_dc.BitBlt((0, 0), (width, height), mfc_dc, (0, 0), win32con.SRCCOPY)

            signed_ints_array = save_bitmap.GetBitmapBits(True)
            img = np.frombuffer(signed_ints_array, dtype='uint8')
            img.shape = (height, width, 4)

            # Safely release Windows GDI object handles
            win32gui.DeleteObject(save_bitmap.GetHandle())
            save_dc.DeleteDC()
            mfc_dc.DeleteDC()
            win32gui.ReleaseDC(hwnd, hwnd_dc)

            img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # Crop black padding borders dynamically
            temp_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            mask = temp_gray > 0
            coords = np.argwhere(mask)

            if coords.size > 0:
                y0, x0 = coords.min(axis=0)
                y1, x1 = coords.max(axis=0) + 1
                img_bgr = img_bgr[y0:y1, x0:x1]

            # Generate aligned grayscale output array
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            return img_gray, img_bgr

        except Exception:
            return None, None