# -*- coding: utf-8 -*-
"""Main Entry Point for Onmyoji Tactical Helper.

This module initializes application configuration, manages DPI awareness,
and orchestrates background screen capture, OCR draft detection, and UI thread safety.
"""

import os

# Limit CPU threading to prevent 100% CPU utilization on low-spec hardware
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

import queue
import sys
import threading
import time
import tkinter as tk
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ai_tactical_manager import AITacticalManager
from auth_module import check_activation
from capture_engine import CaptureEngine
from db_manager import load_cached_data
from logic import build_database, get_recommendation
from mode_controller import ModeController
from rapidocr_onnxruntime import RapidOCR
from ui_module import OnmyojiHelperUI

# ==========================================
# System Performance & State Constants
# ==========================================
INTERVAL_SLEEP_MODE: float = 2.5   # Sleep mode scanning interval in seconds
INTERVAL_ACTIVE_MODE: float = 1.0  # Active battle drafting scanning interval in seconds
COOLDOWN_OPP_STATIC: float = 1.2   # OCR cooldown time when image motion is static

MIN_OPP_ROI_WIDTH: int = 500
MIN_MY_ROI_WIDTH: int = 150

STATE_SLEEP: int = 0
STATE_ACTIVE: int = 1
# ==========================================

# Global System Managers & UI Queue
tactical_mgr: AITacticalManager = AITacticalManager()
mode_controller: ModeController = ModeController(tactical_mgr)
ui_queue: queue.Queue = queue.Queue()

# RapidOCR configuration isolated to DirectML / CPU execution providers
# to prevent CUDA VRAM collisions with background LLM / YOLO models
ocr_engine: RapidOCR = RapidOCR(
    providers=["DmlExecutionProvider", "CPUExecutionProvider"],
    det_box_thresh=0.3,
    det_unclip_ratio=1.8,
    det_thresh=0.3
)


def queue_ui_update(
    remembered_opp_ban: Optional[str],
    opp_draft: List[str],
    my_draft: List[str],
    recs: List[str],
    preds: List[str],
    status: str
) -> None:
    """Queues a UI update payload for consumption by the Tkinter main thread.

    Args:
        remembered_opp_ban: Detected opponent banned character name, if any.
        opp_draft: List of detected opponent drafted characters.
        my_draft: List of detected player drafted characters.
        recs: Recommended counter picks for the player.
        preds: Predicted choices for the opponent.
        status: Status description text displayed on the user interface.
    """
    ui_queue.put((remembered_opp_ban, opp_draft, my_draft, recs, preds, status))


def process_ui_queue(root_window: tk.Tk, ui_app_instance: OnmyojiHelperUI) -> None:
    """Processes pending UI update messages on the main Tkinter thread loop.

    Args:
        root_window: The primary Tkinter root window instance.
        ui_app_instance: The application main user interface instance.
    """
    while not ui_queue.empty():
        try:
            data = ui_queue.get_nowait()
            ui_app_instance.update_info(*data)
        except queue.Empty:
            break
    root_window.after(50, process_ui_queue, root_window, ui_app_instance)


def get_downsampled_blur(img_gray: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Generates a downsampled and Gaussian-blurred image for frame motion analysis.

    Args:
        img_gray: Input grayscale OpenCV frame array.

    Returns:
        Processed downsampled image array, or None if the input array is empty or invalid.
    """
    if img_gray is None or img_gray.size == 0:
        return None
    small = cv2.resize(img_gray, (120, 80), interpolation=cv2.INTER_NEAREST)
    return cv2.GaussianBlur(small, (5, 5), 0)


def ocr_worker(
    ui_app: OnmyojiHelperUI,
    compiled_dbs: Dict[str, Dict[str, Any]],
    full_to_short_map: Dict[str, str]
) -> None:
    """Background thread executing screen capture and OCR draft recognition.

    Args:
        ui_app: Reference to the primary UI application.
        compiled_dbs: Compiled character recommendation database mapping.
        full_to_short_map: Mapping dictionary from character full name to shorthand.
    """
    clicker = CaptureEngine()
    queue_ui_update(None, [], [], [], [], "正在尋找陰陽師視窗...")

    remembered_opp_ban: Optional[str] = None
    opp_draft_fullnames: List[str] = []

    prev_opp_small_blur: Optional[np.ndarray] = None
    prev_my_small_blur: Optional[np.ndarray] = None
    last_ocr_time: float = 0.0

    current_my_draft: List[str] = []
    last_opp_draft_len: int = 0

    current_state: int = STATE_SLEEP
    current_sleep_interval: float = INTERVAL_SLEEP_MODE

    last_window_w: int = 0
    last_window_h: int = 0
    last_wakeup_time: float = time.time()

    has_triggered_five_picks: bool = False

    sorted_dict: List[Tuple[str, str]] = sorted(
        full_to_short_map.items(), key=lambda x: len(x[0]), reverse=True
    )
    print("🚀 RapidOCR thread started (DirectML / CPU memory-isolated high-speed edition)...")

    while True:
        ui_system_choice = ui_app.selected_ban.get()
        current_system = compiled_dbs.get(ui_system_choice)

        if not current_system:
            time.sleep(1)
            continue

        active_db = current_system["db"]
        my_ban = current_system["my_ban"]

        if ui_app.needs_reset:
            remembered_opp_ban = None
            opp_draft_fullnames.clear()
            current_my_draft.clear()
            last_opp_draft_len = 0
            prev_opp_small_blur = None
            prev_my_small_blur = None
            ui_app.needs_reset = False
            current_state = STATE_SLEEP
            has_triggered_five_picks = False
            if hasattr(mode_controller, "clear_draft_data"):
                mode_controller.clear_draft_data()
            queue_ui_update(
                remembered_opp_ban,
                [],
                [],
                [],
                [],
                f"已重置，當前數據：{ui_system_choice}"
            )

        try:
            windows = clicker.find_windows()
            if not windows:
                queue_ui_update(
                    remembered_opp_ban,
                    [],
                    [],
                    [],
                    [],
                    "找不到遊戲視窗，請開啟陰陽師..."
                )
                time.sleep(2)
                continue

            hwnd = windows[0]._hWnd
            screen_gray, screen_color = clicker.capture_window_bitblt(hwnd)

            if screen_color is None:
                time.sleep(0.5)
                continue

            h, w = screen_color.shape[:2]

            if last_window_w != 0 and last_window_h != 0:
                if w != last_window_w or h != last_window_h:
                    prev_opp_small_blur = None
                    prev_my_small_blur = None

            last_window_w = w
            last_window_h = h

            current_sleep_interval = (
                INTERVAL_SLEEP_MODE if current_state == STATE_SLEEP else INTERVAL_ACTIVE_MODE
            )

            roi_opp_color = screen_color[0:int(h * 0.52), int(w * 0.45):int(w * 0.92)]
            roi_opp_gray = screen_gray[0:int(h * 0.52), int(w * 0.45):int(w * 0.92)]

            roi_my_color = screen_color[int(h * 0.10):int(h * 0.52), int(w * 0.10):int(w * 0.20)]
            roi_my_gray = screen_gray[int(h * 0.10):int(h * 0.52), int(w * 0.10):int(w * 0.20)]

            curr_opp_small_blur = get_downsampled_blur(roi_opp_gray)
            curr_my_small_blur = get_downsampled_blur(roi_my_gray)
            run_ocr = True

            if (
                prev_opp_small_blur is not None
                and curr_opp_small_blur is not None
                and prev_my_small_blur is not None
                and curr_my_small_blur is not None
            ):
                diff_opp = cv2.mean(cv2.absdiff(curr_opp_small_blur, prev_opp_small_blur))[0]
                diff_my = cv2.mean(cv2.absdiff(curr_my_small_blur, prev_my_small_blur))[0]

                if (
                    diff_opp < 8.0
                    and diff_my < 8.0
                    and (time.time() - last_ocr_time) < COOLDOWN_OPP_STATIC
                ):
                    run_ocr = False

            ocr_opp = []
            ocr_my = []

            if run_ocr:
                prev_opp_small_blur = curr_opp_small_blur
                prev_my_small_blur = curr_my_small_blur
                last_ocr_time = time.time()

                curr_roi_w_opp = roi_opp_color.shape[1]
                if curr_roi_w_opp < MIN_OPP_ROI_WIDTH:
                    target_w_opp = 640
                    h_opp, w_opp = roi_opp_color.shape[:2]
                    target_h_opp = int(h_opp * (target_w_opp / w_opp))
                else:
                    h_opp, w_opp = roi_opp_color.shape[:2]
                    target_w_opp = int(w_opp * 1.30)
                    target_h_opp = int(h_opp * 1.30)

                img_opp_resized = cv2.resize(
                    roi_opp_color, (target_w_opp, target_h_opp), interpolation=cv2.INTER_LINEAR
                )

                h_my, w_my = roi_my_color.shape[:2]
                target_h_my = target_h_opp
                target_w_my = int(w_my * (target_h_my / h_my))

                img_my_resized = cv2.resize(
                    roi_my_color, (target_w_my, target_h_my), interpolation=cv2.INTER_LINEAR
                )

                combined_roi = np.hstack((img_my_resized, img_opp_resized))
                x_border = target_w_my

                ocr_combined, _ = ocr_engine(combined_roi)

                if ocr_combined:
                    for res in ocr_combined:
                        box = res[0]
                        x_center = sum([pt[0] for pt in box]) / 4.0
                        if x_center < x_border:
                            ocr_my.append(res)
                        else:
                            ocr_opp.append(res)

            if ocr_opp:
                full_text_opp = "".join([res[1] for res in ocr_opp])

                if (
                    "配對中" in full_text_opp
                    or "匹配成功" in full_text_opp
                    or "名士" in full_text_opp
                    or "陰陽師" in full_text_opp
                ):
                    current_state = STATE_ACTIVE
                    last_wakeup_time = time.time()
                    if len(opp_draft_fullnames) >= 5 or len(current_my_draft) >= 5:
                        remembered_opp_ban = None
                        opp_draft_fullnames.clear()
                        current_my_draft.clear()
                        last_opp_draft_len = 0
                        has_triggered_five_picks = False
                        if hasattr(mode_controller, "clear_draft_data"):
                            mode_controller.clear_draft_data()

                elif (
                    "禁" in full_text_opp
                    or "禁選" in full_text_opp
                    or "禁用" in full_text_opp
                ):
                    current_state = STATE_ACTIVE
                    last_wakeup_time = time.time()

                elif "勝利" in full_text_opp or "失敗" in full_text_opp:
                    current_state = STATE_SLEEP
                    if remembered_opp_ban or opp_draft_fullnames or current_my_draft:
                        remembered_opp_ban = None
                        opp_draft_fullnames.clear()
                        current_my_draft.clear()
                        last_opp_draft_len = 0
                        has_triggered_five_picks = False
                        if hasattr(mode_controller, "clear_draft_data"):
                            mode_controller.clear_draft_data()

                if current_state == STATE_ACTIVE:
                    if (
                        not remembered_opp_ban
                        and len(opp_draft_fullnames) == 0
                        and len(current_my_draft) == 0
                    ):
                        if time.time() - last_wakeup_time > 15.0:
                            current_state = STATE_SLEEP

                if "不能上" in full_text_opp:
                    time.sleep(current_sleep_interval)
                    continue

                if (
                    ("禁" in full_text_opp or "禁選" in full_text_opp)
                    and (len(opp_draft_fullnames) > 0 or len(current_my_draft) > 0)
                ):
                    remembered_opp_ban = None
                    opp_draft_fullnames.clear()
                    current_my_draft.clear()
                    last_opp_draft_len = 0
                    current_state = STATE_ACTIVE
                    has_triggered_five_picks = False
                    if hasattr(mode_controller, "clear_draft_data"):
                        mode_controller.clear_draft_data()

                for res in ocr_opp:
                    text = res[1]
                    if not remembered_opp_ban:
                        if (
                            "禁" in full_text_opp
                            or "禁選" in full_text_opp
                            or "禁用" in full_text_opp
                        ):
                            for full_name, short_name in sorted_dict:
                                if full_name in text:
                                    remembered_opp_ban = short_name
                                    if short_name in opp_draft_fullnames:
                                        opp_draft_fullnames.remove(short_name)
                                    break

                    if remembered_opp_ban:
                        if len(opp_draft_fullnames) < 5:
                            for full_name, short_name in sorted_dict:
                                if full_name in text:
                                    if (
                                        "禁" not in full_text_opp
                                        and "禁選" not in full_text_opp
                                    ):
                                        if (
                                            short_name != remembered_opp_ban
                                            and short_name != my_ban
                                            and short_name not in opp_draft_fullnames
                                        ):
                                            opp_draft_fullnames.append(short_name)
                                    break

            current_opp_len = len(opp_draft_fullnames)

            if remembered_opp_ban and len(current_my_draft) < current_opp_len:
                if ocr_my:
                    for res in ocr_my:
                        text = res[1]
                        for full_name, short_name in sorted_dict:
                            if full_name in text:
                                if (
                                    short_name != remembered_opp_ban
                                    and short_name != my_ban
                                    and short_name not in current_my_draft
                                ):
                                    current_my_draft.append(short_name)
                                break

            if len(opp_draft_fullnames) == 5:
                current_state = STATE_SLEEP
                if not has_triggered_five_picks:
                    has_triggered_five_picks = True
                    if hasattr(mode_controller, "save_and_handle_five_picks"):
                        mode_controller.save_and_handle_five_picks(
                            ui_app.root,
                            current_my_draft,
                            opp_draft_fullnames,
                            screen_color,
                            hwnd
                        )
                    else:
                        mode_controller.handle_five_picks(
                            ui_app.root,
                            current_my_draft,
                            opp_draft_fullnames,
                            screen_color,
                            hwnd
                        )

            if remembered_opp_ban:
                opp_ban_str = f"Ban{remembered_opp_ban}"

                my_recs, opp_preds, match_type = get_recommendation(
                    db=active_db,
                    current_ban=opp_ban_str,
                    opp_draft=opp_draft_fullnames,
                    my_draft=current_my_draft,
                    my_ban=my_ban
                )

                my_recs = [c for c in my_recs if c != my_ban and c != remembered_opp_ban]
                opp_preds = [c for c in opp_preds if c != my_ban and c != remembered_opp_ban]

                queue_ui_update(
                    remembered_opp_ban,
                    opp_draft_fullnames,
                    current_my_draft,
                    my_recs,
                    opp_preds,
                    match_type
                )
            else:
                queue_ui_update(None, [], [], [], [], "等待對手 Ban 位資訊...")

        except Exception:
            pass

        time.sleep(current_sleep_interval)


if __name__ == "__main__":
    check_activation()

    try:
        SYSTEMS, FULL_TO_SHORT_MAP, MULTI_CHAR_SHORTS = load_cached_data()
        print("✅ Successfully loaded and decrypted database and dictionaries from AppData!")
    except Exception as e:
        print(f"❌ Failed to load database: {e}")
        sys.exit(1)

    COMPILED_DBS: Dict[str, Dict[str, Any]] = {}
    for sys_name, sys_info in SYSTEMS.items():
        COMPILED_DBS[sys_name] = {
            "my_ban": sys_info["my_ban"],
            "db": build_database(sys_info["raw_data"], MULTI_CHAR_SHORTS)
        }

    available_systems: List[str] = list(SYSTEMS.keys())

    root = tk.Tk()
    app = OnmyojiHelperUI(root, available_systems, mode_change_callback=mode_controller.set_mode)
    mode_controller.set_ui_app(app)

    root.after(200, process_ui_queue, root, app)

    t = threading.Thread(target=ocr_worker, args=(app, COMPILED_DBS, FULL_TO_SHORT_MAP), daemon=True)
    t.start()

    root.mainloop()