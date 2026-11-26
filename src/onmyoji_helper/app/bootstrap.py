"""Application Bootstrapper & Draft Worker Orchestrator for Onmyoji Tactical Helper.

Executes zero-side-effect dependency wiring at runtime:
1. Configures OpenMP/OpenCV thread pools, Per-Monitor DPI awareness, UAC elevation, and
   `BELOW_NORMAL` scheduling priority.
2. Runs hardware license verification and single-pass encrypted Google Sheets cache sync.
3. Compiles XMind playbooks and launches the `DraftOcrScanner` + `DraftStateMachine` worker
   alongside the Tkinter `OnmyojiHelperUI` event loop.
"""

import argparse
import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox
from typing import Any

import cv2

from onmyoji_helper.app.battle_coordinator import AITacticalManager
from onmyoji_helper.app.mode_controller import ModeController
from onmyoji_helper.decision.draft_engine import build_database, get_recommendation
from onmyoji_helper.decision.draft_fsm import DraftStateMachine
from onmyoji_helper.domain.models import DraftPhase
from onmyoji_helper.domain.shikigami_repository import ShikigamiRepository
from onmyoji_helper.infra.auth_service import check_activation
from onmyoji_helper.infra.capture_engine import CaptureEngine
from onmyoji_helper.infra.env_loader import load_dotenv_if_present
from onmyoji_helper.infra.logger_setup import setup_runtime_logger
from onmyoji_helper.infra.sheets_sync_store import (
    has_cloud_sheets_configured,
    load_all_cached_bundle,
)
from onmyoji_helper.infra.win32_privilege import (
    configure_cpu_thread_env,
    enable_per_monitor_dpi_awareness,
    ensure_admin_privileges,
    set_below_normal_process_priority,
)
from onmyoji_helper.ui.demo_sandbox import DemoSandboxWindow
from onmyoji_helper.ui.main_window import OnmyojiHelperUI
from onmyoji_helper.vision.draft_ocr_scanner import DraftOcrScanner

INTERVAL_SLEEP_MODE: float = 2.5
INTERVAL_BAN_WAIT: float = 0.9
INTERVAL_ACTIVE_MODE: float = 1.3
INTERVAL_IN_BATTLE_MODE: float = 4.5


def _process_ui_queue(
    root_window: tk.Tk,
    ui_app: OnmyojiHelperUI,
    ui_queue: "queue.Queue[tuple[str | None, list[str], list[str], list[str], list[str], str]]",
) -> None:
    """Drains pending draft UI updates on the Tkinter main thread every 50ms."""
    while not ui_queue.empty():
        try:
            data = ui_queue.get_nowait()
            ui_app.update_info(*data)
        except queue.Empty:
            break
    root_window.after(50, _process_ui_queue, root_window, ui_app, ui_queue)


def _draft_ocr_worker(
    ui_app: OnmyojiHelperUI,
    compiled_dbs: dict[str, dict[str, Any]],
    repository: ShikigamiRepository,
    mode_controller: ModeController,
    tactical_mgr: AITacticalManager,
    ui_queue: "queue.Queue[tuple[str | None, list[str], list[str], list[str], list[str], str]]",
    logger: Any,
) -> None:
    """Background worker orchestrating GDI capture, `DraftOcrScanner`, and `DraftStateMachine`."""
    clicker = CaptureEngine()
    scanner = DraftOcrScanner(known_tokens=repository.known_ocr_tokens)
    fsm = DraftStateMachine()

    def push_ui(
        opp_ban: str | None,
        opp_draft: list[str],
        my_draft: list[str],
        recs: list[str],
        preds: list[str],
        status: str,
    ) -> None:
        ui_queue.put((opp_ban, opp_draft, my_draft, recs, preds, status))

    push_ui(None, [], [], [], [], "正在尋找陰陽師視窗...")
    logger.info("🚀 RapidOCR 背景掃描執行緒已啟動...")

    last_window_wh: tuple[int, int] = (0, 0)
    logged_capture_fail = False
    logged_black_frame = False
    logged_first_ocr = False

    while True:
        ui_system_choice = ui_app.get_selected_system()
        current_system = compiled_dbs.get(ui_system_choice)
        if not current_system:
            time.sleep(1.0)
            continue

        active_db = current_system["db"]
        my_ban = str(current_system["my_ban"])

        if ui_app.check_and_clear_reset():
            fsm.reset(DraftPhase.SLEEP)
            scanner.reset_motion_cache()
            mode_controller.clear_draft_data()
            tactical_mgr.reset_for_next_match()
            push_ui(None, [], [], [], [], f"已重置，當前數據：{ui_system_choice}")

        poll_interval = fsm.get_recommended_poll_interval(
            sleep_sec=INTERVAL_SLEEP_MODE,
            ban_wait_sec=INTERVAL_BAN_WAIT,
            active_sec=INTERVAL_ACTIVE_MODE,
            in_battle_sec=INTERVAL_IN_BATTLE_MODE,
        )

        try:
            windows = clicker.find_windows()
            if not windows:
                push_ui(fsm.remembered_opp_ban, [], [], [], [], "找不到遊戲視窗，請開啟陰陽師...")
                time.sleep(2.0)
                continue

            hwnd = int(windows[0]._hWnd)
            frame = clicker.capture_frame(hwnd, need_gray=True, window_title=windows[0].title)
            if frame is None:
                if not logged_capture_fail:
                    logged_capture_fail = True
                    titles = [w.title for w in windows]
                    logger.warning(f"⚠️ 截圖回傳為空 (hwnd={hwnd}, 偵測視窗={titles})，等待視窗就緒...")
                time.sleep(0.5)
                continue

            logged_capture_fail = False
            w, h = frame.width, frame.height

            if frame.gray is not None and not frame.gray.any():
                if not logged_black_frame:
                    logged_black_frame = True
                    logger.warning(
                        f"⚠️ 截取到的畫面為全黑 (hwnd={hwnd}, 尺寸={w}x{h})，"
                        "請確認遊戲視窗未被最小化、非獨佔全螢幕，且助手權限與遊戲一致。"
                    )
            else:
                logged_black_frame = False

            if (w, h) != last_window_wh:
                logger.info(f"🎮 已鎖定遊戲視窗並截圖成功: '{windows[0].title}' (hwnd={hwnd}, 尺寸={w}x{h})")
                last_window_wh = (w, h)

            ocr_opp, ocr_my, _ = scanner.scan_draft_rois(
                screen_color=frame.bgr,
                screen_gray=frame.gray,
                is_active_state=fsm.is_active_phase,
                has_locked_opp_ban=(fsm.remembered_opp_ban is not None),
                in_battle_five_picks=fsm.has_triggered_five_picks,
            )

            if ocr_opp and not logged_first_ocr:
                first_txt = "".join(str(r[1]) for r in ocr_opp).strip()
                if first_txt:
                    logged_first_ocr = True
                    logger.info(f"👁️ 首次 OCR 讀取成功 (尺寸={w}x{h}): '{first_txt[:40]}'")

            snapshot, skip_ui_refresh = fsm.step(
                ocr_opp=ocr_opp,
                ocr_my=ocr_my,
                sorted_full_to_short=repository.sorted_full_to_short,
                my_ban=my_ban,
                is_combat_active=bool(
                    tactical_mgr.is_active_session and tactical_mgr.battle_has_started
                ),
            )

            if snapshot.should_reset_battle:
                mode_controller.clear_draft_data()
                tactical_mgr.reset_for_next_match()

            if skip_ui_refresh:
                time.sleep(poll_interval)
                continue

            if snapshot.just_completed_five_picks:
                mode_controller.save_and_handle_five_picks(
                    ui_app.root,
                    snapshot.my_draft,
                    snapshot.opp_draft,
                    None,
                    hwnd,
                )

            if snapshot.opp_ban:
                my_recs, opp_preds, match_type = get_recommendation(
                    db=active_db,
                    current_ban=f"Ban{snapshot.opp_ban}",
                    opp_draft=snapshot.opp_draft,
                    my_draft=snapshot.my_draft,
                    my_ban=my_ban,
                )
                my_recs = [c for c in my_recs if c != my_ban and c != snapshot.opp_ban]
                opp_preds = [c for c in opp_preds if c != my_ban and c != snapshot.opp_ban]
                push_ui(
                    snapshot.opp_ban,
                    snapshot.opp_draft,
                    snapshot.my_draft,
                    my_recs,
                    opp_preds,
                    match_type,
                )
            elif fsm.in_ban_waiting:
                my_recs, opp_preds, match_type = get_recommendation(
                    db=active_db,
                    current_ban="",
                    opp_draft=[],
                    my_draft=[],
                    my_ban=my_ban,
                )
                push_ui(
                    "禁選中",
                    [],
                    [],
                    my_recs,
                    opp_preds,
                    f"已進入禁選階段，等待對手鎖定 Ban 位 ({match_type})",
                )
            else:
                push_ui(None, [], [], [], [], "等待對手 Ban 位資訊...")

        except Exception as loop_err:
            logger.warning(f"⚠️ [OCR Worker] 迴圈迭代發生異常 (已自動保護恢復): {loop_err}", exc_info=True)

        time.sleep(
            fsm.get_recommended_poll_interval(
                sleep_sec=INTERVAL_SLEEP_MODE,
                ban_wait_sec=INTERVAL_BAN_WAIT,
                active_sec=INTERVAL_ACTIVE_MODE,
                in_battle_sec=INTERVAL_IN_BATTLE_MODE,
            )
        )


def _parse_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses runtime CLI flags (`--demo`, `--smoke-exit-ms`)."""
    parser = argparse.ArgumentParser(
        prog="run_app.py",
        description="Onmyoji Tactical Helper — Real-Time Multimodal Edge-AI Assistant",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Launch the offline interactive match sandbox (no game window, GPU, or .env needed).",
    )
    parser.add_argument(
        "--smoke-exit-ms",
        type=int,
        default=int(os.environ.get("ONMYOJI_DEMO_SMOKE_MS", "0") or "0"),
        help=argparse.SUPPRESS,
    )
    args, _ = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    return args


def run_application(argv: list[str] | None = None) -> None:
    """Primary application entrypoint called by `run_app.py`."""
    args = _parse_cli_args(argv)
    demo_mode = bool(args.demo)

    configure_cpu_thread_env()
    enable_per_monitor_dpi_awareness()
    load_dotenv_if_present()

    has_private_cloud = has_cloud_sheets_configured()
    is_public_sandbox = demo_mode or (not has_private_cloud)

    if not is_public_sandbox:
        ensure_admin_privileges()

    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)

    logger = setup_runtime_logger(install_hooks=True)
    set_below_normal_process_priority()
    check_activation(demo_mode=demo_mode)

    if not demo_mode and not has_private_cloud:
        print(
            "[Public Sandbox Mode] No private .env detected — loaded built-in sample playbooks.\n"
            "  -> Tip: Run 'python run_app.py --demo' or click '🧪 Interactive Demo Sandbox' "
            "in the window to simulate live Ban/Pick & 5v5 combat!"
        )

    try:
        systems, full_to_short_map, multi_char_shorts, tags_db = load_all_cached_bundle(
            demo_mode=demo_mode
        )
        logger.info(
            f"✅ 資料庫啟動載入完成 (體系: {len(systems)} 套, 字典: {len(full_to_short_map)} 條, "
            f"多字縮寫: {len(multi_char_shorts)} 個, 標籤: {len(tags_db)} 隻)"
        )
    except Exception as exc:
        logger.critical(f"❌ Failed to load database: {exc}", exc_info=True)
        try:
            err_root = tk.Tk()
            err_root.withdraw()
            messagebox.showerror("資料庫載入失敗", f"無法連線至雲端資料庫且本機尚無快取，請檢查網路連線。\n({exc})")
            err_root.destroy()
        except Exception as ui_err:
            logger.debug(f"[Bootstrap] Error dialog fallback skipped: {ui_err}")
        sys.exit(1)

    repository = ShikigamiRepository(
        tags_db=tags_db,
        full_to_short_map=full_to_short_map,
        multi_char_shorts=multi_char_shorts,
    )

    compiled_dbs: dict[str, dict[str, Any]] = {
        sys_name: {
            "my_ban": sys_info["my_ban"],
            "db": build_database(sys_info["raw_data"], repository.sorted_multi_shorts),
        }
        for sys_name, sys_info in systems.items()
    }

    tactical_mgr = AITacticalManager(repository=repository)
    mode_controller = ModeController(tactical_mgr)
    ui_queue: queue.Queue[tuple[str | None, list[str], list[str], list[str], list[str], str]] = (
        queue.Queue()
    )

    root = tk.Tk()
    sandbox_holder: list[DemoSandboxWindow] = []

    def open_or_focus_sandbox() -> None:
        if sandbox_holder and sandbox_holder[0].winfo_exists():
            sandbox_holder[0].window.lift()
            return
        win = DemoSandboxWindow(
            root=root,
            ui_app=app,
            compiled_dbs=compiled_dbs,
            repository=repository,
        )
        sandbox_holder.clear()
        sandbox_holder.append(win)

    app = OnmyojiHelperUI(
        root,
        list(systems.keys()),
        mode_change_callback=(None if demo_mode else mode_controller.set_mode),
        sandbox_callback=(open_or_focus_sandbox if is_public_sandbox else None),
    )
    mode_controller.set_ui_app(app)

    if demo_mode:
        logger.info("🧪 [Demo Mode] Interactive Match Sandbox launched (--demo).")
        open_or_focus_sandbox()
        if args.smoke_exit_ms > 0:
            sb = sandbox_holder[0]
            root.after(200, lambda: sb.apply_result(sb.controller.select_scenario(1)))
            root.after(400, lambda: sb.apply_result(sb.controller.select_scenario(2)))
            root.after(600, lambda: sb.apply_result(sb.controller.select_scenario(3)))
            root.after(args.smoke_exit_ms, root.destroy)
        root.mainloop()
        return

    if is_public_sandbox:
        app.lbl_status.config(
            text="Public Sandbox Ready — Click [🧪 Interactive Demo Sandbox] or run --demo"
        )

    root.after(200, _process_ui_queue, root, app, ui_queue)

    tactical_mgr.models.warmup_in_background()

    worker_thread = threading.Thread(
        target=_draft_ocr_worker,
        args=(app, compiled_dbs, repository, mode_controller, tactical_mgr, ui_queue, logger),
        name="ocr_worker",
        daemon=True,
    )
    worker_thread.start()

    root.mainloop()
