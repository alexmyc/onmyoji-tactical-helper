"""Windows Process Privilege, DPI Awareness, Priority, and UIPI Inspection Utilities."""

import ctypes
import logging
import os
import subprocess
import sys

from onmyoji_helper.infra.win32_window import get_window_pid

logger = logging.getLogger("OnmyojiHelper")

BELOW_NORMAL_PRIORITY_CLASS: int = 0x00004000
PROCESS_QUERY_LIMITED_INFORMATION: int = 0x1000
TOKEN_QUERY: int = 0x0008
TOKEN_ELEVATION_CLASS: int = 20
ERROR_ACCESS_DENIED: int = 5

if sys.platform == "win32":
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)

    _kernel32.GetCurrentProcess.argtypes = []
    _kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    _kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    _kernel32.SetPriorityClass.restype = ctypes.c_bool

    _kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_bool, ctypes.c_uint32]
    _kernel32.OpenProcess.restype = ctypes.c_void_p
    _kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    _kernel32.CloseHandle.restype = ctypes.c_bool

    _advapi32.OpenProcessToken.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    _advapi32.OpenProcessToken.restype = ctypes.c_bool
    _advapi32.GetTokenInformation.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    _advapi32.GetTokenInformation.restype = ctypes.c_bool
else:
    _kernel32 = None
    _advapi32 = None


def configure_cpu_thread_env() -> None:
    """Caps OpenMP / MKL / OpenBLAS worker threads to prevent CPU sawtooth contention with the game."""
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    os.environ.setdefault("KMP_BLOCKTIME", "0")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def enable_per_monitor_dpi_awareness() -> None:
    """Enables Per-Monitor DPI Awareness v2/v1 on Windows so coordinates match physical pixels."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def is_current_process_elevated() -> bool:
    """Returns True if the current process is running with Windows Administrator privileges."""
    if sys.platform != "win32":
        return True
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def ensure_admin_privileges() -> None:
    """Relaunches the process via Windows UAC ('runas') if not already running as Administrator.

    Exits cleanly after spawning the elevated child process or if the user declines UAC,
    preventing infinite elevation loops. Supports `ONMYOJI_SKIP_UAC_FOR_TEST=1` strictly
    for headless automated test harnesses.
    """
    if sys.platform != "win32" or is_current_process_elevated():
        return
    if os.environ.get("ONMYOJI_SKIP_UAC_FOR_TEST") == "1":
        return
    if os.environ.get("ONMYOJI_UAC_ATTEMPTED") == "1":
        sys.exit(0)

    try:
        shell32 = ctypes.windll.shell32
        shell32.ShellExecuteW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
            ctypes.c_int,
        ]
        shell32.ShellExecuteW.restype = ctypes.c_void_p

        if getattr(sys, "frozen", False):
            exe_path = sys.executable
            params = subprocess.list2cmdline(sys.argv[1:])
            work_dir = os.path.dirname(os.path.abspath(exe_path))
        else:
            exe_path = sys.executable
            script_path = os.path.abspath(sys.argv[0] if sys.argv else __file__)
            params = subprocess.list2cmdline([script_path, *sys.argv[1:]])
            work_dir = os.path.dirname(script_path)

        os.environ["ONMYOJI_UAC_ATTEMPTED"] = "1"
        ret = shell32.ShellExecuteW(None, "runas", exe_path, params, work_dir, 1)
        if ret and int(ret) > 32:
            sys.exit(0)
        sys.exit(0)
    except SystemExit:
        raise
    except Exception:
        sys.exit(0)


def set_below_normal_process_priority() -> None:
    """Lowers process scheduling priority to BELOW_NORMAL so the game always wins CPU slices."""
    if sys.platform != "win32" or _kernel32 is None:
        return
    try:
        handle = _kernel32.GetCurrentProcess()
        if _kernel32.SetPriorityClass(handle, BELOW_NORMAL_PRIORITY_CLASS):
            logger.info("🛡️ 進程優先級已設為 BELOW_NORMAL（低於標準），優先保障遊戲流暢度。")
        else:
            logger.warning("⚠️ 無法設定進程優先級為 BELOW_NORMAL (SetPriorityClass 返回 False)。")
    except Exception as exc:
        logger.debug(f"⚠️ 設定進程優先級略過: {exc}")


def is_uipi_blocked_for_hwnd(hwnd: int) -> bool:
    """Detects whether the target game process is elevated (Admin) while this helper is non-Admin.

    When True, Windows UIPI silently drops SendInput events targeting the elevated game window.
    """
    if sys.platform != "win32" or _kernel32 is None or _advapi32 is None:
        return False
    if is_current_process_elevated():
        return False

    game_pid = get_window_pid(hwnd)
    if game_pid <= 0 or game_pid == os.getpid():
        return False

    h_proc = _kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, game_pid)
    if not h_proc:
        return ctypes.get_last_error() == ERROR_ACCESS_DENIED

    h_token = ctypes.c_void_p(0)
    try:
        if not _advapi32.OpenProcessToken(h_proc, TOKEN_QUERY, ctypes.byref(h_token)):
            return ctypes.get_last_error() == ERROR_ACCESS_DENIED

        elevation = ctypes.c_uint32(0)
        ret_len = ctypes.c_uint32(0)
        ok = _advapi32.GetTokenInformation(
            h_token,
            TOKEN_ELEVATION_CLASS,
            ctypes.byref(elevation),
            ctypes.sizeof(elevation),
            ctypes.byref(ret_len),
        )
        return bool(ok and elevation.value != 0)
    except Exception:
        return False
    finally:
        if h_token and h_token.value:
            try:
                _kernel32.CloseHandle(h_token)
            except Exception:
                pass
        if h_proc:
            try:
                _kernel32.CloseHandle(h_proc)
            except Exception:
                pass
