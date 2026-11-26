"""Minimal Bootstrapper Entrypoint for Onmyoji Tactical Helper.

Performs fast pre-import CPU thread-pool capping and Windows UAC Administrator elevation
before loading `onmyoji_helper.app.bootstrap`, compatible with both local `src/` execution
and frozen PyInstaller (`OnmyojiHelper.exe` + `data/`) builds.
"""

import ctypes
import os
import subprocess
import sys


def _is_admin() -> bool:
    """Returns True if the current process is running with Windows Administrator privileges."""
    if sys.platform != "win32":
        return True
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def _ensure_admin_privileges() -> None:
    """Relaunches via Windows UAC ('runas') if non-elevated; exits cleanly on cancel or error."""
    if sys.platform != "win32" or _is_admin():
        return
    if os.environ.get("ONMYOJI_SKIP_UAC_FOR_TEST") == "1":
        return
    if os.environ.get("ONMYOJI_UAC_ATTEMPTED") == "1":
        sys.exit(0)

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


def _has_local_private_dotenv() -> bool:
    """Returns True when a local `.env` exists (production mode requiring UAC elevation)."""
    if os.environ.get("ONMYOJI_DISABLE_DOTENV") == "1":
        return False
    if os.environ.get("ONMYOJI_AUTH_CSV_URL"):
        return True
    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(os.path.abspath(sys.executable))
        return os.path.isfile(os.path.join(exe_dir, ".env")) or os.path.isfile(
            os.path.join(exe_dir, "data", ".env")
        )
    root_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.isfile(os.path.join(root_dir, ".env")) or os.path.isfile(
        os.path.join(root_dir, "data", ".env")
    )


def _has_demo_default_flag() -> bool:
    """Returns True when `data/demo_default.flag` is present (HR / Recruiter Demo Edition)."""
    if os.environ.get("ONMYOJI_IGNORE_DEMO_FLAG") == "1":
        return False
    if getattr(sys, "frozen", False):
        base = os.path.dirname(os.path.abspath(sys.executable))
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.isfile(os.path.join(base, "data", "demo_default.flag")) or os.path.isfile(
        os.path.join(base, "demo_default.flag")
    )


def _should_skip_uac(argv: list[str]) -> bool:
    """Skips Windows UAC prompt in `--demo`, `--smoke-check`, `demo_default.flag`, `--help`, or Sandbox runs."""
    if any(flag in argv for flag in ("--demo", "--smoke-check", "-h", "--help")) or _has_demo_default_flag():
        return True
    return not _has_local_private_dotenv()


def _configure_import_path() -> None:
    """Injects `src/` onto `sys.path` when running directly from source."""
    if not getattr(sys, "frozen", False):
        root_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.join(root_dir, "src")
        if os.path.isdir(src_dir) and src_dir not in sys.path:
            sys.path.insert(0, src_dir)


if __name__ == "__main__":
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    os.environ.setdefault("KMP_BLOCKTIME", "0")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass

    if not _should_skip_uac(sys.argv[1:]):
        _ensure_admin_privileges()
    _configure_import_path()

    from onmyoji_helper.app.bootstrap import run_application

    run_application(sys.argv[1:])
