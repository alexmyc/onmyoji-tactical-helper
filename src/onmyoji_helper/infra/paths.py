"""Unified path resolution utilities across development and frozen PyInstaller environments."""

import os
import sys


def get_app_base_dir() -> str:
    """Returns the application root directory (executable folder when frozen, project root in dev)."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    # Navigate from src/onmyoji_helper/infra/paths.py -> project root
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", ".."))


def resolve_asset_path(filename: str, extra_dir: str = "") -> str:
    """Resolves a bundled model or asset path across PyInstaller `_MEIPASS`, `data/`, and base dir.

    Search priority:
        1. `sys._MEIPASS/<filename>` (points to `<exe_dir>/data` under `--contents-directory=data`)
        2. `<base_dir>/data/<filename>`
        3. `<base_dir>/<filename>`
        4. `<base_dir>/models/<filename>`
        5. `<extra_dir>/<filename>` (if provided)

    Args:
        filename: Target asset file name (e.g. 'best.pt' or 'gemma-4-E2B-it-Q4_K_M.gguf').
        extra_dir: Optional user-supplied fallback search directory.

    Returns:
        First existing absolute file path, or the primary expected candidate path if missing.
    """
    base_dir = get_app_base_dir()
    candidates: list[str] = []

    meipass_dir = getattr(sys, "_MEIPASS", None)
    if meipass_dir:
        candidates.append(os.path.join(os.path.abspath(meipass_dir), filename))

    candidates.append(os.path.join(base_dir, "data", filename))
    candidates.append(os.path.join(base_dir, filename))
    candidates.append(os.path.join(base_dir, "models", filename))

    if extra_dir:
        candidates.append(os.path.join(os.path.abspath(extra_dir), filename))

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    return candidates[0] if candidates else os.path.join(base_dir, filename)


def get_appdata_cache_path(filename: str = "cache_db.enc") -> str:
    """Returns the %APPDATA%/OnmyojiHelper path for encrypted local cache persistence."""
    appdata = os.getenv("APPDATA", "") or os.path.expanduser("~")
    target_dir = os.path.join(appdata, "OnmyojiHelper")
    os.makedirs(target_dir, exist_ok=True)
    return os.path.join(target_dir, filename)
