"""Zero-Dependency `.env` Loader & Dual-Track Configuration Resolver.

Searches for an optional local `.env` file across frozen executable (`<exe_dir>/.env`,
`data/.env`, `sys._MEIPASS/.env`) and development (`<project_root>/.env`) paths without
requiring `python-dotenv`. When present, private Google Sheets sync and hardware license
endpoints are activated; when absent, the app transparently enters Public Sandbox Mode.
"""

import os
import sys

from onmyoji_helper.infra.paths import get_app_base_dir

PLACEHOLDER_PREFIXES: tuple[str, ...] = ("your_", "<your", "example_", "placeholder")


def _candidate_dotenv_paths() -> list[str]:
    """Returns ordered candidate `.env` file paths across frozen and source layouts."""
    candidates: list[str] = []
    base_dir = get_app_base_dir()

    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(os.path.abspath(sys.executable))
        candidates.append(os.path.join(exe_dir, ".env"))
        candidates.append(os.path.join(exe_dir, "data", ".env"))

    meipass_dir = getattr(sys, "_MEIPASS", None)
    if meipass_dir:
        candidates.append(os.path.join(os.path.abspath(meipass_dir), ".env"))

    candidates.append(os.path.join(base_dir, ".env"))
    candidates.append(os.path.join(base_dir, "data", ".env"))
    return candidates


def parse_dotenv_file(filepath: str) -> dict[str, str]:
    """Parses `KEY=VALUE` pairs from a UTF-8 `.env` file, ignoring comments and placeholders."""
    parsed: dict[str, str] = {}
    if not filepath or not os.path.isfile(filepath):
        return parsed

    try:
        with open(filepath, encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[7:].strip()
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
                    val = val[1:-1].strip()
                if not key or not val:
                    continue
                if any(val.lower().startswith(p) for p in PLACEHOLDER_PREFIXES):
                    continue
                parsed[key] = val
    except Exception:
        return {}
    return parsed


def load_dotenv_if_present(override: bool = False) -> str | None:
    """Loads the first discovered `.env` file into `os.environ` unless disabled for sandbox testing.

    Set `ONMYOJI_DISABLE_DOTENV=1` in tests to simulate a clean `git clone` without `.env`.

    Returns:
        Absolute path of the loaded `.env` file, or `None` if no `.env` file was found.
    """
    if os.environ.get("ONMYOJI_DISABLE_DOTENV") == "1":
        return None

    for candidate in _candidate_dotenv_paths():
        if os.path.isfile(candidate):
            kv = parse_dotenv_file(candidate)
            for key, val in kv.items():
                if override or key not in os.environ or not os.environ[key]:
                    os.environ[key] = val
            return candidate
    return None


def get_clean_env(key: str) -> str:
    """Reads a trimmed, non-placeholder environment variable after `.env` autoloading."""
    load_dotenv_if_present()
    val = os.environ.get(key, "").strip()
    if not val or any(val.lower().startswith(p) for p in PLACEHOLDER_PREFIXES):
        return ""
    return val
