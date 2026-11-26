"""Multi-Profile Cython .pyd & PyInstaller Release Builder for Onmyoji Tactical Helper.

Compiles all Python modules under `src/onmyoji_helper/` into native Windows `.pyd`
C-extensions via Cython 3.3 with anti-reverse-engineering hardening (`/O2 /DNDEBUG`,
`/DEBUG:NONE`, stripped docstrings/bindings/traces), packages `OnmyojiHelper.exe` via
PyInstaller, applies profile-specific asset/credential rules, and executes a strict
post-build security & integrity audit. All terminal output uses pure English ASCII to
prevent codepage garbling in Windows CMD / PowerShell.

Supported Profiles (`--profile`):
  1. `personal`    -> `dist/OnmyojiHelper_Personal/`
                      33 `.pyd` modules + local `.env` + `best.pt` + local `.gguf` (if present).
  2. `hr-demo`     -> `dist/OnmyojiHelper_HR_Demo/` + `dist/OnmyojiHelper_HR_Demo_Win64.zip`
                      33 `.pyd` modules + ZERO `.env` (`.env.example` only) + `data/demo_default.flag`
                      (launches clean Offline Demo directly on double-click without CLI args or UAC).
  3. `friend-full` -> `dist/OnmyojiHelper_Friend_Full/`
                      Hardened 33 `.pyd` modules + `ONMYOJI_LOG_LEVEL=WARNING` + local `.env`
                      (enforces Google Sheets hardware whitelist) + `best.pt` + bundled `.gguf`
                      under `data/models/` with auto-rewritten `LOCAL_LLM_MODEL_PATH`.
  4. `friend-lite` -> `dist/OnmyojiHelper_Friend_Lite/`
                      Hardened 33 `.pyd` modules + `ONMYOJI_LOG_LEVEL=WARNING` + local `.env`
                      (enforces Google Sheets hardware whitelist) + `best.pt` + excludes `llama_cpp`
                      & `.gguf`, injecting `FORCE_LITE_ENGINE=1` for low-spec / non-CUDA PCs.
"""

from __future__ import annotations

import argparse
import glob
import os
import pathlib
import shutil
import subprocess
import sys
import sysconfig
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

APP_NAME: str = "OnmyojiHelper"
PACKAGE_NAME: str = "onmyoji_helper"
CONTENTS_DIR_NAME: str = "data"
BOOTSTRAP_FILE: str = "run_app.py"
ICON_FILE: str = "app.ico"
YOLO_WEIGHT_FILE: str = "best.pt"
DEFAULT_GGUF_FILE: str = "gemma-4-E2B-it-Q4_K_M.gguf"
DEMO_FLAG_FILE: str = "demo_default.flag"


@dataclass(frozen=True)
class BuildProfileSpec:
    """Declarative specification for each standalone release profile."""

    key: str
    menu_number: str
    title: str
    output_folder_name: str
    include_local_env: bool
    include_yolo_weights: bool
    include_local_gguf: bool
    exclude_llama_cpp: bool
    exclude_heavy_cv_runtime: bool
    require_uac_admin: bool
    write_demo_default_flag: bool
    force_lite_engine: bool
    default_log_level: str | None
    create_win64_zip: bool
    zip_filename: str = ""


PROFILES: dict[str, BuildProfileSpec] = {
    "personal": BuildProfileSpec(
        key="personal",
        menu_number="1",
        title="Personal Full Edition (Local .env + YOLO + Optional GGUF)",
        output_folder_name="OnmyojiHelper_Personal",
        include_local_env=True,
        include_yolo_weights=True,
        include_local_gguf=True,
        exclude_llama_cpp=False,
        exclude_heavy_cv_runtime=False,
        require_uac_admin=True,
        write_demo_default_flag=False,
        force_lite_engine=False,
        default_log_level=None,
        create_win64_zip=False,
    ),
    "hr-demo": BuildProfileSpec(
        key="hr-demo",
        menu_number="2",
        title="HR / Recruiter Demo Edition (Zero Credentials + Auto Offline Demo + ZIP)",
        output_folder_name="OnmyojiHelper_HR_Demo",
        include_local_env=False,
        include_yolo_weights=False,
        include_local_gguf=False,
        exclude_llama_cpp=True,
        exclude_heavy_cv_runtime=True,
        require_uac_admin=False,
        write_demo_default_flag=True,
        force_lite_engine=True,
        default_log_level=None,
        create_win64_zip=True,
        zip_filename="OnmyojiHelper_HR_Demo_Win64.zip",
    ),
    "friend-full": BuildProfileSpec(
        key="friend-full",
        menu_number="3",
        title="Friend Full Edition - GPU / CUDA (Hardened .pyd + Whitelist .env + Bundled GGUF)",
        output_folder_name="OnmyojiHelper_Friend_Full",
        include_local_env=True,
        include_yolo_weights=True,
        include_local_gguf=True,
        exclude_llama_cpp=False,
        exclude_heavy_cv_runtime=False,
        require_uac_admin=True,
        write_demo_default_flag=False,
        force_lite_engine=False,
        default_log_level="WARNING",
        create_win64_zip=False,
    ),
    "friend-lite": BuildProfileSpec(
        key="friend-lite",
        menu_number="4",
        title="Friend Lite Edition - Low-Spec / No CUDA (Hardened .pyd + Whitelist .env + Rule Engine)",
        output_folder_name="OnmyojiHelper_Friend_Lite",
        include_local_env=True,
        include_yolo_weights=True,
        include_local_gguf=False,
        exclude_llama_cpp=True,
        exclude_heavy_cv_runtime=False,
        require_uac_admin=True,
        write_demo_default_flag=False,
        force_lite_engine=True,
        default_log_level="WARNING",
        create_win64_zip=False,
    ),
}

MENU_TO_PROFILE: dict[str, str] = {spec.menu_number: key for key, spec in PROFILES.items()}

# Mandatory across ALL 4 profiles (`personal`, `hr-demo`, `friend-full`, `friend-lite`) because
# Cython `.pyd` binaries hide Python-level imports from PyInstaller's static bytecode analyzer!
COMMON_HIDDEN_IMPORTS: list[str] = [
    "win32gui",
    "win32con",
    "win32process",
    "win32api",
    "win32ui",
    "win32clipboard",
    "pywintypes",
    "pythoncom",
    "winreg",
    "pygetwindow",
    "requests",
    "urllib3",
    "certifi",
    "PIL",
    "PIL.Image",
    "cv2",
    "numpy",
    "tkinter",
    "tkinter.font",
    "tkinter.messagebox",
    "argparse",
    "ctypes",
    "urllib.request",
    "urllib.error",
    "urllib.parse",
    "http.client",
    "ssl",
    "csv",
    "dataclasses",
    "enum",
    "functools",
    "json",
    "hashlib",
    "io",
    "logging.handlers",
    "math",
    "queue",
    "random",
    "threading",
    "subprocess",
    "uuid",
    "gc",
    "collections",
]

# Required for all live combat / whitelist profiles (`personal`, `friend-full`, `friend-lite`)
LIVE_COMBAT_HIDDEN_IMPORTS: list[str] = [
    "ultralytics",
    "rapidocr_onnxruntime",
    "onnxruntime",
    "torch",
    "torchvision",
]

LIVE_COMBAT_COLLECT_ALL: list[str] = [
    "ultralytics",
    "rapidocr_onnxruntime",
]

LIVE_COMBAT_COLLECT_BINARIES: list[str] = [
    "onnxruntime",
]

HEAVY_RUNTIME_EXCLUDES: list[str] = [
    "llama_cpp",
    "torch",
    "torchvision",
    "torchaudio",
    "ultralytics",
    "rapidocr_onnxruntime",
    "onnxruntime",
    "scipy",
    "pandas",
    "matplotlib",
]

_SINGLE_MODULE_SETUP_TEMPLATE: str = """from setuptools import Extension, setup
from Cython.Build import cythonize
from Cython.Compiler import Options

Options.docstrings = False
Options.embed_pos_in_docstring = False
Options.c_line_in_traceback = False
Options.emit_code_comments = False

ext = Extension(
    {dotted_name!r},
    [{rel_src_posix!r}],
    extra_compile_args=["/O2", "/DNDEBUG"],
    extra_link_args=["/DEBUG:NONE"],
)

setup(
    name={safe_name!r},
    ext_modules=cythonize(
        [ext],
        compiler_directives={{
            "language_level": "3",
            "embedsignature": False,
            "binding": False,
            "linetrace": False,
            "profile": False,
            "emit_code_comments": False,
            "annotation_typing": False,
        }},
        quiet=True,
    ),
)
"""


def _remove_path_safe(path: str) -> None:
    if not os.path.exists(path):
        return
    try:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            os.remove(path)
    except Exception:
        pass


def _is_module_available(mod_name: str) -> bool:
    try:
        __import__(mod_name)
        return True
    except Exception:
        return False


def discover_package_modules(src_root: str) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Discovers all compilable `.py` modules and package directories under `src/onmyoji_helper`."""
    pkg_root = os.path.join(src_root, PACKAGE_NAME)
    if not os.path.isdir(pkg_root):
        raise FileNotFoundError(f"Core package directory not found: {pkg_root}")

    modules: list[tuple[str, str, str]] = []
    packages: set[str] = set()

    for root_dir, _, files in os.walk(pkg_root):
        if "__pycache__" in root_dir:
            continue
        rel_dir = os.path.relpath(root_dir, src_root)
        rel_dir_posix = rel_dir.replace(os.sep, "/")
        dotted_pkg = rel_dir_posix.replace("/", ".")
        if "__init__.py" in files:
            packages.add(dotted_pkg)

        for fname in sorted(files):
            if not fname.endswith(".py") or fname == "__init__.py":
                continue
            mod_stem = fname[:-3]
            dotted_mod = f"{dotted_pkg}.{mod_stem}"
            abs_src = os.path.join(root_dir, fname)
            modules.append((dotted_mod, rel_dir_posix, abs_src))

    modules.sort(key=lambda item: item[0])
    return modules, sorted(packages)


def _parse_simple_dotenv(dotenv_path: str) -> dict[str, str]:
    result: dict[str, str] = {}
    if not os.path.isfile(dotenv_path):
        return result
    with open(dotenv_path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
                v = v[1:-1].strip()
            if k and v:
                result[k] = v
    return result


def discover_local_gguf_model(
    project_root: str,
    lm_studio_dirs: list[str] | None = None,
) -> str | None:
    """Locates the best quantized `.gguf` LLM model across `.env`, project dirs, and LM Studio.

    Search order:
      1. `LOCAL_LLM_MODEL_PATH` in `<project_root>/.env` (if present and existing).
      2. `<project_root>/gemma-4-E2B-it-Q4_K_M.gguf` or `<project_root>/models/*.gguf` / `*.gguf`
         (ignoring `mmproj*` vision projectors and build/dist folders).
      3. Local LM Studio model cache (`~/.lmstudio/models`, `~/.cache/lm-studio/models`),
         filtering out `mmproj*` / `vocab*` files and prioritizing Gemma / `< 8 GB` chat models.
    """
    dotenv_kv = _parse_simple_dotenv(os.path.join(project_root, ".env"))
    cfg_path = dotenv_kv.get("LOCAL_LLM_MODEL_PATH", "").strip()
    if cfg_path:
        resolved_cfg = (
            cfg_path if os.path.isabs(cfg_path) else os.path.join(project_root, cfg_path)
        )
        if os.path.isfile(resolved_cfg) and resolved_cfg.lower().endswith(".gguf"):
            return os.path.abspath(resolved_cfg)

    preferred_root_gguf = os.path.join(project_root, DEFAULT_GGUF_FILE)
    if os.path.isfile(preferred_root_gguf):
        return os.path.abspath(preferred_root_gguf)

    for sub in ("models", ""):
        scan_dir = os.path.join(project_root, sub) if sub else project_root
        if not os.path.isdir(scan_dir):
            continue
        for fname in sorted(os.listdir(scan_dir)):
            low = fname.lower()
            if low.endswith(".gguf") and not low.startswith(("mmproj", "vocab")):
                cand = os.path.join(scan_dir, fname)
                if os.path.isfile(cand):
                    return os.path.abspath(cand)

    if lm_studio_dirs is None:
        home = pathlib.Path.home()
        local_app = os.environ.get("LOCALAPPDATA", "")
        search_roots = [
            home / ".lmstudio" / "models",
            home / ".cache" / "lm-studio" / "models",
        ]
        if local_app:
            search_roots.append(pathlib.Path(local_app) / "LM-Studio" / "models")
    else:
        search_roots = [pathlib.Path(p) for p in lm_studio_dirs]

    lm_candidates: list[tuple[int, int, str]] = []
    for root_p in search_roots:
        if not root_p.is_dir():
            continue
        for gguf_p in root_p.rglob("*.gguf"):
            low_name = gguf_p.name.lower()
            if low_name.startswith(("mmproj", "vocab")) or "mmproj" in low_name:
                continue
            try:
                size_bytes = gguf_p.stat().st_size
            except OSError:
                continue
            is_gemma_rank = 0 if "gemma" in low_name else 1
            fits_consumer_vram_rank = 0 if size_bytes <= (8 * 1024 ** 3) else 1
            lm_candidates.append((is_gemma_rank + fits_consumer_vram_rank, size_bytes, str(gguf_p.resolve())))

    if lm_candidates:
        lm_candidates.sort(key=lambda item: (item[0], item[1]))
        return lm_candidates[0][2]
    return None


def rewrite_packaged_dotenv(
    src_dotenv_path: str,
    dst_dotenv_path: str,
    updates: dict[str, str],
    remove_keys: set[str] | None = None,
) -> None:
    """Copies `src_dotenv_path` to `dst_dotenv_path` with targeted key overrides (never mutating root `.env`)."""
    removed = {k.strip() for k in (remove_keys or set())}
    out_lines: list[str] = []
    seen_keys: set[str] = set()

    if os.path.isfile(src_dotenv_path):
        with open(src_dotenv_path, encoding="utf-8") as f:
            for raw_line in f:
                stripped = raw_line.strip()
                if stripped and not stripped.startswith("#") and "=" in stripped:
                    check_line = stripped[7:].strip() if stripped.startswith("export ") else stripped
                    k = check_line.split("=", 1)[0].strip()
                    if k in removed:
                        continue
                    if k in updates:
                        out_lines.append(f"{k}={updates[k]}\n")
                        seen_keys.add(k)
                        continue
                out_lines.append(raw_line if raw_line.endswith("\n") else f"{raw_line}\n")

    if out_lines and not out_lines[-1].endswith("\n"):
        out_lines[-1] += "\n"

    missing_updates = [(k, v) for k, v in updates.items() if k not in seen_keys and k not in removed]
    if missing_updates:
        if out_lines:
            out_lines.append("\n# --- Auto-generated packaging profile overrides ---\n")
        for k, v in missing_updates:
            out_lines.append(f"{k}={v}\n")

    os.makedirs(os.path.dirname(os.path.abspath(dst_dotenv_path)), exist_ok=True)
    with open(dst_dotenv_path, "w", encoding="utf-8") as f:
        f.writelines(out_lines)


def _compile_single_module(
    dotted_name: str,
    rel_pkg_dir_posix: str,
    abs_src_path: str,
    cython_tmp_root: str,
    stage_dir: str,
) -> tuple[str, str, str]:
    """Compiles a single package module into a hardened `.pyd` inside an isolated child process."""
    safe_name = dotted_name.replace(".", "__")
    mod_work_dir = os.path.join(cython_tmp_root, safe_name)
    _remove_path_safe(mod_work_dir)

    mod_filename = os.path.basename(abs_src_path)
    rel_pkg_os = rel_pkg_dir_posix.replace("/", os.sep)
    local_pkg_dir = os.path.join(mod_work_dir, rel_pkg_os)
    os.makedirs(local_pkg_dir, exist_ok=True)

    parts = rel_pkg_dir_posix.split("/")
    accum = mod_work_dir
    for part in parts:
        accum = os.path.join(accum, part)
        init_marker = os.path.join(accum, "__init__.py")
        if not os.path.exists(init_marker):
            with open(init_marker, "w", encoding="utf-8") as f:
                f.write("")

    try:
        local_src = os.path.join(local_pkg_dir, mod_filename)
        shutil.copy2(abs_src_path, local_src)
        rel_src_posix = f"{rel_pkg_dir_posix}/{mod_filename}"

        setup_script_path = os.path.join(mod_work_dir, "_setup_mod.py")
        with open(setup_script_path, "w", encoding="utf-8") as f:
            f.write(
                _SINGLE_MODULE_SETUP_TEMPLATE.format(
                    dotted_name=dotted_name,
                    rel_src_posix=rel_src_posix,
                    safe_name=safe_name,
                )
            )

        res = subprocess.run(
            [sys.executable, "_setup_mod.py", "build_ext", "--inplace"],
            cwd=mod_work_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if res.returncode != 0:
            raise RuntimeError(
                f"Cython build failed for {dotted_name}:\n"
                f"STDOUT:\n{res.stdout[-1500:]}\nSTDERR:\n{res.stderr[-1500:]}"
            )

        pyds = [f for f in os.listdir(local_pkg_dir) if f.endswith(".pyd")]
        if not pyds:
            raise RuntimeError(f"Cython build succeeded but no .pyd output found for: {dotted_name}")

        pyd_filename = pyds[0]
        dst_pkg_dir = os.path.join(stage_dir, rel_pkg_os)
        os.makedirs(dst_pkg_dir, exist_ok=True)
        dst_pyd = os.path.join(dst_pkg_dir, pyd_filename)
        shutil.move(os.path.join(local_pkg_dir, pyd_filename), dst_pyd)
        return dotted_name, rel_pkg_dir_posix, dst_pyd
    finally:
        _remove_path_safe(mod_work_dir)


def _preflight_check(
    project_root: str,
    modules: list[tuple[str, str, str]],
    spec: BuildProfileSpec,
) -> str | None:
    print(f"[STEP 1/6] Running preflight checks for profile '{spec.key}'...")

    for pkg in ("Cython", "PyInstaller", "setuptools"):
        try:
            __import__(pkg)
        except ImportError as exc:
            raise RuntimeError(
                f"Missing required packaging dependency '{pkg}'. Run: pip install {pkg}"
            ) from exc

    if not modules:
        raise RuntimeError("No compilable package modules discovered under src/onmyoji_helper/!")

    if not os.path.isfile(os.path.join(project_root, BOOTSTRAP_FILE)):
        raise FileNotFoundError(f"Bootstrap entrypoint missing: {BOOTSTRAP_FILE}")

    if spec.include_yolo_weights and not os.path.isfile(os.path.join(project_root, YOLO_WEIGHT_FILE)):
        print(f"   [WARN] {YOLO_WEIGHT_FILE} not found in project root; weight copy will be skipped.")

    if spec.include_local_env and not os.path.isfile(os.path.join(project_root, ".env")):
        print(
            f"   [WARN] Profile '{spec.key}' expects a local .env for cloud whitelist auth, "
            "but .env was not found in project root."
        )

    discovered_gguf: str | None = None
    if spec.include_local_gguf:
        discovered_gguf = discover_local_gguf_model(project_root)
        if discovered_gguf:
            size_gb = os.path.getsize(discovered_gguf) / (1024 ** 3)
            print(f"   [OK] Discovered local GGUF model: {discovered_gguf} ({size_gb:.2f} GB)")
        else:
            print(
                f"   [WARN] No local .gguf model found in project root or LM Studio cache; "
                f"'{spec.key}' will rely on zero-VRAM rule fallback unless weights are added later."
            )

    print(
        f"   [OK] Preflight passed ({len(modules)} layered modules, target: dist/{spec.output_folder_name})"
    )
    return discovered_gguf


def _compile_all_modules(
    modules: list[tuple[str, str, str]],
    packages: list[str],
    cython_tmp_root: str,
    stage_dir: str,
    max_workers: int,
) -> list[tuple[str, str, str]]:
    workers = max(1, min(2, max_workers))
    print(
        f"\n[STEP 2/6] Compiling {len(modules)} modules into native hardened .pyd binaries "
        f"(workers={workers})..."
    )

    os.makedirs(cython_tmp_root, exist_ok=True)
    os.makedirs(stage_dir, exist_ok=True)

    for dotted_pkg in packages:
        pkg_dir = os.path.join(stage_dir, dotted_pkg.replace(".", os.sep))
        os.makedirs(pkg_dir, exist_ok=True)
        with open(os.path.join(pkg_dir, "__init__.py"), "w", encoding="utf-8") as f:
            f.write("")

    staged_results: list[tuple[str, str, str]] = []
    if workers == 1:
        for idx, (dotted_mod, rel_dir_posix, abs_src) in enumerate(modules, start=1):
            t0 = time.perf_counter()
            res_tuple = _compile_single_module(
                dotted_mod, rel_dir_posix, abs_src, cython_tmp_root, stage_dir
            )
            elapsed = time.perf_counter() - t0
            staged_results.append(res_tuple)
            print(
                f"   [{idx:02d}/{len(modules)}] [OK] {dotted_mod} -> "
                f"{os.path.basename(res_tuple[2])} ({elapsed:.1f}s)"
            )
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _compile_single_module, dotted_mod, rel_dir_posix, abs_src, cython_tmp_root, stage_dir
                ): dotted_mod
                for dotted_mod, rel_dir_posix, abs_src in modules
            }
            done_count = 0
            for fut in as_completed(futures):
                res_tuple = fut.result()
                done_count += 1
                staged_results.append(res_tuple)
                print(
                    f"   [{done_count:02d}/{len(modules)}] [OK] {res_tuple[0]} -> "
                    f"{os.path.basename(res_tuple[2])}"
                )

    staged_results.sort(key=lambda item: item[0])
    return staged_results


def build_pyinstaller_command(
    project_root: str,
    stage_dir: str,
    work_dir: str,
    pyi_dist_holder: str,
    staged_bootstrap: str,
    staged_pyds: list[tuple[str, str, str]],
    packages: list[str],
    spec: BuildProfileSpec,
    has_llama_cpp: bool | None = None,
) -> list[str]:
    """Constructs the complete PyInstaller CLI command with explicit `.pyd`-hidden dependencies."""
    llama_installed = _is_module_available("llama_cpp") if has_llama_cpp is None else has_llama_cpp

    cmd: list[str] = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--windowed",
        f"--contents-directory={CONTENTS_DIR_NAME}",
        f"--name={APP_NAME}",
        f"--distpath={pyi_dist_holder}",
        f"--workpath={work_dir}",
        f"--specpath={work_dir}",
        f"--paths={stage_dir}",
    ]

    if spec.require_uac_admin:
        cmd.append("--uac-admin")

    icon_path = os.path.join(project_root, ICON_FILE)
    if os.path.isfile(icon_path):
        cmd.append(f"--icon={icon_path}")

    for dotted_pkg in packages:
        cmd.append(f"--hidden-import={dotted_pkg}")

    for dotted_mod, rel_dir_posix, pyd_path in staged_pyds:
        cmd.append(f"--add-binary={pyd_path};{rel_dir_posix}")
        cmd.append(f"--hidden-import={dotted_mod}")

    # 1. Mandatory hidden imports & pywin32 DLLs across ALL 4 profiles
    for dep in COMMON_HIDDEN_IMPORTS:
        if _is_module_available(dep.split(".")[0]) or dep in sys.stdlib_module_names:
            cmd.append(f"--hidden-import={dep}")
    if _is_module_available("pywintypes"):
        cmd.append("--collect-binaries=pywin32_system32")
        purelib = sysconfig.get_paths().get("purelib", "")
        pywin32_sys_dir = os.path.join(purelib, "pywin32_system32") if purelib else ""
        if pywin32_sys_dir and os.path.isdir(pywin32_sys_dir):
            for dll_name in sorted(os.listdir(pywin32_sys_dir)):
                if dll_name.lower().endswith(".dll"):
                    cmd.append(f"--add-binary={os.path.join(pywin32_sys_dir, dll_name)};.")

    # 2. Profile-specific CV / OCR / LLM collection vs exclusion
    if spec.exclude_heavy_cv_runtime:
        for ex in HEAVY_RUNTIME_EXCLUDES:
            cmd.append(f"--exclude-module={ex}")
    else:
        for dep in LIVE_COMBAT_HIDDEN_IMPORTS:
            cmd.append(f"--hidden-import={dep}")
        for pkg in LIVE_COMBAT_COLLECT_ALL:
            cmd.append(f"--collect-all={pkg}")
        for bin_pkg in LIVE_COMBAT_COLLECT_BINARIES:
            cmd.append(f"--collect-binaries={bin_pkg}")

        if spec.exclude_llama_cpp:
            cmd.append("--exclude-module=llama_cpp")
        elif llama_installed:
            cmd.append("--hidden-import=llama_cpp")
            cmd.append("--collect-all=llama_cpp")

    cmd.append(staged_bootstrap)
    return cmd


def _run_pyinstaller(
    project_root: str,
    stage_dir: str,
    work_dir: str,
    dist_dir: str,
    staged_pyds: list[tuple[str, str, str]],
    packages: list[str],
    spec: BuildProfileSpec,
) -> str:
    uac_desc = "--uac-admin" if spec.require_uac_admin else "standard-user (no UAC prompt)"
    print(f"\n[STEP 3/6] Freezing {APP_NAME}.exe via PyInstaller (--onedir --windowed, {uac_desc})...")

    staged_bootstrap = os.path.join(stage_dir, BOOTSTRAP_FILE)
    shutil.copy2(os.path.join(project_root, BOOTSTRAP_FILE), staged_bootstrap)

    pyi_dist_holder = os.path.join(work_dir, "_pyi_dist")
    _remove_path_safe(pyi_dist_holder)

    cmd = build_pyinstaller_command(
        project_root=project_root,
        stage_dir=stage_dir,
        work_dir=work_dir,
        pyi_dist_holder=pyi_dist_holder,
        staged_bootstrap=staged_bootstrap,
        staged_pyds=staged_pyds,
        packages=packages,
        spec=spec,
    )

    res = subprocess.run(cmd, cwd=stage_dir)
    if res.returncode != 0:
        raise RuntimeError(f"PyInstaller failed with exit code {res.returncode}")

    built_dir = os.path.join(pyi_dist_holder, APP_NAME)
    target_output_dir = os.path.join(dist_dir, spec.output_folder_name)
    _remove_path_safe(target_output_dir)
    os.makedirs(dist_dir, exist_ok=True)
    shutil.move(built_dir, target_output_dir)

    print(f"   [OK] PyInstaller bundle staged at: {target_output_dir}")
    return target_output_dir


def _copy_profile_assets(
    project_root: str,
    output_app_dir: str,
    spec: BuildProfileSpec,
    discovered_gguf: str | None,
) -> str | None:
    print(f"\n[STEP 4/6] Configuring assets and runtime flags in {CONTENTS_DIR_NAME}/...")

    data_dir = os.path.join(output_app_dir, CONTENTS_DIR_NAME)
    os.makedirs(data_dir, exist_ok=True)

    # 1. YOLOv8 weights (`best.pt`)
    if spec.include_yolo_weights:
        yolo_src = os.path.join(project_root, YOLO_WEIGHT_FILE)
        if os.path.isfile(yolo_src):
            shutil.copy2(yolo_src, os.path.join(data_dir, YOLO_WEIGHT_FILE))
            size_mb = os.path.getsize(yolo_src) / (1024 * 1024)
            print(f"   [OK] Copied {YOLO_WEIGHT_FILE} -> {CONTENTS_DIR_NAME}/{YOLO_WEIGHT_FILE} ({size_mb:.1f} MB)")

    # 2. Always include `.env.example` template
    env_example_src = os.path.join(project_root, ".env.example")
    if os.path.isfile(env_example_src):
        shutil.copy2(env_example_src, os.path.join(data_dir, ".env.example"))

    # 3. Optional local GGUF model copy into `data/models/<filename>.gguf`
    bundled_rel_gguf: str | None = None
    if spec.include_local_gguf and discovered_gguf and os.path.isfile(discovered_gguf):
        models_dir = os.path.join(data_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        gguf_fname = os.path.basename(discovered_gguf)
        dst_gguf = os.path.join(models_dir, gguf_fname)
        size_gb = os.path.getsize(discovered_gguf) / (1024 ** 3)
        print(
            f"   [INFO] Copying quantized GGUF model '{gguf_fname}' ({size_gb:.2f} GB) "
            f"-> {CONTENTS_DIR_NAME}/models/{gguf_fname} ..."
        )
        shutil.copy2(discovered_gguf, dst_gguf)
        bundled_rel_gguf = f"data/models/{gguf_fname}"
        print(f"   [OK] Bundled standalone GGUF model: {bundled_rel_gguf}")
    elif not spec.include_local_gguf:
        print(f"   [OK] Profile '{spec.key}' excludes heavy GGUF weights (zero-VRAM rule tier).")

    # 4. Credential & runtime `.env` policy (never mutates root `.env`!)
    dst_dotenv = os.path.join(data_dir, ".env")
    if spec.include_local_env:
        src_dotenv = os.path.join(project_root, ".env")
        env_updates: dict[str, str] = {}
        env_removals: set[str] = set()

        if bundled_rel_gguf:
            env_updates["LOCAL_LLM_MODEL_PATH"] = bundled_rel_gguf
        if spec.force_lite_engine:
            env_updates["FORCE_LITE_ENGINE"] = "1"
            env_removals.add("LOCAL_LLM_MODEL_PATH")
        else:
            env_removals.add("FORCE_LITE_ENGINE")
        if spec.default_log_level:
            env_updates["ONMYOJI_LOG_LEVEL"] = spec.default_log_level

        if os.path.isfile(src_dotenv) or env_updates:
            rewrite_packaged_dotenv(
                src_dotenv_path=src_dotenv,
                dst_dotenv_path=dst_dotenv,
                updates=env_updates,
                remove_keys=env_removals,
            )
            print(
                f"   [OK] Packaged {CONTENTS_DIR_NAME}/.env "
                f"(overrides={env_updates or 'none'}, root .env untouched)."
            )
    else:
        _remove_path_safe(dst_dotenv)
        print(
            f"   [OK] Credential-free isolation active: excluded .env "
            f"(only {CONTENTS_DIR_NAME}/.env.example included)."
        )

    # 5. HR Demo zero-argument startup marker (`data/demo_default.flag`)
    flag_path = os.path.join(data_dir, DEMO_FLAG_FILE)
    if spec.write_demo_default_flag:
        with open(flag_path, "w", encoding="utf-8") as f:
            f.write("OFFLINE_DEMO_DEFAULT=1\n")
        print(
            f"   [OK] Created {CONTENTS_DIR_NAME}/{DEMO_FLAG_FILE} "
            f"(double-clicking {APP_NAME}.exe launches Offline Demo directly)."
        )
    else:
        _remove_path_safe(flag_path)

    return bundled_rel_gguf


def _create_zip_archive(source_dir: str, zip_output_path: str) -> float:
    """Compresses `source_dir` into `zip_output_path` preserving its top-level folder name."""
    _remove_path_safe(zip_output_path)
    parent_dir = os.path.dirname(os.path.abspath(source_dir))
    with zipfile.ZipFile(zip_output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for root_dir, _, files in os.walk(source_dir):
            for fname in sorted(files):
                abs_file = os.path.join(root_dir, fname)
                arcname = os.path.relpath(abs_file, parent_dir)
                zf.write(abs_file, arcname)
    return os.path.getsize(zip_output_path) / (1024 * 1024)


def _cleanup_build_artifacts(
    project_root: str,
    cython_tmp_root: str,
    stage_dir: str,
    work_dir: str,
) -> None:
    print("\n[STEP 5/6] Cleaning temporary build directories and intermediate C files...")
    for d in (cython_tmp_root, stage_dir, work_dir, os.path.join(project_root, "build")):
        _remove_path_safe(d)

    for pattern in ("*.c", "*.spec", "*.pyd"):
        for fpath in glob.glob(os.path.join(project_root, pattern)):
            _remove_path_safe(fpath)
    print("   [OK] Temporary build artifacts removed.")


def _verify_release_integrity(
    project_root: str,
    output_app_dir: str,
    expected_modules: list[tuple[str, str, str]],
    spec: BuildProfileSpec,
    bundled_rel_gguf: str | None,
) -> None:
    print("\n[STEP 6/6] Running post-build security, manifest, and zero-source-leak audit...")

    exe_name = f"{APP_NAME}.exe"
    exe_path = os.path.join(output_app_dir, exe_name)
    data_dir = os.path.join(output_app_dir, CONTENTS_DIR_NAME)
    pkg_out_dir = os.path.join(data_dir, PACKAGE_NAME)

    if not os.path.isfile(exe_path):
        raise AssertionError(f"Executable missing: {exe_path}")
    if not os.path.isdir(data_dir):
        raise AssertionError(f"Data directory missing: {data_dir}")

    # 1. Verify UAC manifest matches profile privilege policy
    with open(exe_path, "rb") as exe_f:
        exe_bytes = exe_f.read()
    if spec.require_uac_admin and b"requireAdministrator" not in exe_bytes:
        raise AssertionError(f"Missing requireAdministrator UAC manifest in: {exe_path}")
    if not spec.require_uac_admin and b"requireAdministrator" in exe_bytes:
        raise AssertionError(f"HR Demo build must NOT require UAC Administrator elevation: {exe_path}")

    # 2. Verify root directory cleanliness (`['OnmyojiHelper.exe', 'data']` only)
    root_entries = sorted(os.listdir(output_app_dir))
    expected_root = sorted([exe_name, CONTENTS_DIR_NAME])
    if root_entries != expected_root:
        raise AssertionError(f"Unexpected root directory layout: expected={expected_root}, got={root_entries}")

    # 3. Verify zero `.py` source files and 100% `.pyd` module coverage
    leaked_py_files: list[str] = []
    for check_dir in (output_app_dir, data_dir):
        for fname in os.listdir(check_dir):
            full_p = os.path.join(check_dir, fname)
            if os.path.isfile(full_p) and fname.endswith(".py"):
                leaked_py_files.append(full_p)

    found_dotted_pyds: set[str] = set()
    if os.path.isdir(pkg_out_dir):
        for r_dir, _, fnames in os.walk(pkg_out_dir):
            rel_pkg = os.path.relpath(r_dir, data_dir).replace(os.sep, ".")
            for fname in fnames:
                if fname.endswith(".py"):
                    leaked_py_files.append(os.path.join(r_dir, fname))
                elif fname.endswith(".pyd"):
                    stem = fname.split(".")[0]
                    found_dotted_pyds.add(f"{rel_pkg}.{stem}")

    if leaked_py_files:
        raise AssertionError(f"Leaked .py source files detected in output bundle: {leaked_py_files}")

    expected_dotted = {item[0] for item in expected_modules}
    missing_pyds = sorted(expected_dotted - found_dotted_pyds)
    if missing_pyds:
        raise AssertionError(f"Missing compiled .pyd modules in {CONTENTS_DIR_NAME}/{PACKAGE_NAME}/: {missing_pyds}")

    # 4. Verify credential & flag rules
    data_dotenv = os.path.join(data_dir, ".env")
    if spec.include_local_env and os.path.isfile(os.path.join(project_root, ".env")):
        if not os.path.isfile(data_dotenv):
            raise AssertionError(f"Expected packaged .env missing: {data_dotenv}")
        packaged_kv = _parse_simple_dotenv(data_dotenv)
        if spec.force_lite_engine and packaged_kv.get("FORCE_LITE_ENGINE") != "1":
            raise AssertionError("FORCE_LITE_ENGINE=1 missing from friend-lite packaged data/.env")
        if spec.default_log_level and packaged_kv.get("ONMYOJI_LOG_LEVEL") != spec.default_log_level:
            raise AssertionError(f"ONMYOJI_LOG_LEVEL={spec.default_log_level} missing from packaged data/.env")
        if bundled_rel_gguf and packaged_kv.get("LOCAL_LLM_MODEL_PATH") != bundled_rel_gguf:
            raise AssertionError("LOCAL_LLM_MODEL_PATH override missing from packaged data/.env")
    elif not spec.include_local_env:
        if os.path.exists(data_dotenv) or os.path.exists(os.path.join(output_app_dir, ".env")):
            raise AssertionError("CRITICAL: Private .env leaked into credential-free build!")

    if spec.write_demo_default_flag and not os.path.isfile(os.path.join(data_dir, DEMO_FLAG_FILE)):
        raise AssertionError(f"Missing {DEMO_FLAG_FILE} in HR Demo build!")

    if bundled_rel_gguf:
        abs_bundled_gguf = os.path.join(output_app_dir, bundled_rel_gguf.replace("/", os.sep))
        if not os.path.isfile(abs_bundled_gguf):
            raise AssertionError(f"Bundled GGUF file missing: {abs_bundled_gguf}")

    # 5. Verify `win32gui` & `pywintypes` native binaries exist inside `data/` (`_internal/` equivalent)
    win32gui_binaries: list[str] = []
    pywintypes_dlls: list[str] = []
    for r_dir, _, fnames in os.walk(data_dir):
        for fn in fnames:
            low_fn = fn.lower()
            if low_fn.startswith("win32gui") and low_fn.endswith(".pyd"):
                win32gui_binaries.append(os.path.join(r_dir, fn))
            elif low_fn.startswith("pywintypes") and low_fn.endswith(".dll"):
                pywintypes_dlls.append(os.path.join(r_dir, fn))

    if not win32gui_binaries:
        raise AssertionError(f"CRITICAL: win32gui*.pyd missing from bundled {CONTENTS_DIR_NAME}/ directory!")
    if not pywintypes_dlls:
        raise AssertionError(f"CRITICAL: pywintypes*.dll missing from bundled {CONTENTS_DIR_NAME}/ directory!")

    # 6. Execute live `.exe --smoke-check` subprocess before allowing `.zip` packaging!
    print(f"   [OK] Verified bundled pywin32 binaries ({os.path.basename(win32gui_binaries[0])}, {os.path.basename(pywintypes_dlls[0])})")
    print(f"   [SMOKE] Launching {exe_name} --smoke-check to verify zero-crash startup...")
    smoke_env = dict(os.environ)
    smoke_env["__COMPAT_LAYER"] = "RunAsInvoker"
    smoke_env["ONMYOJI_SKIP_UAC_FOR_TEST"] = "1"
    smoke_res = subprocess.run(
        [exe_path, "--smoke-check"],
        cwd=output_app_dir,
        env=smoke_env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=35,
        check=False,
    )
    if smoke_res.returncode != 0:
        raise AssertionError(
            f"Compiled {exe_name} --smoke-check failed with exit code {smoke_res.returncode}!\n"
            f"STDOUT:\n{smoke_res.stdout}\nSTDERR:\n{smoke_res.stderr}"
        )
    print(f"   [OK] {exe_name} --smoke-check exited cleanly with code 0!")

    total_bytes = sum(
        os.path.getsize(os.path.join(r, fn))
        for r, _, fns in os.walk(output_app_dir)
        for fn in fns
    )
    print(f"   [OK] Clean root layout verified: {root_entries}")
    print(f"   [OK] Native .pyd coverage verified: {len(found_dotted_pyds)}/{len(expected_dotted)} (0 .py leaks)")
    print(f"   [OK] Total bundle directory size: {total_bytes / (1024 * 1024):.1f} MB")


def _prompt_interactive_profile() -> str:
    print("============================================================================")
    print("  Onmyoji Tactical Helper - Standalone Release Packaging Menu")
    print("============================================================================")
    print("  [1] Personal Full Edition       -> dist/OnmyojiHelper_Personal")
    print("      (33 .pyd + local .env + best.pt + local .gguf if available)")
    print("  [2] HR / Recruiter Demo Edition -> dist/OnmyojiHelper_HR_Demo + Win64.zip")
    print("      (33 .pyd + ZERO .env + auto-launches Offline Demo without CLI args)")
    print("  [3] Friend Full Edition (CUDA)  -> dist/OnmyojiHelper_Friend_Full")
    print("      (Hardened .pyd + WARNING log + whitelist .env + bundled local .gguf)")
    print("  [4] Friend Lite Edition (No GPU)-> dist/OnmyojiHelper_Friend_Lite")
    print("      (Hardened .pyd + whitelist .env + excludes llama_cpp + FORCE_LITE_ENGINE=1)")
    print("============================================================================")
    choice = input("Select edition [1-4, press Enter for default 2 (hr-demo)]: ").strip()
    if not choice:
        return "hr-demo"
    if choice in MENU_TO_PROFILE:
        return MENU_TO_PROFILE[choice]
    low = choice.lower()
    if low in PROFILES:
        return low
    raise ValueError(f"Invalid selection '{choice}'. Expected 1, 2, 3, 4 or profile name.")


def resolve_profile_spec(args: argparse.Namespace, is_tty: bool | None = None) -> BuildProfileSpec:
    """Resolves the active `BuildProfileSpec` from CLI flags or interactive menu prompt."""
    if args.profile:
        key = args.profile.strip().lower()
        if key in MENU_TO_PROFILE:
            key = MENU_TO_PROFILE[key]
        if key not in PROFILES:
            raise ValueError(f"Unknown profile '{args.profile}'. Valid choices: {list(PROFILES)}")
        return PROFILES[key]

    # Legacy CLI flag compatibility (`--include-local-env` / `--full`)
    if args.include_local_env and args.full:
        return PROFILES["personal"]
    if args.include_local_env and not args.full:
        return PROFILES["friend-lite"]
    if args.full and not args.include_local_env:
        return PROFILES["friend-full"]

    tty = bool(sys.stdin and sys.stdin.isatty()) if is_tty is None else is_tty
    if tty and not args.dry_run:
        chosen_key = _prompt_interactive_profile()
        return PROFILES[chosen_key]

    return PROFILES["hr-demo"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Onmyoji Tactical Helper - Multi-Profile Cython .pyd & PyInstaller Release Builder"
    )
    parser.add_argument(
        "--profile",
        choices=[*PROFILES.keys(), *MENU_TO_PROFILE.keys()],
        default=None,
        help="Target release edition: personal | hr-demo | friend-full | friend-lite (or 1..4).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Maximum parallel Cython compilation workers (default: 1 for low RAM, max: 2).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Legacy alias: selects a full-LLM profile when --profile is omitted.",
    )
    parser.add_argument(
        "--include-local-env",
        action="store_true",
        help="Legacy alias: includes local .env when --profile is omitted.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run preflight module discovery and asset checks without compiling or freezing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    spec = resolve_profile_spec(args)

    project_root = os.path.dirname(os.path.abspath(__file__))
    src_root = os.path.join(project_root, "src")
    cython_tmp_root = os.path.join(project_root, "_cython_tmp")
    stage_dir = os.path.join(project_root, "_release_stage")
    work_dir = os.path.join(project_root, "_pyinstaller_work")
    dist_dir = os.path.join(project_root, "dist")

    start_time = time.perf_counter()
    print("=" * 76)
    print(f"  {APP_NAME} Release Builder | Edition [{spec.menu_number}]: {spec.title}")
    print("=" * 76)

    modules, packages = discover_package_modules(src_root)
    zip_created_path: str | None = None

    try:
        discovered_gguf = _preflight_check(project_root, modules, spec=spec)
        if args.dry_run:
            print(f"\n[OK] [--dry-run] Preflight verification passed for profile '{spec.key}'.")
            return

        _remove_path_safe(stage_dir)
        _remove_path_safe(work_dir)

        staged_pyds = _compile_all_modules(
            modules=modules,
            packages=packages,
            cython_tmp_root=cython_tmp_root,
            stage_dir=stage_dir,
            max_workers=args.workers,
        )
        output_app_dir = _run_pyinstaller(
            project_root=project_root,
            stage_dir=stage_dir,
            work_dir=work_dir,
            dist_dir=dist_dir,
            staged_pyds=staged_pyds,
            packages=packages,
            spec=spec,
        )
        bundled_rel_gguf = _copy_profile_assets(
            project_root=project_root,
            output_app_dir=output_app_dir,
            spec=spec,
            discovered_gguf=discovered_gguf,
        )
        _verify_release_integrity(
            project_root=project_root,
            output_app_dir=output_app_dir,
            expected_modules=modules,
            spec=spec,
            bundled_rel_gguf=bundled_rel_gguf,
        )

        if spec.create_win64_zip and spec.zip_filename:
            zip_created_path = os.path.join(dist_dir, spec.zip_filename)
            print(f"\n[ARCHIVE] Compressing HR Demo bundle -> {zip_created_path} ...")
            zip_mb = _create_zip_archive(output_app_dir, zip_created_path)
            print(f"   [OK] Created portable archive: {zip_created_path} ({zip_mb:.1f} MB)")
    finally:
        if not args.dry_run:
            _cleanup_build_artifacts(project_root, cython_tmp_root, stage_dir, work_dir)

    total_sec = time.perf_counter() - start_time
    final_dir = os.path.join(dist_dir, spec.output_folder_name)
    print("\n" + "=" * 76)
    print(f"  [SUCCESS] Build completed in {total_sec:.1f}s!")
    print(f"  Output Folder : {final_dir}")
    if zip_created_path:
        print(f"  Portable ZIP  : {zip_created_path}")
    print(f"  Executable    : {os.path.join(final_dir, APP_NAME + '.exe')}")
    print("=" * 76)


if __name__ == "__main__":
    main()
