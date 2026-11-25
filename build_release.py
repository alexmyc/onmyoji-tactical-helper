"""Low-RAM, Hardened Package Release Builder for Onmyoji Tactical Helper.

Discovers and compiles all Python modules under `src/onmyoji_helper/` sequentially into
native Windows `.pyd` C-extensions via Cython 3.3 with anti-decompilation hardening
(stripped docstrings, bindings, line traces, code comments, and debug symbols), freezes
`run_app.py` via PyInstaller (`--onedir --windowed --uac-admin --contents-directory=data`),
copies model weights directly into `data/` without archive re-compression, cleans all
temporary build artifacts, and audits the output for strict root cleanliness, embedded
UAC manifest, 100% `.pyd` coverage, and zero `.py` source leaks.

Usage:
    python build_release.py             # Lite build (recommended for friends: excludes 3GB GGUF)
    python build_release.py --full      # Full build (includes 3GB Gemma 4 GGUF + llama_cpp)
    python build_release.py --dry-run   # Validate environment, package tree, and assets only
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

APP_NAME: str = "OnmyojiHelper"
PACKAGE_NAME: str = "onmyoji_helper"
CONTENTS_DIR_NAME: str = "data"
BOOTSTRAP_FILE: str = "run_app.py"
ICON_FILE: str = "app.ico"
YOLO_WEIGHT_FILE: str = "best.pt"
GGUF_MODEL_FILE: str = "gemma-4-E2B-it-Q4_K_M.gguf"

# Third-party / stdlib hidden imports needed because PyInstaller cannot statically parse inside .pyd binaries
THIRD_PARTY_HIDDEN_IMPORTS: list[str] = [
    "cv2",
    "numpy",
    "PIL",
    "PIL.Image",
    "pygetwindow",
    "win32gui",
    "win32ui",
    "win32con",
    "win32api",
    "win32process",
    "winreg",
    "pywintypes",
    "ultralytics",
    "torch",
    "torchvision",
    "rapidocr_onnxruntime",
    "onnxruntime",
    "tkinter",
    "tkinter.font",
    "tkinter.messagebox",
    "argparse",
    "ctypes",
    "urllib.request",
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


def discover_package_modules(src_root: str) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Discovers all compilable `.py` modules and package directories under `src/onmyoji_helper`.

    Returns:
        Tuple of:
        - List of `(dotted_module_name, rel_pkg_dir_posix, abs_src_path)` for non-`__init__` modules.
        - Sorted list of dotted package names (e.g. `['onmyoji_helper', 'onmyoji_helper.app', ...]`).
    """
    pkg_root = os.path.join(src_root, PACKAGE_NAME)
    if not os.path.isdir(pkg_root):
        raise FileNotFoundError(f"找不到核心套件目錄: {pkg_root}")

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

    # Create empty __init__.py markers along the temp package hierarchy so Cython resolves package context
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
                f"Cython 編譯失敗 ({dotted_name}):\n"
                f"STDOUT:\n{res.stdout[-1500:]}\nSTDERR:\n{res.stderr[-1500:]}"
            )

        pyds = [f for f in os.listdir(local_pkg_dir) if f.endswith(".pyd")]
        if not pyds:
            raise RuntimeError(f"編譯完成但找不到產出的 .pyd 檔案: {dotted_name}")

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
    include_full_llm: bool,
) -> None:
    print("🔍 [Step 1/6] 正在執行建置前置檢查...")

    for pkg in ("Cython", "PyInstaller", "setuptools"):
        try:
            __import__(pkg)
        except ImportError as exc:
            raise RuntimeError(f"缺少必要打包工具 '{pkg}'，請先執行: pip install {pkg}") from exc

    if not modules:
        raise RuntimeError("未掃描到任何可編譯的核心模組！")

    if not os.path.isfile(os.path.join(project_root, BOOTSTRAP_FILE)):
        raise FileNotFoundError(f"缺少入口引導檔: {BOOTSTRAP_FILE}")

    if not os.path.isfile(os.path.join(project_root, YOLO_WEIGHT_FILE)):
        print(f"⚠️ 警告: 未找到 {YOLO_WEIGHT_FILE}，打包後將略過複製此權重檔。")

    if include_full_llm and not os.path.isfile(os.path.join(project_root, GGUF_MODEL_FILE)):
        print(f"⚠️ 警告: --full 模式未找到 {GGUF_MODEL_FILE}，執行時將自動退回輕量規則模式。")

    mode_str = "完整版 Full" if include_full_llm else "輕量版 Lite"
    print(f"✅ 前置檢查通過 (共 {len(modules)} 個分層核心模組，模式: {mode_str})")


def _compile_all_modules(
    modules: list[tuple[str, str, str]],
    packages: list[str],
    cython_tmp_root: str,
    stage_dir: str,
    max_workers: int,
) -> list[tuple[str, str, str]]:
    workers = max(1, min(2, max_workers))
    print(f"\n🛡️ [Step 2/6] 正在將 {len(modules)} 個分層模組編譯為原生防反編譯 .pyd (並發數={workers})...")

    os.makedirs(cython_tmp_root, exist_ok=True)
    os.makedirs(stage_dir, exist_ok=True)

    # Create empty __init__.py package markers in stage_dir (packed into PYZ-00.pyz by PyInstaller)
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
                f"   [{idx:02d}/{len(modules)}] ✅ {dotted_mod} -> {os.path.basename(res_tuple[2])} ({elapsed:.1f}s)"
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
                    f"   [{done_count:02d}/{len(modules)}] ✅ {res_tuple[0]} -> {os.path.basename(res_tuple[2])}"
                )

    staged_results.sort(key=lambda item: item[0])
    return staged_results


def _run_pyinstaller(
    project_root: str,
    stage_dir: str,
    work_dir: str,
    dist_dir: str,
    staged_pyds: list[tuple[str, str, str]],
    packages: list[str],
    include_full_llm: bool,
) -> None:
    print(f"\n📦 [Step 3/6] 正在透過 PyInstaller (--onedir --uac-admin) 封裝 {APP_NAME}.exe...")

    staged_bootstrap = os.path.join(stage_dir, BOOTSTRAP_FILE)
    shutil.copy2(os.path.join(project_root, BOOTSTRAP_FILE), staged_bootstrap)

    cmd: list[str] = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--windowed",
        "--uac-admin",
        f"--contents-directory={CONTENTS_DIR_NAME}",
        f"--name={APP_NAME}",
        f"--distpath={dist_dir}",
        f"--workpath={work_dir}",
        f"--specpath={work_dir}",
        f"--paths={stage_dir}",
        "--collect-all=rapidocr_onnxruntime",
    ]

    icon_path = os.path.join(project_root, ICON_FILE)
    if os.path.isfile(icon_path):
        cmd.append(f"--icon={icon_path}")

    for dotted_pkg in packages:
        cmd.append(f"--hidden-import={dotted_pkg}")

    for dotted_mod, rel_dir_posix, pyd_path in staged_pyds:
        cmd.append(f"--add-binary={pyd_path};{rel_dir_posix}")
        cmd.append(f"--hidden-import={dotted_mod}")

    for dep in THIRD_PARTY_HIDDEN_IMPORTS:
        cmd.append(f"--hidden-import={dep}")

    if include_full_llm:
        cmd.append("--collect-all=llama_cpp")
    else:
        cmd.append("--exclude-module=llama_cpp")

    cmd.append(staged_bootstrap)

    res = subprocess.run(cmd, cwd=stage_dir)
    if res.returncode != 0:
        raise RuntimeError(f"PyInstaller 打包失敗 (exit code={res.returncode})")

    print(f"✅ PyInstaller 封裝完成 -> {os.path.join(dist_dir, APP_NAME)} (內部資源收攏至 {CONTENTS_DIR_NAME}/)")


def _copy_model_assets(
    project_root: str,
    output_app_dir: str,
    include_full_llm: bool,
    include_local_env: bool = False,
) -> None:
    print(f"\n📂 [Step 4/6] 正在直接複製模型權重與範本資源至 {CONTENTS_DIR_NAME}/ (不經壓縮封裝，避免爆 RAM)...")

    data_dir = os.path.join(output_app_dir, CONTENTS_DIR_NAME)
    os.makedirs(data_dir, exist_ok=True)

    yolo_src = os.path.join(project_root, YOLO_WEIGHT_FILE)
    if os.path.isfile(yolo_src):
        shutil.copy2(yolo_src, os.path.join(data_dir, YOLO_WEIGHT_FILE))
        size_mb = os.path.getsize(yolo_src) / (1024 * 1024)
        print(f"   ✅ 已複製 {YOLO_WEIGHT_FILE} -> {CONTENTS_DIR_NAME}/{YOLO_WEIGHT_FILE} ({size_mb:.1f} MB)")

    env_example_src = os.path.join(project_root, ".env.example")
    if os.path.isfile(env_example_src):
        shutil.copy2(env_example_src, os.path.join(data_dir, ".env.example"))

    if include_local_env:
        dotenv_src = os.path.join(project_root, ".env")
        if os.path.isfile(dotenv_src):
            shutil.copy2(dotenv_src, os.path.join(data_dir, ".env"))
            print(f"   ✅ [--include-local-env] 已封裝本機私有 .env -> {CONTENTS_DIR_NAME}/.env")
        else:
            print("   ⚠️ 指定了 --include-local-env，但專案根目錄未找到 .env。")
    else:
        print(f"   🔒 預設安全模式：未複製私人 .env（僅附帶 {CONTENTS_DIR_NAME}/.env.example 無憑證範本）。")

    if include_full_llm:
        gguf_src = os.path.join(project_root, GGUF_MODEL_FILE)
        if os.path.isfile(gguf_src):
            print(f"   ⏳ 正在複製大型本地模型 {GGUF_MODEL_FILE} 至 {CONTENTS_DIR_NAME}/，請稍候...")
            shutil.copy2(gguf_src, os.path.join(data_dir, GGUF_MODEL_FILE))
            size_gb = os.path.getsize(gguf_src) / (1024 ** 3)
            print(f"   ✅ 已複製 {GGUF_MODEL_FILE} -> {CONTENTS_DIR_NAME}/{GGUF_MODEL_FILE} ({size_gb:.2f} GB)")
    else:
        print("   ⚡ 輕量版 (Lite) 已略過 3GB 本地 GGUF 模型，將以 0 顯存純規則模式極速運行。")


def _cleanup_build_artifacts(
    project_root: str,
    cython_tmp_root: str,
    stage_dir: str,
    work_dir: str,
) -> None:
    print("\n🧹 [Step 5/6] 正在自動清理暫存建置垃圾...")

    for d in (cython_tmp_root, stage_dir, work_dir, os.path.join(project_root, "build")):
        _remove_path_safe(d)

    for pattern in ("*.c", "*.spec", "*.pyd"):
        for fpath in glob.glob(os.path.join(project_root, pattern)):
            _remove_path_safe(fpath)

    print("✅ 暫存檔 (build/、*.c、*.spec、暫存目錄) 清理完畢！")


def _verify_zero_source_leak(
    project_root: str,
    output_app_dir: str,
    expected_modules: list[tuple[str, str, str]],
    include_full_llm: bool,
    include_local_env: bool = False,
) -> None:
    print("\n🔐 [Step 6/6] 正在執行極簡目錄結構、UAC 清單、零源碼洩漏與產物完整性驗證...")

    exe_name = f"{APP_NAME}.exe"
    exe_path = os.path.join(output_app_dir, exe_name)
    data_dir = os.path.join(output_app_dir, CONTENTS_DIR_NAME)
    pkg_out_dir = os.path.join(data_dir, PACKAGE_NAME)

    if not os.path.isfile(exe_path):
        raise AssertionError(f"找不到最終執行檔: {exe_path}")
    if not os.path.isdir(data_dir):
        raise AssertionError(f"找不到核心資源目錄: {data_dir}")

    # 1. Verify UAC requireAdministrator manifest
    with open(exe_path, "rb") as exe_f:
        exe_bytes = exe_f.read()
    if b"requireAdministrator" not in exe_bytes:
        raise AssertionError(f"執行檔未內建 UAC requireAdministrator 權限清單: {exe_path}")

    # 2. Strict root directory cleanliness check
    root_entries = sorted(os.listdir(output_app_dir))
    expected_root_entries = sorted([exe_name, CONTENTS_DIR_NAME])
    if root_entries != expected_root_entries:
        raise AssertionError(
            f"發佈根目錄存在多餘或缺失項目！預期={expected_root_entries}，實際={root_entries}"
        )

    # 3. Verify zero leaked project .py files and 100% compiled .pyd modules under data/onmyoji_helper/
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
        raise AssertionError(f"偵測到未清理的自研 .py 源碼檔洩漏: {leaked_py_files}")

    expected_dotted = {item[0] for item in expected_modules}
    missing_pyds = sorted(expected_dotted - found_dotted_pyds)
    if missing_pyds:
        raise AssertionError(f"{CONTENTS_DIR_NAME}/{PACKAGE_NAME}/ 缺少已編譯的 .pyd 模組: {missing_pyds}")

    # 4. Verify model weights & local .env inside data/
    if os.path.isfile(os.path.join(project_root, YOLO_WEIGHT_FILE)):
        data_yolo = os.path.join(data_dir, YOLO_WEIGHT_FILE)
        if not os.path.isfile(data_yolo):
            raise AssertionError(f"找不到模型權重檔: {data_yolo}")

    data_dotenv = os.path.join(data_dir, ".env")
    if include_local_env and os.path.isfile(os.path.join(project_root, ".env")):
        if not os.path.isfile(data_dotenv):
            raise AssertionError(f"找不到封裝的私有設定檔: {data_dotenv}")
    elif not include_local_env:
        if os.path.exists(data_dotenv) or os.path.exists(os.path.join(output_app_dir, ".env")):
            raise AssertionError("預設無憑證模式下偵測到私人 .env 被意外複製進發佈目錄！")

    if include_full_llm and os.path.isfile(os.path.join(project_root, GGUF_MODEL_FILE)):
        data_gguf = os.path.join(data_dir, GGUF_MODEL_FILE)
        if not os.path.isfile(data_gguf):
            raise AssertionError(f"完整版缺少大模型權重檔: {data_gguf}")

    total_bytes = sum(
        os.path.getsize(os.path.join(r, fn))
        for r, _, fns in os.walk(output_app_dir)
        for fn in fns
    )
    print(f"   ✅ 極簡根目錄驗證通過: 僅包含 {root_entries}")
    print(f"   ✅ 執行檔存在且已內建 UAC 管理員權限清單 (requireAdministrator): {exe_path}")
    print(
        f"   ✅ {CONTENTS_DIR_NAME}/{PACKAGE_NAME}/ 分層 .pyd 二進制模組齊全: "
        f"{len(found_dotted_pyds)}/{len(expected_dotted)}"
    )
    if os.path.isfile(os.path.join(data_dir, YOLO_WEIGHT_FILE)):
        print(f"   ✅ 模型權重檔就位: {CONTENTS_DIR_NAME}/{YOLO_WEIGHT_FILE}")
    if os.path.isfile(os.path.join(data_dir, ".env")):
        print(f"   ✅ 私有雲端連線設定就位: {CONTENTS_DIR_NAME}/.env")
    print("   ✅ 打包驗證通過：零自研 .py 源碼洩漏！")
    print(f"   📊 最終發佈資料夾總大小: {total_bytes / (1024 * 1024):.1f} MB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Onmyoji Tactical Helper 低記憶體、分層套件防反編譯一鍵打包工具"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="打包完整版（附帶 3GB Gemma 4 GGUF 模型與 llama_cpp；預設為輕量版 Lite）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Cython 編譯最大並發數（預設 1 逐一編譯最省 RAM，上限 2）",
    )
    parser.add_argument(
        "--include-local-env",
        action="store_true",
        help="顯式將本機私有 .env 封裝至 data/.env（預設 False：僅複製 .env.example，確保預設產物無憑證殘留）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="僅執行環境、套件模組與權重前置檢查，不實際編譯與打包",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = os.path.dirname(os.path.abspath(__file__))
    src_root = os.path.join(project_root, "src")
    cython_tmp_root = os.path.join(project_root, "_cython_tmp")
    stage_dir = os.path.join(project_root, "_release_stage")
    work_dir = os.path.join(project_root, "_pyinstaller_work")
    dist_dir = os.path.join(project_root, "dist")
    output_app_dir = os.path.join(dist_dir, APP_NAME)

    start_time = time.perf_counter()
    print("=" * 60)
    print(f"🚀 {APP_NAME} 分層套件防反編譯正式打包程序啟動")
    print("=" * 60)

    modules, packages = discover_package_modules(src_root)

    try:
        _preflight_check(project_root, modules, include_full_llm=args.full)
        if args.dry_run:
            print("\n✅ [--dry-run] 前置環境與套件檔案檢查完全通過，隨時可正式打包！")
            return

        _remove_path_safe(stage_dir)
        _remove_path_safe(work_dir)
        _remove_path_safe(output_app_dir)

        staged_pyds = _compile_all_modules(
            modules=modules,
            packages=packages,
            cython_tmp_root=cython_tmp_root,
            stage_dir=stage_dir,
            max_workers=args.workers,
        )
        _run_pyinstaller(
            project_root=project_root,
            stage_dir=stage_dir,
            work_dir=work_dir,
            dist_dir=dist_dir,
            staged_pyds=staged_pyds,
            packages=packages,
            include_full_llm=args.full,
        )
        _copy_model_assets(
            project_root,
            output_app_dir,
            include_full_llm=args.full,
            include_local_env=args.include_local_env,
        )
        _verify_zero_source_leak(
            project_root,
            output_app_dir,
            expected_modules=modules,
            include_full_llm=args.full,
            include_local_env=args.include_local_env,
        )
    finally:
        if not args.dry_run:
            _cleanup_build_artifacts(project_root, cython_tmp_root, stage_dir, work_dir)

    total_sec = time.perf_counter() - start_time
    print("\n" + "=" * 60)
    print(f"🎉 正式版打包全部完成！(總耗時: {total_sec:.1f} 秒)")
    print(f"📁 輸出路徑: {output_app_dir}")
    print(f"👉 壓縮整個 '{APP_NAME}' 資料夾傳給朋友，解壓雙擊 {APP_NAME}.exe 即可使用！")
    print("=" * 60)


if __name__ == "__main__":
    main()
