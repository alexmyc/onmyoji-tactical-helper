"""Unit Tests for Dual-Track `.env` Isolation, Public Sandbox Mode, and `--demo` Scenarios."""

import logging
import subprocess
import sys
import tomllib
from pathlib import Path

import build_release
import pytest
import run_app

import onmyoji_helper
from onmyoji_helper.decision.draft_engine import build_database
from onmyoji_helper.decision.llm_engine import (
    TacticalEngine,
    evaluate_hardware_tier,
    is_force_lite_engine,
    resolve_configured_gguf_path,
)
from onmyoji_helper.domain.models import OperatingMode
from onmyoji_helper.domain.public_demo_data import (
    PUBLIC_DEMO_SYSTEMS,
    format_bilingual_token,
    get_public_demo_bundle,
)
from onmyoji_helper.domain.shikigami_repository import ShikigamiRepository
from onmyoji_helper.infra.auth_service import check_activation, get_auth_csv_url
from onmyoji_helper.infra.env_loader import parse_dotenv_file
from onmyoji_helper.infra.logger_setup import setup_runtime_logger
from onmyoji_helper.infra.paths import DEMO_DEFAULT_FLAG_FILE, has_demo_default_flag
from onmyoji_helper.infra.sheets_sync_store import (
    clear_memory_cache,
    has_cloud_sheets_configured,
    load_all_cached_bundle,
)
from onmyoji_helper.ui.demo_sandbox import DemoSandboxController

ENV_KEYS = (
    "ONMYOJI_AUTH_CSV_URL",
    "ONMYOJI_SHEET_DICT_MAP",
    "ONMYOJI_SHEET_DB_EXPORT",
    "ONMYOJI_SHEET_MULTI_SHORTS",
    "ONMYOJI_SHEET_SHIKIGAMI_TAGS",
    "ONMYOJI_SHEET_VERSION_CTRL",
)


def test_parse_dotenv_file_handles_quotes_exports_and_placeholders(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# Comment line",
                "ONMYOJI_SHEET_DICT_MAP='real_sheet_id_123'",
                'export ONMYOJI_SHEET_DB_EXPORT="real_db_456"',
                "ONMYOJI_SHEET_MULTI_SHORTS=your_placeholder_value",
                "ONMYOJI_AUTH_CSV_URL=",
            ]
        ),
        encoding="utf-8",
    )
    parsed = parse_dotenv_file(str(env_file))
    assert parsed == {
        "ONMYOJI_SHEET_DICT_MAP": "real_sheet_id_123",
        "ONMYOJI_SHEET_DB_EXPORT": "real_db_456",
    }


def test_public_sandbox_fallback_when_dotenv_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ONMYOJI_DISABLE_DOTENV", "1")
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    clear_memory_cache()

    assert get_auth_csv_url() == ""
    assert has_cloud_sheets_configured() is False

    # Must not raise or perform network calls when .env is absent
    check_activation(demo_mode=False)

    systems, dict_map, multi_shorts, tags_db = load_all_cached_bundle(demo_mode=False)
    assert len(systems) >= 2
    assert "SP06" in multi_shorts
    assert dict_map["封陽君"] == "封"
    assert tags_db["閻"]["suggested_skill"] == "S3"


def test_demo_sandbox_controller_all_four_scenarios() -> None:
    systems, dict_map, multi_shorts, tags_db = get_public_demo_bundle()
    repo = ShikigamiRepository(
        tags_db=tags_db,
        full_to_short_map=dict_map,
        multi_char_shorts=multi_shorts,
    )
    compiled_dbs = {
        name: {
            "my_ban": info["my_ban"],
            "db": build_database(info["raw_data"], repo.sorted_multi_shorts),
        }
        for name, info in systems.items()
    }

    ctrl = DemoSandboxController(compiled_dbs=compiled_dbs, repository=repo)

    # Scenario 0: Step through 5 steps -> final step reaches 5 picks & Yellow Mode
    res = ctrl.select_scenario(0)
    assert res.opp_ban == "禁選中"
    for _ in range(4):
        res = ctrl.next_step()
    assert res.scenario_index == 0
    assert len(res.my_draft) == 5
    assert len(res.opp_draft) == 5
    assert res.mode == OperatingMode.YELLOW.value

    # Scenario 1: Stolen [葛] pivots recommendation to [百羽]
    ctrl.select_scenario(1)
    pivot_res = ctrl.next_step()
    assert pivot_res.my_recs[0] == "百羽"

    # Scenario 2: Mirror detection flags [鏡像/內戰] and predicts [SP06]
    ctrl.select_scenario(2)
    mirror_res = ctrl.next_step()
    assert "鏡像/內戰" in mirror_res.status_text
    assert mirror_res.opp_preds[0] == "SP06"

    # Scenario 3: 5v5 combat turns cover Single-Target S3, Enemy AoE S3, Ally AoE S3, and S1 fallback
    t1 = ctrl.select_scenario(3)
    assert "閻" in t1.status_text and "[S3]" in t1.status_text and "玲" in t1.status_text

    t2 = ctrl.next_step()
    assert "神荒" in t2.status_text and "[S3]" in t2.status_text and "敵方全體" in t2.status_text

    t3 = ctrl.next_step()
    assert "追月" in t3.status_text and "[S3]" in t3.status_text and "我方全體" in t3.status_text

    t4 = ctrl.next_step()
    assert "SP06" in t4.status_text and "[S1]" in t4.status_text and "玲" in t4.status_text

    # Interactive mode switches (Green / Yellow / Red) & Playbook selector work offline without game/GPU
    y_res = ctrl.switch_demo_mode("yellow")
    assert y_res.mode == OperatingMode.YELLOW.value
    assert "SP Enma" in y_res.tactical_advice and "Hatsune Miku" in y_res.tactical_advice

    r_res = ctrl.switch_demo_mode("red")
    assert r_res.mode == OperatingMode.RED.value
    assert "Skill-First" in r_res.tactical_advice and "SP Susabi" in r_res.tactical_advice

    g_res = ctrl.switch_demo_mode("green")
    assert g_res.mode == OperatingMode.GREEN.value
    assert "Green Mode" in g_res.tactical_advice

    alt_pb = list(compiled_dbs.keys())[1]
    pb_res = ctrl.set_active_playbook(alt_pb)
    assert alt_pb in pb_res.telemetry_lines[0]
    assert format_bilingual_token("閻") == "閻 (SP Enma)"


def test_version_consistency_and_cross_platform_demo_smoke() -> None:
    """Ensures `__version__ == '1.0.0'` and `--demo` runs without Win32, GPU, OCR, or `.env` packages."""
    root_dir = Path(__file__).resolve().parent.parent
    pyproject_data = tomllib.loads((root_dir / "pyproject.toml").read_text(encoding="utf-8"))
    assert onmyoji_helper.__version__ == "1.0.0"
    assert pyproject_data["project"]["version"] == onmyoji_helper.__version__

    # Simulate a clean Linux / cross-platform machine (blocking pywin32, pygetwindow, YOLO, Torch, OCR, LLM)
    cross_platform_smoke = """
import builtins, os, sys
import cv2
os.environ["ONMYOJI_DISABLE_DOTENV"] = "1"
os.environ["ONMYOJI_SKIP_UAC_FOR_TEST"] = "1"
_real = builtins.__import__
_blocked = {
    "win32gui", "win32con", "win32ui", "win32api", "win32process",
    "pygetwindow", "ultralytics", "torch", "rapidocr_onnxruntime", "llama_cpp",
}
def _guarded(name, g=None, l=None, fl=(), lv=0):
    if name.split(".")[0] in _blocked:
        raise ModuleNotFoundError(f"Simulated Linux/minimal env missing {name}")
    return _real(name, g, l, fl, lv)
builtins.__import__ = _guarded
sys.platform = "linux"
sys.path.insert(0, "src")
from onmyoji_helper.app.bootstrap import run_application
run_application(["--demo", "--smoke-exit-ms", "800"])
"""
    proc = subprocess.run(
        [sys.executable, "-c", cross_platform_smoke],
        cwd=str(root_dir),
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    if "TclError" not in proc.stderr:
        assert proc.returncode == 0, f"Cross-platform --demo smoke failed:\n{proc.stderr}"


def test_hr_demo_narrow_ui_labels_and_widget_constraints() -> None:
    """Verifies HR Demo playbook/status strings stay compact (`<= 14` chars, no 'Sandbox' bloat)."""
    for sys_key in PUBLIC_DEMO_SYSTEMS:
        assert "Sandbox" not in sys_key
        assert len(sys_key) <= 14, f"Demo playbook label too wide for narrow 520px header: {sys_key!r}"
    assert "Offline Demo" in PUBLIC_DEMO_SYSTEMS


def test_force_lite_engine_local_gguf_path_and_warning_log_level(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verifies `FORCE_LITE_ENGINE=1`, `LOCAL_LLM_MODEL_PATH`, and `ONMYOJI_LOG_LEVEL=WARNING`."""
    monkeypatch.setenv("ONMYOJI_DISABLE_DOTENV", "1")
    monkeypatch.setenv("FORCE_LITE_ENGINE", "1")

    assert is_force_lite_engine() is True
    tier_name, can_load, _ = evaluate_hardware_tier()
    assert can_load is False
    assert "FORCE_LITE_ENGINE=1" in tier_name

    engine = TacticalEngine()
    assert engine.llm is None
    assert "FORCE_LITE_ENGINE=1" in engine.device_mode

    # Relative LOCAL_LLM_MODEL_PATH (`data/models/<name>.gguf`) resolution
    monkeypatch.delenv("FORCE_LITE_ENGINE", raising=False)
    fake_gguf = tmp_path / "data" / "models" / "friend_quant.gguf"
    fake_gguf.parent.mkdir(parents=True)
    fake_gguf.write_bytes(b"GGUF_TEST")
    monkeypatch.setattr("onmyoji_helper.decision.llm_engine.get_app_base_dir", lambda: str(tmp_path))
    monkeypatch.setenv("LOCAL_LLM_MODEL_PATH", "data/models/friend_quant.gguf")
    assert resolve_configured_gguf_path("nonexistent.gguf") == str(fake_gguf)

    # Hardened Friend Edition WARNING log-level override
    monkeypatch.setenv("ONMYOJI_LOG_LEVEL", "WARNING")
    log_instance = setup_runtime_logger(log_dir=str(tmp_path / "logs"), install_hooks=False)
    assert log_instance.level == logging.WARNING


def test_demo_default_flag_zero_arg_startup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verifies `data/demo_default.flag` enables zero-CLI-argument Offline Demo startup & skips UAC."""
    assert has_demo_default_flag(str(tmp_path)) is False
    flag_file = tmp_path / "data" / DEMO_DEFAULT_FLAG_FILE
    flag_file.parent.mkdir(parents=True)
    flag_file.write_text("OFFLINE_DEMO_DEFAULT=1\n", encoding="utf-8")
    assert has_demo_default_flag(str(tmp_path)) is True

    monkeypatch.setattr(run_app, "_has_demo_default_flag", lambda: True)
    assert run_app._should_skip_uac([]) is True


def test_four_release_profiles_pure_ascii_and_dotenv_rewriting(tmp_path: Path) -> None:
    """Verifies pure-ASCII `build_release.py`/`.bat`, 4 profile specs, GGUF filter, and `.env` isolation."""
    root_dir = Path(__file__).resolve().parent.parent
    py_text = (root_dir / "build_release.py").read_text(encoding="utf-8")
    bat_text = (root_dir / "build_release.bat").read_text(encoding="utf-8")
    assert py_text.isascii(), "build_release.py must be 100% pure ASCII to prevent Windows CMD garbling!"
    assert bat_text.isascii(), "build_release.bat must be 100% pure ASCII to prevent Windows CMD garbling!"
    assert "chcp 65001" in bat_text

    # 1. Verify all 4 profile specs
    assert set(build_release.PROFILES.keys()) == {"personal", "hr-demo", "friend-full", "friend-lite"}

    p_personal = build_release.resolve_profile_spec(build_release.parse_args(["--profile", "personal"]), is_tty=False)
    assert p_personal.output_folder_name == "OnmyojiHelper_Personal"
    assert p_personal.include_local_env is True and p_personal.write_demo_default_flag is False

    p_hr = build_release.resolve_profile_spec(build_release.parse_args(["--profile", "hr-demo"]), is_tty=False)
    assert p_hr.output_folder_name == "OnmyojiHelper_HR_Demo"
    assert p_hr.include_local_env is False
    assert p_hr.write_demo_default_flag is True
    assert p_hr.create_win64_zip is True and p_hr.zip_filename == "OnmyojiHelper_HR_Demo_Win64.zip"
    assert p_hr.require_uac_admin is False

    p_ffull = build_release.resolve_profile_spec(build_release.parse_args(["--profile", "friend-full"]), is_tty=False)
    assert p_ffull.output_folder_name == "OnmyojiHelper_Friend_Full"
    assert p_ffull.include_local_env is True and p_ffull.include_local_gguf is True
    assert p_ffull.default_log_level == "WARNING"

    p_flite = build_release.resolve_profile_spec(build_release.parse_args(["--profile", "friend-lite"]), is_tty=False)
    assert p_flite.output_folder_name == "OnmyojiHelper_Friend_Lite"
    assert p_flite.include_local_env is True and p_flite.exclude_llama_cpp is True
    assert p_flite.force_lite_engine is True and p_flite.default_log_level == "WARNING"

    # 2. Verify GGUF discovery ignores `mmproj*` vision projectors and picks chat GGUF
    fake_lm = tmp_path / "lm_studio"
    fake_lm.mkdir()
    (fake_lm / "mmproj-vision-f16.gguf").write_bytes(b"x" * 64)
    chat_gguf = fake_lm / "gemma-4-E2B-Q4_K_M.gguf"
    chat_gguf.write_bytes(b"y" * 128)
    empty_proj = tmp_path / "proj"
    empty_proj.mkdir()
    assert build_release.discover_local_gguf_model(str(empty_proj), lm_studio_dirs=[str(fake_lm)]) == str(
        chat_gguf.resolve()
    )

    # 3. Verify `rewrite_packaged_dotenv` never mutates source `.env` while injecting profile overrides
    src_env = tmp_path / ".env"
    orig_content = "ONMYOJI_AUTH_CSV_URL=mock_whitelist_endpoint\nONMYOJI_SHEET_DICT_MAP=sheet123\n"
    src_env.write_text(orig_content, encoding="utf-8")

    dst_ffull_env = tmp_path / "dist_ffull" / "data" / ".env"
    build_release.rewrite_packaged_dotenv(
        str(src_env),
        str(dst_ffull_env),
        updates={"LOCAL_LLM_MODEL_PATH": "data/models/gemma.gguf", "ONMYOJI_LOG_LEVEL": "WARNING"},
        remove_keys={"FORCE_LITE_ENGINE"},
    )
    assert src_env.read_text(encoding="utf-8") == orig_content
    parsed_ffull = parse_dotenv_file(str(dst_ffull_env))
    assert parsed_ffull["ONMYOJI_AUTH_CSV_URL"] == "mock_whitelist_endpoint"
    assert parsed_ffull["LOCAL_LLM_MODEL_PATH"] == "data/models/gemma.gguf"
    assert parsed_ffull["ONMYOJI_LOG_LEVEL"] == "WARNING"

    dst_flite_env = tmp_path / "dist_flite" / "data" / ".env"
    build_release.rewrite_packaged_dotenv(
        str(dst_ffull_env),
        str(dst_flite_env),
        updates={"FORCE_LITE_ENGINE": "1", "ONMYOJI_LOG_LEVEL": "WARNING"},
        remove_keys={"LOCAL_LLM_MODEL_PATH"},
    )
    parsed_flite = parse_dotenv_file(str(dst_flite_env))
    assert parsed_flite["FORCE_LITE_ENGINE"] == "1"
    assert "LOCAL_LLM_MODEL_PATH" not in parsed_flite
