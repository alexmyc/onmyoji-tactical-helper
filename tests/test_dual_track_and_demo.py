"""Unit Tests for Dual-Track `.env` Isolation, Public Sandbox Mode, and `--demo` Scenarios."""

from pathlib import Path

import pytest

from onmyoji_helper.decision.draft_engine import build_database
from onmyoji_helper.domain.models import OperatingMode
from onmyoji_helper.domain.public_demo_data import get_public_demo_bundle
from onmyoji_helper.domain.shikigami_repository import ShikigamiRepository
from onmyoji_helper.infra.auth_service import check_activation, get_auth_csv_url
from onmyoji_helper.infra.env_loader import parse_dotenv_file
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
