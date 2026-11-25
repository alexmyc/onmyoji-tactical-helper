"""Google Sheets Cloud Sync & Encrypted AppData Cache Store for Onmyoji Tactical Helper.

Supports dual-track operation:
1. Private Cloud Mode (when `.env` defines `ONMYOJI_SHEET_*` IDs): synchronizes live playbooks
   and 251-shikigami mechanism tags from Google Sheets with vectorized NumPy XOR AppData caching.
2. Public Sandbox Mode (when `.env` is absent or `demo_mode=True`): loads the built-in offline
   sandbox dataset (`public_demo_data.py`) without network calls or hardcoded private IDs.
"""

import csv
import io
import json
import logging
import os
import threading
import urllib.request
from typing import Any

import numpy as np

from onmyoji_helper.domain.public_demo_data import get_public_demo_bundle
from onmyoji_helper.infra.env_loader import get_clean_env
from onmyoji_helper.infra.paths import get_appdata_cache_path

logger = logging.getLogger("OnmyojiHelper")

ENCRYPT_KEY: int = 0x57

_CACHE_LOCK: threading.Lock = threading.Lock()
_MEMORY_PAYLOAD_CACHE: dict[str, Any] | None = None


def _sheet_csv_url(sheet_id: str) -> str:
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv" if sheet_id else ""


def get_configured_sheet_urls() -> dict[str, str]:
    """Resolves Google Sheets export URLs dynamically from `.env` / environment variables."""
    return {
        "dict_map": _sheet_csv_url(get_clean_env("ONMYOJI_SHEET_DICT_MAP")),
        "db_export": _sheet_csv_url(get_clean_env("ONMYOJI_SHEET_DB_EXPORT")),
        "multi_shorts": _sheet_csv_url(get_clean_env("ONMYOJI_SHEET_MULTI_SHORTS")),
        "shikigami_tags": _sheet_csv_url(get_clean_env("ONMYOJI_SHEET_SHIKIGAMI_TAGS")),
        "version_ctrl": _sheet_csv_url(get_clean_env("ONMYOJI_SHEET_VERSION_CTRL")),
    }


def has_cloud_sheets_configured() -> bool:
    """Returns True when private Google Sheets endpoints are configured via `.env`."""
    urls = get_configured_sheet_urls()
    return bool(urls["dict_map"] and urls["db_export"] and urls["multi_shorts"])


def clear_memory_cache() -> None:
    """Clears the in-memory payload cache (useful for isolated unit/dual-track tests)."""
    global _MEMORY_PAYLOAD_CACHE
    with _CACHE_LOCK:
        _MEMORY_PAYLOAD_CACHE = None


def xor_transform_bytes(data_bytes: bytes) -> bytes:
    """Applies vectorized NumPy XOR cipher transformation (~50x faster than Python byte loops)."""
    if not data_bytes:
        return data_bytes
    arr = np.frombuffer(data_bytes, dtype=np.uint8)
    return np.bitwise_xor(arr, np.uint8(ENCRYPT_KEY)).tobytes()


def fetch_csv_from_google(url: str, timeout: float = 5.0) -> list[list[str]]:
    """Downloads CSV rows from a Google Sheets export URL."""
    if not url:
        return []
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            content = response.read().decode("utf-8")
        return list(csv.reader(io.StringIO(content)))
    except Exception as exc:
        logger.warning(f"⚠️ Failed to fetch CSV from remote endpoint: {exc}")
        return []


def fetch_remote_version() -> str | None:
    """Retrieves the remote database version identifier from Google Sheets if configured."""
    url_ver = get_configured_sheet_urls()["version_ctrl"]
    if not url_ver:
        return None
    try:
        rows = fetch_csv_from_google(url_ver, timeout=3.0)
        if rows and rows[0]:
            return rows[0][0].strip().lstrip("'")
    except Exception as exc:
        logger.debug(f"⚠️ Failed to fetch remote database version: {exc}")
    return None


def _parse_list_or_str(val: Any) -> list[str]:
    """Parses role preference sequences formatted as JSON arrays or `>`-delimited strings."""
    if not val:
        return []
    val_str = str(val).strip()
    if val_str.startswith("[") and val_str.endswith("]"):
        try:
            parsed = json.loads(val_str)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception as exc:
            logger.debug(f"[SheetsSync] Role list JSON parse fallback: {exc}")
    return [x.strip() for x in val_str.split(">") if x.strip()]


def _atomic_write_cache(enc_path: str, encrypted_bytes: bytes) -> None:
    """Persists encrypted cache bytes atomically via `.tmp` + `fsync` + `os.replace`."""
    tmp_path = f"{enc_path}.tmp"
    with open(tmp_path, "wb") as f:
        f.write(encrypted_bytes)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, enc_path)


def _read_disk_cache(enc_path: str) -> dict[str, Any] | None:
    """Reads and decrypts `cache_db.enc` from disk if valid."""
    if not os.path.exists(enc_path):
        return None
    try:
        with open(enc_path, "rb") as f:
            encrypted_bytes = f.read()
        decrypted_str = xor_transform_bytes(encrypted_bytes).decode("utf-8")
        payload = json.loads(decrypted_str)
        return payload if isinstance(payload, dict) else None
    except Exception as exc:
        logger.warning(f"⚠️ 本地快取解析異常 ({exc})，準備重新下載。")
        return None


def _parse_shikigami_tags_table(rows_tags: list[list[str]]) -> dict[str, dict[str, Any]]:
    """Parses the shikigami mechanism & skill tag CSV table into normalized dictionaries."""
    tags_db: dict[str, dict[str, Any]] = {}
    if not rows_tags or len(rows_tags) <= 1:
        return tags_db

    header = [h.strip().lstrip("'") for h in rows_tags[0]]
    for row in rows_tags[1:]:
        if not row:
            continue
        row_dict = {header[i]: row[i].strip() for i in range(min(len(header), len(row)))}
        clean_name = row_dict.get("Shikigami_Name", "").strip().lstrip("'")
        if not clean_name:
            continue

        prio_str = row_dict.get("Static_Priority", "")
        prio = int(prio_str) if prio_str.isdigit() else 5

        prio_roles_raw = row_dict.get("Priority_Target_Roles", "")
        enemy_pref = _parse_list_or_str(row_dict.get("Enemy_Attack_Pref", ""))
        ally_pref = _parse_list_or_str(row_dict.get("Ally_Buff_Pref", ""))

        if prio_roles_raw.startswith("{") and prio_roles_raw.endswith("}"):
            try:
                prio_roles_json = json.loads(prio_roles_raw)
                enemy_pref = prio_roles_json.get("enemy_attack_pref", enemy_pref)
                ally_pref = prio_roles_json.get("ally_buff_pref", ally_pref)
            except Exception as exc:
                logger.debug(f"[SheetsSync] Priority_Target_Roles JSON fallback: {exc}")

        skill_details_raw = row_dict.get("Skill_Details", "")
        skill_details_parsed: dict[str, Any] = {}
        if skill_details_raw.startswith("{") and skill_details_raw.endswith("}"):
            try:
                skill_details_parsed = json.loads(skill_details_raw)
            except Exception as exc:
                logger.debug(f"[SheetsSync] Skill_Details JSON fallback: {exc}")

        for s_key in ("S1", "S2", "S3", "S4"):
            if s_key not in skill_details_parsed:
                desc = row_dict.get(f"{s_key}_Desc", "")
                scope = row_dict.get(f"{s_key}_Scope", "")
                target_pref = _parse_list_or_str(row_dict.get(f"{s_key}_Target_Pref", ""))
                if desc or scope or target_pref:
                    skill_details_parsed[s_key] = {
                        "desc": desc,
                        "scope": scope,
                        "target_role_pref": target_pref,
                    }

        role_cls = row_dict.get("Role_Class", "").strip()
        tags_db[clean_name] = {
            "role": role_cls,
            "tag": role_cls,
            "static_priority": prio,
            "priority": prio,
            "static_action_tip": row_dict.get("Static_Action_Tip", "").strip(),
            "suggested_skill": row_dict.get("Suggested_Skill", "").strip(),
            "threat_mechanism": row_dict.get("Threat_Mechanism", "").strip(),
            "counter_weakness": row_dict.get("Counter_Weakness", "").strip(),
            "priority_condition": row_dict.get("Priority_Condition", "").strip(),
            "recommended_operation": row_dict.get("Recommended_Operation", "").strip(),
            "priority_target_roles": {
                "enemy_attack_pref": enemy_pref,
                "ally_buff_pref": ally_pref,
            },
            "skill_details": skill_details_parsed,
        }
    return tags_db


def download_and_encrypt_all(
    remote_version: str = "1.0.0",
    timeout: float = 8.0,
) -> dict[str, Any]:
    """Downloads all 4 Google Sheets tables, encrypts to AppData cache, and updates memory cache."""
    global _MEMORY_PAYLOAD_CACHE

    urls = get_configured_sheet_urls()
    if not (urls["dict_map"] and urls["db_export"] and urls["multi_shorts"]):
        raise ValueError("Private Google Sheets endpoints are not configured in .env.")

    logger.info("🌐 正在從 Google Sheets 同步最新資料庫...")

    rows_dict = fetch_csv_from_google(urls["dict_map"], timeout=timeout)
    if not rows_dict:
        raise ConnectionError("Unable to download DictMap from configured Google Sheet.")
    dict_map: dict[str, str] = {}
    for row in rows_dict[1:]:
        if len(row) >= 2:
            full_name = row[0].strip().lstrip("'")
            short_name = row[1].strip().lstrip("'")
            if full_name:
                dict_map[full_name] = short_name

    rows_multi = fetch_csv_from_google(urls["multi_shorts"], timeout=timeout)
    if not rows_multi:
        raise ConnectionError("Unable to download MultiShorts from configured Google Sheet.")
    multi_shorts: list[str] = [
        row[0].strip().lstrip("'") for row in rows_multi[1:] if row and row[0].strip().lstrip("'")
    ]

    rows_db = fetch_csv_from_google(urls["db_export"], timeout=timeout)
    if not rows_db:
        raise ConnectionError("Unable to download DB_Export from configured Google Sheet.")
    temp_systems: dict[str, dict[str, Any]] = {}
    for row in rows_db[1:]:
        if len(row) < 5:
            continue
        sys_name = row[0].strip()
        my_ban = row[1].strip()
        opp_ban = row[2].strip()
        my_draft = row[3].strip().lstrip("'")
        opp_draft = row[4].strip().lstrip("'")

        if sys_name not in temp_systems:
            temp_systems[sys_name] = {"my_ban": my_ban, "opp_bans": {}}

        opp_bans_dict = temp_systems[sys_name]["opp_bans"]
        opp_bans_dict.setdefault(opp_ban, [])
        line_str = f"{my_draft} {opp_draft}".strip() if opp_draft else my_draft
        opp_bans_dict[opp_ban].append(line_str)

    restored_systems: dict[str, Any] = {}
    for sys_name, sys_info in temp_systems.items():
        raw_lines: list[str] = []
        for opp_ban, lines in sys_info["opp_bans"].items():
            raw_lines.append(opp_ban)
            raw_lines.extend(lines)
        restored_systems[sys_name] = {
            "my_ban": sys_info["my_ban"],
            "raw_data": "\n".join(raw_lines),
        }

    rows_tags = fetch_csv_from_google(urls["shikigami_tags"], timeout=timeout) if urls["shikigami_tags"] else []
    tags_db = _parse_shikigami_tags_table(rows_tags)

    cache_payload: dict[str, Any] = {
        "VERSION": remote_version,
        "FULL_TO_SHORT_MAP": dict_map,
        "MULTI_CHAR_SHORTS": multi_shorts,
        "SYSTEMS": restored_systems,
        "SHIKIGAMI_TAGS": tags_db,
    }

    encrypted_bytes = xor_transform_bytes(json.dumps(cache_payload, ensure_ascii=False).encode("utf-8"))
    enc_path = get_appdata_cache_path()
    _atomic_write_cache(enc_path, encrypted_bytes)

    with _CACHE_LOCK:
        _MEMORY_PAYLOAD_CACHE = cache_payload

    logger.info(f"🔒 [v{remote_version}] 最新資料庫已加密寫入本地快取。")
    return cache_payload


def load_all_cached_bundle(
    demo_mode: bool = False,
) -> tuple[dict[str, Any], dict[str, str], list[str], dict[str, dict[str, Any]]]:
    """Loads the dataset bundle from private `.env` cloud/cache or Public Sandbox fallback."""
    global _MEMORY_PAYLOAD_CACHE

    if demo_mode or not has_cloud_sheets_configured():
        logger.info("🧪 [DataStore] Loading built-in Public Sandbox playbooks & 25+ shikigami tags.")
        return get_public_demo_bundle()

    with _CACHE_LOCK:
        if _MEMORY_PAYLOAD_CACHE is not None and _MEMORY_PAYLOAD_CACHE.get("SYSTEMS"):
            return (
                _MEMORY_PAYLOAD_CACHE["SYSTEMS"],
                _MEMORY_PAYLOAD_CACHE["FULL_TO_SHORT_MAP"],
                _MEMORY_PAYLOAD_CACHE["MULTI_CHAR_SHORTS"],
                _MEMORY_PAYLOAD_CACHE.get("SHIKIGAMI_TAGS", {}),
            )

    enc_path = get_appdata_cache_path()
    cached_payload = _read_disk_cache(enc_path)
    local_ver = cached_payload.get("VERSION", "0.0.0") if cached_payload else None
    remote_ver = fetch_remote_version()

    if remote_ver and remote_ver != local_ver:
        logger.info(f"🔄 偵測到遠端資料庫更新 [{remote_ver}] (本機: {local_ver})，開始同步...")
        try:
            fresh = download_and_encrypt_all(remote_version=remote_ver, timeout=10.0)
            return (
                fresh["SYSTEMS"],
                fresh["FULL_TO_SHORT_MAP"],
                fresh["MULTI_CHAR_SHORTS"],
                fresh.get("SHIKIGAMI_TAGS", {}),
            )
        except Exception as exc:
            logger.warning(f"⚠️ 遠端更新失敗 ({exc})，嘗試載入現有本機快取資料。")

    if cached_payload and cached_payload.get("SYSTEMS"):
        with _CACHE_LOCK:
            _MEMORY_PAYLOAD_CACHE = cached_payload
        return (
            cached_payload["SYSTEMS"],
            cached_payload["FULL_TO_SHORT_MAP"],
            cached_payload["MULTI_CHAR_SHORTS"],
            cached_payload.get("SHIKIGAMI_TAGS", {}),
        )

    fresh = download_and_encrypt_all(remote_version=remote_ver or "1.0.0", timeout=10.0)
    return (
        fresh["SYSTEMS"],
        fresh["FULL_TO_SHORT_MAP"],
        fresh["MULTI_CHAR_SHORTS"],
        fresh.get("SHIKIGAMI_TAGS", {}),
    )


def load_cached_data() -> tuple[dict[str, Any], dict[str, str], list[str]]:
    """Backwards-compatible loader returning `(systems, full_to_short_map, multi_char_shorts)`."""
    systems, dict_map, multi_shorts, _ = load_all_cached_bundle()
    return systems, dict_map, multi_shorts


def load_cached_tags() -> dict[str, dict[str, Any]]:
    """Retrieves shikigami tags from the shared memory/AppData bundle or Public Sandbox."""
    if not has_cloud_sheets_configured():
        return get_public_demo_bundle()[3]

    with _CACHE_LOCK:
        if _MEMORY_PAYLOAD_CACHE is not None and _MEMORY_PAYLOAD_CACHE.get("SHIKIGAMI_TAGS"):
            return _MEMORY_PAYLOAD_CACHE["SHIKIGAMI_TAGS"]

    try:
        _, _, _, tags = load_all_cached_bundle()
        return tags
    except Exception as exc:
        logger.error(f"❌ 載入式神標籤失敗: {exc}")
        return {}
