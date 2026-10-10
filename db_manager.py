# -*- coding: utf-8 -*-
"""Database Manager Module for Onmyoji Tactical Helper.

This module manages remote database synchronization from Google Sheets, XOR payload encryption/decryption,
AppData local binary caching, version control checks, and shikigami tag database parsing.
"""

import csv
import io
import json
import os
import socket
import threading
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

# ==========================================
# ⚙️ Google Sheets SHEET_IDs Configurations
# ==========================================
ID_DICT_MAP = os.getenv("ID_DICT_MAP", "YOUR_DICT_MAP_SHEET_ID")
ID_DB_EXPORT = os.getenv("ID_DB_EXPORT", "YOUR_DB_EXPORT_SHEET_ID")
ID_MULTI_SHORTS = os.getenv("ID_MULTI_SHORTS", "YOUR_MULTI_SHORTS_SHEET_ID")
ID_SHIKIGAMI_TAGS = os.getenv("ID_SHIKIGAMI_TAGS", "YOUR_SHIKIGAMI_TAGS_SHEET_ID")
ID_VERSION_CTRL = os.getenv("ID_VERSION_CTRL", "YOUR_VERSION_CTRL_SHEET_ID")

ENCRYPT_KEY = int(os.getenv("ENCRYPT_KEY", "0x57"), 16)
# ==========================================


def _xor_encrypt_decrypt(data_bytes: bytes) -> bytes:
    """Applies simple XOR cipher encryption/decryption to input byte data.

    Args:
        data_bytes: Byte sequence to encrypt or decrypt.

    Returns:
        XOR transformed byte sequence.
    """
    return bytes([b ^ ENCRYPT_KEY for b in data_bytes])


def get_appdata_file_path() -> str:
    """Retrieves the absolute path for the encrypted local cache file in AppData.

    Returns:
        Absolute file system path string to 'cache_db.enc'.
    """
    appdata = os.getenv('APPDATA', '')
    target_dir = os.path.join(appdata, "OnmyojiHelper")
    os.makedirs(target_dir, exist_ok=True)
    return os.path.join(target_dir, "cache_db.enc")


def fetch_csv_from_google(url: str, timeout: float = 5.0) -> List[List[str]]:
    """Downloads CSV data from a specified Google Sheets URL.

    Args:
        url: Public export CSV URL string.
        timeout: HTTP request timeout duration in seconds.

    Returns:
        List of rows, where each row is a list of cell string values.
    """
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        content = response.read().decode('utf-8')
    reader = csv.reader(io.StringIO(content))
    return list(reader)


def fetch_remote_version() -> Optional[str]:
    """Retrieves the remote database version identifier from Google Sheets.

    Returns:
        Version string if successfully retrieved, None otherwise.
    """
    try:
        rows = fetch_csv_from_google(URL_VERSION_CTRL, timeout=3.0)
        if rows and len(rows) > 0 and len(rows[0]) > 0:
            return rows[0][0].strip().lstrip("'")
    except Exception as e:
        print(f"⚠️ Failed to fetch remote database version (skipping check): {e}")
    return None


def _parse_list_or_str(val: Any) -> List[str]:
    """Parses list formatting from raw text, supporting JSON arrays or delimiter sequences.

    Args:
        val: Input value string or raw object.

    Returns:
        Parsed list of string elements.
    """
    if not val:
        return []
    val_str = str(val).strip()
    if val_str.startswith("[") and val_str.endswith("]"):
        try:
            parsed = json.loads(val_str)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed]
        except Exception:
            pass
    return [x.strip() for x in val_str.split(">") if x.strip()]


def download_and_encrypt_all(
    remote_version: str = "0.0.1",
    timeout: float = 6.0
) -> Tuple[Dict[str, Any], Dict[str, str], List[str]]:
    """Downloads fresh database tables from Google Sheets, encrypts the payload, and saves to AppData cache.

    Args:
        remote_version: Database version string.
        timeout: HTTP request timeout duration in seconds.

    Returns:
        Tuple containing (restored_systems, dict_map, multi_shorts).
    """
    print("🌐 正在從 Google Sheets 下載最新資料庫...")

    # 1. Download DictMap
    rows_dict = fetch_csv_from_google(URL_DICT_MAP, timeout=timeout)[1:]
    dict_map: Dict[str, str] = {}
    for row in rows_dict:
        if len(row) >= 2:
            full_name = row[0].strip().lstrip("'")
            short_name = row[1].strip().lstrip("'")
            if full_name:
                dict_map[full_name] = short_name

    # 2. Download MultiShorts
    rows_multi = fetch_csv_from_google(URL_MULTI_SHORTS, timeout=timeout)[1:]
    multi_shorts: List[str] = []
    for row in rows_multi:
        if len(row) >= 1:
            item = row[0].strip().lstrip("'")
            if item:
                multi_shorts.append(item)

    # 3. Download DB_Export
    rows_db = fetch_csv_from_google(URL_DB_EXPORT, timeout=timeout)[1:]
    temp_systems: Dict[str, Dict[str, Any]] = {}
    for row in rows_db:
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
        if opp_ban not in opp_bans_dict:
            opp_bans_dict[opp_ban] = []

        line_str = f"{my_draft} {opp_draft}".strip() if opp_draft else my_draft
        opp_bans_dict[opp_ban].append(line_str)

    restored_systems: Dict[str, Any] = {}
    for sys_name, sys_info in temp_systems.items():
        raw_lines = []
        for opp_ban, lines in sys_info["opp_bans"].items():
            raw_lines.append(opp_ban)
            raw_lines.extend(lines)
        restored_systems[sys_name] = {
            "my_ban": sys_info["my_ban"],
            "raw_data": "\n".join(raw_lines)
        }

    # 4. Download Shikigami Tags
    rows_tags = fetch_csv_from_google(URL_SHIKIGAMI_TAGS, timeout=timeout)
    tags_db: Dict[str, Any] = {}
    if len(rows_tags) > 1:
        header = [h.strip().lstrip("'") for h in rows_tags[0]]
        for row in rows_tags[1:]:
            if not row:
                continue
            row_dict = {header[i]: row[i].strip() for i in range(min(len(header), len(row)))}
            raw_name = row_dict.get("Shikigami_Name", "").strip()
            clean_name = raw_name.lstrip("'")
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
                except Exception:
                    pass

            skill_details_raw = row_dict.get("Skill_Details", "")
            skill_details_parsed: Dict[str, Any] = {}
            if skill_details_raw.startswith("{") and skill_details_raw.endswith("}"):
                try:
                    skill_details_parsed = json.loads(skill_details_raw)
                except Exception:
                    pass

            for s_key in ["S1", "S2", "S3", "S4"]:
                if s_key not in skill_details_parsed:
                    desc = row_dict.get(f"{s_key}_Desc", "")
                    scope = row_dict.get(f"{s_key}_Scope", "")
                    target_pref = _parse_list_or_str(row_dict.get(f"{s_key}_Target_Pref", ""))
                    if desc or scope or target_pref:
                        skill_details_parsed[s_key] = {
                            "desc": desc,
                            "scope": scope,
                            "target_role_pref": target_pref
                        }

            tags_db[clean_name] = {
                "role": row_dict.get("Role_Class", "").strip(),
                "tag": row_dict.get("Role_Class", "").strip(),
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
                    "ally_buff_pref": ally_pref
                },
                "skill_details": skill_details_parsed
            }

    cache_data = {
        "VERSION": remote_version,
        "FULL_TO_SHORT_MAP": dict_map,
        "MULTI_CHAR_SHORTS": multi_shorts,
        "SYSTEMS": restored_systems,
        "SHIKIGAMI_TAGS": tags_db
    }

    json_str = json.dumps(cache_data, ensure_ascii=False)
    encrypted_bytes = _xor_encrypt_decrypt(json_str.encode('utf-8'))

    enc_path = get_appdata_file_path()
    with open(enc_path, "wb") as f:
        f.write(encrypted_bytes)

    print(f"🔒 [v{remote_version}] 最新資料 (含完整增量式神標籤) 已成功加密寫入本地快取！")
    return restored_systems, dict_map, multi_shorts


def load_cached_data() -> Tuple[Dict[str, Any], Dict[str, str], List[str]]:
    """Loads database payload from AppData encrypted cache or triggers remote download if outdated.

    Returns:
        Tuple containing (restored_systems, dict_map, multi_shorts).
    """
    enc_path = get_appdata_file_path()
    local_ver: Optional[str] = None
    cached_payload: Optional[Dict[str, Any]] = None

    if os.path.exists(enc_path):
        try:
            with open(enc_path, "rb") as f:
                encrypted_bytes = f.read()
            decrypted_str = _xor_encrypt_decrypt(encrypted_bytes).decode('utf-8')
            cached_payload = json.loads(decrypted_str)
            local_ver = cached_payload.get("VERSION", "0.0.0")
        except Exception as e:
            print(f"⚠️ Local cache corrupted, preparing re-download: {e}")
            local_ver = None

    remote_ver = fetch_remote_version()

    if remote_ver and remote_ver != local_ver:
        print(
            f"🔄 Remote version update detected [{remote_ver}] "
            f"(Local: {local_ver if local_ver else 'None'}), updating..."
        )
        try:
            return download_and_encrypt_all(remote_version=remote_ver, timeout=10.0)
        except Exception as e:
            print(f"⚠️ Update failed, attempting fallback to local cached data: {e}")

    if cached_payload:
        print(f"📁 Successfully loaded local encrypted cache from AppData (Current Version: v{local_ver})!")
        return cached_payload["SYSTEMS"], cached_payload["FULL_TO_SHORT_MAP"], cached_payload["MULTI_CHAR_SHORTS"]

    return download_and_encrypt_all(remote_version=remote_ver or "0.0.1", timeout=6.0)


def load_cached_tags() -> Dict[str, Any]:
    """Retrieves the complete shikigami tags database from the AppData encrypted cache file.

    Returns:
        Dictionary mapping shikigami names to tag and mechanism metadata.
    """
    enc_path = get_appdata_file_path()
    if os.path.exists(enc_path):
        try:
            with open(enc_path, "rb") as f:
                encrypted_bytes = f.read()
            decrypted_str = _xor_encrypt_decrypt(encrypted_bytes).decode('utf-8')
            cached_payload = json.loads(decrypted_str)
            return cached_payload.get("SHIKIGAMI_TAGS", {})
        except Exception as e:
            print(f"⚠️ Failed to read shikigami tags from local cache: {e}")
    return {}