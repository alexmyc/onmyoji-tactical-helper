"""Unit Tests for `ShikigamiRepository` Substring Boundaries & `ActionBarDetector` Template Matching."""

import cv2
import numpy as np

from onmyoji_helper.domain.models import UNKNOWN_SHIKIGAMI
from onmyoji_helper.domain.shikigami_repository import ShikigamiRepository
from onmyoji_helper.infra.sheets_sync_store import _parse_shikigami_tags_table, xor_transform_bytes
from onmyoji_helper.vision.action_bar_detector import ActionBarDetector


def test_shikigami_repository_prevents_short_token_false_matches() -> None:
    tags_db = {
        "神荒": {"role": "輸出", "priority": 1, "suggested_skill": "S3"},
        "不知火": {"role": "輔助", "priority": 2, "suggested_skill": "S3"},
        "荒骷髏": {"role": "輸出", "priority": 3, "suggested_skill": "S2"},
        "SP荒骷髏": {"role": "控制", "priority": 1, "suggested_skill": "S3"},
        "追月": {"role": "供火", "priority": 4, "suggested_skill": "S3"},
        "SP06": {"role": "輸出", "priority": 1, "suggested_skill": "S3"},
    }
    full_to_short = {
        "天照": "神",
        "神啟荒": "神荒",
        "流光追月神": "追月",
    }
    repo = ShikigamiRepository(tags_db=tags_db, full_to_short_map=full_to_short)

    # 1. Single-char queries ('神', '荒', '火') MUST NOT falsely inherit '神荒' or '不知火'
    assert repo.get_tag_info("神")["priority"] == 99
    assert repo.get_tag_info("荒")["priority"] == 99
    assert repo.get_tag_info("火")["priority"] == 99

    # 2. Bare rarity prefix 'SP' MUST NOT match 'SP06' or 'SP荒骷髏'
    assert repo.get_tag_info("SP")["priority"] == 99

    # 3. Exact alias resolution ('神啟荒' -> '神荒') and bounded root match ('流光追月神' -> '追月') succeed
    assert repo.get_tag_info("神啟荒")["role"] == "輸出"
    assert repo.get_tag_info("流光追月神")["role"] == "供火"

    # 4. Longest specific key ('SP荒骷髏') wins over shorter substring ('荒骷髏')
    sp_res = repo.get_tag_info("SP荒骷髏")
    base_res = repo.get_tag_info("荒骷髏")
    assert sp_res["role"] == "控制" and sp_res["priority"] == 1
    assert base_res["role"] == "輸出" and base_res["priority"] == 3

    # 5. Short 2-char root does NOT falsely match an unrelated long phrase (< 33% overlap)
    assert repo.get_tag_info("追月測試無關超長標題文字框")["priority"] == 99

    # 6. Dynamic update_tags clears LRU cache cleanly
    repo.update_tags({**tags_db, "神": {"role": "輸出", "priority": 1, "suggested_skill": "S3"}})
    assert repo.get_tag_info("神")["priority"] == 1


def _paint_distinct_avatar_tile(canvas: np.ndarray, x: int, y: int, w: int, h: int, idx: int) -> None:
    """Draws a unique, high-contrast geometric pattern per slot for deterministic template matching."""
    tile = np.full((h, w, 3), 30 + idx * 15, dtype=np.uint8)
    if idx == 0:
        cv2.circle(tile, (w // 2, h // 2), min(w, h) // 3, (240, 80, 40), -1)
    elif idx == 1:
        cv2.rectangle(tile, (6, 6), (w - 6, h - 6), (40, 230, 90), 4)
        cv2.line(tile, (0, 0), (w, h), (250, 250, 250), 3)
    elif idx == 2:
        pts = np.array([[w // 2, 5], [5, h - 5], [w - 5, h - 5]], dtype=np.int32)
        cv2.fillPoly(tile, [pts], (60, 120, 245))
    elif idx == 3:
        cv2.line(tile, (0, h // 2), (w, h // 2), (255, 220, 40), 6)
        cv2.line(tile, (w // 2, 0), (w // 2, h), (255, 60, 200), 6)
    else:
        for step in range(4, w, 12):
            cv2.line(tile, (step, 0), (step, h), (240, 240, 120), 3)
    canvas[y : y + h, x : x + w] = tile


def test_action_bar_detector_prep_capture_and_active_turn_matching() -> None:
    detector = ActionBarDetector()
    w, h = 1280, 720
    picks = ["葛", "帝", "骷", "閻", "連"]

    # 1. Blank/dark loading screen is rejected by contrast/brightness gate
    dark_frame = np.zeros((h, w, 3), dtype=np.uint8)
    assert detector.capture_prep_templates(dark_frame, our_picks=picks) is False
    assert detector.identify_active_turn(dark_frame) == (None, UNKNOWN_SHIKIGAMI, 0.0)

    # 2. Populate 5 distinct prep-bar avatar tiles (i=0 -> Slot 5 ... i=4 -> Slot 1)
    prep_frame = np.full((h, w, 3), 20, dtype=np.uint8)
    box_w, box_h = int(w * detector.pw), int(h * detector.ph)
    y_top, gap = int(h * detector.py), int(w * detector.pgap)
    slot_patterns: dict[str, int] = {}
    for i in range(5):
        slot_name = f"Slot {5 - i}"
        x = int(w * detector.px) + i * gap
        _paint_distinct_avatar_tile(prep_frame, x, y_top, box_w, box_h, i)
        slot_patterns[slot_name] = i

    assert detector.capture_prep_templates(prep_frame, our_picks=picks) is True
    assert len(detector.prep_templates) == 5
    assert detector.slot_to_shiki["Slot 4"] == "閻"

    # 3. Paint Slot 4's avatar ('閻') into the bottom-right active turn indicator box
    turn_frame = np.full((h, w, 3), 20, dtype=np.uint8)
    rx, ry = int(w * detector.rx), int(h * detector.ry)
    rw, rh = int(w * detector.rw), int(h * detector.rh)
    _paint_distinct_avatar_tile(turn_frame, rx, ry, rw, rh, slot_patterns["Slot 4"])

    slot, shiki_name, score = detector.identify_active_turn(turn_frame)
    assert slot == "Slot 4"
    assert shiki_name == "閻"
    assert score >= 0.70

    # 4. Reset clears all templates cleanly
    detector.reset()
    assert len(detector.prep_templates) == 0


def test_xor_cipher_roundtrip_and_tags_csv_parser() -> None:
    payload = "陰陽師鬥技助手_XOR_Roundtrip_2026".encode()
    encrypted = xor_transform_bytes(payload)
    assert encrypted != payload
    assert xor_transform_bytes(encrypted) == payload
    assert xor_transform_bytes(b"" if False else payload[:0]) == payload[:0]

    rows = [
        [
            "Shikigami_Name",
            "Role_Class",
            "Static_Priority",
            "Suggested_Skill",
            "Enemy_Attack_Pref",
            "S3_Desc",
            "S3_Scope",
        ],
        ["大夜摩天閻魔", "控制", "1", "S3", "拉條>控制>輸出", "單體沉默變形", "敵方單體"],
    ]
    parsed = _parse_shikigami_tags_table(rows)
    assert parsed["大夜摩天閻魔"]["priority"] == 1
    assert parsed["大夜摩天閻魔"]["priority_target_roles"]["enemy_attack_pref"] == ["拉條", "控制", "輸出"]
    assert parsed["大夜摩天閻魔"]["skill_details"]["S3"]["scope"] == "敵方單體"
