"""Shikigami Knowledge Repository for Onmyoji Tactical Helper.

Consolidates full-to-short name normalization, multi-character token ordering, and
cached mechanism/tag lookups for all 251+ shikigami, eliminating duplicate O(N log N)
key sorting across `battle_context` and `target_selector`.
"""

from functools import lru_cache
from typing import Any

from onmyoji_helper.domain.models import UNKNOWN_SHIKIGAMI

DEFAULT_TAG_METADATA: dict[str, Any] = {
    "priority": 99,
    "static_priority": 99,
    "tag": "一般單位",
    "role": "一般單位",
    "threat_mechanism": "",
    "counter_weakness": "",
    "static_action_tip": "",
    "suggested_skill": "S1",
    "priority_condition": "",
    "recommended_operation": "",
    "priority_target_roles": {"enemy_attack_pref": [], "ally_buff_pref": []},
    "skill_details": {},
}


class ShikigamiRepository:
    """In-memory indexed repository for shikigami aliases and combat mechanism tags."""

    def __init__(
        self,
        tags_db: dict[str, dict[str, Any]] | None = None,
        full_to_short_map: dict[str, str] | None = None,
        multi_char_shorts: list[str] | None = None,
    ) -> None:
        """Initializes the repository and pre-sorts lookup keys by descending length.

        Args:
            tags_db: Mapping from shikigami name to combat role, priority, and skill metadata.
            full_to_short_map: Mapping from full OCR name to canonical shorthand.
            multi_char_shorts: List of multi-character shorthand tokens (e.g. 'SP06', '酒吞').
        """
        self._tags_db: dict[str, dict[str, Any]] = dict(tags_db or {})
        self._sorted_tag_keys: tuple[str, ...] = tuple(
            sorted(self._tags_db.keys(), key=len, reverse=True)
        )

        self.full_to_short_map: dict[str, str] = dict(full_to_short_map or {})
        self.sorted_full_to_short: list[tuple[str, str]] = sorted(
            self.full_to_short_map.items(), key=lambda item: len(item[0]), reverse=True
        )
        self.sorted_multi_shorts: tuple[str, ...] = tuple(
            sorted(set(multi_char_shorts or []), key=len, reverse=True)
        )

        # Bound per-instance LRU lookup cache
        self._cached_lookup = lru_cache(maxsize=512)(self._lookup_uncached)

    @property
    def tags_db(self) -> dict[str, dict[str, Any]]:
        """Returns the underlying raw tags dictionary."""
        return self._tags_db

    def update_tags(self, tags_db: dict[str, dict[str, Any]]) -> None:
        """Updates the tag database and invalidates the internal LRU lookup cache."""
        self._tags_db = dict(tags_db or {})
        self._sorted_tag_keys = tuple(sorted(self._tags_db.keys(), key=len, reverse=True))
        self._cached_lookup.cache_clear()

    def _lookup_uncached(self, clean_name: str) -> dict[str, Any]:
        """Resolves raw tag dictionary entry via exact match followed by length-ordered match."""
        matched_data: dict[str, Any] | None = None
        if clean_name in self._tags_db:
            matched_data = self._tags_db[clean_name]
        else:
            for key in self._sorted_tag_keys:
                if key == clean_name:
                    matched_data = self._tags_db[key]
                    break
                if len(key) >= 2 and (key in clean_name or clean_name in key):
                    matched_data = self._tags_db[key]
                    break

        if not isinstance(matched_data, dict):
            return dict(DEFAULT_TAG_METADATA)

        prio_raw = matched_data.get("priority", matched_data.get("static_priority", 5))
        prio = int(prio_raw) if str(prio_raw).isdigit() else 5
        role_tag = str(matched_data.get("role", matched_data.get("tag", "一般單位")) or "一般單位")

        prio_roles = matched_data.get("priority_target_roles", {})
        if not isinstance(prio_roles, dict):
            prio_roles = {"enemy_attack_pref": [], "ally_buff_pref": []}

        return {
            "priority": prio,
            "static_priority": prio,
            "tag": role_tag,
            "role": role_tag,
            "static_action_tip": matched_data.get("static_action_tip", ""),
            "suggested_skill": matched_data.get("suggested_skill", "S1") or "S1",
            "threat_mechanism": matched_data.get("threat_mechanism", ""),
            "counter_weakness": matched_data.get("counter_weakness", ""),
            "priority_condition": matched_data.get("priority_condition", ""),
            "recommended_operation": matched_data.get("recommended_operation", ""),
            "priority_target_roles": prio_roles,
            "skill_details": matched_data.get("skill_details", {}),
        }

    def get_tag_info(self, name: str | None) -> dict[str, Any]:
        """Retrieves normalized combat metadata for a shikigami in O(1) amortized time.

        Args:
            name: Shikigami full or short name string.

        Returns:
            Normalized metadata dictionary with guaranteed schema keys.
        """
        if (
            not name
            or name == UNKNOWN_SHIKIGAMI
            or "未知" in name
            or "敵方" in name
            or "我方" in name
        ):
            return dict(DEFAULT_TAG_METADATA)
        return self._cached_lookup(name.strip())
