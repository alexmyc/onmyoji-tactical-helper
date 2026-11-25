"""Shikigami Knowledge Repository for Onmyoji Tactical Helper.

Consolidates full-to-short name normalization, multi-character token ordering, and
cached mechanism/tag lookups for all 251+ shikigami, enforcing exact-match priority and
bounded root-containment rules to prevent short-token false positives.
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

_GENERIC_ASCII_PREFIXES: frozenset[str] = frozenset({"sp", "ssr", "sr"})
_MIN_CONTAINMENT_RATIO: float = 0.33


def _is_meaningful_root_match(key: str, query: str) -> bool:
    """Checks whether `key` and `query` share a safe, non-trivial root containment relationship.

    Rules:
    1. Both `key` and `query` must have length >= 2 (single-character names must match via
       exact lookup or explicit `full_to_short_map` alias, never via arbitrary substring).
    2. Pure generic rarity prefixes ('SP', 'SSR', 'SR') are rejected as standalone roots.
    3. Overlap length must be >= 33% of the longer string (`len(shorter) / len(longer) >= 0.33`)
       so 2-character roots match 4..6 char shikigami names ('追月' <-> '流光追月神') without
       contaminating unrelated long phrases.
    """
    if len(key) < 2 or len(query) < 2:
        return False
    shorter, longer = (key, query) if len(key) <= len(query) else (query, key)
    if shorter.lower() in _GENERIC_ASCII_PREFIXES:
        return False
    if shorter.isascii() and len(shorter) < 3:
        return False
    if shorter not in longer:
        return False
    return (len(shorter) / float(len(longer))) >= _MIN_CONTAINMENT_RATIO


class ShikigamiRepository:
    """In-memory indexed repository for shikigami aliases and combat mechanism tags."""

    def __init__(
        self,
        tags_db: dict[str, dict[str, Any]] | None = None,
        full_to_short_map: dict[str, str] | None = None,
        multi_char_shorts: list[str] | None = None,
    ) -> None:
        """Initializes the repository and pre-sorts lookup keys by descending length."""
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
        self._short_to_full_candidates: dict[str, list[str]] = {}
        for full_name, short_name in self.sorted_full_to_short:
            if short_name:
                self._short_to_full_candidates.setdefault(short_name, []).append(full_name)

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

    def _resolve_raw_entry(self, clean_name: str) -> dict[str, Any] | None:
        # 1. Strict exact match
        if clean_name in self._tags_db:
            return self._tags_db[clean_name]

        # 2. Exact alias resolution via full_to_short_map (full -> short and short -> full)
        alias_short = self.full_to_short_map.get(clean_name)
        if alias_short and alias_short in self._tags_db:
            return self._tags_db[alias_short]
        for full_cand in self._short_to_full_candidates.get(clean_name, ()):
            if full_cand in self._tags_db:
                return self._tags_db[full_cand]

        # 3. Prefer prefix/suffix root containment first (longest key wins)
        for key in self._sorted_tag_keys:
            if _is_meaningful_root_match(key, clean_name):
                shorter, longer = (key, clean_name) if len(key) <= len(clean_name) else (clean_name, key)
                if longer.startswith(shorter) or longer.endswith(shorter):
                    return self._tags_db[key]

        # 4. Bounded interior root containment (e.g. '追月' <-> '流光追月神', longest key wins)
        for key in self._sorted_tag_keys:
            if _is_meaningful_root_match(key, clean_name):
                return self._tags_db[key]

        return None

    def _lookup_uncached(self, clean_name: str) -> dict[str, Any]:
        """Resolves raw tag dictionary entry with exact, alias, and bounded-root precedence."""
        matched_data = self._resolve_raw_entry(clean_name)
        if not isinstance(matched_data, dict):
            return dict(DEFAULT_TAG_METADATA)

        prio_raw = matched_data.get("priority", matched_data.get("static_priority", 5))
        prio_str = str(prio_raw)
        prio = int(prio_str) if prio_str.isdigit() else 5
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
        """Retrieves normalized combat metadata for a shikigami in O(1) amortized time."""
        if (
            not name
            or name == UNKNOWN_SHIKIGAMI
            or "未知" in name
            or "敵方" in name
            or "我方" in name
        ):
            return dict(DEFAULT_TAG_METADATA)
        return self._cached_lookup(name.strip())
