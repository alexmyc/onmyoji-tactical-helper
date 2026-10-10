# -*- coding: utf-8 -*-
"""Target Selection Engine Module for Onmyoji Tactical Helper.

This module implements hierarchical target selection algorithms for skills based on active
mechanism scopes, skill availability filters, dynamic team survival status, and static character
priority rankings.
"""

from typing import Any, Dict, List, Optional, Tuple


class TargetSelector:
    """Dynamic target selection engine implementing domain-specific targeting rules."""

    def __init__(self) -> None:
        """Initializes the TargetSelector instance."""
        pass

    def select_target(
        self,
        active_shiki_name: str,
        active_slot: str,
        skill_states: Dict[str, bool],
        my_status: Dict[str, Dict[str, Any]],
        opp_status: Dict[str, Dict[str, Any]],
        tags_db: Dict[str, Any],
        is_my_turn: bool = True
    ) -> Dict[str, Any]:
        """Primary target selection entry point.

        Determines optimal skill selection and target positioning based on game state,
        skill availability, scope mechanism, and priority hierarchy.

        Args:
            active_shiki_name: Name of the active character.
            active_slot: Slot key of the active character (e.g., "Slot 1").
            skill_states: Dictionary mapping skill keys ('s1'-'s4') to availability booleans.
            my_status: Status dictionary of player side units.
            opp_status: Status dictionary of opponent side units.
            tags_db: Reference dictionary containing character metadata and roles.
            is_my_turn: Boolean flag indicating if it is currently the player's turn.

        Returns:
            Dictionary containing target slot, target name, role, chosen skill, scope, and side.
        """
        # 1. Opponent Turn: Default to identifying high-threat opponent units
        if not is_my_turn:
            best_slot, best_name, best_role = self._filter_and_sort_target(
                opp_status, ["控制", "輸出", "拉條", "供火", "輔助"], tags_db, is_enemy=True
            )
            return {
                "target_slot": best_slot,
                "target_name": best_name,
                "target_role": best_role,
                "chosen_skill": "S1",
                "scope": "敵方單體",
                "target_is_enemy": True
            }

        # 2. Player Turn: Retrieve metadata for the active character
        shiki_info = self._get_db_info(active_shiki_name, tags_db)
        prio_roles = shiki_info.get("priority_target_roles", {})
        skills_detail = shiki_info.get("skill_details", {})
        db_suggested = shiki_info.get("suggested_skill", "S3")

        # 3. Determine the optimal active skill
        chosen_skill = self._determine_best_skill(
            skill_states=skill_states,
            skills_detail=skills_detail,
            db_suggested_skill=db_suggested
        )

        skill_info = skills_detail.get(chosen_skill, {}) or skills_detail.get(chosen_skill.lower(), {})
        scope = str(skill_info.get("scope", "")).strip()

        # Handle S4 missing scope default
        if chosen_skill == "S4" and (not scope or scope.lower() in ["", "none", "null"]):
            scope = "自己"
        elif not scope:
            scope = "敵方單體"

        # 4. Scope-based targeting routing

        # Scope A: AoE Skills (敵方全體 / 我方全體) -> No single target lock ring required
        if "全體" in scope:
            target_is_enemy = "敵方" in scope
            return {
                "target_slot": None,
                "target_name": "敵方全體" if target_is_enemy else "我方全體",
                "target_role": "全體",
                "chosen_skill": chosen_skill,
                "scope": scope,
                "target_is_enemy": target_is_enemy
            }

        # Scope B: Self-targeted skills (姿態切換 / 自身 Buff / 臨時第四技能) -> Lock active unit
        if scope == "自己":
            my_info = my_status.get(active_slot, {})
            self_name = my_info.get("name", active_shiki_name)
            return {
                "target_slot": active_slot,
                "target_name": self_name,
                "target_role": "自己",
                "chosen_skill": chosen_skill,
                "scope": scope,
                "target_is_enemy": False
            }

        # Scope C: Ally Single Target (護盾 / 增益 / 拉條) -> Exclude active self
        if "我方" in scope or scope == "我方單體":
            target_role_pref = skill_info.get("target_role_pref", [])
            if not target_role_pref:
                target_role_pref = prio_roles.get(
                    "ally_buff_pref", ["輸出", "控制", "拉條", "輔助", "供火"]
                )

            best_slot, best_name, best_role = self._filter_and_sort_target(
                my_status, target_role_pref, tags_db, is_enemy=False, exclude_slot=active_slot
            )
            return {
                "target_slot": best_slot,
                "target_name": best_name,
                "target_role": best_role,
                "chosen_skill": chosen_skill,
                "scope": scope,
                "target_is_enemy": False
            }

        # Scope D: Enemy Single Target (Single-target attack / CC skills)
        target_role_pref = skill_info.get("target_role_pref", [])
        if not target_role_pref:
            target_role_pref = prio_roles.get(
                "enemy_attack_pref", ["輸出", "控制", "拉條", "供火", "輔助"]
            )

        best_slot, best_name, best_role = self._filter_and_sort_target(
            opp_status, target_role_pref, tags_db, is_enemy=True
        )
        return {
            "target_slot": best_slot,
            "target_name": best_name,
            "target_role": best_role,
            "chosen_skill": chosen_skill,
            "scope": scope,
            "target_is_enemy": True
        }

    def _determine_best_skill(
        self,
        skill_states: Dict[str, bool],
        skills_detail: Dict[str, Any],
        db_suggested_skill: str = "S3"
    ) -> str:
        """Determines best skill selection prioritizing DB recommendations and fallback rules.

        Args:
            skill_states: Dictionary mapping skill keys to UI availability booleans.
            skills_detail: Dictionary containing metadata for character skills.
            db_suggested_skill: DB recommended priority skill (defaults to "S3").

        Returns:
            Selected skill key string (e.g., "S3", "S2", "S4", or "S1").
        """
        valid_skills = {}
        for skill_key in ["S1", "S2", "S3", "S4"]:
            info = skills_detail.get(skill_key, {}) or skills_detail.get(skill_key.lower(), {})
            scope = str(info.get("scope", "")).strip()
            state_key = skill_key.lower()

            if skill_key == "S4" and (not scope or scope.lower() in ["", "none", "null"]):
                scope = "自己"

            is_passive_or_none = any(kw in scope for kw in ["被動", "無"])

            if skill_key == "S1" and (is_passive_or_none or not scope):
                is_passive_or_none = False

            is_valid = bool(skill_states.get(state_key)) and not is_passive_or_none
            valid_skills[skill_key] = is_valid

        # Highest Priority: Use DB suggested skill if currently valid and available
        db_suggested = str(db_suggested_skill).upper() if db_suggested_skill else "S3"
        if valid_skills.get(db_suggested, False):
            return db_suggested

        # Fallback Priority Order: S3 -> S2 -> S4 -> S1
        fallback_priority = ["S3", "S2", "S4", "S1"]
        for sk in fallback_priority:
            if valid_skills.get(sk, False):
                return sk

        return "S1"

    def _get_db_info(self, name: str, tags_db: Dict[str, Any]) -> Dict[str, Any]:
        """Safely retrieves character metadata from the tags database.

        Args:
            name: Character name string.
            tags_db: Tags database mapping dictionary.

        Returns:
            Character metadata dictionary, or empty dictionary if not found.
        """
        if not name or "未知" in name:
            return {}
        for k, v in tags_db.items():
            if k in name or name in k:
                return v if isinstance(v, dict) else {}
        return {}

    def _filter_and_sort_target(
        self,
        status_dict: Dict[str, Dict[str, Any]],
        role_pref_list: Optional[List[str]],
        tags_db: Dict[str, Any],
        is_enemy: bool = True,
        exclude_slot: Optional[str] = None
    ) -> Tuple[Optional[str], str, str]:
        """Hierarchical filtering and fallback sorting algorithm for target selection.

        Tier 1: Filter active units matching preferred roles in priority order.
        Tier 2: If multiple units match a role, select lowest Static_Priority rank.
        Tier 3 (Fallback): Select alive unit with lowest Static_Priority across entire board.

        Args:
            status_dict: Board status dictionary for ally/enemy team.
            role_pref_list: List of preferred role strings in priority order.
            tags_db: Database dictionary containing character tags and priorities.
            is_enemy: Boolean indicating if filtering enemy targets.
            exclude_slot: Optional slot key string to exclude from candidates.

        Returns:
            Tuple of (target_slot, target_name, target_role).
        """
        alive_units = []
        for slot, info in status_dict.items():
            if exclude_slot and slot == exclude_slot:
                continue

            name = info.get("name", "")
            is_alive = info.get("alive", False)

            if is_alive and name and ("敵方" not in name if is_enemy else "我方" not in name):
                t_info = self._get_db_info(name, tags_db)
                role = t_info.get("role", t_info.get("tag", "一般單位"))
                prio_str = t_info.get("static_priority", t_info.get("priority", 5))
                prio = int(prio_str) if str(prio_str).isdigit() else 5

                alive_units.append({
                    "slot": slot,
                    "name": name,
                    "role": role,
                    "priority": prio
                })

        if not alive_units:
            if exclude_slot and exclude_slot in status_dict:
                my_name = status_dict[exclude_slot].get("name", "我方單位")
                return exclude_slot, my_name, "一般單位"
            default_slot = "Slot 1"
            default_name = "敵方單位" if is_enemy else "我方單位"
            return default_slot, default_name, "一般單位"

        # Tier 1 & Tier 2: Role preference matching and static priority sorting
        if role_pref_list and isinstance(role_pref_list, list):
            for pref_role in role_pref_list:
                matched = [u for u in alive_units if u["role"] == pref_role]
                if matched:
                    matched.sort(key=lambda x: x["priority"])
                    best = matched[0]
                    return best["slot"], best["name"], best["role"]

        # Tier 3 Fallback: Global minimum static priority selection
        alive_units.sort(key=lambda x: x["priority"])
        best = alive_units[0]
        return best["slot"], best["name"], best["role"]