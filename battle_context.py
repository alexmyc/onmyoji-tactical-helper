# -*- coding: utf-8 -*-
"""Battle Context Builder Module for Onmyoji Tactical Helper.

This module aggregates real-time battle board state, queries character tags and metadata,
invokes the TargetSelector engine, and constructs rich contextual payloads for LLM inference
and fallback UI displays.
"""

from collections import Counter
from typing import Any, Dict, List, Optional

from db_manager import load_cached_tags
from target_selector import TargetSelector


class BattleContextBuilder:
    """Builds and enriches battle contextual payloads for tactical reasoning."""

    def __init__(self, tag_filename: Optional[str] = None) -> None:
        """Initializes the context builder, loads character tags DB, and binds TargetSelector.

        Args:
            tag_filename: Optional custom filename for tags database loading.
        """
        self.tags_db: Dict[str, Any] = self._load_tags()
        self.target_selector: TargetSelector = TargetSelector()

    def _load_tags(self) -> Dict[str, Any]:
        """Loads cached character tags and metadata dictionary from AppData storage.

        Returns:
            Dictionary containing character tags and mechanism data.
        """
        try:
            tags = load_cached_tags()
            if tags:
                return tags
        except Exception as e:
            print(f"⚠️ [ContextBuilder] Failed to load tags from AppData cache: {e}")
        return {}

    def get_tag_info(self, name: Optional[str]) -> Dict[str, Any]:
        """Queries character tag database for priority rankings, roles, and skill details.

        Args:
            name: Character name string.

        Returns:
            Dictionary containing metadata attributes for the target character.
        """
        default_res: Dict[str, Any] = {
            "priority": 99,
            "tag": "一般單位",
            "role": "一般單位",
            "threat_mechanism": "",
            "counter_weakness": "",
            "static_action_tip": "",
            "suggested_skill": "S1",
            "priority_condition": "",
            "recommended_operation": "",
            "priority_target_roles": {"enemy_attack_pref": [], "ally_buff_pref": []},
            "skill_details": {}
        }
        if not name or "敵方" in name or "我方" in name:
            return default_res

        for key, data in self.tags_db.items():
            if key in name or name in key:
                if isinstance(data, dict):
                    prio = data.get("priority", data.get("static_priority", 5))
                    role_tag = data.get("role", data.get("tag", "一般單位"))

                    prio_roles = data.get("priority_target_roles", {})
                    if not isinstance(prio_roles, dict):
                        prio_roles = {"enemy_attack_pref": [], "ally_buff_pref": []}

                    return {
                        "priority": int(prio) if str(prio).isdigit() else 5,
                        "tag": role_tag if role_tag else "一般單位",
                        "role": role_tag if role_tag else "一般單位",
                        "static_action_tip": data.get("static_action_tip", ""),
                        "suggested_skill": data.get("suggested_skill", "S1"),
                        "threat_mechanism": data.get("threat_mechanism", ""),
                        "counter_weakness": data.get("counter_weakness", ""),
                        "priority_condition": data.get("priority_condition", ""),
                        "recommended_operation": data.get("recommended_operation", ""),
                        "priority_target_roles": prio_roles,
                        "skill_details": data.get("skill_details", {})
                    }
        return default_res

    def _compute_skill_votes(self, enriched_my_team: List[Dict[str, Any]]) -> List[str]:
        """Calculates team skill priority voting based on active team compositions.

        Args:
            enriched_my_team: List of active ally unit dictionaries.

        Returns:
            Ordered list of recommended skill keys ('s3', 's2', 's1').
        """
        votes = Counter()
        for unit in enriched_my_team:
            skill = unit.get("suggested_skill", "").upper()
            if "S3" in skill:
                votes["s3"] += 1
            elif "S2" in skill:
                votes["s2"] += 1
            else:
                votes["s1"] += 1

        if not votes:
            return ["s3", "s2", "s1"]

        sorted_skills = [item[0] for item in votes.most_common()]
        for fallback in ["s3", "s2", "s1"]:
            if fallback not in sorted_skills:
                sorted_skills.append(fallback)
        return sorted_skills

    def build_payload(
        self,
        my_status: Dict[str, Dict[str, Any]],
        opp_status: Dict[str, Dict[str, Any]],
        is_my_turn: bool = False,
        skill_states: Optional[Dict[str, bool]] = None,
        locked_target_slot: Optional[str] = None,
        active_slot: Optional[str] = None,
        active_shiki_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Assembles a comprehensive context payload for tactical decision engines.

        Args:
            my_status: Status dictionary for player team.
            opp_status: Status dictionary for opponent team.
            is_my_turn: Boolean flag indicating active player turn.
            skill_states: Dictionary mapping skill buttons ('s1'-'s4') to availability booleans.
            locked_target_slot: Optional manually locked target slot identifier.
            active_slot: Slot identifier of the active acting character.
            active_shiki_name: Name of the active acting character.

        Returns:
            Structured context payload dictionary.
        """
        self.tags_db = self._load_tags()

        if skill_states is None:
            skill_states = {"s1": False, "s2": False, "s3": False, "s4": False}

        if not active_shiki_name and active_slot:
            active_shiki_name = my_status.get(active_slot, {}).get("name", "未知式神")

        # 1. Invoke TargetSelector for dynamic target filtering
        target_res = self.target_selector.select_target(
            active_shiki_name=active_shiki_name,
            active_slot=active_slot,
            skill_states=skill_states,
            my_status=my_status,
            opp_status=opp_status,
            tags_db=self.tags_db,
            is_my_turn=is_my_turn
        )

        target_is_enemy = target_res.get("target_is_enemy", True)
        target_pool = opp_status if target_is_enemy else my_status

        # Evaluate manually locked target slot validity
        if locked_target_slot and target_pool.get(locked_target_slot, {}).get("alive", False):
            target_slot = locked_target_slot
        else:
            target_slot = target_res["target_slot"]

        # Dynamically retrieve target real name
        if target_slot:
            target_name = target_pool.get(target_slot, {}).get("name", target_res["target_name"])
        else:
            target_name = target_res["target_name"]  # e.g., "敵方全體" or "我方全體"

        target_info = self.get_tag_info(target_name)

        # 2. Assemble fallback tactical advice string
        shiki_info = self.get_tag_info(active_shiki_name)
        rec_op = shiki_info.get("recommended_operation", "")

        if not is_my_turn:
            threat = target_info.get("threat_mechanism", "")
            weakness = target_info.get("counter_weakness", "")
            if threat and weakness:
                fallback_msg = f"【敵方回合】留意 [{target_name}] 核心威脅：{threat}；弱點：{weakness}。"
            elif threat:
                fallback_msg = f"【敵方回合】留意 [{target_name}] 核心機制：{threat}。"
            else:
                fallback_msg = f"【敵方回合】等待敵方行動，預計優先壓制 [{target_name}] ({target_res['target_role']})。"
        else:
            turn_prefix = (
                f"我方 [{active_slot}] {active_shiki_name} 回合！"
                if active_slot and active_shiki_name != "未知"
                else "我方回合！"
            )

            # Optimize fallback message formatting for AoE skills (target_slot is None)
            if not target_slot:
                fallback_msg = f"{turn_prefix}建議發動 [{target_res['chosen_skill']}] 作用於 {target_name}。"
            else:
                action_prefix = "打擊敵方" if target_is_enemy else "指定我方"
                fallback_msg = (
                    f"{turn_prefix}建議發動 [{target_res['chosen_skill']}] "
                    f"{action_prefix} [{target_slot}] {target_name} ({target_res['target_role']})。"
                )

            if rec_op:
                fallback_msg += f"\n📌 戰術連招提示：{rec_op}"

        # 3. Package team metadata structures
        enriched_my_team = [
            {
                "slot": s,
                "name": info.get("name", ""),
                "role": self.get_tag_info(info.get("name", ""))["role"],
                "suggested_skill": self.get_tag_info(info.get("name", ""))["suggested_skill"],
                "is_active_turn": (s == active_slot)
            }
            for s, info in my_status.items() if info.get("alive", False)
        ]

        enriched_opp_team = [
            {
                "slot": s,
                "name": info.get("name", ""),
                "role": self.get_tag_info(info.get("name", ""))["role"],
                "threat": self.get_tag_info(info.get("name", ""))["threat_mechanism"],
                "weakness": self.get_tag_info(info.get("name", ""))["counter_weakness"]
            }
            for s, info in opp_status.items() if info.get("alive", False)
        ]

        return {
            "is_my_turn": is_my_turn,
            "active_shikigami_slot": active_slot,
            "active_shikigami_name": active_shiki_name,
            "active_shikigami_operation": rec_op,
            "active_shikigami_skills": shiki_info.get("skill_details", {}),
            "available_skills": skill_states,
            "recommended_skill_order": self._compute_skill_votes(enriched_my_team),
            "my_player_team": enriched_my_team,
            "opp_alive_units": enriched_opp_team,
            "enemy_target": {
                "slot": target_slot,
                "name": target_name,
                "role": target_info.get("role", "一般單位"),
                "threat": target_info.get("threat_mechanism", ""),
                "weakness": target_info.get("counter_weakness", "")
            },
            "recommended_target_slot": target_slot,
            "recommended_target_name": target_name,
            "recommended_target_tag": target_info.get("role", "一般單位"),
            "target_is_enemy": target_is_enemy,
            "fallback_advice": fallback_msg
        }