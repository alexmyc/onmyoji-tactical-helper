"""Hierarchical Target Selection Engine for Onmyoji Tactical Helper.

Selects the optimal active skill (`S1..S4`) and target unit across single-target enemy,
single-target ally, self, enemy AoE (`敵方全體`), and ally AoE (`我方全體`) scopes using
`ShikigamiRepository` cached metadata.
"""

from typing import Any

from onmyoji_helper.domain.shikigami_repository import ShikigamiRepository


class TargetSelector:
    """Dynamic target selection engine implementing Onmyoji combat priority hierarchies."""

    def __init__(self, repo: ShikigamiRepository | None = None) -> None:
        """Initializes the selector with an optional shared `ShikigamiRepository`."""
        self._repo: ShikigamiRepository | None = repo
        self._adhoc_repo_id: int | None = None
        self._adhoc_repo: ShikigamiRepository | None = None

    def _resolve_repo(self, tags_source: ShikigamiRepository | dict[str, Any]) -> ShikigamiRepository:
        if isinstance(tags_source, ShikigamiRepository):
            return tags_source
        if self._repo is not None and self._repo.tags_db is tags_source:
            return self._repo
        if self._adhoc_repo is None or self._adhoc_repo_id != id(tags_source):
            self._adhoc_repo = ShikigamiRepository(tags_db=tags_source)
            self._adhoc_repo_id = id(tags_source)
        return self._adhoc_repo

    def select_target(
        self,
        active_shiki_name: str | None,
        active_slot: str | None,
        skill_states: dict[str, bool],
        my_status: dict[str, dict[str, Any]],
        opp_status: dict[str, dict[str, Any]],
        tags_db: ShikigamiRepository | dict[str, Any],
        is_my_turn: bool = True,
    ) -> dict[str, Any]:
        """Determines optimal skill selection and target positioning for the current frame."""
        repo = self._resolve_repo(tags_db)

        if not is_my_turn:
            best_slot, best_name, best_role = self._filter_and_sort_target(
                opp_status, ["控制", "輸出", "拉條", "供火", "輔助"], repo, is_enemy=True
            )
            return {
                "target_slot": best_slot,
                "target_name": best_name,
                "target_role": best_role,
                "chosen_skill": "S1",
                "scope": "敵方單體",
                "target_is_enemy": True,
            }

        shiki_info = repo.get_tag_info(active_shiki_name)
        prio_roles = shiki_info.get("priority_target_roles", {})
        skills_detail = shiki_info.get("skill_details", {})
        db_suggested = shiki_info.get("suggested_skill", "S3")

        chosen_skill = self._determine_best_skill(
            skill_states=skill_states,
            skills_detail=skills_detail,
            db_suggested_skill=db_suggested,
        )

        skill_info = skills_detail.get(chosen_skill, {}) or skills_detail.get(chosen_skill.lower(), {})
        scope = str(skill_info.get("scope", "")).strip()

        if chosen_skill == "S4" and (not scope or scope.lower() in ("", "none", "null")):
            scope = "自己"
        elif not scope:
            scope = "敵方單體"

        # Scope A: AoE Skills (敵方全體 / 我方全體) -> AutoExecutor resolves an alive side unit
        if "全體" in scope or "群" in scope:
            is_ally_aoe = any(
                kw in scope for kw in ("我方", "友方", "群輔", "治療", "拉條", "回火")
            )
            target_is_enemy = ("敵方" in scope) or (not is_ally_aoe)
            return {
                "target_slot": None,
                "target_name": "敵方全體" if target_is_enemy else "我方全體",
                "target_role": "全體",
                "chosen_skill": chosen_skill,
                "scope": scope,
                "target_is_enemy": target_is_enemy,
            }

        # Scope B: Self-targeted skills
        if scope == "自己":
            my_info = my_status.get(active_slot or "", {})
            self_name = str(my_info.get("name", active_shiki_name or "我方單位"))
            return {
                "target_slot": active_slot,
                "target_name": self_name,
                "target_role": "自己",
                "chosen_skill": chosen_skill,
                "scope": scope,
                "target_is_enemy": False,
            }

        # Scope C: Ally Single Target (exclude active self unless sole survivor)
        if "我方" in scope or scope == "我方單體":
            target_role_pref = skill_info.get("target_role_pref", []) or prio_roles.get(
                "ally_buff_pref", ["輸出", "控制", "拉條", "輔助", "供火"]
            )
            best_slot, best_name, best_role = self._filter_and_sort_target(
                my_status, target_role_pref, repo, is_enemy=False, exclude_slot=active_slot
            )
            return {
                "target_slot": best_slot,
                "target_name": best_name,
                "target_role": best_role,
                "chosen_skill": chosen_skill,
                "scope": scope,
                "target_is_enemy": False,
            }

        # Scope D: Enemy Single Target
        target_role_pref = skill_info.get("target_role_pref", []) or prio_roles.get(
            "enemy_attack_pref", ["輸出", "控制", "拉條", "供火", "輔助"]
        )
        best_slot, best_name, best_role = self._filter_and_sort_target(
            opp_status, target_role_pref, repo, is_enemy=True
        )
        return {
            "target_slot": best_slot,
            "target_name": best_name,
            "target_role": best_role,
            "chosen_skill": chosen_skill,
            "scope": scope,
            "target_is_enemy": True,
        }

    @staticmethod
    def _determine_best_skill(
        skill_states: dict[str, bool],
        skills_detail: dict[str, Any],
        db_suggested_skill: str = "S3",
    ) -> str:
        """Selects the highest-priority active non-passive skill."""
        valid_skills: dict[str, bool] = {}
        for skill_key in ("S1", "S2", "S3", "S4"):
            info = skills_detail.get(skill_key, {}) or skills_detail.get(skill_key.lower(), {})
            scope = str(info.get("scope", "")).strip()
            if skill_key == "S4" and (not scope or scope.lower() in ("", "none", "null")):
                scope = "自己"

            is_passive_or_none = any(kw in scope for kw in ("被動", "無"))
            if skill_key == "S1" and (is_passive_or_none or not scope):
                is_passive_or_none = False

            valid_skills[skill_key] = bool(skill_states.get(skill_key.lower())) and not is_passive_or_none

        suggested = str(db_suggested_skill).upper() if db_suggested_skill else "S3"
        if valid_skills.get(suggested, False):
            return suggested

        for sk in ("S3", "S2", "S4", "S1"):
            if valid_skills.get(sk, False):
                return sk
        return "S1"

    @staticmethod
    def _filter_and_sort_target(
        status_dict: dict[str, dict[str, Any]],
        role_pref_list: list[str] | None,
        repo: ShikigamiRepository,
        is_enemy: bool = True,
        exclude_slot: str | None = None,
    ) -> tuple[str | None, str, str]:
        """Filters alive units by preferred combat role and ascending static priority."""
        alive_units: list[dict[str, Any]] = []
        for slot, info in status_dict.items():
            if exclude_slot and slot == exclude_slot:
                continue
            name = str(info.get("name", ""))
            is_alive = bool(info.get("alive", False))
            if is_alive and name and ("敵方" not in name if is_enemy else "我方" not in name):
                t_info = repo.get_tag_info(name)
                alive_units.append(
                    {
                        "slot": slot,
                        "name": name,
                        "role": t_info["role"],
                        "priority": int(t_info["priority"]),
                    }
                )

        if not alive_units:
            if exclude_slot and exclude_slot in status_dict:
                my_name = str(status_dict[exclude_slot].get("name", "我方單位"))
                return exclude_slot, my_name, "一般單位"
            return "Slot 1", ("敵方單位" if is_enemy else "我方單位"), "一般單位"

        if role_pref_list and isinstance(role_pref_list, list):
            for pref_role in role_pref_list:
                matched = [u for u in alive_units if u["role"] == pref_role]
                if matched:
                    matched.sort(key=lambda x: x["priority"])
                    best = matched[0]
                    return best["slot"], best["name"], best["role"]

        alive_units.sort(key=lambda x: x["priority"])
        best = alive_units[0]
        return best["slot"], best["name"], best["role"]
