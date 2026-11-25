"""Battle Context Payload Builder for Onmyoji Tactical Helper.

Aggregates real-time board state, `ShikigamiRepository` O(1) cached combat metadata, and
`TargetSelector` decisions into structured payloads for LLM reasoning and instant rule advice.
"""

from collections import Counter
from typing import Any

from onmyoji_helper.decision.target_selector import TargetSelector
from onmyoji_helper.domain.models import UNKNOWN_SHIKIGAMI
from onmyoji_helper.domain.shikigami_repository import ShikigamiRepository
from onmyoji_helper.infra.sheets_sync_store import load_cached_tags


class BattleContextBuilder:
    """Assembles enriched battle contextual payloads without per-frame disk I/O or key re-sorting."""

    def __init__(self, repo: ShikigamiRepository | None = None) -> None:
        """Initializes the context builder with a cached `ShikigamiRepository` and `TargetSelector`."""
        self.repo: ShikigamiRepository = repo or ShikigamiRepository(tags_db=load_cached_tags())
        self.target_selector: TargetSelector = TargetSelector(repo=self.repo)

    def get_tag_info(self, name: str | None) -> dict[str, Any]:
        """Queries cached character metadata via `ShikigamiRepository`."""
        return self.repo.get_tag_info(name)

    @staticmethod
    def _compute_skill_votes(enriched_my_team: list[dict[str, Any]]) -> list[str]:
        votes: Counter = Counter()
        for unit in enriched_my_team:
            skill = str(unit.get("suggested_skill", "")).upper()
            if "S3" in skill:
                votes["s3"] += 1
            elif "S2" in skill:
                votes["s2"] += 1
            else:
                votes["s1"] += 1

        if not votes:
            return ["s3", "s2", "s1"]

        sorted_skills = [item[0] for item in votes.most_common()]
        for fallback in ("s3", "s2", "s1"):
            if fallback not in sorted_skills:
                sorted_skills.append(fallback)
        return sorted_skills

    def build_payload(
        self,
        my_status: dict[str, dict[str, Any]],
        opp_status: dict[str, dict[str, Any]],
        is_my_turn: bool = False,
        skill_states: dict[str, bool] | None = None,
        locked_target_slot: str | None = None,
        active_slot: str | None = None,
        active_shiki_name: str | None = None,
    ) -> dict[str, Any]:
        """Builds a structured tactical context dictionary for a single battle frame."""
        if skill_states is None:
            skill_states = {"s1": False, "s2": False, "s3": False, "s4": False}

        if not active_shiki_name and active_slot:
            active_shiki_name = str(my_status.get(active_slot, {}).get("name", UNKNOWN_SHIKIGAMI))

        target_res = self.target_selector.select_target(
            active_shiki_name=active_shiki_name,
            active_slot=active_slot,
            skill_states=skill_states,
            my_status=my_status,
            opp_status=opp_status,
            tags_db=self.repo,
            is_my_turn=is_my_turn,
        )

        target_is_enemy = bool(target_res.get("target_is_enemy", True))
        target_pool = opp_status if target_is_enemy else my_status

        if locked_target_slot and target_pool.get(locked_target_slot, {}).get("alive", False):
            target_slot = locked_target_slot
        else:
            target_slot = target_res["target_slot"]

        if target_slot:
            target_name = str(target_pool.get(target_slot, {}).get("name", target_res["target_name"]))
        else:
            target_name = str(target_res["target_name"])

        target_info = self.repo.get_tag_info(target_name)
        shiki_info = self.repo.get_tag_info(active_shiki_name)
        rec_op = str(shiki_info.get("recommended_operation", ""))

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
                if active_slot and active_shiki_name and active_shiki_name != UNKNOWN_SHIKIGAMI
                else "我方回合！"
            )
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

        # Single lookup per alive unit (eliminating previous 3x redundant calls per unit)
        enriched_my_team: list[dict[str, Any]] = []
        for s, info in my_status.items():
            if info.get("alive", False):
                u_name = str(info.get("name", ""))
                u_meta = self.repo.get_tag_info(u_name)
                enriched_my_team.append(
                    {
                        "slot": s,
                        "name": u_name,
                        "role": u_meta["role"],
                        "suggested_skill": u_meta["suggested_skill"],
                        "is_active_turn": (s == active_slot),
                    }
                )

        enriched_opp_team: list[dict[str, Any]] = []
        for s, info in opp_status.items():
            if info.get("alive", False):
                u_name = str(info.get("name", ""))
                u_meta = self.repo.get_tag_info(u_name)
                enriched_opp_team.append(
                    {
                        "slot": s,
                        "name": u_name,
                        "role": u_meta["role"],
                        "threat": u_meta["threat_mechanism"],
                        "weakness": u_meta["counter_weakness"],
                    }
                )

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
                "weakness": target_info.get("counter_weakness", ""),
            },
            "recommended_target_slot": target_slot,
            "recommended_target_name": target_name,
            "recommended_target_tag": target_info.get("role", "一般單位"),
            "target_is_enemy": target_is_enemy,
            "chosen_skill": target_res.get("chosen_skill", "S1"),
            "skill_scope": target_res.get("scope", "敵方單體"),
            "fallback_advice": fallback_msg,
        }
