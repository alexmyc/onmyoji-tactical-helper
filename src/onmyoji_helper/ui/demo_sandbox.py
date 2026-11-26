"""Interactive Offline Match Sandbox (`--demo`) for Onmyoji Tactical Helper.

Allows technical interviewers, hiring managers, and contributors to step through live Ban/Pick
counter-tree reasoning, dead-branch viability pivots, mirror-match detection, and 5v5 combat
targeting using the real production engines without requiring the game client, GPU, or `.env`.
"""

import logging
import tkinter as tk
from collections.abc import Callable
from dataclasses import dataclass
from tkinter import font
from typing import Any

from onmyoji_helper.decision.battle_context import BattleContextBuilder
from onmyoji_helper.decision.draft_engine import get_recommendation
from onmyoji_helper.decision.draft_fsm import DraftStateMachine
from onmyoji_helper.decision.llm_engine import TacticalEngine
from onmyoji_helper.domain.models import DraftPhase, OperatingMode
from onmyoji_helper.domain.public_demo_data import (
    format_bilingual_list,
    format_bilingual_token,
)
from onmyoji_helper.domain.shikigami_repository import ShikigamiRepository
from onmyoji_helper.ui.main_window import OnmyojiHelperUI
from onmyoji_helper.ui.tactical_panel import AITacticalUI

logger = logging.getLogger("OnmyojiHelper")


@dataclass(frozen=True)
class SandboxStepResult:
    """Structured telemetry emitted after executing a single interactive sandbox step."""

    scenario_index: int
    scenario_title: str
    step_index: int
    total_steps: int
    headline: str
    telemetry_lines: list[str]
    opp_ban: str | None
    opp_draft: list[str]
    my_draft: list[str]
    my_recs: list[str]
    opp_preds: list[str]
    status_text: str
    mode: str
    tactical_advice: str


class DemoSandboxController:
    """Headless-testable orchestrator driving real Draft & Combat engines across 4 curated scenarios."""

    SCENARIO_TITLES: tuple[str, ...] = (
        "1 · XMind Counter-Chain & Shuffled Enemy Set",
        "2 · Stolen Core Card → Dead-Branch Viability Pivot",
        "3 · Mirror / Copycat Playbook Detection",
        "4 · 5v5 Combat Targeting & Orb-Shortage Fallback",
    )

    def __init__(
        self,
        compiled_dbs: dict[str, dict[str, Any]],
        repository: ShikigamiRepository,
    ) -> None:
        self.compiled_dbs: dict[str, dict[str, Any]] = compiled_dbs
        self.repository: ShikigamiRepository = repository
        self.fsm: DraftStateMachine = DraftStateMachine()
        self.context_builder: BattleContextBuilder = BattleContextBuilder(repo=repository)
        self.tactical_engine: TacticalEngine = TacticalEngine(force_rule_only=True)

        system_keys = list(compiled_dbs.keys())
        self.primary_sys_name: str = system_keys[0]
        self.control_sys_name: str = system_keys[1] if len(system_keys) > 1 else system_keys[0]
        self.active_sys_name: str = self.primary_sys_name

        self.scenario_idx: int = 0
        self.step_idx: int = -1

    def _get_system(self, name: str | None = None) -> tuple[dict[str, Any], str]:
        target_name = name if (name and name in self.compiled_dbs) else self.active_sys_name
        entry = self.compiled_dbs[target_name]
        return entry["db"], str(entry["my_ban"])

    def set_active_playbook(self, playbook_name: str) -> SandboxStepResult:
        """Switches the active playbook in `--demo` and re-evaluates the current draft scenario."""
        if playbook_name in self.compiled_dbs:
            self.active_sys_name = playbook_name
        curr_step = max(0, self.step_idx)
        if self.scenario_idx == 3:
            return self._exec_scenario_combat_turns(curr_step)
        if self.scenario_idx == 1:
            return self._exec_scenario_stolen_pivot(curr_step)
        if self.scenario_idx == 2:
            return self._exec_scenario_mirror_detection(curr_step)
        return self._exec_scenario_counter_chain(curr_step)

    def switch_demo_mode(self, target_mode: str) -> SandboxStepResult:
        """Jumps interactively to a representative Green, Yellow, or Red state in `--demo`."""
        mode_val = target_mode.value if isinstance(target_mode, OperatingMode) else str(target_mode).lower()
        if mode_val == OperatingMode.GREEN.value:
            self.scenario_idx = 0
            self.step_idx = 3
            return self._exec_scenario_counter_chain(3)
        if mode_val == OperatingMode.YELLOW.value:
            self.scenario_idx = 3
            self.step_idx = 0
            return self._exec_scenario_combat_turns(0)
        self.scenario_idx = 3
        self.step_idx = 1
        return self._exec_scenario_combat_turns(1)

    def total_steps_for_scenario(self, scenario_idx: int) -> int:
        counts = (5, 3, 2, 4)
        return counts[scenario_idx % len(counts)]

    def select_scenario(self, scenario_idx: int) -> SandboxStepResult:
        """Resets state to the start of `scenario_idx` and executes its first step (`step 0`)."""
        self.scenario_idx = scenario_idx % len(self.SCENARIO_TITLES)
        self.step_idx = -1
        self.fsm.reset(DraftPhase.SLEEP)
        return self.next_step()

    def reset(self) -> SandboxStepResult:
        """Resets back to Scenario 1, Step 1."""
        return self.select_scenario(0)

    def next_step(self) -> SandboxStepResult:
        """Advances one step in the active scenario (wrapping cleanly across scenarios)."""
        total = self.total_steps_for_scenario(self.scenario_idx)
        if self.step_idx + 1 >= total:
            self.scenario_idx = (self.scenario_idx + 1) % len(self.SCENARIO_TITLES)
            self.step_idx = 0
            self.fsm.reset(DraftPhase.SLEEP)
        else:
            self.step_idx += 1
            if self.step_idx == 0:
                self.fsm.reset(DraftPhase.SLEEP)

        if self.scenario_idx == 0:
            return self._exec_scenario_counter_chain(self.step_idx)
        if self.scenario_idx == 1:
            return self._exec_scenario_stolen_pivot(self.step_idx)
        if self.scenario_idx == 2:
            return self._exec_scenario_mirror_detection(self.step_idx)
        return self._exec_scenario_combat_turns(self.step_idx)

    def _exec_scenario_counter_chain(self, step: int) -> SandboxStepResult:
        db, my_ban = self._get_system()
        ocr_frames: list[tuple[list[tuple[None, str, float]], list[tuple[None, str, float]], str]] = [
            (
                [(None, "禁選階段", 0.99)],
                [],
                "Ban Phase Entered — Strict Ban Hard-Gate blocks premature pick noise.",
            ),
            (
                [(None, "禁用封陽君", 0.99)],
                [],
                "Enemy Locks Ban [封] — Cold-start tree evaluates Ban封 root openers.",
            ),
            (
                [(None, "石長姬", 0.99), (None, "天照", 0.99)],
                [(None, "不知火", 0.99), (None, "天照", 0.99)],
                "Picks 1–2 — Order-invariant set scorer matches shuffled enemy [石, 神].",
            ),
            (
                [(None, "大夜摩天閻魔", 0.99), (None, "龍吟鈴鹿御前", 0.99)],
                [(None, "葛葉", 0.99), (None, "一目連", 0.99)],
                "Picks 3–4 — Ordered ally backbone [離→神→葛→連] narrows 5th-pick counter.",
            ),
            (
                [(None, "神釀星熊童子", 0.99)],
                [(None, "荒骷髏", 0.99)],
                "Pick 5 Locked — DraftStateMachine triggers 5v5 Battle Handoff (Yellow Mode).",
            ),
        ]

        # Replay FSM deterministically from step 0..step if jumped directly
        self.fsm.reset(DraftPhase.SLEEP)
        snapshot = None
        for idx in range(step + 1):
            ocr_opp, ocr_my, _ = ocr_frames[idx]
            snapshot, _ = self.fsm.step(
                ocr_opp=ocr_opp,
                ocr_my=ocr_my,
                sorted_full_to_short=self.repository.sorted_full_to_short,
                my_ban=my_ban,
                now=1000.0 + idx,
            )

        assert snapshot is not None
        headline = ocr_frames[step][2]

        if snapshot.opp_ban:
            recs, preds, match_type = get_recommendation(
                db=db,
                current_ban=f"Ban{snapshot.opp_ban}",
                opp_draft=snapshot.opp_draft,
                my_draft=snapshot.my_draft,
                my_ban=my_ban,
            )
            opp_ban_ui: str | None = snapshot.opp_ban
        else:
            recs, preds, match_type = get_recommendation(
                db=db,
                current_ban="",
                opp_draft=[],
                my_draft=[],
                my_ban=my_ban,
            )
            opp_ban_ui = "禁選中"

        mode = OperatingMode.YELLOW.value if snapshot.just_completed_five_picks else OperatingMode.GREEN.value
        if snapshot.just_completed_five_picks:
            advice = (
                "⚡ [5v5 Battle Handoff Ready — Yellow Mode]\n"
                "• 5-Pick Portrait Templates Locked (`ActionBarDetector`, 5/5 slots)\n"
                "• Waiting for Deterministic Skill-Bar Ready (`is_my_turn == True`) to calibrate baselines\n\n"
                "🎯 Priority Threat: Enemy [Slot 3] 閻 (SP Enma — Top-speed single-target morph/silence)\n"
                "💡 Opening Plan: Use 連 (Ichimokuren) wind shields & 葛 (Kuzunoha) fox-guard on 骷 (Gashadokuro)."
            )
        else:
            top_recs_bi = format_bilingual_list(recs[:3])
            top_preds_bi = format_bilingual_list(preds[:3]) if preds else "—"
            advice = (
                "🟢 [Green Mode — Draft Decision Support (0 MB GPU VRAM)]\n"
                f"• FSM Phase: {snapshot.phase.value} | Poll Cadence: {self.fsm.get_recommended_poll_interval():.1f}s\n"
                f"• Recommended Next Counter: {top_recs_bi}\n"
                f"• Predicted Opponent Pool: {top_preds_bi}"
            )

        telemetry = [
            f"Playbook: {self.active_sys_name} (My Ban: {format_bilingual_token(my_ban)})",
            f"FSM Phase: {snapshot.phase.value} | Opp Ban: {format_bilingual_token(opp_ban_ui)}",
            f"Ally Backbone (Ordered): {format_bilingual_list(snapshot.my_draft)}",
            f"Enemy Formation (Set):   {format_bilingual_list(snapshot.opp_draft)}",
            f"XMind Top Counters:      {format_bilingual_list(recs[:4])}",
            f"Opponent Predictions:    {format_bilingual_list(preds[:4])}",
            f"Branch Score Telemetry:  {match_type}",
        ]
        return SandboxStepResult(
            scenario_index=0,
            scenario_title=self.SCENARIO_TITLES[0],
            step_index=step,
            total_steps=len(ocr_frames),
            headline=headline,
            telemetry_lines=telemetry,
            opp_ban=opp_ban_ui,
            opp_draft=snapshot.opp_draft,
            my_draft=snapshot.my_draft,
            my_recs=recs,
            opp_preds=preds,
            status_text=match_type,
            mode=mode,
            tactical_advice=advice,
        )

    def _exec_scenario_stolen_pivot(self, step: int) -> SandboxStepResult:
        db, my_ban = self._get_system(self.primary_sys_name)
        states = [
            (
                "封",
                [],
                [],
                "Step 1 — Enemy locks Ban [封]; primary branch [離→神→葛→連→骷] recommends [離].",
            ),
            (
                "封",
                ["葛", "神"],
                ["離"],
                "Step 2 — Enemy STEALS [葛]! Viability penalty (0.18^1) decays dead branch → pivots to [百羽].",
            ),
            (
                "封",
                ["葛", "神", "石"],
                ["離", "百羽", "酒吞"],
                "Step 3 — Clean pivot onto [離→百羽→酒吞→追月→連] branch; recommends [追月] next.",
            ),
        ]
        opp_ban, opp_draft, my_draft, headline = states[step]
        recs, preds, match_type = get_recommendation(
            db=db,
            current_ban=f"Ban{opp_ban}",
            opp_draft=opp_draft,
            my_draft=my_draft,
            my_ban=my_ban,
        )
        advice = (
            "🧠 [XMind Dead-Branch Viability Decay]\n"
            f"• Formula: γ_viability = (0.12^n_ban) × (0.18^n_stolen)\n"
            f"• Status: {headline}\n"
            f"• Active Top Recommendation: [{recs[0]}]"
        )
        telemetry = [
            f"Playbook: {self.primary_sys_name} | Ban: Ban{opp_ban} (My Ban: {my_ban})",
            f"Ally Picks:  {my_draft}",
            f"Enemy Picks: {opp_draft}",
            "Broken Branch [離,神,葛,連,骷]: Penalized by 0.18x after [葛] stolen",
            f"Surviving Pivot Recommendation: {recs}",
            f"Opponent Remaining Predictions: {preds}",
            f"Engine Match Tag: {match_type}",
        ]
        return SandboxStepResult(
            scenario_index=1,
            scenario_title=self.SCENARIO_TITLES[1],
            step_index=step,
            total_steps=len(states),
            headline=headline,
            telemetry_lines=telemetry,
            opp_ban=opp_ban,
            opp_draft=opp_draft,
            my_draft=my_draft,
            my_recs=recs,
            opp_preds=preds,
            status_text=match_type,
            mode=OperatingMode.GREEN.value,
            tactical_advice=advice,
        )

    def _exec_scenario_mirror_detection(self, step: int) -> SandboxStepResult:
        db, my_ban = self._get_system(self.primary_sys_name)
        states = [
            (
                "禪",
                ["離", "百羽"],
                ["離"],
                "Step 1 — Opponent opens with our own [離 + 百羽] Tempo backbone.",
            ),
            (
                "禪",
                ["離", "百羽", "酒吞"],
                ["離", "石"],
                "Step 2 — 3-card backbone overlap triggers [鏡像/內戰] Mirror Alert & predicts [SP06]!",
            ),
        ]
        opp_ban, opp_draft, my_draft, headline = states[step]
        recs, preds, match_type = get_recommendation(
            db=db,
            current_ban=f"Ban{opp_ban}",
            opp_draft=opp_draft,
            my_draft=my_draft,
            my_ban=my_ban,
        )
        advice = (
            "🪞 [Mirror / Copycat Playbook Detector]\n"
            "• Opponent is drafting our own recorded `my` backbone [離→百羽→酒吞→SP06→連]!\n"
            f"• Engine Status Tag: {match_type}\n"
            f"• Predicted Opponent Follow-up: {' / '.join(preds[:3])}"
        )
        telemetry = [
            f"Playbook: {self.primary_sys_name} | Ban: Ban{opp_ban}",
            f"Opponent Revealed: {opp_draft} (Matches our `my` column!)",
            f"Mirror Detection Tag: {match_type}",
            f"Predicted Mirror Next Card: {preds}",
            f"Recommended Counter Pivot:  {recs}",
        ]
        return SandboxStepResult(
            scenario_index=2,
            scenario_title=self.SCENARIO_TITLES[2],
            step_index=step,
            total_steps=len(states),
            headline=headline,
            telemetry_lines=telemetry,
            opp_ban=opp_ban,
            opp_draft=opp_draft,
            my_draft=my_draft,
            my_recs=recs,
            opp_preds=preds,
            status_text=match_type,
            mode=OperatingMode.GREEN.value,
            tactical_advice=advice,
        )

    def _exec_scenario_combat_turns(self, step: int) -> SandboxStepResult:
        opp_status: dict[str, dict[str, Any]] = {
            "Slot 1": {"name": "玲", "alive": True, "pos": (420, 210)},
            "Slot 2": {"name": "SP桃", "alive": True, "pos": (520, 195)},
            "Slot 3": {"name": "葛", "alive": True, "pos": (635, 190)},
            "Slot 4": {"name": "酒吞", "alive": True, "pos": (745, 205)},
            "Slot 5": {"name": "季", "alive": True, "pos": (860, 225)},
        }
        combat_cases: list[dict[str, Any]] = [
            {
                "mode": OperatingMode.YELLOW.value,
                "headline": "Turn 1 (Yellow HUD) — Single-Target Speed CC: [閻] locks Priority-1 [玲] with [S3].",
                "my_draft": ["葛", "帝", "骷", "閻", "連"],
                "my_status": {
                    "Slot 1": {"name": "葛", "alive": True, "pos": (280, 440)},
                    "Slot 2": {"name": "帝", "alive": True, "pos": (430, 460)},
                    "Slot 3": {"name": "骷", "alive": True, "pos": (590, 470)},
                    "Slot 4": {"name": "閻", "alive": True, "pos": (750, 455)},
                    "Slot 5": {"name": "連", "alive": True, "pos": (900, 435)},
                },
                "active_slot": "Slot 4",
                "active_name": "閻",
                "skills": {"s1": True, "s2": False, "s3": True, "s4": True},
            },
            {
                "mode": OperatingMode.RED.value,
                "headline": "Turn 2 (Red Auto-Exec Sim) — Enemy AoE: [神荒] casts [S3] (2-step alive enemy lock + cast).",
                "my_draft": ["離", "葛", "帝", "連", "神荒"],
                "my_status": {
                    "Slot 1": {"name": "離", "alive": True, "pos": (280, 440)},
                    "Slot 2": {"name": "葛", "alive": True, "pos": (430, 460)},
                    "Slot 3": {"name": "帝", "alive": True, "pos": (590, 470)},
                    "Slot 4": {"name": "連", "alive": True, "pos": (750, 455)},
                    "Slot 5": {"name": "神荒", "alive": True, "pos": (900, 435)},
                },
                "active_slot": "Slot 5",
                "active_name": "神荒",
                "skills": {"s1": True, "s2": False, "s3": True, "s4": False},
            },
            {
                "mode": OperatingMode.RED.value,
                "headline": "Turn 3 (Red Auto-Exec Sim) — Ally AoE: [追月] casts [S3] (2-step alive ally lock + cast).",
                "my_draft": ["追月", "帝", "閻", "SP06", "連"],
                "my_status": {
                    "Slot 1": {"name": "追月", "alive": True, "pos": (280, 440)},
                    "Slot 2": {"name": "帝", "alive": True, "pos": (430, 460)},
                    "Slot 3": {"name": "閻", "alive": True, "pos": (590, 470)},
                    "Slot 4": {"name": "SP06", "alive": True, "pos": (750, 455)},
                    "Slot 5": {"name": "連", "alive": True, "pos": (900, 435)},
                },
                "active_slot": "Slot 1",
                "active_name": "追月",
                "skills": {"s1": True, "s2": False, "s3": True, "s4": False},
            },
            {
                "mode": OperatingMode.RED.value,
                "headline": "Turn 4 (Orb Shortage Safety) — [SP06] S3 greyed out (0 orbs) → auto-downgrades to [S1] on [玲]!",
                "my_draft": ["追月", "帝", "SP06", "閻", "連"],
                "my_status": {
                    "Slot 1": {"name": "追月", "alive": True, "pos": (280, 440)},
                    "Slot 2": {"name": "帝", "alive": True, "pos": (430, 460)},
                    "Slot 3": {"name": "SP06", "alive": True, "pos": (590, 470)},
                    "Slot 4": {"name": "閻", "alive": True, "pos": (750, 455)},
                    "Slot 5": {"name": "連", "alive": True, "pos": (900, 435)},
                },
                "active_slot": "Slot 3",
                "active_name": "SP06",
                "skills": {"s1": True, "s2": False, "s3": False, "s4": False},
            },
        ]

        case = combat_cases[step]
        payload = self.context_builder.build_payload(
            my_status=case["my_status"],
            opp_status=opp_status,
            is_my_turn=True,
            skill_states=case["skills"],
            active_slot=case["active_slot"],
            active_shiki_name=case["active_name"],
        )
        llm_res = self.tactical_engine.get_tactical_advice(payload, mode="RULE")

        chosen_skill = str(payload["chosen_skill"])
        scope = str(payload["skill_scope"])
        target_slot = payload["recommended_target_slot"]
        target_name = str(payload["recommended_target_name"])
        target_is_enemy = bool(payload["target_is_enemy"])
        actor_bi = format_bilingual_token(str(case["active_name"]))
        target_bi = format_bilingual_token(target_name)
        scope_en_map = {
            "敵方單體": "Enemy Single-Target",
            "敵方全體": "Enemy AoE",
            "我方全體": "Ally AoE",
            "我方單體": "Ally Single-Target",
            "自己": "Self",
        }
        scope_en = scope_en_map.get(scope, scope)

        mode_banner = (
            "🔴 [Red Mode Simulation — Skill-First Dry-Run (Zero OS Mouse Injection)]"
            if case["mode"] == OperatingMode.RED.value
            else "🟡 [Yellow Mode — Non-Activating HUD + Tactical Advisor]"
        )
        en_summary = (
            f"• Active Turn (`ActionBarDetector`): [{case['active_slot']}] {actor_bi}\n"
            f"• Decision: Cast [{chosen_skill}] ({scope_en}) → "
            f"{'Enemy' if target_is_enemy else 'Ally'} {f'[{target_slot}] ' if target_slot else ''}{target_bi}"
        )
        advice_text = (
            f"{mode_banner}\n"
            f"{en_summary}\n"
            f"• Available Skill Bar (`TurnDetector`): {case['skills']}\n"
            f"{'─' * 30}\n"
            f"{payload['fallback_advice']}"
        )

        exec_plan = (
            f"Skill-First 2-Step -> [1] Click [{chosen_skill}] -> 0.24s highlight pause -> "
            f"[2] Click {'Enemy' if target_is_enemy else 'Ally'} {target_slot or 'Slot 1'} ({target_bi})"
        )
        telemetry = [
            "Turn-UI Baseline Gate: Calibrated on deterministic skill-bar ready (is_my_turn=True)",
            f"ActionBarDetector Match: [{case['active_slot']}] {actor_bi} (score=0.84 >= 0.15)",
            f"TargetSelector Output: skill={chosen_skill}, scope={scope_en}, is_enemy={target_is_enemy}",
            f"AutoBattleExecutor Plan: {exec_plan}",
            f"Tactical Engine Tier: {llm_res['mode']} (+ 1.2–1.8s unfinished-turn retry guard)",
        ]
        return SandboxStepResult(
            scenario_index=3,
            scenario_title=self.SCENARIO_TITLES[3],
            step_index=step,
            total_steps=len(combat_cases),
            headline=str(case["headline"]),
            telemetry_lines=telemetry,
            opp_ban="不相",
            opp_draft=["玲", "SP桃", "葛", "酒吞", "季"],
            my_draft=list(case["my_draft"]),
            my_recs=["【武運亨通，旗開得勝】"],
            opp_preds=[],
            status_text=f"5v5 Combat Sim · {case['active_name']} → [{chosen_skill}] → {target_name}",
            mode=str(case["mode"]),
            tactical_advice=advice_text,
        )


class DemoSandboxWindow:
    """Interactive dark-mode Toplevel controller driving `OnmyojiHelperUI` and `AITacticalUI`."""

    def __init__(
        self,
        root: tk.Tk,
        ui_app: OnmyojiHelperUI,
        compiled_dbs: dict[str, dict[str, Any]],
        repository: ShikigamiRepository,
        demo_mode: bool = False,
    ) -> None:
        self.root: tk.Tk = root
        self.ui_app: OnmyojiHelperUI = ui_app
        self.demo_mode: bool = demo_mode
        self.controller: DemoSandboxController = DemoSandboxController(compiled_dbs, repository)
        self.auto_playing: bool = False
        self._after_id: str | None = None

        self.window: tk.Toplevel = tk.Toplevel(root)
        self.window.title("Onmyoji Tactical Helper — Interactive Match Sandbox (--demo)")
        self.window.configure(bg="#121212")
        self.window.attributes("-topmost", True)
        self.window.resizable(False, False)

        # Position MainWindow (top-right), AITacticalUI (bottom-right), and Sandbox (top-left/center)
        screen_w = self.root.winfo_screenwidth()
        right_x = max(580, screen_w - 555)
        self.root.geometry(f"520x545+{right_x}+40")
        if self.demo_mode:
            self.root.title("Onmyoji Tactical Helper — Live Draft & Mode View (--demo)")

        self.ai_panel: AITacticalUI = AITacticalUI(
            self.root,
            close_callback=self._on_ai_panel_closed,
        )
        if self.demo_mode:
            self.ai_panel.window.title("AI Tactical Advisor — Live Telemetry (--demo)")
        self.ai_panel.window.geometry(f"520x310+{right_x}+605")

        left_x = max(25, right_x - 600)
        self.window.geometry(f"575x650+{left_x}+40")
        self.window.protocol("WM_DELETE_WINDOW", self.close)

        self.h_font = font.Font(family="Segoe UI", size=12, weight="bold")
        self.sub_font = font.Font(family="Segoe UI", size=9)
        self.btn_font = font.Font(family="Segoe UI", size=9, weight="bold")
        self.mono_font = font.Font(family="Consolas", size=9)

        self._scenario_btns: list[tk.Button] = []
        self._mode_btns: dict[str, tk.Button] = {}
        self._build_widgets()
        if self.demo_mode or self.ui_app.mode_change_callback is None:
            self._attach_main_window_hooks()
        self.apply_result(self.controller.select_scenario(0))

    def _attach_main_window_hooks(self) -> None:
        """Wires `MainWindow` mode dot, reset button, and playbook menu directly into the sandbox."""
        self.ui_app.mode_change_callback = self.on_mode_requested
        self.ui_app.enable_mode_click()
        if self.demo_mode:
            self.ui_app.btn_reset.config(command=self._on_reset_clicked)
            orig_on_ban = self.ui_app.on_ban_changed

            def _demo_on_ban_changed(selected: str) -> None:
                orig_on_ban(selected)
                self._stop_autoplay()
                self.apply_result(self.controller.set_active_playbook(selected))

            def _make_pb_cmd(val: str) -> Callable[[], None]:
                def _run() -> None:
                    self.ui_app.selected_ban.set(val)
                    _demo_on_ban_changed(val)

                return _run

            menu = self.ui_app.ban_menu["menu"]
            menu.delete(0, "end")
            for pb_name in self.ui_app.available_bans:
                menu.add_command(label=pb_name, command=_make_pb_cmd(pb_name))

    def on_mode_requested(self, target_mode: str) -> None:
        """Handles mode switches triggered either from `MainWindow`'s status dot or Sandbox buttons."""
        self._stop_autoplay()
        self.apply_result(self.controller.switch_demo_mode(target_mode))

    def _on_ai_panel_closed(self) -> None:
        pass

    def winfo_exists(self) -> bool:
        try:
            return bool(self.window.winfo_exists())
        except Exception:
            return False

    def _ensure_ai_panel(self) -> None:
        if not self.ai_panel.winfo_exists():
            screen_w = self.root.winfo_screenwidth()
            right_x = max(580, screen_w - 555)
            self.ai_panel = AITacticalUI(self.root, close_callback=self._on_ai_panel_closed)
            self.ai_panel.window.geometry(f"520x310+{right_x}+605")

    def _build_widgets(self) -> None:
        header = tk.Frame(self.window, bg="#1C1C1E")
        header.pack(fill=tk.X, padx=14, pady=(14, 8), ipadx=8, ipady=8)

        tk.Label(
            header,
            text="🧪 Offline Interactive Match Sandbox (No Game / No .env Needed)",
            bg="#1C1C1E",
            fg="#4CC2FF",
            font=self.h_font,
        ).pack(anchor="w", padx=10)
        tk.Label(
            header,
            text=(
                "Drives the real DraftStateMachine, XMind Counter-Tree, TargetSelector & TacticalEngine.\n"
                "Watch the live Chinese Draft UI (top-right) and AI Tactical Panel (bottom-right) react!"
            ),
            bg="#1C1C1E",
            fg="#AEAEB2",
            font=self.sub_font,
            justify="left",
        ).pack(anchor="w", padx=10, pady=(4, 0))

        scen_frame = tk.Frame(self.window, bg="#121212")
        scen_frame.pack(fill=tk.X, padx=14, pady=4)
        tk.Label(
            scen_frame,
            text="SELECT ENGINEERING SCENARIO:",
            bg="#121212",
            fg="#8E8E93",
            font=self.btn_font,
        ).pack(anchor="w", pady=(0, 4))

        def _make_scenario_cmd(scen_idx: int) -> Callable[[], None]:
            return lambda: self._on_select_scenario(scen_idx)

        for idx, title in enumerate(DemoSandboxController.SCENARIO_TITLES):
            btn = tk.Button(
                scen_frame,
                text=f"  {title}",
                anchor="w",
                bg="#1C1C1E",
                fg="#F5F5F7",
                activebackground="#2C2C2E",
                activeforeground="#FFFFFF",
                font=self.sub_font,
                relief="flat",
                cursor="hand2",
                command=_make_scenario_cmd(idx),
            )
            btn.pack(fill=tk.X, pady=2, ipady=4)
            self._scenario_btns.append(btn)

        ctrl_frame = tk.Frame(self.window, bg="#121212")
        ctrl_frame.pack(fill=tk.X, padx=14, pady=8)

        self.btn_next = tk.Button(
            ctrl_frame,
            text="▶ Next Step",
            bg="#0A84FF",
            fg="#FFFFFF",
            activebackground="#0066CC",
            activeforeground="#FFFFFF",
            font=self.btn_font,
            relief=tk.FLAT,
            cursor="hand2",
            command=self._on_next_clicked,
        )
        self.btn_next.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5), ipady=6)

        self.btn_auto = tk.Button(
            ctrl_frame,
            text="⚡ Auto-Play All",
            bg="#30D158",
            fg="#000000",
            activebackground="#28B14A",
            activeforeground="#000000",
            font=self.btn_font,
            relief=tk.FLAT,
            cursor="hand2",
            command=self.toggle_autoplay,
        )
        self.btn_auto.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5, ipady=6)

        self.btn_reset = tk.Button(
            ctrl_frame,
            text="↺ Reset",
            bg="#3A3A3C",
            fg="#F5F5F7",
            activebackground="#48484A",
            activeforeground="#FFFFFF",
            font=self.btn_font,
            relief=tk.FLAT,
            cursor="hand2",
            command=self._on_reset_clicked,
        )
        self.btn_reset.pack(side=tk.LEFT, ipadx=14, padx=(5, 0), ipady=6)

        mode_bar = tk.Frame(self.window, bg="#121212")
        mode_bar.pack(fill=tk.X, padx=14, pady=(0, 6))
        tk.Label(
            mode_bar,
            text="MODE PREVIEW (or click status dot on Main Window):",
            bg="#121212",
            fg="#8E8E93",
            font=self.sub_font,
        ).pack(side=tk.LEFT, padx=(0, 6))

        def _make_mode_cmd(mk: str) -> Callable[[], None]:
            return lambda: self.on_mode_requested(mk)

        for m_key, m_label, m_fg in (
            (OperatingMode.GREEN.value, "🟢 Green", "#34C759"),
            (OperatingMode.YELLOW.value, "🟡 Yellow", "#FFCC00"),
            (OperatingMode.RED.value, "🔴 Red", "#FF453A"),
        ):
            m_btn = tk.Button(
                mode_bar,
                text=m_label,
                bg="#1C1C1E",
                fg=m_fg,
                activebackground="#2C2C2E",
                activeforeground=m_fg,
                font=self.btn_font,
                relief=tk.FLAT,
                cursor="hand2",
                command=_make_mode_cmd(m_key),
            )
            m_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2, ipady=3)
            self._mode_btns[m_key] = m_btn

        self.lbl_headline = tk.Label(
            self.window,
            text="",
            bg="#242426",
            fg="#FFD60A",
            font=self.btn_font,
            wraplength=535,
            justify="left",
            anchor="w",
            padx=10,
            pady=8,
        )
        self.lbl_headline.pack(fill=tk.X, padx=14, pady=(2, 6))

        log_box = tk.Frame(self.window, bg="#1C1C1E", highlightbackground="#3A3A3C", highlightthickness=1)
        log_box.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 14))

        tk.Label(
            log_box,
            text="LIVE ENGINE TELEMETRY (ENGLISH INSPECTOR):",
            bg="#1C1C1E",
            fg="#8E8E93",
            font=self.btn_font,
        ).pack(anchor="w", padx=10, pady=(8, 2))

        self.lbl_telemetry = tk.Label(
            log_box,
            text="",
            bg="#1C1C1E",
            fg="#E5E5EA",
            font=self.mono_font,
            wraplength=530,
            justify="left",
            anchor="nw",
        )
        self.lbl_telemetry.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))

    def apply_result(self, res: SandboxStepResult) -> None:
        """Pushes step output to `OnmyojiHelperUI`, `AITacticalUI`, and the telemetry inspector."""
        self._ensure_ai_panel()
        self.ui_app.set_mode_ui(res.mode)
        self.ui_app.update_info(
            res.opp_ban,
            res.opp_draft,
            res.my_draft,
            res.my_recs,
            res.opp_preds,
            res.status_text,
        )
        self.ai_panel.update_advice(res.tactical_advice)

        for idx, btn in enumerate(self._scenario_btns):
            is_active = idx == res.scenario_index
            btn.config(
                bg="#2C3E50" if is_active else "#1C1C1E",
                fg="#4CC2FF" if is_active else "#F5F5F7",
            )

        for m_key, m_btn in self._mode_btns.items():
            m_btn.config(bg="#2C3E50" if m_key == res.mode else "#1C1C1E")

        step_badge = f"[Step {res.step_index + 1}/{res.total_steps}] {res.headline}"
        self.lbl_headline.config(text=step_badge)
        self.lbl_telemetry.config(text="\n".join(f"• {line}" for line in res.telemetry_lines))

    def _on_select_scenario(self, idx: int) -> None:
        self._stop_autoplay()
        self.apply_result(self.controller.select_scenario(idx))

    def _on_next_clicked(self) -> None:
        self._stop_autoplay()
        self.step_once()

    def _on_reset_clicked(self) -> None:
        self._stop_autoplay()
        self.apply_result(self.controller.reset())

    def step_once(self) -> SandboxStepResult:
        res = self.controller.next_step()
        self.apply_result(res)
        return res

    def toggle_autoplay(self) -> None:
        if self.auto_playing:
            self._stop_autoplay()
        else:
            self.auto_playing = True
            self.btn_auto.config(text="⏸ Pause Auto-Play", bg="#FF9F0A")
            self._autoplay_tick()

    def _stop_autoplay(self) -> None:
        self.auto_playing = False
        if self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except Exception as exc:
                logger.debug(f"[DemoSandbox] Autoplay timer cancel skipped: {exc}")
            self._after_id = None
        if self.winfo_exists():
            self.btn_auto.config(text="⚡ Auto-Play All", bg="#30D158")

    def _autoplay_tick(self) -> None:
        if not self.auto_playing or not self.winfo_exists():
            return
        self.step_once()
        self._after_id = self.root.after(1350, self._autoplay_tick)

    def close(self) -> None:
        self._stop_autoplay()
        if self.ai_panel.winfo_exists():
            self.ai_panel.destroy()
        if self.winfo_exists():
            self.window.destroy()
