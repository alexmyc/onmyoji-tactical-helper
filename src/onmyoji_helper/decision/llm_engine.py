"""Local Gemma-4 GGUF & Zero-VRAM Rule Tactical Engine for Onmyoji Tactical Helper.

Automatically profiles host VRAM and system RAM to choose between GPU-accelerated local
Gemma-4 inference (`llama-cpp-python`) and zero-latency deterministic rule advice.
"""

import ctypes
import gc
import json
import logging
import os
from typing import Any

from onmyoji_helper.infra.paths import resolve_asset_path

logger = logging.getLogger("OnmyojiHelper")

VALID_SLOTS = frozenset({"Slot 1", "Slot 2", "Slot 3", "Slot 4", "Slot 5"})
VALID_SKILLS = frozenset({"S1", "S2", "S3", "S4"})


def evaluate_hardware_tier() -> tuple[str, bool, str]:
    """Evaluates host RAM and NVIDIA CUDA VRAM to select GPU LLM vs. 0-VRAM Rule tier."""
    force_rule = os.getenv("FORCE_RULE_MODE", "0").lower() in ("1", "true", "yes")
    if force_rule:
        return "輕量規則模式 (手動強制)", False, "使用者設置環境變數 FORCE_RULE_MODE=1"

    total_ram_gb = 8.0
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            total_ram_gb = stat.ullTotalPhys / (1024 ** 3)
    except Exception as exc:
        logger.debug(f"[TacticalEngine] Host RAM probe fallback: {exc}")

    has_cuda = False
    vram_gb = 0.0
    gpu_name = "None"
    try:
        import torch

        if torch.cuda.is_available():
            has_cuda = True
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024 ** 3)
            gpu_name = torch.cuda.get_device_name(0)
    except Exception as exc:
        logger.debug(f"[TacticalEngine] CUDA probe skipped: {exc}")

    if not has_cuda:
        return "輕量規則模式", False, "未偵測到獨立 NVIDIA GPU (內顯環境，自動優化防止遊戲卡頓)"
    if vram_gb < 3.8:
        return "輕量規則模式", False, f"獨立顯卡 ({gpu_name}) 顯存僅 {vram_gb:.1f}GB (< 4GB)，避免顯存溢出"
    if total_ram_gb < 7.5:
        return "輕量規則模式", False, f"系統主記憶體僅 {total_ram_gb:.1f}GB (< 8GB)，優先保障遊戲運行記憶體"

    return "GPU 加速模式", True, f"顯卡: {gpu_name} ({vram_gb:.1f}GB VRAM), 主記憶體: {total_ram_gb:.1f}GB"


class TacticalEngine:
    """Tactical inference engine powered by local Gemma-4 GGUF or zero-VRAM rule fallback."""

    def __init__(
        self,
        model_filename: str = "gemma-4-E2B-it-Q4_K_M.gguf",
        model_dir: str = "",
        force_rule_only: bool = False,
    ) -> None:
        """Initializes hardware tier evaluation and loads GGUF weights if present."""
        self.model_filename: str = model_filename
        self.model_dir: str = model_dir
        self.force_rule_only: bool = force_rule_only
        self.llm: Any = None
        self.device_mode: str = "CPU"
        self._init_llm()

    def _init_llm(self) -> None:
        if self.force_rule_only:
            self.llm = None
            self.device_mode = "輕量規則模式 (Sandbox Instant)"
            return
        _, can_load_llm, reason = evaluate_hardware_tier()
        if not can_load_llm:
            self.llm = None
            self.device_mode = "輕量規則模式 (0顯存/0延遲)"
            logger.info(f"⚡ [Hardware Tier] 已啟動【輕量規則模式】：{reason}。以 0 顯存與零延遲提供精確戰術指示。")
            return

        primary_candidate = resolve_asset_path(self.model_filename, extra_dir=self.model_dir)
        fallback_candidate = resolve_asset_path("mini_llm.gguf", extra_dir=self.model_dir)
        model_path: str | None = None
        for candidate in (primary_candidate, fallback_candidate):
            if os.path.isfile(candidate):
                model_path = candidate
                break

        if not model_path:
            self.llm = None
            self.device_mode = "輕量規則模式 (無權重備援)"
            logger.info(
                f"⚠️ [LLM Diagnosis] 未找到大模型權重檔 '{self.model_filename}'，無縫切換為智慧規則備援模式。"
            )
            return

        try:
            from llama_cpp import Llama

            self.llm = Llama(
                model_path=model_path,
                n_gpu_layers=99,
                n_ctx=2048,
                n_threads=4,
                verbose=False,
            )
            self.device_mode = "GPU 加速模式"
            logger.info(f"🎉 [LLM Diagnosis] 成功加載 Gemma 4 ({self.device_mode})！硬體配置: {reason}")
        except Exception as exc:
            logger.warning(f"⚠️ [LLM Diagnosis] GPU 模型加載異常 ({exc})，安全切換為輕量規則模式。")
            self.llm = None
            self.device_mode = "輕量規則模式 (GPU失敗備援)"

    def release(self) -> None:
        """Releases the Llama instance and reclaims memory."""
        if self.llm is not None:
            try:
                del self.llm
            except Exception as exc:
                logger.debug(f"[TacticalEngine] LLM release cleanup note: {exc}")
            self.llm = None
            gc.collect()

    def get_tactical_advice(self, payload: dict[str, Any], mode: str = "LLM") -> dict[str, Any]:
        """Generates concise tactical advice via Gemma-4 chat completion or deterministic rules."""
        target_slot = payload.get("recommended_target_slot", "Slot 1")
        target_name = str(payload.get("recommended_target_name", "敵方單位"))
        target_tag = str(payload.get("recommended_target_tag", "單位"))

        if self.llm is None or mode != "LLM":
            return self._rule_based_advice(payload, target_slot, target_name, target_tag)

        try:
            default_skill = str(payload.get("chosen_skill", "S1")).upper()
            avail_map = payload.get("available_skills") or {}
            shiki_skills = payload.get("active_shikigami_skills") or {}

            # Structured context including acting shikigami, S1..S4 skill definitions, availability, and board state
            compact_payload = {
                "is_my_turn": payload.get("is_my_turn"),
                "active_shikigami_slot": payload.get("active_shikigami_slot"),
                "active_shikigami_name": payload.get("active_shikigami_name"),
                "active_shikigami_operation": payload.get("active_shikigami_operation"),
                "active_shikigami_skills": shiki_skills,
                "chosen_skill": default_skill,
                "skill_scope": payload.get("skill_scope"),
                "available_skills": avail_map,
                "my_player_team": payload.get("my_player_team"),
                "enemy_target": payload.get("enemy_target"),
                "opp_alive_units": payload.get("opp_alive_units"),
            }

            system_instruction = (
                "你是《陰陽師》頂級戰術指揮官。你的任務是根據「當前行動式神」(active_shikigami_name)、"
                "「可用技能」(available_skills) 與「敵我場上陣容」，決定最佳施放技能與目標。\n\n"
                "【最高原則】：\n"
                "1. 必須嚴格遵照當前行動式神的【連招建議】(active_shikigami_operation) 與【技能範疇】(active_shikigami_skills)。\n"
                "2. chosen_skill 只能從 available_skills 為 true 且非被動的技能中選擇（S1/S2/S3/S4）。\n"
                "3. 若技能範疇為「我方單體」，目標必須為我方單位；若為「敵方單體」，目標為敵方單位。絕對不可混淆！\n"
                "4. 術語白名單：僅限使用 [S1普攻]、[S2技能]、[S3大招]、[S4技能]。絕對禁止出現「奧義」、「必殺技」！\n"
                "5. 我方回合 (is_my_turn == True)：結構必須為：[當前行動者/技能] + [目標位置與名字] + [精練對策/連招提示]。控制在 35 字以內。\n"
                "6. 敵方回合 (is_my_turn == False)：簡述敵方核心威脅與對策。\n\n"
                "【輸出 JSON 格式】：\n"
                "{\n"
                f'  "chosen_skill": "{default_skill}",\n'
                f'  "target_slot": "{target_slot}",\n'
                '  "advice": "戰術提示 (35字以內)"\n'
                "}"
            )

            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": json.dumps(compact_payload, ensure_ascii=False)},
            ]

            response = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=256,
                temperature=0.1,
            )
            raw_text = response["choices"][0]["message"]["content"].strip()

            parsed_slot = target_slot
            parsed_skill = default_skill
            clean_advice = ""
            start_idx = raw_text.find("{")
            end_idx = raw_text.rfind("}") + 1
            if start_idx != -1 and end_idx > start_idx:
                res_json = json.loads(raw_text[start_idx:end_idx])
                candidate_slot = res_json.get("target_slot", target_slot)
                if candidate_slot in VALID_SLOTS or candidate_slot is None:
                    parsed_slot = candidate_slot
                cand_skill = str(res_json.get("chosen_skill", default_skill)).strip().upper()
                if cand_skill in VALID_SKILLS and bool(avail_map.get(cand_skill.lower(), False)):
                    sk_meta = shiki_skills.get(cand_skill, {}) or shiki_skills.get(cand_skill.lower(), {})
                    sk_scope = str(sk_meta.get("scope", ""))
                    if cand_skill == "S1" or not any(p in sk_scope for p in ("被動", "無")):
                        parsed_skill = cand_skill
                clean_advice = str(res_json.get("advice", "")).strip()

            junk_keywords = (
                "50字內",
                "核心對策",
                "精練核心",
                "請輸出",
                "格式如下",
                "戰術對策：",
                "提示",
                "奧義",
                "必殺技",
                "指揮「",
            )
            is_broken_json = (
                clean_advice.startswith("{")
                or "target" in clean_advice
                or "advice" in clean_advice
            )
            if not clean_advice or is_broken_json or any(kw in clean_advice for kw in junk_keywords):
                clean_advice = self._build_dynamic_fallback(payload, target_name, target_tag)

            return {
                "mode": f"Gemma-4 ({self.device_mode})",
                "chosen_skill": parsed_skill,
                "target_slot": parsed_slot,
                "target_name": target_name,
                "target_tag": target_tag,
                "advice": clean_advice,
            }
        except Exception as exc:
            logger.debug(f"⚠️ [LLM Engine] 推論異常，自動切換規則備援: {exc}")
            return self._rule_based_advice(payload, target_slot, target_name, target_tag)

    @staticmethod
    def _build_dynamic_fallback(payload: dict[str, Any], target_name: str, target_tag: str) -> str:
        fallback_msg = str(payload.get("fallback_advice", ""))
        if fallback_msg:
            return fallback_msg

        is_my_turn = bool(payload.get("is_my_turn", False))
        active_slot = payload.get("active_shikigami_slot")
        active_unit_name = payload.get("active_shikigami_name", "")
        skills = payload.get("available_skills", {})
        vote_order = payload.get("recommended_skill_order", ["s3", "s2", "s1"])
        enemy_target = payload.get("enemy_target", {})
        threat = enemy_target.get("threat", "")
        weakness = enemy_target.get("weakness", "")

        if is_my_turn:
            chosen_skill = next((s for s in vote_order if skills.get(s, False)), None)
            if not chosen_skill:
                chosen_skill = next((s for s in ("s3", "s2", "s1", "s4") if skills.get(s, False)), "s1")
            name_map = {"s1": "S1 普攻", "s2": "S2 技能", "s3": "S3 大招", "s4": "S4 技能"}
            skill_text = name_map.get(chosen_skill, "S1 普攻")

            action_tip = "集中火力壓制，保持陣型優勢"
            if "控制" in target_tag:
                action_tip = "優先硬控封印，防止打亂節奏"
            elif "輸出" in target_tag:
                action_tip = "集中火力秒殺，剪除敵方主力"
            elif "拉條" in target_tag or "供火" in target_tag:
                action_tip = "壓制推條，斷絕鬼火與行動條"

            turn_prefix = (
                f"我方 [{active_slot}] {active_unit_name} 回合！"
                if active_slot and active_unit_name
                else "我方回合！"
            )
            return f"{turn_prefix}建議點擊 [{skill_text}] 打擊 [{target_name}]，{action_tip}。"

        if threat and weakness:
            return f"【敵方回合】留意 [{target_name}] 威脅：{threat}；弱點：{weakness}。"
        if threat:
            return f"【敵方回合】留意 [{target_name}] 核心機制：{threat}。"
        return f"【敵方回合】等待行動，預計優先壓制 [{target_name}]（{target_tag}）。"

    def _rule_based_advice(
        self,
        payload: dict[str, Any],
        target_slot: str | None,
        target_name: str,
        target_tag: str,
    ) -> dict[str, Any]:
        return {
            "mode": self.device_mode,
            "chosen_skill": str(payload.get("chosen_skill", "S1")).upper(),
            "target_slot": target_slot,
            "target_name": target_name,
            "target_tag": target_tag,
            "advice": self._build_dynamic_fallback(payload, target_name, target_tag),
        }
