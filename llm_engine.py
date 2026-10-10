# -*- coding: utf-8 -*-
"""LLM Tactical Engine Module for Onmyoji Tactical Helper.

This module manages local LLM inference using Gemma 4 via llama-cpp-python,
handling GPU/CPU hardware acceleration, chat completions, response JSON parsing,
and rule-based fallback generation.
"""

import gc
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional


class TacticalEngine:
    """Tactical inference engine powered by local LLM or rule-based fallback logic."""

    def __init__(
        self,
        model_filename: str = "gemma-4-E2B-it-Q4_K_M.gguf",
        model_dir: str = ""
    ) -> None:
        """Initializes the tactical engine, searches for GGUF model files, and loads the model.

        Args:
            model_filename: Filename of the GGUF quant model.
            model_dir: Optional additional directory to search for the model.
        """
        self.model_filename: str = model_filename
        self.model_dir: str = model_dir
        self.llm: Any = None
        self.device_mode: str = "CPU"
        self._init_llm()

    def _init_llm(self) -> None:
        """Locates the GGUF model file and initializes llama-cpp-python with GPU or CPU execution."""
        base_dir = os.path.dirname(
            os.path.abspath(sys.executable if getattr(sys, 'frozen', False) else __file__)
        )

        possible_paths = [
            os.path.join(base_dir, self.model_filename),
            os.path.join(base_dir, "models", self.model_filename),
            os.path.join(self.model_dir, self.model_filename) if self.model_dir else "",
            os.path.join(base_dir, "mini_llm.gguf"),
            os.path.join(base_dir, "models", "mini_llm.gguf")
        ]
        possible_paths = [p for p in possible_paths if p]

        model_path: Optional[str] = None
        for p in possible_paths:
            if os.path.exists(p):
                model_path = p
                break

        print("\n🔍 [LLM Diagnosis] Checking Gemma 4 model file path...")

        if not model_path:
            print(f"⚠️ [LLM Diagnosis] Model file '{self.model_filename}' not found! Fully enabling smart rule fallback.")
            return

        print(f"✅ [LLM Diagnosis] Successfully located model file: {model_path}")

        try:
            from llama_cpp import Llama
            self.llm = Llama(
                model_path=model_path,
                n_gpu_layers=99,
                n_ctx=2048,
                n_threads=4,
                verbose=False
            )
            self.device_mode = "GPU Acceleration"
            print("🎉 [LLM Diagnosis] Successfully initialized Gemma 4 (GPU Hardware Acceleration)!")
        except Exception as e:
            print(f"⚠️ [LLM Diagnosis] GPU initialization failed ({e}). Attempting fallback to CPU mode...")
            try:
                from llama_cpp import Llama
                self.llm = Llama(
                    model_path=model_path,
                    n_gpu_layers=0,
                    n_ctx=2048,
                    n_threads=4,
                    verbose=False
                )
                self.device_mode = "CPU"
                print("⚙️ [LLM Diagnosis] Successfully initialized Gemma 4 (CPU Mode)!")
            except Exception:
                self.llm = None
                print("❌ [LLM Diagnosis] Model initialization failed. Switching to smart Rule fallback mode.")

    def release(self) -> None:
        """Safely destroys the Llama instance and collects garbage to free VRAM/RAM."""
        if self.llm is not None:
            try:
                del self.llm
            except Exception:
                pass
            self.llm = None
            gc.collect()

    def get_tactical_advice(self, payload: Dict[str, Any], mode: str = "LLM") -> Dict[str, Any]:
        """Generates real-time tactical advice using Gemma 4 chat completion or rule fallback.

        Args:
            payload: Structured battle context payload.
            mode: Operational mode string ("LLM" or "Rule").

        Returns:
            Dictionary containing inference mode, target slot, target name, tag, and advice text.
        """
        target_slot = payload.get("recommended_target_slot", "Slot 1")
        target_name = payload.get("recommended_target_name", "敵方單位")
        target_tag = payload.get("recommended_target_tag", "單位")

        if self.llm is None or mode != "LLM":
            return self._rule_based_advice(payload, target_slot, target_name, target_tag)

        try:
            active_slot = payload.get("active_shikigami_slot")
            active_unit_name = payload.get("active_shikigami_name", "")

            system_instruction = (
                "你是《陰陽師》頂級戰術指揮官。你的任務是指導「我方玩家」進行精準操作。\n\n"
                "【最高原則】：\n"
                "1. 必須嚴格遵照當前行動式神的【連招建議】(active_shikigami_operation) 與【技能範疇】(scope)。\n"
                "2. 若技能範疇為「我方單體」，目標必須為我方單位；若為「敵方單體」，目標為敵方單位。絕對不可混淆！\n"
                "3. 術語白名單：僅限使用 [S1普攻]、[S2技能]、[S3大招]、[S4技能]。絕對禁止出現「奧義」、「必殺技」！\n"
                "4. 我方回合 (is_my_turn == True)：結構必須為：[當前行動者/技能] + [目標位置與名字] + [精練對策/連招提示]。控制在 35 字以內。\n"
                "5. 敵方回合 (is_my_turn == False)：簡述敵方核心威脅與對策。\n\n"
                "【Few-Shot 範例 1 (敵方打擊技能)】\n"
                'User: {"is_my_turn": true, "active_shikigami_slot": "Slot 3", "active_shikigami_name": "須佐之男", "available_skills": {"s3": true}, "enemy_target": {"slot": "Slot 2", "name": "帝釋天", "role": "控制"}}\n'
                'Assistant: {"target_slot": "Slot 2", "advice": "我方 [Slot 3] 須佐之男回合！建議使用 [S3大招] 集火敵方 [Slot 2] 帝釋天，優先硬控封印。"}\n\n'
                "【Few-Shot 範例 2 (我方增益/護盾技能)】\n"
                'User: {"is_my_turn": true, "active_shikigami_slot": "Slot 1", "active_shikigami_name": "葛葉", "active_shikigami_operation": "開局首輪優先施放 S2 指定我方最高優先級單位以解鎖 S3", "available_skills": {"s2": true}}\n'
                'Assistant: {"target_slot": "Slot 3", "advice": "我方 [Slot 1] 葛葉回合！建議使用 [S2技能] 指定我方 [Slot 3] 主力單位，解鎖三技能。"}\n\n'
                "【Few-Shot 範例 3 (敵方回合)】\n"
                'User: {"is_my_turn": false, "enemy_target": {"slot": "Slot 3", "name": "須佐之男", "role": "輸出", "threat": "天雷單體秒殺", "weakness": "怕硬控"}}\n'
                'Assistant: {"target_slot": "Slot 3", "advice": "【敵方回合】留意 [須佐之男] 核心威脅：天雷單體秒殺；弱點：怕硬控。"}\n\n'
                "【輸出 JSON 格式】：\n"
                "{\n"
                f'  "target_slot": "{target_slot}",\n'
                '  "advice": "戰術提示 (35字以內)"\n'
                "}"
            )

            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
            ]

            response = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=256,
                temperature=0.1
            )

            raw_text = response["choices"][0]["message"]["content"].strip()

            parsed_slot = target_slot
            clean_advice = ""

            # Extract JSON block safely
            try:
                start_idx = raw_text.find('{')
                end_idx = raw_text.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = raw_text[start_idx:end_idx]
                    res_json = json.loads(json_str)
                    parsed_slot = res_json.get("target_slot", target_slot)
                    clean_advice = str(res_json.get("advice", "")).strip()
                else:
                    clean_advice = ""
            except Exception:
                clean_advice = ""

            # Validate response string against forbidden artifacts and junk phrases
            junk_keywords = [
                "50字內", "核心對策", "精練核心", "請輸出", "格式如下",
                "戰術對策：", "提示", "奧義", "必殺技", "指揮「"
            ]
            is_broken_json = (
                clean_advice.startswith("{")
                or "target" in clean_advice
                or "advice" in clean_advice
            )

            if not clean_advice or is_broken_json or any(kw in clean_advice for kw in junk_keywords):
                clean_advice = self._build_dynamic_fallback(payload, target_name, target_tag)

            return {
                "mode": f"Gemma-4 ({self.device_mode})",
                "target_slot": parsed_slot,
                "target_name": target_name,
                "target_tag": target_tag,
                "advice": clean_advice
            }

        except Exception:
            return self._rule_based_advice(payload, target_slot, target_name, target_tag)

    def _build_dynamic_fallback(
        self,
        payload: Dict[str, Any],
        target_name: str,
        target_tag: str
    ) -> str:
        """Constructs fallback tactical advice if LLM inference is unavailable or produces invalid JSON.

        Args:
            payload: Battle context dictionary.
            target_name: Name of the primary target.
            target_tag: Role/Tag of the primary target.

        Returns:
            Formatted fallback tactical advice string in Traditional Chinese.
        """
        fallback_msg = payload.get("fallback_advice", "")
        if fallback_msg:
            return fallback_msg

        is_my_turn = payload.get("is_my_turn", False)
        active_slot = payload.get("active_shikigami_slot")
        active_unit_name = payload.get("active_shikigami_name", "")
        skills = payload.get("available_skills", {})
        vote_order = payload.get("recommended_skill_order", ["s3", "s2", "s1"])
        enemy_target = payload.get("enemy_target", {})

        threat = enemy_target.get("threat", "")
        weakness = enemy_target.get("weakness", "")

        if is_my_turn:
            chosen_skill = None
            for s in vote_order:
                if skills.get(s, False):
                    chosen_skill = s
                    break

            if not chosen_skill:
                for s in ["s3", "s2", "s1", "s4"]:
                    if skills.get(s, False):
                        chosen_skill = s
                        break

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
        else:
            if threat and weakness:
                return f"【敵方回合】留意 [{target_name}] 威脅：{threat}；弱點：{weakness}。"
            elif threat:
                return f"【敵方回合】留意 [{target_name}] 核心機制：{threat}。"
            else:
                return f"【敵方回合】等待行動，預計優先壓制 [{target_name}]（{target_tag}）。"

    def _rule_based_advice(
        self,
        payload: Dict[str, Any],
        target_slot: str,
        target_name: str,
        target_tag: str
    ) -> Dict[str, Any]:
        """Generates a rule-based advice dictionary response.

        Args:
            payload: Battle context dictionary.
            target_slot: Target slot key identifier.
            target_name: Primary target character name.
            target_tag: Primary target role/tag string.

        Returns:
            Dictionary payload for rule-based advice output.
        """
        advice_text = self._build_dynamic_fallback(payload, target_name, target_tag)
        return {
            "mode": "Rule (Smart Fallback)",
            "target_slot": target_slot,
            "target_name": target_name,
            "target_tag": target_tag,
            "advice": advice_text
        }