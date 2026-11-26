"""Built-In Public Sandbox Playbooks & Shikigami Metadata for Offline Demo Mode.

Provides a realistic, self-contained dataset (zero private cloud credentials required) so
recruiters, interviewers, and open-source contributors can run `python run_app.py --demo`
out of the box after `git clone` and experience XMind counter-tree scoring, dead-branch
pivots, mirror-match detection, and 5v5 tactical targeting live.
"""

from typing import Any

PUBLIC_DEMO_FULL_TO_SHORT: dict[str, str] = {
    "封陽君": "封",
    "不知火": "離",
    "天照": "神",
    "神啟荒": "神荒",
    "龍吟鈴鹿御前": "SP06",
    "石長姬": "石",
    "大夜摩天閻魔": "閻",
    "福悅座敷童子": "火",
    "葛葉": "葛",
    "帝釋天": "帝",
    "荒骷髏": "骷",
    "一目連": "連",
    "酒吞童子": "酒吞",
    "百羽鳳凰火": "百羽",
    "流光追月神": "追月",
    "須佐之男": "須",
    "千姬": "千",
    "神釀星熊童子": "SP熊",
    "渺念螢草": "SP草",
    "紡願緣結神": "SP緣",
    "心狩鬼女紅葉": "SP葉",
    "言靈": "言",
    "季": "季",
    "瀧": "瀧",
    "荒": "荒",
    "灼華桃花妖": "SP桃",
    "初音未來": "玲",
}

PUBLIC_DEMO_MULTI_SHORTS: list[str] = [
    "SP06",
    "神荒",
    "酒吞",
    "百羽",
    "追月",
    "SP熊",
    "SP草",
    "SP緣",
    "SP葉",
    "SP桃",
]

PUBLIC_DEMO_EN_GLOSSARY: dict[str, str] = {
    "封": "Fengyangjun",
    "離": "Shiranui",
    "神": "Amaterasu",
    "神荒": "SP Susabi",
    "SP06": "SP Otakemaru",
    "石": "Iwannagahime",
    "閻": "SP Enma",
    "火": "SP Zashiki",
    "葛": "Kuzunoha",
    "帝": "Taishakuten",
    "骷": "Gashadokuro",
    "連": "Ichimokuren",
    "酒吞": "Shuten Doji",
    "百羽": "SP Hououka",
    "追月": "SP Oitsuki",
    "須": "Susanoo",
    "千": "Senhime",
    "SP熊": "SP Hoshiguma",
    "SP草": "SP Kusa",
    "SP緣": "SP Enmusubi",
    "SP葉": "SP Momiji",
    "言": "Kotodama",
    "季": "Jinkougyou",
    "瀧": "Taki",
    "荒": "Susabi",
    "SP桃": "SP Momo",
    "玲": "Hatsune Miku",
    "禪": "Ungaikyo",
    "不相": "Mismatched",
}


def format_bilingual_token(token: str | None) -> str:
    """Formats a short shikigami token with its English name for recruiter-friendly telemetry."""
    if not token:
        return "None"
    en = PUBLIC_DEMO_EN_GLOSSARY.get(token)
    return f"{token} ({en})" if en else token


def format_bilingual_list(tokens: list[str]) -> str:
    """Formats a list of short shikigami tokens into a concise bilingual string."""
    if not tokens:
        return "(empty)"
    return ", ".join(format_bilingual_token(t) for t in tokens)

PUBLIC_DEMO_SYSTEMS: dict[str, dict[str, Any]] = {
    "Sandbox · Ban禪 Tempo Playbook": {
        "my_ban": "禪",
        "raw_data": "\n".join(
            [
                "Ban封",
                "離神葛連骷 神SP06石閻SP熊",
                "離百羽酒吞追月連 神SP06石連SP熊",
                "離火葛帝連 玲SP桃葛酒吞季",
                "Ban須",
                "千連SP06酒吞神荒 離酒吞千帝言",
                "追月神SP06葛連 離SP葉石閻季",
                "Ban禪",
                "離百羽酒吞SP06連 離石閻SP熊葛",
            ]
        ),
    },
    "Sandbox · Ban不相 Control Playbook": {
        "my_ban": "不相",
        "raw_data": "\n".join(
            [
                "Ban不相",
                "葛帝骷閻連 玲SP桃葛酒吞季",
                "離葛帝連神荒 神SP06石閻SP桃",
                "Ban封",
                "追月帝閻SP06連 離酒吞石葛SP熊",
            ]
        ),
    },
}

PUBLIC_DEMO_TAGS: dict[str, dict[str, Any]] = {
    "閻": {
        "role": "控制",
        "tag": "控制",
        "priority": 1,
        "static_priority": 1,
        "suggested_skill": "S3",
        "threat_mechanism": "Top-speed single-target silence & morph lock",
        "counter_weakness": "Shields / immunity layers or faster puller",
        "recommended_operation": "搶到一速→S3優先控敵Static_Priority=1核心（輸出/拉條）；無一速時S2干擾鬼火/節奏",
        "priority_target_roles": {
            "enemy_attack_pref": ["拉條", "控制", "輸出", "供火", "輔助"],
            "ally_buff_pref": ["輸出", "控制"],
        },
        "skill_details": {
            "S1": {"desc": "Normal attack", "scope": "敵方單體", "target_role_pref": ["控制", "輸出"]},
            "S2": {"desc": "Passive orb pressure", "scope": "被動", "target_role_pref": []},
            "S3": {"desc": "Single-target morph + silence", "scope": "敵方單體", "target_role_pref": ["拉條", "輸出", "控制"]},
            "S4": {"desc": "Self immunity stance", "scope": "自己", "target_role_pref": []},
        },
    },
    "玲": {
        "role": "輸出",
        "tag": "輸出",
        "priority": 1,
        "static_priority": 1,
        "suggested_skill": "S3",
        "threat_mechanism": "High-tempo action-bar pull & burst amplification",
        "counter_weakness": "Vulnerable to turn-1 hard morph/taunt lock",
        "recommended_operation": "開局優先施放S3拉條搶節奏",
        "priority_target_roles": {"enemy_attack_pref": ["輸出", "控制"], "ally_buff_pref": ["輸出"]},
        "skill_details": {
            "S1": {"desc": "Normal attack", "scope": "敵方單體", "target_role_pref": ["輸出"]},
            "S3": {"desc": "Team tempo boost", "scope": "我方全體", "target_role_pref": []},
        },
    },
    "神荒": {
        "role": "輸出",
        "tag": "輸出",
        "priority": 1,
        "static_priority": 1,
        "suggested_skill": "S3",
        "threat_mechanism": "Punishes non-basic skills with AoE starfall",
        "counter_weakness": "Burst down through shields or isolate orbs",
        "recommended_operation": "優先施放S3開啟星爆全體壓血線",
        "priority_target_roles": {"enemy_attack_pref": ["輸出", "控制"], "ally_buff_pref": ["輸出"]},
        "skill_details": {
            "S1": {"desc": "Normal attack", "scope": "敵方單體", "target_role_pref": ["輸出"]},
            "S2": {"desc": "Star array", "scope": "被動", "target_role_pref": []},
            "S3": {"desc": "Celestial AoE detonation", "scope": "敵方全體", "target_role_pref": []},
        },
    },
    "追月": {
        "role": "供火",
        "tag": "供火",
        "priority": 3,
        "static_priority": 3,
        "suggested_skill": "S3",
        "threat_mechanism": "Continuous orb supply + team ATK/SPD aura",
        "counter_weakness": "Dispel or turn-push suppression",
        "recommended_operation": "首輪施放S3為全隊供火加速，缺火時自動降級S1普攻",
        "priority_target_roles": {"enemy_attack_pref": ["控制", "輸出"], "ally_buff_pref": ["輸出"]},
        "skill_details": {
            "S1": {"desc": "Normal attack", "scope": "敵方單體", "target_role_pref": ["控制", "輸出"]},
            "S2": {"desc": "Lunar passive", "scope": "被動", "target_role_pref": []},
            "S3": {"desc": "All-ally orb & speed blessing", "scope": "我方全體", "target_role_pref": []},
        },
    },
    "葛": {
        "role": "控制",
        "tag": "控制",
        "priority": 1,
        "static_priority": 1,
        "suggested_skill": "S2",
        "threat_mechanism": "Fox mark link & turn disruption",
        "counter_weakness": "Cleanse / immunity shields",
        "recommended_operation": "開局首輪優先施放S2指定我方主力輸出解鎖S3",
        "priority_target_roles": {"enemy_attack_pref": ["控制", "輸出"], "ally_buff_pref": ["輸出", "控制"]},
        "skill_details": {
            "S1": {"desc": "Normal attack", "scope": "敵方單體", "target_role_pref": ["輸出"]},
            "S2": {"desc": "Ally fox-guard buff", "scope": "我方單體", "target_role_pref": ["輸出", "控制"]},
            "S3": {"desc": "Enemy AoE disruption", "scope": "敵方全體", "target_role_pref": []},
        },
    },
    "帝": {
        "role": "控制",
        "tag": "控制",
        "priority": 1,
        "static_priority": 1,
        "suggested_skill": "S2",
        "threat_mechanism": "Golden lotus mind-control hijack",
        "counter_weakness": "High-EFFRES cleanse or burst focus",
        "recommended_operation": "施放S2金蓮鎖定敵方核心控制或主力輸出",
        "priority_target_roles": {"enemy_attack_pref": ["控制", "輸出", "拉條"], "ally_buff_pref": ["輸出"]},
        "skill_details": {
            "S1": {"desc": "Normal attack", "scope": "敵方單體", "target_role_pref": ["控制", "輸出"]},
            "S2": {"desc": "Golden Lotus single-target control", "scope": "敵方單體", "target_role_pref": ["控制", "輸出"]},
            "S3": {"desc": "AoE lotus burst", "scope": "敵方全體", "target_role_pref": []},
        },
    },
    "SP06": {
        "role": "輸出",
        "tag": "輸出",
        "priority": 1,
        "static_priority": 1,
        "suggested_skill": "S3",
        "threat_mechanism": "Multi-hit AoE armor-piercing volley",
        "counter_weakness": "Shield stacking & single-target lockdown",
        "recommended_operation": "鬼火充足時施放S3群體壓制",
        "priority_target_roles": {"enemy_attack_pref": ["輸出", "控制"], "ally_buff_pref": ["輸出"]},
        "skill_details": {
            "S1": {"desc": "Normal attack", "scope": "敵方單體", "target_role_pref": ["輸出"]},
            "S3": {"desc": "Dragon chant AoE", "scope": "敵方全體", "target_role_pref": []},
        },
    },
    "SP桃": {
        "role": "輔助",
        "tag": "輔助",
        "priority": 2,
        "static_priority": 2,
        "suggested_skill": "S3",
        "threat_mechanism": "Team heal & revival sustain",
        "counter_weakness": "Anti-heal or CC before turn",
        "recommended_operation": "施放S3維持全隊血線與復活印記",
        "priority_target_roles": {"enemy_attack_pref": ["輸出"], "ally_buff_pref": ["輸出"]},
        "skill_details": {
            "S1": {"desc": "Normal attack", "scope": "敵方單體", "target_role_pref": ["輸出"]},
            "S3": {"desc": "All-ally heal & revive", "scope": "我方全體", "target_role_pref": []},
        },
    },
    "季": {
        "role": "輸出",
        "tag": "輸出",
        "priority": 1,
        "static_priority": 1,
        "suggested_skill": "S3",
        "threat_mechanism": "Seasonal field true-damage ticks on turn end",
        "counter_weakness": "Burst before field matures",
        "recommended_operation": "開局施放S3展開四時結界",
        "priority_target_roles": {"enemy_attack_pref": ["輸出", "控制"], "ally_buff_pref": ["輸出"]},
        "skill_details": {
            "S1": {"desc": "Normal attack", "scope": "敵方單體", "target_role_pref": ["輸出"]},
            "S3": {"desc": "Four-seasons AoE domain", "scope": "敵方全體", "target_role_pref": []},
        },
    },
    "連": {
        "role": "輔助",
        "tag": "輔助",
        "priority": 3,
        "static_priority": 3,
        "suggested_skill": "S3",
        "threat_mechanism": "Thick wind barriers absorbing AoE burst",
        "counter_weakness": "Ignore-shield damage or CC",
        "recommended_operation": "施放S3全體風盾保護全隊免受爆發秒殺",
        "priority_target_roles": {"enemy_attack_pref": ["輸出"], "ally_buff_pref": ["輸出"]},
        "skill_details": {
            "S1": {"desc": "Normal attack", "scope": "敵方單體", "target_role_pref": ["輸出"]},
            "S3": {"desc": "All-ally wind shield", "scope": "我方全體", "target_role_pref": []},
        },
    },
    "骷": {
        "role": "輸出",
        "tag": "輸出",
        "priority": 2,
        "static_priority": 2,
        "suggested_skill": "S3",
        "threat_mechanism": "Counter-bruiser frontline pressure",
        "counter_weakness": "Hard morph control",
        "recommended_operation": "施放S3壓制敵方脆皮核心",
        "priority_target_roles": {"enemy_attack_pref": ["輸出", "控制"], "ally_buff_pref": ["輸出"]},
        "skill_details": {
            "S1": {"desc": "Normal attack", "scope": "敵方單體", "target_role_pref": ["輸出"]},
            "S3": {"desc": "Heavy cleave", "scope": "敵方全體", "target_role_pref": []},
        },
    },
    "離": {
        "role": "輔助",
        "tag": "輔助",
        "priority": 2,
        "static_priority": 2,
        "suggested_skill": "S3",
        "threat_mechanism": "Double-basic-attack starlight realm",
        "counter_weakness": "Realm replacement",
        "recommended_operation": "維持星火結界增益普攻體系",
        "priority_target_roles": {"enemy_attack_pref": ["輸出"], "ally_buff_pref": ["輸出"]},
        "skill_details": {
            "S1": {"desc": "Normal attack", "scope": "敵方單體", "target_role_pref": ["輸出"]},
            "S3": {"desc": "Starfire realm", "scope": "我方全體", "target_role_pref": []},
        },
    },
    "酒吞": {
        "role": "輸出",
        "tag": "輸出",
        "priority": 2,
        "static_priority": 2,
        "suggested_skill": "S1",
        "threat_mechanism": "Zero-orb multi-stack single-target frenzy",
        "counter_weakness": "Dispel gourd stacks or taunt",
        "recommended_operation": "普攻S1零火連擊點殺敵方核心",
        "priority_target_roles": {"enemy_attack_pref": ["控制", "輸出"], "ally_buff_pref": ["輸出"]},
        "skill_details": {
            "S1": {"desc": "Multi-hit gourd strike", "scope": "敵方單體", "target_role_pref": ["控制", "輸出"]},
            "S3": {"desc": "Self heal & frenzy", "scope": "自己", "target_role_pref": []},
        },
    },
}


def get_public_demo_bundle() -> tuple[dict[str, Any], dict[str, str], list[str], dict[str, dict[str, Any]]]:
    """Returns a deep-copied `(systems, full_to_short_map, multi_shorts, tags_db)` sandbox bundle."""
    systems = {k: dict(v) for k, v in PUBLIC_DEMO_SYSTEMS.items()}
    dict_map = dict(PUBLIC_DEMO_FULL_TO_SHORT)
    multi_shorts = list(PUBLIC_DEMO_MULTI_SHORTS)
    tags_db = {k: dict(v) for k, v in PUBLIC_DEMO_TAGS.items()}
    return systems, dict_map, multi_shorts, tags_db
