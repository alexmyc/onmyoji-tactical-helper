"""XMind Tree-Structured Draft Decision & Opponent Prediction Engine.

Models draft playbooks as Ban-partitioned counter-pick trees:
1. Ally lineups (`my[0..4]`) act as ordered 5-step sequential counter-chains.
2. Opponent lineups (`set(opp)`) act as order-independent opening formation sets.
3. Enforces a global strict availability mask (`opp_ban | my_ban | my_picks | opp_picks`).
4. Penalizes broken branches whose remaining core cards were banned or stolen.
5. Detects mirror/copycat matches where the opponent drafts our own `my` backbone.
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class DraftScoringConfig:
    """Immutable hyperparameter configuration for XMind tree scoring and prediction."""

    max_recommendations: int = 5
    max_predictions: int = 5

    primary_ban_mult: float = 1.00
    fallback_my_ban_mult: float = 0.85
    cross_ban_mult: float = 0.55

    penalty_my_banned_per_card: float = 0.12
    penalty_my_stolen_per_card: float = 0.18
    penalty_immediate_next_dead: float = 0.35
    penalty_opp_contains_banned: float = 0.55

    pred_primary_ban_mult: float = 1.15
    pred_cross_ban_mult: float = 0.70
    pred_mirror_my_ban_mult: float = 1.20
    pred_mirror_opp_ban_mult: float = 1.05
    pred_mirror_other_ban_mult: float = 0.90
    pred_dead_banned_penalty: float = 0.20
    pred_dead_taken_penalty: float = 0.45

    agg_top_tier_ratio: float = 0.50


DEFAULT_SCORING_CONFIG = DraftScoringConfig()


def _normalize_ban_name(ban_str: str | None) -> str:
    """Strips optional 'Ban' prefix and whitespace from a Ban identifier."""
    if not ban_str:
        return ""
    cleaned = str(ban_str).strip()
    if cleaned.lower().startswith("ban"):
        cleaned = cleaned[3:].strip()
    return cleaned


def parse_draft_string(draft_str: str, multi_char_shorts: Sequence[str], _presorted: bool = False) -> list[str]:
    """Tokenizes a continuous lineup shorthand string into individual character tokens."""
    ordered_multi = (
        multi_char_shorts if _presorted else sorted(multi_char_shorts, key=len, reverse=True)
    )
    result: list[str] = []
    i = 0
    n = len(draft_str)
    while i < n:
        matched = False
        for mc in ordered_multi:
            if draft_str.startswith(mc, i):
                result.append(mc)
                i += len(mc)
                matched = True
                break
        if not matched:
            result.append(draft_str[i])
            i += 1
    return result


def build_database(
    raw_data: str,
    multi_char_shorts: Sequence[str],
) -> dict[str, list[dict[str, list[str]]]]:
    """Compiles raw multiline Ban/lineup text into an indexed tree dictionary."""
    sorted_multi = tuple(sorted(set(multi_char_shorts), key=len, reverse=True))
    db: dict[str, list[dict[str, list[str]]]] = {}
    current_ban: str | None = None

    for raw_line in raw_data.strip().split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Ban"):
            current_ban = line
            db.setdefault(current_ban, [])
        elif current_ban:
            parts = line.split()
            if len(parts) >= 2:
                db[current_ban].append(
                    {
                        "my": parse_draft_string(parts[0], sorted_multi, _presorted=True),
                        "opp": parse_draft_string(parts[1], sorted_multi, _presorted=True),
                    }
                )
            elif len(parts) == 1:
                db[current_ban].append(
                    {
                        "my": parse_draft_string(parts[0], sorted_multi, _presorted=True),
                        "opp": [],
                    }
                )
    return db


def _score_my_backbone(my_rec: list[str], my_picks: list[str]) -> tuple[float, bool]:
    """Scores how closely `my_picks` follows the ordered counter-pick backbone `my_rec`."""
    k = len(my_picks)
    if k == 0:
        return 40.0, True
    if len(my_rec) >= k and my_rec[:k] == my_picks:
        return 160.0 + (40.0 * k), True

    my_picks_set = set(my_picks)
    my_rec_set = set(my_rec)
    set_overlap = len(my_picks_set & my_rec_set)
    if set_overlap == 0:
        return 0.0, False

    prefix_exact = sum(1 for idx in range(min(k, len(my_rec))) if my_rec[idx] == my_picks[idx])
    if set_overlap == k:
        return 120.0 + (30.0 * k) + (15.0 * prefix_exact), False

    subset_ratio = set_overlap / float(k)
    return (set_overlap * 45.0) + (prefix_exact * 20.0) + (subset_ratio * 30.0), False


def _score_opp_unordered(opp_rec: list[str], opp_picks_set: set[str]) -> tuple[float, int]:
    """Scores order-independent set overlap between revealed `opp_picks` and `opp_rec`."""
    if not opp_picks_set or not opp_rec:
        return 0.0, 0
    opp_rec_set = set(opp_rec)
    hits = len(opp_picks_set & opp_rec_set)
    if hits == 0:
        return 0.0, 0
    precision = hits / float(len(opp_picks_set))
    recall = hits / float(len(opp_rec_set))
    return (hits * 140.0) + (precision * 110.0) + (recall * 40.0), hits


def _rank_candidates(
    score_map: dict[str, float],
    locked_unavailable: set[str],
    limit: int,
) -> list[str]:
    """Applies the global strict availability mask and sorts candidates descending by score."""
    ranked = [
        (card, sc)
        for card, sc in score_map.items()
        if card and sc > 0.0 and card not in locked_unavailable
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return [card for card, _ in ranked[:limit]]


def _cold_start_recommendation(
    db: dict[str, list[dict[str, list[str]]]],
    active_ban_key: str,
    is_fallback: bool,
    banned_cards: set[str],
    locked_unavailable: set[str],
    cfg: DraftScoringConfig,
) -> tuple[list[str], list[str], str]:
    """Computes Turn-1 opener recommendations and likely enemy openers right after Ban phase."""
    records = db.get(active_ban_key, [])
    used_all_db = False
    if not records:
        records = [rec for recs in db.values() for rec in recs]
        used_all_db = True

    if not records:
        return ["無匹配"], ["未知陣容"], "未收錄此 Ban 位，請手動盲翻"

    my_scores: dict[str, float] = defaultdict(float)
    opp_scores: dict[str, float] = defaultdict(float)

    for rec in records:
        my_rec = rec.get("my", [])
        opp_rec = rec.get("opp", [])
        if not my_rec:
            continue

        ban_conflicts_my = sum(1 for c in my_rec if c in banned_cards)
        ban_conflicts_opp = sum(1 for c in opp_rec if c in banned_cards)
        branch_w = (cfg.penalty_my_banned_per_card ** ban_conflicts_my) * (
            cfg.penalty_opp_contains_banned if ban_conflicts_opp > 0 else 1.0
        )

        if my_rec[0] not in locked_unavailable:
            my_scores[my_rec[0]] += 100.0 * branch_w
        else:
            for c in my_rec[1:]:
                if c not in locked_unavailable:
                    my_scores[c] += 25.0 * branch_w
                    break

        for idx, c in enumerate(opp_rec):
            if c not in locked_unavailable:
                pos_w = 1.4 if idx == 0 else 0.7
                opp_scores[c] += 50.0 * branch_w * pos_w

    final_my_recs = _rank_candidates(my_scores, locked_unavailable, cfg.max_recommendations)
    final_opp_preds = _rank_candidates(opp_scores, locked_unavailable, cfg.max_predictions)
    if not final_my_recs:
        final_my_recs = ["暫無數據匹配"]

    if used_all_db:
        status_msg = "跨庫首手盲翻"
    elif is_fallback:
        status_msg = f"啟用默認 ({active_ban_key}) 首手盲翻"
    else:
        status_msg = f"常規 {active_ban_key} 首手盲翻"

    return final_my_recs, final_opp_preds, status_msg


def get_recommendation(
    db: dict[str, list[dict[str, list[str]]]],
    current_ban: str | None,
    opp_draft: list[str],
    my_draft: list[str] | None = None,
    my_ban: str = "禪",
    config: DraftScoringConfig = DEFAULT_SCORING_CONFIG,
) -> tuple[list[str], list[str], str]:
    """Calculates XMind tree-weighted next-pick recommendations and opponent predictions."""
    my_picks: list[str] = [c for c in (my_draft or []) if c]
    opp_picks: list[str] = [c for c in (opp_draft or []) if c]
    my_picks_set: set[str] = set(my_picks)
    opp_picks_set: set[str] = set(opp_picks)

    if len(my_picks) >= 5:
        return ["【武運亨通，旗開得勝】"], [], "選卡結束，準備戰鬥"

    opp_ban_card = _normalize_ban_name(current_ban)
    my_ban_card = _normalize_ban_name(my_ban)
    banned_cards: set[str] = {c for c in (opp_ban_card, my_ban_card) if c}
    locked_unavailable: set[str] = banned_cards | my_picks_set | opp_picks_set

    opp_ban_key = f"Ban{opp_ban_card}" if opp_ban_card else ""
    my_ban_key = f"Ban{my_ban_card}" if my_ban_card else ""

    has_primary_ban = bool(opp_ban_key and opp_ban_key in db and db[opp_ban_key])
    active_ban_key = opp_ban_key if has_primary_ban else my_ban_key
    is_fallback = not has_primary_ban

    if len(my_picks) == 0 and len(opp_picks) == 0:
        return _cold_start_recommendation(
            db=db,
            active_ban_key=active_ban_key,
            is_fallback=is_fallback,
            banned_cards=banned_cards,
            locked_unavailable=locked_unavailable,
            cfg=config,
        )

    step = len(my_picks)
    rec_branch_candidates: list[tuple[float, str, bool, bool]] = []
    best_rec_weight: float = 0.0

    for ban_key, records in db.items():
        is_primary_ban = (ban_key == opp_ban_key) if has_primary_ban else (ban_key == my_ban_key)
        if has_primary_ban:
            ban_mult = config.primary_ban_mult if is_primary_ban else config.cross_ban_mult
        else:
            ban_mult = config.fallback_my_ban_mult if is_primary_ban else config.cross_ban_mult

        for rec in records:
            my_rec = rec.get("my", [])
            opp_rec = rec.get("opp", [])
            if not my_rec:
                continue

            remaining_my = [c for c in my_rec if c not in my_picks_set]
            available_remaining_my = [c for c in remaining_my if c not in locked_unavailable]
            if not available_remaining_my:
                continue

            backbone_score, is_exact_prefix = _score_my_backbone(my_rec, my_picks)
            opp_score, opp_hits = _score_opp_unordered(opp_rec, opp_picks_set)

            if opp_picks_set and opp_hits == 0 and not is_primary_ban:
                continue

            raw_score = backbone_score + opp_score
            if raw_score <= 0.0:
                continue

            if is_exact_prefix and step < len(my_rec) and my_rec[step] not in my_picks_set:
                intended_next = my_rec[step]
            else:
                intended_next = remaining_my[0]

            dead_banned = sum(1 for c in remaining_my if c in banned_cards)
            dead_stolen = sum(1 for c in remaining_my if c in opp_picks_set)
            immediate_dead = intended_next in locked_unavailable

            viability_mult = 1.0
            if dead_banned > 0:
                viability_mult *= config.penalty_my_banned_per_card ** dead_banned
            if dead_stolen > 0:
                viability_mult *= config.penalty_my_stolen_per_card ** dead_stolen
            if immediate_dead:
                viability_mult *= config.penalty_immediate_next_dead
            if any(c in banned_cards for c in opp_rec):
                viability_mult *= config.penalty_opp_contains_banned

            branch_weight = raw_score * ban_mult * viability_mult
            target_card = intended_next if not immediate_dead else available_remaining_my[0]
            is_viable = dead_banned == 0 and dead_stolen == 0 and not immediate_dead

            rec_branch_candidates.append((branch_weight, target_card, is_primary_ban, is_viable))
            if branch_weight > best_rec_weight:
                best_rec_weight = branch_weight

    my_rec_scores: dict[str, float] = defaultdict(float)
    used_cross_ban_fallback = False
    if best_rec_weight > 0.0:
        min_keep = best_rec_weight * config.agg_top_tier_ratio
        best_winner_is_primary = False
        for weight, card, is_primary, _ in rec_branch_candidates:
            if weight >= min_keep:
                sharpened = weight * ((weight / best_rec_weight) ** 2)
                my_rec_scores[card] += sharpened
                if weight == best_rec_weight and is_primary:
                    best_winner_is_primary = True
        if has_primary_ban and not best_winner_is_primary:
            used_cross_ban_fallback = True

    final_my_recs = _rank_candidates(my_rec_scores, locked_unavailable, config.max_recommendations)
    if not final_my_recs:
        final_my_recs = ["暫無數據匹配"]

    pred_branch_candidates: list[tuple[float, list[tuple[str, float]], bool]] = []
    best_normal_pred: float = 0.0
    best_mirror_pred: float = 0.0
    best_mirror_hits: int = 0

    if len(opp_picks) < 5:
        for ban_key, records in db.items():
            normal_ban_mult = (
                config.pred_primary_ban_mult
                if (has_primary_ban and ban_key == opp_ban_key)
                else config.pred_cross_ban_mult
            )
            if my_ban_key and ban_key == my_ban_key:
                mirror_ban_mult = config.pred_mirror_my_ban_mult
            elif has_primary_ban and ban_key == opp_ban_key:
                mirror_ban_mult = config.pred_mirror_opp_ban_mult
            else:
                mirror_ban_mult = config.pred_mirror_other_ban_mult

            for rec in records:
                my_rec = rec.get("my", [])
                opp_rec = rec.get("opp", [])

                if opp_rec:
                    opp_score_a, hits_a = _score_opp_unordered(opp_rec, opp_picks_set)
                    if hits_a > 0 or not opp_picks_set:
                        rem_opp = [c for c in opp_rec if c not in opp_picks_set]
                        avail_opp = [c for c in rem_opp if c not in locked_unavailable]
                        if avail_opp:
                            my_ctx_hits = len(my_picks_set & set(my_rec))
                            dead_b = sum(1 for c in rem_opp if c in banned_cards)
                            dead_t = sum(1 for c in rem_opp if c in my_picks_set)
                            viab_a = (config.pred_dead_banned_penalty ** dead_b) * (
                                config.pred_dead_taken_penalty ** dead_t
                            )
                            base_a = (opp_score_a if opp_picks_set else 60.0) + (my_ctx_hits * 25.0)
                            w_a = base_a * normal_ban_mult * viab_a
                            card_weights_a = [(c, 1.0) for c in avail_opp]
                            pred_branch_candidates.append((w_a, card_weights_a, False))
                            if w_a > best_normal_pred:
                                best_normal_pred = w_a

                if my_rec and opp_picks_set:
                    my_backbone_set = set(my_rec)
                    mirror_hits = len(opp_picks_set & my_backbone_set)
                    if mirror_hits > 0:
                        rem_mirror = [c for c in my_rec if c not in opp_picks_set]
                        avail_mirror = [c for c in rem_mirror if c not in locked_unavailable]
                        if avail_mirror:
                            m_precision = mirror_hits / float(len(opp_picks_set))
                            m_recall = mirror_hits / float(len(my_backbone_set))
                            opener_hits = sum(1 for c in my_rec[:2] if c in opp_picks_set)
                            opener_bonus = opener_hits * 35.0
                            dead_b = sum(1 for c in rem_mirror if c in banned_cards)
                            dead_t = sum(1 for c in rem_mirror if c in my_picks_set)
                            viab_b = (config.pred_dead_banned_penalty ** dead_b) * (
                                config.pred_dead_taken_penalty ** dead_t
                            )
                            base_b = (
                                (mirror_hits * 145.0)
                                + (m_precision * 115.0)
                                + (m_recall * 40.0)
                                + opener_bonus
                            )
                            w_b = base_b * mirror_ban_mult * viab_b
                            card_weights_b = [
                                (c, 1.25 if idx == 0 else 0.80) for idx, c in enumerate(avail_mirror)
                            ]
                            pred_branch_candidates.append((w_b, card_weights_b, True))
                            if w_b > best_mirror_pred:
                                best_mirror_pred = w_b
                                best_mirror_hits = mirror_hits

    best_pred_weight = max(best_normal_pred, best_mirror_pred)
    opp_pred_scores: dict[str, float] = defaultdict(float)
    mirror_detected = (
        best_mirror_pred > 0.0
        and best_mirror_hits >= 2
        and best_mirror_pred >= (best_normal_pred * 0.85)
    ) or (
        best_mirror_pred > 0.0
        and best_mirror_hits >= 1
        and best_mirror_pred >= best_normal_pred
    )

    if best_pred_weight > 0.0:
        min_pred_keep = best_pred_weight * config.agg_top_tier_ratio
        for weight, card_weights, _ in pred_branch_candidates:
            if weight >= min_pred_keep:
                sharpened = weight * ((weight / best_pred_weight) ** 2)
                for card, rel_w in card_weights:
                    opp_pred_scores[card] += sharpened * rel_w

    final_opp_preds = _rank_candidates(opp_pred_scores, locked_unavailable, config.max_predictions)

    tags: list[str] = []
    if is_fallback:
        tags.append(f"預設Ban{my_ban_card}")
    elif used_cross_ban_fallback:
        tags.append("跨Ban變陣")
    if mirror_detected:
        tags.append("鏡像/內戰")

    tag_str = f" [{'·'.join(tags)}]" if tags else ""
    match_type = f"推:{best_rec_weight:.1f} | 預:{best_pred_weight:.1f}{tag_str}"

    return final_my_recs, final_opp_preds, match_type
