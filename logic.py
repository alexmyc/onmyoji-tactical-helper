# -*- coding: utf-8 -*-
"""Draft Decision Logic Module for Onmyoji Tactical Helper.

This module provides core computer algorithms for parsing draft strings, constructing
line-up databases, and calculating asymmetric weighted recommendations and opponent predictions.
"""

from collections import Counter
from typing import Dict, List, Optional, Tuple


def parse_draft_string(draft_str: str, multi_char_shorts: List[str]) -> List[str]:
    """Parses a continuous draft string into individual character short names.

    Args:
        draft_str: Continuous string representing a sequence of drafted characters.
        multi_char_shorts: List of multi-character short names matched first by length.

    Returns:
        List of tokenized character short name strings.
    """
    sorted_multi_chars = sorted(multi_char_shorts, key=len, reverse=True)
    result: List[str] = []
    i = 0
    while i < len(draft_str):
        matched = False
        for mc in sorted_multi_chars:
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
    multi_char_shorts: List[str]
) -> Dict[str, List[Dict[str, List[str]]]]:
    """Builds a structured draft line-up database from raw text data.

    Args:
        raw_data: Raw multiline text data containing Ban headers and line-up pairs.
        multi_char_shorts: List of multi-character short names for string parsing.

    Returns:
        Dictionary mapping Ban keys (e.g. 'Ban禪') to lists of draft record dictionaries.
    """
    db: Dict[str, List[Dict[str, List[str]]]] = {}
    current_ban: Optional[str] = None
    for line in raw_data.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        if line.startswith("Ban"):
            current_ban = line
            db[current_ban] = []
        else:
            parts = line.split()
            if len(parts) == 2 and current_ban:
                db[current_ban].append({
                    "my": parse_draft_string(parts[0], multi_char_shorts),
                    "opp": parse_draft_string(parts[1], multi_char_shorts)
                })
    return db


def get_recommendation(
    db: Dict[str, List[Dict[str, List[str]]]],
    current_ban: Optional[str],
    opp_draft: List[str],
    my_draft: Optional[List[str]] = None,
    my_ban: str = "禪"
) -> Tuple[List[str], List[str], str]:
    """Calculates draft recommendations and opponent predictions using asymmetric scoring.

    Args:
        db: Line-up database mapping Ban keys to draft records.
        current_ban: Ban key string identified for the opponent (e.g. 'Ban須').
        opp_draft: List of characters currently drafted by the opponent.
        my_draft: List of characters currently drafted by the player.
        my_ban: Default fallback ban character name string if opponent ban is unspecified.

    Returns:
        Tuple containing:
        - List of recommended character choices for the player.
        - List of predicted character picks for the opponent.
        - Match status or score summary string.
    """
    if my_draft is None:
        my_draft = []

    turn_index = len(opp_draft)

    # Return completion status if 5 picks are completed
    if turn_index >= 5:
        return ["【武運亨通，旗開得勝】"], [], "選卡結束，準備戰鬥"

    # Blacklist processing
    banned_cards = set()

    actual_banned_card = current_ban.replace("Ban", "") if current_ban else None
    if actual_banned_card:
        banned_cards.add(actual_banned_card)

    is_fallback = False
    if not current_ban or current_ban not in db or not db[current_ban]:
        current_ban = f"Ban{my_ban}"
        is_fallback = True
        banned_cards.add(my_ban)

    my_recommendations: List[str] = []
    opp_predictions: List[str] = []

    # Turn 1: Blind Pick Logic
    if turn_index == 0:
        if current_ban in db:
            for record in db[current_ban]:
                if len(record["my"]) > 0:
                    my_recommendations.append(record["my"][0])
                if len(record["opp"]) > 0:
                    opp_predictions.append(record["opp"][0])

            status_msg = f"啟用默認 ({current_ban}) 首手盲翻" if is_fallback else f"常規 {current_ban} 首手盲翻"

            final_my_recs = [
                c for c in dict.fromkeys(my_recommendations)
                if c not in my_draft and c not in banned_cards
            ]
            final_opp_preds = [
                c for c in dict.fromkeys(opp_predictions)
                if c not in banned_cards
            ]

            if not final_my_recs:
                final_my_recs.append("暫無數據匹配")

            return final_my_recs, final_opp_preds, status_msg
        else:
            return ["無匹配"], ["未知陣容"], "未收錄此 Ban 位，請手動盲翻"

    # Asymmetric Scoring Algorithm
    best_rec_score: float = -1.0
    best_pred_score: float = -1.0

    matched_rec_records = []
    matched_pred_records = []

    for ban_key, records in db.items():
        is_current_ban = (ban_key == current_ban)

        for record in records:
            opp_record = record["opp"]
            my_record = record["my"]

            if any(b_card in opp_record or b_card in my_record for b_card in banned_cards):
                continue

            # 1. Calculate opponent score
            pos_opp = sum(1 for i, card in enumerate(opp_draft) if i < len(opp_record) and opp_record[i] == card)
            int_opp = len(set(opp_draft).intersection(set(opp_record)))
            score_opp = (int_opp * 100) + (pos_opp * 10)

            # 2. Calculate player score
            pos_my = sum(1 for i, card in enumerate(my_draft) if i < len(my_record) and my_record[i] == card)
            int_my = len(set(my_draft).intersection(set(my_record)))
            score_my = (int_my * 100) + (pos_my * 10)

            # 3. Calculate cross opponent score
            pos_opp_cross = sum(1 for i, card in enumerate(opp_draft) if i < len(my_record) and my_record[i] == card)
            int_opp_cross = len(set(opp_draft).intersection(set(my_record)))
            score_opp_cross = (int_opp_cross * 100) + (pos_opp_cross * 10)

            # Recommendation Score calculation
            base_rec_score = (score_opp * 1.5) + score_my
            if base_rec_score > 0:
                final_rec_score = base_rec_score + (0.1 if is_current_ban else 0)
                if final_rec_score > best_rec_score:
                    best_rec_score = final_rec_score
                    matched_rec_records = [record]
                elif final_rec_score == best_rec_score:
                    matched_rec_records.append(record)

            # Prediction Score calculation
            use_cross = score_opp_cross > score_opp
            active_opp_score = score_opp_cross if use_cross else score_opp

            base_pred_score = active_opp_score + score_my
            if base_pred_score > 0:
                final_pred_score = base_pred_score + (0.1 if is_current_ban else 0)
                if final_pred_score > best_pred_score:
                    best_pred_score = final_pred_score
                    matched_pred_records = [(record, use_cross)]
                elif final_pred_score == best_pred_score:
                    matched_pred_records.append((record, use_cross))

    # Generate recommendation results
    if best_rec_score > 0:
        for record in matched_rec_records:
            for card in record["my"]:
                if card not in my_draft:
                    my_recommendations.append(card)
                    break

    # Generate prediction results
    if best_pred_score > 0:
        for record, use_cross in matched_pred_records:
            target_record = record["my"] if use_cross else record["opp"]
            for card in target_record:
                if card not in opp_draft:
                    opp_predictions.append(card)
                    break

    # Sort candidates by frequency
    sorted_my_recs = [card for card, count in Counter(my_recommendations).most_common()]
    sorted_opp_preds = [card for card, count in Counter(opp_predictions).most_common()]

    final_my_recs = [
        card for card in sorted_my_recs
        if card not in my_draft and card not in banned_cards
    ]
    final_opp_preds = [
        card for card in sorted_opp_preds
        if card not in opp_draft and card not in banned_cards
    ]

    if not final_my_recs:
        final_my_recs.append("暫無數據匹配")

    match_type = f"推:{best_rec_score:.1f} | 預:{best_pred_score:.1f}"
    if is_fallback:
        match_type += f" (預設Ban{my_ban})"

    return final_my_recs, final_opp_preds, match_type