# id=ind23 island=C fitness_f1=0.484 conditions=6 penalized=0.472
PARAM_SPECS = {
    "tight_space_distance_m": (1.0, 5.0, 2.5),
    "space_after_beat_m": (2.0, 10.0, 4.0),
    "crowd_limit_count": (1.0, 5.0, 3.0),
    "sideline_risk_m": (1.0, 6.0, 2.5),
    "danger_zone_dist_to_goal_m": (10.0, 35.0, 25.0),
    "support_min_count": (0.0, 3.0, 1.0),
    "closing_speed_threshold_mps": (-2.0, 2.0, 0.5),
}

def predict_take_on(f: dict, p: dict) -> int:
    is_under_pressure = f["distance_m"] < p["tight_space_distance_m"]

    if not is_under_pressure:
        return 0

    has_speed_advantage = f["carrier_speed_mps"] > f["opponent_speed_mps"]
    is_closing_fast = f["closing_speed_mps"] > p["closing_speed_threshold_mps"]
    favorable_duel = has_speed_advantage or (not is_closing_fast)

    has_space_after_beat = f["second_nearest_dist_m"] > p["space_after_beat_m"]
    is_crowded = f["n_opponents_nearby_10m"] >= p["crowd_limit_count"]
    space_ok = has_space_after_beat and (not is_crowded)

    has_support = f["n_supporting_teammates"] >= p["support_min_count"]

    is_pinned_to_sideline = f["dist_to_sideline_m"] < p["sideline_risk_m"]
    is_in_attacking_third = f["dist_to_goal_line_m"] < p["danger_zone_dist_to_goal_m"]
    positional_ok = (not is_pinned_to_sideline) or is_in_attacking_third

    was_previous_pass = bool(f["prev_event_was_pass"])

    take_on_score = 0
    if favorable_duel:
        take_on_score += 1
    if space_ok:
        take_on_score += 1
    if positional_ok:
        take_on_score += 1
    if is_in_attacking_third:
        take_on_score += 1
    if has_support:
        take_on_score += 1
    if was_previous_pass:
        take_on_score -= 1
    if is_crowded:
        take_on_score -= 1

    should_take_on = take_on_score >= 2

    return 1 if should_take_on else 0
