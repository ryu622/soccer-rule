# id=ind4 island=B fitness_f1=0.391 conditions=20 penalized=0.351
PARAM_SPECS = {
    "tight_space_distance_m": (0.5, 5.0, 2.5),
    "frontal_approach_angle_deg": (10.0, 90.0, 45.0),
    "fast_closing_speed_mps": (0.5, 4.0, 1.5),
    "open_space_second_dist_m": (2.0, 12.0, 6.0),
    "support_min_count": (0.0, 4.0, 1.0),
    "sideline_danger_dist_m": (1.0, 8.0, 3.0),
    "attacking_third_dist_to_goal_m": (10.0, 40.0, 25.0),
}

def predict_take_on(f: dict, p: dict) -> int:
    is_tight_space = f["distance_m"] < p["tight_space_distance_m"]
    is_frontal_approach = f["approach_angle_deg"] < p["frontal_approach_angle_deg"]
    is_defender_closing_fast = f["closing_speed_mps"] > p["fast_closing_speed_mps"]
    is_defender_retreating = f["closing_speed_mps"] < 0

    has_space_behind_defender = f["second_nearest_dist_m"] > p["open_space_second_dist_m"]
    has_support = f["n_supporting_teammates"] >= p["support_min_count"]

    is_pinned_to_sideline = f["dist_to_sideline_m"] < p["sideline_danger_dist_m"]
    is_in_attacking_third = f["dist_to_goal_line_m"] < p["attacking_third_dist_to_goal_m"]

    is_low_density = f["n_opponents_nearby_10m"] <= 1
    is_faster_than_opponent = f["carrier_speed_mps"] > f["opponent_speed_mps"]

    was_previous_pass = bool(f["prev_event_was_pass"])

    pattern_isolated_defender = (
        is_tight_space and is_frontal_approach and has_space_behind_defender and is_low_density
    )

    pattern_speed_advantage = (
        is_tight_space and is_defender_closing_fast and is_faster_than_opponent and has_space_behind_defender
    )

    pattern_supported_risk_take = (
        is_tight_space and has_support and is_in_attacking_third and not is_pinned_to_sideline
    )

    pattern_defender_backing_off = (
        is_tight_space and is_defender_retreating and was_previous_pass and has_space_behind_defender
    )

    take_on = (
        pattern_isolated_defender
        or pattern_speed_advantage
        or pattern_supported_risk_take
        or pattern_defender_backing_off
    )

    return 1 if take_on else 0
