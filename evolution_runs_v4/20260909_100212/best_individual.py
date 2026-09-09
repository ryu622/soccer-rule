# id=ind4 island=A fitness_f1=0.439 conditions=10 penalized=0.419
PARAM_SPECS = {
    "tight_space_distance_m": (1.0, 8.0, 4.0),
    "direct_approach_angle_deg": (10.0, 90.0, 45.0),
    "high_closing_speed_mps": (0.0, 4.0, 1.5),
    "space_after_beat_m": (2.0, 12.0, 6.0),
    "support_min_count": (0.0, 5.0, 1.0),
    "sideline_risk_distance_m": (0.5, 8.0, 3.0),
    "attacking_third_distance_m": (10.0, 40.0, 25.0),
}

def predict_take_on(f: dict, p: dict) -> int:
    is_tight_space = f["distance_m"] < p["tight_space_distance_m"]
    is_direct_approach = f["approach_angle_deg"] < p["direct_approach_angle_deg"]
    is_high_closing_speed = f["closing_speed_mps"] > p["high_closing_speed_mps"]
    has_space_after_beat = f["second_nearest_dist_m"] > p["space_after_beat_m"]
    has_enough_support = f["n_supporting_teammates"] >= p["support_min_count"]
    is_pinned_to_sideline = f["dist_to_sideline_m"] < p["sideline_risk_distance_m"]
    is_in_attacking_third = f["dist_to_goal_line_m"] < p["attacking_third_distance_m"]

    is_dangerous_pressure = is_tight_space and is_direct_approach and is_high_closing_speed

    favorable_space = has_space_after_beat and not is_pinned_to_sideline

    is_worth_risk = has_enough_support or is_in_attacking_third

    should_take_on = (not is_dangerous_pressure) and favorable_space and is_worth_risk

    return 1 if should_take_on else 0
