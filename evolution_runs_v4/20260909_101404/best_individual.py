# id=ind12 island=B fitness_f1=0.455 conditions=16 penalized=0.423
PARAM_SPECS = {
    "tight_space_distance_m": (0.5, 4.0, 2.0),
    "head_on_approach_angle_deg": (10.0, 60.0, 35.0),
    "escape_space_ratio": (1.0, 3.0, 1.5),
    "min_supporting_teammates": (0.0, 3.0, 1.0),
    "high_closing_speed_mps": (0.5, 3.0, 1.5),
    "sideline_danger_distance_m": (2.0, 8.0, 4.0),
    "attacking_third_distance_m": (15.0, 35.0, 25.0),
}

def predict_take_on(f: dict, p: dict) -> int:
    is_tight_space = f["distance_m"] < p["tight_space_distance_m"]
    is_head_on_approach = f["approach_angle_deg"] < p["head_on_approach_angle_deg"]
    is_closing_fast = f["closing_speed_mps"] > p["high_closing_speed_mps"]
    has_escape_space = f["second_nearest_dist_m"] > f["distance_m"] * p["escape_space_ratio"]
    has_support = f["n_supporting_teammates"] >= p["min_supporting_teammates"]
    is_pinned_to_sideline = f["dist_to_sideline_m"] < p["sideline_danger_distance_m"]
    is_carrier_faster = f["carrier_speed_mps"] > f["opponent_speed_mps"]
    is_in_attacking_third = f["dist_to_goal_line_m"] < p["attacking_third_distance_m"]

    pattern_space_exploit = is_tight_space and has_escape_space and is_carrier_faster

    pattern_supported_confrontation = (
        is_head_on_approach and is_closing_fast and has_support
    )

    pattern_final_third_opportunism = (
        is_in_attacking_third and has_escape_space and is_carrier_faster
    )

    pattern_avoid_sideline_trap = (
        is_tight_space and not is_pinned_to_sideline and has_escape_space
    )

    should_take_on = (
        pattern_space_exploit
        or pattern_supported_confrontation
        or pattern_final_third_opportunism
        or pattern_avoid_sideline_trap
    )

    return 1 if should_take_on else 0
