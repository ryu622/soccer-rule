# id=ind15 island=B fitness_f1=0.486 conditions=19 penalized=0.486
PARAM_SPECS = {
    "tight_space_distance_m": (1.0, 8.0, 4.5),
    "fast_closing_speed_mps": (0.0, 3.0, 0.8),
    "open_space_second_dist_m": (2.0, 12.0, 5.0),
    "support_min_count": (0.0, 3.0, 1.0),
    "sideline_trap_distance_m": (1.0, 8.0, 5.0),
    "attacking_third_distance_m": (10.0, 40.0, 30.0),
    "speed_advantage_margin_mps": (0.0, 2.0, 0.3),
}

def predict_take_on(f: dict, p: dict) -> int:
    is_tight_space = f["distance_m"] < p["tight_space_distance_m"]
    is_fast_closing = f["closing_speed_mps"] > p["fast_closing_speed_mps"]
    is_carrier_faster = (
        f["carrier_speed_mps"] - f["opponent_speed_mps"] > p["speed_advantage_margin_mps"]
    )
    has_open_space_behind = f["second_nearest_dist_m"] > p["open_space_second_dist_m"]
    has_support = f["n_supporting_teammates"] >= p["support_min_count"]
    is_sideline_trapped = f["dist_to_sideline_m"] < p["sideline_trap_distance_m"]
    is_attacking_third = f["dist_to_goal_line_m"] < p["attacking_third_distance_m"]
    is_retreating_defender = f["closing_speed_mps"] < 0

    pattern_speed_advantage = (
        is_tight_space and is_carrier_faster and has_open_space_behind
    )

    pattern_pressure_with_support = (
        is_tight_space and is_fast_closing and has_support
    )

    pattern_attacking_third_opportunity = (
        has_open_space_behind and is_attacking_third and (has_support or is_carrier_faster)
    )

    pattern_desperate_sideline_escape = (
        is_sideline_trapped and (is_fast_closing or is_tight_space) and has_open_space_behind
    )

    pattern_free_space_takeon = (
        not is_tight_space and has_open_space_behind and is_retreating_defender
    )

    score = (
        int(pattern_speed_advantage)
        + int(pattern_pressure_with_support)
        + int(pattern_attacking_third_opportunity)
        + int(pattern_desperate_sideline_escape)
        + int(pattern_free_space_takeon)
    )

    should_take_on = score >= 1

    return 1 if should_take_on else 0
