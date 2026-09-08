# id=ind90 island=C fitness_f1=0.598
# fitted_params={'tight_space_distance_m': 1.6117385656982066, 'head_on_approach_angle_deg': 56.311830142635166, 'high_closing_speed_mps': 0.9460245380409014, 'space_after_beat_distance_m': 2.0, 'support_min_count': 0.0, 'sideline_danger_distance_m': 2.2669773907787016, 'attacking_third_distance_m': 20.750672082911407}
PARAM_SPECS = {
    "tight_space_distance_m": (0.5, 5.0, 2.1),
    "head_on_approach_angle_deg": (10.0, 70.0, 36.0),
    "high_closing_speed_mps": (0.5, 3.0, 1.2),
    "space_after_beat_distance_m": (2.0, 10.0, 4.8),
    "support_min_count": (0.0, 4.0, 1.0),
    "sideline_danger_distance_m": (1.0, 8.0, 2.8),
    "attacking_third_distance_m": (10.0, 40.0, 26.0),
}

def predict_take_on(f: dict, p: dict) -> int:
    is_tight_space = f["distance_m"] < p["tight_space_distance_m"]
    is_head_on_approach = f["approach_angle_deg"] < p["head_on_approach_angle_deg"]
    is_closing_fast = f["closing_speed_mps"] > p["high_closing_speed_mps"]
    is_opponent_retreating = f["closing_speed_mps"] < 0

    has_space_after_beat = f["second_nearest_dist_m"] > p["space_after_beat_distance_m"]
    has_some_space_after_beat = f["second_nearest_dist_m"] > (p["space_after_beat_distance_m"] * 0.65)
    has_support = f["n_supporting_teammates"] >= p["support_min_count"]

    is_pinned_to_sideline = f["dist_to_sideline_m"] < p["sideline_danger_distance_m"]
    is_in_attacking_third = f["dist_to_goal_line_m"] < p["attacking_third_distance_m"]

    carrier_faster_than_opponent = f["carrier_speed_mps"] > f["opponent_speed_mps"]
    is_under_pressure = is_head_on_approach and is_closing_fast
    has_safety_net = has_space_after_beat or has_support

    pattern_committed_defender = (
        is_tight_space
        and is_under_pressure
        and has_space_after_beat
        and not is_pinned_to_sideline
    )

    pattern_supported_isolation = (
        is_tight_space
        and has_space_after_beat
        and has_support
        and not is_pinned_to_sideline
    )

    pattern_attacking_third_opportunity = (
        is_in_attacking_third
        and has_some_space_after_beat
        and carrier_faster_than_opponent
        and (has_support or not is_pinned_to_sideline)
    )

    pattern_exploit_retreating_defender = (
        is_opponent_retreating
        and has_space_after_beat
        and not is_pinned_to_sideline
        and (has_support or carrier_faster_than_opponent)
    )

    pattern_desperate_sideline_escape = (
        is_pinned_to_sideline
        and is_tight_space
        and is_head_on_approach
        and not is_closing_fast
        and has_support
        and carrier_faster_than_opponent
    )

    pattern_tight_space_favorable = (
        is_tight_space
        and not is_pinned_to_sideline
        and carrier_faster_than_opponent
        and has_safety_net
    )

    should_take_on = (
        pattern_committed_defender
        or pattern_supported_isolation
        or pattern_attacking_third_opportunity
        or pattern_exploit_retreating_defender
        or pattern_desperate_sideline_escape
        or pattern_tight_space_favorable
    )

    return 1 if should_take_on else 0
