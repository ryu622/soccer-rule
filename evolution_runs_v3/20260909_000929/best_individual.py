# id=ind27 island=B fitness_f1=0.499 conditions=21 penalized=0.499
PARAM_SPECS = {
    "tight_space_distance_m": (0.5, 5.0, 2.5),
    "head_on_angle_deg": (10.0, 70.0, 40.0),
    "retreating_angle_deg": (90.0, 170.0, 120.0),
    "open_space_dist_m": (2.0, 10.0, 4.0),
    "support_min_count": (0.0, 4.0, 1.0),
    "sideline_danger_dist_m": (1.0, 6.0, 2.5),
    "speed_advantage_mps": (-1.0, 2.0, 0.0),
}

def predict_take_on(f: dict, p: dict) -> int:
    is_tight_space = f["distance_m"] < p["tight_space_distance_m"]
    is_head_on_approach = f["approach_angle_deg"] < p["head_on_angle_deg"]
    is_defender_retreating = f["approach_angle_deg"] > p["retreating_angle_deg"]
    has_space_beyond = f["second_nearest_dist_m"] > p["open_space_dist_m"]
    has_support = f["n_supporting_teammates"] >= p["support_min_count"]
    is_pinned_to_sideline = f["dist_to_sideline_m"] < p["sideline_danger_dist_m"]
    is_carrier_faster_enough = (
        f["carrier_speed_mps"] - f["opponent_speed_mps"] > p["speed_advantage_mps"]
    )

    pattern_beat_defender_headon = (
        is_tight_space
        and is_head_on_approach
        and is_carrier_faster_enough
        and has_space_beyond
    )

    pattern_exploit_retreating_defender = (
        is_defender_retreating and has_space_beyond and not is_pinned_to_sideline
    )

    pattern_supported_risk_take = (
        is_tight_space and has_support and has_space_beyond
    )

    pattern_escape_sideline_trap = (
        is_pinned_to_sideline
        and is_tight_space
        and is_head_on_approach
        and has_space_beyond
        and (has_support or is_carrier_faster_enough)
    )

    should_take_on = (
        pattern_beat_defender_headon
        or pattern_exploit_retreating_defender
        or pattern_supported_risk_take
        or pattern_escape_sideline_trap
    )

    return 1 if should_take_on else 0
