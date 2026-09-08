PARAM_SPECS = {
    "tight_space_distance_m": (1.0, 6.0, 3.5),
    "closing_fast_mps": (0.0, 3.0, 1.0),
    "head_on_angle_deg": (20.0, 90.0, 45.0),
    "escape_space_dist_m": (3.0, 10.0, 6.0),
    "support_min_count": (0.0, 4.0, 1.0),
    "danger_zone_dist_m": (10.0, 35.0, 22.0),
    "min_votes": (2.0, 6.0, 3.0),
}

def predict_take_on(f: dict, p: dict) -> int:
    is_tight_space = f["distance_m"] < p["tight_space_distance_m"]

    is_closing_fast = f["closing_speed_mps"] > p["closing_fast_mps"]

    is_head_on_approach = f["approach_angle_deg"] < p["head_on_angle_deg"]

    has_escape_space = f["second_nearest_dist_m"] > p["escape_space_dist_m"]

    has_support = f["n_supporting_teammates"] >= p["support_min_count"]

    speed_advantage = f["carrier_speed_mps"] > f["opponent_speed_mps"]

    is_attacking_third = f["dist_to_goal_line_m"] < p["danger_zone_dist_m"]

    not_pinned_sideline = f["dist_to_sideline_m"] > p["tight_space_distance_m"]

    votes = (
        int(is_tight_space)
        + int(is_closing_fast)
        + int(is_head_on_approach)
        + int(has_escape_space)
        + int(has_support)
        + int(speed_advantage)
        + int(is_attacking_third)
        + int(not_pinned_sideline)
    )

    return 1 if votes >= p["min_votes"] else 0
