PARAM_SPECS = {
    "tight_space_distance_m": (1.0, 6.0, 3.0),
    "closing_speed_threshold_mps": (-1.0, 3.0, 1.0),
    "head_on_angle_deg": (10.0, 90.0, 45.0),
    "space_after_beat_m": (3.0, 12.0, 6.0),
    "support_min_count": (0.0, 4.0, 1.0),
    "sideline_trap_distance_m": (1.0, 8.0, 4.0),
    "min_votes": (1.0, 6.0, 3.0),
}

def predict_take_on(f: dict, p: dict) -> int:
    is_tight_space = f["distance_m"] < p["tight_space_distance_m"]
    is_closing_fast = f["closing_speed_mps"] > p["closing_speed_threshold_mps"]
    is_head_on_approach = f["approach_angle_deg"] < p["head_on_angle_deg"]
    has_space_beyond = f["second_nearest_dist_m"] > p["space_after_beat_m"]
    has_support = f["n_supporting_teammates"] >= p["support_min_count"]
    is_carrier_faster = f["carrier_speed_mps"] > f["opponent_speed_mps"]
    not_pinned_sideline = f["dist_to_sideline_m"] > p["sideline_trap_distance_m"]

    votes = (
        int(is_tight_space)
        + int(is_closing_fast)
        + int(is_head_on_approach)
        + int(has_space_beyond)
        + int(has_support)
        + int(is_carrier_faster)
        + int(not_pinned_sideline)
    )

    return 1 if votes >= p["min_votes"] else 0
