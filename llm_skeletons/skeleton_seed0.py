PARAM_SPECS = {
    "tight_space_distance_m": (1.0, 6.0, 3.0),
    "closing_fast_mps": (0.5, 3.0, 1.5),
    "frontal_approach_deg": (20.0, 90.0, 45.0),
    "open_space_dist_m": (3.0, 10.0, 6.0),
    "support_min_count": (0.0, 4.0, 2.0),
    "attacking_third_dist_m": (10.0, 40.0, 25.0),
    "min_votes": (1.0, 6.0, 3.0),
}

def predict_take_on(f: dict, p: dict) -> int:
    is_tight_space = f["distance_m"] < p["tight_space_distance_m"]
    is_closing_fast = f["closing_speed_mps"] > p["closing_fast_mps"]
    is_frontal_approach = f["approach_angle_deg"] < p["frontal_approach_deg"]
    has_space_beyond = f["second_nearest_dist_m"] > p["open_space_dist_m"]
    has_backup = f["n_supporting_teammates"] >= p["support_min_count"]
    is_speed_advantage = f["carrier_speed_mps"] > f["opponent_speed_mps"]
    is_dangerous_zone = f["dist_to_goal_line_m"] < p["attacking_third_dist_m"]
    is_not_sideline_trapped = f["dist_to_sideline_m"] > p["tight_space_distance_m"]

    votes = 0
    votes += int(is_tight_space and is_closing_fast)
    votes += int(is_frontal_approach)
    votes += int(has_space_beyond)
    votes += int(has_backup)
    votes += int(is_speed_advantage)
    votes += int(is_dangerous_zone and is_not_sideline_trapped)

    return 1 if votes >= p["min_votes"] else 0
