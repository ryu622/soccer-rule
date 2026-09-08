PARAM_SPECS = {
    "tight_space_distance_m": (1.0, 6.0, 3.0),
    "direct_approach_angle_deg": (10.0, 60.0, 30.0),
    "fast_closing_speed_mps": (0.5, 3.0, 1.5),
    "space_after_beat_m": (3.0, 10.0, 6.0),
    "support_min_count": (0.0, 3.0, 1.0),
    "sideline_danger_m": (2.0, 8.0, 4.0),
    "attacking_third_dist_m": (15.0, 40.0, 25.0),
}

def predict_take_on(f: dict, p: dict) -> int:
    is_tight_pressure = f["distance_m"] < p["tight_space_distance_m"]
    is_direct_approach = f["approach_angle_deg"] < p["direct_approach_angle_deg"]
    is_closing_fast = f["closing_speed_mps"] > p["fast_closing_speed_mps"]

    has_space_behind_defender = f["second_nearest_dist_m"] > p["space_after_beat_m"]
    has_supporting_teammates = f["n_supporting_teammates"] >= p["support_min_count"]

    is_pinned_to_sideline = f["dist_to_sideline_m"] < p["sideline_danger_m"]
    is_in_attacking_third = f["dist_to_goal_line_m"] < p["attacking_third_dist_m"]

    carrier_has_speed_advantage = f["carrier_speed_mps"] >= f["opponent_speed_mps"]

    defender_engaged = is_tight_pressure and (is_direct_approach or is_closing_fast)

    favorable_space = has_space_behind_defender or has_supporting_teammates

    high_reward_zone = is_in_attacking_third or carrier_has_speed_advantage

    risky_position = is_pinned_to_sideline and not has_supporting_teammates

    should_take_on = defender_engaged and favorable_space and high_reward_zone and not risky_position

    return 1 if should_take_on else 0
