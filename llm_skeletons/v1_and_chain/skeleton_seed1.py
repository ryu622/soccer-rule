PARAM_SPECS = {
    "tight_space_distance_m": (1.0, 5.0, 2.5),
    "frontal_approach_angle_deg": (20.0, 80.0, 45.0),
    "fleeing_approach_angle_deg": (100.0, 170.0, 135.0),
    "fast_closing_speed_mps": (0.5, 3.0, 1.5),
    "speed_advantage_margin_mps": (0.0, 2.0, 0.5),
    "escape_space_distance_m": (3.0, 10.0, 6.0),
    "support_min_count": (0.0, 3.0, 1.0),
    "sideline_danger_distance_m": (1.0, 6.0, 3.0),
    "attacking_third_distance_m": (15.0, 40.0, 25.0),
}

def predict_take_on(f: dict, p: dict) -> int:
    is_tight_space = f["distance_m"] < p["tight_space_distance_m"]

    is_frontal_pressure = f["approach_angle_deg"] < p["frontal_approach_angle_deg"]
    is_defender_fleeing = f["approach_angle_deg"] > p["fleeing_approach_angle_deg"]

    is_fast_closing = f["closing_speed_mps"] > p["fast_closing_speed_mps"]

    has_speed_advantage = (
        f["carrier_speed_mps"] - f["opponent_speed_mps"] > p["speed_advantage_margin_mps"]
    )

    has_escape_space = f["second_nearest_dist_m"] > p["escape_space_distance_m"]

    has_support = f["n_supporting_teammates"] >= p["support_min_count"]

    is_pinned_to_sideline = f["dist_to_sideline_m"] < p["sideline_danger_distance_m"]

    is_in_attacking_third = f["dist_to_goal_line_m"] < p["attacking_third_distance_m"]

    favorable_duel = (
        is_tight_space
        and is_frontal_pressure
        and not is_fast_closing
        and has_speed_advantage
    )

    exploitable_gap = (
        is_tight_space
        and is_defender_fleeing
        and has_escape_space
    )

    safety_net = has_support or has_escape_space

    high_reward_zone = is_in_attacking_third and not is_pinned_to_sideline

    take_on = (favorable_duel or exploitable_gap) and safety_net and (
        high_reward_zone or has_escape_space
    )

    return 1 if take_on else 0
