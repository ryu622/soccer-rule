PARAM_SPECS = {
    "tight_space_distance_m": (0.5, 5.0, 2.5),
    "direct_approach_angle_deg": (10.0, 80.0, 40.0),
    "fast_closing_speed_mps": (0.5, 4.0, 1.5),
    "open_space_ratio": (1.1, 3.0, 1.5),
    "support_min_count": (0.0, 4.0, 1.0),
    "sideline_risk_distance_m": (1.0, 8.0, 4.0),
    "attacking_third_distance_m": (10.0, 40.0, 25.0),
}

def predict_take_on(f: dict, p: dict) -> int:
    is_tight_pressure = f["distance_m"] < p["tight_space_distance_m"]

    is_direct_threat = (
        f["approach_angle_deg"] < p["direct_approach_angle_deg"]
        and f["closing_speed_mps"] > p["fast_closing_speed_mps"]
    )

    is_defender_passive = (
        f["approach_angle_deg"] > p["direct_approach_angle_deg"]
        or f["closing_speed_mps"] <= 0.0
    )

    has_space_advantage = (
        f["second_nearest_dist_m"] > f["distance_m"] * p["open_space_ratio"]
    )

    has_supporting_teammates = (
        f["n_supporting_teammates"] >= p["support_min_count"]
    )

    is_pinned_to_sideline = f["dist_to_sideline_m"] < p["sideline_risk_distance_m"]

    is_dangerous_area = f["dist_to_goal_line_m"] < p["attacking_third_distance_m"]

    is_speed_advantage = f["carrier_speed_mps"] > f["opponent_speed_mps"]

    favorable_engagement = (
        (is_tight_pressure and is_defender_passive)
        or (is_tight_pressure and has_space_advantage and is_speed_advantage)
    )

    risk_acceptable = (
        (has_supporting_teammates or is_dangerous_area)
        and not (is_pinned_to_sideline and is_direct_threat)
    )

    should_take_on = favorable_engagement and risk_acceptable

    return 1 if should_take_on else 0
