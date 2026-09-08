# id=ind4 island=B fitness_f1=0.430
# fitted_params={'tight_space_distance_m': 2.108410335447504, 'frontal_approach_angle_deg': 58.735273845502896, 'escape_angle_deg': 110.01318096429848, 'high_closing_speed_mps': 2.292597637006766, 'space_behind_min_m': 3.0, 'support_min_count': 0.6799508369954164, 'sideline_trap_distance_m': 2.6929341846793227, 'attacking_third_distance_m': 10.0}
PARAM_SPECS = {
    "tight_space_distance_m": (0.5, 4.0, 2.0),
    "frontal_approach_angle_deg": (10.0, 60.0, 35.0),
    "escape_angle_deg": (100.0, 170.0, 140.0),
    "high_closing_speed_mps": (0.5, 3.5, 1.5),
    "space_behind_min_m": (3.0, 10.0, 6.0),
    "support_min_count": (0.0, 3.0, 1.0),
    "sideline_trap_distance_m": (1.0, 6.0, 3.0),
    "attacking_third_distance_m": (10.0, 35.0, 25.0),
}

def predict_take_on(f: dict, p: dict) -> int:
    is_tight_space = f["distance_m"] < p["tight_space_distance_m"]
    is_frontal_approach = f["approach_angle_deg"] < p["frontal_approach_angle_deg"]
    is_defender_retreating = f["approach_angle_deg"] > p["escape_angle_deg"]
    is_closing_fast = f["closing_speed_mps"] > p["high_closing_speed_mps"]
    has_space_behind = f["second_nearest_dist_m"] > p["space_behind_min_m"]
    has_support = f["n_supporting_teammates"] >= p["support_min_count"]
    is_sideline_trapped = f["dist_to_sideline_m"] < p["sideline_trap_distance_m"]
    is_attacking_third = f["dist_to_goal_line_m"] < p["attacking_third_distance_m"]
    carrier_faster_than_opponent = f["carrier_speed_mps"] > f["opponent_speed_mps"]

    pattern_speed_advantage = (
        is_tight_space
        and is_frontal_approach
        and carrier_faster_than_opponent
        and has_space_behind
    )

    pattern_defender_off_balance = (
        is_defender_retreating
        and has_space_behind
        and not is_sideline_trapped
    )

    pattern_supported_risk_take = (
        is_tight_space
        and has_support
        and has_space_behind
        and not is_closing_fast
    )

    pattern_attacking_third_opportunity = (
        is_attacking_third
        and is_frontal_approach
        and has_space_behind
        and not is_sideline_trapped
    )

    pattern_forced_by_sideline = (
        is_sideline_trapped
        and is_tight_space
        and is_frontal_approach
        and carrier_faster_than_opponent
    )

    should_take_on = (
        pattern_speed_advantage
        or pattern_defender_off_balance
        or pattern_supported_risk_take
        or pattern_attacking_third_opportunity
        or pattern_forced_by_sideline
    )

    return 1 if should_take_on else 0
