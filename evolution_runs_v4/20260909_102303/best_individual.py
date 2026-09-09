# id=ind64 island=A fitness_f1=0.557 conditions=18 penalized=0.521
PARAM_SPECS = {
    "tight_space_distance_m": (1.0, 8.0, 2.5),
    "clear_retreat_closing_speed_mps": (-6.0, 0.0, -1.5),
    "space_after_beat_distance_m": (1.0, 12.0, 5.0),
    "support_min_count": (0.0, 5.0, 1.0),
    "sideline_risk_distance_m": (0.0, 6.0, 2.5),
    "speed_disadvantage_margin_mps": (0.0, 8.0, 1.0),
    "retreating_defender_angle_deg": (90.0, 180.0, 135.0),
}

def predict_take_on(f: dict, p: dict) -> int:
    is_tight_space = f["distance_m"] < p["tight_space_distance_m"]

    is_defender_clearly_retreating = (
        f["closing_speed_mps"] < p["clear_retreat_closing_speed_mps"]
        or f["approach_angle_deg"] > p["retreating_defender_angle_deg"]
    )

    has_space_after_beating = f["second_nearest_dist_m"] > p["space_after_beat_distance_m"]

    has_back_support = f["n_supporting_teammates"] >= p["support_min_count"]

    is_pinned_to_sideline = f["dist_to_sideline_m"] < p["sideline_risk_distance_m"]

    speed_deficit = f["opponent_speed_mps"] - f["carrier_speed_mps"]
    carrier_not_outpaced = speed_deficit < p["speed_disadvantage_margin_mps"]

    engagement_opportunity = is_tight_space or is_defender_clearly_retreating

    favorable_space = has_space_after_beating or (not is_pinned_to_sideline and has_back_support)

    risk_acceptable = carrier_not_outpaced or (has_back_support and has_space_after_beating)

    severe_sideline_trap = is_pinned_to_sideline and not has_space_after_beating

    should_take_on = (
        engagement_opportunity
        and favorable_space
        and risk_acceptable
        and not severe_sideline_trap
    )

    return 1 if should_take_on else 0
