# id=ind19 island=A fitness_f1=0.491 conditions=10 penalized=0.471
PARAM_SPECS = {
    "tight_space_distance_m": (1.0, 10.0, 5.0),
    "closing_speed_threshold_mps": (-1.0, 3.0, -0.5),
    "escape_space_distance_m": (1.0, 10.0, 4.0),
    "sideline_danger_distance_m": (0.5, 6.0, 1.0),
    "congestion_max_opponents": (0.0, 6.0, 2.0),
    "direct_approach_angle_deg": (0.0, 180.0, 60.0),
    "support_min_count": (0.0, 5.0, 1.0),
}

def predict_take_on(f: dict, p: dict) -> int:
    is_tight_space = f["distance_m"] < p["tight_space_distance_m"]
    is_closing = f["closing_speed_mps"] > p["closing_speed_threshold_mps"]
    has_escape_space = f["second_nearest_dist_m"] > p["escape_space_distance_m"]
    is_pinned_to_sideline = f["dist_to_sideline_m"] < p["sideline_danger_distance_m"]
    is_congested = f["n_opponents_nearby_10m"] > p["congestion_max_opponents"]
    is_direct_approach = f["approach_angle_deg"] < p["direct_approach_angle_deg"]
    has_support = f["n_supporting_teammates"] >= p["support_min_count"]
    was_previous_pass = bool(f["prev_event_was_pass"])

    favorable_situation = has_escape_space or (not is_congested) or has_support

    pressure_trigger = is_tight_space and (is_closing or is_direct_approach)

    should_take_on = (
        pressure_trigger
        and favorable_situation
        and not is_pinned_to_sideline
    )

    return 1 if should_take_on else 0
