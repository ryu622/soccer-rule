# id=ind17 island=C fitness_f1=0.424 conditions=12 penalized=0.400
PARAM_SPECS = {
    "pressure_distance_m": (1.0, 6.0, 3.5),
    "safe_approach_angle_deg": (100.0, 170.0, 140.0),
    "closing_speed_threat_mps": (0.2, 2.5, 1.2),
    "space_after_beat_m": (2.0, 8.0, 4.0),
    "support_min_count": (0.0, 3.0, 0.0),
    "sideline_risk_distance_m": (2.0, 8.0, 3.0),
    "congestion_max_opponents": (1.0, 5.0, 3.0),
}

def predict_take_on(f: dict, p: dict) -> int:
    # 第一段階: 間合いによる状況判定
    is_under_pressure = f["distance_m"] < p["pressure_distance_m"]

    # 第二段階: 守備者の脅威度評価
    is_defender_closing_in = f["closing_speed_mps"] > p["closing_speed_threat_mps"]
    is_defender_facing_away = f["approach_angle_deg"] > p["safe_approach_angle_deg"]
    is_carrier_faster = f["carrier_speed_mps"] > f["opponent_speed_mps"]

    manageable_threat = (not is_defender_closing_in) or is_defender_facing_away or is_carrier_faster

    # 第三段階: 突破後の空間的余裕(密集度と2番目の守備者距離を独立に評価)
    has_space_beyond = f["second_nearest_dist_m"] > p["space_after_beat_m"]
    is_not_congested = f["n_opponents_nearby_10m"] <= p["congestion_max_opponents"]

    # 第四段階: 位置的リスクとサポート状況
    has_support = f["n_supporting_teammates"] >= p["support_min_count"]
    is_pinned_to_sideline = f["dist_to_sideline_m"] < p["sideline_risk_distance_m"]

    positional_risk_acceptable = has_support or (not is_pinned_to_sideline) or has_space_beyond

    # 直前のプレー文脈
    was_previous_pass = bool(f["prev_event_was_pass"])

    should_take_on = (
        is_under_pressure
        and manageable_threat
        and is_not_congested
        and positional_risk_acceptable
    )

    if was_previous_pass and not has_space_beyond:
        should_take_on = False

    return 1 if should_take_on else 0
