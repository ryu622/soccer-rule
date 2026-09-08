# id=ind48 island=C fitness_f1=0.483 conditions=16 penalized=0.451
PARAM_SPECS = {
    "pressure_distance_m": (1.0, 6.0, 3.5),
    "safe_approach_angle_deg": (90.0, 160.0, 130.0),
    "fast_closing_speed_mps": (0.5, 3.0, 1.5),
    "open_space_dist_m": (3.0, 10.0, 6.0),
    "support_min_count": (0.0, 3.0, 1.0),
    "sideline_trap_dist_m": (1.0, 8.0, 4.0),
    "congestion_count_max": (1.0, 5.0, 3.0),
}

def predict_take_on(f: dict, p: dict) -> int:
    # 大枠: 間合いが詰まっているプレッシャー局面かどうか
    is_under_pressure = f["distance_m"] < p["pressure_distance_m"]

    if not is_under_pressure:
        # 間合いに余裕がある場合は、基本的に急いで仕掛ける必要がない
        return 0

    # ここからはプレッシャー下での詳細判断

    # 守備者の寄せ方: 正対して詰めてきているか、それとも外れていく動きか
    defender_committed = f["approach_angle_deg"] < p["safe_approach_angle_deg"]

    # 守備者が急速に距離を詰めてきているか(勢いのある寄せ)
    is_closing_fast = f["closing_speed_mps"] > p["fast_closing_speed_mps"]

    # ボール保持者が守備者より優位な速度を持っているか(振り切れる可能性)
    has_speed_advantage = f["carrier_speed_mps"] > f["opponent_speed_mps"]

    # 1人目を抜いた後のスペースがあるか
    has_space_beyond = f["second_nearest_dist_m"] > p["open_space_dist_m"]

    # 後方サポートが十分か
    has_support = f["n_supporting_teammates"] >= p["support_min_count"]

    # ピッチ上のリスク: サイドに追い込まれていないか
    is_pinned_to_sideline = f["dist_to_sideline_m"] < p["sideline_trap_dist_m"]

    # 密集度: 周囲に相手が多すぎないか
    is_congested = f["n_opponents_nearby_10m"] > p["congestion_count_max"]

    # 直前パスを受けた直後は、味方の意図的な仕掛け準備の可能性が高い
    was_previous_pass = bool(f["prev_event_was_pass"])

    # 攻撃的優位性の判断: スピード優位 or 空間的余裕があること
    has_dribble_advantage = has_speed_advantage or has_space_beyond

    # リスク要因: サイドに詰められている、かつ密集している、かつサポートがない
    is_high_risk = is_pinned_to_sideline and is_congested and not has_support

    # 守備者の寄せが甘い(離れていく、または遅い)場合は仕掛けやすい
    defender_is_vulnerable = (not defender_committed) or (not is_closing_fast)

    should_take_on = (
        is_under_pressure
        and not is_high_risk
        and (has_dribble_advantage or defender_is_vulnerable)
        and (has_support or was_previous_pass or has_space_beyond)
    )

    return 1 if should_take_on else 0
