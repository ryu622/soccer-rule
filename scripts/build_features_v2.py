"""1v1デュエルタスクの特徴量セットを拡張し、決定木-XGBoostの差が広がるか確認するための
診断用データセットを構築する(documents/depth_sweep_results.md ・ phase3b_evolution_results.md
3.4.3/3.4.4節の議論を受けたレイヤー2/3切り分け実験)。

正例・負例の抽出ロジック(どの瞬間を1v1サンプルとみなすか)はbuild_features.pyと完全に同一。
特徴量だけを以下の4カテゴリで追加する:

    A. トレンド/加速度: distance_m・closing_speed_mpsの直近0.8〜1.0秒の推移
    B. 守備者の方向転換率: 進行方向ベクトルの変化角度
    C. 3番目に近い守備者・周辺密集度
    D. ボール保持者の直前の選択履歴: 直前イベントの種類・経過時間

uv run python scripts/build_features_v2.py
"""

from __future__ import annotations

import bisect
import math
from datetime import timedelta

import pandas as pd
from kloppy import sportec
from kloppy.domain import EventDataset, Frame, Player, TrackingDataset

from build_features import (
    MATCH_IDS,
    SUPPORT_RADIUS_M,
    SYNC_ERROR_FLAG_DISTANCE_M,
    attack_sign,
    frames_by_period,
    heading_vector,
    nearest_frame_index,
)
from build_labels import build_negative_samples, find_player

OUT_PATH = "data/tackling_features_v2.csv"

DENSITY_RADIUS_M = 10.0  # 「周辺の密集度」を数える半径


def lookback_frame_by(frames: list[Frame], ref_index: int, ref_ts, seconds: float) -> Frame | None:
    target_ts = ref_ts - type(ref_ts)(seconds=seconds)
    for i in range(ref_index, -1, -1):
        if frames[i].timestamp <= target_ts:
            return frames[i]
    return None


def angle_between(v1: tuple[float, float] | None, v2: tuple[float, float] | None) -> float | None:
    if v1 is None or v2 is None:
        return None
    cos_angle = max(-1.0, min(1.0, v1[0] * v2[0] + v1[1] * v2[1]))
    return math.degrees(math.acos(cos_angle))


def build_event_index(events: EventDataset) -> dict[int, tuple[list, list[str]]]:
    """period_id -> (timestamp昇順のtimedeltaリスト, 対応するevent_name(or型名)リスト)"""
    by_period: dict[int, list[tuple]] = {}
    for e in events.events:
        if e.timestamp < timedelta(0):
            continue
        name = getattr(e, "event_name", None) or type(e).__name__
        by_period.setdefault(e.period.id, []).append((e.timestamp, name))
    index: dict[int, tuple[list, list[str]]] = {}
    for period_id, items in by_period.items():
        items.sort(key=lambda x: x[0])
        index[period_id] = ([x[0] for x in items], [x[1] for x in items])
    return index


def prev_event_lookup(event_index: dict[int, tuple[list, list[str]]], period_id: int, ts) -> tuple[str | None, float | None]:
    ts_list, name_list = event_index.get(period_id, ([], []))
    i = bisect.bisect_left(ts_list, ts)
    if i == 0:
        return None, None
    prev_ts = ts_list[i - 1]
    prev_name = name_list[i - 1]
    return prev_name, (ts - prev_ts).total_seconds()


def compute_features_v2(
    events: EventDataset,
    tracking: TrackingDataset,
    frames_lookup: dict[int, list[Frame]],
    event_index: dict[int, tuple[list, list[str]]],
    match_id: str,
    period_id: int,
    ts,
    carrier: Player,
    opponent: Player,
    label: int,
) -> dict | None:
    frames = frames_lookup.get(period_id)
    if not frames:
        return None
    ref_i = nearest_frame_index(frames, ts)
    ref_frame = frames[ref_i]

    carrier_pt = ref_frame.players_coordinates.get(carrier)
    opponent_pt = ref_frame.players_coordinates.get(opponent)
    if carrier_pt is None or opponent_pt is None:
        return None

    carrier_data = ref_frame.players_data.get(carrier)
    opponent_data = ref_frame.players_data.get(opponent)
    carrier_speed = carrier_data.speed if carrier_data else None
    opponent_speed = opponent_data.speed if opponent_data else None

    frame_04 = lookback_frame_by(frames, ref_i, ref_frame.timestamp, 0.4)
    frame_08 = lookback_frame_by(frames, ref_i, ref_frame.timestamp, 0.8)
    frame_10 = lookback_frame_by(frames, ref_i, ref_frame.timestamp, 1.0)

    def pt_of(frame, player):
        return frame.players_coordinates.get(player) if frame else None

    carrier_04, opponent_04 = pt_of(frame_04, carrier), pt_of(frame_04, opponent)
    carrier_08, opponent_08 = pt_of(frame_08, carrier), pt_of(frame_08, opponent)
    carrier_10, opponent_10 = pt_of(frame_10, carrier), pt_of(frame_10, opponent)

    opponent_heading_late = heading_vector(opponent_pt, opponent_04)  # [-0.4s, 0]
    distance = math.hypot(carrier_pt.x - opponent_pt.x, carrier_pt.y - opponent_pt.y)

    approach_angle_deg = None
    if opponent_heading_late is not None:
        to_carrier = (carrier_pt.x - opponent_pt.x, carrier_pt.y - opponent_pt.y)
        norm = math.hypot(*to_carrier)
        if norm > 1e-6:
            to_carrier_unit = (to_carrier[0] / norm, to_carrier[1] / norm)
            cos_angle = max(-1.0, min(1.0, opponent_heading_late[0] * to_carrier_unit[0] + opponent_heading_late[1] * to_carrier_unit[1]))
            approach_angle_deg = math.degrees(math.acos(cos_angle))

    def dist_at(c_pt, o_pt):
        if c_pt is None or o_pt is None:
            return None
        return math.hypot(c_pt.x - o_pt.x, c_pt.y - o_pt.y)

    distance_04 = dist_at(carrier_04, opponent_04)
    distance_08 = dist_at(carrier_08, opponent_08)
    distance_10 = dist_at(carrier_10, opponent_10)

    closing_speed = None
    if distance_04 is not None and frame_04 is not None:
        dt = (ref_frame.timestamp - frame_04.timestamp).total_seconds()
        if dt > 1e-6:
            closing_speed = (distance_04 - distance) / dt

    closing_speed_early = None
    if distance_08 is not None and distance_04 is not None and frame_04 is not None and frame_08 is not None:
        dt_early = (frame_04.timestamp - frame_08.timestamp).total_seconds()
        if dt_early > 1e-6:
            closing_speed_early = (distance_08 - distance_04) / dt_early

    closing_accel_mps2 = None
    if closing_speed is not None and closing_speed_early is not None and frame_04 is not None:
        dt = (ref_frame.timestamp - frame_04.timestamp).total_seconds()
        if dt > 1e-6:
            closing_accel_mps2 = (closing_speed - closing_speed_early) / dt

    distance_trend_1s_mps = None
    if distance_10 is not None and frame_10 is not None:
        dt = (ref_frame.timestamp - frame_10.timestamp).total_seconds()
        if dt > 1e-6:
            distance_trend_1s_mps = (distance_10 - distance) / dt

    # 守備者の方向転換率: [-0.8,-0.4] の進行方向 vs [-0.4,0] の進行方向
    opponent_heading_early = heading_vector(opponent_04, opponent_08) if opponent_04 and opponent_08 else None
    opponent_heading_change_deg = angle_between(opponent_heading_early, opponent_heading_late)

    # 3番目に近い相手選手・周辺密集度
    opponent_dists = []
    for player, point in ref_frame.players_coordinates.items():
        if player.team == carrier.team:
            continue
        opponent_dists.append(math.hypot(point.x - carrier_pt.x, point.y - carrier_pt.y))
    opponent_dists.sort()
    second_nearest = opponent_dists[1] if len(opponent_dists) >= 2 else None
    third_nearest = opponent_dists[2] if len(opponent_dists) >= 3 else None
    n_opponents_nearby = sum(1 for d in opponent_dists if d <= DENSITY_RADIUS_M)

    # 後方サポート人数
    sign = attack_sign(ref_frame, carrier.team)
    n_supporting = 0
    for player, point in ref_frame.players_coordinates.items():
        if player.team != carrier.team or player == carrier:
            continue
        d = math.hypot(point.x - carrier_pt.x, point.y - carrier_pt.y)
        if d > SUPPORT_RADIUS_M:
            continue
        if (point.x - carrier_pt.x) * sign < 0:
            n_supporting += 1

    dist_to_sideline = 34.0 - abs(carrier_pt.y)
    goal_line_x = 52.5 * sign
    dist_to_goal_line = abs(goal_line_x - carrier_pt.x)

    prev_event_name, seconds_since_prev_event = prev_event_lookup(event_index, period_id, ts)
    prev_event_was_pass = 1 if prev_event_name == "pass" else 0

    return {
        "match_id": match_id,
        "period_id": period_id,
        "timestamp": str(ts),
        "carrier_id": carrier.player_id,
        "opponent_id": opponent.player_id,
        # --- 元の9特徴量 ---
        "distance_m": distance,
        "approach_angle_deg": approach_angle_deg,
        "carrier_speed_mps": carrier_speed,
        "opponent_speed_mps": opponent_speed,
        "closing_speed_mps": closing_speed,
        "second_nearest_dist_m": second_nearest,
        "n_supporting_teammates": n_supporting,
        "dist_to_sideline_m": dist_to_sideline,
        "dist_to_goal_line_m": dist_to_goal_line,
        # --- A. トレンド/加速度 ---
        "distance_trend_1s_mps": distance_trend_1s_mps,
        "closing_accel_mps2": closing_accel_mps2,
        # --- B. 守備者の方向転換率 ---
        "opponent_heading_change_deg": opponent_heading_change_deg,
        # --- C. 3番目に近い守備者・密集度 ---
        "third_nearest_dist_m": third_nearest,
        "n_opponents_nearby_10m": n_opponents_nearby,
        # --- D. 直前の選択履歴 ---
        "seconds_since_prev_event": seconds_since_prev_event,
        "prev_event_was_pass": prev_event_was_pass,
        # --- メタ ---
        "carrier_x": carrier_pt.x,
        "carrier_y": carrier_pt.y,
        "sync_error_suspect": distance > SYNC_ERROR_FLAG_DISTANCE_M,
        "label": label,
    }


def build_match_features(match_id: str) -> list[dict]:
    events = sportec.load_open_event_data(match_id=match_id, coordinates="sportec")
    tracking = sportec.load_open_tracking_data(match_id=match_id, coordinates="sportec")
    frames_lookup = frames_by_period(tracking)
    event_index = build_event_index(events)

    rows: list[dict] = []

    positives = [
        e
        for e in events.events
        if getattr(e, "event_name", None) == "TacklingGame" and e.raw_event.get("Type") == "ground"
    ]
    for e in positives:
        if e.timestamp < timedelta(0):
            continue
        winner = find_player(events, e.raw_event.get("Winner"))
        loser = find_player(events, e.raw_event.get("Loser"))
        if winner is None or loser is None:
            continue
        carrier = winner if e.raw_event.get("WinnerRole") == "withBallControl" else loser
        opponent = loser if carrier is winner else winner
        row = compute_features_v2(
            events, tracking, frames_lookup, event_index, match_id, e.period.id, e.timestamp, carrier, opponent, label=1
        )
        if row is not None:
            rows.append(row)

    negatives = build_negative_samples(match_id, events=events, tracking=tracking)
    for n in negatives:
        carrier = find_player(events, n.carrier_id)
        opponent = find_player(events, n.opponent_id)
        if carrier is None or opponent is None:
            continue
        row = compute_features_v2(
            events, tracking, frames_lookup, event_index, match_id, n.period_id, n.timestamp, carrier, opponent, label=0
        )
        if row is not None:
            rows.append(row)

    return rows


def main() -> None:
    all_rows: list[dict] = []
    for match_id in MATCH_IDS:
        rows = build_match_features(match_id)
        n_pos = sum(1 for r in rows if r["label"] == 1)
        n_neg = sum(1 for r in rows if r["label"] == 0)
        print(f"{match_id}: pos={n_pos} neg={n_neg}")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_PATH, index=False)
    print()
    print(f"saved {len(df)} rows to {OUT_PATH}")
    print(df["label"].value_counts())
    print()
    print("missing values per new column:")
    new_cols = [
        "distance_trend_1s_mps", "closing_accel_mps2", "opponent_heading_change_deg",
        "third_nearest_dist_m", "n_opponents_nearby_10m", "seconds_since_prev_event", "prev_event_was_pass",
    ]
    print(df[new_cols].isna().sum())


if __name__ == "__main__":
    main()
