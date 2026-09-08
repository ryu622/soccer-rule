"""1v1デュエル(仕掛けるvs待つ)タスクの特徴量テーブルを構築する。

research_plan.md 3.2節「データ最適化」・4.2節に対応。全7試合について
正例(ground TacklingGame)・負例(見送り、build_labels.py)それぞれの
基準フレームで特徴量を計算し、DataFrameとして data/tackling_features.csv に保存する。

特徴量:
    distance_m              間合い(保持者-最近接相手の距離)
    approach_angle_deg      進入角度(相手の進行方向が保持者にどれだけ正対しているか。0°=直進で接近)
    carrier_speed_mps       保持者の速度
    opponent_speed_mps      相手の速度
    closing_speed_mps       間合いが詰まる速さ(正=接近中、負=離れている)
    second_nearest_dist_m   2番目に近い相手選手との距離(空間的余裕の目安)
    n_supporting_teammates  保持者より後方(自陣側)SUPPORT_RADIUS_M以内にいる味方の数
    dist_to_sideline_m      タッチラインまでの距離
    dist_to_goal_line_m     攻撃方向のゴールラインまでの距離
    carrier_x, carrier_y    保持者のピッチ座標(中心原点)
    label                   1=仕掛けた(正例) / 0=見送った(負例)
    match_id, period_id, timestamp, carrier_id, opponent_id  識別・分割用メタ情報
"""

from __future__ import annotations

import math
from datetime import timedelta

import pandas as pd
from kloppy import sportec
from kloppy.domain import EventDataset, Frame, Player, TrackingDataset

from build_labels import build_negative_samples, find_player

MATCH_IDS = ["J03WPY", "J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WQQ", "J03WR9"]

HEADING_LOOKBACK_SECONDS = 0.4  # 進行方向・詰め速度を推定するための遡り時間
SUPPORT_RADIUS_M = 15.0  # 後方サポートとみなす半径
SYNC_ERROR_FLAG_DISTANCE_M = 10.0  # これを超えるdistance_mは1v1として物理的に不自然であり、
# 先行研究(Bassek et al., 2025)が報告するトラッキング-イベント間の同期誤差(平均2.61±3.60m, 最大75.35m)
# に由来するノイズの可能性が高い。除外はせずフラグ列として残し、モデリング時に判断できるようにする。
OUT_PATH = "data/tackling_features.csv"


def frames_by_period(tracking: TrackingDataset) -> dict[int, list[Frame]]:
    result: dict[int, list[Frame]] = {}
    for frame in tracking.frames:
        result.setdefault(frame.period.id, []).append(frame)
    for frames in result.values():
        frames.sort(key=lambda f: f.timestamp)
    return result


def nearest_frame_index(frames: list[Frame], ts) -> int:
    # frames はtimestamp昇順なので二分探索でもよいが、件数が少ないためlinear minで十分
    best_i, best_d = 0, None
    for i, f in enumerate(frames):
        d = abs(f.timestamp - ts)
        if best_d is None or d < best_d:
            best_d, best_i = d, i
    return best_i


def lookback_frame(frames: list[Frame], ref_index: int, ref_ts) -> Frame | None:
    target_ts = ref_ts - type(ref_ts)(seconds=HEADING_LOOKBACK_SECONDS)
    for i in range(ref_index, -1, -1):
        if frames[i].timestamp <= target_ts:
            return frames[i]
    return None


def heading_vector(ref_point, past_point) -> tuple[float, float] | None:
    if past_point is None:
        return None
    dx, dy = ref_point.x - past_point.x, ref_point.y - past_point.y
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return None
    return dx / norm, dy / norm


def attack_sign(frame: Frame, team) -> float:
    """teamの攻撃方向が+x側なら+1, -x側なら-1を返す。"""
    home_team_is_rtl = str(frame.attacking_direction) == "AttackingDirection.RTL"
    home_forward_sign = -1.0 if home_team_is_rtl else 1.0
    if str(team.ground) == "home":
        return home_forward_sign
    return -home_forward_sign


def compute_features(
    events: EventDataset,
    tracking: TrackingDataset,
    frames_lookup: dict[int, list[Frame]],
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

    past_frame = lookback_frame(frames, ref_i, ref_frame.timestamp)
    carrier_past_pt = past_frame.players_coordinates.get(carrier) if past_frame else None
    opponent_past_pt = past_frame.players_coordinates.get(opponent) if past_frame else None

    opponent_heading = heading_vector(opponent_pt, opponent_past_pt)
    distance = math.hypot(carrier_pt.x - opponent_pt.x, carrier_pt.y - opponent_pt.y)

    approach_angle_deg = None
    if opponent_heading is not None:
        to_carrier = (carrier_pt.x - opponent_pt.x, carrier_pt.y - opponent_pt.y)
        norm = math.hypot(*to_carrier)
        if norm > 1e-6:
            to_carrier_unit = (to_carrier[0] / norm, to_carrier[1] / norm)
            cos_angle = max(-1.0, min(1.0, opponent_heading[0] * to_carrier_unit[0] + opponent_heading[1] * to_carrier_unit[1]))
            approach_angle_deg = math.degrees(math.acos(cos_angle))

    closing_speed = None
    if past_frame is not None and carrier_past_pt is not None and opponent_past_pt is not None:
        past_distance = math.hypot(carrier_past_pt.x - opponent_past_pt.x, carrier_past_pt.y - opponent_past_pt.y)
        dt = (ref_frame.timestamp - past_frame.timestamp).total_seconds()
        if dt > 1e-6:
            closing_speed = (past_distance - distance) / dt

    # 2番目に近い相手選手
    opponent_dists = []
    for player, point in ref_frame.players_coordinates.items():
        if player.team == carrier.team:
            continue
        opponent_dists.append(math.hypot(point.x - carrier_pt.x, point.y - carrier_pt.y))
    opponent_dists.sort()
    second_nearest = opponent_dists[1] if len(opponent_dists) >= 2 else None

    # 後方サポート人数
    sign = attack_sign(ref_frame, carrier.team)
    n_supporting = 0
    for player, point in ref_frame.players_coordinates.items():
        if player.team != carrier.team or player == carrier:
            continue
        d = math.hypot(point.x - carrier_pt.x, point.y - carrier_pt.y)
        if d > SUPPORT_RADIUS_M:
            continue
        # 自陣側(ゴール方向と逆)にいる = (player.x - carrier.x) * sign < 0
        if (point.x - carrier_pt.x) * sign < 0:
            n_supporting += 1

    dist_to_sideline = 34.0 - abs(carrier_pt.y)
    goal_line_x = 52.5 * sign
    dist_to_goal_line = abs(goal_line_x - carrier_pt.x)

    return {
        "match_id": match_id,
        "period_id": period_id,
        "timestamp": str(ts),
        "carrier_id": carrier.player_id,
        "opponent_id": opponent.player_id,
        "distance_m": distance,
        "approach_angle_deg": approach_angle_deg,
        "carrier_speed_mps": carrier_speed,
        "opponent_speed_mps": opponent_speed,
        "closing_speed_mps": closing_speed,
        "second_nearest_dist_m": second_nearest,
        "n_supporting_teammates": n_supporting,
        "dist_to_sideline_m": dist_to_sideline,
        "dist_to_goal_line_m": dist_to_goal_line,
        "carrier_x": carrier_pt.x,
        "carrier_y": carrier_pt.y,
        "sync_error_suspect": distance > SYNC_ERROR_FLAG_DISTANCE_M,
        "label": label,
    }


def build_match_features(match_id: str) -> list[dict]:
    events = sportec.load_open_event_data(match_id=match_id, coordinates="sportec")
    tracking = sportec.load_open_tracking_data(match_id=match_id, coordinates="sportec")
    frames_lookup = frames_by_period(tracking)

    rows: list[dict] = []

    positives = [
        e
        for e in events.events
        if getattr(e, "event_name", None) == "TacklingGame" and e.raw_event.get("Type") == "ground"
    ]
    for e in positives:
        if e.timestamp < timedelta(0):
            # 生データの一部に期間開始前を指す不正なタイムスタンプが付与された
            # (後から挿入された訂正イベントと思われる)ものがあり、
            # トラッキングフレームとの対応付けが破綻するため除外する。
            continue
        winner = find_player(events, e.raw_event.get("Winner"))
        loser = find_player(events, e.raw_event.get("Loser"))
        if winner is None or loser is None:
            continue
        carrier = winner if e.raw_event.get("WinnerRole") == "withBallControl" else loser
        opponent = loser if carrier is winner else winner
        row = compute_features(
            events, tracking, frames_lookup, match_id, e.period.id, e.timestamp, carrier, opponent, label=1
        )
        if row is not None:
            rows.append(row)

    negatives = build_negative_samples(match_id, events=events, tracking=tracking)
    for n in negatives:
        carrier = find_player(events, n.carrier_id)
        opponent = find_player(events, n.opponent_id)
        if carrier is None or opponent is None:
            continue
        row = compute_features(
            events, tracking, frames_lookup, match_id, n.period_id, n.timestamp, carrier, opponent, label=0
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
    print(f"sync_error_suspect (distance_m > {SYNC_ERROR_FLAG_DISTANCE_M}m): {df['sync_error_suspect'].sum()} rows")
    print(df.groupby("label")["sync_error_suspect"].sum())
    print()
    print(df.describe())


if __name__ == "__main__":
    main()
