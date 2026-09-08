"""1v1デュエル(仕掛けるvs待つ)タスクの正例・負例ラベルを構築するプロトタイプ。

research_plan.md 3.3節「負例の定義」に対応。

正例(仕掛けた/TAKE_ON):
    TacklingGame イベントのうち Type == "ground"(地上での1v1)。
    ボール保持者(WinnerRole または LoserRole が "withBallControl" の側)が
    実際に相手と接触/対峙した瞬間。

負例(見送り):
    ボール保持スペル(同一選手が連続してボールを保持しているフレーム区間)の中で、
    最近接の相手選手との距離が閾値(PRESSURE_DISTANCE_M)以下になった最初の瞬間
    (「プレッシャー開始」)を機会窓の起点とする。
    その後 WINDOW_SECONDS 秒以内にそのボール保持者が関与する ground の
    TacklingGame イベントが発生しなければ、「見送った(待った)」負例として
    プレッシャー開始フレームを1件採用する。

閾値はJ03WPYの実際のTacklingGame(ground)発生時における
保持者-対峙相手間の距離分布(中央値約1.55m, p75約3.58m, p90約6.76m)を参考に、
「接触の瞬間」より一回り広い「間合いを詰められ始めた」範囲としてPRESSURE_DISTANCE_M=5.0mを採用。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from kloppy import sportec
from kloppy.domain import EventDataset, Frame, Player, TrackingDataset

PRESSURE_DISTANCE_M = 5.0  # この距離以下を「プレッシャー下」とみなす
WINDOW_SECONDS = 1.0  # プレッシャー開始からこの秒数以内にTacklingGameが無ければ「見送り」
MIN_CARRY_RADIUS_M = 2.5  # ボールとの距離がこれ以下なら「保持している」とみなす
MAX_BALL_HEIGHT_M = 0.3  # ボール高さがこれ以下のフレームのみ「地上でのボール保持」とみなす(ヘディング等の空中戦を除外)
MIN_SPELL_FRAMES = 5  # 0.2秒未満の保持スペルはノイズとして除外


@dataclass
class NegativeSample:
    match_id: str
    period_id: int
    timestamp: object
    carrier_id: str
    carrier_name: str
    opponent_id: str
    opponent_name: str
    nearest_opponent_distance: float


def find_player(events: EventDataset, player_id: str) -> Player | None:
    for team in events.metadata.teams:
        for p in team.players:
            if p.player_id == player_id:
                return p
    return None


def nearest_opponent_distance(frame: Frame, carrier: Player) -> tuple[float, Player] | None:
    carrier_point = frame.players_coordinates.get(carrier)
    if carrier_point is None:
        return None
    best = None
    for player, point in frame.players_coordinates.items():
        if player.team == carrier.team:
            continue
        d = math.hypot(point.x - carrier_point.x, point.y - carrier_point.y)
        if best is None or d < best[0]:
            best = (d, player)
    return best


def build_carrier_per_frame(tracking: TrackingDataset) -> list[tuple[Frame, Player | None]]:
    result = []
    for frame in tracking.frames:
        if frame.ball_state is None or str(frame.ball_state) != "BallState.ALIVE":
            result.append((frame, None))
            continue
        owning_team = frame.ball_owning_team
        if owning_team is None or frame.ball_coordinates is None:
            result.append((frame, None))
            continue
        ball_z = frame.ball_coordinates.z
        if ball_z is not None and ball_z > MAX_BALL_HEIGHT_M:
            result.append((frame, None))
            continue
        best_player, best_dist = None, None
        for player, point in frame.players_coordinates.items():
            if player.team != owning_team:
                continue
            d = math.hypot(
                point.x - frame.ball_coordinates.x, point.y - frame.ball_coordinates.y
            )
            if best_dist is None or d < best_dist:
                best_dist, best_player = d, player
        if best_player is not None and best_dist <= MIN_CARRY_RADIUS_M:
            result.append((frame, best_player))
        else:
            result.append((frame, None))
    return result


def segment_spells(carrier_frames: list[tuple[Frame, Player | None]]):
    """連続して同一選手が保持しているフレーム区間(スペル)に分割する。周期(period)をまたがない。"""
    spells = []
    current_player = None
    current_period = None
    current_frames: list[Frame] = []
    for frame, player in carrier_frames:
        if player is not None and player is current_player and frame.period.id == current_period:
            current_frames.append(frame)
        else:
            if current_player is not None and len(current_frames) >= MIN_SPELL_FRAMES:
                spells.append((current_player, current_period, current_frames))
            current_player = player
            current_period = frame.period.id if player is not None else None
            current_frames = [frame] if player is not None else []
    if current_player is not None and len(current_frames) >= MIN_SPELL_FRAMES:
        spells.append((current_player, current_period, current_frames))
    return spells


def carrier_involved_in_ground_tackle(
    events: EventDataset, carrier: Player, period_id: int, start_ts, end_ts
) -> bool:
    for e in events.events:
        if getattr(e, "event_name", None) != "TacklingGame":
            continue
        if e.raw_event.get("Type") != "ground":
            continue
        if e.period.id != period_id:
            continue
        if not (start_ts <= e.timestamp <= end_ts):
            continue
        winner_id = e.raw_event.get("Winner")
        loser_id = e.raw_event.get("Loser")
        if carrier.player_id in (winner_id, loser_id):
            return True
    return False


def build_negative_samples(
    match_id: str, events: EventDataset | None = None, tracking: TrackingDataset | None = None
) -> list[NegativeSample]:
    if events is None:
        events = sportec.load_open_event_data(match_id=match_id, coordinates="sportec")
    if tracking is None:
        tracking = sportec.load_open_tracking_data(match_id=match_id, coordinates="sportec")

    carrier_frames = build_carrier_per_frame(tracking)
    spells = segment_spells(carrier_frames)

    negatives: list[NegativeSample] = []
    for carrier, period_id, frames in spells:
        onset_frame = None
        onset_dist = None
        onset_opponent = None
        for frame in frames:
            res = nearest_opponent_distance(frame, carrier)
            if res is None:
                continue
            dist, opponent = res
            if dist <= PRESSURE_DISTANCE_M:
                onset_frame = frame
                onset_dist = dist
                onset_opponent = opponent
                break
        if onset_frame is None:
            continue

        window_end = onset_frame.timestamp + type(onset_frame.timestamp)(seconds=WINDOW_SECONDS)
        if carrier_involved_in_ground_tackle(
            events, carrier, period_id, onset_frame.timestamp, window_end
        ):
            continue

        negatives.append(
            NegativeSample(
                match_id=match_id,
                period_id=period_id,
                timestamp=onset_frame.timestamp,
                carrier_id=carrier.player_id,
                carrier_name=carrier.name,
                opponent_id=onset_opponent.player_id,
                opponent_name=onset_opponent.name,
                nearest_opponent_distance=onset_dist,
            )
        )
    return negatives


def main() -> None:
    match_id = "J03WPY"
    events = sportec.load_open_event_data(match_id=match_id, coordinates="sportec")
    n_positive = sum(
        1
        for e in events.events
        if getattr(e, "event_name", None) == "TacklingGame" and e.raw_event.get("Type") == "ground"
    )

    negatives = build_negative_samples(match_id)

    print(f"[{match_id}] positive (ground TacklingGame): {n_positive}")
    print(f"[{match_id}] negative (見送り) candidates: {len(negatives)}")
    print(f"  class balance (pos:neg) = 1:{len(negatives)/n_positive:.2f}")
    print()
    print("negative sample preview:")
    for n in negatives[:10]:
        print(f"  period={n.period_id} t={n.timestamp} carrier={n.carrier_name} dist={n.nearest_opponent_distance:.2f}m")


if __name__ == "__main__":
    main()
