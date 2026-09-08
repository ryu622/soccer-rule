"""idsse-data の TacklingGame(1v1デュエル)イベントを確認するための探索スクリプト。

research_plan.md 4.2節の1〜3に対応:
  1. TacklingGame イベントのカラム構成確認
  2. 件数・クラス分布(仕掛けた/待った、および付随するWinnerResult等)確認
  3. トラッキング-イベント間の同期誤差の分布確認

uv run python scripts/explore_tackling_game.py
"""

from __future__ import annotations

import math
from collections import Counter

from kloppy import sportec
from kloppy.domain import EventDataset, TrackingDataset

MATCH_IDS = [
    "J03WPY",
    "J03WMX",
    "J03WN1",
    "J03WOH",
    "J03WOY",
    "J03WQQ",
    "J03WR9",
]


def summarize_tackling_events(events: EventDataset, match_id: str) -> list:
    tackling = [e for e in events.events if getattr(e, "event_name", None) == "TacklingGame"]
    type_dist = Counter(e.raw_event.get("Type") for e in tackling)
    result_dist = Counter(e.raw_event.get("WinnerResult") for e in tackling)
    print(f"[{match_id}] total_events={len(events.events)} tackling={len(tackling)}")
    print(f"  Type: {dict(type_dist)}")
    print(f"  WinnerResult: {dict(result_dist)}")
    return tackling


def find_player(events: EventDataset, player_id: str):
    for team in events.metadata.teams:
        for player in team.players:
            if player.player_id == player_id:
                return player
    return None


def sync_error_sample(match_id: str, tackling_events: list, max_samples: int = 60) -> list[float]:
    """TacklingGame イベントの記録座標と、その時刻のWinner選手のトラッキング座標との距離(m)を計算する。"""
    tracking: TrackingDataset = sportec.load_open_tracking_data(
        match_id=match_id, coordinates="sportec"
    )

    # period -> sorted frames (timestamp is a timedelta relative to period start)
    frames_by_period: dict[int, list] = {}
    for frame in tracking.frames:
        frames_by_period.setdefault(frame.period.id, []).append(frame)
    for period_frames in frames_by_period.values():
        period_frames.sort(key=lambda f: f.timestamp)

    distances = []
    events = sportec.load_open_event_data(match_id=match_id, coordinates="sportec")

    sampled = tackling_events[:max_samples]
    for e in sampled:
        winner_id = e.raw_event.get("Winner")
        winner = find_player(events, winner_id)
        if winner is None:
            continue
        period_frames = frames_by_period.get(e.period.id, [])
        if not period_frames:
            continue
        # nearest frame by timestamp (linear scan is fine for a small sample)
        nearest = min(period_frames, key=lambda f: abs(f.timestamp - e.timestamp))
        player_point = nearest.players_coordinates.get(winner)
        if player_point is None:
            continue
        # raw event X-Position/Y-Position are corner-origin (0-105, 0-68);
        # tracking data loaded with coordinates="sportec" is centered (-52.5..52.5, -34..34).
        event_x = float(e.raw_event["X-Position"]) - 52.5
        event_y = float(e.raw_event["Y-Position"]) - 34.0
        dx = player_point.x - event_x
        dy = player_point.y - event_y
        distances.append(math.hypot(dx, dy))

    return distances


def main() -> None:
    all_tackling_counts = []
    for match_id in MATCH_IDS:
        events = sportec.load_open_event_data(match_id=match_id, coordinates="sportec")
        tackling = summarize_tackling_events(events, match_id)
        all_tackling_counts.append(len(tackling))

    print()
    print(f"total TacklingGame events across {len(MATCH_IDS)} matches: {sum(all_tackling_counts)}")

    # sync error check on one match (first) as a representative sample
    first_match = MATCH_IDS[0]
    events = sportec.load_open_event_data(match_id=first_match, coordinates="sportec")
    tackling = [e for e in events.events if getattr(e, "event_name", None) == "TacklingGame"]
    print()
    print(f"[{first_match}] computing tracking-event sync error for {min(60, len(tackling))} sample events...")
    distances = sync_error_sample(first_match, tackling)
    if distances:
        mean = sum(distances) / len(distances)
        var = sum((d - mean) ** 2 for d in distances) / len(distances)
        std = math.sqrt(var)
        print(f"  n={len(distances)} mean={mean:.2f}m std={std:.2f}m max={max(distances):.2f}m min={min(distances):.2f}m")
    else:
        print("  no distances computed (player/frame lookup failed)")


if __name__ == "__main__":
    main()
