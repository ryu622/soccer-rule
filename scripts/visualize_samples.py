"""正例(ground TacklingGame)・負例(見送り)サンプルをピッチ図上にプロットして目視確認する。

uv run python scripts/visualize_samples.py
"""

from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

from kloppy import sportec

from build_labels import build_negative_samples, find_player

MATCH_ID = "J03WPY"
N_SAMPLES_EACH = 3
OUT_PATH = "scratch_samples.png"

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


def nearest_frame(tracking, period_id, ts):
    candidates = [f for f in tracking.frames if f.period.id == period_id]
    return min(candidates, key=lambda f: abs(f.timestamp - ts))


def draw_pitch(ax):
    ax.add_patch(
        Rectangle((-PITCH_LENGTH / 2, -PITCH_WIDTH / 2), PITCH_LENGTH, PITCH_WIDTH, fill=False, color="gray")
    )
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.add_patch(Circle((0, 0), 9.15, fill=False, color="gray", linewidth=0.8))
    ax.set_xlim(-PITCH_LENGTH / 2 - 3, PITCH_LENGTH / 2 + 3)
    ax.set_ylim(-PITCH_WIDTH / 2 - 3, PITCH_WIDTH / 2 + 3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_frame(ax, frame, carrier, highlight_opponent, title):
    draw_pitch(ax)
    teams = list({p.team for p in frame.players_coordinates.keys()})
    team_color = {}
    if len(teams) >= 1:
        team_color[teams[0]] = "tab:blue"
    if len(teams) >= 2:
        team_color[teams[1]] = "tab:red"

    for player, point in frame.players_coordinates.items():
        color = team_color.get(player.team, "gray")
        ax.scatter(point.x, point.y, color=color, s=60, zorder=3)

    if frame.ball_coordinates is not None:
        ax.scatter(
            frame.ball_coordinates.x, frame.ball_coordinates.y, color="black", s=25, marker="o", zorder=4
        )

    carrier_point = frame.players_coordinates.get(carrier)
    if carrier_point is not None:
        ax.scatter(
            carrier_point.x, carrier_point.y, facecolors="none", edgecolors="gold", linewidths=2.5, s=220, zorder=5
        )

    if highlight_opponent is not None:
        opp_point = frame.players_coordinates.get(highlight_opponent)
        if opp_point is not None:
            ax.scatter(
                opp_point.x, opp_point.y, facecolors="none", edgecolors="lime", linewidths=2.5, s=220, zorder=5
            )

    ax.set_title(title, fontsize=9)


def main() -> None:
    events = sportec.load_open_event_data(match_id=MATCH_ID, coordinates="sportec")
    tracking = sportec.load_open_tracking_data(match_id=MATCH_ID, coordinates="sportec")

    positives = [
        e
        for e in events.events
        if getattr(e, "event_name", None) == "TacklingGame" and e.raw_event.get("Type") == "ground"
    ][:N_SAMPLES_EACH]

    negatives = build_negative_samples(MATCH_ID)[:N_SAMPLES_EACH]

    fig, axes = plt.subplots(2, N_SAMPLES_EACH, figsize=(5 * N_SAMPLES_EACH, 9))

    for i, e in enumerate(positives):
        winner = find_player(events, e.raw_event.get("Winner"))
        loser = find_player(events, e.raw_event.get("Loser"))
        carrier = winner if e.raw_event.get("WinnerRole") == "withBallControl" else loser
        opponent = loser if carrier is winner else winner
        frame = nearest_frame(tracking, e.period.id, e.timestamp)
        p1 = frame.players_coordinates.get(carrier)
        p2 = frame.players_coordinates.get(opponent)
        d = math.hypot(p1.x - p2.x, p1.y - p2.y) if p1 and p2 else float("nan")
        plot_frame(
            axes[0, i],
            frame,
            carrier,
            opponent,
            f"POS #{i+1}: {carrier.name} vs {opponent.name}\nt={e.timestamp} dist={d:.2f}m result={e.raw_event.get('WinnerResult')}",
        )

    for i, n in enumerate(negatives):
        carrier = find_player(events, n.carrier_id)
        frame = nearest_frame(tracking, n.period_id, n.timestamp)
        # recompute nearest opponent at this frame for highlighting
        best = None
        for player, point in frame.players_coordinates.items():
            if player.team == carrier.team:
                continue
            cp = frame.players_coordinates.get(carrier)
            d = math.hypot(point.x - cp.x, point.y - cp.y)
            if best is None or d < best[0]:
                best = (d, player)
        opponent = best[1] if best else None
        plot_frame(
            axes[1, i],
            frame,
            carrier,
            opponent,
            f"NEG #{i+1}: {carrier.name}\nt={n.timestamp} dist={n.nearest_opponent_distance:.2f}m (no tackle)",
        )

    fig.suptitle(f"{MATCH_ID}: positive (ground TacklingGame) vs negative (no tackle) samples", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=130)
    print(f"saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
