"""Source-backed football plots and a data-only completed-pass network."""

import json
from collections import defaultdict
import numpy as np
from app.analytics.tactics import location
from app.analytics.tactical_relations import clock
from .pitch import draw_pitch, direction_arrow, save_figure
from .style import COLORS, apply_style, decorate

colors = [COLORS["blue"], COLORS["orange"]]


def completed_pass_network(events, team_id, network_window=None):
    positions = defaultdict(list)
    edges = defaultdict(int)
    for event in events:
        if (event.get("team") or {}).get("id") != team_id or (
            event.get("type") or {}
        ).get("name") != "Pass":
            continue
        if network_window is not None:
            seconds = clock(event)
            if (
                event.get("period") != 1
                or seconds is None
                or not network_window[0] * 60 <= seconds < network_window[1] * 60
            ):
                continue
        details = event.get("pass") or {}
        point = location(event.get("location"))
        player = (event.get("player") or {}).get("id")
        receiver = (details.get("recipient") or {}).get("id")
        if not details.get("outcome") and point and player:
            positions[player].append(point)
            if receiver:
                edges[(player, receiver)] += 1
    centroids = {key: np.mean(points, axis=0) for key, points in positions.items()}
    return positions, edges, centroids


def shot_map(events, chosen, *, scouting_root, output_dir):
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    apply_style()
    teams = chosen["teams"][:2]
    located_shots = [location(event.get("location")) for event in events
                     if (event.get("type") or {}).get("name") == "Shot" and event.get("period") != 5]
    located_shots = [point for point in located_shots if point is not None]
    half_pitch = bool(located_shots) and min(point[0] for point in located_shots) >= 60
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 8.3) if half_pitch else (14, 6.8))
    decorate(fig, "Shot locations and supplied xG",
             f"{chosen['home_team']} {chosen['home_score']}–{chosen['away_score']} {chosen['away_team']}  ·  {chosen['date']}  ·  supplied StatsBomb xG")
    fig.subplots_adjust(left=.055, right=.96, top=.79, bottom=.25, wspace=.13)
    for ax, team, color in zip(axes, teams, colors):
        draw_pitch(ax)
        if half_pitch:
            ax.set_xlim(60, 122.5)
        goals = []
        for event in events:
            if ((event.get("team") or {}).get("id") != team["team_id"]
                or (event.get("type") or {}).get("name") != "Shot"
                or event.get("period") == 5):
                continue
            point = location(event.get("location"))
            if point is None:
                continue
            shot = event.get("shot") or {}
            goal = (shot.get("outcome") or {}).get("name") == "Goal"
            ax.scatter(*point, s=35 + 900 * shot.get("statsbomb_xg", 0),
                       facecolor=color if goal else "white", edgecolor=color,
                       lw=1.6, alpha=.95, zorder=4 if goal else 3)
            if goal:
                goals.append((event, point))
        label_goals = sorted(goals, key=lambda item: item[1][1]) if half_pitch else goals
        for index, (event, point) in enumerate(label_goals):
            name = (event.get("player") or {}).get("name", "Goal")
            goal_label = f"{name}\n{event['minute']:02d}:{event['second']:02d}"
            ax.annotate(goal_label, point,
                        xytext=(63, 8 + index * min(14, 64 / max(1, len(goals) - 1))) if half_pitch else (-12, 20 if index % 2 == 0 else -24),
                        textcoords="data" if half_pitch else "offset points", ha="left" if half_pitch else "right", fontsize=10,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 2},
                        arrowprops={"arrowstyle": "-", "lw": .7, "color": color}, zorder=5)
        ax.set_title(f"{team['team']}  ·  {team['shots']} shots  ·  {team['xg']:.2f} xG",
                     loc="left", color=color, pad=12)
        direction_arrow(ax)
    key = fig.add_axes([.055, .13, .91, .065])
    key.axis("off")
    key.text(0, .5, "xG per shot", va="center", fontsize=11, transform=key.transAxes)
    for x, value in zip([.16, .28, .40], [.05, .20, .50]):
        key.scatter(x, .5, s=35 + 900 * value, facecolor="white", edgecolor=COLORS["muted"], transform=key.transAxes)
        key.text(x + .025, .5, f"{value:.2f}", va="center", transform=key.transAxes)
    key.legend(handles=[Line2D([], [], marker="o", linestyle="none", markerfacecolor=COLORS["ink"],
                               markeredgecolor=COLORS["ink"], markersize=8, label="Goal"),
                        Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
                               markeredgecolor=COLORS["ink"], markersize=8, label="Other shot")],
               loc="center right", ncol=2, frameon=False, borderaxespad=0)
    view_note = "Attacking halves shown; all located shots retained. " if half_pitch else "Full pitches shown. "
    return save_figure(fig, "shot-map",
                       "Source: StatsBomb Open Data. Circle area encodes supplied xG with a minimum visible size; filled circles are goals.\n"
                       + view_note + "Both teams attack right in 120 × 80 provider units.\nGoal labels use elapsed match clock (MM:SS). Shootouts excluded.",
                       scouting_root=scouting_root, output_dir=output_dir)


def _number_tag_positions(centroids):
    """Move number tags only; dots and pass arrows retain measured centroids."""
    ids = list(centroids)
    if not ids:
        return {}
    anchors = np.asarray([centroids[key] for key in ids], dtype=float)
    tags = anchors + np.array([0, -4.5])
    for _ in range(100):
        move = (anchors + [0, -4.5] - tags) * .025
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                delta = tags[a] - tags[b]
                distance = np.linalg.norm(delta)
                if distance < 9:
                    direction = delta / distance if distance > .01 else np.array([1., 0.])
                    shift = direction * (9 - distance) * .5
                    move[a] += shift
                    move[b] -= shift
        tags += move
        tags[:, 0] = np.clip(tags[:, 0], 3, 117)
        tags[:, 1] = np.clip(tags[:, 1], 3, 77)
    return dict(zip(ids, tags))


def passing_network(events, chosen, lineup_names, network_window, *, scouting_root, output_dir):
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch

    apply_style()
    teams = chosen["teams"][:2]
    networks = [completed_pass_network(events, team["team_id"], network_window) for team in teams]
    max_players = max((sum(len(positions[p]) >= 3 for p in centroids)
                       for positions, _, centroids in networks), default=0)
    roster_rows = max(1, int(np.ceil(max_players / 2)))
    fig = plt.figure(figsize=(14, 7.8 + max(0, roster_rows - 6) * .28))
    grid = fig.add_gridspec(2, 2, height_ratios=[4.1, max(1.5, roster_rows * .24)])
    phase = (f"First half, {network_window[0]}–{network_window[1]} minutes"
             if network_window else "Full match, including substitutes")
    decorate(fig, "Who connects the passing game?", phase + "  ·  completed passes only")
    fig.subplots_adjust(left=.055, right=.96, top=.80, bottom=.19, wspace=.13, hspace=.42)
    for side, (team, color, network) in enumerate(zip(teams, colors, networks)):
        ax = fig.add_subplot(grid[0, side])
        roster_ax = fig.add_subplot(grid[1, side])
        roster_ax.axis("off")
        draw_pitch(ax)
        positions, edges, centroids = network
        displayed = {p: q for p, q in centroids.items() if len(positions[p]) >= 3}
        for (a, b), count in edges.items():
            if count >= 3 and a in centroids and b in centroids:
                ax.add_patch(FancyArrowPatch(centroids[a], centroids[b], arrowstyle="-|>",
                             connectionstyle="arc3,rad=.16", mutation_scale=10,
                             shrinkA=6, shrinkB=6, color=color, alpha=.55,
                             lw=.6 + count / 6, zorder=2))
        tags = _number_tag_positions(displayed)
        roster = []
        for player, point in displayed.items():
            ax.scatter(*point, s=48, color=color, edgecolor="white", lw=.6, zorder=4)
            name, jersey = lineup_names.get(player, (str(player), ""))
            number = str(jersey) if jersey is not None and str(jersey) else str(player)
            tag = tags[player]
            ax.annotate(number, point, xytext=tag, textcoords="data", ha="center", va="center",
                        fontsize=10, color=color, weight="semibold",
                        bbox={"boxstyle": "round,pad=.18", "facecolor": "white", "edgecolor": color, "lw": .8},
                        arrowprops={"arrowstyle": "-", "color": color, "lw": .65}, zorder=5)
            roster.append((number, name))
        roster.sort(key=lambda row: (int(row[0]) if row[0].isdigit() else 999, row[1]))
        for index, (number, name) in enumerate(roster):
            column, row = divmod(index, roster_rows)
            roster_ax.text(column * .52, 1 - row / max(roster_rows, 1), f"{number:>2}  {name}",
                           transform=roster_ax.transAxes, va="top", fontsize=10, color=COLORS["ink"])
        if not roster:
            roster_ax.text(0, .8, "No players meet the display threshold", color=COLORS["muted"], fontsize=10)
        ax.set_title(team["team"], loc="left", color=color, pad=12)
        direction_arrow(ax)
    fig.legend(handles=[Line2D([], [], color=COLORS["muted"], lw=.6 + n / 6, label=f"{n} passes") for n in [3, 10]],
               loc="lower left", bbox_to_anchor=(.05, .105), ncol=2, frameon=False, handlelength=3)
    return save_figure(fig, "passing-network",
                       "Source: StatsBomb Open Data. Dots = mean completed-pass origin; number tags are moved for legibility. Nodes require ≥3 origins.\nArrows show ≥3 passes in that direction; width encodes count. Mean ball-action locations do not establish a fixed formation.",
                       scouting_root=scouting_root, output_dir=output_dir)


def visible_space(events, frames, *, scouting_root, output_dir, chosen=None):
    from matplotlib import pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    from matplotlib.lines import Line2D

    apply_style()
    frame_map = {f["event_uuid"]: f for f in frames}
    candidates = []
    for event in events:
        if event.get("id") not in frame_map or (event.get("type") or {}).get("name") != "Pass":
            continue
        start = location(event.get("location"))
        end = location((event.get("pass") or {}).get("end_location"))
        if start and end and not (event.get("pass") or {}).get("outcome"):
            candidates.append((end[0] - start[0], event, start, end))
    if not candidates:
        return []
    _, event, start, end = max(candidates, key=lambda item: item[0])
    frame = frame_map[event["id"]]
    team_id = (event.get("team") or {}).get("id")
    team_colors = {team["team_id"]: color for team, color in zip(chosen["teams"][:2], colors)} if chosen else {}
    teammate_color = team_colors.get(team_id, colors[0])
    opponent_color = next((color for identity, color in team_colors.items() if identity != team_id), colors[1])
    fig = plt.figure(figsize=(13, 7.3))
    ax = fig.add_axes([.055, .20, .67, .59])
    info = fig.add_axes([.765, .22, .21, .54])
    info.axis("off")
    decorate(fig, "The camera sees only part of the pitch",
             f"{(event.get('team') or {}).get('name', '')}  ·  {(event.get('player') or {}).get('name', '')}  ·  {event.get('minute', 0)}:{event.get('second', 0):02d}")
    draw_pitch(ax)
    area = frame.get("visible_area", [])
    if len(area) >= 6 and len(area) % 2 == 0:
        ax.add_patch(Rectangle((0, 0), 120, 80, color="#DFE5EB", zorder=.1))
        field = Rectangle((0, 0), 120, 80, transform=ax.transData)
        polygon = Polygon(np.asarray(area).reshape(-1, 2), facecolor="white",
                          edgecolor=COLORS["muted"], lw=1.1, zorder=.2)
        polygon.set_clip_path(field)
        ax.add_patch(polygon)
    counts = []
    for teammate, color, marker in [(True, teammate_color, "o"), (False, opponent_color, "s")]:
        points = [location(p.get("location")) for p in frame["freeze_frame"] if p["teammate"] == teammate]
        points = [p for p in points if p]
        counts.append(len(points))
        if points:
            ax.scatter(*zip(*points), s=80, color=color, marker=marker, edgecolor="white", lw=.8, zorder=3)
    ax.scatter(*start, s=140, marker="*", color=COLORS["ink"], edgecolor="white", lw=.7, zorder=5)
    ax.scatter(*end, s=130, facecolor="none", edgecolor=COLORS["ink"], lw=1.5, zorder=4)
    ax.annotate("", end, start, arrowprops={"arrowstyle": "->", "color": COLORS["ink"], "lw": 1.8}, zorder=4)
    direction_arrow(ax)
    info.text(0, 1, "Visible in this frame", fontsize=12, weight="semibold", va="top")
    handles = [Line2D([], [], marker="o", color="none", markerfacecolor=teammate_color, markersize=8,
                      label=f"Teammates  ·  {counts[0]}"),
               Line2D([], [], marker="s", color="none", markerfacecolor=opponent_color, markersize=8,
                      label=f"Opponents  ·  {counts[1]}"),
               Line2D([], [], marker="*", color="none", markerfacecolor=COLORS["ink"], markersize=11, label="Pass origin"),
               Line2D([], [], marker="o", color="none", markerfacecolor="none", markeredgecolor=COLORS["ink"],
                      markersize=9, label="Recorded endpoint")]
    info.legend(handles=handles, loc="upper left", bbox_to_anchor=(-.06, .91), frameon=False,
                borderaxespad=0, handletextpad=.7, labelspacing=1.0, fontsize=11)
    info.text(0, .42, "White: camera field of view\nGrey: outside camera view" if area else "Camera field of view unavailable",
              fontsize=11, va="top", linespacing=1.7)
    info.text(0, .21, "Only observed players are shown.\nUnseen positions remain unknown.", fontsize=10,
              va="top", linespacing=1.6, color=COLORS["muted"])
    return save_figure(fig, "visible-space",
                       "Source: StatsBomb 360 / Open Data. Illustrative frame: the match's largest forward completed pass by x displacement.\nPlayer locations are event-time snapshots, not trajectories or pitch control. Coordinates: 120 × 80 provider units.",
                       scouting_root=scouting_root, output_dir=output_dir)


def running_position(*, scouting_root, output_dir):
    from matplotlib import pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    apply_style()
    sample = json.loads((scouting_root / "movement" / "metrica-1.json").read_text())
    candidates = [row for row in sample["players"] if row["team"] == "home" and row["detected_samples"] > 1000]
    player = max(candidates, key=lambda row: row["distance_m"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.4))
    decorate(fig, "Recorded positions of an anonymous player",
             f"Metrica sample 1  ·  {player['name']}  ·  {player['distance_m'] / 1000:.2f} km from recorded coordinates  ·  {player['detected_samples']:,} finite-coordinate samples")
    fig.subplots_adjust(left=.055, right=.96, top=.78, bottom=.29, wspace=.13)
    for ax in axes:
        draw_pitch(ax, 105, 68)
    grid = np.asarray(player["occupancy_grid"], dtype=float)
    grid = 100 * grid / max(1.0, grid.sum())
    cmap = LinearSegmentedColormap.from_list("position_share", ["#FFFFFF", "#CFDEEC", COLORS["blue"]])
    image = axes[0].imshow(grid, extent=(0, 105, 68, 0), origin="upper", cmap=cmap,
                           vmin=0, vmax=grid.max(), interpolation="nearest", zorder=.2)
    trajectory = player["trajectory_example"]
    segments, current = [], []
    for point in trajectory:
        if current and point[0] - current[-1][0] > 1.0:
            segments.append(current)
            current = []
        current.append(point)
    if current:
        segments.append(current)
    for segment in segments:
        axes[1].plot([point[1] + 52.5 for point in segment], [point[2] + 34 for point in segment],
                     color=colors[0], lw=1.5, alpha=.85, zorder=2)
    if trajectory:
        for point, label, color, offset in [(trajectory[0], "Start", colors[0], (-6, 12)),
                                             (trajectory[-1], "End", COLORS["ink"], (6, -18))]:
            target = (point[1] + 52.5, point[2] + 34)
            axes[1].scatter(*target, s=55, color=color, edgecolor="white", zorder=4)
            axes[1].annotate(f"{label} · {point[0]:.1f}s", target, xytext=offset,
                             textcoords="offset points", ha="right" if label == "Start" else "left",
                             fontsize=10, zorder=5, bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})
        # Recorded sample positions can briefly be outside the playing area.
        axes[1].set_xlim(min(-2.5, min(p[1] + 52.5 for p in trajectory) - 2),
                         max(107.5, max(p[1] + 52.5 for p in trajectory) + 2))
    axes[0].set_title(f"Full-match occupancy · {grid.shape[1]} × {grid.shape[0]} bins", loc="left")
    axes[1].set_title("First 120 seconds  ·  sampled path", loc="left")
    cax = fig.add_axes([.075, .205, .38, .023])
    colorbar = fig.colorbar(image, cax=cax, orientation="horizontal")
    colorbar.outline.set_visible(False)
    colorbar.set_label("Share of finite-coordinate samples per bin (%)", labelpad=7)
    fig.text(.55, .205, "Stadium frame  ·  nominal 105 × 68 m\nNo change of attacking direction between halves applied",
             fontsize=10, color=COLORS["muted"], va="top", linespacing=1.6)
    return save_figure(fig, "running-position",
                       "Source: Metrica Sports sample data. Anonymous home player selected by distance from recorded coordinates.\nSampled gaps >1s remain open. Occupancy describes recorded positions, not tactical role. Supplier detection flags are unavailable.\nOff-pitch coordinates enter boundary bins. No Gaussian smoothing; nearest-cell display, without interpolated intensity.",
                       scouting_root=scouting_root, output_dir=output_dir)
