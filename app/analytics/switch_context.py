"""Visible opposition context and detected movement around cross-field passes."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import zlib

from .artifacts import archive_outputs, file_digest, cached_json
from .movement import _number, _boolean
from .tactical_relations import point
from .tracking import _seconds
from app.data_service.warehouse import Warehouse


def segment_distance(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    scale = dx * dx + dy * dy
    t = (
        max(0, min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / scale))
        if scale
        else 0
    )
    return math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy)


def local_visible(p, area, radius=5):
    if not isinstance(area, list) or len(area) < 6 or len(area) % 2:
        return False
    vertices = [point(area[i : i + 2]) for i in range(0, len(area), 2)]
    if any(v is None for v in vertices):
        return False
    inside = False
    edges = list(zip(vertices, vertices[1:] + vertices[:1]))
    for a, b in edges:
        if (a[1] > p[1]) != (b[1] > p[1]) and p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (
            b[1] - a[1]
        ) + a[0]:
            inside = not inside
    return inside and min(segment_distance(p, a, b) for a, b in edges) >= radius


def snapshot_context(row, frame):
    if frame is None:
        return {"snapshot_available": False}
    start, end = row["start"], row["end"]
    opponents, teammates = [], []
    for player in frame.get("freeze_frame", []):
        p = point(player.get("location"))
        if p is None or player.get("keeper") is True:
            continue
        if player.get("teammate") is False:
            opponents.append(p)
        elif player.get("teammate") is True and player.get("actor") is not True:
            teammates.append(p)
    ball_side = -1 if start[1] < 80 / 3 else 1 if start[1] >= 160 / 3 else None

    def side_count(players, side):
        return (
            sum((p[1] - 40) * side > 80 / 6 for p in players)
            if side is not None
            else None
        )

    def nearby(players, target):
        return sum(math.dist(p, target) <= 5 for p in players)

    return {
        "snapshot_available": True,
        "opponents_visible": len(opponents),
        "teammates_visible_excluding_passer": len(teammates),
        "nearest_visible_opponent_to_origin_units": min(
            (math.dist(p, start) for p in opponents), default=None
        ),
        "origin_opponents_within5_units": nearby(opponents, start),
        "target_opponents_within5_units": nearby(opponents, end),
        "target_teammates_within5_units": nearby(teammates, end),
        "origin_local_visible": local_visible(start, frame.get("visible_area")),
        "target_local_visible": local_visible(end, frame.get("visible_area")),
        "ball_side_teammates_visible": side_count(teammates, ball_side),
        "far_side_teammates_visible": side_count(
            teammates, -ball_side if ball_side else None
        ),
        "ball_side_opponents_visible": side_count(opponents, ball_side),
        "far_side_opponents_visible": side_count(
            opponents, -ball_side if ball_side else None
        ),
        "opponent_visible_width_units": (
            max(p[1] for p in opponents) - min(p[1] for p in opponents)
            if opponents
            else None
        ),
        "opponent_visible_depth_units": (
            max(p[0] for p in opponents) - min(p[0] for p in opponents)
            if opponents
            else None
        ),
        "continuous_runs_available": False,
    }


def continuous_motion(frames, player_id, sign):
    positions = []
    for f in frames:
        player = next(
            (
                p
                for p in f.get("player_data", [])
                if str(p.get("player_id")) == str(player_id)
            ),
            None,
        )
        if player is None or player.get("is_detected") is not True:
            return None
        x, y = _number(player.get("x")), _number(player.get("y"))
        if x is None or y is None:
            return None
        positions.append((_seconds(f["timestamp"]), sign * x, sign * y))
    if len(positions) < 2:
        return None
    for a, b in zip(positions, positions[1:]):
        dt = b[0] - a[0]
        if not 0 < dt <= 0.15 or math.hypot(b[1] - a[1], b[2] - a[2]) / dt > 12:
            return None
    return {
        "start": positions[0][1:],
        "end": positions[-1][1:],
        "dx": positions[-1][1] - positions[0][1],
        "dy": positions[-1][2] - positions[0][2],
    }


def tracked_context(case, snapshots, metadata):
    f = case["pass_frame"]
    before = [snapshots.get((case["period"], i)) for i in range(f - 10, f + 1)]
    window = [snapshots.get((case["period"], i)) for i in range(f - 10, f + 11)]
    at = snapshots.get((case["period"], f))
    if at is None:
        return {"pass_frame_available": False, "movement_coverage": "missing"}
    sign = -1 if case["attacking_side"] == "right_to_left" else 1
    ball = at.get("ball_data") or {}
    bx, by = _number(ball.get("x")), _number(ball.get("y"))
    distance = (
        math.hypot(sign * bx - case["x"], sign * by - case["y"])
        if bx is not None and by is not None
        else None
    )
    group = (
        "home team"
        if str(metadata["home_team"]["id"]) == case["team_id"]
        else "away team"
    )
    aligned = (
        ball.get("is_detected") is True
        and (at.get("possession") or {}).get("group") == group
        and distance is not None
        and distance <= 5
    )
    lookup = {str(p["id"]): p for p in metadata["players"]}
    teams = defaultdict(list)
    for player in at.get("player_data", []):
        identity = lookup.get(str(player.get("player_id")))
        if (
            identity is None
            or (identity.get("player_role") or {}).get("acronym") == "GK"
            or player.get("is_detected") is not True
        ):
            continue
        x, y = _number(player.get("x")), _number(player.get("y"))
        if x is not None and y is not None:
            teams[str(identity["team_id"])].append(
                (str(player["player_id"]), sign * x, sign * y)
            )
    own = teams[case["team_id"]]
    opposition = [
        p for team, players in teams.items() if team != case["team_id"] for p in players
    ]
    side = 1 if case["y"] > 0 else -1
    possession_groups = {
        (w.get("possession") or {}).get("group") for w in window if w is not None
    }
    other_group = "away team" if group == "home team" else "home team"
    movement_context = (
        "possession_changed"
        if other_group in possession_groups
        else "own_or_unassigned" if group in possession_groups else "unknown"
    )
    observed = Counter()
    run_examples = []
    closing = []
    if aligned and all(w is not None for w in window):
        for pid, x, y in own:
            if pid == case["player_id"]:
                continue
            movement = continuous_motion(window, pid, sign)
            if movement is None:
                continue
            zone = (
                "ball_side"
                if movement["start"][1] * side > metadata["pitch_width"] / 6
                else (
                    "far_side"
                    if movement["start"][1] * side < -metadata["pitch_width"] / 6
                    else "central"
                )
            )
            observed[zone + "_players_continuously_observed"] += 1
            if movement["dx"] >= 2 or abs(movement["dy"]) >= 2:
                observed[zone + "_moving_players"] += 1
                pre = continuous_motion(before, pid, sign)
                run_examples.append(
                    {
                        "player_id": pid,
                        "role_in_pass": (
                            "identified_target"
                            if pid == case.get("target_player_id")
                            else "other_teammate"
                        ),
                        "side_at_start": zone,
                        "pre_release_dx": pre["dx"],
                        "pre_release_dy": pre["dy"],
                        "post_release_dx": movement["dx"] - pre["dx"],
                        "post_release_dy": movement["dy"] - pre["dy"],
                        **movement,
                    }
                )
    if aligned and all(w is not None for w in before):
        b0 = before[0].get("ball_data") or {}
        x0, y0 = _number(b0.get("x")), _number(b0.get("y"))
        if b0.get("is_detected") is True and x0 is not None and y0 is not None:
            for pid, x, y in opposition:
                movement = continuous_motion(before, pid, sign)
                if movement is None:
                    continue
                d0 = math.dist(movement["start"], (sign * x0, sign * y0))
                d1 = math.dist(movement["end"], (sign * bx, sign * by))
                speed = (d0 - d1) / (
                    _seconds(before[-1]["timestamp"]) - _seconds(before[0]["timestamp"])
                )
                closing.append(
                    {
                        "player_id": pid,
                        "distance_m": d1,
                        "closing_speed_mps_proxy": speed,
                        "constant_closing_time_seconds_proxy": (
                            d1 / speed if speed > 0.5 else None
                        ),
                        "start": movement["start"],
                        "end": movement["end"],
                    }
                )
    return {
        "pass_frame_available": True,
        "ball_anchor_distance_m": distance,
        "usable_ball_anchor": aligned,
        "tracked_ball_at_release": (
            (sign * bx, sign * by) if bx is not None and by is not None else None
        ),
        "detected_positions_at_release": [
            {"player_id": p[0], "x": p[1], "y": p[2], "teammate": True} for p in own
        ]
        + [
            {"player_id": p[0], "x": p[1], "y": p[2], "teammate": False}
            for p in opposition
        ],
        "teammates_detected_at_pass": len(own),
        "opponents_detected_at_pass": len(opposition),
        "opponent_detected_width_m": (
            max(p[2] for p in opposition) - min(p[2] for p in opposition)
            if opposition
            else None
        ),
        "opponent_detected_depth_m": (
            max(p[1] for p in opposition) - min(p[1] for p in opposition)
            if opposition
            else None
        ),
        "ball_side_teammates_detected": sum(
            p[2] * side > metadata["pitch_width"] / 6 for p in own
        ),
        "far_side_teammates_detected": sum(
            p[2] * side < -metadata["pitch_width"] / 6 for p in own
        ),
        "ball_side_opponents_detected": sum(
            p[2] * side > metadata["pitch_width"] / 6 for p in opposition
        ),
        "far_side_opponents_detected": sum(
            p[2] * side < -metadata["pitch_width"] / 6 for p in opposition
        ),
        "continuous_movement_counts": dict(observed),
        "movement_window_team_context": movement_context,
        "observed_movements": run_examples,
        "opponent_closing": closing,
        "movement_coverage": (
            "some_teammates" if observed else "no_complete_teammate_window"
        ),
    }


def selected_frames(connection, asset_id, needed):
    try:
        import orjson

        loads = orjson.loads
    except ImportError:
        loads = json.loads
    result = {}
    buffer = b""
    for (content,) in connection.execute(
        "SELECT content FROM raw_chunks WHERE asset_id=? ORDER BY ordinal", (asset_id,)
    ):
        buffer += zlib.decompress(content)
        lines = buffer.split(b"\n")
        buffer = lines.pop()
        for line in lines:
            if not line.strip():
                continue
            item = loads(line)
            key = (item.get("period"), item.get("frame"))
            if key in needed and item.get("timestamp") is not None:
                result[key] = item
    if buffer.strip():
        item = loads(buffer)
        key = (item.get("period"), item.get("frame"))
        if key in needed and item.get("timestamp") is not None:
            result[key] = item
    return result


def draw_example(cases, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D

    eligible = [
        r
        for r in cases
        if r.get("usable_ball_anchor")
        and r.get("target_x_at_pass") is not None
        and any(
            m["side_at_start"] == "far_side" for m in r.get("observed_movements", [])
        )
    ]
    if not eligible:
        return None
    case = max(
        eligible,
        key=lambda r: r["teammates_detected_at_pass"] + r["opponents_detected_at_pass"],
    )
    length, width = case["pitch_length"], case["pitch_width"]
    from .visualization.pitch import draw_pitch, direction_arrow
    from .visualization.style import COLORS, apply_style, decorate, save

    apply_style()
    fig = plt.figure(figsize=(13, 8.3))
    ax = fig.add_axes([.055, .265, .685, .51])
    info = fig.add_axes([.775, .26, .20, .51])
    info.axis("off")
    decorate(fig, "A cross-field pass and the movement we can observe",
             f"{case['team']} vs {case['opponent']}  ·  {case['date'][:10]}")
    fig.text(.055, .84,
             f"Match {case['match_id']} · event {case['event_id']} · provider pressure: {case['vendor_pressure'].replace('_', ' ')} · outcome: {case['pass_outcome']}",
             fontsize=10, color=COLORS["muted"])
    draw_pitch(ax, length, width, centered=True, invert_y=False)
    ball_side = 1 if case["y"] > 0 else -1
    for side in (1, -1):
        start_y = width / 6 if side > 0 else -width / 2
        ax.add_patch(Rectangle((-length / 2, start_y), length, width / 3,
                               color="#EAF1F7" if side == -ball_side else "#F1F4F7", zorder=.1))
        ax.text(-length / 2 + 2.5, side * width / 3,
                "Ball side" if side == ball_side else "Far side",
                color=COLORS["muted"], fontsize=11, va="center")
    for own, color, marker in [(True, COLORS["blue"], "o"), (False, COLORS["orange"], "s")]:
        points = [p for p in case["detected_positions_at_release"] if p["teammate"] == own]
        ax.scatter([p["x"] for p in points], [p["y"] for p in points],
                   color=color, s=70, marker=marker, zorder=3, edgecolor="white", linewidth=.8)
    for motion in case["observed_movements"]:
        ax.annotate("", xy=motion["end"], xytext=motion["start"],
                    arrowprops={"arrowstyle": "->", "lw": 2, "color": COLORS["blue"]}, zorder=4)
    for motion in case["opponent_closing"]:
        if motion["distance_m"] <= 10:
            ax.annotate("", xy=motion["end"], xytext=motion["start"],
                        arrowprops={"arrowstyle": "->", "lw": 1.8, "color": COLORS["orange"]}, zorder=4)
    ball = case["tracked_ball_at_release"]
    target = (case["target_x_at_pass"], case["target_y_at_pass"])
    ax.scatter(*ball, color=COLORS["ink"], s=65, zorder=5, edgecolor="white", lw=.8)
    ax.annotate("", xy=target, xytext=ball,
                arrowprops={"arrowstyle": "->", "linestyle": "--", "lw": 1.5, "color": COLORS["ink"]}, zorder=2)
    ax.scatter(*target, s=170, facecolors="none", edgecolors=COLORS["ink"], linewidths=1.4, zorder=4)
    ax.annotate("Release", ball, xytext=(-30, 18), textcoords="offset points", fontsize=10,
                color=COLORS["ink"], bbox={"facecolor": "white", "edgecolor": "none", "pad": 2}, zorder=6)
    ax.annotate("Identified target", target, xytext=(-10, -26), textcoords="offset points", fontsize=10, ha="right",
                color=COLORS["ink"], bbox={"facecolor": "white", "edgecolor": "none", "pad": 2},
                arrowprops={"arrowstyle": "-", "color": COLORS["muted"], "lw": .7}, zorder=6)
    if case.get("attacking_side") in {"left_to_right", "right_to_left"}:
        direction_arrow(ax, "Attack · provider direction normalized")
    else:
        ax.text(.5, -.09, "Attacking direction unavailable", transform=ax.transAxes,
                ha="center", color=COLORS["muted"], fontsize=10)
    info.text(0, 1, "Detected at release", fontsize=12, weight="semibold", va="top")
    info.text(0, .87, f"{case['teammates_detected_at_pass']} teammates\n{case['opponents_detected_at_pass']} opponents",
              fontsize=12, va="top", linespacing=1.8)
    info.text(0, .60, "Tracking alignment", fontsize=11, weight="semibold", va="top")
    info.text(0, .51, f"Ball anchor error: {case['ball_anchor_distance_m']:.2f} m\nDetected players only",
              fontsize=10, color=COLORS["muted"], va="top", linespacing=1.7)
    counts = case.get("continuous_movement_counts", {})
    observed = sum(counts.get(f"{side}_players_continuously_observed", 0) for side in ["central", "far_side", "ball_side"])
    info.text(0, .30, "Movement availability", fontsize=11, weight="semibold", va="top")
    context = case.get("movement_window_team_context", "unknown").replace("_", " ")
    info.text(0, .21, f"{observed} complete teammate windows\nPossession: {context}", fontsize=10,
              color=COLORS["muted"], va="top", linespacing=1.7, wrap=True)
    handles = [Line2D([], [], marker="o", linestyle="none", color=COLORS["blue"], markersize=7, label="Teammate at release"),
               Line2D([], [], marker="s", linestyle="none", color=COLORS["orange"], markersize=7, label="Opponent at release"),
               Line2D([], [], color=COLORS["ink"], linestyle="--", label="Release to identified target"),
               Line2D([], [], color=COLORS["blue"], lw=2, label="Teammate motion: −1 to +1s"),
               Line2D([], [], color=COLORS["orange"], lw=2, label="Nearby opponent: −1s to release")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.05, .118), ncol=3,
               frameon=False, fontsize=10, columnspacing=2.0, handlelength=2.5, labelspacing=1.0)
    fig.text(.055, .035,
             "Source: SkillCorner open data. Arrows require continuous detections; unavailable or extrapolated positions are not filled.\nDashed link is not a measured ball trajectory. Ball/far side does not establish an overload. Illustrative selection by detection coverage.",
             fontsize=10, color=COLORS["muted"], va="bottom", linespacing=1.5)
    save(fig, output, "switch-context", formats=("png", "pdf", "svg"))
    return {
        "match_id": case["match_id"],
        "event_id": case["event_id"],
        "selection": "most detected outfield players among usable cases with observed far-side movement; illustrative, not an outcome-based sample",
    }


def run(database="data/football.sqlite3", output=".tmp/switch-context"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    receipts = []
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as c:
        c.row_factory = sqlite3.Row
        for document in c.execute(
            "SELECT d.asset_id,a.dataset_id,a.path,a.sha256,a.bytes,d.payload FROM documents d JOIN assets a ON a.id=d.asset_id WHERE d.kind='skillcorner_document' AND a.path LIKE 'data/matches/%_match.json'"
        ).fetchall():
            m = json.loads(document["payload"])
            if (
                isinstance(m.get("id"), bool)
                or not isinstance(m.get("id"), int)
                or m["id"] <= 0
            ):
                raise ValueError(
                    "SkillCorner match identity must be a positive integer"
                )
            mid = str(m["id"])
            if document["path"] != f"data/matches/{mid}/{mid}_match.json":
                raise ValueError(
                    "SkillCorner metadata identity differs from its source path"
                )
            width = m["pitch_width"]
            assets = c.execute(
                "SELECT id,sha256,bytes FROM assets WHERE dataset_id=? AND path IN (?,?)",
                (
                    document["dataset_id"],
                    f"data/matches/{mid}/{mid}_dynamic_events.csv",
                    f"data/matches/{mid}/{mid}_tracking_extrapolated.jsonl",
                ),
            ).fetchall()
            track = c.execute(
                "SELECT id FROM assets WHERE dataset_id=? AND path=?",
                (
                    document["dataset_id"],
                    f"data/matches/{mid}/{mid}_tracking_extrapolated.jsonl",
                ),
            ).fetchone()
            if track is None:
                continue
            dynamic = c.execute(
                "SELECT id FROM assets WHERE dataset_id=? AND path=?",
                (
                    document["dataset_id"],
                    f"data/matches/{mid}/{mid}_dynamic_events.csv",
                ),
            ).fetchone()
            if dynamic is None:
                continue
            records = [
                dict(r)
                for r in c.execute(
                    """SELECT asset_id,event_id,event_type,event_subtype,frame_start,frame_end,
                    period,attacking_side,team_id,team_shortname,player_id,player_position,
                    x_end,y_end,player_targeted_x_pass,player_targeted_y_pass,
                    player_targeted_id,player_targeted_position,
                    associated_player_possession_event_id,end_type,pass_outcome,
                    overall_pressure_start,overall_pressure_end,reception_difficulty_start,
                    space_constraint_start,time_to_impact_end,passing_option_ease_end,
                    defensive_structure,team_out_of_possession_phase_type,fully_extrapolated
                    FROM skillcorner_dynamic_events WHERE asset_id=? AND match_id=?
                    AND event_type IN ('player_possession','off_ball_run') ORDER BY row_number""",
                    (dynamic["id"], mid),
                )
            ]
            by_possession = defaultdict(list)
            for r in records:
                if r["event_type"] == "off_ball_run":
                    by_possession[r["associated_player_possession_event_id"]].append(r)
            selected = []
            for r in records:
                y, target, x = (
                    _number(r["y_end"]),
                    _number(r["player_targeted_y_pass"]),
                    _number(r["x_end"]),
                )
                if (
                    r["event_type"] != "player_possession"
                    or r["end_type"] != "pass"
                    or y is None
                    or target is None
                    or x is None
                    or abs(y) < width / 6
                    or abs(target) < width / 6
                    or y * target >= 0
                    or r["attacking_side"] not in {"left_to_right", "right_to_left"}
                ):
                    continue
                coach_side = (
                    "home" if r["team_id"] == str(m["home_team"]["id"]) else "away"
                )
                coach = m.get(coach_side + "_team_coach") or {}
                case = {
                    "match_id": mid,
                    "event_id": r["event_id"],
                    "asset_id": r["asset_id"],
                    "date": m["date_time"],
                    "team_id": r["team_id"],
                    "player_id": r["player_id"],
                    "period": int(r["period"]),
                    "pass_frame": int(r["frame_end"]),
                    "attacking_side": r["attacking_side"],
                    "x": x,
                    "y": y,
                    "target_y_at_pass": target,
                    "target_player_id": r["player_targeted_id"],
                    "passer_position": r["player_position"],
                    "target_position": r["player_targeted_position"],
                    "target_x_at_pass": _number(r["player_targeted_x_pass"]),
                    "opponent": m[
                        ("away" if coach_side == "home" else "home") + "_team"
                    ]["name"],
                    "pitch_length": m["pitch_length"],
                    "pitch_width": width,
                    "team": r["team_shortname"],
                    "coach": coach,
                    "pass_outcome": r["pass_outcome"],
                    "vendor_pressure": r["overall_pressure_end"],
                    "vendor_pressure_at_possession_start": r["overall_pressure_start"],
                    "vendor_reception_difficulty_at_start": r[
                        "reception_difficulty_start"
                    ],
                    "vendor_space_constraint_at_start": r["space_constraint_start"],
                    "vendor_time_to_impact": r["time_to_impact_end"],
                    "vendor_passing_option_ease": r["passing_option_ease_end"],
                    "vendor_defensive_structure": r["defensive_structure"],
                    "vendor_defensive_phase": r["team_out_of_possession_phase_type"],
                    "fully_extrapolated": _boolean(r["fully_extrapolated"]),
                }
                active = [
                    b
                    for b in by_possession[r["event_id"]]
                    if b["team_id"] == r["team_id"]
                    and int(b["frame_start"])
                    <= case["pass_frame"]
                    <= int(b["frame_end"])
                ]
                case["provider_active_run_ids"] = [b["event_id"] for b in active]
                case["provider_active_run_subtypes"] = dict(
                    Counter(b["event_subtype"] or "unknown" for b in active)
                )
                selected.append(case)
            needed = {
                (r["period"], f)
                for r in selected
                for f in range(r["pass_frame"] - 10, r["pass_frame"] + 11)
            }
            source = c.execute(
                "SELECT sha256,bytes FROM assets WHERE id=?", (track["id"],)
            ).fetchone()
            cache = output / "frames"
            cache.mkdir(exist_ok=True)
            signature = {
                "tracking_sha256": source["sha256"],
                "bytes": source["bytes"],
                "frame_selection_sha256": hashlib.sha256(
                    json.dumps(sorted(needed)).encode()
                ).hexdigest(),
                "reader_version": 1,
            }
            frame_list = cached_json(
                cache / f"{mid}.json",
                signature,
                lambda: list(selected_frames(c, track["id"], needed).values()),
            )
            snapshots = {(r["period"], r["frame"]): r for r in frame_list}
            for case in selected:
                case.update(tracked_context(case, snapshots, m))
            cases.extend(selected)
            receipts.extend([list(r) for r in assets])
            receipts.append(
                [document["asset_id"], document["sha256"], document["bytes"]]
            )
            print("Context", mid, len(selected), flush=True)
    groups = []
    for pressure in sorted({r["vendor_pressure"] or "unknown" for r in cases}):
        rows = [r for r in cases if (r["vendor_pressure"] or "unknown") == pressure]
        known = [r for r in rows if r["pass_outcome"] in {"successful", "unsuccessful"}]
        groups.append(
            {
                "vendor_pressure": pressure,
                "attempts": len(rows),
                "known_outcomes": len(known),
                "successful": sum(r["pass_outcome"] == "successful" for r in known),
                "observed_movement_cases": sum(
                    bool(r.get("observed_movements")) for r in rows
                ),
            }
        )
    contexts = defaultdict(list)
    for case in cases:
        contexts[
            (
                case["vendor_defensive_phase"] or "unknown",
                case["vendor_defensive_structure"] or "unknown",
                case["vendor_pressure"] or "unknown",
            )
        ].append(case)
    report = {
        "matches": len({r["match_id"] for r in cases}),
        "cross_field_attempts": len(cases),
        "pressure_groups": groups,
        "defensive_context_groups": [
            {
                "opponent_phase": key[0],
                "vendor_defensive_structure": key[1],
                "vendor_pressure": key[2],
                "attempts": len(rows),
                "successful": sum(r["pass_outcome"] == "successful" for r in rows),
                "far_side_motion_observed": sum(
                    any(
                        m["side_at_start"] == "far_side"
                        for m in r.get("observed_movements", [])
                    )
                    for r in rows
                ),
            }
            for key, rows in sorted(contexts.items())
        ],
        "coach_metadata_cases": sum(bool(r["coach"]) for r in cases),
        "tracked_anchor_usable": sum(r.get("usable_ball_anchor", False) for r in cases),
        "cases_with_continuous_teammate_motion": sum(
            bool(r.get("continuous_movement_counts")) for r in cases
        ),
        "cases_with_observed_far_side_movement": sum(
            any(
                x["side_at_start"] == "far_side"
                for x in r.get("observed_movements", [])
            )
            for r in cases
        ),
        "cases_with_opponent_closing": sum(
            bool(r.get("opponent_closing")) for r in cases
        ),
        "movement_window_contexts": dict(
            Counter(r.get("movement_window_team_context", "unknown") for r in cases)
        ),
        "far_side_movement_without_observed_team_change": sum(
            r.get("movement_window_team_context") == "own_or_unassigned"
            and any(
                x["side_at_start"] == "far_side"
                for x in r.get("observed_movements", [])
            )
            for r in cases
        ),
        "provider_active_run_cases": sum(
            bool(r["provider_active_run_ids"]) for r in cases
        ),
        "source_fingerprint": hashlib.sha256(json.dumps(receipts).encode()).hexdigest(),
        "extractor_sha256": file_digest(__file__),
        "references": [
            "https://github.com/SkillCorner/opendata",
            "https://skillcorner.com/us/articles/introducing-skillcorners-new-pressure-metrics",
        ],
        "method": "Attempted passes ending a player possession, ball holder and targeted player on opposite outer pitch thirds at release. Dynamic coordinates are attacker-normalized; raw tracking is rotated by attacking_side. Movement requires21 consecutive detected frames from1s before to1s after release, gap<=0.15s, speed<=12m/s. Forward/lateral displacement>=2m is a movement candidate. Opponent closing uses11 detected pre-release frames and relative distance to the detected ball.",
        "limits": [
            "Ball side and far side are spatial proxies; observed counts do not prove a tactical overload or coach intent.",
            "Broadcast visibility can omit far-side players; absent continuous motion means unavailable observation, not no run.",
            "Vendor pressure is a multi-component label, not interchangeable with StatsBomb binary under_pressure or this project's constant-closing-speed proxy.",
            "Shape spans include only detected outfield players; provider defensive structure may use extrapolated coordinates.",
            "Dynamic event IDs are unique only within a match. SkillCorner and Euro2024 cohorts are separate; no fixture/player cross-provider merge is attempted.",
            "Post-release motion is descriptive evidence and cannot be used as a pre-release prediction feature.",
            "Movement before/after release is stored separately. A possession change can turn support movement into defensive recovery; observed motion is not automatically an attacking run.",
            "Identified targets condition inclusion; these attempts are not all attempted switches, so success rates can have selection bias.",
        ],
    }
    if not cases:
        raise ValueError("No SkillCorner cross-field attempts")
    report["example_case"] = draw_example(cases, output)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    (output / "cases.json").write_text(json.dumps(cases, indent=2, allow_nan=False))
    run_id = file_digest(output / "report.json")[:16]
    with Warehouse(database) as db:
        db.connection.executescript(
            "CREATE TABLE IF NOT EXISTS tactical_switch_context_runs(run_id TEXT PRIMARY KEY,payload TEXT);CREATE TABLE IF NOT EXISTS tactical_switch_context_cases(run_id TEXT REFERENCES tactical_switch_context_runs(run_id),provider TEXT,match_id TEXT,event_id TEXT,payload TEXT,PRIMARY KEY(run_id,provider,match_id,event_id));"
        )
        db.connection.execute(
            "INSERT OR REPLACE INTO tactical_switch_context_runs VALUES(?,?)",
            (run_id, json.dumps(report)),
        )
        db.connection.executemany(
            "INSERT OR REPLACE INTO tactical_switch_context_cases VALUES(?,?,?,?,?)",
            [
                (run_id, "skillcorner", r["match_id"], r["event_id"], json.dumps(r))
                for r in cases
            ],
        )
        archive_outputs(
            db, "switch_context_artifacts", output, "switch_context", run_id
        )
        if report["example_case"]:
            ds = db.dataset(
                "switch_context_artifacts", "derived", "soccer_analytics", run_id
            )
            db.asset(ds, output / "switch-context.pdf", "switch-context.pdf")
        db.connection.commit()
        db.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    print(json.dumps({"run_id": run_id, **report}, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--database", default="data/football.sqlite3")
    p.add_argument("--output", default=".tmp/switch-context")
    a = p.parse_args()
    run(a.database, a.output)
