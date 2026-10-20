"""StatsBomb event/360 metrics. Snapshot geometry is not continuous tracking."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np


def location(value):
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        if any(isinstance(v, bool) for v in value[:2]):
            return None
        x, y = float(value[0]), float(value[1])
        return (
            (min(120.0, max(0.0, x)), min(80.0, max(0.0, y)))
            if math.isfinite(x) and math.isfinite(y)
            else None
        )
    except (TypeError, ValueError):
        return None


def _box(point):
    return point is not None and point[0] >= 102 and 18 <= point[1] <= 62


def _progressive(start, end):
    if start is None or end is None:
        return False
    before, after = math.hypot(120 - start[0], 40 - start[1]), math.hypot(
        120 - end[0], 40 - end[1]
    )
    return before - after >= 10 and after <= 0.75 * before


def _totals(identity):
    return {
        **identity,
        **dict.fromkeys(
            (
                "events",
                "passes",
                "completed_passes",
                "progressive_passes",
                "carries",
                "progressive_carries",
                "box_entries",
                "pressures",
                "counterpressures",
                "recoveries",
                "shots",
                "goals",
                "xg",
            ),
            0,
        ),
        "action_grid": [[0] * 12 for _ in range(8)],
    }


def shot_geometry(point):
    dx, dy = 120 - point[0], point[1] - 40
    return math.hypot(dx, dy), math.atan2(abs(8 * dx), dx * dx + dy * dy - 16)


def analyze_event_match(
    events: list[dict], frames: list[dict], *, match_id: int
) -> dict:
    if not isinstance(events, list) or not isinstance(frames, list):
        raise ValueError("Events and 360 frames must be arrays")
    teams, players, network, shots, moves, spatial, formations = (
        {},
        {},
        {},
        [],
        [],
        [],
        [],
    )
    indexed = {
        frame["event_uuid"]: frame
        for frame in frames
        if isinstance(frame, dict) and "event_uuid" in frame
    }
    missing_locations = 0
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("Invalid event record")
        if event.get("period") == 5:
            continue
        team = event.get("team") or {}
        if "id" not in team:
            continue
        team_id = team["id"]
        if team_id not in teams:
            teams[team_id] = _totals({"team_id": team_id, "team": team.get("name")})
        team_row = teams[team_id]
        player = event.get("player") or {}
        if "id" in player and player["id"] not in players:
            players[player["id"]] = _totals(
                {
                    "player_id": player["id"],
                    "player": player.get("name"),
                    "team_id": team_id,
                    "position": (event.get("position") or {}).get("name"),
                }
            )
        player_row = players.get(player.get("id"))
        targets = [team_row] + ([player_row] if player_row else [])
        kind = (event.get("type") or {}).get("name")
        start = location(event.get("location"))
        for row in targets:
            row["events"] += 1
            if start is not None:
                row["action_grid"][min(7, int(start[1] / 10))][
                    min(11, int(start[0] / 10))
                ] += 1
        if start is None:
            missing_locations += 1
        if kind in {"Starting XI", "Tactical Shift"}:
            formations.append(
                {
                    "team_id": team_id,
                    "minute": event.get("minute", 0),
                    "formation": (event.get("tactics") or {}).get("formation"),
                }
            )
        if kind in {"Pass", "Carry"}:
            details = event.get(kind.lower()) or {}
            end = location(details.get("end_location"))
            completed = kind == "Carry" or not details.get("outcome")
            for row in targets:
                row["passes" if kind == "Pass" else "carries"] += 1
                if kind == "Pass" and completed:
                    row["completed_passes"] += 1
                if completed and _progressive(start, end):
                    row[
                        (
                            "progressive_passes"
                            if kind == "Pass"
                            else "progressive_carries"
                        )
                    ] += 1
                if completed and _box(end) and not _box(start):
                    row["box_entries"] += 1
            if start is not None and end is not None:
                moves.append(
                    {
                        "player_id": player.get("id"),
                        "team_id": team_id,
                        "kind": kind,
                        "start": start,
                        "end": end,
                        "completed": completed,
                    }
                )
            receiver = details.get("recipient") or {}
            if kind == "Pass" and completed and player.get("id") and receiver.get("id"):
                key = (team_id, player["id"], receiver["id"])
                edge = network.setdefault(
                    key,
                    {
                        "team_id": team_id,
                        "passer_id": player["id"],
                        "passer": player.get("name"),
                        "receiver_id": receiver["id"],
                        "receiver": receiver.get("name"),
                        "completed_passes": 0,
                    },
                )
                edge["completed_passes"] += 1
        if kind == "Pressure":
            for row in targets:
                row["pressures"] += 1
                row["counterpressures"] += bool(event.get("counterpress"))
        if kind == "Ball Recovery":
            for row in targets:
                row["recoveries"] += 1
        if kind == "Shot":
            details = event.get("shot") or {}
            xg = details.get("statsbomb_xg")
            xg = (
                float(xg)
                if isinstance(xg, (float, int))
                and not isinstance(xg, bool)
                and math.isfinite(xg)
                and 0 <= xg <= 1
                else None
            )
            goal = (details.get("outcome") or {}).get("name") == "Goal"
            for row in targets:
                row["shots"] += 1
                row["goals"] += goal
                row["xg"] += xg or 0.0
            if start is not None:
                distance, angle = shot_geometry(start)
                shots.append(
                    {
                        "match_id": match_id,
                        "event_id": event.get("id"),
                        "player_id": player.get("id"),
                        "team_id": team_id,
                        "start": start,
                        "distance_units": distance,
                        "angle_radians": angle,
                        "goal": int(goal),
                        "provider_xg": xg,
                        "header": int(
                            (details.get("body_part") or {}).get("name") == "Head"
                        ),
                        "penalty": (details.get("type") or {}).get("name") == "Penalty",
                        "set_piece": (details.get("type") or {}).get("name")
                        in {"Free Kick", "Corner"},
                        "under_pressure": bool(event.get("under_pressure")),
                        "period": event.get("period", 1),
                    }
                )
        frame = indexed.get(event.get("id"))
        if frame and start is not None:
            opponents = [
                location(p.get("location"))
                for p in frame.get("freeze_frame", [])
                if p.get("teammate") is False
            ]
            opponents = [point for point in opponents if point is not None]
            teammates = [
                location(p.get("location"))
                for p in frame.get("freeze_frame", [])
                if p.get("teammate") is True
            ]
            teammates = [point for point in teammates if point is not None]
            end = location((event.get("pass") or {}).get("end_location"))
            blockers = None
            if end is not None and math.dist(start, end) > 0:
                dx, dy = end[0] - start[0], end[1] - start[1]
                length_sq = dx * dx + dy * dy
                blockers = sum(
                    0
                    < ((p[0] - start[0]) * dx + (p[1] - start[1]) * dy) / length_sq
                    < 1
                    and abs(dx * (start[1] - p[1]) - (start[0] - p[0]) * dy)
                    / math.sqrt(length_sq)
                    <= 1.5
                    for p in opponents
                )
            spatial.append(
                {
                    "event_id": event.get("id"),
                    "team_id": team_id,
                    "visible_opponents": len(opponents),
                    "nearest_visible_opponent_units": min(
                        (math.dist(start, point) for point in opponents), default=None
                    ),
                    "visible_teammates": sum(
                        p.get("teammate") is True for p in frame.get("freeze_frame", [])
                    ),
                    "visible_teammate_width_units": (
                        max(p[1] for p in teammates) - min(p[1] for p in teammates)
                        if len(teammates) >= 2
                        else None
                    ),
                    "visible_teammate_depth_units": (
                        max(p[0] for p in teammates) - min(p[0] for p in teammates)
                        if len(teammates) >= 2
                        else None
                    ),
                    "visible_lane_blockers_snapshot": blockers,
                    "continuous_tracking": False,
                }
            )
    return {
        "match_id": match_id,
        "event_count": len(events),
        "events_without_locations": missing_locations,
        "teams": list(teams.values()),
        "players": list(players.values()),
        "shots": shots,
        "moves": moves,
        "passing_network": list(network.values()),
        "spatial_context": spatial,
        "formations": formations,
        "coordinate_system": "StatsBomb 120x80 units, event-team attacking orientation",
        "progressive_rule": "Successful action reduces goal distance by >=10 units and >=25%",
        "box_rule": "End in x>=102 and 18<=y<=62, start outside box",
        "360_note": "Visible-player snapshot at an event; no speed, running distance or full-pitch completeness implied.",
    }


def grid_index(point):
    return min(7, int(point[1] / 10)) * 12 + min(11, int(point[0] / 10))


def fit_expected_threat(
    move_counts: np.ndarray, attempts: np.ndarray, shots: np.ndarray, goals: np.ndarray
) -> dict:
    """Solve the minimal empirical goal probability, validating count conservation.

    A small value-iteration step is not an absolute-error certificate when moves
    almost always stay in the same cell. Solve the count system directly instead.
    States with no path to an observed goal have minimal nonnegative value zero;
    excluding them also removes singular closed non-scoring communicating classes.
    """
    counts = [np.asarray(a) for a in (move_counts, attempts, shots, goals)]
    if any(a.shape != shape or a.dtype.kind not in 'iuf' for a, shape in zip(counts, ((96, 96), (96,), (96,), (96,)))):
        raise ValueError('Invalid xT count shapes or types')
    move_counts, attempts, shots, goals = [a.astype(np.float64) for a in counts]
    if any(not np.isfinite(a).all() or np.any(a < 0) for a in (move_counts, attempts, shots, goals)):
        raise ValueError('Invalid xT nonnegative finite counts')
    with np.errstate(over='ignore'):
        total, completed = attempts + shots, move_counts.sum(axis=1)
    if not np.isfinite(total).all() or np.any(completed > attempts) or np.any(goals > shots):
        raise ValueError('Invalid xT count conservation')

    reachable = goals > 0
    while True:
        previous = reachable.copy()
        reachable |= np.any(move_counts[:, reachable] > 0, axis=1)
        if np.array_equal(previous, reachable):
            break
    values = np.zeros(96)
    amplification = 0.
    diagonal = (attempts - np.diag(move_counts)) + shots
    if reachable.any():
        system = -move_counts[np.ix_(reachable, reachable)]
        np.fill_diagonal(system, diagonal[reachable])
        # Form the diagonal without adding and subtracting a large self-loop.
        # For this transient M-matrix, A^-1 D is nonnegative. Its infinity norm
        # is max(A^-1 D 1), giving a residual-based error bound for the system.
        try:
            # Cancellation across near-closed cycles can round a bad solution's
            # residual to zero. Screen conditioning separately; this is a
            # conservative refusal rule, not a certified roundoff bound.
            condition = float(np.linalg.cond(system, p=np.inf))
            if not math.isfinite(condition) or condition * np.finfo(float).eps > 1e-9:
                raise ValueError('Unstable xT count-system conditioning')
            solved = np.linalg.solve(system, np.column_stack([goals[reachable], total[reachable]]))
        except np.linalg.LinAlgError as error:
            raise ValueError('Unstable xT numerical system') from error
        if (not np.isfinite(solved).all() or np.any(solved[:, 0] < -1e-12)
                or np.any(solved[:, 0] > 1 + 1e-12) or np.any(solved[:, 1] < 1 - 1e-12)):
            raise ValueError('Unstable xT numerical solution')
        values[reachable] = np.clip(solved[:, 0], 0, 1)
        amplification = float(np.max(solved[:, 1]))
    off_diagonal = move_counts.copy()
    np.fill_diagonal(off_diagonal, 0)
    residual = goals - (diagonal * values - off_diagonal @ values)
    residual_max = float(np.max(np.divide(np.abs(residual), total, out=np.zeros(96), where=total > 0)))
    error_bound = residual_max * amplification
    if not math.isfinite(error_bound) or error_bound > 1e-9:
        raise ValueError('Unstable xT residual bound')
    return {
        "grid": values.reshape(8, 12).tolist(),
        "iterations": 0,
        "converged": True,
        "solver": "direct_count_system_on_goal_reachable_cells",
        "diagnostics": {
            "observed_cells": int(np.count_nonzero(total)),
            "unobserved_cells": int(np.count_nonzero(total == 0)),
            "zero_shot_cells": int(np.count_nonzero(shots == 0)),
            "zero_goal_cells": int(np.count_nonzero(goals == 0)),
            "goal_reachable_cells": int(reachable.sum()),
            "bellman_residual_max": residual_max,
            "residual_amplification": amplification,
            "residual_error_bound": error_bound,
            "scope": "Float64 residual bound for the fitted system only; excludes roundoff, sampling and modelling uncertainty. Unobserved cells are not evidence of no threat.",
        },
        "method": "Empirical shot/move Markov model; failed moves terminate with zero goal value; direct minimum nonnegative solution",
        "interpretation": "Retrospective pooled event value, not calibrated possession-outcome probability.",
    }
