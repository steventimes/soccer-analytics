"""Formation, possession, direction and opponent analysis with causal-time prefixes."""

from __future__ import annotations
from collections import defaultdict, Counter, deque
import json
import math
from pathlib import Path
import sqlite3
from .tactics import _progressive, _box


def clock(event):
    value = event.get("timestamp")
    if not isinstance(value, str):
        return None
    try:
        parts = value.split(":")
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    except (ValueError, IndexError):
        return None


def point(value):
    if not isinstance(value, list) or len(value) < 2:
        return None
    try:
        x, y = float(value[0]), float(value[1])
    except (ValueError, TypeError):
        return None
    return (
        (x, y)
        if math.isfinite(x) and math.isfinite(y) and 0 <= x <= 120 and 0 <= y <= 80
        else None
    )


def channel(p):
    return "low_y" if p[1] < 80 / 3 else "high_y" if p[1] >= 160 / 3 else "central"


def analyze_match(events, meta, context=None):
    home = meta["home_team"]["home_team_id"]
    away = meta["away_team"]["away_team_id"]
    ids = [home, away]
    names = {
        home: meta["home_team"]["home_team_name"],
        away: meta["away_team"]["away_team_name"],
    }
    context = context or {}
    forms = {tid: None for tid in ids}
    scores = {tid: 0 for tid in ids}
    teams = {}
    possessions = {}
    prefixes = []
    changes = []
    positions = defaultdict(list)
    audit = Counter()
    for tid in ids:
        teams[tid] = {
            "team_id": tid,
            "team": names[tid],
            "opponent_id": away if tid == home else home,
            "opponent": names[away if tid == home else home],
            "home": tid == home,
            "starting_formation": None,
            "formation_changes": 0,
            "passes": 0,
            "carries": 0,
            "completed_moves": 0,
            "forward_moves": 0,
            "low_y_moves": 0,
            "central_moves": 0,
            "high_y_moves": 0,
            "box_entries": 0,
            "pressures": 0,
            "shots": 0,
            "np_shots": 0,
            "xg_known_shots": 0,
            "npxg": 0.0,
            "possession_clock_seconds": 0.0,
            "on_ball_actions": 0,
            "possessions": 0,
            "shot_possessions": 0,
            "action_locations": 0,
            "origin_x_sum": 0.0,
        }
    valid = [e for e in events if e.get("period") in {1, 2, 3, 4}]
    period_extents = {}
    for i, event in enumerate(valid):
        tid = (event.get("team") or {}).get("id")
        owner = (event.get("possession_team") or {}).get("id")
        kind = (event.get("type") or {}).get("name")
        period = event["period"]
        t = clock(event)
        if t is not None:
            low, high = period_extents.get(period, (t, t))
            period_extents[period] = (min(low, t), max(high, t))
        if tid not in teams:
            continue
        row = teams[tid]
        if kind in {"Starting XI", "Tactical Shift"}:
            form = (event.get("tactics") or {}).get("formation")
            if form:
                forms[tid] = str(form)
                if kind == "Starting XI":
                    row["starting_formation"] = str(form)
                else:
                    row["formation_changes"] += 1
                changes.append(
                    {
                        "team_id": tid,
                        "period": period,
                        "clock": t,
                        "event_index": event.get("index", i),
                        "formation": str(form),
                        "type": kind,
                    }
                )
        # An inferred clock share is not an official possession percentage.
        if (
            i + 1 < len(valid)
            and owner in teams
            and t is not None
            and kind
            not in {
                "Half End",
                "Half Start",
                "Foul Committed",
                "Injury Stoppage",
                "Substitution",
                "Ball Out",
            }
        ):
            nxt = valid[i + 1]
            nt = clock(nxt)
            if nt is not None and nxt["period"] == period:
                delta = nt - t
                if 0 <= delta <= 30:
                    teams[owner]["possession_clock_seconds"] += delta
                elif delta > 30:
                    audit["unassigned_long_clock_gap"] += 1
        start = point(event.get("location"))
        if kind in {"Pass", "Carry"}:
            row["on_ball_actions"] += 1
            row["passes" if kind == "Pass" else "carries"] += 1
            details = event.get(kind.lower()) or {}
            end = point(details.get("end_location"))
            completed = kind == "Carry" or not details.get("outcome")
            if start:
                row["action_locations"] += 1
                row["origin_x_sum"] += start[0]
                position = (event.get("position") or {}).get("name")
                if position:
                    positions[(tid, position, forms[tid])].append(start)
            if start and end and completed:
                row["completed_moves"] += 1
                row[channel(end) + "_moves"] += 1
                row["forward_moves"] += end[0] - start[0] >= 5
                row["box_entries"] += _box(end) and not _box(start)
        if kind == "Pressure":
            row["pressures"] += 1
        if owner in teams and event.get("possession") is not None:
            key = (period, event["possession"], owner)
            if key not in possessions:
                opponent = away if owner == home else home
                possessions[key] = {
                    "owner": owner,
                    "period": period,
                    "possession_id": event["possession"],
                    "formation": forms[owner],
                    "opponent_formation": forms[opponent],
                    "score_state": scores[owner] - scores[opponent],
                    "actions": [],
                    "shots": [],
                }
            p = possessions[key]
            if tid == owner and kind in {"Pass", "Carry"} and start:
                d = event.get(kind.lower()) or {}
                end = point(d.get("end_location"))
                duration = event.get("duration")
                p["actions"].append(
                    {
                        "index": event.get("index", i),
                        "time": t,
                        "duration": duration,
                        "start": start,
                        "end": end,
                        "kind": kind,
                        "complete": kind == "Carry" or not d.get("outcome"),
                        "formation": forms[owner],
                        "opponent_formation": forms[away if owner == home else home],
                    }
                )
            if tid == owner and kind == "Shot":
                shot = event.get("shot") or {}
                xg = shot.get("statsbomb_xg")
                penalty = (shot.get("type") or {}).get("name") == "Penalty"
                p["shots"].append(
                    {
                        "index": event.get("index", i),
                        "time": t,
                        "xg": xg,
                        "penalty": penalty,
                    }
                )
        if kind == "Shot":
            d = event.get("shot") or {}
            row["shots"] += 1
            if (d.get("type") or {}).get("name") != "Penalty":
                row["np_shots"] += 1
                xg = d.get("statsbomb_xg")
                if (
                    isinstance(xg, (int, float))
                    and not isinstance(xg, bool)
                    and math.isfinite(xg)
                    and 0 <= xg <= 1
                ):
                    row["xg_known_shots"] += 1
                    row["npxg"] += xg
            if (d.get("outcome") or {}).get("name") == "Goal":
                scores[tid] += 1
        if kind == "Own Goal Against":
            scores[away if tid == home else home] += 1
    for key, p in possessions.items():
        owner = p["owner"]
        row = teams[owner]
        row["possessions"] += 1
        row["shot_possessions"] += any(not s["penalty"] for s in p["shots"])
        if len(p["actions"]) < 3:
            continue
        actions = p["actions"][:3]
        last = actions[-1]
        if any(
            a["time"] is None
            or not isinstance(a["duration"], (int, float))
            or not math.isfinite(a["duration"])
            or a["duration"] < 0
            for a in actions
        ):
            audit["prefix_unknown_action_end_time"] += 1
            continue
        decision = last["time"] + last["duration"]
        if any(
            s["index"] <= last["index"]
            or (s["time"] is not None and s["time"] <= decision)
            for s in p["shots"]
        ):
            audit["shot_already_observed_at_prefix_end"] += 1
            continue
        later = [
            s
            for s in p["shots"]
            if not s["penalty"]
            and s["time"] is not None
            and s["time"] > decision
            and s["index"] > last["index"]
        ]
        valid_ends = [a["end"] for a in actions if a["end"]]
        if len(valid_ends) < 3:
            audit["incomplete_prefix_geometry"] += 1
            continue
        geometry = [a["start"] for a in actions] + valid_ends
        opponent = away if owner == home else home
        base = {
            "start_x": actions[0]["start"][0],
            "start_y": actions[0]["start"][1],
            "home": float(owner == home),
            "score_state": p["score_state"],
            "period": p["period"],
            "clock": decision,
            "competition": str(meta["competition"]["competition_id"]),
            "gender": meta.get("home_team", {}).get("home_team_gender", "unknown"),
            **context.get(owner, {}),
        }
        full = {
            **base,
            "formation": last["formation"] or "unknown",
            "opponent_formation": last["opponent_formation"] or "unknown",
            "prefix_forward_fraction": sum(
                a["end"][0] - a["start"][0] >= 5 for a in actions
            )
            / 3,
            "prefix_completion_fraction": sum(a["complete"] for a in actions) / 3,
            "prefix_carry_fraction": sum(a["kind"] == "Carry" for a in actions) / 3,
            "prefix_max_x": max(g[0] for g in geometry),
            "prefix_width_units": max(g[1] for g in geometry)
            - min(g[1] for g in geometry),
            "prefix_low_y_fraction": sum(channel(g) == "low_y" for g in valid_ends) / 3,
            "prefix_central_fraction": sum(channel(g) == "central" for g in valid_ends)
            / 3,
            "prefix_elapsed_seconds": decision - actions[0]["time"],
        }
        prefixes.append(
            {
                "match_id": meta["match_id"],
                "date": meta["match_date"],
                "team_id": owner,
                "opponent_id": opponent,
                "period": p["period"],
                "possession_id": p["possession_id"],
                "decision_clock": decision,
                "last_prefix_event_index": last["index"],
                "features": full,
                "baseline_features": base,
                "shot_created": int(bool(later)),
                "later_npxg": sum(
                    s["xg"] for s in later if isinstance(s["xg"], (int, float))
                ),
            }
        )
    total_clock = sum(r["possession_clock_seconds"] for r in teams.values())
    total_actions = sum(r["on_ball_actions"] for r in teams.values())
    match_seconds = sum(high - low for low, high in period_extents.values())
    for tid, row in teams.items():
        opponent = teams[row["opponent_id"]]
        moves = row["completed_moves"]
        row.update(
            {
                "match_id": meta["match_id"],
                "date": meta["match_date"],
                "competition": meta["competition"]["competition_name"],
                "competition_id": meta["competition"]["competition_id"],
                "season": meta["season"]["season_name"],
                "season_id": meta["season"]["season_id"],
                "on_ball_action_share_proxy": (
                    row["on_ball_actions"] / total_actions if total_actions else None
                ),
                "possession_clock_share_proxy": (
                    row["possession_clock_seconds"] / total_clock
                    if total_clock
                    else None
                ),
                "clock_coverage_fraction": (
                    total_clock / match_seconds if match_seconds else None
                ),
                "official_possession_percentage": None,
                "mean_on_ball_origin_x": (
                    row["origin_x_sum"] / row["action_locations"]
                    if row["action_locations"]
                    else None
                ),
                "low_y_attack_share": row["low_y_moves"] / moves if moves else None,
                "central_attack_share": row["central_moves"] / moves if moves else None,
                "high_y_attack_share": row["high_y_moves"] / moves if moves else None,
                "shot_possession_rate": (
                    row["shot_possessions"] / row["possessions"]
                    if row["possessions"]
                    else None
                ),
                "npxg_per100_possessions": (
                    row["npxg"] / row["possessions"] * 100
                    if row["possessions"] and row["xg_known_shots"] == row["np_shots"]
                    else None
                ),
                "opponent_starting_formation": opponent["starting_formation"],
                "opponent_npxg": opponent["npxg"],
                "goals_in_event_log": scores[tid],
                "pressure_per100_opponent_passes": (
                    row["pressures"] / opponent["passes"] * 100
                    if opponent["passes"]
                    else None
                ),
                **context.get(tid, {}),
            }
        )
    centroids = [
        {
            "team_id": k[0],
            "position": k[1],
            "formation": k[2],
            "events": len(points),
            "mean_x": sum(p[0] for p in points) / len(points),
            "mean_y": sum(p[1] for p in points) / len(points),
        }
        for k, points in positions.items()
    ]
    return {
        "match_id": meta["match_id"],
        "teams": list(teams.values()),
        "prefixes": prefixes,
        "formation_changes": changes,
        "role_action_centroids": centroids,
        "audit": dict(audit),
    }


def collect(database="data/football.sqlite3", output=".tmp/tactical-relations"):
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    metadata = {}
    assets = {}
    for r in connection.execute(
        "SELECT payload FROM documents WHERE kind='statsbomb_matches'"
    ):
        for m in json.loads(r[0]):
            metadata[m["match_id"]] = m
    for r in connection.execute(
        "SELECT d.asset_id,a.path FROM documents d JOIN assets a ON a.id=d.asset_id WHERE d.kind='statsbomb_events'"
    ):
        try:
            mid = int(Path(r["path"]).stem)
        except ValueError:
            continue
        if mid in metadata:
            assets[mid] = r["asset_id"]
    history = defaultdict(lambda: deque(maxlen=8))
    elos = {}
    team_rows = []
    prefixes = []
    centroids = []
    formations = []
    audits = Counter()
    ordered = sorted(assets, key=lambda mid: (metadata[mid]["match_date"], mid))
    dates = defaultdict(list)
    for mid in ordered:
        dates[metadata[mid]["match_date"]].append(mid)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for day, mids in sorted(dates.items()):
        updates = []
        for mid in mids:
            meta = metadata[mid]
            comp = meta["competition"]["competition_id"]
            home = meta["home_team"]["home_team_id"]
            away = meta["away_team"]["away_team_id"]
            context = {}
            for tid, opp in [(home, away), (away, home)]:
                key = (comp, tid)
                ok = (comp, opp)
                past = list(history[ok])
                difference = elos.get(key, 1500.0) - elos.get(ok, 1500.0)
                context[tid] = {
                    "pre_match_archive_elo_difference": difference,
                    "opponent_prior_matches": len(past),
                    "opponent_prior_pressure_proxy": (
                        sum(p["pressure_per100_opponent_passes"] for p in past)
                        / len(past)
                        if past
                        and all(
                            p["pressure_per100_opponent_passes"] is not None
                            for p in past
                        )
                        else None
                    ),
                    "opponent_prior_possession_proxy": (
                        sum(p["possession_clock_share_proxy"] for p in past) / len(past)
                        if past
                        and all(
                            p["possession_clock_share_proxy"] is not None for p in past
                        )
                        else None
                    ),
                }
            payload = connection.execute(
                "SELECT payload FROM documents WHERE asset_id=?", (assets[mid],)
            ).fetchone()[0]
            result = analyze_match(json.loads(payload), meta, context)
            team_rows.extend(result["teams"])
            prefixes.extend(result["prefixes"])
            centroids.extend(
                [{"match_id": mid, **r} for r in result["role_action_centroids"]]
            )
            formations.extend(
                [{"match_id": mid, **r} for r in result["formation_changes"]]
            )
            audits.update(result["audit"])
            score = (
                1
                if meta["home_score"] > meta["away_score"]
                else 0.5 if meta["home_score"] == meta["away_score"] else 0
            )
            expected = 1 / (
                1
                + 10
                ** (-(context[home]["pre_match_archive_elo_difference"] + 60) / 400)
            )
            updates.append((comp, home, away, 20 * (score - expected), result["teams"]))
        # All outcomes update only after the whole calendar date, preserving unknown kickoff order.
        for comp, home, away, delta, rows in updates:
            elos[(comp, home)] = elos.get((comp, home), 1500) + delta
            elos[(comp, away)] = elos.get((comp, away), 1500) - delta
            for row in rows:
                history[(comp, row["team_id"])].append(row)
        if len(team_rows) % 200 < 2:
            print("Tactical fixtures processed", len(team_rows) // 2, flush=True)
    tracking = []
    for r in connection.execute(
        "SELECT d.payload FROM documents d JOIN assets a ON a.id=d.asset_id WHERE a.path LIKE 'movement/skillcorner-%.json'"
    ):
        report = json.loads(r[0])
        phase = report.get("provided_phase_profiles", {}).get("profiles", [])
        shapes = {str(p["team"]): p for p in report.get("team_shape", [])}
        teams = {str(p["team_id"]) for p in phase}
        total = sum(
            p["duration_seconds"] for p in phase if p["state"] == "in_possession"
        )
        for tid in teams:
            own = [p for p in phase if str(p["team_id"]) == tid]
            duration = sum(
                p["duration_seconds"] for p in own if p["state"] == "in_possession"
            )
            tracking.append(
                {
                    "match_id": report["match_id"],
                    "date": report["date"],
                    "team_id": tid,
                    "opponent_id": next((t for t in teams if t != tid), None),
                    "classified_phase_possession_share": (
                        duration / total if total else None
                    ),
                    "detected_shape": shapes.get(tid),
                    "provided_phase_shapes": own,
                    "scope": "Classified phase-time share and partial visible-player shape; vendor shapes may include extrapolation.",
                }
            )
    data = {
        "matches": len(ordered),
        "team_match_rows": team_rows,
        "prefixes": prefixes,
        "role_action_centroids": centroids,
        "formation_changes": formations,
        "tracking_context": tracking,
        "audit": dict(audits),
    }
    (output / "data.json").write_text(json.dumps(data))
    connection.close()
    return data
