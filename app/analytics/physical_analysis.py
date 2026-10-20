"""Link detected tracking workload to spatial exchanges and transition phases."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
import hashlib
import json
import logging
import math
from pathlib import Path
import sqlite3

from .movement_lab import external_load, rotation_candidates, smooth_segments
from .tracking import _seconds

logger = logging.getLogger(__name__)

METHOD_SOURCES = [
    {
        "title": "Croatia: Evolving third-player combinations",
        "publisher": "FIFA",
        "url": "https://www.fifatrainingcentre.com/en/game/game-analysis/world-class-tactics/croatia-evolving-third-player-combinations.php",
        "published": "2026-05-19",
        "reviewed": "2026-10-09",
        "application": "Interpret movement relative to pressure, space and team-mate actions. A pair-order exchange is only a candidate for video review.",
    },
    {
        "title": "Attacking and defensive transitions, Adrian Boothroyd",
        "publisher": "FIFA",
        "url": "https://www.fifatrainingcentre.com/en/practice/elite-sessions/transition-to-attacking/transitions.php",
        "published": "2022-04-19",
        "reviewed": "2026-10-09",
        "application": "Measure support after regaining possession and defensive reorganisation after loss; link geometry with counter-press events.",
    },
    {
        "title": "EURO 2024 physical analysis report",
        "publisher": "UEFA",
        "url": "https://editorial.uefa.com/resources/0297-1d4e3592fbf1-f11d4e1c826a-1000/uefa_euro_2024_physical_analysis_report_20250318094958.pdf",
        "published": "2025-03-18",
        "reviewed": "2026-10-09",
        "pages": [9, 34, 37, 42, 43],
        "application": "Speed zones >20 and >25 km/h; 1, 3, 5 and 10 minute rolling demands; positional and possession context. This optical sample uses different filtering and cannot reproduce UEFA tournament benchmarks.",
    },
    {
        "title": "Is it worth the effort? Understanding and contextualizing physical metrics in soccer",
        "publisher": "Llana et al., arXiv",
        "url": "https://arxiv.org/abs/2204.02313",
        "published": "2022-04-05",
        "reviewed": "2026-10-09",
        "application": "Associate running with technical-tactical context. We do not reproduce the paper possession-value model or interpret vendor EPV as ground truth.",
    },
]


def read_match(connection, metadata_asset):
    metadata = json.loads(
        connection.execute(
            "SELECT payload FROM documents WHERE asset_id=?", (metadata_asset["id"],)
        ).fetchone()[0]
    )
    mid = metadata["id"]
    assets = {
        r["path"].split("/")[-1]: dict(r)
        for r in connection.execute(
            "SELECT * FROM assets WHERE dataset_id=? AND path LIKE ?",
            (metadata_asset["dataset_id"], f"data/matches/{mid}/%"),
        )
    }
    tracking = assets[f"{mid}_tracking_extrapolated.jsonl"]
    event_asset = assets[f"{mid}_dynamic_events.csv"]
    phase_asset = assets[f"{mid}_phases_of_play.csv"]
    events = [
        dict(r)
        for r in connection.execute(
            "SELECT * FROM skillcorner_dynamic_events WHERE asset_id=?",
            (event_asset["id"],),
        )
    ]
    phases = [
        dict(r)
        for r in connection.execute(
            "SELECT * FROM skillcorner_phases WHERE asset_id=? ORDER BY CAST(frame_start AS INTEGER)",
            (phase_asset["id"],),
        )
    ]
    lookup = {p["id"]: p for p in metadata["players"]}
    points, snapshots = defaultdict(list), []
    audit = {
        "frames": 0,
        "invalid_period_frames": 0,
        "detected_points": 0,
        "excluded_extrapolated_or_invalid_points": 0,
        "possession_known_snapshots": 0,
    }
    try:
        import orjson

        loads = orjson.loads
    except ImportError:
        loads = json.loads
    for (raw,) in connection.execute(
        "SELECT payload FROM json_lines WHERE asset_id=? ORDER BY row_number",
        (tracking["id"],),
    ):
        frame = loads(raw)
        audit["frames"] += 1
        period = frame.get("period")
        if period not in (1, 2) or frame.get("timestamp") is None:
            audit["invalid_period_frames"] += 1
            continue
        fid, time = frame["frame"], _seconds(frame["timestamp"])
        team_points = defaultdict(dict)
        for point in frame.get("player_data", []):
            pid, x, y = point.get("player_id"), point.get("x"), point.get("y")
            player = lookup.get(pid)
            valid = (
                player
                and point.get("is_detected") is True
                and isinstance(x, (int, float))
                and isinstance(y, (int, float))
                and math.isfinite(x)
                and math.isfinite(y)
            )
            if not valid:
                audit["excluded_extrapolated_or_invalid_points"] += 1
                continue
            audit["detected_points"] += 1
            points[pid].append((period, fid, time, x, y))
            if fid % 10 or player["player_role"]["acronym"] == "GK":
                continue
            team = player["team_id"]
            sign = (
                1 if metadata["home_team_side"][period - 1] == "left_to_right" else -1
            )
            if team != metadata["home_team"]["id"]:
                sign *= -1
            team_points[team][pid] = (sign * x, sign * y)
        if fid % 10:
            continue
        owner_player = lookup.get((frame.get("possession") or {}).get("player_id"))
        owner = owner_player["team_id"] if owner_player else None
        ball = frame.get("ball_data") or {}
        ball_x = ball.get("x") if ball.get("is_detected") is True else None
        snapshots.append(
            {
                "period": period,
                "frame": fid,
                "seconds": time,
                "owner": owner,
                "teams": dict(team_points),
                "ball_x": ball_x,
            }
        )
        audit["possession_known_snapshots"] += owner is not None
    return (
        metadata,
        points,
        snapshots,
        events,
        phases,
        audit,
        [metadata_asset, tracking, event_asset, phase_asset],
    )


def geometry(snapshot, team, metadata):
    positions = list(snapshot["teams"].get(team, {}).values())
    if len(positions) < 8:
        return {
            "visible_outfield_players": len(positions),
            "width_m": None,
            "depth_m": None,
            "players_behind_ball": None,
        }
    xs, ys = zip(*positions)
    behind = None
    if snapshot["ball_x"] is not None:
        sign = (
            1
            if metadata["home_team_side"][snapshot["period"] - 1] == "left_to_right"
            else -1
        )
        if team != metadata["home_team"]["id"]:
            sign *= -1
        behind = sum(x < snapshot["ball_x"] * sign for x in xs)
    return {
        "visible_outfield_players": len(positions),
        "width_m": max(ys) - min(ys),
        "depth_m": max(xs) - min(xs),
        "players_behind_ball": behind,
    }


def interval_load(segments, period, start, end):
    selected = [
        r
        for r in segments
        if r["period"] == period and r["frame"] - 5 >= start and r["frame"] <= end
    ]
    observed = sum(r["duration"] for r in selected)
    distance = sum(r["duration"] * r["speed_mps"] for r in selected)
    high_speed = sum(
        r["duration"] * r["speed_mps"] for r in selected if r["speed_mps"] > 20 / 3.6
    )
    return {
        "observed_seconds": observed,
        "distance_m": distance,
        "high_speed_distance_m": high_speed,
        "distance_m_per_observed_minute": (
            distance * 60 / observed if observed else None
        ),
    }


def contextual_load(metadata, segments, phases):
    by_period = defaultdict(list)
    for phase in phases:
        by_period[int(phase["period"])].append(phase)
    starts = {p: [int(r["frame_start"]) for r in rows] for p, rows in by_period.items()}
    lookup = {p["id"]: p for p in metadata["players"]}
    groups = defaultdict(
        lambda: {
            "observed_seconds": 0.0,
            "distance_m": 0.0,
            "high_speed_distance_m": 0.0,
        }
    )
    classified = defaultdict(float)
    for pid, player_segments in segments.items():
        player = lookup[pid]
        for segment in player_segments:
            period, frame = segment["period"], segment["frame"]
            index = bisect_right(starts.get(period, []), frame) - 1
            if index < 0:
                continue
            phase = by_period[period][index]
            if frame - 5 < int(phase["frame_start"]) or frame > int(phase["frame_end"]):
                continue
            own = player["team_id"] == int(phase["team_in_possession_id"])
            state = "in_possession" if own else "out_of_possession"
            label = phase[
                (
                    "team_in_possession_phase_type"
                    if own
                    else "team_out_of_possession_phase_type"
                )
            ]
            key = (
                player["team_id"],
                player["player_role"]["position_group"],
                state,
                label,
            )
            group = groups[key]
            dt, speed = segment["duration"], segment["speed_mps"]
            group["observed_seconds"] += dt
            group["distance_m"] += dt * speed
            group["high_speed_distance_m"] += dt * speed if speed > 20 / 3.6 else 0
            classified[pid] += dt
    rows = [
        {
            "match_id": metadata["id"],
            "team_id": k[0],
            "position_group": k[1],
            "state": k[2],
            "phase": k[3],
            **v,
            "distance_m_per_observed_player_minute": v["distance_m"]
            * 60
            / v["observed_seconds"],
        }
        for k, v in groups.items()
    ]
    return rows, classified


def annotate_windows(metadata, rotations, segments, snapshots, events, phases):
    by_frame = {s["frame"]: s for s in snapshots}
    possession_events = [e for e in events if e["event_type"] == "player_possession"]
    shots = [e for e in possession_events if e["end_type"] == "shot"]
    for rotation in rotations:
        end = rotation["decision_frame"]
        team = rotation["team_id"]
        period = rotation["period"]
        stop = next(
            (
                int(e["frame_end"])
                for e in sorted(possession_events, key=lambda e: int(e["frame_end"]))
                if int(e["period"]) == period
                and int(e["frame_end"]) > end
                and e["game_interruption_after"]
            ),
            None,
        )
        observed_end = max(
            (s["frame"] for s in snapshots if s["period"] == period), default=end
        )
        complete = observed_end >= end + 100 and (stop is None or stop >= end + 100)
        future_shot = any(
            int(e["team_id"]) == team
            and int(e["period"]) == period
            and end
            < int(e["frame_end"])
            <= min(end + 100, stop if stop is not None else end + 100)
            for e in shots
        )
        rotation.update(
            {
                "match_id": metadata["id"],
                "followup_10s_complete": complete,
                "shot_within_next10s": (
                    True if future_shot else False if complete else None
                ),
                "vendor_run_types_near_exchange": sorted(
                    {
                        e["event_subtype"]
                        for e in events
                        if e["event_type"] == "off_ball_run"
                        and int(e["team_id"]) == team
                        and int(e["period"]) == period
                        and int(e["frame_start"]) <= end
                        and int(e["frame_end"]) >= rotation["start_frame"]
                    }
                ),
            }
        )
    windows = []
    team_ids = {metadata["home_team"]["id"], metadata["away_team"]["id"]}
    lookup = {p["id"]: p for p in metadata["players"]}
    for phase in phases:
        if phase["team_in_possession_phase_type"] not in {"transition", "quick_break"}:
            continue
        start, period = int(phase["frame_start"]), int(phase["period"])
        end = min(start + 100, int(phase["frame_end"]))
        if end - start < 30:
            continue
        gaining = int(phase["team_in_possession_id"])
        losing = next(t for t in team_ids if t != gaining)
        nearest = by_frame.get(((start + 5) // 10) * 10)
        nearest_end = by_frame.get((end // 10) * 10)
        # A classified attacking transition does not prove that the immediately
        # preceding state was an open-play turnover; retain provider taxonomy.
        window = {
            "match_id": metadata["id"],
            "period": period,
            "start_frame": start,
            "end_frame": end,
            "duration_seconds": (end - start) / 10,
            "team_id": gaining,
            "opponent_id": losing,
            "phase": phase["team_in_possession_phase_type"],
            "phase_ends_with_shot": phase["team_possession_lead_to_shot"] == "True",
            "shot_in_window": any(
                int(e["team_id"]) == gaining
                and int(e["period"]) == period
                and start < int(e["frame_end"]) <= end
                for e in shots
            ),
            "defending_shape_at_start": (
                geometry(nearest, losing, metadata)
                if nearest and nearest["period"] == period
                else None
            ),
            "defending_shape_at_end": (
                geometry(nearest_end, losing, metadata)
                if nearest_end and nearest_end["period"] == period
                else None
            ),
            "vendor_counterpress_events_first5s": sum(
                e["event_type"] == "on_ball_engagement"
                and e["event_subtype"] == "counter_press"
                and int(e["team_id"]) == losing
                and int(e["period"]) == period
                and start <= int(e["frame_start"]) <= min(end, start + 50)
                for e in events
            ),
            "player_load": [],
        }
        for pid, series in segments.items():
            load = interval_load(series, period, start, end)
            if load["observed_seconds"]:
                window["player_load"].append(
                    {
                        "player_id": pid,
                        "team_id": lookup[pid]["team_id"],
                        "position_group": lookup[pid]["player_role"]["position_group"],
                        "window_coverage_fraction": load["observed_seconds"]
                        / window["duration_seconds"],
                        **load,
                    }
                )
        windows.append(window)
    return windows


def persist(database, report, players, rotations, transitions, phases, output):
    from app.data_service.warehouse import Warehouse
    from .artifacts import archive_outputs

    run_id = report["run_id"]
    with Warehouse(database) as db:
        db.connection.execute(
            "CREATE TABLE IF NOT EXISTS movement_runs(run_id TEXT PRIMARY KEY,payload TEXT)"
        )
        db.connection.execute(
            "INSERT INTO movement_runs VALUES(?,?) ON CONFLICT(run_id) DO UPDATE SET payload=excluded.payload",
            (run_id, json.dumps(report)),
        )
        for table, rows in [
            ("player_external_load", players),
            ("movement_rotations", rotations),
            ("movement_transitions", transitions),
            ("movement_phase_load", phases),
            ("professional_method_sources", METHOD_SOURCES),
        ]:
            db.connection.execute(
                f"CREATE TABLE IF NOT EXISTS {table}(run_id TEXT REFERENCES movement_runs(run_id),row_number INTEGER,payload TEXT,PRIMARY KEY(run_id,row_number))"
            )
            db.connection.execute(f"DELETE FROM {table} WHERE run_id=?", (run_id,))
            db.connection.executemany(
                f"INSERT INTO {table} VALUES(?,?,?)",
                [(run_id, i, json.dumps(row)) for i, row in enumerate(rows, 1)],
            )
        archive_outputs(
            db, "movement_lab_artifacts", output, "movement_lab_artifact", run_id
        )
        db.connection.commit()
        db.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def run(database="data/football.sqlite3", output=".tmp/movement-lab"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(f"file:{Path(database).resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    assets = [
        dict(r)
        for r in connection.execute(
            "SELECT a.* FROM assets a JOIN datasets d ON d.id=a.dataset_id WHERE d.provider='skillcorner' AND a.path LIKE '%_match.json' AND a.complete=1 ORDER BY a.path"
        )
    ]
    if not assets:
        raise ValueError("No SkillCorner match metadata in database")
    players, rotations, transitions, phase_load, audits, manifest = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for asset in assets:
        metadata, points, snapshots, events, phases, audit, inputs = read_match(
            connection, asset
        )
        segments = {pid: smooth_segments(series) for pid, series in points.items()}
        context, classified = contextual_load(metadata, segments, phases)
        phase_load.extend(context)
        candidates = rotation_candidates(snapshots)
        transitions.extend(
            annotate_windows(metadata, candidates, segments, snapshots, events, phases)
        )
        rotations.extend(candidates)
        for player in metadata["players"]:
            series = segments.get(player["id"], [])
            nominal = ((player.get("playing_time") or {}).get("total") or {}).get(
                "minutes_played"
            )
            if not nominal:
                continue
            load = external_load(series, nominal)
            players.append(
                {
                    "match_id": metadata["id"],
                    "date": metadata["date_time"][:10],
                    "player_id": player["id"],
                    "name": player["short_name"],
                    "team_id": player["team_id"],
                    "position_group": player["player_role"]["position_group"],
                    "phase_classified_observed_seconds": classified[player["id"]],
                    "phase_classified_fraction": (
                        classified[player["id"]] / load["observed_seconds"]
                        if load["observed_seconds"]
                        else None
                    ),
                    **load,
                }
            )
        audits.append(
            {
                "match_id": metadata["id"],
                "date": metadata["date_time"][:10],
                "snapshot_count": len(snapshots),
                **audit,
            }
        )
        manifest.extend(
            {k: r[k] for k in ("id", "dataset_id", "path", "sha256", "bytes")}
            for r in inputs
        )
        logger.info(
            "Match %s: %s rotations, %s observed players",
            metadata["id"],
            len(candidates),
            len(segments),
        )
    connection.close()
    signature = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
        + Path(__file__).read_bytes()
        + Path(__file__).with_name("movement_lab.py").read_bytes()
    ).hexdigest()[:16]
    eligible_rotations = [r for r in rotations if r["shot_within_next10s"] is not None]
    report = {
        "run_id": signature,
        "matches": len(assets),
        "player_match_rows": len(players),
        "rotation_candidates": len(rotations),
        "classified_transition_windows_ge3s": len(transitions),
        "phase_load_groups": len(phase_load),
        "audits": audits,
        "inputs": manifest,
        "rotation_followup": {
            "eligible": len(eligible_rotations),
            "shots_next10s": sum(r["shot_within_next10s"] for r in eligible_rotations),
            "interpretation": "Descriptive selected exchanges; no matched control or causal effect estimate.",
        },
        "protocol": {
            "tracking": "Detected positions only; 10 Hz input, consecutive six-point half-second chord velocity, scalar acceleration between adjacent windows. Raw spikes >12m/s and all gaps reset filters.",
            "speed_zones": "UEFA-compatible >20 and >25 km/h; sensitivity distances >19.8 and >25.2 km/h are retained.",
            "acceleration": "Local exploratory threshold +/-3m/s2 sustained >=1s, not an adopted UEFA cutoff.",
            "sprint_repeats": "Local definition: >=3 bouts above25km/h lasting>=1s with <=20s recovery, within one period.",
            "peaks": "Rolling1,3,5,10min clock windows, 1Hz anchors, >=80% detected segment exposure; distances are not scaled to fill missing time.",
            "rotation": "Five consecutive 1Hz possession samples, persistent3m pair-order change, both move>=3m, pair<=25m, cross-axis gap<=15m,20s cooldown.",
            "transition": "Provider quick_break/transition phases, first<=10s, minimum3s, variable horizon retained. Counterpress labels and phase taxonomy are vendor-derived.",
            "shape": "At least8 detected outfield players; rest-defense count additionally needs a detected ball. Both axes orient toward each team attacking goal.",
            "fitness_limit": "Camera-selected external match load is not conditioning capacity, fatigue diagnosis, injury risk or complete GPS workload. No heart rate, RPE, training load or fitness-test measurements are available.",
            "sample_limit": "20 selected A-League2024/25 fixtures; no unseen league or season validation.",
        },
        "professional_sources": METHOD_SOURCES,
    }
    for name, value in [
        ("report", report),
        ("players", players),
        ("rotations", rotations),
        ("transitions", transitions),
        ("phase-load", phase_load),
    ]:
        (output / f"{name}.json").write_text(json.dumps(value, indent=2))
    persist(database, report, players, rotations, transitions, phase_load, output)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "run_id",
                    "matches",
                    "player_match_rows",
                    "rotation_candidates",
                    "classified_transition_windows_ge3s",
                    "phase_load_groups",
                )
            },
            indent=2,
        )
    )
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="data/football.sqlite3")
    parser.add_argument("--output", default=".tmp/movement-lab")
    args = parser.parse_args()
    run(args.database, args.output)
