"""Tracking sample analysis plus SkillCorner's supplied off-ball/physical aggregates."""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

from app.analytics.tracking import (
    metrica_csv_frames,
    metrica_epts_frames,
    skillcorner_frames,
    summarize_tracking,
)
from app.data_service.archive import read_asset_json, verify_asset_file
from app.file_io import write_json_atomic as _atomic_write_json

logger = logging.getLogger(__name__)


def _boolean(value):
    if str(value).lower() in {"true", "1"}:
        return True
    if str(value).lower() in {"false", "0"}:
        return False
    return None


def _number(value):
    try:
        import math

        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def analyze_dynamic_events(path: Path) -> dict:
    types, subtypes, players = Counter(), Counter(), {}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            kind = row["event_type"]
            types[kind] += 1
            if kind != "off_ball_run":
                continue
            subtype = row.get("event_subtype") or "unknown"
            subtypes[subtype] += 1
            key = (row["team_id"], row["player_id"])
            if key not in players:
                players[key] = {
                    "team_id": row["team_id"],
                    "player_id": row["player_id"],
                    "player": row.get("player_name"),
                    "position": row.get("player_position"),
                    "off_ball_runs": 0,
                    "run_subtypes": {},
                    "targeted_runs": 0,
                    "runs_received": 0,
                    "targeted_status_known": 0,
                    "received_status_known": 0,
                    "runs_leading_to_shot": 0,
                    "fully_extrapolated_runs": 0,
                    "unknown_extrapolation_status": 0,
                    "duration_seconds": 0.0,
                }
            player = players[key]
            player["off_ball_runs"] += 1
            player["run_subtypes"][subtype] = player["run_subtypes"].get(subtype, 0) + 1
            targeted, received = _boolean(row.get("targeted")), _boolean(
                row.get("received")
            )
            player["targeted_runs"] += targeted is True
            player["runs_received"] += received is True
            player["targeted_status_known"] += targeted is not None
            player["received_status_known"] += received is not None
            player["runs_leading_to_shot"] += _boolean(row.get("lead_to_shot")) is True
            extrapolated = _boolean(row.get("fully_extrapolated"))
            player["fully_extrapolated_runs"] += extrapolated is True
            player["unknown_extrapolation_status"] += extrapolated is None
            player["duration_seconds"] += _number(row.get("duration")) or 0.0
    for player in players.values():
        player["targeted_fraction_known"] = (
            player["targeted_runs"] / player["targeted_status_known"]
            if player["targeted_status_known"]
            else None
        )
        player["received_fraction_known"] = (
            player["runs_received"] / player["received_status_known"]
            if player["received_status_known"]
            else None
        )
    return {
        "event_types": dict(types),
        "run_subtypes": dict(subtypes),
        "players": list(players.values()),
        "note": "Provider-classified dynamic events, distinct from detected-only physical distance estimates. Empty quality flags are unknown.",
    }


def analyze_provider_aggregates(root: Path, index: dict) -> dict:
    reports = {}
    for relative in sorted(index):
        if not relative.startswith("data/aggregates/") or not relative.endswith(".csv"):
            continue
        path = verify_asset_file(root, relative, index)
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows = []
            for original in reader:
                # Public birthdays are unnecessary for this scouting output.
                row = {
                    key: (
                        _number(value) if _number(value) is not None else value or None
                    )
                    for key, value in original.items()
                    if key != "player_birthdate"
                }
                rows.append(row)
        reports[Path(relative).stem] = {
            "players": rows,
            "rows": len(rows),
            "scope": "Provider season aggregates for AUS A-League 2024/25; supplied inclusion/QC rules apply.",
            "units_note": "Keep vendor field definitions. Average performance metrics are not season totals; p30tip fields use 30 minutes in possession.",
        }
    return reports


def analyze_phases(path: Path, team_ids: list[int]) -> dict:
    groups = {}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            owner = _number(row.get("team_in_possession_id"))
            duration = _number(row.get("duration"))
            if (
                owner is None
                or duration is None
                or duration < 0
                or int(owner) not in team_ids
            ):
                continue
            for state, team in [
                ("in_possession", int(owner)),
                (
                    "out_of_possession",
                    next((team for team in team_ids if team != int(owner)), None),
                ),
            ]:
                if team is None:
                    continue
                key = (team, state, row.get(f"team_{state}_phase_type") or "unknown")
                if key not in groups:
                    groups[key] = {
                        "team_id": team,
                        "state": state,
                        "phase": key[2],
                        "phases": 0,
                        "duration_seconds": 0.0,
                        "width_sum": 0.0,
                        "width_seconds": 0.0,
                        "depth_sum": 0.0,
                        "depth_seconds": 0.0,
                        "shot_true": 0,
                        "shot_known": 0,
                    }
                group = groups[key]
                group["phases"] += 1
                group["duration_seconds"] += duration
                for name, column in [
                    ("width", f"team_{state}_width_start"),
                    ("depth", f"team_{state}_length_start"),
                ]:
                    value = _number(row.get(column))
                    if value is not None and value >= 0:
                        group[f"{name}_sum"] += value * duration
                        group[f"{name}_seconds"] += duration
                shot = _boolean(row.get("team_possession_lead_to_shot"))
                group["shot_true"] += shot is True
                group["shot_known"] += shot is not None
    profiles = []
    for group in groups.values():
        result = {
            name: group[name]
            for name in ("team_id", "state", "phase", "phases", "duration_seconds")
        }
        for name in ("width", "depth"):
            result[f"provided_mean_{name}_m"] = (
                group[f"{name}_sum"] / group[f"{name}_seconds"]
                if group[f"{name}_seconds"]
                else None
            )
        result["shot_ending_fraction_known"] = (
            group["shot_true"] / group["shot_known"] if group["shot_known"] else None
        )
        profiles.append(result)
    return {
        "profiles": profiles,
        "note": "Provider-classified phases and supplied shape estimates, duration-weighted. They can include extrapolated positions and differ from detected-only tracking summaries.",
    }


def analyze_movement(data_root: Path, output_dir: Path) -> dict:
    data_root, output_dir = Path(data_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matches, errors = [], []
    for source in ("skillcorner", "metrica"):
        root = data_root / source
        manifest = json.loads((root / "manifest.json").read_text())
        index = {entry["path"]: entry for entry in manifest["files"]}
        if source == "skillcorner":
            metadata_paths = [
                path
                for path in sorted(index)
                if path.startswith("data/matches/") and path.endswith("_match.json")
            ]
            for metadata_path in metadata_paths:
                metadata = read_asset_json(root, metadata_path, index)
                mid = metadata["id"]
                prefix = f"data/matches/{mid}/{mid}"
                try:
                    tracking = verify_asset_file(
                        root, prefix + "_tracking_extrapolated.jsonl", index
                    )
                    report = summarize_tracking(skillcorner_frames(tracking, metadata))
                    dynamic = analyze_dynamic_events(
                        verify_asset_file(root, prefix + "_dynamic_events.csv", index)
                    )
                    report.update(
                        {
                            "source": source,
                            "match_id": mid,
                            "date": metadata["date_time"],
                            "home": metadata["home_team"]["name"],
                            "away": metadata["away_team"]["name"],
                            "home_score": metadata["home_team_score"],
                            "away_score": metadata["away_team_score"],
                            "pitch_length": metadata["pitch_length"],
                            "pitch_width": metadata["pitch_width"],
                            "dynamic_events": dynamic,
                            "revision": manifest["revision"],
                        }
                    )
                    report["provided_phase_profiles"] = analyze_phases(
                        verify_asset_file(root, prefix + "_phases_of_play.csv", index),
                        [metadata["home_team"]["id"], metadata["away_team"]["id"]],
                    )
                    _atomic_write_json(output_dir / f"{source}-{mid}.json", report)
                    matches.append(
                        {
                            "source": source,
                            "match_id": mid,
                            "frames": report["frames"],
                            "players": len(report["players"]),
                            "home": report["home"],
                            "away": report["away"],
                            "off_ball_runs": sum(dynamic["run_subtypes"].values()),
                            "shape_samples": sum(
                                r["qualified_shape_samples"]
                                for r in report["team_shape"]
                            ),
                        }
                    )
                    logger.info(
                        "Tracking analyzed %s %s: %d frames",
                        source,
                        mid,
                        report["frames"],
                    )
                except Exception as exc:
                    errors.append(
                        {"source": source, "match_id": mid, "message": str(exc)}
                    )
            aggregates = analyze_provider_aggregates(root, index)
            _atomic_write_json(
                output_dir / "skillcorner-season-aggregates.json", aggregates
            )
        else:
            for game in (1, 2, 3):
                prefix = f"data/Sample_Game_{game}/Sample_Game_{game}"
                try:
                    if game < 3:
                        home = verify_asset_file(
                            root, prefix + "_RawTrackingData_Home_Team.csv", index
                        )
                        away = verify_asset_file(
                            root, prefix + "_RawTrackingData_Away_Team.csv", index
                        )
                        frames = metrica_csv_frames(home, away)
                    else:
                        frames = metrica_epts_frames(
                            verify_asset_file(root, prefix + "_metadata.xml", index),
                            verify_asset_file(root, prefix + "_tracking.txt", index),
                        )
                    report = summarize_tracking(frames)
                    report.update(
                        {
                            "source": source,
                            "match_id": game,
                            "identity_note": "Anonymized sample; jersey identities cannot be linked across games.",
                            "pitch_length": 105,
                            "pitch_width": 68,
                            "revision": manifest["revision"],
                        }
                    )
                    _atomic_write_json(output_dir / f"{source}-{game}.json", report)
                    matches.append(
                        {
                            "source": source,
                            "match_id": game,
                            "frames": report["frames"],
                            "players": len(report["players"]),
                            "shape_samples": sum(
                                r["qualified_shape_samples"]
                                for r in report["team_shape"]
                            ),
                        }
                    )
                    logger.info(
                        "Tracking analyzed %s %s: %d frames",
                        source,
                        game,
                        report["frames"],
                    )
                except Exception as exc:
                    errors.append(
                        {"source": source, "match_id": game, "message": str(exc)}
                    )
    result = {
        "matches": matches,
        "errors": errors,
        "notes": [
            "SkillCorner broadcast visibility and extrapolation are explicitly separated.",
            "Observed distances are not full-match fitness rankings when detected coverage is incomplete.",
            "Season aggregate and dynamic-event definitions remain provider-specific.",
            "Metrica CSV games lack goalkeeper role labels; their shape estimates include tracked keepers.",
        ],
    }
    _atomic_write_json(output_dir / "report.json", result)
    return result
