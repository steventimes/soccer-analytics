"""Dataset-wide scouting summaries and a chronological binary shot-quality experiment."""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from app.analytics.tactics import analyze_event_match, fit_expected_threat, grid_index
from app.data_service.archive import read_asset_json
from app.file_io import write_json_atomic as _atomic_write_json

logger = logging.getLogger(__name__)
COUNTS = (
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
)


def _minute(value):
    if value is None:
        return None
    fields = str(value).split(":")
    return float(fields[0]) + (float(fields[1]) / 60 if len(fields) > 1 else 0)


def _minutes(lineups, final_minute):
    result = {}
    for team in lineups:
        for player in team.get("lineup", []):
            intervals = []
            for position in player.get("positions", []):
                start = _minute(position.get("from"))
                end = _minute(position.get("to"))
                if start is not None:
                    intervals.append((start, end if end is not None else final_minute))
            merged = []
            for start, end in sorted(intervals):
                if end < start:
                    continue
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            result[player["player_id"]] = (
                sum(end - start for start, end in merged) if merged else None
            )
    return result


def fit_shot_models(shots: list[dict]) -> dict:
    rows = [
        row
        for row in shots
        if not row["penalty"] and row["period"] <= 4 and row["provider_xg"] is not None
    ]
    matches = sorted({(row["date"], row["match_id"]) for row in rows})
    if len(matches) < 10:
        return {"available": False, "reason": "fewer_than_ten_matches"}
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    split = max(1, int(len(matches) * 0.8))
    cutoff = matches[split][0]
    train = [row for row in rows if row["date"] < cutoff]
    test = [row for row in rows if row["date"] >= cutoff]
    names = ["distance_units", "angle_radians", "header", "set_piece", "under_pressure"]
    matrix = lambda data: [[float(row[name]) for name in names] for row in data]
    y_train, y_test = [row["goal"] for row in train], [row["goal"] for row in test]
    if len(set(y_train)) != 2 or len(set(y_test)) != 2:
        return {"available": False, "reason": "missing_goal_or_nongoal_class"}
    results = []
    probabilities = {"provider_statsbomb": [row["provider_xg"] for row in test]}
    candidates = {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=150,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=42,
        ),
    }
    for name, estimator in candidates.items():
        estimator.fit(matrix(train), y_train)
        probabilities[name] = estimator.predict_proba(matrix(test))[:, 1].tolist()
    for name, prediction in probabilities.items():
        results.append(
            {
                "model": name,
                "log_loss": float(log_loss(y_test, prediction)),
                "brier_score": float(brier_score_loss(y_test, prediction)),
                "auc": float(roc_auc_score(y_test, prediction)),
                "mean_probability": float(np.mean(prediction)),
                "observed_goal_rate": float(np.mean(y_test)),
            }
        )
    return {
        "available": True,
        "train_shots": len(train),
        "test_shots": len(test),
        "cutoff_date": cutoff,
        "train_matches": len({row["match_id"] for row in train}),
        "test_matches": len({row["match_id"] for row in test}),
        "features": names,
        "results": results,
        "protocol": "Whole matches split by date. Same-day matches stay together. Non-penalty shots with provider xG only. No shot outcomes/endpoints or provider xG enter model features.",
        "note": "Pooled public historical samples; vendor xG is a supplied reference with external training provenance.",
    }


def analyze_statsbomb(
    root: Path,
    output_dir: Path,
    *,
    maximum_matches: int | None = None,
    competition_id: int | None = None,
    season_id: int | None = None,
) -> dict:
    root, output_dir = Path(root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "matches").mkdir(exist_ok=True)
    manifest = json.loads((root / "manifest.json").read_text())
    index = {entry["path"]: entry for entry in manifest["files"]}
    metadata = {}
    for path in sorted(index):
        if path.startswith("data/matches/") and path.endswith(".json"):
            for match in read_asset_json(root, path, index):
                mid = match["match_id"]
                if (
                    mid in metadata
                    and metadata[mid]["competition"]["competition_id"]
                    != match["competition"]["competition_id"]
                ):
                    raise ValueError("Match belongs to conflicting competitions")
                metadata[mid] = match
    ordered = sorted(
        metadata.values(), key=lambda match: (match["match_date"], match["match_id"])
    )
    ordered = [
        match
        for match in ordered
        if (
            competition_id is None
            or match["competition"]["competition_id"] == competition_id
        )
        and (season_id is None or match["season"]["season_id"] == season_id)
    ]
    if not ordered:
        raise ValueError("No matches in selected competition/season")
    if maximum_matches is not None:
        ordered = ordered[:maximum_matches]
    players, teams, shots, summaries, errors, spatial_summaries = {}, {}, [], [], [], []
    quality_exclusions = []
    transitions, attempts, shot_bins, goal_bins = (
        np.zeros((96, 96)),
        np.zeros(96),
        np.zeros(96),
        np.zeros(96),
    )
    player_net_bins = {}
    for number, match in enumerate(ordered, 1):
        mid = match["match_id"]
        try:
            events = read_asset_json(root, f"data/events/{mid}.json", index)
            frame_path = f"data/three-sixty/{mid}.json"
            try:
                frames = (
                    read_asset_json(root, frame_path, index)
                    if frame_path in index
                    else []
                )
            except json.JSONDecodeError as exc:
                frames = []
                quality_exclusions.append(
                    {
                        "match_id": mid,
                        "asset": frame_path,
                        "reason": "invalid_optional_360_json",
                        "message": str(exc),
                    }
                )
                logger.warning("隔离可选站位文件 %s: %s", frame_path, exc)
            lineups = read_asset_json(root, f"data/lineups/{mid}.json", index)
            report = analyze_event_match(events, frames, match_id=mid)
            group = {
                "competition_id": match["competition"]["competition_id"],
                "competition": match["competition"]["competition_name"],
                "season_id": match["season"]["season_id"],
                "season": match["season"]["season_name"],
                "gender": match["home_team"].get("home_team_gender"),
                "date": match["match_date"],
            }
            final_minute = max(
                (
                    event.get("minute", 0) + event.get("second", 0) / 60
                    for event in events
                    if event.get("period", 1) <= 4
                ),
                default=90,
            )
            minutes = _minutes(lineups, final_minute)
            for row in report["players"]:
                key = (
                    group["competition_id"],
                    group["season_id"],
                    row["team_id"],
                    row["player_id"],
                )
                if key not in players:
                    players[key] = {
                        **group,
                        **{
                            name: row[name]
                            for name in ("player_id", "player", "team_id", "position")
                        },
                        **dict.fromkeys(COUNTS, 0),
                        "matches": 0,
                        "covered_minutes": 0.0,
                        "matches_with_minutes": 0,
                    }
                    player_net_bins[key] = np.zeros(96)
                aggregate = players[key]
                for name in COUNTS:
                    aggregate[name] += row[name]
                aggregate["matches"] += 1
                if minutes.get(row["player_id"]) is not None:
                    aggregate["covered_minutes"] += minutes[row["player_id"]]
                    aggregate["matches_with_minutes"] += 1
            for row in report["teams"]:
                key = (group["competition_id"], group["season_id"], row["team_id"])
                if key not in teams:
                    teams[key] = {
                        **group,
                        "team_id": row["team_id"],
                        "team": row["team"],
                        **dict.fromkeys(COUNTS, 0),
                        "matches": 0,
                    }
                for name in COUNTS:
                    teams[key][name] += row[name]
                teams[key]["matches"] += 1
            for move in report["moves"]:
                start, end = grid_index(move["start"]), grid_index(move["end"])
                attempts[start] += 1
                if move["completed"]:
                    transitions[start, end] += 1
                    key = (
                        group["competition_id"],
                        group["season_id"],
                        move["team_id"],
                        move["player_id"],
                    )
                    if key in player_net_bins:
                        player_net_bins[key][end] += 1
                        player_net_bins[key][start] -= 1
            for shot in report["shots"]:
                shots.append({**group, **shot})
                if not shot["penalty"] and shot["period"] <= 4:
                    cell = grid_index(shot["start"])
                    shot_bins[cell] += 1
                    goal_bins[cell] += shot["goal"]
            match_summary = {
                "match_id": mid,
                **group,
                "home_team": match["home_team"]["home_team_name"],
                "away_team": match["away_team"]["away_team_name"],
                "home_score": match["home_score"],
                "away_score": match["away_score"],
                "events": report["event_count"],
                "teams": [
                    {k: v for k, v in row.items() if k != "action_grid"}
                    for row in report["teams"]
                ],
                "360_events": len(report["spatial_context"]),
                "formations": report["formations"],
            }
            summaries.append(match_summary)
            if frames:
                distances = [
                    row["nearest_visible_opponent_units"]
                    for row in report["spatial_context"]
                    if row["nearest_visible_opponent_units"] is not None
                ]
                spatial_summaries.append(
                    {
                        "match_id": mid,
                        "events_with_frames": len(report["spatial_context"]),
                        "mean_nearest_visible_opponent_units": (
                            sum(distances) / len(distances) if distances else None
                        ),
                        "continuous_tracking": False,
                    }
                )
            # Match-level drilldown retains the network and spatial examples without duplicating raw moves.
            if frames or match.get("competition_stage", {}).get("name") == "Final":
                _atomic_write_json(
                    output_dir / "matches" / f"{mid}.json",
                    {
                        **match_summary,
                        "passing_network": report["passing_network"],
                        "spatial_context": report["spatial_context"],
                        "action_grids": {
                            str(row["team_id"]): row["action_grid"]
                            for row in report["teams"]
                        },
                        "shots": report["shots"],
                    },
                )
        except Exception as exc:
            errors.append({"match_id": mid, "message": str(exc)})
        if number % 100 == 0:
            logger.info(
                "StatsBomb analysis %d/%d matches, %d shots, %d errors",
                number,
                len(ordered),
                len(shots),
                len(errors),
            )
    xt = fit_expected_threat(transitions, attempts, shot_bins, goal_bins)
    values = np.asarray(xt["grid"]).reshape(96)
    for key, row in players.items():
        row["net_xt_added"] = float(player_net_bins[key] @ values)
        known_minutes = (
            row["covered_minutes"]
            if row["matches_with_minutes"] == row["matches"]
            else 0.0
        )
        row["net_xt_per90"] = (
            row["net_xt_added"] * 90 / known_minutes if known_minutes else None
        )
        row["progressive_actions_per90"] = (
            (row["progressive_passes"] + row["progressive_carries"])
            * 90
            / known_minutes
            if known_minutes
            else None
        )
    xg = fit_shot_models(shots)
    result = {
        "source": "StatsBomb Open Data",
        "repository": manifest["repository"],
        "revision": manifest["revision"],
        "scope": {
            "competition_id": competition_id,
            "season_id": season_id,
            "maximum_matches": maximum_matches,
        },
        "matches_analyzed": len(summaries),
        "events": sum(row["events"] for row in summaries),
        "shots": len(shots),
        "players": list(players.values()),
        "teams": list(teams.values()),
        "matches": summaries,
        "spatial_context": spatial_summaries,
        "expected_threat": xt,
        "shot_models": xg,
        "errors": errors,
        "quality_exclusions": quality_exclusions,
        "unreferenced_event_files": sorted(
            path
            for path in index
            if path.startswith("data/events/") and int(Path(path).stem) not in metadata
        ),
        "notes": [
            "Player rates use covered lineup intervals only; unknown minutes remain null.",
            "Compare within competition-season, position and adequate exposure; archive selection is uneven.",
            "xT is fitted retrospectively on the analyzed archive, not an independent predictive validation.",
            "360 observations are event snapshots and cannot measure running velocity.",
        ],
    }
    _atomic_write_json(output_dir / "report.json", result)
    player_rows = list(players.values())
    with (output_dir / "players.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(player_rows[0]) if player_rows else ["player_id"]
        )
        writer.writeheader()
        writer.writerows(player_rows)
    _atomic_write_json(
        output_dir / "shots.json", {"source": "StatsBomb", "shots": shots}
    )
    return result
