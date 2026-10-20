"""Exploratory findings from verified SoccerData assets in the local warehouse.

Keep publisher placeholders and SDK information loss out of the analytical sample.
Player repeatability is conditional on returning with enough minutes in the same
league; whole-match possession associations do not establish tactical causality.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile

from app.data_service.database import export_asset
from .artifacts import archive_outputs, file_digest

ALIASES = {
    "AFC Bournemouth": "Bournemouth",
    "Brighton & Hove Albion": "Brighton",
    "Ipswich Town": "Ipswich",
    "Leicester City": "Leicester",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
}
SITUATIONS = {
    "OpenPlay": "Open Play",
    "FromCorner": "From Corner",
    "SetPiece": "Set Piece",
    "DirectFreekick": "Direct Freekick",
    "Penalty": "Penalty",
}


def number(value):
    return float(value) if value not in (None, "") else None


def possession_eligible(values):
    return (
        len(values) == 2
        and all(v is not None and 0 <= v <= 100 for v in values)
        and abs(sum(values) - 100) <= 0.3
    )


def consecutive_pairs(rows, minimum_minutes=900):
    panel = {}
    for row in rows:
        key = (row["league"], int(row["season_id"]), row["player_id"])
        if key in panel:
            raise ValueError("Duplicate league/player/season source identity")
        panel[key] = row
    return [
        (row, panel[(league, year + 1, pid)])
        for (league, year, pid), row in sorted(panel.items())
        if (league, year + 1, pid) in panel
        and min(float(row["minutes"]), float(panel[(league, year + 1, pid)]["minutes"]))
        >= minimum_minutes
    ]


def restore_situations(shots, raw):
    result, recovered = [], 0
    for shot in shots:
        expected = SITUATIONS.get(raw.get(shot["shot_id"]))
        if expected is None or (shot["situation"] and shot["situation"] != expected):
            raise ValueError("Shot situation cannot be reconciled by source shot ID")
        recovered += not bool(shot["situation"])
        result.append({**shot, "situation": expected})
    return result, recovered


def load_frame(database, connection, run, method, receipt):
    row = connection.execute(
        """SELECT s.asset_id,s.rows,a.sha256,a.path FROM soccerdata_imports s
        JOIN assets a ON a.id=s.asset_id WHERE s.run_directory=? AND s.method=?
        ORDER BY s.rowid DESC LIMIT 1""",
        (run, method),
    ).fetchone()
    if row is None:
        raise ValueError(f"Missing source frame: {run}/{method}")
    aid, expected_rows, digest, path = row
    with tempfile.TemporaryDirectory() as directory:
        exported = Path(directory) / "frame.csv"
        export_asset(database, aid, exported)
        with exported.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            headers = {h: h.split("_", 2)[2] for h in reader.fieldnames}
            if len(set(headers.values())) != len(headers):
                raise ValueError("Ambiguous analytical column names")
            rows = [{headers[k]: v for k, v in row.items()} for row in reader]
    if len(rows) != expected_rows:
        raise ValueError("Frame receipt does not match exported population")
    receipt[aid] = {"asset_id": aid, "sha256": digest, "path": path, "rows": len(rows)}
    return rows


def cached_documents(connection, provider, prefix, receipt):
    latest = {}
    for row in connection.execute(
        """SELECT a.id,a.path,a.sha256,d.payload FROM assets a
        JOIN datasets s ON s.id=a.dataset_id JOIN documents d ON d.asset_id=a.id
        WHERE s.name=? AND a.path LIKE ? AND a.complete=1 ORDER BY a.id""",
        ("soccerdata:" + provider, prefix + "%"),
    ):
        latest[row[1].split("@")[0]] = row
    for aid, path, digest, payload in latest.values():
        # These SDK caches are UTF-8 JSON; validate the stored document against
        # the original asset receipt before using raw labels or score metadata.
        if hashlib.sha256(payload.encode()).hexdigest() != digest:
            raise ValueError("Cached JSON document does not match its source hash")
        receipt[aid] = {"asset_id": aid, "sha256": digest, "path": path}
        yield json.loads(payload)


def team_panel(stats):
    rows = []
    for match in stats:
        for side, other in [("home", "away"), ("away", "home")]:
            rows.append(
                {
                    "league": match["league"],
                    "year": int(match["season_id"]),
                    "game_id": match["game_id"],
                    "game": match["game"],
                    "date": match["date"][:10],
                    "side": side,
                    "team": match[side + "_team"],
                    "opponent": match[other + "_team"],
                    "home_team": match["home_team"],
                    "away_team": match["away_team"],
                    "xg": float(match[side + "_xg"]),
                    "xga": float(match[other + "_xg"]),
                    "npxg": float(match[side + "_np_xg"]),
                    "goals": float(match[side + "_goals"]),
                    "against": float(match[other + "_goals"]),
                    "ppda": number(match[side + "_ppda"]),
                    "deep": float(match[side + "_deep_completions"]),
                }
            )
    return rows


def correlation(x, y, clusters, draws=1000):
    import numpy as np

    x, y = np.asarray(x, float), np.asarray(y, float)
    labels = sorted(set(clusters))
    ids = {label: i for i, label in enumerate(labels)}
    sufficient = np.zeros((len(labels), 6))
    np.add.at(
        sufficient,
        [ids[c] for c in clusters],
        np.column_stack((np.ones(len(x)), x, y, x * y, x * x, y * y)),
    )

    def estimate(s):
        n, sx, sy, sxy, sxx, syy = s
        denominator = (n * sxx - sx * sx) * (n * syy - sy * sy)
        return (n * sxy - sx * sy) / np.sqrt(denominator) if denominator > 0 else None

    rng = np.random.default_rng(20261009)
    values = [
        estimate(sufficient[rng.integers(len(labels), size=len(labels))].sum(axis=0))
        for _ in range(draws)
    ]
    values = [v for v in values if v is not None]
    return {
        "r": estimate(sufficient.sum(axis=0)),
        "cluster_bootstrap_ci95": (
            np.quantile(values, [0.025, 0.975]).tolist() if values else None
        ),
        "rows": len(x),
        "clusters": len(labels),
        "bootstrap_draws": draws,
    }


def rate(row, metric):
    value = (
        (float(row["np_goals"]) - float(row["np_xg"]))
        if metric == "finishing_residual"
        else float(row[metric])
    )
    return value * 90 / float(row["minutes"])


def repeatability(players):
    pairs = consecutive_pairs(players)
    metrics = ["np_goals", "np_xg", "xa", "shots", "key_passes", "finishing_residual"]
    groups = {
        "minimum900": pairs,
        "minimum1800": consecutive_pairs(players, 1800),
        "forwards900": [
            (a, b)
            for a, b in pairs
            if "F" in a["position"].split() and "F" in b["position"].split()
        ],
    }
    result = {}
    for label, selected in groups.items():
        result[label] = {
            m: correlation(
                [rate(a, m) for a, b in selected],
                [rate(b, m) for a, b in selected],
                [a["player_id"] for a, b in selected],
            )
            for m in metrics
        }
    return result, pairs


def temporal_models(pairs):
    import numpy as np

    subsets = {
        "train": [(a, b) for a, b in pairs if int(a["season_id"]) <= 2021],
        "validation": [(a, b) for a, b in pairs if int(a["season_id"]) == 2022],
        "test": [(a, b) for a, b in pairs if int(a["season_id"]) >= 2023],
    }
    models, losses = {}, {}
    for label, fields in {
        "goals": ["np_goals"],
        "opportunities": ["np_xg"],
        "opportunities_and_finishing": ["np_xg", "finishing_residual"],
    }.items():
        train = subsets["train"]
        matrix = np.array([[1] + [rate(a, f) for f in fields] for a, b in train])
        target = np.array([rate(b, "np_goals") for a, b in train])
        weights = np.linalg.lstsq(matrix, target, rcond=None)[0]
        scores = {}
        for split in ["validation", "test"]:
            selected = subsets[split]
            prediction = np.maximum(
                0,
                np.array([[1] + [rate(a, f) for f in fields] for a, b in selected])
                @ weights,
            )
            error = abs(
                prediction - np.array([rate(b, "np_goals") for a, b in selected])
            )
            scores[split + "_mae"] = float(error.mean())
            if split == "test":
                losses[label] = error
        models[label] = {"features": fields, "coefficients": weights.tolist(), **scores}
    test = subsets["test"]
    baseline = np.array(
        [abs(rate(a, "np_goals") - rate(b, "np_goals")) for a, b in test]
    )
    labels = sorted({a["player_id"] for a, b in test})
    contributions = defaultdict(list)
    for i, (a, b) in enumerate(test):
        contributions[a["player_id"]].append(i)
    selected_model = min(models, key=lambda name: models[name]["validation_mae"])
    diffs = {}
    rng = np.random.default_rng(20261009)
    for a, b in [
        ("opportunities", "goals"),
        ("opportunities_and_finishing", "opportunities"),
    ]:
        delta = losses[a] - losses[b]
        sums = np.array([delta[contributions[label]].sum() for label in labels])
        counts = np.array([len(contributions[label]) for label in labels])
        estimates = []
        for _ in range(1000):
            ids = rng.integers(len(labels), size=len(labels))
            estimates.append(sums[ids].sum() / counts[ids].sum())
        diffs[a + "_minus_" + b] = {
            "mae_delta": float(delta.mean()),
            "player_cluster_ci95": np.quantile(estimates, [0.025, 0.975]).tolist(),
        }
    return {
        "target": "next-season non-penalty goals per90",
        "sample": {k: len(v) for k, v in subsets.items()},
        "train_input_seasons": "2014/15-2021/22",
        "validation_input_season": "2022/23",
        "test_target_seasons": ["2024/25", "2025/26"],
        "models": models,
        "validation_selected_model": selected_model,
        "carry_forward_goals_test_mae": float(baseline.mean()),
        "paired_model_comparisons": diffs,
        "scope": "Exploratory fixed linear models, no future xG as an input; same-league returners with >=900 minutes in both years. Not a youth-potential model.",
    }


def possession_analysis(teamrows, schedules, espn, shots, summaries):
    import numpy as np

    canon = lambda name: ALIASES.get(name, name)
    by_fixture = {
        (r["date"], r["home_team"], r["away_team"], r["side"]): r
        for r in teamrows
        if r["league"] == "ENG-Premier League" and r["year"] == 2024
    }
    schedules = {r["game"]: r for r in schedules}
    by_shots = defaultdict(list)
    for shot in shots:
        by_shots[(shot["game_id"], shot["team"])].append(shot)
    original_scores = {}
    placeholder_games = set()
    for summary in summaries:
        competition = summary["header"]["competitions"][0]
        gid = summary["header"]["id"]
        original_scores[gid] = {
            r["homeAway"]: float(r["score"]) for r in competition["competitors"]
        }
        values = [
            [
                number(s.get("displayValue"))
                for s in team.get("statistics", [])
                if s["name"] == "possessionPct"
            ]
            for team in summary["boxscore"]["teams"]
        ]
        if len(values) == 2 and values == [[0.0], [0.0]]:
            placeholder_games.add(gid)
    joined, exclusions, seen, discrepancies = [], [], set(), 0
    xg_differences = []
    for row in espn:
        schedule = schedules[row["game"]]
        side = "home" if row["is_home"] == "True" else "away"
        key = (
            schedule["date"][:10],
            canon(schedule["home_team"]),
            canon(schedule["away_team"]),
            side,
        )
        match = by_fixture.get(key)
        if match is None or key in seen or match["team"] != canon(row["team"]):
            raise ValueError(
                "Ambiguous or unmatched ESPN/Understat fixture/team identity"
            )
        seen.add(key)
        if original_scores[schedule["game_id"]][side] != match["goals"]:
            raise ValueError("ESPN and Understat final scores disagree")
        events = [
            e
            for e in by_shots[(match["game_id"], match["team"])]
            if e["result"] != "Own Goal"
        ]
        shot_total = sum(float(e["xg"]) for e in events)
        xg_differences.append(shot_total - match["xg"])
        np_events = [e for e in events if e["situation"] != "Penalty"]
        np_total = sum(float(e["xg"]) for e in np_events)
        discrepancies += number(row["total_shots"]) != len(events)
        joined.append(
            {
                **match,
                "team_reported_xg": match["xg"],
                "shot_xg": shot_total,
                "np_shot_xg": np_total,
                "espn_game_id": schedule["game_id"],
                "possession": number(row["possession_pct"]),
                "shots": len(events),
                "np_shots": len(np_events),
                "np_quality": np_total / len(np_events) if np_events else None,
            }
        )
    fixtures = defaultdict(list)
    for row in joined:
        fixtures[row["game_id"]].append(row)
    for gid, rows in fixtures.items():
        if not possession_eligible([r["possession"] for r in rows]):
            exclusions.append(
                {
                    "understat_game_id": gid,
                    "game": rows[0]["game"],
                    "reason": "invalid paired possession total",
                    "values": [r["possession"] for r in rows],
                }
            )
    excluded = {r["understat_game_id"] for r in exclusions}
    valid = [r for r in joined if r["game_id"] not in excluded]
    if len(joined) != 760 or len(fixtures) != 380:
        raise ValueError(
            "Expected complete 2024/25 English league fixture reconciliation"
        )
    metrics = {}
    for field in ["shots", "shot_xg", "np_shot_xg", "np_quality"]:
        selected = [r for r in valid if r[field] is not None]
        metrics[field] = correlation(
            [r["possession"] for r in selected],
            [r[field] for r in selected],
            [r["game_id"] for r in selected],
        )
    clubs = []
    for team in sorted({r["team"] for r in valid}):
        selected = [r for r in valid if r["team"] == team]
        clubs.append(
            {
                "team": team,
                "matches": len(selected),
                "possession_pct": float(np.mean([r["possession"] for r in selected])),
                "shot_xg_per_match": float(np.mean([r["shot_xg"] for r in selected])),
                "np_xg_per_shot": sum(r["np_shot_xg"] for r in selected)
                / sum(r["np_shots"] for r in selected),
            }
        )
    return {
        "matched_fixtures": len(fixtures),
        "score_conflicts": 0,
        "excluded_possession_fixtures": exclusions,
        "raw_espn_both_zero_fixtures": len(placeholder_games),
        "eligible_fixtures": len(valid) // 2,
        "shot_count_disagreement_team_rows": int(discrepancies),
        "shot_sum_vs_reported_team_xg": {
            "team_rows_differing_more_than_0_001": sum(
                abs(d) > 0.001 for d in xg_differences
            ),
            "mean_shot_sum_minus_team_reported": float(np.mean(xg_differences)),
            "maximum_absolute_difference": max(abs(d) for d in xg_differences),
            "note": "Both measures are preserved. Publisher aggregation corrections are unverified; shot quality uses the sum of the same shot values as its denominator, excluding own goals and penalties.",
        },
        "correlations": metrics,
        "team_summaries": clubs,
        "scope": "Whole-match descriptive associations; game state, opponent approach, red cards and selection can affect possession. Chance volume uses attacking shots; quality uses non-penalty shot-level xG divided by those same shots. Understat team-reported totals remain separate.",
    }


def team_changes(teamrows):
    import numpy as np

    output = []
    for team in ["Manchester City", "Arsenal", "Liverpool"]:
        periods = [
            [
                r
                for r in teamrows
                if r["team"] == team
                and r["year"] == year
                and r["league"] == "ENG-Premier League"
            ]
            for year in [2023, 2024]
        ]
        summaries = [
            {
                "season": season,
                "matches": len(rows),
                "xg_per_match": float(np.mean([r["xg"] for r in rows])),
                "xga_per_match": float(np.mean([r["xga"] for r in rows])),
            }
            for rows, season in zip(periods, ["2023/24", "2024/25"])
        ]
        for summary in summaries:
            summary["xgd_per_match"] = (
                summary["xg_per_match"] - summary["xga_per_match"]
            )
        maps = [{(r["opponent"], r["side"]): r for r in rows} for rows in periods]
        keys = sorted(maps[0].keys() & maps[1].keys())
        labels = sorted({k[0] for k in keys})
        values = {
            op: [
                maps[1][k]["xg"]
                - maps[1][k]["xga"]
                - maps[0][k]["xg"]
                + maps[0][k]["xga"]
                for k in keys
                if k[0] == op
            ]
            for op in labels
        }
        rng = np.random.default_rng(20261009)
        boot = [
            np.mean(
                [
                    v
                    for i in rng.integers(len(labels), size=len(labels))
                    for v in values[labels[i]]
                ]
            )
            for _ in range(1000)
        ]
        output.append(
            {
                "team": team,
                "seasons": summaries,
                "attack_xg_change": summaries[1]["xg_per_match"]
                - summaries[0]["xg_per_match"],
                "defensive_xga_change": summaries[1]["xga_per_match"]
                - summaries[0]["xga_per_match"],
                "matched_opponent_home_away": {
                    "matches_per_season": len(keys),
                    "opponents": len(labels),
                    "xgd_change": float(
                        np.mean([v for group in values.values() for v in group])
                    ),
                    "opponent_cluster_ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
                },
            }
        )
    return output


def run(database="data/football.sqlite3", output=".tmp/new-source-analysis"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    receipt = {}
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as c:
        players = load_frame(
            database, c, "Understat-history", "player_season_stats", receipt
        )
        stats = load_frame(
            database, c, "Understat-history", "team_match_stats", receipt
        )
        shots = load_frame(database, c, "Understat-details", "shot_events", receipt)
        schedules = load_frame(database, c, "ESPN", "schedule", receipt)
        espn = load_frame(database, c, "ESPN", "matchsheets", receipt)
        raw_shots = {}
        for doc in cached_documents(
            c, "Understat", "Understat-details/cache/match_", receipt
        ):
            for events in doc["shots"].values():
                for event in events:
                    if event["id"] in raw_shots:
                        raise ValueError("Duplicate original shot ID")
                    raw_shots[event["id"]] = event["situation"]
        restored, recovered = restore_situations(shots, raw_shots)
        multi_team = sum(
            "," in p["team_title"]
            for doc in cached_documents(
                c, "Understat", "Understat-history/cache/league_%_season_", receipt
            )
            for p in doc["players"]
        )
        summaries = list(cached_documents(c, "ESPN", "ESPN/cache/Summary_", receipt))
    panel = team_panel(stats)
    persistence, pairs = repeatability(players)
    report = {
        "snapshot_date": "2026-10-09",
        "analysis_type": "exploratory; observational; no causal claims",
        "repeatability": persistence,
        "temporal_prediction": temporal_models(pairs),
        "possession": possession_analysis(panel, schedules, espn, restored, summaries),
        "team_changes": team_changes(panel),
        "data_quality": {
            "recovered_penalty_labels": recovered,
            "multi_team_player_season_rows": multi_team,
            "player_season_grain": "league/player/source-season totals, potentially multiple clubs; SDK team label is first-listed club and must not receive all those totals",
        },
        "input_receipt": {
            "assets": len(receipt),
            "frames": [r for r in receipt.values() if "rows" in r],
            "asset_set_sha256": hashlib.sha256(
                json.dumps(
                    sorted(receipt.values(), key=lambda r: r["asset_id"]),
                    sort_keys=True,
                ).encode()
            ).hexdigest(),
            "extractor_sha256": file_digest(__file__),
        },
        "limits": [
            "Player samples require sufficient minutes in both years and omit league switchers and disappearing players.",
            "Correlation can reflect role persistence; forward-only and higher-minute sensitivity checks are included.",
            "Possession is a whole-match outcome and cannot show that retaining possession causes shot quality to change.",
            "Team xG changes do not identify coaching, injury, fitness or tactical causes.",
            "Understat model revisions are unknown; estimates are publisher xG, not directly measured chance probabilities.",
        ],
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    from app.data_service.warehouse import Warehouse

    run_id = file_digest(report_path)[:16]
    with Warehouse(database) as db:
        archive_outputs(
            db, "new_source_findings", output, "new_source_findings", run_id
        )
        db.record_analysis_inputs("new_source_findings", run_id, receipt)
        db.connection.commit()
        db.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    print(
        json.dumps(
            {
                "run_id": run_id,
                "player_pairs": len(pairs),
                "eligible_possession_fixtures": report["possession"][
                    "eligible_fixtures"
                ],
                "data_quality": report["data_quality"],
            },
            indent=2,
        )
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="data/football.sqlite3")
    parser.add_argument("--output", default=".tmp/new-source-analysis")
    args = parser.parse_args()
    run(args.database, args.output)
