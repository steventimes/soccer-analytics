"""Run player/team assessment models against the existing self-contained database."""

from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import sqlite3
from datetime import datetime, timezone
from .assessment_data import (
    load_players,
    player_panels,
    fit_priors,
    player_examples,
    load_team_matches,
    team_examples,
    team_rows,
    rolling_team_features,
)
from .assessment_models import METHODS, benchmark, heldout_player, clip_prediction
from .assessment_profiles import season_profiles, profile_scores, styles
from .team_assessment import poisson_ratings, tactical_profiles, cluster_styles

logger = logging.getLogger(__name__)
PLAYER_TARGETS = (
    "goals_per90",
    "progressive_passes_per90",
    "pass_completion",
    "duel_win_rate",
    "keeper_success_rate",
)
TEAM_TARGETS = ("goals_for", "goals_against", "points")


def attach_game(connection, examples):
    cache = {}
    for pid, aid, row_number in connection.execute(
        "SELECT real_player_id,game_asset_id,game_row_number FROM game_real_links WHERE real_provider='wyscout' AND game_provider='fifa'"
    ):
        p = connection.execute(
            "SELECT effective_date FROM game_profiles WHERE asset_id=? AND row_number=?",
            (aid, row_number),
        ).fetchone()
        table = connection.execute(
            "SELECT kind FROM imports WHERE asset_id=? AND kind LIKE '%players'", (aid,)
        ).fetchone()
        if not table:
            continue
        record = connection.execute(
            'SELECT * FROM "'
            + table[0].replace('"', '""')
            + '" WHERE asset_id=? AND row_number=?',
            (aid, row_number),
        ).fetchone()
        if not record:
            continue
        values = {}
        for name in [
            "overall",
            "movement_sprint_speed",
            "power_stamina",
            "attacking_short_passing",
            "mentality_vision",
            "attacking_finishing",
            "defending_standing_tackle",
        ]:
            try:
                values["game_" + name] = (
                    float(record[name]) if record[name] is not None else None
                )
            except (ValueError, KeyError, IndexError):
                values["game_" + name] = None
        cache[pid] = (p[0], values)
    for r in examples:
        value = cache.get(r["entity_id"])
        r["has_game_data"] = bool(
            value and value[0] and value[0][:10] < r["feature_asof"]
        )
        r["game_features"] = value[1] if r["has_game_data"] else {}
    return sum(r["has_game_data"] for r in examples)


def save_results(
    database, output, report, profiles, team_ratings, team_styles, examples
):
    from app.data_service.warehouse import Warehouse
    from .artifacts import archive_outputs

    run_id = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()[
        :16
    ]
    with Warehouse(database) as db:
        db.connection.executescript("""
        CREATE TABLE IF NOT EXISTS assessment_runs(run_id TEXT PRIMARY KEY,created_at TEXT,protocol TEXT,report TEXT);
        CREATE TABLE IF NOT EXISTS player_assessments(run_id TEXT REFERENCES assessment_runs(run_id),provider TEXT,player_id TEXT,league TEXT,as_of TEXT,payload TEXT,PRIMARY KEY(run_id,provider,player_id,league));
        CREATE TABLE IF NOT EXISTS team_assessments(run_id TEXT REFERENCES assessment_runs(run_id),provider TEXT,team_id TEXT,league TEXT,as_of TEXT,payload TEXT,PRIMARY KEY(run_id,provider,team_id,league));
        CREATE TABLE IF NOT EXISTS assessment_experiments(run_id TEXT REFERENCES assessment_runs(run_id),experiment TEXT,payload TEXT,PRIMARY KEY(run_id,experiment));
        CREATE TABLE IF NOT EXISTS assessment_examples(run_id TEXT REFERENCES assessment_runs(run_id),kind TEXT,row_number INTEGER,payload TEXT,PRIMARY KEY(run_id,kind,row_number));
        """)
        # UPSERT preserves references when repeating the same versioned run.
        db.connection.execute(
            "INSERT INTO assessment_runs VALUES(?,?,?,?) ON CONFLICT(run_id) DO UPDATE SET created_at=excluded.created_at,protocol=excluded.protocol,report=excluded.report",
            (
                run_id,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(report["protocol"]),
                json.dumps(report),
            ),
        )
        for row in profiles:
            db.connection.execute(
                "INSERT OR REPLACE INTO player_assessments VALUES(?,?,?,?,?,?)",
                (
                    run_id,
                    "wyscout",
                    row["entity_id"],
                    row["league"],
                    row["as_of"],
                    json.dumps(row),
                ),
            )
        for row in team_ratings:
            db.connection.execute(
                "INSERT OR REPLACE INTO team_assessments VALUES(?,?,?,?,?,?)",
                (
                    run_id,
                    "football-csv",
                    row["entity_id"],
                    row["league"],
                    row["as_of"],
                    json.dumps(row),
                ),
            )
        for row in team_styles:
            db.connection.execute(
                "INSERT OR REPLACE INTO team_assessments VALUES(?,?,?,?,?,?)",
                (
                    run_id,
                    "wyscout",
                    row["entity_id"],
                    row["league"],
                    "2018-05-31",
                    json.dumps(row),
                ),
            )
        for name, result in report["benchmarks"].items():
            db.connection.execute(
                "INSERT OR REPLACE INTO assessment_experiments VALUES(?,?,?)",
                (run_id, name, json.dumps(result)),
            )
        for kind, rows in examples.items():
            db.connection.executemany(
                "INSERT OR REPLACE INTO assessment_examples VALUES(?,?,?,?)",
                [(run_id, kind, i, json.dumps(row)) for i, row in enumerate(rows, 1)],
            )
        archive_outputs(
            db, "assessment_artifacts", output, "assessment_artifact", run_id
        )
        db.connection.commit()
        db.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return run_id


def run(
    database="data/football.sqlite3",
    output=".tmp/assessments",
    methods=METHODS,
    train_cap=5000,
):
    from .artifacts import cached_json, source_signature, file_digest

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    player_inputs = {
        "sources": source_signature(connection, ("wyscout",)),
        "extractor_sha256": file_digest(Path(__file__).with_name("assessment_data.py")),
    }
    team_inputs = {
        "sources": source_signature(connection, ("football-csv",)),
        "as_of": "2026-10-09",
        "extractor_sha256": player_inputs["extractor_sha256"],
    }

    def build_player_data():
        people, matches, obs, teamstyles, audit = load_players(connection)
        return {
            "people": people,
            "matches": matches,
            "observations": obs,
            "panels": player_panels(obs, people),
            "team_styles": [
                {"match_id": str(k[0]), "team_id": k[1], **v}
                for k, v in teamstyles.items()
            ],
            "audit": audit,
        }

    player = cached_json(output / "player-data.json", player_inputs, build_player_data)
    if not any(p["position"] == "Goalkeeper" for p in player["panels"]):
        raise ValueError(
            "Player cache predates goalkeeper exposure reconstruction; rebuild it"
        )
    panels = player["panels"]
    priors = fit_priors([p for p in panels if p["block"] == 0])
    profiles = profile_scores(season_profiles(panels), priors)
    player_style = styles(profiles, output / "player-styles.json")
    (output / "player-profiles.json").write_text(json.dumps(profiles, indent=2))

    def build_team_data():
        matches, audit = load_team_matches(connection, team_inputs["as_of"])
        return {"matches": matches, "examples": team_examples(matches), "audit": audit}

    team = cached_json(output / "team-data.json", team_inputs, build_team_data)
    bench = {}
    all_examples = {}
    models = {}
    for mode in ["temporal", "unseen_players"]:
        training_panels = [
            p
            for p in panels
            if p["block"] == 0
            and (mode != "unseen_players" or not heldout_player(p["entity_id"]))
        ]
        own_priors = fit_priors(training_panels)
        pairs = player_examples(panels, own_priors)
        attach_game(connection, pairs)
        all_examples["player_" + mode] = pairs
        logger.info("Starting player benchmark %s (%d pairs)", mode, len(pairs))
        result, _ = benchmark(
            pairs,
            PLAYER_TARGETS,
            kind="player",
            mode=mode,
            output=output / "models",
            methods=methods,
            train_cap=train_cap,
        )
        bench["player_" + mode] = result
    pairs = all_examples["player_temporal"]
    for with_game in [False, True]:
        label = "player_game_ablation_" + ("game" if with_game else "real")
        result, _ = benchmark(
            pairs,
            PLAYER_TARGETS,
            kind="player",
            mode="temporal",
            game_features=with_game,
            matched_only=True,
            output=output / "game-models",
            methods=("ridge", "hist_gradient_boosting"),
            train_cap=train_cap,
        )
        bench[label] = result
    all_examples["team"] = team["examples"]
    logger.info("Starting team benchmark %d windows", len(team["examples"]))
    bench["team_temporal"], models = benchmark(
        team["examples"],
        TEAM_TARGETS,
        kind="team",
        output=output / "models",
        methods=methods,
        train_cap=train_cap,
    )
    poisson = poisson_ratings(team["matches"], output / "models")
    (output / "team-poisson.json").write_text(json.dumps(poisson, indent=2))
    team_names = {
        str(row["entity_id"]): json.loads(row["payload"])["name"]
        for row in connection.execute(
            "SELECT entity_id,payload FROM wyscout_entities WHERE entity_type='teams'"
        )
    }
    tactics = cluster_styles(
        tactical_profiles(player["matches"], player["team_styles"], team_names),
        output / "team-styles.joblib",
    )
    (output / "team-styles.json").write_text(json.dumps(tactics, indent=2))
    team_ratings = poisson["ratings"]
    # Show frozen-model forecasts only for each league's most recent covered season.
    current = {}
    for key, rows in team_rows(team["matches"]).items():
        if len(rows) >= 8:
            snapshot = {
                "entity_id": key[2],
                "league": key[0],
                "season": key[1],
                "name": rows[-1]["name"],
                "as_of": rows[-1]["date"],
                "features": rolling_team_features(rows[-8:]),
            }
            identity = (key[0], key[2])
            if (
                identity not in current
                or snapshot["as_of"] > current[identity]["as_of"]
            ):
                current[identity] = snapshot
    import numpy as np

    league_latest = {
        league: max(r["season"] for r in team["matches"] if r["competition"] == league)
        for league in {r["competition"] for r in team["matches"]}
    }
    forecasts = []
    for row in current.values():
        if row["season"] != league_latest[row["league"]]:
            continue
        prediction = {}
        for target, bundle in models.items():
            x = np.array(
                [[row["features"].get(c) for c in bundle["columns"]]], dtype=object
            )
            prediction[target] = float(
                clip_prediction(bundle["model"].predict(x), target)[0]
            )
        forecasts.append(
            {
                **row,
                "next_five_match_mean_forecast": prediction,
                "model_scope": "Frozen model fitted through validation season 2024; extrapolation to partial 2026 season has no future-label validation.",
            }
        )
    (output / "team-forecasts.json").write_text(json.dumps(forecasts, indent=2))
    protocol = {
        "player_scope": "2017/18 five European leagues; full-season profiles and next-window observations",
        "player_windows": [
            "through 2017-10-31",
            "Nov-Dec2017",
            "Jan-Feb2018",
            "Mar-May2018",
        ],
        "player_training": "window0 ->1",
        "validation": "window1 ->2",
        "test": "window2 ->3",
        "unseen_players": "fixed SHA256 20% player holdout excluded from training, validation and empirical priors",
        "target_exposure": "Both adjacent player windows require 270 minutes; future ratios require sufficient attempts. Conditional performance, not availability prediction.",
        "goalkeeper_exposure": "Nominal 90-minute lineup/substitution/red-card reconstruction; unrecorded stoppage time is excluded. Other players use supplied PlayeRank minutes, but no PlayeRank score enters targets.",
        "team_training": "2008-2023 seasons, labels before2024July1",
        "team_validation": "2024season labels before2025July1",
        "team_test": "2025season labels before2026July1",
        "team_windows": "8 prior matches ->5 future matches; five-match stride, same league-season, strict date separation",
        "train_cap": train_cap,
        "methods": list(methods),
        "selection": "Validation only; each algorithm uses two predeclared parameter settings; all tested methods reported",
        "empirical_priors": "Player league-role priors use window0 only; unseen-player priors exclude held-out identities.",
        "rating": "Descriptive position/league percentiles with explicitly subjective role weights, no absolute ability calibration",
        "style": "KMeans, diagonal Gaussian mixture, PCA and NMF; no good/bad ordering",
        "limits": [
            "Only one complete multi-league player-event season is covered.",
            "Roster position metadata has no per-match historical timestamp; evaluations are retrospective and not proven prospective.",
            "Conditional beta/gamma intervals assume exchangeable attempts/counts and do not capture all opponent or measurement error.",
            "Goalkeeper success rates lack shot-difficulty adjustment.",
            "Historical profiles and neighbour suggestions are not current transfer recommendations.",
        ],
        "runtime": {
            p: importlib.metadata.version(p)
            for p in ["scikit-learn", "xgboost-cpu", "numpy", "scipy"]
        },
        "inputs": {
            "player": player_inputs,
            "team": team_inputs,
            "game": source_signature(connection, ("game-ratings",)),
        },
    }
    report = {
        "protocol": protocol,
        "player_observations": len(player["observations"]),
        "player_profiles": len(profiles),
        "goalkeeper_profiles": sum(p["position"] == "Goalkeeper" for p in profiles),
        "player_data_audit": player["audit"],
        "team_matches": len(team["matches"]),
        "team_windows": len(team["examples"]),
        "team_ratings": len(team_ratings),
        "team_style_profiles": len(tactics["profiles"]),
        "benchmarks": bench,
        "player_styles": player_style,
        "team_poisson": poisson,
        "team_style_selection": {k: v for k, v in tactics.items() if k != "profiles"},
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    connection.close()
    run_id = save_results(
        database,
        output,
        report,
        profiles,
        team_ratings,
        tactics["profiles"],
        all_examples,
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "report": str(output / "report.json"),
                "player_profiles": len(profiles),
                "goalkeepers": report["goalkeeper_profiles"],
                "team_ratings": len(team_ratings),
                "failures": {k: len(v["failures"]) for k, v in bench.items()},
            },
            indent=2,
        )
    )
    return report


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="data/football.sqlite3")
    parser.add_argument("--output", default=".tmp/assessments")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--train-cap", type=int, default=5000)
    args = parser.parse_args()
    selected = args.methods.split(",")
    if not selected or any(s not in METHODS for s in selected):
        parser.error("Unknown model name")
    run(args.database, args.output, tuple(selected), args.train_cap)
