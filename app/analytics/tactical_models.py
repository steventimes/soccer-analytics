"""Whole-match temporal chance models and descriptive opponent dossiers."""

import argparse
from collections import defaultdict, Counter
import json
import logging
from pathlib import Path
import warnings

logger = logging.getLogger(__name__)


def model(name):
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import (
        RandomForestClassifier,
        ExtraTreesClassifier,
        HistGradientBoostingClassifier,
    )
    from sklearn.neural_network import MLPClassifier

    if name == "logistic":
        return LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    if name == "sgd":
        return SGDClassifier(
            loss="log_loss", max_iter=2000, alpha=0.001, random_state=42
        )
    if name == "naive_bayes":
        return GaussianNB(var_smoothing=1e-6)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=10,
            n_jobs=1,
            random_state=42,
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=100, min_samples_leaf=10, n_jobs=1, random_state=42
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=120,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=42,
        )
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.05,
            n_jobs=1,
            random_state=42,
            eval_metric="logloss",
        )
    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(32,),
            max_iter=250,
            random_state=42,
            alpha=0.01,
            early_stopping=False,
        )
    raise ValueError(name)


def fit_chances(prefixes, output, train_cap=15000, methods=None):
    import numpy as np
    import joblib
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
    from threadpoolctl import threadpool_limits

    rows = [r for r in prefixes if r["date"] >= "2015-01-01"]
    days = sorted({r["date"] for r in rows})
    valid_start = days[int(len(days) * 0.6)]
    test_start = days[int(len(days) * 0.8)]
    train = [r for r in rows if r["date"] < valid_start]
    valid = [r for r in rows if valid_start <= r["date"] < test_start]
    test = [r for r in rows if r["date"] >= test_start]
    if len(train) > train_cap:
        indices = np.random.default_rng(42).choice(len(train), train_cap, replace=False)
        train = [train[i] for i in sorted(indices)]
    y = np.array([r["shot_created"] for r in train])
    yv = np.array([r["shot_created"] for r in valid])
    yt = np.array([r["shot_created"] for r in test])
    results = []
    failures = []
    objects = {}
    test_predictions = {}

    def metrics(actual, pred):
        return {
            "log_loss": float(log_loss(actual, pred, labels=[0, 1])),
            "brier": float(brier_score_loss(actual, pred)),
            "auc": float(roc_auc_score(actual, pred)) if len(set(actual)) > 1 else None,
        }

    constant = np.full(len(test), np.mean(y))
    baseline_predictions = None
    selected_methods = methods or [
        "logistic",
        "sgd",
        "naive_bayes",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
        "xgboost",
        "mlp",
    ]
    for name in ["context_baseline", *selected_methods]:
        try:
            source = "baseline_features" if name == "context_baseline" else "features"
            features = dict(train[0][source])
            removals = {
                "without_formation": ["formation", "opponent_formation"],
                "without_channels": [
                    "prefix_low_y_fraction",
                    "prefix_central_fraction",
                    "prefix_width_units",
                ],
                "without_opponent": [
                    "opponent_prior_matches",
                    "opponent_prior_pressure_proxy",
                    "opponent_prior_possession_proxy",
                    "opponent_formation",
                    "pre_match_archive_elo_difference",
                ],
            }
            if name == "formation_only":
                features = {
                    **train[0]["baseline_features"],
                    "formation": features["formation"],
                    "opponent_formation": features["opponent_formation"],
                }
            for key in removals.get(name, []):
                features.pop(key, None)
            numeric = [k for k, v in features.items() if not isinstance(v, str)]
            category = [k for k, v in features.items() if isinstance(v, str)]
            columns = numeric + category
            transform = ColumnTransformer(
                [
                    (
                        "numeric",
                        make_pipeline(
                            SimpleImputer(strategy="median", add_indicator=True),
                            StandardScaler(),
                        ),
                        list(range(len(numeric))),
                    ),
                    (
                        "category",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        list(range(len(numeric), len(columns))),
                    ),
                ]
            )
            estimator = make_pipeline(
                transform,
                model(
                    "logistic"
                    if name
                    in {
                        "context_baseline",
                        "without_formation",
                        "without_channels",
                        "without_opponent",
                        "formation_only",
                    }
                    else name
                ),
            )

            def matrix(data):
                return np.array(
                    [[r[source].get(k) for k in columns] for r in data], dtype=object
                )

            with threadpool_limits(limits=1), warnings.catch_warnings(
                record=True
            ) as recorded:
                warnings.simplefilter("always")
                estimator.fit(matrix(train), y)
                pv = estimator.predict_proba(matrix(valid))[:, 1]
            if not np.all(np.isfinite(pv)):
                raise ValueError("Invalid probability")
            result = {
                "model": name,
                "validation": metrics(yv, np.clip(pv, 1e-6, 1 - 1e-6)),
                "warnings": sorted({str(r.message) for r in recorded}),
                "features": columns,
                "train_rows": len(train),
                "validation_rows": len(valid),
                "test_rows": len(test),
            }
            objects[name] = (estimator, columns, source)
            results.append(result)
        except Exception as exc:
            failures.append(
                {"model": name, "type": type(exc).__name__, "message": str(exc)}
            )
    selected = min(
        (r for r in results if r["model"] != "context_baseline"),
        key=lambda r: r["validation"]["log_loss"],
    )["model"]
    # Final fit still precedes all held-out match dates. Use a common budget for
    # every algorithm, and do not change the selected model after test scoring.
    combined = train + valid
    if len(combined) > train_cap:
        ix = np.random.default_rng(43).choice(len(combined), train_cap, replace=False)
        combined = [combined[i] for i in sorted(ix)]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for result in results:
        estimator, columns, source = objects[result["model"]]

        def matrix(data):
            return np.array(
                [[r[source].get(k) for k in columns] for r in data], dtype=object
            )

        with threadpool_limits(limits=1), warnings.catch_warnings(
            record=True
        ) as recorded:
            warnings.simplefilter("always")
            estimator.fit(
                matrix(combined), np.array([r["shot_created"] for r in combined])
            )
            pred = estimator.predict_proba(matrix(test))[:, 1]
        if not np.all(np.isfinite(pred)):
            raise ValueError("Invalid test probability")
        pred = np.clip(pred, 1e-6, 1 - 1e-6)
        test_predictions[result["model"]] = pred
        result["test"] = metrics(yt, pred)
        result["selected_by_validation"] = result["model"] == selected
        result["refit_warnings"] = sorted({str(r.message) for r in recorded})
        if result["model"] == "context_baseline":
            baseline_predictions = pred
        if result["model"] == selected:
            joblib.dump(
                {
                    "pipeline": estimator,
                    "columns": columns,
                    "source": source,
                    "prediction_point": "end of first three on-ball actions",
                    "validation_selected": True,
                    "training_end_exclusive": test_start,
                },
                output / "chance-model.joblib",
            )
    clusters = defaultdict(list)
    for i, row in enumerate(test):
        clusters[row["match_id"]].append(i)
    keys = sorted(clusters)
    rng = np.random.default_rng(42)
    for result in results:
        pred = test_predictions[result["model"]]
        if baseline_predictions is not None:
            difference = (pred - yt) ** 2 - (baseline_predictions - yt) ** 2
            samples = []
            for _ in range(200):
                chosen = rng.choice(keys, len(keys), replace=True)
                indices = np.concatenate([clusters[k] for k in chosen])
                samples.append(float(np.mean(difference[indices])))
            result["brier_difference_vs_context"] = float(np.mean(difference))
            result["match_bootstrap_ci95"] = np.quantile(
                samples, [0.025, 0.975]
            ).tolist()
    selected_prediction = test_predictions[selected]
    (output / "chance-predictions.json").write_text(
        json.dumps(
            [
                {
                    "match_id": r["match_id"],
                    "team_id": r["team_id"],
                    "period": r["period"],
                    "possession_id": r["possession_id"],
                    "date": r["date"],
                    "shot_created": int(a),
                    "probability": float(p),
                }
                for r, a, p in zip(test, yt, selected_prediction)
            ]
        )
    )
    return {
        "selected_model": selected,
        "results": results,
        "failures": failures,
        "constant_baseline": metrics(yt, constant),
        "split": {
            "validation_start": valid_start,
            "test_start": test_start,
            "train_matches": len({r["match_id"] for r in train}),
            "validation_matches": len({r["match_id"] for r in valid}),
            "test_matches": len(clusters),
            "common_fit_budget": train_cap,
        },
        "notes": [
            "Labels are later non-penalty shots in the same possession, after the third action ends.",
            "Possessions with a shot already observed by the decision clock are excluded.",
            "No final-possession length, future position, future formation or future xG enters features.",
            "All possessions from a fixture stay in one chronological split; confidence resamples fixtures.",
            "Formation and channel associations do not establish the effect of changing a tactic.",
        ],
    }


def relationships(rows):
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import Ridge

    groups = defaultdict(list)
    opponents = defaultdict(list)
    teams = defaultdict(list)
    for row in rows:
        key = (
            row["competition_id"],
            row["season_id"],
            row["starting_formation"],
            row["opponent_starting_formation"],
        )
        groups[key].append(row)
        opponents[
            (
                row["competition_id"],
                row["season_id"],
                row["team_id"],
                row["opponent_id"],
            )
        ].append(row)
        teams[(row["competition_id"], row["season_id"], row["team_id"])].append(row)

    def summary(sample):
        def mean(key):
            values = [r[key] for r in sample if r.get(key) is not None]
            return float(np.mean(values)) if values else None

        return {
            "team_match_rows": len(sample),
            "unique_matches": len({r["match_id"] for r in sample}),
            "home_rows": sum(r["home"] for r in sample),
            "away_rows": sum(not r["home"] for r in sample),
            "npxg_per100_possessions": mean("npxg_per100_possessions"),
            "possession_clock_share_proxy": mean("possession_clock_share_proxy"),
            "clock_coverage_fraction": mean("clock_coverage_fraction"),
            "low_y_attack_share": mean("low_y_attack_share"),
            "central_attack_share": mean("central_attack_share"),
            "high_y_attack_share": mean("high_y_attack_share"),
            "shot_possession_rate": mean("shot_possession_rate"),
            "pressure_per100_opponent_passes": mean("pressure_per100_opponent_passes"),
        }

    formation = [
        {
            "competition_id": key[0],
            "season_id": key[1],
            "formation": key[2],
            "opponent_formation": key[3],
            **summary(sample),
        }
        for key, sample in groups.items()
        if len(sample) >= 5
    ]
    dossier = [
        {
            "competition_id": k[0],
            "season_id": k[1],
            "season": sample[0]["season"],
            "team_id": k[2],
            "team": sample[0]["team"],
            "opponent_id": k[3],
            "opponent": sample[0]["opponent"],
            "formation_choices": dict(
                Counter(r["starting_formation"] or "unknown" for r in sample)
            ),
            **summary(sample),
        }
        for k, sample in opponents.items()
    ]
    variation = []
    for key, sample in teams.items():
        if len(sample) < 8:
            continue
        by_opp = defaultdict(list)
        for r in sample:
            by_opp[r["opponent_id"]].append(r)
        values = [
            summary(v)["possession_clock_share_proxy"]
            for v in by_opp.values()
            if len(v) >= 2
        ]
        values = [v for v in values if v is not None]
        variation.append(
            {
                "competition_id": key[0],
                "season_id": key[1],
                "season": sample[0]["season"],
                "team_id": key[2],
                "team": sample[0]["team"],
                "matches": len(sample),
                "opponents": len(by_opp),
                "repeated_opponent_possession_range": (
                    max(values) - min(values) if len(values) >= 2 else None
                ),
                "formations": dict(
                    Counter(r["starting_formation"] or "unknown" for r in sample)
                ),
            }
        )
    numeric = [
        "possession_clock_share_proxy",
        "central_attack_share",
        "low_y_attack_share",
        "mean_on_ball_origin_x",
        "pre_match_archive_elo_difference",
        "home",
    ]
    category = [
        "team_id",
        "competition_id",
        "season_id",
        "starting_formation",
        "opponent_starting_formation",
    ]
    eligible = [
        r
        for r in rows
        if r["npxg_per100_possessions"] is not None and r["possessions"] >= 10
    ]
    x = np.array(
        [
            [r.get(k) for k in numeric] + [str(r.get(k) or "unknown") for k in category]
            for r in eligible
        ],
        dtype=object,
    )
    y = np.array([r["npxg_per100_possessions"] for r in eligible])
    transformer = ColumnTransformer(
        [
            (
                "numeric",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                list(range(len(numeric))),
            ),
            (
                "fixed_context",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(range(len(numeric), len(numeric) + len(category))),
            ),
        ]
    )
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        fit = make_pipeline(transformer, Ridge(alpha=10.0)).fit(x, y)
    coefficients = dict(zip(numeric, map(float, fit[-1].coef_[: len(numeric)])))
    return {
        "formation_matchups": formation,
        "opponent_dossiers": dossier,
        "team_adjustment_profiles": variation,
        "descriptive_adjusted_associations": {
            "rows": len(eligible),
            "outcome": "non-penalty xG per100possessions",
            "numeric_coefficients_per_standard_deviation": coefficients,
            "fixed_context": category,
            "method": "Regularized observational regression; no causal effect or optimal formation claim.",
        },
    }


def save_results(database, output, report, data):
    from app.data_service.warehouse import Warehouse
    from .artifacts import archive_outputs, file_digest

    run_id = "tactics-" + file_digest(Path(output) / "report.json")[:16]
    relations = report["relations"]
    with Warehouse(database) as db:
        db.connection.executescript(
            """CREATE TABLE IF NOT EXISTS tactical_assessment_runs(run_id TEXT PRIMARY KEY,payload TEXT);CREATE TABLE IF NOT EXISTS tactical_team_matches(run_id TEXT REFERENCES tactical_assessment_runs(run_id),match_id INTEGER,team_id INTEGER,payload TEXT,PRIMARY KEY(run_id,match_id,team_id));CREATE TABLE IF NOT EXISTS tactical_opponent_dossiers(run_id TEXT REFERENCES tactical_assessment_runs(run_id),row_number INTEGER,payload TEXT,PRIMARY KEY(run_id,row_number));CREATE TABLE IF NOT EXISTS tactical_tracking_context(run_id TEXT REFERENCES tactical_assessment_runs(run_id),row_number INTEGER,payload TEXT,PRIMARY KEY(run_id,row_number));"""
        )
        db.connection.execute(
            "INSERT INTO tactical_assessment_runs VALUES(?,?) ON CONFLICT(run_id) DO UPDATE SET payload=excluded.payload",
            (run_id, json.dumps(report)),
        )
        db.connection.executemany(
            "INSERT OR REPLACE INTO tactical_team_matches VALUES(?,?,?,?)",
            [
                (run_id, r["match_id"], r["team_id"], json.dumps(r))
                for r in data["team_match_rows"]
            ],
        )
        db.connection.execute(
            "DELETE FROM tactical_opponent_dossiers WHERE run_id=?", (run_id,)
        )
        db.connection.executemany(
            "INSERT INTO tactical_opponent_dossiers VALUES(?,?,?)",
            [
                (run_id, i, json.dumps(r))
                for i, r in enumerate(relations["opponent_dossiers"], 1)
            ],
        )
        db.connection.executemany(
            "INSERT OR REPLACE INTO tactical_tracking_context VALUES(?,?,?)",
            [
                (run_id, i, json.dumps(r))
                for i, r in enumerate(data["tracking_context"], 1)
            ],
        )
        archive_outputs(
            db, "tactical_assessment_artifacts", output, "tactical_artifact", run_id
        )
        db.connection.commit()
        db.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return run_id


def run(
    database="data/football.sqlite3", output=".tmp/tactical-relations", train_cap=15000
):
    import sqlite3
    from .tactical_relations import collect
    from .artifacts import cached_json, source_signature, file_digest

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(
        f"file:{Path(database).resolve()}?mode=ro", uri=True
    ) as connection:
        inputs = source_signature(connection, ("statsbomb", "skillcorner"))
    extraction_signature = {
        "sources": inputs,
        "extractor_sha256": file_digest(
            Path(__file__).with_name("tactical_relations.py")
        ),
    }
    data = cached_json(
        output / "data.json", extraction_signature, lambda: collect(database, output)
    )
    model_signature = {
        "extraction": extraction_signature,
        "model_sha256": file_digest(__file__),
        "train_cap": train_cap,
    }
    chances = cached_json(
        output / "chance-benchmark.json",
        model_signature,
        lambda: fit_chances(data["prefixes"], output, train_cap),
    )
    ablations = cached_json(
        output / "ablations.json",
        model_signature,
        lambda: fit_chances(
            data["prefixes"],
            output / "ablations",
            train_cap,
            methods=[
                "logistic",
                "formation_only",
                "without_formation",
                "without_channels",
                "without_opponent",
            ],
        ),
    )
    relations = relationships(data["team_match_rows"])
    report = {
        "fixtures": data["matches"],
        "team_match_rows": len(data["team_match_rows"]),
        "eligible_prefixes": len(data["prefixes"]),
        "tracking_team_matches": len(data["tracking_context"]),
        "formation_records": len(data["formation_changes"]),
        "audit": data["audit"],
        "chance_models": chances,
        "feature_ablations": ablations,
        "relations": relations,
        "input_receipt": model_signature,
        "definitions": {
            "nominal_formation": "Provider Starting XI and Tactical Shift codes; not inferred from sparse positions.",
            "direction": "Completed move endpoints in low-y, central, high-y thirds, in StatsBomb attacking orientation. Forward moves require >=5 x units.",
            "possession": "Event-clock and on-ball-action share proxies; official possession remains unknown. Unassigned long/dead-time gaps and clock coverage are reported.",
            "role_positions": "Means of on-ball action locations, not average off-ball standing positions.",
            "tracking": "Detected-player team shape and vendor-classified attack/defense phase geometry; camera coverage and extrapolation caveats retained.",
            "opponent_context": "Archive-conditioned Elo and up to8 previously observed opponent fixtures; all same-day outcomes update after forecasts.",
            "association": "Observational within context; tactics respond to score, opponent strength and selection.",
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    save_results(database, output, report, data)
    print(
        json.dumps(
            {
                "fixtures": report["fixtures"],
                "prefixes": report["eligible_prefixes"],
                "selected_chance_model": chances["selected_model"],
                "failures": chances["failures"],
                "report": str(output / "report.json"),
            },
            indent=2,
        )
    )
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="data/football.sqlite3")
    parser.add_argument("--output", default=".tmp/tactical-relations")
    parser.add_argument("--train-cap", type=int, default=15000)
    args = parser.parse_args()
    run(args.database, args.output, args.train_cap)
