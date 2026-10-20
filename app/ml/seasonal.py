"""Whole-season walk-forward comparisons of history length and cross-league pooling."""

from __future__ import annotations

import logging
import sys
from collections import Counter, defaultdict
from importlib.metadata import PackageNotFoundError, version
from statistics import mean, pstdev

from app.data_service.fetch.public import _identity
from app.data_service.history import audit_history, canonical_team, digest, match_key
from app.ml.alternatives import METHODS, estimator_factory
from app.ml.evaluation import elo_baseline_predictions
from app.ml.feature_engineering import FeatureEngineer
from app.ml.model import (
    accuracy_score,
    fit_estimator,
    multiclass_brier_score,
    multiclass_log_loss,
    ranked_probability_score,
)

logger = logging.getLogger(__name__)


def runtime_versions() -> dict:
    versions = {"python": sys.version.split()[0]}
    for package in ("numpy", "xgboost-cpu", "scikit-learn"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def prepare_matches(matches, source: str, competition: str) -> list[dict]:
    prepared = []
    for match in matches:
        # These identities are experimental only; provider-specific production bundles stay unchanged.
        item = dict(match)
        for venue in ("homeTeam", "awayTeam"):
            team = dict(item[venue])
            team["id"] = _identity(source, competition, canonical_team(team["name"]))
            item[venue] = team
        item["utcDate"] = (
            match.get("source_date", match["utcDate"][:10]) + "T00:00:00+00:00"
        )
        prepared.append(item)
    return prepared


def summarize_matrix(results: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for result in results:
        groups[
            (result["source"], result["model"], result["window"], result["strategy"])
        ].append(result)
    summary = []
    for (source, model, window, strategy), rows in sorted(groups.items()):
        total = sum(row["test_rows"] for row in rows)
        summary.append(
            {
                "source": source,
                "model": model,
                "window": window,
                "strategy": strategy,
                "league_seasons": len(rows),
                "test_rows": total,
                **{
                    metric: sum(row[metric] * row["test_rows"] for row in rows) / total
                    for metric in (
                        "accuracy",
                        "log_loss",
                        "brier_score",
                        "rps",
                        "majority_accuracy",
                        "elo_accuracy",
                    )
                },
                "league_season_log_loss_stddev": pstdev(
                    row["log_loss"] for row in rows
                ),
                "league_season_accuracy_stddev": pstdev(
                    row["accuracy"] for row in rows
                ),
            }
        )
    return summary


def paired_comparisons(results: list[dict]) -> list[dict]:
    lookup = {
        (
            row["source"],
            row["competition"],
            row["test_season"],
            row["model"],
            row["window"],
            row["strategy"],
        ): row
        for row in results
    }
    groups = defaultdict(list)
    for key, row in lookup.items():
        source, code, season, model, window, strategy = key
        reference_key = (source, code, season, model, "1", "local")
        if key == reference_key or reference_key not in lookup:
            continue
        reference = lookup[reference_key]
        if row["test_population_sha256"] != reference["test_population_sha256"]:
            raise ValueError("配对实验使用了不同的测试比赛")
        groups[(source, model, window, strategy)].append(
            row["log_loss"] - reference["log_loss"]
        )
    return [
        {
            "source": source,
            "model": model,
            "window": window,
            "strategy": strategy,
            "reference": "one-season local model",
            "pairs": len(values),
            "mean_log_loss_delta": mean(values),
            "log_loss_delta_stddev": pstdev(values),
            "improved_pairs": sum(value < 0 for value in values),
            "worsened_pairs": sum(value > 0 for value in values),
        }
        for (source, model, window, strategy), values in sorted(groups.items())
    ]


def run_season_matrix(
    collections: dict,
    *,
    test_seasons=None,
    windows=("1", "3", "all"),
    models=("xgboost", "sgd", "random_forest", "gaussian_nb"),
    seed: int = 42,
    pooled: bool = True,
    factory=estimator_factory,
    progress=None,
    reference_collections=None,
) -> dict:
    sources = sorted({key[0] for key in collections})
    competitions = sorted({key[1] for key in collections})
    seasons = sorted({key[2] for key in collections})
    tests = sorted(set(test_seasons or seasons[-3:]))
    windows, models = list(dict.fromkeys(windows)), list(dict.fromkeys(models))
    if (
        not collections
        or not models
        or any(
            model not in METHODS
            for model in models
        )
    ):
        raise ValueError("没有数据或模型列表非法")
    if not windows or any(
        window != "all" and (not window.isdigit() or int(window) <= 0)
        for window in windows
    ):
        raise ValueError("window 必须是正整数或 all")
    if any(season not in seasons for season in tests):
        raise ValueError("测试赛季不在历史快照中")
    audit = audit_history(
        reference_collections if reference_collections is not None else collections
    )
    excluded = defaultdict(set)
    for record in audit["excluded_records"]:
        excluded[(record["source"], record["competition"], record["season"])].add(
            record["match_id"]
        )
    collections = {
        key: [match for match in matches if match["id"] not in excluded[key]]
        for key, matches in collections.items()
    }
    features = FeatureEngineer().features + tuple(
        f"league_{code}" for code in competitions
    )
    results, errors = [], []
    for source in sources:
        for test_season in tests:
            for window in windows:
                packs = {}
                expected_history = [
                    season for season in seasons if season < test_season
                ]
                required = expected_history
                if window != "all":
                    required = [
                        str(year)
                        for year in range(
                            int(test_season) - int(window), int(test_season)
                        )
                    ]
                for code in competitions:
                    try:
                        if window == "all":
                            required = [
                                season
                                for season in expected_history
                                if (source, code, season) in collections
                            ]
                        if not required or any(
                            (source, code, season) not in collections
                            for season in [*required, test_season]
                        ):
                            raise ValueError("请求的训练或测试赛季快照不完整")
                        history = []
                        season_by_id = {}
                        for season in [*required, test_season]:
                            for match in collections[(source, code, season)]:
                                if (
                                    match["id"] in season_by_id
                                    and season_by_id[match["id"]] != season
                                ):
                                    raise ValueError("跨赛季重复比赛 ID")
                                season_by_id[match["id"]] = season
                                history.append(match)
                        dataset = FeatureEngineer().build_training_dataset(
                            prepare_matches(history, source, code)
                        )
                        tagged = [
                            (
                                kickoff,
                                {
                                    **row,
                                    **{
                                        f"league_{c}": float(c == code)
                                        for c in competitions
                                    },
                                },
                                target,
                                season_by_id[mid],
                            )
                            for kickoff, row, target, mid in zip(
                                dataset.kickoffs,
                                dataset.rows,
                                dataset.targets,
                                dataset.match_ids,
                            )
                        ]
                        train = [item for item in tagged if item[3] != test_season]
                        test = [item for item in tagged if item[3] == test_season]
                        if len(train) < 30 or not test:
                            raise ValueError("赛季比赛不足以训练或评估")
                        test_population = sorted(
                            (
                                match_key(m),
                                m["score"]["fullTime"]["home"],
                                m["score"]["fullTime"]["away"],
                            )
                            for m in collections[(source, code, test_season)]
                        )
                        packs[code] = {
                            "train": train,
                            "test": test,
                            "training_seasons": required,
                            "missing_history_seasons": (
                                [
                                    season
                                    for season in expected_history
                                    if season not in required
                                ]
                                if window == "all"
                                else []
                            ),
                            "test_population_sha256": digest(test_population),
                        }
                    except Exception as exc:
                        errors.append(
                            {
                                "source": source,
                                "competition": code,
                                "test_season": test_season,
                                "window": window,
                                "stage": "dataset",
                                "message": str(exc),
                            }
                        )
                if not packs:
                    continue
                # A pooled fit must finish before the earliest held-out match in any league.
                cutoff = min(pack["test"][0][0] for pack in packs.values())
                for pack in packs.values():
                    pack["train"] = [item for item in pack["train"] if item[0] < cutoff]
                all_train = sorted(
                    (item for pack in packs.values() for item in pack["train"]),
                    key=lambda item: item[0],
                )
                for model in models:
                    # This preset is fixed for the matrix; no held-out season selects hyperparameters.
                    params = {"alpha": 0.1} if model == "sgd" else {}
                    pooled_estimator = None
                    if pooled:
                        try:
                            pooled_estimator = factory(model, seed=seed, params=params)(
                                features
                            )
                            fit_estimator(
                                pooled_estimator,
                                [item[1] for item in all_train],
                                [item[2] for item in all_train],
                                kickoffs=[item[0] for item in all_train],
                            )
                        except Exception as exc:
                            errors.append(
                                {
                                    "source": source,
                                    "test_season": test_season,
                                    "window": window,
                                    "model": model,
                                    "stage": "pooled_fit",
                                    "message": str(exc),
                                }
                            )
                            pooled_estimator = None
                    for code, pack in packs.items():
                        for strategy in (("local", "pooled") if pooled else ("local",)):
                            try:
                                training = (
                                    pack["train"] if strategy == "local" else all_train
                                )
                                if len(training) < 30:
                                    raise ValueError("截止时间之前的训练比赛不足")
                                if strategy == "pooled":
                                    if pooled_estimator is None:
                                        continue
                                    estimator = pooled_estimator
                                else:
                                    estimator = factory(
                                        model, seed=seed, params=params
                                    )(features)
                                    fit_estimator(
                                        estimator,
                                        [item[1] for item in training],
                                        [item[2] for item in training],
                                        kickoffs=[item[0] for item in training],
                                    )
                                test_rows = [item[1] for item in pack["test"]]
                                targets = [item[2] for item in pack["test"]]
                                probabilities = estimator.predict_proba(test_rows)
                                train_targets = [item[2] for item in pack["train"]]
                                majority = Counter(train_targets).most_common(1)[0][0]
                                result = {
                                    "source": source,
                                    "competition": code,
                                    "test_season": test_season,
                                    "window": window,
                                    "strategy": strategy,
                                    "model": model,
                                    "params": params,
                                    "training_seasons": pack["training_seasons"],
                                    "training_rows": len(training),
                                    "missing_history_seasons": pack[
                                        "missing_history_seasons"
                                    ],
                                    "pool_training_seasons": (
                                        {
                                            c: p["training_seasons"]
                                            for c, p in packs.items()
                                        }
                                        if strategy == "pooled"
                                        else {code: pack["training_seasons"]}
                                    ),
                                    "training_end": training[-1][0],
                                    "test_start": pack["test"][0][0],
                                    "test_end": pack["test"][-1][0],
                                    "test_rows": len(targets),
                                    "test_population_sha256": pack[
                                        "test_population_sha256"
                                    ],
                                    "pool_competitions": (
                                        sorted(packs)
                                        if strategy == "pooled"
                                        else [code]
                                    ),
                                    "accuracy": accuracy_score(
                                        targets, estimator.predict(test_rows)
                                    ),
                                    "log_loss": multiclass_log_loss(
                                        targets, probabilities
                                    ),
                                    "brier_score": multiclass_brier_score(
                                        targets, probabilities
                                    ),
                                    "rps": ranked_probability_score(
                                        targets, probabilities
                                    ),
                                    "majority_accuracy": accuracy_score(
                                        targets, [majority] * len(targets)
                                    ),
                                    "elo_accuracy": accuracy_score(
                                        targets,
                                        elo_baseline_predictions(
                                            [item[1] for item in pack["train"]],
                                            train_targets,
                                            test_rows,
                                        ),
                                    ),
                                }
                                results.append(result)
                            except Exception as exc:
                                errors.append(
                                    {
                                        "source": source,
                                        "competition": code,
                                        "test_season": test_season,
                                        "window": window,
                                        "model": model,
                                        "strategy": strategy,
                                        "stage": "evaluate",
                                        "message": str(exc),
                                    }
                                )
                logger.info(
                    "赛季对比 %s %s history=%s: 累计 %d 个结果",
                    source,
                    test_season,
                    window,
                    len(results),
                )
                if progress:
                    progress({"results": results, "errors": errors})
    return {
        "seed": seed,
        "runtime_versions": runtime_versions(),
        "results": results,
        "errors": errors,
        "summary": summarize_matrix(results),
        "source_comparisons": audit["comparisons"],
        "data_quality": {
            key: value for key, value in audit.items() if key != "comparisons"
        },
        "paired_comparisons": paired_comparisons(results),
        "protocol": {
            "test_seasons": tests,
            "windows": windows,
            "models": models,
            "time_precision": "day",
            "fit_cutoff": "Before earliest held-out fixture across participating leagues",
            "features": "Pre-match rolling form/Elo; earlier held-out results update team state, never the estimator",
            "sgd_params": {"alpha": 0.1},
            "hyperparameter_selection": "Fixed exploratory presets, no holdout selection",
            "preset_origin": "SGD alpha=0.1 reuses prior PL exploration; all comparisons are retrospective",
            "provider_pooling": "Never combine providers; each provider is evaluated separately",
            "note": "Repeated retrospective comparisons, not independent future validation or production release.",
        },
    }
