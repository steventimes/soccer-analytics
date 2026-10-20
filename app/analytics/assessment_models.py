"""Comparable regression experiments; validation selects models, test measures them."""

from __future__ import annotations
import hashlib
import json
import logging
from pathlib import Path
import time
import warnings

logger = logging.getLogger(__name__)
METHODS = (
    "ridge",
    "bayesian_ridge",
    "elastic_net",
    "sgd",
    "svr",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "xgboost",
    "mlp",
)


def choose_model(results):
    eligible = [r for r in results if r.get("validation_mae") is not None]
    if not eligible:
        raise ValueError("No valid validation result")
    return min(eligible, key=lambda r: (r["validation_mae"], r["model"]))["model"]


def heldout_player(entity_id):
    return int(hashlib.sha256(str(entity_id).encode()).hexdigest()[:8], 16) % 5 == 0


def estimator(name, variant=0):
    from sklearn.linear_model import Ridge, BayesianRidge, ElasticNet, SGDRegressor
    from sklearn.svm import SVR
    from sklearn.ensemble import (
        RandomForestRegressor,
        ExtraTreesRegressor,
        HistGradientBoostingRegressor,
    )
    from sklearn.neural_network import MLPRegressor

    if name == "ridge":
        return Ridge(alpha=(1.0, 10.0)[variant])
    if name == "bayesian_ridge":
        return BayesianRidge(lambda_1=(1e-6, 1e-3)[variant])
    if name == "elastic_net":
        return ElasticNet(
            alpha=(0.01, 0.1)[variant], l1_ratio=0.5, max_iter=3000, random_state=42
        )
    if name == "sgd":
        return SGDRegressor(
            alpha=(1e-4, 1e-2)[variant],
            penalty="elasticnet",
            learning_rate="adaptive",
            eta0=0.01,
            max_iter=3000,
            tol=1e-4,
            random_state=42,
        )
    if name == "svr":
        return SVR(C=(1.0, 10.0)[variant], epsilon=0.1)
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=(None, 10)[variant],
            min_samples_leaf=5,
            n_jobs=1,
            random_state=42,
        )
    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=120,
            min_samples_leaf=(3, 8)[variant],
            n_jobs=1,
            random_state=42,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=120,
            max_leaf_nodes=15,
            l2_regularization=(0.0, 1.0)[variant],
            early_stopping=False,
            random_state=42,
        )
    if name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=150,
            max_depth=(3, 5)[variant],
            learning_rate=0.05,
            n_jobs=1,
            objective="reg:squarederror",
            random_state=42,
        )
    if name == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=((32,), (64, 32))[variant],
            max_iter=400,
            alpha=0.01,
            early_stopping=False,
            random_state=42,
        )
    raise ValueError(name)


def pipeline(name, features, variant):
    from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
                list(range(len(numeric), len(columns))),
            ),
        ]
    )
    model = TransformedTargetRegressor(
        regressor=make_pipeline(transform, estimator(name, variant)),
        transformer=StandardScaler(),
    )
    return model, columns


def clip_prediction(values, target):
    import numpy as np

    if target in {"pass_completion", "duel_win_rate", "keeper_success_rate"}:
        return np.clip(values, 0, 1)
    if target == "points":
        return np.clip(values, 0, 3)
    return np.maximum(0, values)


def metrics(actual, predicted):
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": (
            float(r2_score(actual, predicted))
            if len(actual) > 1 and np.var(actual) > 0
            else None
        ),
    }


def feature_matrix(rows, columns, game_features=False):
    import numpy as np

    values = []
    for row in rows:
        features = dict(row["features"])
        if game_features:
            features.update(row.get("game_features", {}))
        values.append([features.get(column) for column in columns])
    return np.array(values, dtype=object)


def benchmark(
    examples,
    targets,
    *,
    kind,
    output,
    methods=METHODS,
    mode="temporal",
    game_features=False,
    matched_only=False,
    train_cap=12000,
):
    import numpy as np
    import joblib
    from threadpoolctl import threadpool_limits

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    winners = {}
    forecast_models = {}
    for target in targets:
        eligible = [
            r
            for r in examples
            if r["targets"].get(target) is not None
            and (not (game_features or matched_only) or r.get("has_game_data"))
        ]
        if kind == "player":
            train = [r for r in eligible if r["phase"] == 0]
            valid = [r for r in eligible if r["phase"] == 1]
            test = [r for r in eligible if r["phase"] == 2]
            if mode == "unseen_players":
                train = [r for r in train if not heldout_player(r["entity_id"])]
                valid = [r for r in valid if not heldout_player(r["entity_id"])]
                test = [r for r in test if heldout_player(r["entity_id"])]
        else:
            train = [
                r
                for r in eligible
                if 2008 <= r["season"] <= 2023 and r["target_end"] < "2024-07-01"
            ]
            valid = [
                r
                for r in eligible
                if r["season"] == 2024
                and r["feature_asof"] >= "2024-07-01"
                and r["target_end"] < "2025-07-01"
            ]
            test = [
                r
                for r in eligible
                if r["season"] == 2025
                and r["feature_asof"] >= "2025-07-01"
                and r["target_end"] < "2026-07-01"
            ]
        if len(train) < 40 or len(valid) < 20 or len(test) < 20:
            rows.append(
                {
                    "target": target,
                    "status": "insufficient_rows",
                    "train": len(train),
                    "validation": len(valid),
                    "test": len(test),
                }
            )
            continue
        # Deterministic common training budget for every algorithm, including SVM.
        if len(train) > train_cap:
            ids = np.random.default_rng(42).choice(len(train), train_cap, replace=False)
            train = [train[i] for i in sorted(ids)]
        yt = np.array([r["targets"][target] for r in train])
        yv = np.array([r["targets"][target] for r in valid])
        ys = np.array([r["targets"][target] for r in test])
        grouped = {}
        for r in train:
            key = (r["league"], r.get("position", "team"))
            grouped.setdefault(key, []).append(r["targets"][target])
        means = {key: float(np.mean(values)) for key, values in grouped.items()}

        def baselines(data):
            mean = np.array(
                [
                    means.get(
                        (r["league"], r.get("position", "team")), float(np.mean(yt))
                    )
                    for r in data
                ]
            )
            carry = np.array(
                [
                    (
                        r["carry_forward"].get(target)
                        if r["carry_forward"].get(target) is not None
                        else mean[i]
                    )
                    for i, r in enumerate(data)
                ]
            )
            shrunk = (
                np.array(
                    [
                        (
                            r["features"].get(target)
                            if r["features"].get(target) is not None
                            else mean[i]
                        )
                        for i, r in enumerate(data)
                    ]
                )
                if kind == "player"
                else (8 * carry + 5 * mean) / 13
            )
            return {
                k: clip_prediction(v, target)
                for k, v in {
                    "carry_forward": carry,
                    "cohort_mean": mean,
                    "empirical_bayes": shrunk,
                }.items()
            }

        validation_baselines = baselines(valid)
        test_baselines = baselines(test)
        baseline_name = min(
            validation_baselines,
            key=lambda k: metrics(yv, validation_baselines[k])["mae"],
        )
        baseline = test_baselines[baseline_name]
        candidates = []
        for name in methods:
            best = None
            for variant in [0, 1]:
                start = time.perf_counter()
                try:
                    features = dict(train[0]["features"])
                    if game_features:
                        features.update(train[0]["game_features"])
                    model, columns = pipeline(name, features, variant)
                    xt, xv = feature_matrix(
                        train, columns, game_features
                    ), feature_matrix(valid, columns, game_features)
                    with threadpool_limits(limits=1), warnings.catch_warnings(
                        record=True
                    ) as notices:
                        warnings.simplefilter("always")
                        model.fit(xt, yt)
                        pred = clip_prediction(model.predict(xv), target)
                    score = metrics(yv, pred)["mae"]
                    trial = {
                        "model": name,
                        "variant": variant,
                        "validation_mae": score,
                        "fit_seconds": time.perf_counter() - start,
                        "warnings": sorted({str(n.message) for n in notices}),
                    }
                    if best is None or score < best["validation_mae"]:
                        best = {**trial, "model_object": model, "columns": columns}
                except Exception as exc:
                    failures.append(
                        {
                            "target": target,
                            "model": name,
                            "variant": variant,
                            "error": type(exc).__name__,
                            "message": str(exc),
                        }
                    )
            if best:
                candidates.append(best)
        if not candidates:
            continue
        selected = choose_model(candidates)
        chosen_predictions = None
        for candidate in candidates:
            name = candidate["model"]
            columns = candidate["columns"]
            model = candidate["model_object"]
            combined = train + valid
            with threadpool_limits(limits=1):
                model.fit(
                    feature_matrix(combined, columns, game_features),
                    np.array([r["targets"][target] for r in combined]),
                )
                pred = clip_prediction(
                    model.predict(feature_matrix(test, columns, game_features)), target
                )
            difference = np.abs(pred - ys) - np.abs(baseline - ys)
            rng = np.random.default_rng(42)
            # Team rows are sampled as entity clusters because each club has repeated windows.
            groups = {r["entity_id"] for r in test}
            by_group = {
                g: np.flatnonzero([r["entity_id"] == g for r in test]) for g in groups
            }
            keys = sorted(groups)
            samples = []
            for _ in range(200):
                picked = rng.choice(keys, len(keys), replace=True)
                indices = np.concatenate([by_group[g] for g in picked])
                samples.append(float(np.mean(difference[indices])))
            result = {
                "target": target,
                "model": name,
                "variant": candidate["variant"],
                "validation_mae": candidate["validation_mae"],
                "test": metrics(ys, pred),
                "baseline": metrics(ys, baseline),
                "baseline_name": baseline_name,
                "all_baselines": {k: metrics(ys, v) for k, v in test_baselines.items()},
                "mae_difference": float(np.mean(difference)),
                "baseline_selected_by": "validation_mae",
                "cluster_bootstrap_ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
                "train": len(train),
                "validation": len(valid),
                "test_rows": len(test),
                "selected_by_validation": name == selected,
                "fit_seconds": candidate["fit_seconds"],
                "warnings": candidate["warnings"],
                "mode": mode,
                "game_features": game_features,
            }
            rows.append(result)
            if name == selected:
                path = (
                    output
                    / f'{kind}-{mode}-{target}-{"game" if game_features else "real"}.joblib'
                )
                joblib.dump(
                    {
                        "model": model,
                        "columns": columns,
                        "target": target,
                        "kind": kind,
                        "game_features": game_features,
                        "selected_by": "validation_mae",
                        "train_target_end": max(r["target_end"] for r in combined),
                    },
                    path,
                )
                winners[target] = {k: v for k, v in result.items()}
                winners[target]["model_path"] = str(path)
                forecast_models[target] = {
                    "model": model,
                    "columns": columns,
                    "game_features": game_features,
                }
                chosen_predictions = [
                    {
                        "entity_id": r["entity_id"],
                        "name": r["name"],
                        "league": r["league"],
                        "as_of": r["feature_asof"],
                        "target_end": r["target_end"],
                        "actual": float(a),
                        "predicted": float(p),
                    }
                    for r, a, p in zip(test, ys, pred)
                ]
        if chosen_predictions:
            (output / f"{kind}-{mode}-{target}-predictions.json").write_text(
                json.dumps(chosen_predictions, indent=2)
            )
        logger.info(
            "%s %s target %s: %d/%d/%d rows, selected %s",
            kind,
            mode,
            target,
            len(train),
            len(valid),
            len(test),
            selected,
        )
    result = {
        "kind": kind,
        "mode": mode,
        "game_features": game_features,
        "matched_only": matched_only,
        "methods": list(methods),
        "results": rows,
        "selected_models": winners,
        "failures": failures,
        "notes": [
            "Hyperparameters and model names selected by validation MAE only; test results are reported for all candidates.",
            "All algorithms share the same train/validation/test rows and deterministic train cap.",
            "Scalers, imputers and category encoders fit only on training data, then train+validation for final test.",
            "Intervals resample entity clusters; they are conditional on fitted models.",
            "Targets condition on sufficient future exposure in the covered competitions; they do not predict availability or injury.",
        ],
    }
    return result, forecast_models
