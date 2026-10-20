"""Bounded, fixed-output native TreeSHAP research on a frozen temporal holdout."""

import hashlib
import json
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import xgboost as xgb

from app.ml.experiments import chronological_split
from app.ml.feature_engineering import FeatureEngineer
from app.ml.model import ModelTrainingError, XGBoostClassifier, fit_estimator, estimator_runtime
from app.ml.score_models import prepare_score_matches


CLASS_ORDER = ("A", "D", "H")


def _unit(name):
    if name not in FeatureEngineer.FEATURES:
        return "source units not declared"
    if "elo" in name:
        return "Elo points"
    if "rest" in name:
        return "days (capped at 21)"
    if "rate" in name:
        return "fraction"
    if "matches_seen" in name or name == "experience_diff":
        return "completed matches"
    if name == "home_advantage":
        return "indicator"
    if "points" in name:
        return "points per recent match"
    return "goals per recent match"


def explain_cohort(estimator, rows, kickoffs, *, class_index=2, against=None):
    """Keep the same class/contrast for every row, including interaction pairs."""
    if type(class_index) is not int or class_index not in range(3):
        raise ValueError("class_index 必须是 0、1、2")
    if against is not None and (type(against) is not int or against not in range(3) or against == class_index):
        raise ValueError("对照类别必须有效且不同于固定类别")
    if not rows or len(rows) > 200 or len(kickoffs) != len(rows) or len(estimator.feature_names) > 50:
        raise ValueError("解释必须包含 1–200 行、对应时间和至多 50 个特征")
    matrix = np.asarray(estimator._matrix(rows), dtype=float)
    sample = xgb.DMatrix(matrix, feature_names=list(estimator.feature_names))
    booster = estimator._booster()
    contributions = booster.predict(sample, pred_contribs=True, strict_shape=True)
    interactions = booster.predict(sample, pred_interactions=True, strict_shape=True)
    margins = booster.predict(sample, output_margin=True, strict_shape=True)
    n, f = matrix.shape
    if contributions.shape != (n, 3, f + 1) or interactions.shape != (n, 3, f + 1, f + 1) or margins.shape != (n, 3):
        raise ModelTrainingError("原生 TreeSHAP 或交互张量形状非法")
    values = contributions[:, class_index].astype(float)
    pairs = interactions[:, class_index].astype(float)
    output = margins[:, class_index].astype(float)
    if against is not None:
        values -= contributions[:, against]
        pairs -= interactions[:, against]
        output -= margins[:, against]
    if (not all(np.isfinite(a).all() for a in (values, pairs, output))
            or not np.allclose(values.sum(axis=1), output, rtol=1e-5, atol=1e-5)
            or not np.allclose(pairs.sum(axis=2), values, rtol=1e-5, atol=1e-5)
            or not np.allclose(pairs, pairs.swapaxes(1, 2), rtol=1e-5, atol=1e-5)):
        raise ModelTrainingError("TreeSHAP、显式交互和模型原始输出不闭合")
    features = []
    for i, name in enumerate(estimator.feature_names):
        features.append({"feature": name, "input_unit": _unit(name),
                         "raw_values": matrix[:, i].tolist(), "contributions": values[:, i].tolist(),
                         "main_effects": pairs[:, i, i].tolist(),
                         "mean_absolute_contribution": float(np.mean(np.abs(values[:, i])))})
    pair_summary = [{"feature_a": estimator.feature_names[i], "feature_b": estimator.feature_names[j],
                     "mean_pair_total": float(np.mean(2 * pairs[:, i, j])),
                     "mean_absolute_pair_total": float(np.mean(np.abs(2 * pairs[:, i, j])))}
                    for i in range(f) for j in range(i + 1, f)]
    model_hash = hashlib.sha256(bytes(booster.save_raw())).hexdigest()
    identity = {"model_sha256": model_hash, "class_index": class_index, "against": against,
                "feature_names": list(estimator.feature_names), "kickoffs": list(kickoffs),
                "input_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
                "engine": estimator_runtime(estimator)}
    return {"schema": "SoccerAnalytics.GlobalExplanation.v1",
            "run_id": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20],
            "identity": identity, "class_order": list(CLASS_ORDER),
            "fixed_output": CLASS_ORDER[class_index] + (" - " + CLASS_ORDER[against] if against is not None else ""),
            "output_unit": "raw margin", "feature_dependence": "native_tree_path_dependent",
            "reference": "Path cover statistics and baseline of this fitted booster, not an interventional background.",
            "rows": n, "kickoffs": list(kickoffs), "raw_outputs": output.tolist(),
            "baseline": values[:, -1].tolist(), "features": features, "interaction_pairs": pair_summary,
            "interaction_convention": "Native off-diagonal cells each hold half the pair interaction; reported pair totals multiply one cell by two. Diagonal is the main effect.",
            "closure_max_absolute_error": float(np.max(np.abs(values.sum(axis=1) - output))),
            "interaction_row_sum_max_absolute_error": float(np.max(np.abs(pairs.sum(axis=2) - values))),
            "limits": ["Contributions are raw-score units, not probability points or causal effects.",
                       "Fixed output for the whole cohort; never mix each row's winning class.",
                       "Dependence plots show observed feature values and contributions, not interventions.",
                       "Only the first bounded subset of the final holdout is explained; no sample is selected by outcome or explanation size."]}


def run_explanations(matches, *, class_index=2, against=None, max_rows=120, seed=42):
    if type(max_rows) is not int or not 1 <= max_rows <= 200:
        raise ValueError("max_rows 必须是 1–200 的整数")
    records, counts = prepare_score_matches(matches)
    times = tuple(r.kickoff.isoformat() for r in records)
    split = chronological_split(SimpleNamespace(rows=records, kickoffs=times))
    data = FeatureEngineer().build_training_dataset(records)
    model = XGBoostClassifier(FeatureEngineer.FEATURES, seed=seed)
    fit_started = perf_counter()
    fit_estimator(model, data.rows[:split], data.targets[:split], kickoffs=times[:split])
    fit_seconds = perf_counter() - fit_started
    end = min(len(records), split + max_rows)
    report = explain_cohort(model, data.rows[split:end], times[split:end], class_index=class_index, against=against)
    report.update(counts=counts, seed=seed, training_rows=split, holdout_rows=len(records) - split,
                  explained_rows=end - split, training_last_kickoff=times[split - 1],
                  holdout_first_kickoff=times[split], holdout_last_kickoff=times[-1],
                  cohort_policy="first_k_holdout_rows_in_kickoff_then_match_id_order",
                  model_policy="fixed_default_XGBoost_fitted_only_on_development;not_a_release_or_model_comparison",
                  feature_window=5, model_fit_seconds=fit_seconds,
                  scope="Frozen model on a chronological holdout. Features advance only with earlier kickoff results; kickoff does not certify final-whistle or source-arrival availability.")
    return report
