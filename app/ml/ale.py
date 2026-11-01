"""Bounded first-order numerical ALE, Apley–Zhu (2019), equations 15–16."""
from time import perf_counter

import numpy as np

from app.ml.alternatives import estimator_factory
from app.ml.availability import build_available_dataset, training_indices
from app.ml.experiments import chronological_split
from app.ml.feature_engineering import FeatureEngineer
from app.ml.model import fit_estimator, multiclass_brier_score, estimator_runtime
from app.run_receipts import content_digest


def accumulated_local_effects(estimator, rows, *, features, class_index=2, bins=10):
    names = tuple(estimator.feature_names)
    if (type(class_index) is not int or class_index not in range(3)
            or type(bins) is not int or not 2 <= bins <= 50
            or not rows or len(rows) > 5000 or not 1 <= len(features) <= 8
            or len(set(features)) != len(features) or any(f not in names for f in features)
            or 2 * len(rows) * len(features) > 80000):
        raise ValueError("ALE 需要有效类别、2–50 个箱、1–8 个唯一特征和有界样本/预测预算")
    if any(isinstance(row[name], bool) for row in rows for name in names):
        raise ValueError("ALE 数值输入不能是布尔值")
    matrix = np.asarray([[row[name] for name in names] for row in rows], dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("ALE 输入必须是有限数值")
    output, evaluated = [], 0
    for feature in features:
        values = matrix[:, names.index(feature)]
        edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1), method='nearest'))
        if len(edges) == 1:
            output.append({'feature': feature, 'status': 'constant', 'edges': edges.tolist(),
                           'counts': [len(rows)], 'effects': [0.], 'local_differences': [0.]})
            continue
        membership = np.clip(np.searchsorted(edges, values, side='left') - 1, 0, len(edges) - 2)
        low = [{**row, feature: float(edges[k])} for row, k in zip(rows, membership)]
        high = [{**row, feature: float(edges[k + 1])} for row, k in zip(rows, membership)]
        probabilities = estimator.predict_proba(low + high)
        multiclass_brier_score([0] * (2 * len(rows)), probabilities)
        probabilities = np.asarray(probabilities, dtype=float)
        differences = probabilities[len(rows):, class_index] - probabilities[:len(rows), class_index]
        counts = np.bincount(membership, minlength=len(edges) - 1)
        if (counts == 0).any():
            raise ValueError("ALE 分箱存在无支持区间；需减少箱数")
        local = np.bincount(membership, weights=differences, minlength=len(counts)) / counts
        accumulated = np.cumsum(local)
        centered = accumulated - np.average(accumulated, weights=counts)
        evaluated += 2 * len(rows)
        output.append({'feature': feature, 'status': 'estimated', 'edges': edges.tolist(),
                       'counts': counts.tolist(), 'local_differences': local.tolist(),
                       'effects': centered.tolist(), 'centering': 'sample_count_weighted_upper_bin_step'})
    return {'schema': 'SoccerAnalytics.NumericALE.v1', 'class_order': ['A', 'D', 'H'],
            'fixed_class': ['A', 'D', 'H'][class_index], 'output_unit': 'probability difference',
            'rows': len(rows), 'prediction_rows_evaluated': evaluated,
            'input_sha256': content_digest(rows), 'requested_bins': bins, 'features': output,
            'limits': ['Local model response on the declared cohort; not a causal effect.',
                       'One numerical coordinate at a time; redundant engineered columns can violate joint constraints even locally.',
                       'No confidence intervals, categorical ordering, or second-order interaction claim.']}


def run_ale_study(matches, *, features=('elo_diff',), model='xgboost', params=None,
                  class_index=2, bins=10, max_rows=120, seed=42, as_of=None):
    if type(max_rows) is not int or not 1 <= max_rows <= 5000:
        raise ValueError("ALE max_rows 必须是 1–5000")
    engineer = FeatureEngineer()
    data = (build_available_dataset(matches, as_of=as_of, engineer=engineer) if as_of is not None
            else engineer.build_training_dataset(matches))
    split = chronological_split(data)
    eligible = (training_indices(data, range(split), data.kickoffs[split])
                if data.result_available_at else list(range(split)))
    if len(eligible) < 30:
        raise ValueError("ALE 留出开始前实际可用标签不足 30 场")
    estimator = estimator_factory(model, seed=seed, params=params)(engineer.features)
    started = perf_counter()
    fit_estimator(estimator, [data.rows[i] for i in eligible], [data.targets[i] for i in eligible],
                  kickoffs=[data.kickoffs[i] for i in eligible])
    report = accumulated_local_effects(estimator, data.rows[split:split + max_rows],
                                       features=features, class_index=class_index, bins=bins)
    report.update(model=model, params=params or {}, seed=seed, runtime=estimator_runtime(estimator),
                  elapsed_seconds=perf_counter() - started, training_rows=len(eligible),
                  training_input_sha256=content_digest({'rows': [data.rows[i] for i in eligible],
                                                        'targets': [data.targets[i] for i in eligible]}),
                  feature_recipe=engineer.to_recipe(), holdout_rows=len(data.rows) - split,
                  cohort_policy='first_k_final_temporal_holdout_rows',
                  timing_policy='observed_result' if as_of is not None else 'legacy_kickoff_order',
                  as_of=as_of, production_model_saved=False)
    return report
