"""Conditional paired match resampling of fixed test predictions."""

import hashlib
import json

import numpy as np
import scipy
from scipy.stats import bootstrap

LOSS_METRICS = ("completion_log_loss", "completion_brier", "forward_displacement_mae_pitch_pct")


def paired_match_ranges(differences, match_ids, *, replicates=2000, seed=20261015):
    """Pool sampled pass losses within each resampled match collection.

    Match totals and counts use the same draws. This conditions on the observed
    pass subset and fitted models; it does not refit or resample training seeds.
    """
    values, groups = np.asarray(differences, dtype=float), np.asarray(match_ids)
    if (values.ndim != 2 or not len(values) or not values.shape[1]
            or not np.isfinite(values).all() or groups.shape != (len(values),)
            or groups.dtype.kind not in "iu" or np.any(groups < 1)
            or type(replicates) is not int or replicates < 200
            or type(seed) is not int or seed < 0):
        raise ValueError("Invalid paired match losses, identities or resampling options")
    ids, inverse = np.unique(groups, return_inverse=True)
    if len(ids) < 2:
        raise ValueError("Paired resampling requires at least two test matches")
    counts = np.bincount(inverse)
    totals = np.array([np.bincount(inverse, weights=column) for column in values.T])

    def pooled_difference(sums, sizes, axis=-1):
        return sums.sum(axis=axis) / sizes.sum(axis=axis)

    result = bootstrap((totals, counts), pooled_difference, axis=-1, paired=True,
                       vectorized=True, method="percentile", confidence_level=.95,
                       n_resamples=replicates, batch=200, rng=np.random.default_rng(seed))
    return {"difference": values.mean(axis=0).tolist(),
            "lower": np.asarray(result.confidence_interval.low).tolist(),
            "upper": np.asarray(result.confidence_interval.high).tolist()}


def compare_test_predictions(labels, match_ids, predictions, *,
                             reference="hist_gradient_boosting_context",
                             replicates=2000, seed=20261015):
    """Average losses across fixed seeds, rather than ensemble probabilities."""
    y = np.asarray(labels, dtype=float)
    groups = np.asarray(match_ids)
    if (y.ndim != 2 or y.shape[1] != 2 or not len(y) or not np.isfinite(y).all()
            or not np.isin(y[:, 0], [0, 1]).all() or np.any(np.abs(y[:, 1]) > 1)
            or not isinstance(predictions, dict) or len(predictions) < 2
            or reference not in predictions):
        raise ValueError("Invalid paired test labels or reference")
    losses, digest = {}, hashlib.sha256()
    for name, runs in predictions.items():
        if not isinstance(name, str) or not name or not isinstance(runs, list) or not runs:
            raise ValueError("Every comparison method needs named fixed predictions")
        parts = []
        digest.update(json.dumps([name, len(runs)]).encode())
        for prediction in runs:
            raw = np.asarray(prediction)
            p = np.asarray(raw, dtype=float)
            if (p.shape != y.shape or not np.isfinite(p).all()
                    or np.any(p[:, 0] < 0) or np.any(p[:, 0] > 1)):
                raise ValueError("Invalid paired test predictions")
            eps = max(1e-7, np.finfo(raw.dtype).eps if raw.dtype.kind == "f" else 1e-7)
            probability = np.clip(p[:, 0], eps, 1 - eps)
            parts.append(np.column_stack((
                -(y[:, 0] * np.log(probability) + (1 - y[:, 0]) * np.log1p(-probability)),
                (probability - y[:, 0]) ** 2,
                np.abs(p[:, 1] - y[:, 1]) * 100,
            )))
            digest.update(p.astype('<f8', copy=False).tobytes())
        losses[name] = np.mean(parts, axis=0)
    rows = []
    for name, values in losses.items():
        if name == reference:
            continue
        interval = paired_match_ranges(values - losses[reference], groups,
                                       replicates=replicates, seed=seed)
        rows.append({"method": name, "runs": len(predictions[name]),
                     **{metric: {key: interval[key][i] for key in ("difference", "lower", "upper")}
                        for i, metric in enumerate(LOSS_METRICS)}})
    cohort = hashlib.sha256(groups.astype('<i8').tobytes() + y.astype('<f8').tobytes()).hexdigest()
    return {
        "method": "paired_match_case_bootstrap", "reference": reference,
        "test_sequences": len(y), "test_matches": len(np.unique(groups)),
        "replicates": replicates, "seed": seed, "rng": "numpy.PCG64",
        "scipy_version": scipy.__version__, "quantiles": [.025, .975],
        "interval_method": "percentile", "weighting": "pooled_sampled_passes",
        "seed_aggregation": "mean_loss_across_fixed_seeds",
        "cohort_sha256": cohort, "predictions_sha256": digest.hexdigest(),
        "method_url": "https://journal.r-project.org/articles/RJ-2023-015/",
        "engine_url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html",
        "scope": "差值为各方法减去参照模型；正值表示误差更大。先逐传球平均固定种子的损失，再按整场测试比赛配对重采样，按抽到的传球数汇总；不是概率集成，也不是各场均值的等权平均。",
        "limits": [
            "范围是逐项2.5%–97.5%重采样百分位，不是同时置信带或已验证覆盖率的置信区间。",
            "固定已训练模型、种子和抽样传球；不覆盖训练、选参、种子总体或场内再次抽样的不确定性。",
            "重复球队的跨场相关和时间漂移未建模；不能据此推断未来赛季或因果效果。",
            "树基线在先前测试对照后选作参照；这些是事后探索比较，没有多重比较显著性结论。",
        ], "comparisons": rows,
    }
