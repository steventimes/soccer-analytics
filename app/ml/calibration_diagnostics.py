"""Marginal and joint histogram diagnostics for fixed three-outcome probabilities."""

import math


def _equal_mass_indices(values, bins):
    """Choose nearest cumulative-count cuts between distinct scores, preserving ties."""
    import numpy as np

    unique, counts = np.unique(values, return_counts=True)
    if len(unique) == 1:
        return np.zeros(len(values), dtype=int), 1
    cumulative = counts.cumsum()[:-1]
    cuts = sorted({int(np.argmin(np.abs(cumulative - len(values) * j / bins)))
                   for j in range(1, bins)})
    thresholds = unique[np.array(cuts, dtype=int) + 1]
    return np.searchsorted(thresholds, values, side='right'), len(thresholds) + 1


def _summarize(probabilities, observations, indices, *, scale, minimum_bin_count, original_brier=None):
    import numpy as np

    if indices.ndim == 1:
        indices = indices[:, None]
    keys, inverse, counts = np.unique(indices, axis=0, return_inverse=True, return_counts=True)
    sums = np.column_stack([np.bincount(inverse, weights=col, minlength=len(keys))
                            for col in probabilities.T])
    positives = np.column_stack([np.bincount(inverse, weights=col, minlength=len(keys))
                                 for col in observations.T]).astype(int)
    predicted, observed = sums / counts[:, None], positives / counts[:, None]
    distances = scale * np.abs(predicted - observed).sum(axis=1)
    keep = counts >= minimum_bin_count
    summary = {'ece': float(np.dot(counts / len(probabilities), distances)),
            'occupied_bins': len(keys), 'minimum_occupied_count': int(counts.min()),
            'maximum_occupied_count': int(counts.max()),
            'displayed_bins': int(keep.sum()), 'displayed_rows': int(counts[keep].sum()),
            'suppressed_bins': int((~keep).sum()), 'suppressed_rows': int(counts[~keep].sum()),
            'cells': [{'bin_index': key.tolist(), 'count': int(count),
                       'mean_probability': mean.tolist(), 'observed_rate': rate.tolist(),
                       'observed_counts': positive.tolist()}
                      for key, count, mean, rate, positive in zip(keys[keep], counts[keep], predicted[keep], observed[keep], positives[keep])]}
    if original_brier is not None:
        # This empirical decomposition describes forecasts replaced by their
        # bin means. Within-bin variation can change the original Brier score.
        weights = counts / len(probabilities)
        class_rates = observations.mean(axis=0)
        uncertainty = float(np.sum(class_rates * (1 - class_rates)))
        reliability = float(np.dot(weights, ((predicted - observed) ** 2).sum(axis=1)))
        resolution = float(np.dot(weights, ((observed - class_rates) ** 2).sum(axis=1)))
        binned_brier = float(np.dot(weights,
                                   (((predicted - observed) ** 2)
                                    + observed * (1 - observed)).sum(axis=1)))
        summary['brier_decomposition'] = {
            'raw_brier': original_brier, 'binned_forecast_brier': binned_brier,
            'uncertainty': uncertainty, 'reliability': reliability, 'resolution': resolution,
            'within_bin_remainder': original_brier - binned_brier,
            'empirical_class_rates': class_rates.tolist(),
        }
    return summary


def calibration_diagnostics(targets, probabilities, *, bins=(5, 10, 20), minimum_bin_count=10):
    """Describe binning sensitivity, without using diagnostics to select or fit a model.

    Joint TV uses the complete vector in a permutation-equivariant cube-grid
    partition intersected with the simplex. Marginal/top-label ECE uses scalar
    absolute gaps. Equal-mass cuts depend on predictions only and preserve ties.
    Every occupied bin contributes to ECE; small cells are omitted from display.
    Brier components describe bin-mean forecasts, with a signed remainder for
    the original forecasts. They do not estimate population calibration.
    """
    import numpy as np
    from app.ml.model import multiclass_brier_score

    bins = tuple(bins)
    if (not bins or len(set(bins)) != len(bins)
            or any(type(b) is not int or not 2 <= b <= 50 for b in bins)
            or type(minimum_bin_count) is not int or minimum_bin_count < 2):
        raise ValueError('分箱数须为不重复的2–50整数；展示阈值至少为2')
    y, p = np.asarray(targets), np.asarray(probabilities)
    if (y.ndim != 1 or not len(y) or y.dtype.kind not in 'iu'
            or p.shape != (len(y), 3) or p.dtype.kind not in 'fiu'):
        raise ValueError('诊断需要等长的整数三分类标签与数值概率矩阵')
    # Keep the shared range/sum tolerance; do not renormalize frozen forecasts.
    original_brier = multiclass_brier_score(y.tolist(), p.tolist())
    p = p.astype(float)
    observed = np.eye(3)[y]
    prediction = p.argmax(axis=1)  # Matches existing first-index tie policy.
    confidence = p[np.arange(len(y)), prediction]
    correctness = (prediction == y).astype(int)
    results = []
    for resolution in bins:
        joint = _summarize(p, observed, np.minimum(resolution - 1, (p * resolution).astype(int)),
                           scale=.5, minimum_bin_count=minimum_bin_count, original_brier=original_brier)
        row = {'resolution': resolution, 'joint_tv': joint}
        for strategy in ['equal_width', 'equal_mass']:
            def scalar(values, outcomes):
                if strategy == 'equal_width':
                    indices = np.minimum(resolution - 1, (values * resolution).astype(int))
                    total_bins = resolution
                else:
                    indices, total_bins = _equal_mass_indices(values, resolution)
                summary = _summarize(values[:, None], outcomes[:, None], indices,
                                     scale=1., minimum_bin_count=minimum_bin_count)
                return {'total_bins': total_bins, 'empty_bins': total_bins - summary['occupied_bins'], **summary}
            classes = [dict(class_index=k, **scalar(p[:, k], observed[:, k])) for k in range(3)]
            row[strategy] = {'classwise_ece': math.fsum(c['ece'] for c in classes) / 3,
                             'classes': classes, 'top_label': scalar(confidence, correctness)}
        results.append(row)
    return {'schema': 'SoccerAnalytics.CalibrationDiagnostics.v1', 'rows': len(y),
            'minimum_bin_count': minimum_bin_count, 'results': results,
            'protocol': {'class_order': ['away_win', 'draw', 'home_win'],
                'joint_metric': 'Match-weighted total variation gap between bin mean vector and observed class frequencies; not the same statistic as averaged marginal ECE.',
                'joint_partition': 'floor(B*p) on all3coordinates, with1clamped into last bin; cube-grid intersections with the simplex. B is axis resolution, not total cell count.',
                'equal_mass': 'Nearest cumulative-count boundaries between distinct prediction scores; ties stay together, lower boundary wins an equal-distance tie. No labels used to choose cuts.',
                'empty_and_small_bins': 'Empty bins contribute zero; every occupied bin contributes to ECE. Only display cells with count>=minimum_bin_count.',
                'brier_decomposition': 'Class-summed mean squared error on all rows. Binned-forecast Brier = uncertainty - resolution + reliability; raw Brier adds the signed within-bin remainder. Empirical outcome frequencies describe this evaluated cohort, not a prior forecast. Components are sample/bin dependent and are not population calibration estimates or selection criteria.',
                'inference': 'Descriptive reused-sample diagnostics only. No p-values, coverage guarantee, population calibration certificate or model selection; no iid resampling assumption applied to football.'}}


def compact_diagnostics(report):
    """Keep routine train/calibrate reports small; full cell tables use the fixed-data CLI."""
    return [{'resolution': row['resolution'], 'joint_tv_ece': row['joint_tv']['ece'],
             'joint_occupied_bins': row['joint_tv']['occupied_bins'],
             'joint_small_bin_rows': row['joint_tv']['suppressed_rows'],
             **({'brier_decomposition': row['joint_tv']['brier_decomposition']}
                if 'brier_decomposition' in row['joint_tv'] else {}),
             **{strategy: {'top_label_ece': row[strategy]['top_label']['ece'],
                           'classwise_ece': row[strategy]['classwise_ece']}
                for strategy in ['equal_width', 'equal_mass']}}
            for row in report['results']]
