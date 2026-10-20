"""Paired diagnostics for fixed outcome forecasts on observed UTC match-days."""

from datetime import UTC, datetime
import hashlib
import json

METRICS = ("log_loss", "brier_score", "rps")


def _integer_options(values, minimum, maximum, label):
    values = tuple(values)
    if (not values or len(set(values)) != len(values)
            or any(type(v) is not int or not minimum <= v <= maximum for v in values)):
        raise ValueError(f"Invalid {label}")
    return values


def _stationary_indices(groups, replicates, block_length, rng):
    """Uniform restart with probability 1/L, otherwise continue around the circle."""
    import numpy as np

    indices = np.empty((replicates, groups), dtype=np.int64)
    indices[:, 0] = rng.integers(groups, size=replicates)
    for t in range(1, groups):
        restart = rng.random(replicates) < 1 / block_length
        proposed = rng.integers(groups, size=replicates)
        indices[:, t] = np.where(restart, proposed, (indices[:, t - 1] + 1) % groups)
    return indices


def paired_outcome_comparison(targets, match_ids, kickoffs, probabilities, model_names,
                              *, reference, replicates=2000, seed=20261017,
                              block_lengths=(1, 3, 7), hac_lags=(0, 1, 3, 7)):
    """Compare match-weighted losses, preserving whole days and paired forecasts.

    Let S_g be a day's sum of paired loss differences, n_g its match count,
    and d=sum(S_g)/sum(n_g). The ratio influence is u_g=S_g-n_g*d.
    Bartlett variance of d is [sum(u_g²)+2 sum_l w_l sum_g u_g*u_(g-l)]/N².
    For one match per day this is the usual biased-autocovariance HAC variance
    of the sample mean. ACF describes these same influences on observed days;
    zero influence variance leaves it undefined. No p-value or coverage claim.
    """
    import numpy as np
    from app.ml.model import _validate_probability_rows

    y, ids, times = np.asarray(targets), np.asarray(match_ids), np.asarray(kickoffs)
    names, p = np.asarray(model_names), np.asarray(probabilities)
    n = len(y) if y.ndim else 0
    if (y.shape != (n,) or n < 2 or y.dtype.kind not in "iu" or not np.isin(y, [0, 1, 2]).all()
            or ids.shape != (n,) or ids.dtype.kind not in "iu" or np.any(ids < 1)
            or len(np.unique(ids)) != n or times.shape != (n,) or times.dtype.kind != "U"
            or names.ndim != 1 or len(names) < 2 or names.dtype.kind != "U"
            or len(set(names)) != len(names) or any(not s.strip() for s in names)
            or not isinstance(reference, str) or reference not in names
            or p.shape != (len(names), n, 3) or p.dtype.kind not in "fiu"
            or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1)
            or type(replicates) is not int or replicates < 200
            or type(seed) is not int or seed < 0):
        raise ValueError("Invalid paired outcome identities, labels, probabilities or options")
    for matrix in p:
        _validate_probability_rows(matrix)
    dates = []
    for value in times:
        dt = datetime.fromisoformat(str(value))
        if dt.tzinfo is None or dt.utcoffset() is None:
            raise ValueError("Kickoffs require an explicit time zone")
        dates.append(dt.astimezone(UTC))
    order = np.lexsort((ids, [dt.timestamp() for dt in dates]))
    ids, y = ids[order], y[order]
    p = p[:, order].astype(float)
    dates = [dates[i] for i in order]
    days, inverse, counts = np.unique([dt.date().isoformat() for dt in dates],
                                      return_inverse=True, return_counts=True)
    g = len(days)
    block_lengths = _integer_options(block_lengths, 1, g, "block lengths")
    hac_lags = _integer_options(hac_lags, 0, g - 1, "HAC lags")
    if g < 2:
        raise ValueError("Comparison requires at least two UTC match-days")
    onehot = np.eye(3)[y]
    losses = np.stack((-np.log(np.clip(p[:, np.arange(n), y], 1e-15, 1)),
                       ((p - onehot) ** 2).sum(axis=2),
                       ((p.cumsum(axis=2)[:, :, :2] - onehot.cumsum(axis=1)[:, :2]) ** 2).mean(axis=2)), axis=2)
    baseline = list(names).index(reference)
    candidates = [i for i in range(len(names)) if i != baseline]
    differences = losses[candidates] - losses[baseline]
    points = differences.mean(axis=1)
    totals = np.array([[np.bincount(inverse, weights=col, minlength=g) for col in rows.T]
                       for rows in differences])  # candidate, metric, day
    influence = totals - points[:, :, None] * counts
    lag_products = [(influence ** 2).sum(axis=2)]
    for lag in range(1, max(hac_lags) + 1):
        lag_products.append((influence[:, :, lag:] * influence[:, :, :-lag]).sum(axis=2))
    hac = []
    for lag in hac_lags:
        variance = lag_products[0].copy()
        for k in range(1, lag + 1):
            variance += 2 * (1 - k / (lag + 1)) * lag_products[k]
        if np.any(variance < -1e-10):
            raise ValueError("Bartlett variance is negative beyond rounding tolerance")
        hac.append(np.sqrt(np.maximum(variance, 0)) / n)
    ranges = []
    for length in block_lengths:
        # Every method/metric uses the same draws; each replicate pools its match counts.
        rng = np.random.default_rng(seed)
        samples = []
        for offset in range(0, replicates, 200):
            indices = _stationary_indices(g, min(200, replicates - offset), length, rng)
            samples.append(totals[:, :, indices].sum(axis=3) / counts[indices].sum(axis=1))
        samples = np.concatenate(samples, axis=2)
        ranges.append(np.quantile(samples, [.025, .975], axis=2, method="linear"))
    comparisons = []
    for j, i in enumerate(candidates):
        comparisons.append({"model": str(names[i]), "metrics": {
            metric: {"difference": float(points[j, k]),
                     "day_influence_acf": [
                         {"lag": lag,
                          "correlation": (float(product[j, k] / lag_products[0][j, k])
                                          if lag_products[0][j, k] > 0 else None)}
                         for lag, product in enumerate(lag_products)],
                     "hac": [{"lag": lag, "standard_error": float(hac[z][j, k])}
                             for z, lag in enumerate(hac_lags)],
                     "stationary_bootstrap": [{"mean_block_days": length,
                                               "lower": float(ranges[z][0, j, k]),
                                               "upper": float(ranges[z][1, j, k])}
                                              for z, length in enumerate(block_lengths)]}
            for k, metric in enumerate(METRICS)}})
    canonical = json.dumps({"match_ids": ids.tolist(), "targets": y.tolist(),
                            "kickoffs": [d.isoformat(timespec="microseconds") for d in dates]},
                           sort_keys=True, separators=(",", ":")).encode()
    predictions = json.dumps(names.tolist(), ensure_ascii=False).encode() + p.astype('<f8').tobytes()
    gaps = [(datetime.fromisoformat(str(b)) - datetime.fromisoformat(str(a))).days
            for a, b in zip(days, days[1:])]
    return {
        "schema": "SoccerAnalytics.PairedOutcomeComparison.v1", "reference": reference,
        "class_order": ["L", "D", "W"], "difference_direction": "candidate minus reference; lower is better",
        "matches": n, "utc_match_days": g, "first_kickoff": dates[0].isoformat(),
        "last_kickoff": dates[-1].isoformat(), "max_calendar_gap_days": max(gaps),
        "matches_per_day": {"min": int(counts.min()), "max": int(counts.max())},
        "cohort_sha256": hashlib.sha256(canonical).hexdigest(),
        "predictions_sha256": hashlib.sha256(predictions).hexdigest(),
        "replicates": replicates, "seed": seed, "rng": "numpy.PCG64",
        "block_unit": "Ordered observed UTC match-days, not elapsed calendar days",
        "weighting": "Pooled matches in each resampled set of days; unequal daily counts retained",
        "probability_floor_for_log_loss": 1e-15, "quantiles": [.025, .975],
        "probability_normalization": "Original rows retained within existing evaluator tolerance1e-6",
        "max_input_row_mass_error": float(np.abs(p.sum(axis=2) - 1).max()),
        "scores": [{"model": str(name), **{metric: float(losses[i, :, k].mean())
                                           for k, metric in enumerate(METRICS)}}
                   for i, name in enumerate(names)],
        "comparisons": comparisons,
        "limits": [
            "Fixed previously fitted forecasts and paired historical labels; no refitting or selection uncertainty.",
            "Stationary resampling assumes weak dependence across ordered observed days, wraps at the end and does not preserve calendar gaps or season breaks.",
            "HAC uses match-weighted day-total ratio influences with Bartlett lag weights; it is a local grouped adaptation, not the original equally spaced DM test.",
            "ACF uses those same day influences and a common lag-zero sum-of-squares denominator; zero variance gives null correlations. Lags count observed days, including across calendar gaps. No significance bands or stationarity conclusion.",
            "Lag and block grids are sensitivity settings, not estimated optimal values. Percentiles are pointwise descriptive ranges, not validated or simultaneous confidence coverage.",
            "No p-values, rejection labels, model reselection or prospective superiority claims. Repeated historical evaluation and long-range team dependence remain limitations.",
        ],
    }


def compare_outcome_file(path, **options):
    """Read only numeric/string NPZ arrays. IDs and individual forecasts stay local."""
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        required = ("targets", "match_ids", "kickoffs", "probabilities", "model_names")
        if not all(name in data for name in required):
            raise ValueError("Paired NPZ needs targets, match_ids, kickoffs, probabilities and model_names")
        return paired_outcome_comparison(**{name: data[name] for name in required}, **options)
