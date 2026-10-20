"""Proportional-odds match probabilities using a single pre-match Elo covariate."""

import math
from time import perf_counter

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit, log_expit

from app.ml.model import ModelTrainingError, NotFittedError, _validate_classes


def _log_probabilities(theta, x):
    coefficient, first, log_gap = theta
    gap = math.exp(log_gap)
    low = first - coefficient * x
    high = low + gap
    # The middle class uses a log difference identity, avoiding CDF cancellation.
    return np.column_stack((log_expit(low),
                            log_expit(high) + log_expit(-low) + math.log(-math.expm1(-gap)),
                            log_expit(-high)))


def _objective(theta, x, targets, ridge):
    coefficient, first, log_gap = theta
    gap = math.exp(log_gap)
    low = first - coefficient * x
    high = low + gap
    a, b = expit(low), expit(high)
    slope = np.where(targets == 0, 1 - a, np.where(targets == 1, 1 - a - b, -b))
    ratio = gap * math.exp(-gap) / (-math.expm1(-gap))
    gap_slope = np.where(targets == 1, gap * (1 - b) + ratio,
                         np.where(targets == 2, -gap * b, 0.))
    logs = _log_probabilities(theta, x)[np.arange(len(x)), targets]
    loss = -float(np.mean(logs)) + .5 * ridge * coefficient ** 2
    gradient = np.array([np.mean(x * slope) + ridge * coefficient,
                         -np.mean(slope), -np.mean(gap_slope)])
    return loss, gradient


class EloOrderedLogit:
    technology = 'elo_ordered_logit'
    technology_version = scipy.__version__
    explanation_method = 'unavailable'
    used_feature_names = ('elo_diff',)
    bounds = ((-30., 30.), (-30., 30.), (-12., 6.))

    def __init__(self, feature_names, *, ridge=.001, max_iter=1000,
                 feature_name='elo_diff', unit_scale=400.):
        if not feature_names or len(set(feature_names)) != len(feature_names) or feature_name not in feature_names:
            raise ModelTrainingError('特征必须非空、唯一并包含所选评级差')
        if isinstance(unit_scale, bool) or not isinstance(unit_scale, (int, float)) or not math.isfinite(unit_scale) or unit_scale <= 0:
            raise ModelTrainingError('评级单位缩放必须是有限正数')
        if isinstance(ridge, bool) or not isinstance(ridge, (int, float)) or not math.isfinite(ridge) or ridge < 0:
            raise ModelTrainingError('ridge 必须是有限非负数')
        if type(max_iter) is not int or max_iter < 1:
            raise ModelTrainingError('max_iter 必须是正整数')
        self.feature_names = tuple(feature_names)
        self.used_feature_names = (feature_name,)
        self.rating_feature = feature_name
        self.unit_scale = float(unit_scale)
        self.ridge, self.max_iter = ridge, max_iter
        self.is_fitted_ = False
        self.parameters_ = None
        self.fit_diagnostics_ = {}

    def _matrix(self, rows):
        # Existing saved Elo estimators predate the optional rating-covariate fields.
        feature = getattr(self, 'rating_feature', 'elo_diff')
        scale = getattr(self, 'unit_scale', 400.)
        try:
            if not rows or any(isinstance(row[feature], bool) for row in rows):
                raise ValueError('empty rows or boolean Elo')
            values = np.array([float(row[feature]) / scale for row in rows])
            if not np.isfinite(values).all():
                raise ValueError('nonfinite Elo')
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ModelTrainingError('评级差值必须是有限数值且输入非空') from exc
        return values

    def fit(self, rows, targets):
        from app.ml.alternatives import ModelConvergenceError

        self.is_fitted_ = False
        self.parameters_ = None
        self.fit_diagnostics_ = {}
        self.rating_feature = getattr(self, 'rating_feature', 'elo_diff')
        self.unit_scale = getattr(self, 'unit_scale', 400.)
        if not rows or len(rows) != len(targets):
            raise ModelTrainingError('训练行与目标必须非空且长度一致')
        _validate_classes(targets, '目标')
        if set(targets) != {0, 1, 2}:
            raise ModelTrainingError('有序三分类拟合需要客胜、平局、主胜三个类别')
        x = self._matrix(rows)
        y = np.asarray(targets, dtype=int)
        frequencies = np.bincount(y, minlength=3) / len(y)
        first = math.log(frequencies[0] / (1 - frequencies[0]))
        second = math.log((1 - frequencies[2]) / frequencies[2])
        initial = [0., first, math.log(second - first)]
        fitted = minimize(_objective, initial, args=(x, y, self.ridge), jac=True,
                          method='L-BFGS-B', bounds=self.bounds,
                          options={'maxiter': self.max_iter, 'gtol': 1e-8, 'ftol': 1e-12, 'maxls': 40})
        if (not fitted.success or not np.isfinite(fitted.x).all()
                or not np.isfinite(fitted.jac).all() or not math.isfinite(fitted.fun)):
            raise ModelConvergenceError('有序逻辑回归未收敛或产生非有限数值')
        self.parameters_ = fitted.x.copy()
        self.fit_diagnostics_ = {'iterations': int(fitted.nit), 'objective': float(fitted.fun),
                                 'max_abs_gradient': float(np.max(np.abs(fitted.jac))),
                                 'coefficient_per400elo': float(fitted.x[0]) if self.rating_feature == 'elo_diff' and self.unit_scale == 400 else None,
                                 'rating_feature': self.rating_feature, 'unit_scale': self.unit_scale,
                                 'coefficient_per_scaled_rating': float(fitted.x[0]),
                                 'thresholds': [float(fitted.x[1]), float(fitted.x[1] + math.exp(fitted.x[2]))],
                                 'at_bound': [bool(abs(v - low) < 1e-7 or abs(v - high) < 1e-7)
                                              for v, (low, high) in zip(fitted.x, self.bounds)]}
        self.is_fitted_ = True
        return self

    def predict_proba(self, rows):
        if not self.is_fitted_:
            raise NotFittedError('模型尚未训练')
        probabilities = np.exp(_log_probabilities(self.parameters_, self._matrix(rows)))
        # Only correct roundoff in the analytically normalized cumulative probabilities.
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        if not np.isfinite(probabilities).all():
            raise ModelTrainingError('有序逻辑回归输出非有限概率')
        return probabilities.tolist()

    def predict(self, rows):
        return [max(range(3), key=row.__getitem__) for row in self.predict_proba(rows)]


def run_rating_experiments(matches, *, ridge=.001, folds=3, seed=42, prediction_sink=None,
                           rating_variants=('basic',), warmup_matches=0):
    """Compare one-covariate ordinal/multinomial models with full-feature controls."""
    if tuple(rating_variants) != ('basic',) or warmup_matches != 0:
        from app.ml.rating_variants import run_rating_variants
        return run_rating_variants(matches, variants=rating_variants, warmup_matches=warmup_matches,
                                   ridge=ridge, folds=folds, seed=seed, prediction_sink=prediction_sink)
    from types import SimpleNamespace
    from app.ml.alternatives import ModelConvergenceError, estimator_factory
    from app.ml.evaluation import evaluate_expanding_window
    from app.ml.experiments import chronological_split
    from app.ml.feature_engineering import FeatureEngineer
    from app.ml.model import accuracy_score, fit_estimator, estimator_runtime, multiclass_brier_score, multiclass_log_loss, ranked_probability_score
    from app.ml.score_models import prepare_score_matches

    if prediction_sink is not None and not callable(prediction_sink):
        raise ValueError('prediction_sink 必须可调用')
    # Validate ordinal settings even if another baseline could otherwise run.
    EloOrderedLogit(('elo_diff',), ridge=ridge)
    records, counts = prepare_score_matches(matches)
    times = tuple(r.kickoff.isoformat() for r in records)
    split = chronological_split(SimpleNamespace(rows=records, kickoffs=times))
    development = FeatureEngineer().build_training_dataset(records[:split])
    specifications = [
        ('elo_ordered_logit', 'elo_ordered_logit', ('elo_diff',), {'ridge': ridge}),
        ('elo_multinomial', 'logistic_l2', ('elo_diff',), {}),
        ('logistic_l2', 'logistic_l2', FeatureEngineer.FEATURES, {}),
        ('xgboost', 'xgboost', FeatureEngineer.FEATURES, {}),
    ]
    candidates = []
    for name, method, features, params in specifications:
        candidate = {'model': name, 'estimator': method, 'feature_names': list(features), 'params': params}
        try:
            evaluation = evaluate_expanding_window(development.rows, development.targets, features,
                           estimator_factory(method, seed=seed, params=params), folds=folds, kickoffs=times[:split])
            candidate.update(selection_log_loss=evaluation.log_loss,
                             development_folds=list(evaluation.fold_metrics))
        except ModelConvergenceError as exc:
            candidate.update(selection_log_loss=None, selection_error=str(exc))
        candidates.append(candidate)
    valid = [c for c in candidates if c['selection_log_loss'] is not None]
    if not valid:
        raise ModelConvergenceError('所有评级基线均未收敛')
    selected = min(valid, key=lambda c: c['selection_log_loss'])['model']
    # No heldout feature or label is supplied to the candidate-selection stage.
    data = FeatureEngineer().build_training_dataset(records)
    for candidate in valid:
        model = estimator_factory(candidate['estimator'], seed=seed, params=candidate['params'])(tuple(candidate['feature_names']))
        fit_started = perf_counter()
        try:
            fit_estimator(model, data.rows[:split], data.targets[:split], kickoffs=times[:split])
        except ModelConvergenceError as exc:
            candidate['holdout_error'] = str(exc)
            candidate['runtime'] = {**estimator_runtime(model),
                                    'final_fit_seconds': perf_counter() - fit_started,
                                    'holdout_inference_seconds': None}
            continue
        fit_seconds = perf_counter() - fit_started
        inference_started = perf_counter()
        probabilities = model.predict_proba(data.rows[split:])
        candidate['runtime'] = {**estimator_runtime(model),
                                'final_fit_seconds': fit_seconds,
                                'holdout_inference_seconds': perf_counter() - inference_started}
        targets = data.targets[split:]
        predictions = [max(range(3), key=row.__getitem__) for row in probabilities]
        candidate.update(holdout_rows=len(targets), holdout_accuracy=accuracy_score(targets, predictions),
                         holdout_log_loss=multiclass_log_loss(targets, probabilities),
                         holdout_brier_score=multiclass_brier_score(targets, probabilities),
                         holdout_rps=ranked_probability_score(targets, probabilities))
        if isinstance(model, EloOrderedLogit):
            candidate['fit_diagnostics'] = model.fit_diagnostics_
        if prediction_sink is not None:
            prediction_sink({'model': candidate['model'], 'match_ids': data.match_ids[split:],
                             'kickoffs': times[split:], 'targets': targets, 'probabilities': probabilities})
    return {'seed': seed, 'counts': counts, 'selection_rows': split, 'holdout_rows': len(records) - split,
            'rating_definition': {'version': 'basic_elo_v1', 'initial_rating': 1500., 'k': 20.,
                                  'home_offset': 80., 'expected_score': 'logistic_base10_scale400',
                                  'goal_margin_weighting': False, 'warmup_matches': 0,
                                  'probability_mapping': 'separately_fitted_ordered_or_multinomial_logit',
                                  'neutral_venue_supported': False},
            'selected': selected, 'selected_on': 'development_expanding_window_log_loss',
            'development_last_kickoff': times[split - 1], 'holdout_first_kickoff': times[split],
            'holdout_last_kickoff': times[-1], 'results': candidates,
            'holdout_protocol': 'Fixed model with advancing earlier-kickoff history; kickoff does not prove final-whistle/source-arrival availability.',
            'comparison_scope': 'Complete configurations: proportional odds versus multinomial, and one Elo feature versus30features. Scaling and regularization also differ; not an isolated causal effect of model family.'}
