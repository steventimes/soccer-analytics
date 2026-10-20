"""Joint feature/model research using development-only search and fixed holdouts."""

from time import perf_counter

from app.ml.alternatives import ModelConvergenceError, estimator_factory
from app.ml.evaluation import evaluate_expanding_window, expanding_window_boundaries
from app.ml.experiments import OPTIMIZERS, SPACES, chronological_split, search_parameters
from app.ml.feature_engineering import FeatureEngineer
from app.ml.model import accuracy_score, fit_estimator, estimator_runtime, multiclass_brier_score, multiclass_log_loss, ranked_probability_score
from app.ml.score_models import prepare_score_matches


DEFAULT_FEATURES = {"window": 5, "elo_k": 20., "home_advantage": 80.}


def feature_dataset(matches, *, window=5, elo_k=20., home_advantage=80.):
    """Research-only recipe; never saves a production bundle with unbound settings."""
    import math

    if type(window) is not int or window < 1:
        raise ValueError("window 必须是正整数")
    for value in (elo_k, home_advantage):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("Elo 参数必须是有限非负数")
    engineer = FeatureEngineer(window)
    # Specialize the existing update constants on this research instance only.
    engineer.ELO_K = elo_k
    engineer.HOME_ELO_ADVANTAGE = home_advantage
    return engineer.build_training_dataset(matches)


def _recipe(params):
    model_params = {k: v for k, v in params.items() if not k.startswith('feature_')}
    features = {**DEFAULT_FEATURES, **{k.removeprefix('feature_'): v for k, v in params.items()
                                    if k.startswith('feature_')}}
    return model_params, features


def run_feature_experiments(matches, *, methods=("xgboost", "logistic_l2"),
                            optimizer="optuna", trials=12, folds=3, seed=42,
                            prediction_sink=None):
    """Compare fixed defaults, model-only and joint search without saving models.

    The optional sink receives individual final-holdout arrays for local audits;
    these private rows are deliberately excluded from the aggregate return value.
    """
    if not methods or any(method not in SPACES for method in methods) or optimizer not in OPTIMIZERS:
        raise ValueError("未知或空的实验模型/优化方法")
    if type(trials) is not int or trials < 4:
        raise ValueError("trials 必须是至少 4 的整数")
    if prediction_sink is not None and not callable(prediction_sink):
        raise ValueError("prediction_sink 必须可调用")
    records, counts = prepare_score_matches(matches)
    # Partition by canonical match identity/time before any feature/model search.
    from types import SimpleNamespace

    times = tuple(r.kickoff.isoformat() for r in records)
    split = chronological_split(SimpleNamespace(rows=records, kickoffs=times))
    development = records[:split]
    boundaries = expanding_window_boundaries(split, folds=folds, kickoffs=times[:split])
    cache = {}

    def development_data(recipe):
        key = tuple(sorted(recipe.items()))
        if key not in cache:
            cache[key] = feature_dataset(development, **recipe)
        return cache[key]

    candidates = []
    for method in dict.fromkeys(methods):
        def objective(params):
            model_params, features = _recipe(params)
            data = development_data(features)
            return evaluate_expanding_window(
                data.rows, data.targets, FeatureEngineer.FEATURES,
                estimator_factory(method, seed=seed, params=model_params),
                folds=folds, kickoffs=data.kickoffs,
            )

        default = {"model": method, "scope": "default", "params": {}, "features": DEFAULT_FEATURES.copy()}
        default_started = perf_counter()
        try:
            evaluation = objective({})
            default['selection_log_loss'] = evaluation.log_loss
            default['development_folds'] = list(evaluation.fold_metrics)
        except ModelConvergenceError as exc:
            default.update(selection_log_loss=None, selection_error=str(exc))
        default['development_objective_seconds'] = perf_counter() - default_started
        candidates.append(default)
        for scope in ('model', 'joint'):
            try:
                search = search_parameters(method, optimizer, objective, trials=trials, seed=seed,
                                           include_features=scope == 'joint')
                model_params, features = _recipe(search['best_params'])
                candidate = {"model": method, "scope": scope, "params": model_params, "features": features,
                             "selection_log_loss": search['best_loss'], "search": search}
            except ModelConvergenceError as exc:
                candidate = {"model": method, "scope": scope, "params": None, "features": None,
                             "selection_log_loss": None, "selection_error": str(exc),
                             "search": exc.search}
            candidates.append(candidate)
    valid = [c for c in candidates if c['selection_log_loss'] is not None]
    if not valid:
        raise ModelConvergenceError("所有特征/模型配置均未收敛")
    selected = min(valid, key=lambda c: c['selection_log_loss'])
    identity = {k: selected[k] for k in ['model', 'scope', 'params', 'features']}
    # Freeze the choice before building any held-out feature or scoring its label.
    for candidate in valid:
        data = feature_dataset(records, **candidate['features'])
        estimator = estimator_factory(candidate['model'], seed=seed, params=candidate['params'])(FeatureEngineer.FEATURES)
        fit_started = perf_counter()
        try:
            fit_estimator(estimator, data.rows[:split], data.targets[:split], kickoffs=times[:split])
        except ModelConvergenceError as exc:
            candidate['holdout_error'] = str(exc)
            candidate['runtime'] = {**estimator_runtime(estimator),
                                    'final_fit_seconds': perf_counter() - fit_started,
                                    'holdout_inference_seconds': None}
            continue
        fit_seconds = perf_counter() - fit_started
        inference_started = perf_counter()
        probabilities = estimator.predict_proba(data.rows[split:])
        candidate['runtime'] = {**estimator_runtime(estimator),
                                'final_fit_seconds': fit_seconds,
                                'holdout_inference_seconds': perf_counter() - inference_started}
        targets = data.targets[split:]
        predictions = [max(range(3), key=lambda i: row[i]) for row in probabilities]
        candidate.update(holdout_rows=len(targets), holdout_accuracy=accuracy_score(targets, predictions),
                         holdout_log_loss=multiclass_log_loss(targets, probabilities),
                         holdout_brier_score=multiclass_brier_score(targets, probabilities),
                         holdout_rps=ranked_probability_score(targets, probabilities))
        if prediction_sink is not None:
            prediction_sink({**{k: candidate[k] for k in ['model', 'scope', 'params', 'features']},
                             'match_ids': data.match_ids[split:], 'kickoffs': times[split:],
                             'targets': targets, 'probabilities': probabilities})
    return {
        'seed': seed, 'optimizer': optimizer, 'trials_per_search': trials, 'counts': counts,
        'selection_rows': split, 'holdout_rows': len(records) - split,
        'selected_on': 'development_expanding_window_log_loss', 'selected': identity,
        'holdout_protocol': 'Fixed model; pre-match features advance after earlier kickoff results. Earlier kickoff does not prove final-whistle or source-arrival availability.',
        'feature_space': {'window': [2, 12], 'elo_k': [5., 50.], 'home_advantage': [0., 150.]},
        'elo_only_window_policy': 'elo_ordered_logit ignores recent-form features; its recent-match window stays5 and is excluded from joint search.',
        'feature_semantics': 'window counts the last N matches, not N seasons; elo_k is update magnitude; home_advantage offsets the Elo expectation, not a probability.',
        'development_first_kickoff': times[0], 'development_last_kickoff': times[split - 1],
        'holdout_first_kickoff': times[split], 'holdout_last_kickoff': times[-1],
        'folds': [{'training_rows': a, 'validation_rows': b - a,
                   'training_last_kickoff': times[a - 1], 'validation_first_kickoff': times[a],
                   'validation_last_kickoff': times[b - 1]} for a, b in zip(boundaries, boundaries[1:])],
        'results': candidates,
    }
