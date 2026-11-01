"""Joint feature/model research using development-only search and fixed holdouts."""

from time import perf_counter
from types import SimpleNamespace

from app.ml.alternatives import ModelConvergenceError, estimator_factory
from app.ml.availability import aware_time, available_records, build_available_dataset, training_indices
from app.ml.evaluation import evaluate_expanding_window, expanding_window_boundaries
from app.ml.experiments import OPTIMIZERS, SPACES, chronological_split, search_parameters, shared_initial_parameters
from app.ml.feature_engineering import FeatureEngineer
from app.ml.model import accuracy_score, fit_estimator, estimator_runtime, multiclass_brier_score, multiclass_log_loss, ranked_probability_score
from app.ml.score_models import prepare_score_matches


DEFAULT_FEATURES = {"window": 5, "elo_k": 20., "home_advantage": 80.}


def feature_dataset(matches, *, window=5, elo_k=20., home_advantage=80., as_of=None):
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
    return (build_available_dataset(matches, as_of=as_of, engineer=engineer)
            if as_of is not None else engineer.build_training_dataset(matches))


def _recipe(params):
    model_params = {k: v for k, v in params.items() if not k.startswith('feature_')}
    features = {**DEFAULT_FEATURES, **{k.removeprefix('feature_'): v for k, v in params.items()
                                    if k.startswith('feature_')}}
    return model_params, features


def run_feature_experiments(matches, *, methods=("xgboost", "logistic_l2"),
                            optimizer="optuna", trials=12, folds=3, seed=42,
                            prediction_sink=None, as_of=None, shared_start=False,
                            model_seed=None, beam_width=None):
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
    if type(shared_start) is not bool or (model_seed is not None and type(model_seed) is not int):
        raise ValueError('非法共同初始化或模型种子')
    if optimizer == 'beam' and (type(beam_width) is not int or beam_width not in (1, 2)):
        raise ValueError('beam 需要显式 width=1 或 2')
    if optimizer != 'beam' and beam_width is not None:
        raise ValueError('beam_width 仅用于 beam')
    model_seed = seed if model_seed is None else model_seed
    matches = tuple(matches)
    cutoff = aware_time(as_of) if as_of is not None else None
    # Validate source precision before canonical records discard those fields.
    strict_records = available_records(matches) if cutoff is not None else None
    records, counts = prepare_score_matches(matches)
    if strict_records is not None:
        records = tuple(r for r in strict_records if r.result_available_at <= cutoff)
        counts.update(as_of_excluded=len(strict_records) - len(records), observed_rows=len(records))
    # Partition by canonical match identity/time before any feature/model search.
    times = tuple(r.kickoff.isoformat() for r in records)
    metadata = SimpleNamespace(rows=records, kickoffs=times, result_available_at=(
        tuple(r.result_available_at.isoformat() for r in records) if cutoff is not None else ()))
    split = chronological_split(metadata)
    eligible = training_indices(metadata, range(split), times[split])
    if len(eligible) < 30:
        raise ValueError("选择时点实际可用的开发标签不足 30 场")
    development = tuple(records[i] for i in eligible)
    development_times = tuple(times[i] for i in eligible)
    development_metadata = SimpleNamespace(rows=development, result_available_at=(
        tuple(r.result_available_at.isoformat() for r in development) if cutoff is not None else ()))
    boundaries = expanding_window_boundaries(len(development), folds=folds, kickoffs=development_times)
    fold_summary = []
    for start, end in zip(boundaries, boundaries[1:]):
        training = training_indices(development_metadata, range(start), development_times[start])
        if not training:
            raise ValueError("开发折开始时没有实际可用的训练标签")
        fold_summary.append({
            'training_rows': len(training), 'validation_rows': end - start,
            'training_last_kickoff': development_times[training[-1]],
            'validation_first_kickoff': development_times[start],
            'validation_last_kickoff': development_times[end - 1],
            **({'training_rows_before_purge': start, 'labels_purged_at_fit': start - len(training)}
               if cutoff is not None else {}),
        })
    cache = {}

    def development_data(recipe):
        key = tuple(sorted(recipe.items()))
        if key not in cache:
            cache[key] = feature_dataset(development, as_of=cutoff, **recipe)
        return cache[key]

    candidates = []
    fit_events = []
    for method in dict.fromkeys(methods):
        def objective(params):
            model_params, features = _recipe(params)
            data = development_data(features)
            return evaluate_expanding_window(
                data.rows, data.targets, FeatureEngineer.FEATURES,
                estimator_factory(method, seed=model_seed, params=model_params),
                folds=folds, kickoffs=data.kickoffs, fit_observer=lambda fold: fit_events.append(fold),
                **({'result_available_at': data.result_available_at} if cutoff is not None else {}),
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
                                           include_features=scope == 'joint',
                                           initial_points=shared_initial_parameters(
                                               method, seed=seed, include_features=scope == 'joint') if shared_start or optimizer == 'beam' else None,
                                           beam_width=beam_width, fit_counter=lambda: len(fit_events))
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
        data = feature_dataset(records, as_of=cutoff, **candidate['features'])
        estimator = estimator_factory(candidate['model'], seed=model_seed, params=candidate['params'])(FeatureEngineer.FEATURES)
        fit_started = perf_counter()
        try:
            fit_estimator(estimator, tuple(data.rows[i] for i in eligible),
                          tuple(data.targets[i] for i in eligible), kickoffs=development_times)
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
        'seed': seed, 'model_seed': model_seed, 'shared_start': shared_start or optimizer == 'beam',
        'optimizer': optimizer, 'trials_per_search': trials, 'counts': counts,
        'selection_rows': len(development), 'holdout_rows': len(records) - split,
        'selected_on': 'development_expanding_window_log_loss', 'selected': identity,
        'timing_policy': 'observed_result' if cutoff is not None else 'legacy_kickoff_order',
        **({'as_of': cutoff.isoformat(), 'selection_as_of': times[split],
            'development_rows_before_purge': split, 'labels_purged_at_selection': split - len(eligible)}
           if cutoff is not None else {}),
        'holdout_protocol': (
            'Fixed model; development labels must arrive before selection and each fold fit; '
            'features advance only with results strictly available before each kickoff.'
            if cutoff is not None else
            'Fixed model; pre-match features advance after earlier kickoff results. Earlier kickoff does not prove final-whistle or source-arrival availability.'),
        'feature_space': {'window': [2, 12], 'elo_k': [5., 50.], 'home_advantage': [0., 150.]},
        'elo_only_window_policy': 'elo_ordered_logit ignores recent-form features; its recent-match window stays5 and is excluded from joint search.',
        'feature_semantics': 'window counts the last N matches, not N seasons; elo_k is update magnitude; home_advantage offsets the Elo expectation, not a probability.',
        'development_first_kickoff': development_times[0], 'development_last_kickoff': development_times[-1],
        'holdout_first_kickoff': times[split], 'holdout_last_kickoff': times[-1],
        'folds': fold_summary,
        'results': candidates,
    }


COMPARISON_ARMS = ('random', 'bayesian', 'optuna', 'genetic', 'beam1', 'beam2')


def _comparison_readiness(development, followup, *, cutoff, boundaries):
    """Describe the exact objective inputs without constructing an estimator."""
    from app.run_receipts import content_digest

    data = feature_dataset(development, as_of=cutoff)
    windows = []
    for left, right in zip(boundaries, boundaries[1:]):
        indices = training_indices(data, range(left), data.kickoffs[left])
        constant = [name for name in FeatureEngineer.FEATURES
                    if len({data.rows[i][name] for i in indices}) == 1]
        validation_constant = [name for name in FeatureEngineer.FEATURES
                    if len({row[name] for row in data.rows[left:right]}) == 1]
        no_history = sum(data.rows[i]['home_matches_seen'] == 0
                         and data.rows[i]['away_matches_seen'] == 0 for i in indices)
        windows.append({
            'fit_as_of': data.kickoffs[left],
            'training_match_ids': [data.match_ids[i] for i in indices],
            'validation_match_ids': list(data.match_ids[left:right]),
            'training_rows': len(indices), 'validation_rows': right-left,
            'labels_purged_at_fit': left-len(indices),
            'training_rows_without_history': no_history,
            'constant_features': constant,
            'varying_features': [name for name in FeatureEngineer.FEATURES if name not in constant],
            'validation_constant_features': validation_constant,
            'available_history_match_ids_at_fit': [r.match_id for r in development
                                                  if r.result_available_at < aware_time(data.kickoffs[left])],
        })
    cold = all(row['training_rows_without_history'] == row['training_rows'] for row in windows)
    observed = [r for r in followup if r.result_available_at <= cutoff]
    reasons = ['NO_OBSERVED_HISTORY_IN_TRAINING_ROWS'] if cold else []
    warnings = ['INDEPENDENT_CONFIRMATION_NOT_REGISTERED']
    if not observed:
        warnings.append('NO_OBSERVED_FOLLOWUP_BEFORE_FINAL_HOLDOUT')
    if any(len(row['constant_features']) == len(FeatureEngineer.FEATURES) for row in windows):
        warnings.append('CONSTANT_BASELINE_TRAINING_FEATURES')
    result = {
        'state': 'insufficient' if cold else 'not_blocked', 'model_fit_attempts': 0,
        'reason_codes': reasons, 'warning_codes': warnings,
        'baseline_recipe': DEFAULT_FEATURES.copy(), 'folds': windows,
        'unique_training_rows': len({i for row in windows for i in row['training_match_ids']}),
        'unique_validation_rows': len({i for row in windows for i in row['validation_match_ids']}),
        'row_history': [{'match_id': match_id, 'home_matches_seen': int(row['home_matches_seen']),
                         'away_matches_seen': int(row['away_matches_seen'])}
                        for match_id, row in zip(data.match_ids, data.rows)],
        'followup': {'state': 'available_unregistered' if observed else 'unavailable',
                     'observed_rows': len(observed), 'pending_rows': len(followup)-len(observed),
                     'match_ids': [r.match_id for r in observed], 'registered': False},
        'scope': 'Baseline recipe diagnostic, not optimizer efficacy. No per-fold sample minimum imposed. '
                 'Overlapping training folds are not independent datasets. Final holdout untouched. '
                 'No history in every training row prevents this feature-search adoption study; '
                 'constant baseline columns alone do not invalidate every algorithm. '
                 'Followup labels are inventory only, not a registered confirmation experiment.',
    }
    return {**result, 'sha256': content_digest(result)}


def _comparison_cohort(matches, *, as_of, evaluation_start, evaluation_end, final_holdout_start, folds, initial_training_rows=None):
    """Admit an explicit development-only outer interval, not the final test set."""
    from app.run_receipts import content_digest

    cutoff, start, end, final = map(aware_time, (as_of, evaluation_start, evaluation_end, final_holdout_start))
    if not start < end <= final or end > cutoff:
        raise ValueError('要求 evaluation_start < evaluation_end <= final_holdout_start 且 evaluation_end <= as_of')
    config = {'as_of': cutoff.isoformat(), 'evaluation_start': start.isoformat(),
              'evaluation_end': end.isoformat(), 'final_holdout_start': final.isoformat(), 'folds': folds}
    if initial_training_rows is not None:
        config['initial_training_rows'] = initial_training_rows
    report = {'state': 'blocked', 'timing_policy': 'observed_result', 'configuration': config,
              'readiness': {'state': 'unavailable', 'reason_codes': ['TIMING_ADMISSION_BLOCKED'], 'model_fit_attempts': 0}}
    try:
        source_records = tuple(available_records(matches))
        records = source_records
        pending = [r for r in records if start <= r.kickoff < end and r.result_available_at > cutoff]
        records = tuple(r for r in records if r.kickoff < end and r.result_available_at <= cutoff)
        development = tuple(r for r in records if r.kickoff < start and r.result_available_at < start)
        outer = tuple(r for r in records if start <= r.kickoff < end)
        report.update(development_rows=len(development), outer_rows=len(outer), pending_outer_rows=len(pending))
        if len(development) < 30:
            raise ValueError('选择时点实际可用的开发标签不足 30 场')
        if not outer or pending:
            raise ValueError('所选外层区间没有完整可观察的结果队列')
        times = tuple(r.kickoff.isoformat() for r in development)
        metadata = SimpleNamespace(rows=development, result_available_at=tuple(r.result_available_at.isoformat() for r in development))
        boundaries = expanding_window_boundaries(len(development), folds=folds, kickoffs=times, initial_training_rows=initial_training_rows)
        windows = []
        for left, right in zip(boundaries, boundaries[1:]):
            indices = training_indices(metadata, range(left), times[left])
            if not indices:
                raise ValueError('开发折开始时没有实际可用训练标签')
            windows.append({'training_rows': len(indices), 'validation_rows': right-left,
                            'fit_as_of': times[left], 'validation_last_kickoff': times[right-1]})
        def identity(rows):
            return content_digest([{'id': r.match_id, 'kickoff': r.kickoff.isoformat(),
                'available': r.result_available_at.isoformat(), 'home': r.home_team_id, 'away': r.away_team_id,
                'home_score': r.home_score, 'away_score': r.away_score} for r in rows])
        readiness = _comparison_readiness(development,
            tuple(r for r in source_records if end <= r.kickoff < final), cutoff=cutoff, boundaries=boundaries)
        report.update(readiness=readiness)
        report.update(state='admitted', folds=windows, validation_rows=len(development)-boundaries[0],
                      effective_initial_training_rows=boundaries[0],
                      development_sha256=identity(development),
                      validation_identity=content_digest([r.match_id for r in development[boundaries[0]:]]),
                      outer_identity=content_digest([r.match_id for r in outer]))
        return report, (records, development, outer)
    except ValueError as exc:
        report.update(code=type(exc).__name__, reason=str(exc))
        return report, None


def run_search_comparison(matches, *, as_of, evaluation_start, evaluation_end, final_holdout_start,
                          search_seeds=(11, 23, 37, 53, 71), model_seed=42, trials=12, folds=3,
                          arms=COMPARISON_ARMS, admit_only=False, prediction_sink=None, initial_training_rows=None):
    """Shared-start XGBoost ablation; each arm has a fresh model/feature history.

    Outer intervals belong to development. This entry never evaluates the final
    holdout and never saves or changes a production model. Cached full trials,
    LLM proposals and low-fidelity evaluations are deliberately absent.
    """
    import importlib
    from app.run_receipts import content_digest, build_run_receipt

    if (type(trials) is not int or trials < 4 or type(folds) is not int or folds < 2
            or type(model_seed) is not int or type(admit_only) is not bool
            or not isinstance(search_seeds, (list, tuple)) or not search_seeds
            or any(type(seed) is not int for seed in search_seeds) or len(set(search_seeds)) != len(search_seeds)
            or not isinstance(arms, (list, tuple)) or not arms or len(set(arms)) != len(arms)
            or any(arm not in COMPARISON_ARMS for arm in arms)):
        raise ValueError('非法搜索预算、种子或比较臂')
    if prediction_sink is not None and not callable(prediction_sink):
        raise ValueError('prediction_sink 必须可调用')
    admission, cohort = _comparison_cohort(tuple(matches), as_of=as_of, evaluation_start=evaluation_start,
        evaluation_end=evaluation_end, final_holdout_start=final_holdout_start, folds=folds,
        initial_training_rows=initial_training_rows)
    configuration = {**admission['configuration'], 'model': 'xgboost', 'model_seed': model_seed,
                     'search_seeds': list(search_seeds), 'trials': trials, 'arms': list(arms),
                     'initialization': 'declared_default_plus_three_common_random_points',
                     'evaluation_level': 'full', 'duplicate_policy': 'count_and_evaluate',
                     'timing_policy': 'observed_result', 'parallel_trials': 1,
                     'readiness_policy': 'observed_history_baseline_v1'}
    receipt = build_run_receipt('search-comparison', configuration=configuration,
        inputs={**{k: admission[k] for k in ('development_sha256', 'validation_identity', 'outer_identity') if k in admission},
                **({'readiness_sha256': admission['readiness']['sha256']} if 'sha256' in admission['readiness'] else {})},
        stages={'admission': admission['state'],
                'readiness': admission.get('readiness', {}).get('state', 'unavailable')})
    report = {'schema': 'SoccerAnalytics.SearchComparison.v1', 'status': admission['state'],
              'admission': admission, 'protocol': configuration, 'protocol_sha256': content_digest(configuration),
              'run_receipt': receipt, 'runs': [], 'adoption_decision': 'not_determined',
              'scope': 'Development outer interval only; final holdout not evaluated. Independent arms, same process with library imports before per-arm timers, no full-trial cache. Not an AgentTune reproduction.'}
    if admit_only or cohort is None:
        return report
    if admission['readiness']['state'] == 'insufficient':
        report.update(status='not_ready')
        return report
    # Fail on missing native/search dependencies before the first paid fit.
    for module in ('xgboost', 'skopt', 'optuna'):
        importlib.import_module(module)
    records, development, outer = cohort
    cutoff = aware_time(as_of)
    times = tuple(r.kickoff.isoformat() for r in development)
    expected_validation = tuple(r.match_id for r in development[expanding_window_boundaries(
        len(development), folds=folds, kickoffs=times, initial_training_rows=initial_training_rows)[0]:])
    for search_seed in search_seeds:
        initial = shared_initial_parameters('xgboost', seed=search_seed, include_features=True)
        for arm in arms:
            # Feature-only caching is isolated per arm; no model score reuse.
            cache, fit_events = {}, []
            objective_requests = 0
            def objective(params):
                nonlocal objective_requests
                objective_requests += 1
                model_params, features = _recipe(params)
                key = tuple(sorted(features.items()))
                if key not in cache:
                    cache[key] = feature_dataset(development, as_of=cutoff, **features)
                data = cache[key]
                left = expanding_window_boundaries(len(data.rows), folds=folds, kickoffs=data.kickoffs, initial_training_rows=initial_training_rows)[0]
                if tuple(data.match_ids[left:]) != expected_validation:
                    raise ValueError('不同配方的验证比赛集合不一致')
                return evaluate_expanding_window(data.rows, data.targets, FeatureEngineer.FEATURES,
                    estimator_factory('xgboost', seed=model_seed, params=model_params), folds=folds,
                    kickoffs=data.kickoffs, result_available_at=data.result_available_at,
                    fit_observer=lambda fold: fit_events.append(fold), initial_training_rows=initial_training_rows)
            optimizer = 'beam' if arm in ('beam1', 'beam2') else arm
            row = {'arm': arm, 'search_seed': search_seed, 'model_seed': model_seed,
                   'state': 'complete', 'outer_fit_attempts': 0}
            started = perf_counter()
            try:
                search = search_parameters('xgboost', optimizer, objective, trials=trials, seed=search_seed,
                    include_features=True, initial_points=initial,
                    beam_width=int(arm[-1]) if optimizer == 'beam' else None,
                    fit_counter=lambda: len(fit_events),
                    context={'development_sha256': admission['development_sha256'],
                             'validation_identity': admission['validation_identity'],
                             'model_seed': model_seed, 'protocol_sha256': report['protocol_sha256'],
                             'application_sha256': receipt['code']['application_sha256']})
                row['search'] = search
                row['selection_log_loss'] = search['best_loss']
                model_params, recipe = _recipe(search['best_params'])
                # Only the inner-selected configuration receives an outer fit.
                data = feature_dataset(records, as_of=cutoff, **recipe)
                lookup = {mid: i for i, mid in enumerate(data.match_ids)}
                training = [lookup[r.match_id] for r in development]
                evaluation = [lookup[r.match_id] for r in outer]
                model = estimator_factory('xgboost', seed=model_seed, params=model_params)(FeatureEngineer.FEATURES)
                row['outer_fit_attempts'] = 1
                outer_started = perf_counter()
                fit_estimator(model, tuple(data.rows[i] for i in training), tuple(data.targets[i] for i in training), kickoffs=times)
                probabilities = model.predict_proba(tuple(data.rows[i] for i in evaluation))
                targets = tuple(data.targets[i] for i in evaluation)
                row.update(outer_seconds=perf_counter()-outer_started,
                           outer_log_loss=multiclass_log_loss(targets, probabilities),
                           outer_brier_score=multiclass_brier_score(targets, probabilities),
                           outer_rows=len(evaluation))
                if prediction_sink is not None:
                    prediction_sink({'arm': arm, 'search_seed': search_seed,
                        'model_seed': model_seed, 'protocol_sha256': report['protocol_sha256'],
                        'development_sha256': admission['development_sha256'],
                        'outer_identity': admission['outer_identity'],
                        'match_ids': tuple(r.match_id for r in outer), 'targets': targets,
                        'probabilities': probabilities})
            except ModelConvergenceError as exc:
                row.update(state='failed', error=str(exc))
                if 'search' not in row:
                    row['search'] = exc.search
            row.update(elapsed_seconds=perf_counter()-started,
                       inner_fit_attempts=len(fit_events), objective_requests=objective_requests)
            report['runs'].append(row)
    report['status'] = 'complete' if all(r['state'] == 'complete' for r in report['runs']) else 'partial'
    report['budget'] = {'objective_requests': sum(r['objective_requests'] for r in report['runs']),
                        'inner_fit_attempts': sum(r['inner_fit_attempts'] for r in report['runs']),
                        'outer_fit_attempts': sum(r['outer_fit_attempts'] for r in report['runs']),
                        'cached_full_evaluations': 0}
    return report
