"""Reproducible optimization on development windows, followed by one final holdout."""

from __future__ import annotations

import math
import random
from time import perf_counter
from typing import Callable, Sequence

from app.ml.alternatives import ModelConvergenceError, estimator_factory
from app.ml.evaluation import TemporalEvaluation, evaluate_expanding_window, elo_baseline_predictions, chronological_holdout_boundary
from app.ml.model import (
    accuracy_score,
    fit_estimator,
    estimator_runtime,
    multiclass_brier_score,
    multiclass_log_loss,
    ranked_probability_score,
)

# All genes are normalized to [0,1]; log scales keep regularization searches useful.
SPACES = {
    "xgboost": [
        ("max_depth", 2, 6, "int"),
        ("learning_rate", 0.01, 0.2, "log"),
        ("l2", 0.1, 10.0, "log"),
    ],
    "sgd": [("alpha", 1e-5, 0.1, "log")],
    "random_forest": [("max_depth", 2, 12, "int"), ("min_samples_leaf", 2, 20, "int")],
    "gaussian_nb": [("var_smoothing", 1e-10, 1e-2, "log")],
    "logistic_l1": [("C", 1e-3, 100.0, "log")],
    "logistic_l2": [("C", 1e-3, 100.0, "log")],
    "logistic_elasticnet": [("C", 1e-3, 100.0, "log"), ("l1_ratio", 0.0, 1.0, "linear")],
    "ebm": [("learning_rate", .01, .1, "log"), ("reg_lambda", .1, 10., "log"),
            ("min_samples_leaf", 4, 20, "int")],
    "elo_ordered_logit": [("ridge", 1e-5, 1.0, "log")],
    "lightgbm": [("num_leaves", 4, 31, "int"), ("learning_rate", .01, .2, "log"),
                 ("reg_lambda", .1, 10., "log")],
    "lightgbm_goss": [("num_leaves", 4, 31, "int"), ("learning_rate", .01, .2, "log"),
                      ("reg_lambda", .1, 10., "log")],
    "catboost": [("depth", 2, 6, "int"), ("learning_rate", .01, .2, "log"),
                 ("l2_leaf_reg", .1, 10., "log")],
}
OPTIMIZERS = ("bayesian", "genetic", "optuna", "random", "beam")
FEATURE_SPACE = [
    ("feature_window", 2, 12, "int"),
    ("feature_elo_k", 5.0, 50.0, "log"),
    ("feature_home_advantage", 0.0, 150.0, "linear"),
]


def _search_dimensions(model, include_features):
    if model not in SPACES or type(include_features) is not bool:
        raise ValueError("未知模型或非法特征搜索开关")
    features = [d for d in FEATURE_SPACE if model != 'elo_ordered_logit' or d[0] != 'feature_window']
    return SPACES[model] + (features if include_features else [])


def _decode_genes(genes, dimensions):
    params = {}
    for gene, (name, low, high, scale) in zip(genes, dimensions):
        value = math.exp(math.log(low) + gene * math.log(high / low)) if scale == 'log' else low + gene * (high - low)
        value = min(high, max(low, value))  # exp(log(high)) can exceed a bound by one ULP.
        params[name] = int(round(value)) if scale == 'int' else float(value)
    return params


def _encode_parameters(params, dimensions):
    if not isinstance(params, dict) or set(params) != {d[0] for d in dimensions}:
        raise ValueError("初始配置必须完整匹配当前搜索空间")
    genes = []
    for name, low, high, scale in dimensions:
        value = params[name]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError("初始配置包含越界或非有限参数")
        if scale == 'int' and type(value) is not int:
            raise ValueError("离散初始参数必须是整数")
        genes.append(math.log(value / low) / math.log(high / low) if scale == 'log' else (value - low) / (high - low))
    return genes


def shared_initial_parameters(model, *, seed=42, include_features=False, count=4):
    """A declared reference plus common random points, never selected on outcomes."""
    references = {'xgboost': {'max_depth': 3, 'learning_rate': .04, 'l2': 1.},
                  'gaussian_nb': {'var_smoothing': 1e-9},
                  'logistic_l1': {'C': 1.}, 'logistic_l2': {'C': 1.},
                  'logistic_elasticnet': {'C': 1., 'l1_ratio': .5},
                  'sgd': {'alpha': 1e-4}, 'random_forest': {'max_depth': 6, 'min_samples_leaf': 2},
                  'ebm': {'learning_rate': .04, 'reg_lambda': 1., 'min_samples_leaf': 4},
                  'elo_ordered_logit': {'ridge': .001},
                  'lightgbm': {'num_leaves': 15, 'learning_rate': .04, 'reg_lambda': 1.},
                  'lightgbm_goss': {'num_leaves': 15, 'learning_rate': .04, 'reg_lambda': 1.},
                  'catboost': {'depth': 4, 'learning_rate': .04, 'l2_leaf_reg': 3.}}
    if model not in references or type(count) is not int or count < 1 or type(seed) is not int:
        raise ValueError("共同参考配置仅支持声明的模型与正整数数量")
    dimensions = _search_dimensions(model, include_features)
    first = references[model].copy()
    if include_features:
        first.update(feature_window=5, feature_elo_k=20., feature_home_advantage=80.)
    first = {name: first[name] for name, *_ in dimensions}
    _encode_parameters(first, dimensions)
    rng = random.Random(seed)
    points = [first]
    for _ in range(count - 1):
        points.append({name: rng.randint(low, high) if scale == 'int' else
                       math.exp(rng.uniform(math.log(low), math.log(high))) if scale == 'log' else rng.uniform(low, high)
                       for name, low, high, scale in dimensions})
    return points


def search_parameters(
    model: str,
    optimizer: str,
    objective: Callable[[dict], float | TemporalEvaluation],
    *,
    trials: int = 12,
    seed: int = 42,
    include_features: bool = False,
    initial_points=None,
    beam_width=None,
    fit_counter=None,
    context=None,
) -> dict:
    """Sequential full evaluations; duplicates and failed attempts consume budget.

    Explicit points count towards GP/TPE startup and GA's initial population.
    Beam is an opt-in local full-configuration ablation, not AgentTune's ranking.
    """
    from app.run_receipts import content_digest

    if optimizer not in OPTIMIZERS:
        raise ValueError("未知模型或优化方法")
    if type(trials) is not int or trials < 4 or type(seed) is not int:
        raise ValueError("trials 必须是至少 4 的整数，seed 必须是整数")
    dimensions = _search_dimensions(model, include_features)
    if fit_counter is not None and not callable(fit_counter):
        raise ValueError("fit_counter 必须可调用")
    if optimizer == 'beam':
        if type(beam_width) is not int or beam_width not in (1, 2) or initial_points is None:
            raise ValueError("beam 需要显式初始配置与 width=1 或 2")
    elif beam_width is not None:
        raise ValueError("beam_width 仅用于 beam 搜索")
    points = None
    if initial_points is not None:
        if not isinstance(initial_points, (list, tuple)) or not 1 <= len(initial_points) <= trials:
            raise ValueError("初始配置数量必须在尝试预算以内")
        points = [dict(p) if isinstance(p, dict) else p for p in initial_points]
        for point in points:
            _encode_parameters(point, dimensions)
        if len({content_digest(p) for p in points}) != len(points):
            raise ValueError("显式初始配置不能重复")
        if optimizer in ('genetic', 'beam') and len(points) < 2:
            raise ValueError("种群/beam 初始化至少需要两个配置")
    configuration = {'model': model, 'optimizer': optimizer, 'seed': seed, 'dimensions': dimensions,
                     'initial_points': points, 'beam_width': beam_width, 'context': context}
    context_id = content_digest(configuration)
    history, population, offspring, beam_parents, beam_rounds = [], [], [], [], []
    rng = random.Random(f"search-proposals:{seed}") if points is not None else random.Random(seed)
    started = perf_counter()
    startup = len(points) if points is not None else min(4, trials - 1)
    generation_size = len(points) if points is not None else min(6, max(4, trials // 2))
    if optimizer == 'bayesian':
        from skopt import Optimizer
        from skopt.space import Real
        search = Optimizer([Real(0., 1.) for _ in dimensions], base_estimator='GP',
                           n_initial_points=startup, acq_func='EI', random_state=seed)
    elif optimizer == 'optuna':
        import optuna
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(
            seed=seed, n_startup_trials=startup, multivariate=False), pruner=optuna.pruners.NopPruner())
        for point in points or ():
            study.enqueue_trial(point)
    seen = set()
    for index in range(trials):
        parent = None
        if optimizer == 'optuna':
            trial = study.ask()
            params = {name: trial.suggest_int(name, low, high) if scale == 'int' else
                      trial.suggest_float(name, low, high, log=scale == 'log')
                      for name, low, high, scale in dimensions}
        elif points is not None and index < len(points):
            params = points[index].copy()
            genes = _encode_parameters(params, dimensions)
        elif optimizer == 'bayesian':
            genes = search.ask()
            params = _decode_genes(genes, dimensions)
        elif optimizer == 'random':
            params = {name: rng.randint(low, high) if scale == 'int' else
                      math.exp(rng.uniform(math.log(low), math.log(high))) if scale == 'log' else rng.uniform(low, high)
                      for name, low, high, scale in dimensions}
        elif optimizer == 'beam':
            if not beam_parents:
                unique = {}
                for row in sorted((r for r in history if r['log_loss'] is not None), key=lambda r: (r['log_loss'], r['trial_index'])):
                    unique.setdefault(content_digest(row['params']), row)
                parents = list(unique.values())[:beam_width]
                if not parents:
                    break
                distance = math.sqrt(sum((a-b)**2 for a,b in zip(
                    _encode_parameters(parents[0]['params'], dimensions),
                    _encode_parameters(parents[-1]['params'], dimensions))))
                beam_rounds.append({'round_index': len(beam_rounds), 'parent_count': len(parents),
                                    'parent_trial_ids': [p['trial_id'] for p in parents],
                                    'parent_distance_unit_l2': distance})
                beam_parents = [parents[0], parents[-1]]
            parent = beam_parents.pop(0)
            genes = _encode_parameters(parent['params'], dimensions)
            genes = [min(1., max(0., gene + rng.gauss(0, .15))) for gene in genes]
            params = _decode_genes(genes, dimensions)
        else:
            if index < generation_size:
                genes = [rng.random() for _ in dimensions]
            else:
                if not offspring:
                    parents = sorted(population, key=lambda item: item[0])[:max(2, generation_size // 2)]
                    offspring = [parents[0][1][:]]
                    while len(offspring) < generation_size:
                        mother, father = rng.sample(parents, 2)
                        child = [rng.choice((a, b)) for a, b in zip(mother[1], father[1])]
                        child = [min(1., max(0., g + rng.gauss(0, .15))) if rng.random() < .4 else g for g in child]
                        offspring.append(child)
                    population = []
                genes = offspring.pop(0)
            params = _decode_genes(genes, dimensions)
        identity = content_digest(params)
        row = {'trial_index': index, 'trial_id': content_digest({'context': context_id, 'index': index, 'params': params}),
               'parent_trial_id': parent['trial_id'] if parent else None,
               'depth': parent['depth'] + 1 if parent else 0, 'params': params,
               'is_duplicate': identity in seen, 'state': 'complete', 'evaluation_level': 'full',
               'cache_hit': False}
        seen.add(identity)
        before_fits = fit_counter() if fit_counter else None
        if before_fits is not None and (type(before_fits) is not int or before_fits < 0):
            raise ValueError("拟合计数必须是非负整数")
        objective_started = perf_counter()
        try:
            evaluation = objective(dict(params))
            row['objective_seconds'] = perf_counter() - objective_started
            loss = float(evaluation.log_loss if isinstance(evaluation, TemporalEvaluation) else evaluation)
            if not math.isfinite(loss):
                raise ValueError("搜索目标必须是有限数值")
            row['log_loss'] = loss
            if isinstance(evaluation, TemporalEvaluation):
                row['development_folds'] = list(evaluation.fold_metrics)
        except ModelConvergenceError as exc:
            row['objective_seconds'] = perf_counter() - objective_started
            penalty = max(36., max((r['log_loss'] for r in history if r['log_loss'] is not None), default=0.) + 1.)
            row.update(log_loss=None, state='failed', error=str(exc))
            if optimizer == 'optuna':
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
            elif optimizer == 'bayesian':
                row['surrogate_penalty'] = penalty
                search.tell(genes, penalty)
            elif optimizer == 'genetic':
                row['surrogate_penalty'] = penalty
                population.append((penalty, genes))
        else:
            if optimizer == 'bayesian':
                search.tell(genes, loss)
            elif optimizer == 'optuna':
                study.tell(trial, loss)
            elif optimizer == 'genetic':
                population.append((loss, genes))
        finally:
            row.setdefault('objective_seconds', perf_counter() - objective_started)
            if fit_counter:
                after_fits = fit_counter()
                if type(after_fits) is not int or after_fits < before_fits:
                    raise ValueError("拟合计数必须单调递增")
                row['fit_attempts'] = after_fits - before_fits
            else:
                row['fit_attempts'] = len(row['development_folds']) if 'development_folds' in row else None
        row['elapsed_seconds'] = perf_counter() - started
        history.append(row)
    completed = [r for r in history if r['log_loss'] is not None]
    known_fit_counts = all(r['fit_attempts'] is not None for r in history)
    result = {'trials': history, 'completed_trials': len(completed), 'failed_trials': len(history) - len(completed),
              'elapsed_seconds': perf_counter() - started,
              'budget': {'requested_trials': trials, 'attempted_trials': len(history), 'seed': seed,
                         'parallel_trials': 1, 'pruning': 'none', 'proposed_candidates': len(history),
                         'unique_candidates': len(seen), 'invalid_candidates': 0, 'duplicate_candidates': len(history) - len(seen),
                         'objective_requests': len(history), 'new_full_evaluations': len(completed),
                         'cached_full_evaluations': 0, 'fit_attempts': sum(r['fit_attempts'] for r in history) if known_fit_counts else None},
              'initialization': {'policy': 'explicit_shared' if points is not None else 'legacy_sampler',
                                 'points': points, 'gp_tpe_startup': startup if optimizer in ('bayesian', 'optuna') else None,
                                 'ga_population': generation_size if optimizer == 'genetic' else None},
              'protocol': {'optimizer': optimizer, 'context_sha256': context_id,
                           'beam_width': beam_width, 'beam_children_per_round': 2 if optimizer == 'beam' else None,
                           'beam_step_in_unit_space': .15 if optimizer == 'beam' else None,
                           'duplicate_policy': 'count_and_evaluate',
                           'proposal_stream': 'separate_from_initialization' if points is not None else 'legacy',
                           'timing_used_for_selection': False},
              'timing_used_for_selection': False}
    if optimizer == 'beam':
        result['beam_rounds'] = beam_rounds
    result['search_overhead_seconds'] = max(0., result['elapsed_seconds'] - sum(r['objective_seconds'] for r in history))
    result['optimizer_fit_attempts'] = None  # Native surrogate internals are not predictive-model fits.
    if not completed:
        raise ModelConvergenceError("搜索中没有收敛的参数配置", trials=history, search=result)
    best = min(completed, key=lambda r: (r['log_loss'], r['trial_index']))
    return {**result, 'best_params': best['params'], 'best_loss': best['log_loss']}


def chronological_split(dataset, fraction: float = 0.2) -> int:
    count = len(dataset.rows)
    if count < 60:
        raise ValueError("实验至少需要 60 场比赛")
    split = chronological_holdout_boundary(count, fraction=fraction, kickoffs=dataset.kickoffs)
    if split < 30:
        raise ValueError("独立开球时间不足以形成开发集和留出集")
    return split


def run_experiments(
    dataset,
    feature_names: tuple[str, ...],
    *,
    methods: Sequence[str] = ("xgboost", "sgd", "random_forest", "gaussian_nb"),
    optimizers: Sequence[str] = ("bayesian", "genetic"),
    trials: int = 12,
    seed: int = 42,
    shared_start: bool = False,
    model_seed: int | None = None,
    beam_width: int | None = None,
) -> dict:
    if type(shared_start) is not bool or (model_seed is not None and type(model_seed) is not int):
        raise ValueError("非法模型种子或共同起点开关")
    if "beam" in optimizers and (type(beam_width) is not int or beam_width not in (1, 2)):
        raise ValueError("beam 需要显式 width=1 或 2")
    if "beam" not in optimizers and beam_width is not None:
        raise ValueError("beam_width 仅用于 beam 搜索")
    model_seed = seed if model_seed is None else model_seed
    fit_events = []
    split = chronological_split(dataset)
    from app.ml.availability import training_indices
    eligible = training_indices(dataset, range(split), dataset.kickoffs[split]) if getattr(dataset, "result_available_at", ()) else list(range(split))
    if len(eligible) < 30:
        raise ValueError("最终留出前实际可用训练标签不足 30 场")
    development_rows, development_targets = (
        tuple(dataset.rows[i] for i in eligible),
        tuple(dataset.targets[i] for i in eligible),
    )
    development_times = tuple(dataset.kickoffs[i] for i in eligible) if dataset.kickoffs else ()
    timing = ({"result_available_at": tuple(dataset.result_available_at[i] for i in eligible)}
              if getattr(dataset, "result_available_at", ()) else {})
    holdout_rows, holdout_targets = dataset.rows[split:], dataset.targets[split:]
    if (
        not methods
        or any(m not in SPACES for m in methods)
        or any(o not in OPTIMIZERS for o in optimizers)
    ):
        raise ValueError("未知或空的实验模型/优化方法")
    candidates = []
    for method in dict.fromkeys(methods):

        def objective(params):
            return evaluate_expanding_window(
                development_rows,
                development_targets,
                feature_names,
                estimator_factory(method, seed=model_seed, params=params),
                folds=3,
                kickoffs=development_times, fit_observer=lambda fold: fit_events.append(fold),
                **timing,
            )

        default = {"model": method, "optimizer": "default", "params": {}}
        default_started = perf_counter()
        try:
            evaluation = objective({})
            default["selection_log_loss"] = evaluation.log_loss
            default["development_folds"] = list(evaluation.fold_metrics)
        except ModelConvergenceError as exc:
            default.update(selection_log_loss=None, selection_error=str(exc))
        default["development_objective_seconds"] = perf_counter() - default_started
        candidates.append(default)
        for optimizer in dict.fromkeys(optimizers):
            try:
                search = search_parameters(
                    method, optimizer, objective, trials=trials, seed=seed,
                    initial_points=shared_initial_parameters(method, seed=seed) if shared_start or optimizer == "beam" else None,
                    beam_width=beam_width if optimizer == "beam" else None, fit_counter=lambda: len(fit_events),
                )
            except ModelConvergenceError as exc:
                candidates.append({"model": method, "optimizer": optimizer, "params": None,
                                   "selection_log_loss": None, "selection_error": str(exc),
                                   "search": exc.search})
                continue
            candidates.append(
                {
                    "model": method,
                    "optimizer": optimizer,
                    "params": search["best_params"],
                    "selection_log_loss": search["best_loss"],
                    "search": search,
                }
            )
    valid = [item for item in candidates if item["selection_log_loss"] is not None]
    if not valid:
        raise ModelConvergenceError("所有实验配置均未收敛")
    selected = min(valid, key=lambda item: item["selection_log_loss"])
    # Selection is locked before any candidate is evaluated on final holdout.
    selected_identity = {
        "model": selected["model"],
        "optimizer": selected["optimizer"],
        "params": selected["params"],
    }
    for candidate in candidates:
        if candidate["selection_log_loss"] is None:
            continue
        estimator = estimator_factory(
            candidate["model"], seed=model_seed, params=candidate["params"]
        )(feature_names)
        fit_started = perf_counter()
        try:
            fit_estimator(estimator, development_rows, development_targets,
                          kickoffs=development_times)
        except ModelConvergenceError as exc:
            candidate['holdout_error'] = str(exc)
            candidate['runtime'] = {**estimator_runtime(estimator),
                                    'final_fit_seconds': perf_counter() - fit_started,
                                    'holdout_inference_seconds': None}
            continue
        fit_seconds = perf_counter() - fit_started
        inference_started = perf_counter()
        probabilities = estimator.predict_proba(holdout_rows)
        candidate["runtime"] = {**estimator_runtime(estimator),
                                "final_fit_seconds": fit_seconds,
                                "holdout_inference_seconds": perf_counter() - inference_started}
        candidate.update(
            {
                "holdout_rows": len(holdout_rows),
                "holdout_accuracy": accuracy_score(
                    holdout_targets, [max(range(3), key=row.__getitem__) for row in probabilities]
                ),
                "holdout_log_loss": multiclass_log_loss(holdout_targets, probabilities),
                "holdout_brier_score": multiclass_brier_score(
                    holdout_targets, probabilities
                ),
                "holdout_rps": ranked_probability_score(holdout_targets, probabilities),
            }
        )
    from collections import Counter

    majority = Counter(development_targets).most_common(1)[0][0]
    return {
        "seed": seed, "model_seed": model_seed, "shared_start": shared_start,
        "search_budget_per_model_optimizer": trials,
        "timing_scope": "Wall elapsed seconds; search includes optimizer overhead, objectives include temporal folds; no speed ranking or timing-based selection.",
        "selection_rows": len(development_rows),
        **({"labels_purged_at_holdout": split - len(eligible)} if timing else {}),
        "holdout_rows": len(holdout_rows),
        "selected_on": "expanding_window_log_loss",
        "selected": selected_identity,
        "results": candidates,
        "baselines": {
            "majority_accuracy": accuracy_score(
                holdout_targets, [majority] * len(holdout_targets)
            ),
            "elo_accuracy": accuracy_score(
                holdout_targets,
                elo_baseline_predictions(
                    development_rows, development_targets, holdout_rows
                ),
            ),
        },
        "note": "Exploratory comparison only; no production model is saved and release gates remain required.",
    }
