"""Reproducible optimization on development windows, followed by one final holdout."""

from __future__ import annotations

import math
import random
from time import perf_counter
from typing import Callable, Sequence

from app.ml.alternatives import ModelConvergenceError, estimator_factory
from app.ml.evaluation import TemporalEvaluation, evaluate_expanding_window, elo_baseline_predictions
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
    "elo_ordered_logit": [("ridge", 1e-5, 1.0, "log")],
    "lightgbm": [("num_leaves", 4, 31, "int"), ("learning_rate", .01, .2, "log"),
                 ("reg_lambda", .1, 10., "log")],
    "lightgbm_goss": [("num_leaves", 4, 31, "int"), ("learning_rate", .01, .2, "log"),
                      ("reg_lambda", .1, 10., "log")],
    "catboost": [("depth", 2, 6, "int"), ("learning_rate", .01, .2, "log"),
                 ("l2_leaf_reg", .1, 10., "log")],
}
OPTIMIZERS = ("bayesian", "genetic", "optuna")
FEATURE_SPACE = [
    ("feature_window", 2, 12, "int"),
    ("feature_elo_k", 5.0, 50.0, "log"),
    ("feature_home_advantage", 0.0, 150.0, "linear"),
]


def search_parameters(
    model: str,
    optimizer: str,
    objective: Callable[[dict], float | TemporalEvaluation],
    *,
    trials: int = 12,
    seed: int = 42,
    include_features: bool = False,
) -> dict:
    if model not in SPACES or optimizer not in OPTIMIZERS:
        raise ValueError("未知模型或优化方法")
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 4:
        raise ValueError("trials 必须是至少 4 的整数")
    rng = random.Random(seed)
    if type(include_features) is not bool:
        raise ValueError("include_features 必须是布尔值")
    feature_dimensions = [d for d in FEATURE_SPACE if model != 'elo_ordered_logit' or d[0] != 'feature_window']
    dimensions = SPACES[model] + (feature_dimensions if include_features else [])
    history = []
    search_started = perf_counter()
    population = []
    if optimizer == "bayesian":
        from skopt import Optimizer
        from skopt.space import Real

        search = Optimizer(
            [Real(0.0, 1.0) for _ in dimensions],
            base_estimator="GP",
            n_initial_points=min(4, trials - 1),
            acq_func="EI",
            random_state=seed,
        )
    elif optimizer == "optuna":
        import optuna

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(
                seed=seed, n_startup_trials=min(4, trials - 1), multivariate=False,
            ),
            pruner=optuna.pruners.NopPruner(),
        )
    generation_size = min(6, max(4, trials // 2))
    offspring = []
    for index in range(trials):
        if optimizer == "optuna":
            trial = study.ask()
            params = {
                name: (
                    trial.suggest_int(name, low, high)
                    if scale == "int"
                    else trial.suggest_float(name, low, high, log=scale == "log")
                )
                for name, low, high, scale in dimensions
            }
        elif optimizer == "bayesian":
            genes = search.ask()
        elif index < generation_size:
            genes = [rng.random() for _ in dimensions]
        else:
            if not offspring:
                # Elitist selection, uniform crossover, Gaussian mutation, generational replacement.
                parents = sorted(population, key=lambda item: item[0])[
                    : max(2, generation_size // 2)
                ]
                offspring = [parents[0][1][:]]
                while len(offspring) < generation_size:
                    mother, father = rng.sample(parents, 2)
                    child = [rng.choice((a, b)) for a, b in zip(mother[1], father[1])]
                    child = [
                        (
                            min(1.0, max(0.0, gene + rng.gauss(0, 0.15)))
                            if rng.random() < 0.4
                            else gene
                        )
                        for gene in child
                    ]
                    offspring.append(child)
                population = []
            genes = offspring.pop(0)
        if optimizer != "optuna":
            params = {}
            for gene, (name, low, high, scale) in zip(genes, dimensions):
                value = (
                    math.exp(math.log(low) + gene * math.log(high / low))
                    if scale == "log"
                    else low + gene * (high - low)
                )
                params[name] = int(round(value)) if scale == "int" else float(value)
        objective_started = perf_counter()
        try:
            evaluation = objective(params)
            loss = float(evaluation.log_loss if isinstance(evaluation, TemporalEvaluation) else evaluation)
        except ModelConvergenceError as exc:
            # Valid clipped multiclass log loss is at most -log(1e-15).
            # The penalty guides GP/GA only; it is never a selectable score.
            penalty = max(36.0, max((r['log_loss'] for r in history
                                     if r['log_loss'] is not None), default=0.0) + 1.0)
            failed = {"params": params, "log_loss": None, "state": "failed", "error": str(exc),
                      "objective_seconds": perf_counter() - objective_started}
            if optimizer == "optuna":
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
            else:
                failed["surrogate_penalty"] = penalty
                if optimizer == "bayesian":
                    search.tell(genes, penalty)
                else:
                    population.append((penalty, genes))
            history.append(failed)
            continue
        if not math.isfinite(loss):
            raise ValueError("搜索目标必须是有限数值")
        result = {"params": params, "log_loss": loss,
                  "objective_seconds": perf_counter() - objective_started}
        if isinstance(evaluation, TemporalEvaluation):
            result["development_folds"] = list(evaluation.fold_metrics)
        history.append(result)
        if optimizer == "bayesian":
            search.tell(genes, loss)
        elif optimizer == "optuna":
            study.tell(trial, loss)
        else:
            population.append((loss, genes))
    completed = [item for item in history if item["log_loss"] is not None]
    result = {
        "trials": history,
        "completed_trials": len(completed),
        "failed_trials": len(history) - len(completed),
        "elapsed_seconds": perf_counter() - search_started,
        "budget": {"requested_trials": trials, "attempted_trials": len(history),
                   "seed": seed, "parallel_trials": 1, "pruning": "none"},
        "timing_used_for_selection": False,
    }
    if not completed:
        raise ModelConvergenceError("搜索中没有收敛的参数配置", trials=history, search=result)
    best = min(completed, key=lambda item: item["log_loss"])
    return {**result, "best_params": best["params"], "best_loss": best["log_loss"]}


def chronological_split(dataset, fraction: float = 0.2) -> int:
    count = len(dataset.rows)
    if count < 60:
        raise ValueError("实验至少需要 60 场比赛")
    split = count - max(1, round(count * fraction))
    if dataset.kickoffs:
        while split > 0 and dataset.kickoffs[split - 1] == dataset.kickoffs[split]:
            split -= 1
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
) -> dict:
    split = chronological_split(dataset)
    development_rows, development_targets = (
        dataset.rows[:split],
        dataset.targets[:split],
    )
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
                estimator_factory(method, seed=seed, params=params),
                folds=3,
                kickoffs=dataset.kickoffs[:split],
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
                    method, optimizer, objective, trials=trials, seed=seed
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
            candidate["model"], seed=seed, params=candidate["params"]
        )(feature_names)
        fit_started = perf_counter()
        try:
            fit_estimator(estimator, development_rows, development_targets,
                          kickoffs=dataset.kickoffs[:split])
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
        "seed": seed,
        "search_budget_per_model_optimizer": trials,
        "timing_scope": "Wall elapsed seconds; search includes optimizer overhead, objectives include temporal folds; no speed ranking or timing-based selection.",
        "selection_rows": split,
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
