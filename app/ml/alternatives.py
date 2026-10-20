"""Alternative estimators sharing the existing dictionary-row / L,D,W contract."""

from __future__ import annotations

from typing import Mapping, Sequence
import warnings

import numpy as np

from app.ml.model import ModelTrainingError, NotFittedError, _validate_classes

METHODS = (
    "xgboost", "sgd", "random_forest", "gaussian_nb",
    "logistic_l1", "logistic_l2", "logistic_elasticnet",
    "elo_ordered_logit",
    "lightgbm", "lightgbm_goss", "catboost",
)


class ModelConvergenceError(ModelTrainingError):
    def __init__(self, message: str, *, trials=None, search=None):
        super().__init__(message)
        self.trials = trials
        self.search = search


class SklearnClassifier:
    explanation_method = "unavailable"

    def __init__(
        self,
        feature_names: tuple[str, ...],
        *,
        method: str = "sgd",
        seed: int = 42,
        **params,
    ):
        import sklearn
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression, SGDClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        if not feature_names or len(set(feature_names)) != len(feature_names):
            raise ModelTrainingError("feature_names 必须非空且唯一")
        self.feature_names = feature_names
        self.seed = seed
        self.technology_version = sklearn.__version__
        self.technology = method
        self.requires_kickoffs = method == "catboost"
        self.execution_policy = {"threads": 1 if method in {"random_forest", "lightgbm", "lightgbm_goss", "catboost"}
                                 else "library_default", "input": "existing_numeric_features_only"}
        if method == "sgd":
            estimator = SGDClassifier(
                **{
                    "loss": "log_loss",
                    "max_iter": 3000,
                    "tol": 1e-5,
                    "random_state": seed,
                    **params,
                }
            )
        elif method == "random_forest":
            estimator = RandomForestClassifier(
                **{"n_estimators": 100, "n_jobs": 1, "random_state": seed, **params}
            )
        elif method == "gaussian_nb":
            estimator = GaussianNB(**params)
        elif method in {"lightgbm", "lightgbm_goss"}:
            import lightgbm

            fixed = {"objective": "multiclass", "n_jobs": 1, "random_state": seed,
                     "deterministic": True, "force_col_wise": True, "verbosity": -1,
                     "boosting_type": "gbdt",
                     "data_sample_strategy": "goss" if method == "lightgbm_goss" else "bagging"}
            allowed = {"n_estimators", "num_leaves", "max_depth", "learning_rate",
                       "min_child_samples", "reg_alpha", "reg_lambda"}
            if set(params) - allowed:
                raise ModelTrainingError("LightGBM 仅接受声明的数值树参数，线程与采样策略固定")
            estimator = lightgbm.LGBMClassifier(**{"n_estimators": 160, "num_leaves": 15,
                                                "learning_rate": .04, "reg_lambda": 1.,
                                                **params, **fixed})
            self.technology_version = lightgbm.__version__
            self.execution_policy.update(data_sample_strategy=fixed["data_sample_strategy"],
                                         deterministic=True, force_col_wise=True)
        elif method == "catboost":
            import catboost

            allowed = {"iterations", "depth", "learning_rate", "l2_leaf_reg", "random_strength"}
            if set(params) - allowed:
                raise ModelTrainingError("CatBoost 仅接受声明的数值树参数，时间与资源策略固定")
            estimator = catboost.CatBoostClassifier(
                **{"iterations": 160, "depth": 4, "learning_rate": .04, "l2_leaf_reg": 3.,
                   **params}, loss_function="MultiClass", classes_count=3,
                boosting_type="Ordered", has_time=True, thread_count=1,
                random_seed=seed, allow_writing_files=False, task_type="CPU", verbose=False)
            self.technology_version = catboost.__version__
            self.execution_policy.update(boosting_type="Ordered", has_time=True,
                                         training_order="validated_nondecreasing_UTC_kickoff",
                                         allow_writing_files=False, prediction_threads=1,
                                         categorical_features_used=False)
        elif method in {"logistic_l1", "logistic_l2", "logistic_elasticnet"}:
            ratio = {"logistic_l1": 1.0, "logistic_l2": 0.0, "logistic_elasticnet": 0.5}[method]
            if "penalty" in params or (
                method != "logistic_elasticnet" and params.get("l1_ratio", ratio) != ratio
            ):
                raise ModelTrainingError("使用 l1_ratio 配置正则化；L1/L2 模型的 l1_ratio 固定")
            estimator = LogisticRegression(
                **{
                    "solver": "lbfgs" if method == "logistic_l2" else "saga",
                    "l1_ratio": ratio, "C": 1.0,
                    "max_iter": 20000, "tol": 1e-4, "random_state": seed,
                    **params,
                }
            )
        else:
            raise ModelTrainingError(f"未知模型 {method}")
        # Scaling is fit only on the fit partition, never the holdout.
        self.pipeline = (make_pipeline(estimator) if method in {"lightgbm", "lightgbm_goss", "catboost"}
                         else make_pipeline(StandardScaler(), estimator))
        options = estimator.get_params()
        self.execution_policy["configured_iteration_cap"] = next(
            (options[name] for name in ("iterations", "n_estimators", "max_iter") if name in options), None)
        self.is_fitted_ = False

    def _matrix(self, rows: Sequence[Mapping[str, float]]) -> np.ndarray:
        try:
            if any(
                isinstance(row[name], bool)
                for row in rows
                for name in self.feature_names
            ):
                raise ValueError("boolean feature")
            matrix = np.asarray(
                [[row[name] for name in self.feature_names] for row in rows],
                dtype=float,
            )
            if not len(rows) or not np.isfinite(matrix).all():
                raise ValueError("empty or nonfinite features")
        except (KeyError, ValueError, TypeError) as exc:
            raise ModelTrainingError(f"非法特征: {exc}") from exc
        return matrix

    def fit(self, rows, targets, *, kickoffs=()):
        self.is_fitted_ = False
        if not rows or len(rows) != len(targets):
            raise ModelTrainingError("训练行与目标必须非空且长度一致")
        _validate_classes(targets, "目标")
        if len(set(targets)) < 2:
            raise ModelTrainingError("至少需要两个目标类别")
        if getattr(self, 'requires_kickoffs', False):
            from datetime import UTC, datetime
            from collections.abc import Sequence

            if isinstance(kickoffs, (str, bytes)) or not isinstance(kickoffs, Sequence) or len(kickoffs) != len(rows):
                raise ModelTrainingError("CatBoost Ordered 需要与训练行对应的开球时间")
            try:
                times = [datetime.fromisoformat(value) for value in kickoffs]
                if any(t.tzinfo is None or t.utcoffset() is None for t in times):
                    raise ValueError("missing timezone")
                times = [t.astimezone(UTC) for t in times]
                if times != sorted(times):
                    raise ValueError("unordered kickoff")
            except (TypeError, ValueError, OverflowError) as exc:
                raise ModelTrainingError("CatBoost 输入必须按带时区的开球时间排序") from exc
        if self.technology.startswith("logistic_"):
            from sklearn.exceptions import ConvergenceWarning

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    self.pipeline.fit(self._matrix(rows), list(targets))
            except ConvergenceWarning as exc:
                raise ModelConvergenceError("逻辑回归未收敛，请检查参数或增加 max_iter") from exc
        elif self.technology in {"lightgbm", "lightgbm_goss", "catboost"}:
            self.pipeline[-1].fit(self._matrix(rows), list(targets))
        else:
            self.pipeline.fit(self._matrix(rows), list(targets))
        self.is_fitted_ = True
        return self

    def predict_proba(self, rows):
        if not self.is_fitted_:
            raise NotFittedError("模型尚未训练")
        matrix = self._matrix(rows)
        if self.technology == "catboost":
            raw = self.pipeline[-1].predict_proba(matrix, thread_count=1)
        elif self.technology.startswith("lightgbm"):
            raw = self.pipeline[-1].predict_proba(matrix, num_threads=1)
        else:
            raw = self.pipeline.predict_proba(matrix)
        probabilities = np.zeros((len(rows), 3))
        classes = self.pipeline[-1].classes_
        if raw.shape != (len(rows), len(classes)):
            raise ModelTrainingError("模型概率列与类别标识不一致")
        for column, label in enumerate(classes):
            probabilities[:, int(label)] = raw[:, column]
        if not np.isfinite(probabilities).all():
            raise ModelTrainingError("模型输出非有限概率")
        totals = probabilities.sum(axis=1, keepdims=True)
        if (probabilities < 0).any() or (totals <= 0).any():
            raise ModelTrainingError("模型输出非法概率")
        # Gaussian log-likelihood normalization can lose precision when extrapolating.
        probabilities /= totals
        return probabilities.tolist()

    def predict(self, rows):
        return [max(range(3), key=row.__getitem__) for row in self.predict_proba(rows)]


def estimator_factory(method: str, *, seed: int = 42, params: dict | None = None):
    if method not in METHODS:
        raise ValueError(f"未知模型 {method}")

    def create(features):
        if method == "elo_ordered_logit":
            from app.ml.ordinal import EloOrderedLogit

            return EloOrderedLogit(features, **(params or {}))
        if method == "xgboost":
            from app.ml.model import XGBoostClassifier

            return XGBoostClassifier(features, seed=seed, **(params or {}))
        return SklearnClassifier(features, method=method, seed=seed, **(params or {}))

    return create
