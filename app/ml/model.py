from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import xgboost as xgb


class ModelTrainingError(ValueError):
    """训练数据或模型状态不满足概率分类契约。"""


class NotFittedError(ModelTrainingError):
    """模型尚未完成训练。"""


def _argmax(values: Sequence[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


class XGBoostClassifier:
    """时间感知的三分类 XGBoost 封装，输出概率与精确 TreeSHAP 边际贡献。"""

    CLASS_COUNT = 3
    technology = "xgboost"
    technology_version = xgb.__version__
    explanation_method = "tree_shap_margin"

    def __init__(
        self,
        feature_names: tuple[str, ...],
        *,
        rounds: int = 160,
        early_stopping_rounds: int = 20,
        max_depth: int = 3,
        learning_rate: float = 0.04,
        subsample: float = 0.85,
        column_subsample: float = 0.9,
        minimum_child_weight: float = 2.0,
        l1: float = 0.02,
        l2: float = 1.0,
        seed: int = 20260817,
    ) -> None:
        if not feature_names or len(set(feature_names)) != len(feature_names):
            raise ModelTrainingError("feature_names 必须非空且唯一")
        integer_parameters = (rounds, early_stopping_rounds, max_depth)
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in integer_parameters
        ):
            raise ModelTrainingError("训练轮数、提前停止轮数与树深必须为正整数")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ModelTrainingError("seed 必须是非负整数")
        numeric_parameters = (
            learning_rate,
            subsample,
            column_subsample,
            minimum_child_weight,
            l1,
            l2,
        )
        try:
            numeric = tuple(
                float(value)
                for value in numeric_parameters
                if not isinstance(value, bool)
            )
        except (TypeError, ValueError) as exc:
            raise ModelTrainingError("XGBoost 超参数必须是有限数值") from exc
        if len(numeric) != len(numeric_parameters) or any(
            not math.isfinite(value) for value in numeric
        ):
            raise ModelTrainingError("XGBoost 超参数必须是有限数值")
        (
            learning_rate,
            subsample,
            column_subsample,
            minimum_child_weight,
            l1,
            l2,
        ) = numeric
        if (
            learning_rate <= 0
            or not 0 < subsample <= 1
            or not 0 < column_subsample <= 1
            or minimum_child_weight < 0
            or l1 < 0
            or l2 < 0
        ):
            raise ModelTrainingError("XGBoost 超参数超出允许范围")

        self.feature_names = tuple(feature_names)
        self.rounds = int(rounds)
        self.early_stopping_rounds = int(early_stopping_rounds)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.subsample = float(subsample)
        self.column_subsample = float(column_subsample)
        self.minimum_child_weight = float(minimum_child_weight)
        self.l1 = float(l1)
        self.l2 = float(l2)
        self.seed = int(seed)
        self.booster_: xgb.Booster | None = None
        self.rounds_used_: int = 0
        self.training_loss_: float | None = None
        self.is_fitted_: bool = False

    def _matrix(self, rows: Sequence[Mapping[str, float]]) -> list[list[float]]:
        if not rows:
            raise ModelTrainingError("特征行不能为空")
        matrix: list[list[float]] = []
        for row_index, row in enumerate(rows):
            values: list[float] = []
            for name in self.feature_names:
                if name not in row:
                    raise ModelTrainingError(f"第 {row_index} 行缺少特征 {name}")
                value = row[name]
                if isinstance(value, bool):
                    raise ModelTrainingError(f"特征 {name} 不能是布尔值")
                try:
                    numeric = float(value)
                except (TypeError, ValueError) as exc:
                    raise ModelTrainingError(f"特征 {name} 不是数值") from exc
                if not math.isfinite(numeric) or abs(numeric) > 3e38:
                    raise ModelTrainingError(
                        f"特征 {name} 必须是 XGBoost 可表示的有限数值"
                    )
                values.append(numeric)
            matrix.append(values)
        return matrix

    def _dmatrix(
        self,
        rows: Sequence[Mapping[str, float]],
        targets: Sequence[int] | None = None,
        weights: Sequence[float] | None = None,
    ) -> xgb.DMatrix:
        return xgb.DMatrix(
            self._matrix(rows),
            label=None if targets is None else list(targets),
            weight=None if weights is None else list(weights),
            feature_names=list(self.feature_names),
        )

    def _parameters(self) -> dict[str, Any]:
        return {
            "objective": "multi:softprob",
            "num_class": self.CLASS_COUNT,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "max_depth": self.max_depth,
            "eta": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.column_subsample,
            "min_child_weight": self.minimum_child_weight,
            "alpha": self.l1,
            "lambda": self.l2,
            "seed": self.seed,
            "nthread": 1,
            "verbosity": 0,
        }

    @staticmethod
    def _sample_weights(targets: Sequence[int]) -> list[float]:
        counts = [targets.count(index) for index in range(3)]
        row_count = len(targets)
        class_weights = [
            row_count / (3 * count) if count else 0.0
            for count in counts
        ]
        return [class_weights[target] for target in targets]

    def fit(
        self,
        rows: Sequence[Mapping[str, float]],
        targets: Sequence[int],
    ) -> XGBoostClassifier:
        if not rows or len(rows) != len(targets):
            raise ModelTrainingError("训练行与目标必须非空且长度一致")
        _validate_classes(targets, "目标")
        weights = self._sample_weights(targets)
        rounds_used = self.rounds

        if len(rows) >= 15:
            validation_count = max(3, int(round(len(rows) * 0.2)))
            split = len(rows) - validation_count
            training = self._dmatrix(
                rows[:split],
                targets[:split],
                weights[:split],
            )
            validation = self._dmatrix(
                rows[split:],
                targets[split:],
                weights[split:],
            )
            probe = xgb.train(
                self._parameters(),
                training,
                num_boost_round=self.rounds,
                evals=[(validation, "validation")],
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval=False,
            )
            rounds_used = min(self.rounds, int(probe.best_iteration) + 1)

        evaluations: dict[str, dict[str, list[float]]] = {}
        full = self._dmatrix(rows, targets, weights)
        self.booster_ = xgb.train(
            self._parameters(),
            full,
            num_boost_round=rounds_used,
            evals=[(full, "training")],
            evals_result=evaluations,
            verbose_eval=False,
        )
        losses = evaluations.get("training", {}).get("mlogloss", [])
        self.training_loss_ = float(losses[-1]) if losses else None
        self.rounds_used_ = rounds_used
        self.is_fitted_ = True
        return self

    def _booster(self) -> xgb.Booster:
        if not self.is_fitted_ or self.booster_ is None:
            raise NotFittedError("模型尚未训练")
        return self.booster_

    def predict_proba(
        self,
        rows: Sequence[Mapping[str, float]],
    ) -> list[list[float]]:
        raw = self._booster().predict(
            self._dmatrix(rows),
            strict_shape=True,
        )
        probabilities = [
            [float(value) for value in probability_row]
            for probability_row in raw.tolist()
        ]
        _validate_probability_rows(probabilities)
        return probabilities

    def predict(self, rows: Sequence[Mapping[str, float]]) -> list[int]:
        return [_argmax(probabilities) for probabilities in self.predict_proba(rows)]

    def explain_prediction(
        self,
        row: Mapping[str, float],
        *,
        top_n: int = 3,
    ) -> dict[str, Any]:
        if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0:
            raise ModelTrainingError("top_n 必须是正整数")
        matrix = self._matrix([row])
        sample = xgb.DMatrix(matrix, feature_names=list(self.feature_names))
        probabilities = self.predict_proba([row])[0]
        ranked = sorted(
            range(self.CLASS_COUNT),
            key=probabilities.__getitem__,
            reverse=True,
        )
        predicted, runner_up = ranked[:2]
        contributions = self._booster().predict(
            sample,
            pred_contribs=True,
            strict_shape=True,
        )
        shape = tuple(int(value) for value in contributions.shape)
        expected_shape = (1, self.CLASS_COUNT, len(self.feature_names) + 1)
        if shape != expected_shape:
            raise ModelTrainingError(
                f"TreeSHAP 形状非法: {shape}，期望 {expected_shape}"
            )
        winner = contributions[0][predicted]
        challenger = contributions[0][runner_up]
        drivers = [
            {
                "feature": name,
                "raw_value": matrix[0][index],
                "contribution": float(winner[index] - challenger[index]),
            }
            for index, name in enumerate(self.feature_names)
        ]
        for item in drivers:
            item["direction"] = (
                "supports_prediction"
                if float(item["contribution"]) >= 0.0
                else "opposes_prediction"
            )
        bias_contribution = float(winner[-1] - challenger[-1])
        logit_margin = bias_contribution + math.fsum(
            float(item["contribution"]) for item in drivers
        )
        return {
            "method": self.explanation_method,
            "predicted_class_index": predicted,
            "runner_up_class_index": runner_up,
            "logit_margin": logit_margin,
            "bias_contribution": bias_contribution,
            "drivers": drivers,
            "top_drivers": sorted(
                drivers,
                key=lambda item: abs(float(item["contribution"])),
                reverse=True,
            )[:top_n],
        }


def _validate_classes(values: Sequence[int], label: str) -> None:
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1, 2)
        for value in values
    ):
        raise ModelTrainingError(f"{label}类别只能是整数 0、1、2")


def _validate_metric_inputs(
    targets: Sequence[int],
    values: Sequence[Sequence[float]] | Sequence[int],
) -> None:
    if not targets or len(targets) != len(values):
        raise ModelTrainingError("指标输入必须非空且长度一致")
    _validate_classes(targets, "真实")


def _validate_probability_rows(probabilities: Sequence[Sequence[float]]) -> None:
    for row in probabilities:
        if len(row) != 3:
            raise ModelTrainingError("概率指标只接受三分类输入")
        normalized: list[float] = []
        for value in row:
            if isinstance(value, bool):
                raise ModelTrainingError("概率必须是有限数值")
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ModelTrainingError("概率必须是有限数值") from exc
            if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
                raise ModelTrainingError("概率必须位于 [0, 1]")
            normalized.append(numeric)
        if not math.isclose(math.fsum(normalized), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ModelTrainingError("每行概率之和必须为 1")


def accuracy_score(targets: Sequence[int], predictions: Sequence[int]) -> float:
    _validate_metric_inputs(targets, predictions)
    _validate_classes(predictions, "预测")
    return sum(
        actual == predicted for actual, predicted in zip(targets, predictions)
    ) / len(targets)


def multiclass_log_loss(
    targets: Sequence[int],
    probabilities: Sequence[Sequence[float]],
) -> float:
    _validate_metric_inputs(targets, probabilities)
    _validate_probability_rows(probabilities)
    return -math.fsum(
        math.log(max(float(row[target]), 1e-15))
        for target, row in zip(targets, probabilities)
    ) / len(targets)


def multiclass_brier_score(
    targets: Sequence[int],
    probabilities: Sequence[Sequence[float]],
) -> float:
    _validate_metric_inputs(targets, probabilities)
    _validate_probability_rows(probabilities)
    return math.fsum(
        (float(probability) - (1.0 if class_index == target else 0.0)) ** 2
        for target, row in zip(targets, probabilities)
        for class_index, probability in enumerate(row)
    ) / len(targets)


def ranked_probability_score(
    targets: Sequence[int],
    probabilities: Sequence[Sequence[float]],
) -> float:
    """计算有序三分类 RPS；比分越低，概率排序质量越好。"""
    _validate_metric_inputs(targets, probabilities)
    _validate_probability_rows(probabilities)
    return math.fsum(
        math.fsum(
            (
                math.fsum(float(value) for value in row[: boundary + 1])
                - (1.0 if target <= boundary else 0.0)
            )
            ** 2
            for boundary in range(2)
        )
        / 2.0
        for target, row in zip(targets, probabilities)
    ) / len(targets)
