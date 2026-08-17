from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from app.ml.model import (
    accuracy_score,
    multiclass_brier_score,
    multiclass_log_loss,
    ranked_probability_score,
)


class EvaluationError(ValueError):
    """时间序列评估参数不足以形成可信的扩展窗口。"""


@dataclass(frozen=True)
class TemporalEvaluation:
    fold_count: int
    evaluated_rows: int
    accuracy: float
    baseline_accuracy: float
    majority_baseline_accuracy: float
    elo_baseline_accuracy: float
    log_loss: float
    brier_score: float
    ranked_probability_score: float
    calibration_error: float
    classwise_calibration_error: float
    sharpness: float
    reliability: tuple[dict[str, Any], ...]
    fold_metrics: tuple[dict[str, Any], ...]


EstimatorFactory = Callable[[tuple[str, ...]], Any]


def elo_baseline_predictions(
    training_rows: Sequence[Mapping[str, float]],
    training_targets: Sequence[int],
    validation_rows: Sequence[Mapping[str, float]],
) -> list[int]:
    """只用历史 Elo 差值拟合三分类阈值，作为可部署的领域基线。"""
    if not training_rows or len(training_rows) != len(training_targets):
        raise EvaluationError("Elo 基线训练行与目标必须非空且长度一致")
    majority_class = Counter(training_targets).most_common(1)[0][0]
    try:
        training_diffs = [float(row["elo_diff"]) for row in training_rows]
        validation_diffs = [float(row["elo_diff"]) for row in validation_rows]
    except (KeyError, TypeError, ValueError):
        return [majority_class] * len(validation_rows)
    if any(not math.isfinite(value) for value in training_diffs + validation_diffs):
        return [majority_class] * len(validation_rows)

    def classify(value: float, threshold: float) -> int:
        if value < -threshold:
            return 0
        if value > threshold:
            return 2
        return 1

    candidates = sorted({0.0, *(abs(value) for value in training_diffs)})
    threshold = max(
        candidates,
        key=lambda candidate: (
            sum(
                classify(value, candidate) == target
                for value, target in zip(training_diffs, training_targets)
            ),
            -candidate,
        ),
    )
    return [classify(value, threshold) for value in validation_diffs]


def classwise_reliability(
    targets: Sequence[int],
    probabilities: Sequence[Sequence[float]],
    *,
    bins: int = 10,
) -> tuple[dict[str, Any], ...]:
    """按客胜、平局、主胜分别统计 one-vs-rest 可靠性。"""
    if isinstance(bins, bool) or not isinstance(bins, int) or bins <= 0:
        raise EvaluationError("bins 必须是正整数")
    multiclass_brier_score(targets, probabilities)
    summaries: list[dict[str, Any]] = []
    for class_index in range(3):
        buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
        for target, row in zip(targets, probabilities):
            probability = float(row[class_index])
            bucket_index = min(bins - 1, int(probability * bins))
            buckets[bucket_index].append((probability, target == class_index))
        populated = []
        for bucket_index, bucket in enumerate(buckets):
            if not bucket:
                continue
            populated.append(
                {
                    "lower_bound": bucket_index / bins,
                    "upper_bound": (bucket_index + 1) / bins,
                    "count": len(bucket),
                    "mean_probability": math.fsum(
                        probability for probability, _ in bucket
                    )
                    / len(bucket),
                    "observed_rate": sum(
                        observed for _, observed in bucket
                    )
                    / len(bucket),
                }
            )
        calibration_error = math.fsum(
            (item["count"] / len(targets))
            * abs(item["observed_rate"] - item["mean_probability"])
            for item in populated
        )
        summaries.append(
            {
                "class_index": class_index,
                "calibration_error": calibration_error,
                "mean_probability": math.fsum(
                    float(row[class_index]) for row in probabilities
                )
                / len(targets),
                "observed_rate": sum(
                    target == class_index for target in targets
                )
                / len(targets),
                "bins": populated,
            }
        )
    return tuple(summaries)


def probability_sharpness(
    targets: Sequence[int],
    probabilities: Sequence[Sequence[float]],
) -> float:
    """用平均最高类别概率表达预测分布的锐度，不把它冒充准确率。"""
    multiclass_brier_score(targets, probabilities)
    return math.fsum(max(float(value) for value in row) for row in probabilities) / len(targets)


def top_label_calibration_error(
    targets: Sequence[int],
    probabilities: Sequence[Sequence[float]],
    *,
    bins: int = 10,
) -> float:
    """计算多分类最高置信度的期望校准误差。"""
    if isinstance(bins, bool) or not isinstance(bins, int) or bins <= 0:
        raise EvaluationError("bins 必须是正整数")
    # 复用 Brier 指标的严格概率契约，避免两个模块产生不同的容错语义。
    multiclass_brier_score(targets, probabilities)
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for target, row in zip(targets, probabilities):
        confidence = max(float(value) for value in row)
        predicted = max(range(len(row)), key=lambda index: float(row[index]))
        bucket_index = min(bins - 1, int(confidence * bins))
        buckets[bucket_index].append((confidence, predicted == target))

    row_count = len(targets)
    return math.fsum(
        (len(bucket) / row_count)
        * abs(
            sum(correct for _, correct in bucket) / len(bucket)
            - math.fsum(confidence for confidence, _ in bucket) / len(bucket)
        )
        for bucket in buckets
        if bucket
    )


def evaluate_expanding_window(
    rows: Sequence[Mapping[str, float]],
    targets: Sequence[int],
    feature_names: tuple[str, ...],
    model_factory: EstimatorFactory,
    *,
    folds: int = 3,
) -> TemporalEvaluation:
    """以前半段为初始历史，用扩展窗口覆盖后半段的全部样本。"""
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
        raise EvaluationError("folds 至少为 2")
    if len(rows) != len(targets) or len(rows) < folds * 2:
        raise EvaluationError("样本不足以形成请求的扩展窗口")
    if not feature_names:
        raise EvaluationError("feature_names 不能为空")

    initial_rows = len(rows) // 2
    remaining_rows = len(rows) - initial_rows
    base_size, extra = divmod(remaining_rows, folds)
    fold_sizes = [base_size + (1 if index < extra else 0) for index in range(folds)]

    all_targets: list[int] = []
    all_probabilities: list[list[float]] = []
    all_predictions: list[int] = []
    all_majority_baselines: list[int] = []
    all_elo_baselines: list[int] = []
    fold_metrics: list[dict[str, Any]] = []
    validation_start = initial_rows

    for fold_index, validation_size in enumerate(fold_sizes, start=1):
        validation_end = validation_start + validation_size
        training_rows = rows[:validation_start]
        training_targets = targets[:validation_start]
        validation_rows = rows[validation_start:validation_end]
        validation_targets = list(targets[validation_start:validation_end])

        estimator = model_factory(feature_names)
        estimator.fit(training_rows, training_targets)
        probabilities = [
            [float(value) for value in probability_row]
            for probability_row in estimator.predict_proba(validation_rows)
        ]
        predictions = [int(value) for value in estimator.predict(validation_rows)]
        majority_class = Counter(training_targets).most_common(1)[0][0]
        majority_baselines = [majority_class] * validation_size
        elo_baselines = elo_baseline_predictions(
            training_rows,
            training_targets,
            validation_rows,
        )
        majority_baseline_accuracy = accuracy_score(
            validation_targets,
            majority_baselines,
        )
        elo_baseline_accuracy = accuracy_score(
            validation_targets,
            elo_baselines,
        )
        fold_reliability = classwise_reliability(
            validation_targets,
            probabilities,
        )

        fold_metrics.append(
            {
                "fold": fold_index,
                "training_rows": len(training_rows),
                "validation_rows": validation_size,
                "accuracy": accuracy_score(validation_targets, predictions),
                "baseline_accuracy": max(
                    majority_baseline_accuracy,
                    elo_baseline_accuracy,
                ),
                "majority_baseline_accuracy": majority_baseline_accuracy,
                "elo_baseline_accuracy": elo_baseline_accuracy,
                "log_loss": multiclass_log_loss(validation_targets, probabilities),
                "brier_score": multiclass_brier_score(validation_targets, probabilities),
                "ranked_probability_score": ranked_probability_score(
                    validation_targets,
                    probabilities,
                ),
                "calibration_error": top_label_calibration_error(
                    validation_targets,
                    probabilities,
                ),
                "classwise_calibration_error": math.fsum(
                    item["calibration_error"] for item in fold_reliability
                )
                / 3.0,
                "sharpness": probability_sharpness(
                    validation_targets,
                    probabilities,
                ),
            }
        )
        all_targets.extend(validation_targets)
        all_probabilities.extend(probabilities)
        all_predictions.extend(predictions)
        all_majority_baselines.extend(majority_baselines)
        all_elo_baselines.extend(elo_baselines)
        validation_start = validation_end

    reliability = classwise_reliability(all_targets, all_probabilities)
    majority_baseline_accuracy = accuracy_score(
        all_targets,
        all_majority_baselines,
    )
    elo_baseline_accuracy = accuracy_score(all_targets, all_elo_baselines)
    return TemporalEvaluation(
        fold_count=folds,
        evaluated_rows=len(all_targets),
        accuracy=accuracy_score(all_targets, all_predictions),
        baseline_accuracy=max(
            majority_baseline_accuracy,
            elo_baseline_accuracy,
        ),
        majority_baseline_accuracy=majority_baseline_accuracy,
        elo_baseline_accuracy=elo_baseline_accuracy,
        log_loss=multiclass_log_loss(all_targets, all_probabilities),
        brier_score=multiclass_brier_score(all_targets, all_probabilities),
        ranked_probability_score=ranked_probability_score(
            all_targets,
            all_probabilities,
        ),
        calibration_error=top_label_calibration_error(
            all_targets,
            all_probabilities,
        ),
        classwise_calibration_error=math.fsum(
            item["calibration_error"] for item in reliability
        )
        / 3.0,
        sharpness=probability_sharpness(all_targets, all_probabilities),
        reliability=reliability,
        fold_metrics=tuple(fold_metrics),
    )
