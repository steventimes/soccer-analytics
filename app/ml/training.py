from __future__ import annotations

import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import pstdev
from typing import Any, Callable, Sequence

from app.config import load_settings
from app.data_service.fetch.public import create_data_client
from app.ml.artifacts import ModelBundle, save_model_bundle
from app.ml.evaluation import EvaluationError, elo_baseline_predictions, evaluate_expanding_window
from app.ml.feature_engineering import FeatureDataset, FeatureEngineer
from app.ml.model import (
    XGBoostClassifier,
    accuracy_score,
    fit_estimator,
    multiclass_brier_score,
    multiclass_log_loss,
    ranked_probability_score,
)


class TrainingError(RuntimeError):
    """训练输入不足或无法产出可验证模型。"""


EstimatorFactory = Callable[[tuple[str, ...]], Any]


class ModelTrainer:
    """按时间留出评估，再用全部历史重训并生成自描述模型包。"""

    def __init__(
        self,
        *,
        client: Any | None = None,
        feature_engineer: FeatureEngineer | None = None,
        model_factory: EstimatorFactory | None = None,
        model_dir: Path | None = None,
        minimum_matches: int = 60,
        holdout_fraction: float = 0.2,
        backtest_folds: int = 3,
        maximum_calibration_error: float = 0.2,
        maximum_classwise_calibration_error: float = 0.2,
        minimum_worst_fold_accuracy_lift: float = -0.15,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if minimum_matches < 6:
            raise TrainingError("minimum_matches 不能小于 6")
        if not 0.1 <= holdout_fraction <= 0.5:
            raise TrainingError("holdout_fraction 必须在 0.1 到 0.5 之间")
        if isinstance(backtest_folds, bool) or not 2 <= backtest_folds <= 5:
            raise TrainingError("backtest_folds 必须在 2 到 5 之间")
        if (
            isinstance(maximum_calibration_error, bool)
            or not isinstance(maximum_calibration_error, (int, float))
            or not math.isfinite(float(maximum_calibration_error))
            or not 0.0 <= float(maximum_calibration_error) <= 1.0
        ):
            raise TrainingError("maximum_calibration_error 必须在 0 到 1 之间")
        if (
            isinstance(maximum_classwise_calibration_error, bool)
            or not isinstance(maximum_classwise_calibration_error, (int, float))
            or not math.isfinite(float(maximum_classwise_calibration_error))
            or not 0.0 <= float(maximum_classwise_calibration_error) <= 1.0
        ):
            raise TrainingError(
                "maximum_classwise_calibration_error 必须在 0 到 1 之间"
            )
        if (
            isinstance(minimum_worst_fold_accuracy_lift, bool)
            or not isinstance(minimum_worst_fold_accuracy_lift, (int, float))
            or not math.isfinite(float(minimum_worst_fold_accuracy_lift))
            or not -1.0 <= float(minimum_worst_fold_accuracy_lift) <= 1.0
        ):
            raise TrainingError("minimum_worst_fold_accuracy_lift 必须在 -1 到 1 之间")
        settings = load_settings()
        self.client = client or create_data_client()
        self.feature_engineer = feature_engineer or FeatureEngineer()
        self.model_factory = model_factory or (
            lambda feature_names: XGBoostClassifier(feature_names)
        )
        self.model_dir = Path(model_dir or settings.model_dir)
        self.minimum_matches = minimum_matches
        self.holdout_fraction = holdout_fraction
        self.backtest_folds = int(backtest_folds)
        self.maximum_calibration_error = float(maximum_calibration_error)
        self.maximum_classwise_calibration_error = float(
            maximum_classwise_calibration_error
        )
        self.minimum_worst_fold_accuracy_lift = float(
            minimum_worst_fold_accuracy_lift
        )
        self.clock = clock or (lambda: datetime.now(UTC))

    def build_dataset(
        self,
        competition_code: str,
        seasons: Sequence[str],
    ) -> FeatureDataset:
        code = competition_code.strip().upper()
        if not code or not seasons:
            raise TrainingError("联赛代码和训练赛季不能为空")
        matches: list[dict[str, Any]] = []
        for season in seasons:
            matches.extend(
                self.client.fetch_matches(
                    code,
                    season=str(season),
                    status="FINISHED",
                )
            )
        return self.feature_engineer.build_training_dataset(matches)

    def train_competition(
        self,
        competition_code: str,
        competition_id: int,
        seasons: Sequence[str],
        *,
        save: bool = True,
    ) -> ModelBundle:
        code = competition_code.strip().upper()
        if not code or competition_id <= 0:
            raise TrainingError("联赛代码或 ID 非法")

        dataset = self.build_dataset(code, seasons)
        row_count = len(dataset.rows)
        if row_count < self.minimum_matches:
            raise TrainingError(
                f"{code} 只有 {row_count} 场有效比赛，至少需要 {self.minimum_matches} 场"
            )

        # 先封存最终留出集；回测和放行前置检查只使用更早的开发数据。
        holdout_count = max(1, int(round(row_count * self.holdout_fraction)))
        split_index = row_count - holdout_count
        if dataset.kickoffs:
            while split_index > 0 and dataset.kickoffs[split_index - 1] == dataset.kickoffs[split_index]:
                split_index -= 1
            if split_index == 0:
                raise TrainingError("独立开球时间不足以形成时间留出集")
        train_rows = dataset.rows[:split_index]
        train_targets = dataset.targets[:split_index]
        validation_rows = dataset.rows[split_index:]
        validation_targets = dataset.targets[split_index:]

        # 扩展窗口覆盖开发集后半段，防止单一时间切点偶然掩盖模型退化。
        try:
            backtest = evaluate_expanding_window(
                train_rows,
                train_targets,
                self.feature_engineer.features,
                self.model_factory,
                folds=self.backtest_folds,
                kickoffs=dataset.kickoffs[:split_index],
            )
        except EvaluationError as exc:
            raise TrainingError(f"{code} 留出集之前的开发集无法完成回测：{exc}") from exc
        if backtest.calibration_error > self.maximum_calibration_error:
            raise TrainingError(
                f"{code} 扩展窗口回测校准误差 {backtest.calibration_error:.4f} "
                f"高于阈值 {self.maximum_calibration_error:.4f}；拒绝发布模型"
            )
        max_class_calibration_error = max(
            float(item["calibration_error"])
            for item in backtest.reliability
        )
        if (
            max_class_calibration_error
            > self.maximum_classwise_calibration_error
        ):
            raise TrainingError(
                f"{code} 扩展窗口最差类别校准误差 "
                f"{max_class_calibration_error:.4f} 高于阈值 "
                f"{self.maximum_classwise_calibration_error:.4f}；拒绝发布模型"
            )
        fold_accuracy_lifts = [
            float(item["accuracy"]) - float(item["baseline_accuracy"])
            for item in backtest.fold_metrics
        ]
        worst_fold_accuracy_lift = min(fold_accuracy_lifts)
        if worst_fold_accuracy_lift < self.minimum_worst_fold_accuracy_lift:
            raise TrainingError(
                f"{code} 扩展窗口最差折准确率增益 {worst_fold_accuracy_lift:.4f} "
                f"低于稳定性阈值 {self.minimum_worst_fold_accuracy_lift:.4f}；拒绝发布模型"
            )
        if backtest.accuracy < backtest.baseline_accuracy:
            raise TrainingError(
                f"{code} 扩展窗口回测准确率 {backtest.accuracy:.4f} "
                f"低于多数类/Elo 强基线 {backtest.baseline_accuracy:.4f}；拒绝发布模型"
            )

        # 验证模型只看时间切点之前的数据；验证完成后再单独创建最终模型。
        validation_model = self.model_factory(self.feature_engineer.features)
        fit_estimator(validation_model, train_rows, train_targets,
                      kickoffs=dataset.kickoffs[:split_index])
        probabilities = validation_model.predict_proba(validation_rows)
        predictions = validation_model.predict(validation_rows)

        majority_class = Counter(train_targets).most_common(1)[0][0]
        majority_baseline_predictions = [majority_class] * len(validation_targets)
        elo_baseline = elo_baseline_predictions(
            train_rows,
            train_targets,
            validation_rows,
        )
        validation_accuracy = accuracy_score(validation_targets, predictions)
        majority_baseline_accuracy = accuracy_score(
            validation_targets,
            majority_baseline_predictions,
        )
        elo_baseline_accuracy = accuracy_score(validation_targets, elo_baseline)
        baseline_accuracy = max(
            majority_baseline_accuracy,
            elo_baseline_accuracy,
        )
        metrics: dict[str, Any] = {
            "evaluation_protocol": "development_backtest_then_temporal_holdout_v1",
            "fit_rows": len(train_rows),
            "validation_rows": len(validation_rows),
            "validation_accuracy": round(validation_accuracy, 6),
            "validation_log_loss": round(
                multiclass_log_loss(validation_targets, probabilities),
                6,
            ),
            "validation_brier_score": round(
                multiclass_brier_score(validation_targets, probabilities),
                6,
            ),
            "validation_ranked_probability_score": round(
                ranked_probability_score(validation_targets, probabilities),
                6,
            ),
            "baseline_accuracy": round(baseline_accuracy, 6),
            "majority_baseline_accuracy": round(majority_baseline_accuracy, 6),
            "elo_baseline_accuracy": round(elo_baseline_accuracy, 6),
            "accuracy_lift": round(validation_accuracy - baseline_accuracy, 6),
            "backtest_folds": backtest.fold_count,
            "backtest_rows": backtest.evaluated_rows,
            "backtest_accuracy": round(backtest.accuracy, 6),
            "backtest_baseline_accuracy": round(backtest.baseline_accuracy, 6),
            "backtest_majority_baseline_accuracy": round(
                backtest.majority_baseline_accuracy,
                6,
            ),
            "backtest_elo_baseline_accuracy": round(
                backtest.elo_baseline_accuracy,
                6,
            ),
            "backtest_accuracy_lift": round(
                backtest.accuracy - backtest.baseline_accuracy,
                6,
            ),
            "backtest_log_loss": round(backtest.log_loss, 6),
            "backtest_brier_score": round(backtest.brier_score, 6),
            "backtest_ranked_probability_score": round(
                backtest.ranked_probability_score,
                6,
            ),
            "backtest_calibration_error": round(backtest.calibration_error, 6),
            "backtest_classwise_calibration_error": round(
                backtest.classwise_calibration_error,
                6,
            ),
            "backtest_max_class_calibration_error": round(
                max_class_calibration_error,
                6,
            ),
            "backtest_sharpness": round(backtest.sharpness, 6),
            "backtest_worst_fold_accuracy_lift": round(
                worst_fold_accuracy_lift,
                6,
            ),
            "backtest_accuracy_lift_stddev": round(
                pstdev(fold_accuracy_lifts),
                6,
            ),
            "backtest_reliability": [
                {
                    "class_index": item["class_index"],
                    "calibration_error": round(item["calibration_error"], 6),
                    "mean_probability": round(item["mean_probability"], 6),
                    "observed_rate": round(item["observed_rate"], 6),
                    "bins": [
                        {
                            "lower_bound": round(bucket["lower_bound"], 6),
                            "upper_bound": round(bucket["upper_bound"], 6),
                            "count": bucket["count"],
                            "mean_probability": round(bucket["mean_probability"], 6),
                            "observed_rate": round(bucket["observed_rate"], 6),
                        }
                        for bucket in item["bins"]
                    ],
                }
                for item in backtest.reliability
            ],
            "backtest_fold_metrics": [
                {
                    name: value
                    if not isinstance(value, float)
                    else round(float(value), 6)
                    for name, value in item.items()
                }
                for item in backtest.fold_metrics
            ],
            "loss_count": dataset.targets.count(0),
            "draw_count": dataset.targets.count(1),
            "win_count": dataset.targets.count(2),
        }
        # 时间留出集无法超过简单领域基线时，禁止覆盖磁盘上的已知可用模型。
        if validation_accuracy < baseline_accuracy:
            raise TrainingError(
                f"{code} 时间留出验证准确率 {validation_accuracy:.4f} "
                f"低于多数类/Elo 强基线 {baseline_accuracy:.4f}；拒绝发布模型"
            )

        final_model = self.model_factory(self.feature_engineer.features)
        fit_estimator(final_model, dataset.rows, dataset.targets,
                      kickoffs=dataset.kickoffs)
        metrics.update(
            {
                "data_source": getattr(self.client, "source_name", "football-data.org"),
                "training_seasons": list(seasons),
                "estimator": str(
                    getattr(final_model, "technology", type(final_model).__name__)
                ),
                "estimator_version": str(
                    getattr(final_model, "technology_version", "unknown")
                ),
                "explanation_method": str(
                    getattr(final_model, "explanation_method", "unavailable")
                ),
                "boosting_rounds": getattr(final_model, "rounds_used_", None),
            }
        )
        trained_at = self.clock()
        if trained_at.tzinfo is None:
            trained_at = trained_at.replace(tzinfo=UTC)
        trained_at = trained_at.astimezone(UTC)

        bundle = ModelBundle(
            competition_code=code,
            competition_id=int(competition_id),
            estimator=final_model,
            feature_names=self.feature_engineer.features,
            classes=("L", "D", "W"),
            trained_at=trained_at.isoformat(),
            metrics=metrics,
            team_states=dataset.team_states,
            feature_defaults=dataset.feature_defaults,
            training_rows=row_count,
        )
        if save:
            save_model_bundle(bundle, self.model_dir / f"{code.lower()}.bundle")
        return bundle
