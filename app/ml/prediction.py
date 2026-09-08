from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

from app.config import load_settings
from app.data_service.fetch.fetcher import DataSourceError, FootballDataClient
from app.ml.artifacts import (
    LoadedModelBundle,
    ModelArtifactError,
    ModelBundle,
    load_model_bundle_with_fallback,
)
from app.ml.feature_engineering import (
    FeatureEngineer,
    FeatureEngineeringError,
    normalize_match,
)
from app.ml.forecast_identity import feature_fingerprint, forecast_id_for_prediction

logger = logging.getLogger(__name__)


class PredictionError(RuntimeError):
    """单个联赛推理失败，上层可记录后继续处理其他联赛。"""

    def __init__(self, message: str, *, code: str = "PREDICTION_FAILED") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class PredictionRun:
    generated_at: str
    predictions: tuple[dict[str, Any], ...]
    errors: tuple[dict[str, str], ...]
    models_loaded: int
    competitions_requested: int
    model_reports: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "predictions": list(self.predictions),
            "errors": list(self.errors),
            "models_loaded": self.models_loaded,
            "competitions_requested": self.competitions_requested,
            "model_reports": list(self.model_reports),
        }


class PredictionEngine:
    def __init__(
        self,
        *,
        models_dir: Path | None = None,
        client: Any | None = None,
        feature_engineer: FeatureEngineer | None = None,
    ) -> None:
        settings = load_settings()
        self.models_dir = Path(models_dir or settings.model_dir)
        self.client = client or FootballDataClient()
        self.feature_engineer = feature_engineer or FeatureEngineer()

    def _load_bundle(self, competition_code: str) -> LoadedModelBundle:
        code = competition_code.strip().upper()
        try:
            loaded = load_model_bundle_with_fallback(
                self.models_dir / f"{code.lower()}.bundle"
            )
        except ModelArtifactError as exc:
            raise PredictionError(str(exc), code="MODEL_UNAVAILABLE") from exc
        bundle = loaded.bundle
        if bundle.competition_code != code:
            raise PredictionError("模型联赛代码与请求不一致", code="MODEL_MISMATCH")
        if bundle.feature_names != self.feature_engineer.features:
            raise PredictionError(
                "模型特征版本与当前代码不一致，请重新训练",
                code="FEATURE_SCHEMA_MISMATCH",
            )
        return loaded

    @staticmethod
    def _model_report(loaded: LoadedModelBundle) -> dict[str, Any]:
        bundle = loaded.bundle
        metrics = bundle.metrics
        return {
            "competition": bundle.competition_code,
            "schema_version": bundle.schema_version,
            "trained_at": bundle.trained_at,
            "training_rows": bundle.training_rows,
            "recovered_from_previous": loaded.recovered_from_previous,
            "model_source": "previous" if loaded.recovered_from_previous else "active",
            "model_sha256": loaded.artifact_sha256,
            "estimator": metrics.get("estimator"),
            "estimator_version": metrics.get("estimator_version"),
            "explanation_method": metrics.get("explanation_method"),
            "boosting_rounds": metrics.get("boosting_rounds"),
            "validation_accuracy": metrics.get("validation_accuracy"),
            "validation_log_loss": metrics.get("validation_log_loss"),
            "validation_brier_score": metrics.get("validation_brier_score"),
            "validation_ranked_probability_score": metrics.get(
                "validation_ranked_probability_score"
            ),
            "baseline_accuracy": metrics.get("baseline_accuracy"),
            "majority_baseline_accuracy": metrics.get("majority_baseline_accuracy"),
            "elo_baseline_accuracy": metrics.get("elo_baseline_accuracy"),
            "backtest_folds": metrics.get("backtest_folds"),
            "backtest_rows": metrics.get("backtest_rows"),
            "backtest_accuracy": metrics.get("backtest_accuracy"),
            "backtest_baseline_accuracy": metrics.get("backtest_baseline_accuracy"),
            "backtest_majority_baseline_accuracy": metrics.get(
                "backtest_majority_baseline_accuracy"
            ),
            "backtest_elo_baseline_accuracy": metrics.get(
                "backtest_elo_baseline_accuracy"
            ),
            "backtest_log_loss": metrics.get("backtest_log_loss"),
            "backtest_brier_score": metrics.get("backtest_brier_score"),
            "backtest_ranked_probability_score": metrics.get(
                "backtest_ranked_probability_score"
            ),
            "backtest_calibration_error": metrics.get("backtest_calibration_error"),
            "backtest_classwise_calibration_error": metrics.get(
                "backtest_classwise_calibration_error"
            ),
            "backtest_max_class_calibration_error": metrics.get(
                "backtest_max_class_calibration_error"
            ),
            "backtest_sharpness": metrics.get("backtest_sharpness"),
            "backtest_worst_fold_accuracy_lift": metrics.get(
                "backtest_worst_fold_accuracy_lift"
            ),
            "backtest_accuracy_lift_stddev": metrics.get(
                "backtest_accuracy_lift_stddev"
            ),
            "backtest_reliability": metrics.get("backtest_reliability"),
            "backtest_fold_metrics": list(metrics.get("backtest_fold_metrics") or []),
        }

    @staticmethod
    def _explanation(
        bundle: ModelBundle,
        features: dict[str, float],
        labels: tuple[str, str, str],
        ranked_indices: list[int],
    ) -> dict[str, Any]:
        explain = getattr(bundle.estimator, "explain_prediction", None)
        if not callable(explain):
            return {
                "available": False,
                "reason": "estimator_not_explainable",
                "top_drivers": [],
            }
        try:
            raw = explain(features, top_n=3)
            method = raw.get("method")
            predicted = int(raw["predicted_class_index"])
            runner_up = int(raw["runner_up_class_index"])
            logit_margin = float(raw["logit_margin"])
            top_drivers = []
            seen_features: set[str] = set()
            for item in raw.get("top_drivers", []):
                feature = str(item["feature"])
                contribution = float(item["contribution"])
                raw_value = float(item["raw_value"])
                if (
                    feature not in bundle.feature_names
                    or not math.isfinite(contribution)
                    or not math.isfinite(raw_value)
                    or feature in seen_features
                ):
                    raise ValueError("解释特征非法")
                seen_features.add(feature)
                top_drivers.append(
                    {
                        "feature": feature,
                        "raw_value": round(raw_value, 6),
                        "contribution": round(contribution, 6),
                        "direction": (
                            "supports_prediction"
                            if contribution >= 0.0
                            else "opposes_prediction"
                        ),
                    }
                )
            if (
                not isinstance(method, str)
                or not method.strip()
                or predicted not in range(3)
                or runner_up not in range(3)
                or predicted != ranked_indices[0]
                or runner_up != ranked_indices[1]
                or not math.isfinite(logit_margin)
                or logit_margin < 0.0
            ):
                raise ValueError("解释类别非法")
            return {
                "available": True,
                "method": method.strip(),
                "predicted_vs_runner_up": (
                    f"{labels[predicted]} vs {labels[runner_up]}"
                ),
                "logit_margin": round(logit_margin, 6),
                "top_drivers": top_drivers,
            }
        except Exception:
            # 解释是辅助证据；任何插件异常都降级为不可用，不能抹掉主概率。
            return {
                "available": False,
                "reason": "explanation_contract_failed",
                "top_drivers": [],
            }

    def _probabilities(self, bundle: ModelBundle, features: dict[str, float]) -> list[float]:
        try:
            raw = [float(value) for value in bundle.estimator.predict_proba([features])[0]]
        except (IndexError, TypeError, ValueError, AttributeError) as exc:
            raise PredictionError("模型未返回有效概率", code="INVALID_PROBABILITIES") from exc
        if (
            len(raw) != 3
            or any(not math.isfinite(value) or value < 0 for value in raw)
            or sum(raw) <= 0
        ):
            raise PredictionError("模型概率必须是三个非负有限值", code="INVALID_PROBABILITIES")
        total = sum(raw)
        normalized = raw if abs(total - 1.0) <= 1e-12 else [value / total for value in raw]
        return [round(value, 10) for value in normalized]

    def predict_competition(
        self,
        competition_code: str,
        *,
        days: int = 3,
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        code = competition_code.strip().upper()
        loaded = self._load_bundle(code)
        return self._predict_competition_with_bundle(
            code,
            loaded,
            days=days,
            now=now,
        )

    def _predict_competition_with_bundle(
        self,
        code: str,
        loaded: LoadedModelBundle,
        *,
        days: int,
        now: datetime | None,
    ) -> list[dict[str, Any]]:
        if days <= 0:
            raise PredictionError("预测天数必须大于 0", code="INVALID_RANGE")
        bundle = loaded.bundle
        model_report = self._model_report(loaded)
        generated_at = now or datetime.now(UTC)
        if generated_at.tzinfo is None:
            generated_at = generated_at.replace(tzinfo=UTC)
        generated_at = generated_at.astimezone(UTC)
        date_from = generated_at.date().isoformat()
        date_to = (generated_at + timedelta(days=days)).date().isoformat()
        range_end = datetime.fromisoformat(date_to).replace(tzinfo=UTC)

        try:
            fixtures = self.client.fetch_scheduled_matches(code, date_from, date_to)
        except DataSourceError as exc:
            raise PredictionError(str(exc), code="DATA_SOURCE_ERROR") from exc

        predictions: list[dict[str, Any]] = []
        labels = ("AWAY_WIN", "DRAW", "HOME_WIN")
        seen_fixtures: dict[int, Any] = {}
        for payload in fixtures:
            if payload.get("status") not in {"SCHEDULED", "TIMED"}:
                continue
            try:
                record = normalize_match(payload)
            except FeatureEngineeringError as exc:
                raise PredictionError(
                    f"{code} 含无法解析的赛程: {exc}",
                    code="INVALID_FIXTURE",
                ) from exc
            if not generated_at < record.kickoff < range_end:
                continue
            existing = seen_fixtures.get(record.match_id)
            if existing is not None:
                if existing == record:
                    continue
                raise PredictionError(
                    f"{code} 比赛 {record.match_id} 存在冲突重复赛程",
                    code="DUPLICATE_FIXTURE_CONFLICT",
                )
            seen_fixtures[record.match_id] = record
            try:
                features = self.feature_engineer.build_fixture_features(
                    record.home_team_id,
                    record.away_team_id,
                    record.kickoff,
                    bundle.team_states,
                )
            except FeatureEngineeringError as exc:
                raise PredictionError(
                    f"{code} 模型球队状态无效: {exc}",
                    code="INVALID_MODEL_STATE",
                ) from exc
            probabilities = self._probabilities(bundle, features)
            ranked_indices = sorted(
                range(3),
                key=probabilities.__getitem__,
                reverse=True,
            )
            best_index = ranked_indices[0]
            probability_margin = (
                probabilities[best_index] - probabilities[ranked_indices[1]]
            )
            feature_digest = feature_fingerprint(bundle.feature_names, features)
            entropy = -sum(
                probability * math.log(probability)
                for probability in probabilities
                if probability > 0
            ) / math.log(3)

            home_state = bundle.team_states.get(str(record.home_team_id), {})
            away_state = bundle.team_states.get(str(record.away_team_id), {})
            home_history = int(home_state.get("matches_played", 0))
            away_history = int(away_state.get("matches_played", 0))
            home_payload = payload.get("homeTeam") or {}
            away_payload = payload.get("awayTeam") or {}
            prediction = {
                    "match_id": record.match_id,
                    "feature_fingerprint": feature_digest,
                    "predicted_at": generated_at.isoformat(),
                    "competition": code,
                    "utc_date": record.kickoff.isoformat(),
                    "home_team_id": record.home_team_id,
                    "away_team_id": record.away_team_id,
                    "home_team": home_payload.get("name") or f"Team {record.home_team_id}",
                    "away_team": away_payload.get("name") or f"Team {record.away_team_id}",
                    "prediction": labels[best_index],
                    "confidence": probabilities[best_index],
                    "confidence_semantics": "top_outcome_probability",
                    "probability_margin": round(probability_margin, 10),
                    "uncertainty": round(entropy, 6),
                    "explanation": self._explanation(
                        bundle,
                        features,
                        labels,
                        ranked_indices,
                    ),
                    "probabilities": {
                        "home_win": probabilities[2],
                        "draw": probabilities[1],
                        "away_win": probabilities[0],
                    },
                    "data_quality": {
                        "is_cold_start": home_history == 0 or away_history == 0,
                        "home_history_matches": home_history,
                        "away_history_matches": away_history,
                    },
                    "data_provenance": {
                        "source": getattr(
                            self.client,
                            "source_name",
                            "football-data.org",
                        ),
                        "source_match_id": record.match_id,
                        "source_updated_at": (
                            payload.get("lastUpdated")
                            if isinstance(payload.get("lastUpdated"), str)
                            else None
                        ),
                        "snapshot_at": generated_at.isoformat(),
                    },
                    "model": {
                        name: value
                        for name, value in model_report.items()
                        if name not in {
                            "backtest_reliability",
                            "backtest_fold_metrics",
                        }
                    },
                }
            prediction["forecast_id"] = forecast_id_for_prediction(prediction)
            predictions.append(prediction)

        return sorted(
            predictions,
            key=lambda item: (item["utc_date"], item["match_id"]),
        )

    def predict_all(
        self,
        competitions: Mapping[str, int] | None = None,
        *,
        days: int = 3,
        now: datetime | None = None,
    ) -> PredictionRun:
        active_competitions = competitions or load_settings().competitions_map
        generated_at = now or datetime.now(UTC)
        if generated_at.tzinfo is None:
            generated_at = generated_at.replace(tzinfo=UTC)
        generated_at = generated_at.astimezone(UTC)

        predictions: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        model_reports: list[dict[str, Any]] = []
        models_loaded = 0
        for code in active_competitions:
            normalized_code = code.upper()
            try:
                loaded = self._load_bundle(normalized_code)
                model_reports.append(self._model_report(loaded))
                competition_predictions = self._predict_competition_with_bundle(
                    normalized_code,
                    loaded,
                    days=days,
                    now=generated_at,
                )
                models_loaded += 1
                predictions.extend(competition_predictions)
            except PredictionError as exc:
                errors.append(
                    {
                        "competition": normalized_code,
                        "code": exc.code,
                        "message": str(exc),
                    }
                )
            except Exception as exc:
                # 单个模型的未知运行时故障也必须隔离，不能让后续联赛全部消失。
                logger.exception("%s 预测发生未归类异常", normalized_code)
                errors.append(
                    {
                        "competition": normalized_code,
                        "code": "UNEXPECTED_PREDICTION_ERROR",
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                )

        predictions.sort(
            key=lambda item: (item["utc_date"], item["competition"], item["match_id"])
        )
        return PredictionRun(
            generated_at=generated_at.isoformat(),
            predictions=tuple(predictions),
            errors=tuple(errors),
            models_loaded=models_loaded,
            competitions_requested=len(active_competitions),
            model_reports=tuple(model_reports),
        )
