from __future__ import annotations

import math
from collections import Counter
from datetime import UTC, datetime
from typing import Any, Mapping, Sequence

from app.ml.forecast_identity import (
    ForecastIdentityError,
    cold_start_evidence,
    forecast_id_matches,
)
from app.web.preset_questions import HIGH_CONFIDENCE_THRESHOLD


def build_operations_summary(
    predictions: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    *,
    models_loaded: int,
    competitions_requested: int,
    prediction_errors: Sequence[Mapping[str, Any]] = (),
    score_errors: Sequence[Mapping[str, Any]] = (),
    model_refresh_errors: Sequence[Mapping[str, Any]] = (),
    model_reports: Sequence[Mapping[str, Any]] = (),
    generated_at: str | None = None,
    high_confidence_threshold: float = HIGH_CONFIDENCE_THRESHOLD,
    low_confidence_threshold: float = 0.5,
    cold_start_warning_ratio: float = 0.5,
    maximum_calibration_error: float = 0.2,
    maximum_classwise_calibration_error: float = 0.2,
    maximum_model_age_hours: float = 72.0,
    minimum_worst_fold_accuracy_lift: float = -0.15,
) -> dict[str, Any]:
    generated = generated_at or datetime.now(UTC).isoformat()
    audited_predictions, duplicate_forecast_count, forecast_conflict_count = (
        _deduplicate_predictions(predictions)
    )
    invalid_prediction_identity_count = sum(
        _prediction_model_identity(item) is None for item in audited_predictions
    )
    invalid_prediction_probability_count = sum(
        not _valid_prediction_probability_contract(item)
        for item in audited_predictions
    )
    cold_start_values: list[bool | None] = []
    for item in audited_predictions:
        try:
            cold_start_values.append(cold_start_evidence(item))
        except ForecastIdentityError:
            cold_start_values.append(None)
    invalid_prediction_quality_count = sum(
        value is None for value in cold_start_values
    )
    confidence_values = [
        value
        for item in audited_predictions
        if _valid_prediction_probability_contract(item)
        and (value := _number(item.get("confidence"))) is not None
    ]
    uncertainty_values = [
        value
        for item in audited_predictions
        if (value := _number(item.get("uncertainty"))) is not None
    ]
    competitions = Counter(
        str(item.get("competition"))
        for item in audited_predictions
        if item.get("competition")
    )
    high_confidence = [value for value in confidence_values if value >= high_confidence_threshold]
    low_confidence = [value for value in confidence_values if value < low_confidence_threshold]

    requested = max(0, int(competitions_requested))
    loaded = max(0, int(models_loaded))
    coverage_ratio = None if requested == 0 else min(1.0, loaded / requested)

    cold_start_count = sum(value is True for value in cold_start_values)
    cold_start_ratio = (
        cold_start_count / len(audited_predictions)
        if audited_predictions
        else 0.0
    )
    (
        quality_rows,
        models_below_baseline,
        models_without_quality_metrics,
        models_below_backtest_baseline,
        models_with_poor_calibration,
        models_recovered_from_previous,
        models_stale,
        models_temporally_unstable,
        models_with_poor_classwise_calibration,
    ) = _model_quality(
        audited_predictions,
        model_reports,
        generated_at=generated,
        maximum_calibration_error=maximum_calibration_error,
        maximum_classwise_calibration_error=maximum_classwise_calibration_error,
        maximum_model_age_hours=maximum_model_age_hours,
        minimum_worst_fold_accuracy_lift=minimum_worst_fold_accuracy_lift,
    )
    quality_identities = {
        (row["competition"], row["model_sha256"]) for row in quality_rows
    }
    mismatched_competitions = {
        identity[0]
        for item in audited_predictions
        if (identity := _prediction_model_identity(item)) is not None
        and identity not in quality_identities
    }
    models_without_quality_metrics = sorted(
        set(models_without_quality_metrics) | mismatched_competitions
    )

    alerts = _build_alerts(
        prediction_count=len(audited_predictions),
        duplicate_forecast_count=duplicate_forecast_count,
        forecast_conflict_count=forecast_conflict_count,
        invalid_prediction_identity_count=invalid_prediction_identity_count,
        invalid_prediction_probability_count=invalid_prediction_probability_count,
        invalid_prediction_quality_count=invalid_prediction_quality_count,
        score_count=len(scores),
        confidence_count=len(confidence_values),
        high_confidence_count=len(high_confidence),
        low_confidence_count=len(low_confidence),
        competition_count=len(competitions),
        error_count=len(prediction_errors),
        score_error_count=len(score_errors),
        model_refresh_error_count=len(model_refresh_errors),
        loaded=loaded,
        requested=requested,
        cold_start_ratio=cold_start_ratio,
        cold_start_warning_ratio=cold_start_warning_ratio,
        models_below_baseline=models_below_baseline,
        models_without_quality_metrics=models_without_quality_metrics,
        models_below_backtest_baseline=models_below_backtest_baseline,
        models_with_poor_calibration=models_with_poor_calibration,
        models_recovered_from_previous=models_recovered_from_previous,
        models_stale=models_stale,
        models_temporally_unstable=models_temporally_unstable,
        models_with_poor_classwise_calibration=(
            models_with_poor_classwise_calibration
        ),
    )
    status = (
        "critical"
        if any(alert["level"] == "critical" for alert in alerts)
        else "attention" if alerts else "healthy"
    )

    return {
        "schema": "SoccerAnalytics.Operations.v3",
        "generated_at": generated,
        "status": status,
        "prediction_count": len(audited_predictions),
        "raw_prediction_count": len(predictions),
        "duplicate_forecast_count": duplicate_forecast_count,
        "forecast_conflict_count": forecast_conflict_count,
        "invalid_prediction_identity_count": invalid_prediction_identity_count,
        "invalid_prediction_probability_count": invalid_prediction_probability_count,
        "invalid_prediction_quality_count": invalid_prediction_quality_count,
        "score_count": len(scores),
        "competition_count": len(competitions),
        "prediction_error_count": len(prediction_errors),
        "score_error_count": len(score_errors),
        "model_refresh_error_count": len(model_refresh_errors),
        "average_confidence": _round_or_none(_average(confidence_values)),
        "average_uncertainty": _round_or_none(_average(uncertainty_values)),
        "high_confidence_count": len(high_confidence),
        "low_confidence_count": len(low_confidence),
        "cold_start_count": cold_start_count,
        "cold_start_ratio": round(cold_start_ratio, 4),
        "model_coverage": {
            "loaded": loaded,
            "requested": requested,
            "ratio": None if coverage_ratio is None else round(coverage_ratio, 4),
        },
        "model_quality_by_competition": quality_rows,
        "models_below_baseline": models_below_baseline,
        "models_without_quality_metrics": models_without_quality_metrics,
        "models_below_backtest_baseline": models_below_backtest_baseline,
        "models_with_poor_calibration": models_with_poor_calibration,
        "models_recovered_from_previous": models_recovered_from_previous,
        "models_stale": models_stale,
        "models_temporally_unstable": models_temporally_unstable,
        "models_with_poor_classwise_calibration": (
            models_with_poor_classwise_calibration
        ),
        "coverage_by_competition": [
            {"competition": competition, "prediction_count": count}
            for competition, count in sorted(competitions.items())
        ],
        "alerts": alerts,
        "thresholds": {
            "high_confidence": high_confidence_threshold,
            "low_confidence": low_confidence_threshold,
            "cold_start_warning_ratio": cold_start_warning_ratio,
            "maximum_calibration_error": maximum_calibration_error,
            "maximum_classwise_calibration_error": (
                maximum_classwise_calibration_error
            ),
            "maximum_model_age_hours": maximum_model_age_hours,
            "minimum_worst_fold_accuracy_lift": minimum_worst_fold_accuracy_lift,
        },
    }


def _model_quality(
    predictions: list[dict[str, Any]],
    model_reports: Sequence[Mapping[str, Any]],
    *,
    generated_at: str,
    maximum_calibration_error: float,
    maximum_classwise_calibration_error: float,
    maximum_model_age_hours: float,
    minimum_worst_fold_accuracy_lift: float,
) -> tuple[
    list[dict[str, Any]],
    list[str],
    list[str],
    list[str],
    list[str],
    list[str],
    list[str],
    list[str],
    list[str],
]:
    predicted_competitions = {
        str(item["competition"])
        for item in predictions
        if item.get("competition")
    }
    reports_are_explicit = bool(model_reports)
    sources: list[Mapping[str, Any]] = list(model_reports)

    by_competition: dict[str, Mapping[str, Any]] = {}
    for source in sources:
        competition = source.get("competition")
        if competition:
            by_competition[str(competition)] = source

    rows: list[dict[str, Any]] = []
    below_validation: list[str] = []
    below_backtest: list[str] = []
    poor_calibration: list[str] = []
    recovered: list[str] = []
    stale: list[str] = []
    temporally_unstable: list[str] = []
    poor_classwise_calibration: list[str] = []
    without_quality: list[str] = []
    required_backtest_fields = (
        "validation_log_loss",
        "validation_brier_score",
        "validation_ranked_probability_score",
        "backtest_rows",
        "backtest_accuracy",
        "backtest_baseline_accuracy",
        "backtest_log_loss",
        "backtest_brier_score",
        "backtest_ranked_probability_score",
        "backtest_calibration_error",
        "backtest_classwise_calibration_error",
        "backtest_max_class_calibration_error",
        "backtest_sharpness",
        "backtest_worst_fold_accuracy_lift",
        "backtest_accuracy_lift_stddev",
    )
    snapshot_time = _timestamp(generated_at)

    for code, source in sorted(by_competition.items()):
        if source.get("recovered_from_previous") is True:
            recovered.append(code)
        trained_at = _timestamp(source.get("trained_at"))
        model_age_hours: float | None = None
        if (
            snapshot_time is not None
            and trained_at is not None
            and trained_at <= snapshot_time
        ):
            model_age_hours = (
                snapshot_time - trained_at
            ).total_seconds() / 3600
            if model_age_hours > maximum_model_age_hours:
                stale.append(code)

        validation = _number(source.get("validation_accuracy"))
        baseline = _number(source.get("baseline_accuracy"))
        schema_version = source.get("schema_version")
        model_sha256 = _sha256(source.get("model_sha256"))
        backtest_values = {
            name: _number(source.get(name)) for name in required_backtest_fields
        }
        folds = _positive_int(source.get("backtest_folds"))
        backtest_rows = _positive_int(source.get("backtest_rows"))
        reliability = _validated_reliability(
            source.get("backtest_reliability"),
            backtest_rows,
        )
        if (
            validation is None
            or baseline is None
            or isinstance(schema_version, bool)
            or schema_version != 3
            or model_sha256 is None
            or folds is None
            or backtest_rows is None
            or reliability is None
            or any(value is None for value in backtest_values.values())
            or trained_at is None
            or model_age_hours is None
        ):
            without_quality.append(code)
            continue
        reliability_average = sum(
            item["calibration_error"] for item in reliability
        ) / len(reliability)
        reliability_maximum = max(
            item["calibration_error"] for item in reliability
        )
        if (
            not math.isclose(
                float(backtest_values["backtest_classwise_calibration_error"]),
                reliability_average,
                rel_tol=0.0,
                abs_tol=1e-4,
            )
            or not math.isclose(
                float(backtest_values["backtest_max_class_calibration_error"]),
                reliability_maximum,
                rel_tol=0.0,
                abs_tol=1e-4,
            )
        ):
            without_quality.append(code)
            continue
        if validation < baseline:
            below_validation.append(code)

        row: dict[str, Any] = {
            "competition": code,
            "schema_version": schema_version,
            "model_sha256": model_sha256,
            "validation_accuracy": round(validation, 4),
            "baseline_accuracy": round(baseline, 4),
            "accuracy_lift": round(validation - baseline, 4),
        }
        for name in ("estimator", "estimator_version", "explanation_method"):
            value = source.get(name)
            if isinstance(value, str) and value.strip():
                row[name] = value.strip()
        boosting_rounds = _positive_int(source.get("boosting_rounds"))
        if boosting_rounds is not None:
            row["boosting_rounds"] = boosting_rounds
        row.update(
            {name: round(float(value), 4) for name, value in backtest_values.items()}
        )
        for name in (
            "majority_baseline_accuracy",
            "elo_baseline_accuracy",
            "backtest_majority_baseline_accuracy",
            "backtest_elo_baseline_accuracy",
        ):
            value = _number(source.get(name))
            if value is not None and 0.0 <= value <= 1.0:
                row[name] = round(value, 4)
        row["backtest_folds"] = folds
        row["backtest_rows"] = backtest_rows
        row["backtest_reliability"] = reliability
        row["backtest_max_class_calibration_error"] = round(
            reliability_maximum,
            4,
        )
        row["trained_at"] = trained_at.isoformat()
        row["model_age_hours"] = round(model_age_hours, 2)
        row["backtest_accuracy_lift"] = round(
            float(backtest_values["backtest_accuracy"])
            - float(backtest_values["backtest_baseline_accuracy"]),
            4,
        )
        if backtest_values["backtest_accuracy"] < backtest_values["backtest_baseline_accuracy"]:
            below_backtest.append(code)
        if backtest_values["backtest_calibration_error"] > maximum_calibration_error:
            poor_calibration.append(code)
        if (
            reliability_maximum
            > maximum_classwise_calibration_error
        ):
            poor_classwise_calibration.append(code)
        if (
            backtest_values["backtest_worst_fold_accuracy_lift"]
            < minimum_worst_fold_accuracy_lift
        ):
            temporally_unstable.append(code)
        rows.append(row)

    expected_competitions = (
        set(by_competition) if reports_are_explicit else predicted_competitions
    )
    without_quality = sorted(set(without_quality) | (expected_competitions - {row["competition"] for row in rows}))
    return (
        rows,
        below_validation,
        without_quality,
        below_backtest,
        poor_calibration,
        recovered,
        stale,
        temporally_unstable,
        poor_classwise_calibration,
    )


def _build_alerts(
    *,
    prediction_count: int,
    duplicate_forecast_count: int,
    forecast_conflict_count: int,
    invalid_prediction_identity_count: int,
    invalid_prediction_probability_count: int,
    invalid_prediction_quality_count: int,
    score_count: int,
    confidence_count: int,
    high_confidence_count: int,
    low_confidence_count: int,
    competition_count: int,
    error_count: int,
    score_error_count: int,
    model_refresh_error_count: int,
    loaded: int,
    requested: int,
    cold_start_ratio: float,
    cold_start_warning_ratio: float,
    models_below_baseline: list[str],
    models_without_quality_metrics: list[str],
    models_below_backtest_baseline: list[str],
    models_with_poor_calibration: list[str],
    models_recovered_from_previous: list[str],
    models_stale: list[str],
    models_temporally_unstable: list[str],
    models_with_poor_classwise_calibration: list[str],
) -> list[dict[str, str]]:
    alerts: list[dict[str, str]] = []
    if forecast_conflict_count:
        alerts.append({"level": "critical", "code": "FORECAST_ID_CONFLICT", "message": f"{forecast_conflict_count} duplicate forecast identifiers contain conflicting payloads."})
    elif duplicate_forecast_count:
        alerts.append({"level": "critical", "code": "DUPLICATE_FORECASTS", "message": f"{duplicate_forecast_count} duplicate forecasts were removed before quality and release counts."})
    if invalid_prediction_identity_count:
        alerts.append({"level": "critical", "code": "PREDICTION_IDENTITY_INVALID", "message": f"{invalid_prediction_identity_count} predictions lack a valid forecast, match, competition, or model identity."})
    if invalid_prediction_probability_count:
        alerts.append({"level": "critical", "code": "PREDICTION_PROBABILITIES_INVALID", "message": f"{invalid_prediction_probability_count} predictions violate the probability, confidence, or outcome contract."})
    if invalid_prediction_quality_count:
        alerts.append({"level": "critical", "code": "PREDICTION_DATA_QUALITY_INVALID", "message": f"{invalid_prediction_quality_count} predictions contain missing or inconsistent cold-start history evidence."})
    if prediction_count == 0:
        alerts.append({"level": "critical", "code": "NO_PREDICTIONS", "message": "No upcoming predictions were exported; check model artifacts and fixture ingestion."})
    if score_count == 0:
        alerts.append({"level": "warning", "code": "NO_SCORES", "message": "No latest scores were exported; verify the scores data source or cache."})
    if requested > loaded:
        alerts.append({"level": "critical", "code": "MODEL_COVERAGE_GAP", "message": f"Only {loaded} of {requested} requested competition models completed prediction."})
    if error_count:
        alerts.append({"level": "warning", "code": "PREDICTION_ERRORS", "message": f"{error_count} competition prediction errors were exported for diagnosis."})
    if score_error_count:
        alerts.append({"level": "warning", "code": "SCORE_FETCH_ERRORS", "message": f"{score_error_count} score source errors were exported for diagnosis."})
    if model_refresh_error_count:
        alerts.append({"level": "warning", "code": "MODEL_REFRESH_ERRORS", "message": f"{model_refresh_error_count} model refresh errors were exported; the active snapshot may be stale."})
    if models_without_quality_metrics:
        alerts.append({"level": "critical", "code": "MODEL_QUALITY_UNAVAILABLE", "message": "Validation and baseline accuracy are unavailable for: " + ", ".join(models_without_quality_metrics)})
    if models_below_baseline:
        alerts.append({"level": "critical", "code": "MODEL_BELOW_BASELINE", "message": "Validation accuracy is below baseline for: " + ", ".join(models_below_baseline)})
    if models_below_backtest_baseline:
        alerts.append({"level": "critical", "code": "MODEL_BACKTEST_BELOW_BASELINE", "message": "Expanding-window accuracy is below its temporal baseline for: " + ", ".join(models_below_backtest_baseline)})
    if models_with_poor_calibration:
        alerts.append({"level": "critical", "code": "MODEL_CALIBRATION_POOR", "message": "Backtest calibration error exceeds the release threshold for: " + ", ".join(models_with_poor_calibration)})
    if models_recovered_from_previous:
        alerts.append({"level": "warning", "code": "MODEL_FALLBACK_ACTIVE", "message": "Prediction is using the previous model version for: " + ", ".join(models_recovered_from_previous)})
    if models_stale:
        alerts.append({"level": "critical", "code": "MODEL_STALE", "message": "Model age exceeds the release threshold for: " + ", ".join(models_stale)})
    if models_temporally_unstable:
        alerts.append({"level": "critical", "code": "MODEL_TEMPORALLY_UNSTABLE", "message": "At least one expanding-window fold fell below the stability threshold for: " + ", ".join(models_temporally_unstable)})
    if models_with_poor_classwise_calibration:
        alerts.append({"level": "critical", "code": "MODEL_CLASSWISE_CALIBRATION_POOR", "message": "At least one outcome class exceeds the calibration threshold for: " + ", ".join(models_with_poor_classwise_calibration)})
    if prediction_count and cold_start_ratio > cold_start_warning_ratio:
        alerts.append({"level": "warning", "code": "COLD_START_CLUSTER", "message": f"Cold-start fixtures account for {cold_start_ratio:.0%} of predictions."})
    if prediction_count > 0 and confidence_count < prediction_count:
        alerts.append({"level": "warning", "code": "MISSING_CONFIDENCE", "message": "Some predictions are missing finite numeric confidence values."})
    if prediction_count > 0 and high_confidence_count == 0:
        alerts.append({"level": "warning", "code": "NO_HIGH_CONFIDENCE_PICKS", "message": "No prediction cleared the high-confidence threshold for recommendations."})
    if prediction_count > 0 and low_confidence_count > prediction_count / 2:
        alerts.append({"level": "warning", "code": "LOW_CONFIDENCE_CLUSTER", "message": "More than half of exported predictions are below the low-confidence threshold."})
    if prediction_count > 0 and competition_count == 0:
        alerts.append({"level": "warning", "code": "MISSING_COMPETITION_LABELS", "message": "Predictions are missing competition labels, which weakens filtering and reporting."})
    return alerts


def _deduplicate_predictions(
    predictions: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    """按 forecast_id 或比赛键去重，并区分完全重复与内容冲突。"""
    unique: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}
    forecast_identities: dict[str, str] = {}
    duplicate_count = 0
    conflict_count = 0
    for index, item in enumerate(predictions):
        row = dict(item)
        forecast_id = item.get("forecast_id")
        canonical_forecast_id = _sha256(forecast_id)
        if item.get("competition") and item.get("match_id") is not None:
            identity = f"match:{item.get('competition')}:{item.get('match_id')}"
        elif canonical_forecast_id is not None:
            identity = f"forecast:{canonical_forecast_id}"
        else:
            identity = f"invalid:{index}"
        if canonical_forecast_id is not None:
            normalized_forecast_id = canonical_forecast_id
            prior_identity = forecast_identities.get(normalized_forecast_id)
            if prior_identity is not None and prior_identity != identity:
                duplicate_count += 1
                conflict_count += 1
                continue
            forecast_identities[normalized_forecast_id] = identity
        elif isinstance(forecast_id, str) and forecast_id.strip():
            normalized_forecast_id = forecast_id.strip()
            prior_identity = forecast_identities.get(normalized_forecast_id)
            if prior_identity is not None and prior_identity != identity:
                duplicate_count += 1
                conflict_count += 1
                continue
            forecast_identities[normalized_forecast_id] = identity
        existing = seen.get(identity)
        if existing is None:
            seen[identity] = row
            unique.append(row)
            continue
        duplicate_count += 1
        if existing != row:
            conflict_count += 1
    return unique, duplicate_count, conflict_count


def _sha256(value: Any) -> str | None:
    if not isinstance(value, str) or len(value) != 64:
        return None
    normalized = value.lower()
    return normalized if all(character in "0123456789abcdef" for character in normalized) else None


def _prediction_model_identity(
    prediction: Mapping[str, Any],
) -> tuple[str, str] | None:
    competition = prediction.get("competition")
    model = prediction.get("model")
    if (
        not isinstance(competition, str)
        or not competition
        or competition != competition.strip().upper()
        or _positive_int(prediction.get("match_id")) is None
        or _sha256(prediction.get("forecast_id")) is None
        or not isinstance(model, Mapping)
        or isinstance(model.get("schema_version"), bool)
        or model.get("schema_version") != 3
        or not forecast_id_matches(prediction)
    ):
        return None
    model_sha256 = _sha256(model.get("model_sha256"))
    return (competition, model_sha256) if model_sha256 is not None else None


def _valid_prediction_probability_contract(
    prediction: Mapping[str, Any],
) -> bool:
    confidence = _number(prediction.get("confidence"))
    probabilities = prediction.get("probabilities")
    probability_keys = ("away_win", "draw", "home_win")
    if (
        confidence is None
        or not 0.0 <= confidence <= 1.0
        or not isinstance(probabilities, Mapping)
        or set(probabilities) != set(probability_keys)
    ):
        return False
    values = [_number(probabilities.get(name)) for name in probability_keys]
    if (
        any(value is None or not 0.0 <= value <= 1.0 for value in values)
        or not math.isclose(
            math.fsum(float(value) for value in values),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    ):
        return False
    normalized = [float(value) for value in values]
    best_index = max(range(3), key=normalized.__getitem__)
    return (
        math.isclose(
            confidence,
            normalized[best_index],
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        and prediction.get("prediction")
        == ("AWAY_WIN", "DRAW", "HOME_WIN")[best_index]
    )


def _validated_reliability(
    value: Any,
    backtest_rows: int | None,
) -> list[dict[str, Any]] | None:
    """验证三类别可靠性分箱，并压缩为前端所需摘要。"""
    if backtest_rows is None or not isinstance(value, list) or len(value) != 3:
        return None
    summaries: list[dict[str, Any]] = []
    seen_classes: set[int] = set()
    for item in value:
        if not isinstance(item, Mapping):
            return None
        class_index = item.get("class_index")
        calibration_error = _number(item.get("calibration_error"))
        bins = item.get("bins")
        if (
            isinstance(class_index, bool)
            or class_index not in (0, 1, 2)
            or class_index in seen_classes
            or calibration_error is None
            or not 0.0 <= calibration_error <= 1.0
            or not isinstance(bins, list)
            or not bins
        ):
            return None
        sample_count = 0
        computed_calibration_error = 0.0
        for bucket in bins:
            if not isinstance(bucket, Mapping):
                return None
            count = _positive_int(bucket.get("count"))
            lower = _number(bucket.get("lower_bound"))
            upper = _number(bucket.get("upper_bound"))
            predicted = _number(bucket.get("mean_probability"))
            observed = _number(bucket.get("observed_rate"))
            if (
                count is None
                or lower is None
                or upper is None
                or predicted is None
                or observed is None
                or not 0.0 <= lower < upper <= 1.0
                or not 0.0 <= predicted <= 1.0
                or not 0.0 <= observed <= 1.0
            ):
                return None
            sample_count += count
            computed_calibration_error += (
                count / backtest_rows
            ) * abs(observed - predicted)
        if sample_count != backtest_rows:
            return None
        if not math.isclose(
            calibration_error,
            computed_calibration_error,
            rel_tol=0.0,
            abs_tol=1e-4,
        ):
            return None
        seen_classes.add(class_index)
        summaries.append(
            {
                "class_index": class_index,
                "calibration_error": round(calibration_error, 4),
                "sample_count": sample_count,
            }
        )
    return sorted(summaries, key=lambda item: item["class_index"])


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _average(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _round_or_none(value: float | None) -> float | None:
    return None if value is None else round(value, 4)
