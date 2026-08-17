from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any, Mapping

from app.ml.forecast_identity import (
    ForecastIdentityError,
    cold_start_evidence,
    forecast_id_matches,
)


def build_release_governance(
    predictions: list[dict[str, Any]],
    operations_summary: dict[str, Any],
    *,
    generated_at: str | None = None,
    minimum_predictions: int = 3,
    minimum_high_confidence_picks: int = 1,
    minimum_average_confidence: float = 0.55,
    maximum_cold_start_ratio: float = 0.5,
) -> dict[str, Any]:
    generated = generated_at or datetime.now(UTC).isoformat()
    (
        unique_predictions,
        duplicate_forecast_count,
        forecast_conflict_count,
        missing_identity_count,
    ) = _unique_predictions(predictions)
    invalid_probability_count = sum(
        not _valid_prediction_probability_contract(item)
        for item in unique_predictions
    )
    blockers = _build_blockers(
        predictions=unique_predictions,
        raw_prediction_count=len(predictions),
        duplicate_forecast_count=duplicate_forecast_count,
        forecast_conflict_count=forecast_conflict_count,
        missing_identity_count=missing_identity_count,
        invalid_probability_count=invalid_probability_count,
        operations_summary=operations_summary,
        generated_at=generated,
        minimum_predictions=minimum_predictions,
        minimum_high_confidence_picks=minimum_high_confidence_picks,
        minimum_average_confidence=minimum_average_confidence,
        maximum_cold_start_ratio=maximum_cold_start_ratio,
    )
    warnings = _build_warnings(operations_summary)
    approved = not blockers
    confidence_values = [
        value
        for item in unique_predictions
        if (value := _number(item.get("confidence"))) is not None
    ]
    high_threshold = _number(
        (operations_summary.get("thresholds") or {}).get("high_confidence")
        if isinstance(operations_summary.get("thresholds"), Mapping)
        else None
    )
    if high_threshold is None:
        high_threshold = 0.6
    unique_high_confidence_count = sum(
        value >= high_threshold for value in confidence_values
    )
    unique_average_confidence = (
        sum(confidence_values) / len(confidence_values)
        if confidence_values
        else None
    )
    return {
        "schema": "SoccerAnalytics.ReleaseGovernance.v3",
        "generated_at": generated,
        "release_decision": "approved" if approved else "blocked",
        "can_publish_recommendations": approved,
        "blockers": blockers,
        "warnings": warnings,
        "thresholds": {
            "minimum_predictions": minimum_predictions,
            "minimum_high_confidence_picks": minimum_high_confidence_picks,
            "minimum_average_confidence": minimum_average_confidence,
            "maximum_cold_start_ratio": maximum_cold_start_ratio,
        },
        "risk_disclosure": {
            "summary": "Model picks are probabilistic decision support, not guaranteed outcomes or financial advice.",
            "responsible_use": [
                "Do not publish picks when release_decision is blocked.",
                "Do not present confidence as certainty.",
                "Show bankroll and loss-limit guidance outside the model output.",
                "Keep human review in the publishing workflow for paid or public recommendations.",
            ],
        },
        "audit": {
            "prediction_count": len(predictions),
            "unique_prediction_count": len(unique_predictions),
            "duplicate_forecast_count": duplicate_forecast_count,
            "forecast_conflict_count": forecast_conflict_count,
            "missing_forecast_identity_count": missing_identity_count,
            "invalid_prediction_probability_count": invalid_probability_count,
            "operation_status": operations_summary.get("status", "unknown"),
            "average_confidence": unique_average_confidence,
            "high_confidence_count": unique_high_confidence_count,
            "model_coverage": operations_summary.get("model_coverage"),
            "cold_start_ratio": operations_summary.get("cold_start_ratio"),
            "models_below_baseline": operations_summary.get("models_below_baseline", []),
            "models_without_quality_metrics": operations_summary.get("models_without_quality_metrics", []),
            "models_below_backtest_baseline": operations_summary.get("models_below_backtest_baseline", []),
            "models_with_poor_calibration": operations_summary.get("models_with_poor_calibration", []),
            "models_recovered_from_previous": operations_summary.get("models_recovered_from_previous", []),
            "models_stale": operations_summary.get("models_stale", []),
            "models_temporally_unstable": operations_summary.get(
                "models_temporally_unstable", []
            ),
            "models_with_poor_classwise_calibration": operations_summary.get(
                "models_with_poor_classwise_calibration", []
            ),
            "unique_average_confidence": unique_average_confidence,
            "unique_high_confidence_count": unique_high_confidence_count,
            "model_refresh_error_count": operations_summary.get(
                "model_refresh_error_count", 0
            ),
            "alert_codes": [alert.get("code") for alert in operations_summary.get("alerts", [])],
        },
        "verification_commands": [
            "python3 -m unittest discover -s tests -v",
            "python3 -m app.pipeline export-site --days 3",
            "node --check docs/app.js",
        ],
    }


def _build_blockers(
    *,
    predictions: list[dict[str, Any]],
    raw_prediction_count: int,
    duplicate_forecast_count: int,
    forecast_conflict_count: int,
    missing_identity_count: int,
    invalid_probability_count: int,
    operations_summary: dict[str, Any],
    generated_at: str,
    minimum_predictions: int,
    minimum_high_confidence_picks: int,
    minimum_average_confidence: float,
    maximum_cold_start_ratio: float,
) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    if operations_summary.get("schema") != "SoccerAnalytics.Operations.v3":
        blockers.append({"code": "OPERATIONS_SCHEMA_INVALID", "message": "Release governance requires a validated SoccerAnalytics.Operations.v3 summary."})
    release_snapshot = _timestamp(generated_at)
    operations_snapshot = _timestamp(operations_summary.get("generated_at"))
    snapshots_match = (
        release_snapshot is not None
        and operations_snapshot is not None
        and release_snapshot == operations_snapshot
    )
    if not snapshots_match:
        blockers.append({"code": "OPERATIONS_SNAPSHOT_MISMATCH", "message": "Operations and release governance must use the same valid snapshot timestamp."})
    if forecast_conflict_count:
        blockers.append({"code": "FORECAST_ID_CONFLICT", "message": f"{forecast_conflict_count} forecast identifiers refer to conflicting payloads."})
    elif duplicate_forecast_count:
        blockers.append({"code": "DUPLICATE_FORECASTS", "message": f"{duplicate_forecast_count} duplicate forecasts cannot contribute to release coverage."})
    if missing_identity_count:
        blockers.append({"code": "FORECAST_IDENTITY_INCOMPLETE", "message": f"{missing_identity_count} predictions lack both forecast_id and a competition/match identity."})
    if invalid_probability_count:
        blockers.append({"code": "PREDICTION_PROBABILITIES_INVALID", "message": f"{invalid_probability_count} predictions violate the probability, confidence, or outcome contract."})
    if len(predictions) < minimum_predictions:
        blockers.append({"code": "INSUFFICIENT_FIXTURE_COVERAGE", "message": f"Only {len(predictions)} predictions were exported; require at least {minimum_predictions} before publishing picks."})
    if operations_summary.get("status") != "healthy":
        blockers.append({"code": "OPERATIONS_NOT_HEALTHY", "message": "Latest operations summary is not healthy; resolve data quality alerts before publishing."})

    high_threshold = _number(
        (operations_summary.get("thresholds") or {}).get("high_confidence")
        if isinstance(operations_summary.get("thresholds"), Mapping)
        else None
    )
    if high_threshold is None:
        high_threshold = 0.6
    confidence_values = [
        value
        for item in predictions
        if (value := _number(item.get("confidence"))) is not None
    ]
    high_confidence_count = sum(
        value >= high_threshold for value in confidence_values
    )
    if high_confidence_count < minimum_high_confidence_picks:
        blockers.append({"code": "NO_PUBLISHABLE_PICKS", "message": f"Only {high_confidence_count} high-confidence picks cleared the threshold; require {minimum_high_confidence_picks}."})
    average_confidence = (
        sum(confidence_values) / len(confidence_values)
        if confidence_values
        else None
    )
    if average_confidence is None or average_confidence < minimum_average_confidence:
        reported_average = operations_summary.get("average_confidence")
        display_average = "unavailable" if average_confidence is None else reported_average
        blockers.append({"code": "LOW_AVERAGE_CONFIDENCE", "message": f"Average confidence is {display_average}; require at least {minimum_average_confidence}."})
    if (
        _nonnegative_int(operations_summary.get("prediction_count")) != len(predictions)
        or _nonnegative_int(operations_summary.get("high_confidence_count"))
        != high_confidence_count
        or _number(operations_summary.get("average_confidence")) is None
        or not math.isclose(
            float(operations_summary.get("average_confidence")),
            average_confidence if average_confidence is not None else math.nan,
            rel_tol=0.0,
            abs_tol=1e-4,
        )
        or raw_prediction_count < len(predictions)
    ):
        blockers.append({"code": "OPERATIONS_PREDICTION_AUDIT_MISMATCH", "message": "Operations counts do not match the unique release prediction set."})

    coverage = operations_summary.get("model_coverage")
    loaded = 0
    if isinstance(coverage, Mapping):
        requested_value = _nonnegative_int(coverage.get("requested"))
        loaded_value = _nonnegative_int(coverage.get("loaded"))
        ratio = _number(coverage.get("ratio"))
        if (
            requested_value is None
            or loaded_value is None
            or ratio is None
            or not 0.0 <= ratio <= 1.0
            or loaded_value > requested_value
            or not math.isclose(
                ratio,
                loaded_value / requested_value if requested_value else 0.0,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ):
            blockers.append({"code": "MODEL_COVERAGE_INVALID", "message": "Model coverage requires finite non-negative loaded/requested counts and a ratio in [0, 1]."})
            requested = 0
        else:
            requested = requested_value
            loaded = loaded_value
        if requested > loaded:
            blockers.append({"code": "MODEL_COVERAGE_INCOMPLETE", "message": f"Only {loaded} of {requested} requested models completed prediction."})
    else:
        blockers.append({"code": "MODEL_COVERAGE_INVALID", "message": "Model coverage evidence is missing."})
    evidence_rows = operations_summary.get("model_quality_by_competition")
    quality_thresholds = _quality_thresholds(operations_summary.get("thresholds"))
    complete_evidence_rows = [
        row
        for row in evidence_rows
        if isinstance(row, Mapping)
        and quality_thresholds is not None
        and snapshots_match
        and release_snapshot is not None
        and _quality_evidence_complete(
            row,
            quality_thresholds,
            snapshot_at=release_snapshot,
        )
    ] if isinstance(evidence_rows, list) else []
    evidence_competitions = {
        str(row.get("competition")) for row in complete_evidence_rows
    }
    evidence_identities = {
        identity
        for row in complete_evidence_rows
        if (identity := _quality_model_identity(row)) is not None
    }
    prediction_identities = {
        identity
        for item in predictions
        if (identity := _prediction_model_identity(item)) is not None
    }
    if (
        loaded <= 0
        or len(complete_evidence_rows) != loaded
        or len(evidence_competitions) != loaded
        or len(evidence_identities) != len(complete_evidence_rows)
        or not prediction_identities.issubset(evidence_identities)
    ):
        blockers.append(
            {
                "code": "MODEL_EVIDENCE_INCOMPLETE",
                "message": (
                    f"Quality evidence covers {len(evidence_competitions)} of "
                    f"{loaded} loaded models."
                ),
            }
        )
    below = [str(code) for code in operations_summary.get("models_below_baseline", [])]
    if below:
        blockers.append({"code": "MODEL_QUALITY_BELOW_BASELINE", "message": "Models below their validation baseline: " + ", ".join(below)})
    without_quality = [
        str(code)
        for code in operations_summary.get("models_without_quality_metrics", [])
    ]
    if without_quality:
        blockers.append({"code": "MODEL_QUALITY_UNAVAILABLE", "message": "Models without finite validation and baseline accuracy: " + ", ".join(without_quality)})
    below_backtest = [str(code) for code in operations_summary.get("models_below_backtest_baseline", [])]
    if below_backtest:
        blockers.append({"code": "MODEL_BACKTEST_BELOW_BASELINE", "message": "Models below their expanding-window baseline: " + ", ".join(below_backtest)})
    poor_calibration = [str(code) for code in operations_summary.get("models_with_poor_calibration", [])]
    if poor_calibration:
        blockers.append({"code": "MODEL_CALIBRATION_TOO_HIGH", "message": "Models above the calibration-error limit: " + ", ".join(poor_calibration)})
    poor_classwise = [str(code) for code in operations_summary.get("models_with_poor_classwise_calibration", [])]
    if poor_classwise:
        blockers.append({"code": "MODEL_CLASSWISE_CALIBRATION_TOO_HIGH", "message": "Models with an outcome class above the calibration limit: " + ", ".join(poor_classwise)})
    recovered = [str(code) for code in operations_summary.get("models_recovered_from_previous", [])]
    if recovered:
        blockers.append({"code": "MODEL_FALLBACK_ACTIVE", "message": "Previous model versions are active for: " + ", ".join(recovered)})
    stale = [str(code) for code in operations_summary.get("models_stale", [])]
    if stale:
        blockers.append({"code": "MODEL_STALE", "message": "Models older than the allowed snapshot age: " + ", ".join(stale)})
    unstable = [
        str(code)
        for code in operations_summary.get("models_temporally_unstable", [])
    ]
    if unstable:
        blockers.append({"code": "MODEL_TEMPORALLY_UNSTABLE", "message": "Models with an unstable expanding-window fold: " + ", ".join(unstable)})
    cold_start_values: list[bool] = []
    invalid_cold_start_evidence = 0
    for prediction in predictions:
        try:
            cold_start_values.append(cold_start_evidence(prediction))
        except ForecastIdentityError:
            invalid_cold_start_evidence += 1
    if invalid_cold_start_evidence:
        blockers.append({"code": "PREDICTION_DATA_QUALITY_INVALID", "message": f"{invalid_cold_start_evidence} predictions contain missing or inconsistent cold-start history evidence."})
    computed_cold_start_count = sum(cold_start_values)
    computed_cold_start_ratio = (
        computed_cold_start_count / len(predictions) if predictions else 0.0
    )
    reported_cold_start_count = _nonnegative_int(
        operations_summary.get("cold_start_count")
    )
    reported_cold_start_ratio = _number(
        operations_summary.get("cold_start_ratio")
    )
    if (
        reported_cold_start_count != computed_cold_start_count
        or reported_cold_start_ratio is None
        or not math.isclose(
            reported_cold_start_ratio,
            round(computed_cold_start_ratio, 4),
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    ):
        blockers.append({"code": "COLD_START_AUDIT_MISMATCH", "message": "Operations cold-start counts do not match the unique release prediction set."})
    if computed_cold_start_ratio > maximum_cold_start_ratio:
        blockers.append({"code": "EXCESSIVE_COLD_START", "message": f"Cold-start ratio {computed_cold_start_ratio:.0%} exceeds the {maximum_cold_start_ratio:.0%} limit."})

    for alert in operations_summary.get("alerts", []):
        if alert.get("level") == "critical":
            blockers.append({"code": f"CRITICAL_ALERT_{alert.get('code', 'UNKNOWN')}", "message": str(alert.get("message", "A critical operations alert is present."))})
    return blockers


_REQUIRED_QUALITY_NUMBERS = (
    "validation_accuracy",
    "baseline_accuracy",
    "validation_log_loss",
    "validation_brier_score",
    "validation_ranked_probability_score",
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
    "model_age_hours",
)


def _quality_evidence_complete(
    row: Mapping[str, Any],
    thresholds: Mapping[str, float],
    *,
    snapshot_at: datetime,
) -> bool:
    values = {
        name: _number(row.get(name)) for name in _REQUIRED_QUALITY_NUMBERS
    }
    if _quality_model_identity(row) is None or any(
        value is None for value in values.values()
    ):
        return False
    if (
        not all(0.0 <= float(values[name]) <= 1.0 for name in (
            "validation_accuracy",
            "baseline_accuracy",
            "validation_ranked_probability_score",
            "backtest_accuracy",
            "backtest_baseline_accuracy",
            "backtest_ranked_probability_score",
            "backtest_calibration_error",
            "backtest_classwise_calibration_error",
            "backtest_max_class_calibration_error",
            "backtest_sharpness",
        ))
        or any(float(values[name]) < 0.0 for name in (
            "validation_log_loss",
            "validation_brier_score",
            "backtest_log_loss",
            "backtest_brier_score",
            "backtest_accuracy_lift_stddev",
            "model_age_hours",
        ))
        or not -1.0 <= float(values["backtest_worst_fold_accuracy_lift"]) <= 1.0
    ):
        return False
    if _positive_int(row.get("backtest_folds")) is None:
        return False
    backtest_rows = _positive_int(row.get("backtest_rows"))
    trained_at = _timestamp(row.get("trained_at"))
    if backtest_rows is None or trained_at is None or trained_at > snapshot_at:
        return False
    actual_model_age_hours = (snapshot_at - trained_at).total_seconds() / 3600.0
    if not math.isclose(
        float(values["model_age_hours"]),
        round(actual_model_age_hours, 2),
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        return False
    reliability = row.get("backtest_reliability")
    if not isinstance(reliability, list) or len(reliability) != 3:
        return False
    classes = set()
    calibration_errors: list[float] = []
    for item in reliability:
        if not isinstance(item, Mapping):
            return False
        class_index = item.get("class_index")
        calibration_error = _number(item.get("calibration_error"))
        if (
            isinstance(class_index, bool)
            or class_index not in (0, 1, 2)
            or class_index in classes
            or calibration_error is None
            or not 0.0 <= calibration_error <= 1.0
            or item.get("sample_count") != backtest_rows
        ):
            return False
        classes.add(class_index)
        calibration_errors.append(calibration_error)
    if classes != {0, 1, 2}:
        return False
    reliability_average = sum(calibration_errors) / len(calibration_errors)
    reliability_maximum = max(calibration_errors)
    if (
        not math.isclose(
            float(values["backtest_classwise_calibration_error"]),
            reliability_average,
            rel_tol=0.0,
            abs_tol=1e-4,
        )
        or not math.isclose(
            float(values["backtest_max_class_calibration_error"]),
            reliability_maximum,
            rel_tol=0.0,
            abs_tol=1e-4,
        )
    ):
        return False
    return (
        float(values["validation_accuracy"])
        >= float(values["baseline_accuracy"])
        and float(values["backtest_accuracy"])
        >= float(values["backtest_baseline_accuracy"])
        and float(values["backtest_calibration_error"])
        <= thresholds["maximum_calibration_error"]
        and reliability_maximum
        <= thresholds["maximum_classwise_calibration_error"]
        and actual_model_age_hours <= thresholds["maximum_model_age_hours"]
        and float(values["backtest_worst_fold_accuracy_lift"])
        >= thresholds["minimum_worst_fold_accuracy_lift"]
    )


def _sha256(value: Any) -> str | None:
    if not isinstance(value, str) or len(value) != 64:
        return None
    normalized = value.lower()
    return normalized if all(character in "0123456789abcdef" for character in normalized) else None


def _quality_model_identity(row: Mapping[str, Any]) -> tuple[str, str] | None:
    competition = row.get("competition")
    if (
        not isinstance(competition, str)
        or not competition
        or competition != competition.strip().upper()
        or isinstance(row.get("schema_version"), bool)
        or row.get("schema_version") != 3
    ):
        return None
    model_sha256 = _sha256(row.get("model_sha256"))
    return (competition, model_sha256) if model_sha256 is not None else None


def _prediction_model_identity(
    prediction: Mapping[str, Any],
) -> tuple[str, str] | None:
    model = prediction.get("model")
    if (
        _sha256(prediction.get("forecast_id")) is None
        or _positive_int(prediction.get("match_id")) is None
        or not isinstance(model, Mapping)
        or isinstance(model.get("schema_version"), bool)
        or model.get("schema_version") != 3
        or not forecast_id_matches(prediction)
    ):
        return None
    return _quality_model_identity(
        {
            "competition": prediction.get("competition"),
            "schema_version": model.get("schema_version"),
            "model_sha256": model.get("model_sha256"),
        }
    )


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


def _quality_thresholds(value: Any) -> dict[str, float] | None:
    if not isinstance(value, Mapping):
        return None
    names = (
        "maximum_calibration_error",
        "maximum_classwise_calibration_error",
        "maximum_model_age_hours",
        "minimum_worst_fold_accuracy_lift",
    )
    thresholds = {name: _number(value.get(name)) for name in names}
    if any(item is None for item in thresholds.values()):
        return None
    normalized = {name: float(item) for name, item in thresholds.items()}
    if (
        not 0.0 <= normalized["maximum_calibration_error"] <= 1.0
        or not 0.0 <= normalized["maximum_classwise_calibration_error"] <= 1.0
        or normalized["maximum_model_age_hours"] <= 0.0
        or not -1.0 <= normalized["minimum_worst_fold_accuracy_lift"] <= 1.0
    ):
        return None
    return normalized


def _unique_predictions(
    predictions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int, int]:
    unique: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}
    forecast_identities: dict[str, str] = {}
    duplicates = 0
    conflicts = 0
    missing = 0
    for index, item in enumerate(predictions):
        forecast_id = item.get("forecast_id")
        canonical_forecast_id = _sha256(forecast_id)
        if _prediction_model_identity(item) is None:
            missing += 1
        if item.get("competition") and item.get("match_id") is not None:
            identity = f"match:{item.get('competition')}:{item.get('match_id')}"
        elif canonical_forecast_id is not None:
            identity = f"forecast:{canonical_forecast_id}"
        else:
            identity = f"missing:{index}"
        if canonical_forecast_id is not None:
            normalized_forecast_id = canonical_forecast_id
            prior_identity = forecast_identities.get(normalized_forecast_id)
            if prior_identity is not None and prior_identity != identity:
                duplicates += 1
                conflicts += 1
                continue
            forecast_identities[normalized_forecast_id] = identity
        elif isinstance(forecast_id, str) and forecast_id.strip():
            normalized_forecast_id = forecast_id.strip()
            prior_identity = forecast_identities.get(normalized_forecast_id)
            if prior_identity is not None and prior_identity != identity:
                duplicates += 1
                conflicts += 1
                continue
            forecast_identities[normalized_forecast_id] = identity
        existing = seen.get(identity)
        if existing is None:
            row = dict(item)
            seen[identity] = row
            unique.append(row)
            continue
        duplicates += 1
        if existing != item:
            conflicts += 1
    return unique, duplicates, conflicts, missing


def _build_warnings(operations_summary: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"code": str(alert.get("code", "UNKNOWN")), "message": str(alert.get("message", "No details available."))}
        for alert in operations_summary.get("alerts", [])
        if alert.get("level") != "critical"
    ]


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


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
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
