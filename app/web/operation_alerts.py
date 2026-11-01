"""Ordered release alerts composed from the operations summary evidence."""


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
