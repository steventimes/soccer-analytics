"""Shared quality and evidence validation for operations and release governance."""
import math
from datetime import UTC, datetime
from typing import Any, Mapping


def probability_quality_passes(row):
    if row.get("schema_version") == 3:
        return True  # Historical evidence keeps its original, disclosed policy.
    if row.get("probability_baseline") != "training_frequency_laplace_v1" or row.get("probability_quality_passed") is not True:
        return False
    fields = ("validation_log_loss", "validation_brier_score",
              "validation_baseline_log_loss", "validation_baseline_brier_score")
    values = [row.get(name) for name in fields]
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) or v < 0 for v in values):
        return False
    loss, brier, baseline_loss, baseline_brier = values
    return loss <= baseline_loss + 1e-6 and brier <= baseline_brier + 1e-6


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


def _sha256(value: Any) -> str | None:
    if not isinstance(value, str) or len(value) != 64:
        return None
    normalized = value.lower()
    return normalized if all(character in "0123456789abcdef" for character in normalized) else None


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
