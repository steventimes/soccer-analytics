from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Any


def build_operations_summary(
    predictions: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    *,
    generated_at: str | None = None,
    high_confidence_threshold: float = 0.6,
    low_confidence_threshold: float = 0.5,
) -> dict[str, Any]:
    generated = generated_at or datetime.now(UTC).isoformat()
    confidence_values = [
        float(item["confidence"])
        for item in predictions
        if _is_number(item.get("confidence"))
    ]
    competitions = Counter(
        str(item.get("competition"))
        for item in predictions
        if item.get("competition")
    )
    high_confidence = [value for value in confidence_values if value >= high_confidence_threshold]
    low_confidence = [value for value in confidence_values if value < low_confidence_threshold]

    alerts = _build_alerts(
        prediction_count=len(predictions),
        score_count=len(scores),
        confidence_count=len(confidence_values),
        high_confidence_count=len(high_confidence),
        low_confidence_count=len(low_confidence),
        competition_count=len(competitions),
    )

    return {
        "generated_at": generated,
        "status": "healthy" if not alerts else "attention",
        "prediction_count": len(predictions),
        "score_count": len(scores),
        "competition_count": len(competitions),
        "average_confidence": _round_or_none(_average(confidence_values)),
        "high_confidence_count": len(high_confidence),
        "low_confidence_count": len(low_confidence),
        "coverage_by_competition": [
            {"competition": competition, "prediction_count": count}
            for competition, count in sorted(competitions.items())
        ],
        "alerts": alerts,
        "thresholds": {
            "high_confidence": high_confidence_threshold,
            "low_confidence": low_confidence_threshold,
        },
    }


def _build_alerts(
    *,
    prediction_count: int,
    score_count: int,
    confidence_count: int,
    high_confidence_count: int,
    low_confidence_count: int,
    competition_count: int,
) -> list[dict[str, str]]:
    alerts: list[dict[str, str]] = []

    if prediction_count == 0:
        alerts.append(
            {
                "level": "critical",
                "code": "NO_PREDICTIONS",
                "message": "No upcoming predictions were exported; check model artifacts and fixture ingestion.",
            }
        )
    if score_count == 0:
        alerts.append(
            {
                "level": "warning",
                "code": "NO_SCORES",
                "message": "No latest scores were exported; verify the scores data source or cache.",
            }
        )
    if prediction_count > 0 and confidence_count < prediction_count:
        alerts.append(
            {
                "level": "warning",
                "code": "MISSING_CONFIDENCE",
                "message": "Some predictions are missing numeric confidence values.",
            }
        )
    if prediction_count > 0 and high_confidence_count == 0:
        alerts.append(
            {
                "level": "warning",
                "code": "NO_HIGH_CONFIDENCE_PICKS",
                "message": "No prediction cleared the high-confidence threshold for recommendations.",
            }
        )
    if prediction_count > 0 and low_confidence_count > prediction_count / 2:
        alerts.append(
            {
                "level": "warning",
                "code": "LOW_CONFIDENCE_CLUSTER",
                "message": "More than half of exported predictions are below the low-confidence threshold.",
            }
        )
    if prediction_count > 0 and competition_count == 0:
        alerts.append(
            {
                "level": "warning",
                "code": "MISSING_COMPETITION_LABELS",
                "message": "Predictions are missing competition labels, which weakens filtering and reporting.",
            }
        )

    return alerts


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _round_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 4)
