from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def build_release_governance(
    predictions: list[dict[str, Any]],
    operations_summary: dict[str, Any],
    *,
    generated_at: str | None = None,
    minimum_predictions: int = 3,
    minimum_high_confidence_picks: int = 1,
    minimum_average_confidence: float = 0.55,
) -> dict[str, Any]:
    generated = generated_at or datetime.now(UTC).isoformat()
    blockers = _build_blockers(
        predictions=predictions,
        operations_summary=operations_summary,
        minimum_predictions=minimum_predictions,
        minimum_high_confidence_picks=minimum_high_confidence_picks,
        minimum_average_confidence=minimum_average_confidence,
    )
    warnings = _build_warnings(operations_summary)
    approved = not blockers

    return {
        "schema": "SoccerAnalytics.ReleaseGovernance.v1",
        "generated_at": generated,
        "release_decision": "approved" if approved else "blocked",
        "can_publish_recommendations": approved,
        "blockers": blockers,
        "warnings": warnings,
        "thresholds": {
            "minimum_predictions": minimum_predictions,
            "minimum_high_confidence_picks": minimum_high_confidence_picks,
            "minimum_average_confidence": minimum_average_confidence,
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
            "operation_status": operations_summary.get("status", "unknown"),
            "average_confidence": operations_summary.get("average_confidence"),
            "high_confidence_count": operations_summary.get("high_confidence_count", 0),
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
    operations_summary: dict[str, Any],
    minimum_predictions: int,
    minimum_high_confidence_picks: int,
    minimum_average_confidence: float,
) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []

    if len(predictions) < minimum_predictions:
        blockers.append(
            {
                "code": "INSUFFICIENT_FIXTURE_COVERAGE",
                "message": f"Only {len(predictions)} predictions were exported; require at least {minimum_predictions} before publishing picks.",
            }
        )

    if operations_summary.get("status") != "healthy":
        blockers.append(
            {
                "code": "OPERATIONS_NOT_HEALTHY",
                "message": "Latest operations summary is not healthy; resolve data quality alerts before publishing.",
            }
        )

    high_confidence_count = int(operations_summary.get("high_confidence_count") or 0)
    if high_confidence_count < minimum_high_confidence_picks:
        blockers.append(
            {
                "code": "NO_PUBLISHABLE_PICKS",
                "message": f"Only {high_confidence_count} high-confidence picks cleared the threshold; require {minimum_high_confidence_picks}.",
            }
        )

    average_confidence = operations_summary.get("average_confidence")
    if average_confidence is None or float(average_confidence) < minimum_average_confidence:
        blockers.append(
            {
                "code": "LOW_AVERAGE_CONFIDENCE",
                "message": f"Average confidence is {average_confidence}; require at least {minimum_average_confidence}.",
            }
        )

    for alert in operations_summary.get("alerts", []):
        if alert.get("level") == "critical":
            blockers.append(
                {
                    "code": f"CRITICAL_ALERT_{alert.get('code', 'UNKNOWN')}",
                    "message": alert.get("message", "A critical operations alert is present."),
                }
            )

    return blockers


def _build_warnings(operations_summary: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "code": str(alert.get("code", "UNKNOWN")),
            "message": str(alert.get("message", "No details available.")),
        }
        for alert in operations_summary.get("alerts", [])
        if alert.get("level") != "critical"
    ]
