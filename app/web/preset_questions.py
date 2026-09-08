from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HIGH_CONFIDENCE_THRESHOLD = 0.6


def _utc(now: datetime | None = None) -> datetime:
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    return current.astimezone(UTC)


def _today_key(now: datetime | None = None) -> str:
    return _utc(now).strftime("%Y-%m-%d")


def load_cached_questions(
    cache_path: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) and payload.get("date") == _today_key(now) else None


def build_preset_questions(
    predictions: list[dict[str, Any]],
    threshold: float = HIGH_CONFIDENCE_THRESHOLD,
    *,
    now: datetime | None = None,
    can_publish_recommendations: bool = False,
) -> dict[str, Any]:
    current = _utc(now)
    recommended = sorted(
        (
            prediction
            for prediction in predictions
            if can_publish_recommendations
            and isinstance(prediction.get("confidence"), (int, float))
            and not isinstance(prediction.get("confidence"), bool)
            and math.isfinite(float(prediction["confidence"]))
            and 0.0 <= float(prediction["confidence"]) <= 1.0
            and prediction["confidence"] >= threshold
        ),
        key=lambda item: item["confidence"],
        reverse=True,
    )
    recommendations = [
        {
            "match": f"{item.get('home_team')} vs {item.get('away_team')}",
            "competition": item.get("competition"),
            "prediction": item.get("prediction"),
            "confidence": item.get("confidence"),
        }
        for item in recommended[:10]
    ]
    if can_publish_recommendations and not recommendations:
        recommendations = [
            {
                "match": "No high-confidence picks yet.",
                "competition": None,
                "prediction": None,
                "confidence": None,
            }
        ]

    return {
        "schema": "SoccerAnalytics.PresetQuestions.v2",
        "date": current.strftime("%Y-%m-%d"),
        "generated_at": current.isoformat(),
        "can_publish_recommendations": can_publish_recommendations,
        "questions": [
            {
                "id": "recommended_bets_today",
                "title": "Recommended picks today",
                "answer": (
                    "High-confidence model picks from the latest verified snapshot."
                    if can_publish_recommendations
                    else "Recommendations are withheld because the release gate is blocked."
                ),
                "items": recommendations,
            },
            {
                "id": "matches_scanned",
                "title": "How many matches were scanned?",
                "answer": f"{len(predictions)} upcoming fixtures were evaluated.",
                "items": [],
            },
        ],
    }
