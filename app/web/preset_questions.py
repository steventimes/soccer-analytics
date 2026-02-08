from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import json


def _today_key() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")


def load_cached_questions(cache_path: Path) -> dict[str, Any] | None:
    if not cache_path.exists():
        return None

    try:
        payload = json.loads(cache_path.read_text())
    except json.JSONDecodeError:
        return None

    if payload.get("date") != _today_key():
        return None

    return payload


def build_preset_questions(
    predictions: list[dict[str, Any]],
    threshold: float = 0.55,
) -> dict[str, Any]:
    recommended = [
        prediction
        for prediction in predictions
        if prediction.get("confidence", 0) >= threshold
    ]
    recommended = sorted(
        recommended, key=lambda item: item.get("confidence", 0), reverse=True
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

    if not recommendations:
        recommendations = [
            {
                "match": "No high-confidence picks yet.",
                "competition": None,
                "prediction": None,
                "confidence": None,
            }
        ]

    questions = [
        {
            "id": "recommended_bets_today",
            "title": "Recommended bettings today",
            "answer": "High-confidence model picks generated in the morning.",
            "items": recommendations,
        },
        {
            "id": "matches_scanned",
            "title": "How many matches were scanned?",
            "answer": f"{len(predictions)} upcoming fixtures were evaluated.",
            "items": [],
        },
    ]

    return {
        "date": _today_key(),
        "generated_at": datetime.utcnow().isoformat(),
        "questions": questions,
    }