from __future__ import annotations

from datetime import datetime
from typing import Any

import requests


def fetch_daily_scores(date: datetime | None = None) -> list[dict[str, Any]]:
    target_date = (date or datetime.utcnow()).strftime("%Y-%m-%d")
    url = "https://www.thesportsdb.com/api/v1/json/3/eventsday.php"
    params = {"d": target_date, "s": "Soccer"}

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return []

    events = payload.get("events") or []
    results = []
    for event in events:
        results.append(
            {
                "league": event.get("strLeague"),
                "home_team": event.get("strHomeTeam"),
                "away_team": event.get("strAwayTeam"),
                "home_score": event.get("intHomeScore"),
                "away_score": event.get("intAwayScore"),
                "time": event.get("strTime"),
                "date": event.get("dateEvent"),
            }
        )
    return results