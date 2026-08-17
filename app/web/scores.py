from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import requests


@dataclass(frozen=True)
class ScoreFetchResult:
    scores: tuple[dict[str, Any], ...]
    errors: tuple[dict[str, str], ...]


def fetch_daily_scores_with_diagnostics(
    date: datetime | None = None,
) -> ScoreFetchResult:
    """抓取当日比分，并保留足以定位上游故障的结构化错误。"""
    target_date = (date or datetime.now(UTC)).strftime("%Y-%m-%d")
    url = "https://www.thesportsdb.com/api/v1/json/3/eventsday.php"
    params = {"d": target_date, "s": "Soccer"}

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        payload = response.json()
    except requests.exceptions.JSONDecodeError as exc:
        # Requests 的 JSONDecodeError 同时继承 RequestException，必须先分类。
        return ScoreFetchResult(
            scores=(),
            errors=(
                {
                    "code": "SCORE_SOURCE_INVALID_PAYLOAD",
                    "message": f"比分源返回的 JSON 无法解析: {exc}",
                },
            ),
        )
    except requests.RequestException as exc:
        return ScoreFetchResult(
            scores=(),
            errors=(
                {
                    "code": "SCORE_SOURCE_UNAVAILABLE",
                    "message": f"比分源请求失败: {exc}",
                },
            ),
        )
    except ValueError as exc:
        return ScoreFetchResult(
            scores=(),
            errors=(
                {
                    "code": "SCORE_SOURCE_INVALID_PAYLOAD",
                    "message": f"比分源返回的 JSON 无法解析: {exc}",
                },
            ),
        )
    if not isinstance(payload, dict):
        return ScoreFetchResult(
            scores=(),
            errors=(
                {
                    "code": "SCORE_SOURCE_INVALID_PAYLOAD",
                    "message": "比分源响应必须是 JSON 对象",
                },
            ),
        )

    events = payload.get("events")
    if events is None:
        events = []
    elif not isinstance(events, list):
        return ScoreFetchResult(
            scores=(),
            errors=(
                {
                    "code": "SCORE_SOURCE_INVALID_PAYLOAD",
                    "message": "比分源响应的 events 字段必须是数组",
                },
            ),
        )
    results = []
    errors: list[dict[str, str]] = []
    for index, event in enumerate(events):
        # 上游偶尔会混入缺少对阵双方的占位事件，不能把空比分写入站点。
        if not isinstance(event, dict) or not event.get("strHomeTeam") or not event.get("strAwayTeam"):
            event_id = event.get("idEvent") if isinstance(event, dict) else None
            identifier = str(event_id or f"index-{index}")
            errors.append(
                {
                    "code": "SCORE_EVENT_INVALID",
                    "message": f"已跳过缺少对阵双方的比分事件: {identifier}",
                }
            )
            continue
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
    return ScoreFetchResult(scores=tuple(results), errors=tuple(errors))
