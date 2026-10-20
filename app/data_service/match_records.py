"""Canonical match records and validation, independent of prediction models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Mapping


class FeatureEngineeringError(ValueError):
    """比赛数据无法安全转换为赛前特征。"""


@dataclass(frozen=True)
class MatchRecord:
    match_id: int
    kickoff: datetime
    status: str
    home_team_id: int
    away_team_id: int
    home_score: int | None
    away_score: int | None


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise FeatureEngineeringError(f"非法比赛时间: {value!r}") from exc
    else:
        raise FeatureEngineeringError("比赛时间不能为空")

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _positive_team_id(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise FeatureEngineeringError(f"{label}球队 ID 非法")
    try:
        team_id = int(value)
    except (TypeError, ValueError) as exc:
        raise FeatureEngineeringError(f"{label}球队 ID 非法") from exc
    if team_id <= 0:
        raise FeatureEngineeringError(f"{label}球队 ID 必须大于 0")
    return team_id


def _score(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FeatureEngineeringError(f"已完成比赛缺少有效{label}比分")
    return value


def normalize_match(payload: Mapping[str, Any] | MatchRecord) -> MatchRecord:
    if isinstance(payload, MatchRecord):
        return payload
    if not isinstance(payload, Mapping):
        raise FeatureEngineeringError("比赛记录必须是对象")

    raw_id = payload.get("id", payload.get("match_id"))
    if isinstance(raw_id, bool):
        raise FeatureEngineeringError("比赛 ID 非法")
    try:
        match_id = int(raw_id)
    except (TypeError, ValueError) as exc:
        raise FeatureEngineeringError("比赛 ID 非法") from exc
    if match_id <= 0:
        raise FeatureEngineeringError("比赛 ID 必须大于 0")

    home_payload = payload.get("homeTeam") or {}
    away_payload = payload.get("awayTeam") or {}
    home_id = payload.get("home_team_id", home_payload.get("id"))
    away_id = payload.get("away_team_id", away_payload.get("id"))
    home_team_id = _positive_team_id(home_id, "主队")
    away_team_id = _positive_team_id(away_id, "客队")
    if home_team_id == away_team_id:
        raise FeatureEngineeringError("主客队不能相同")

    kickoff = _parse_datetime(payload.get("utcDate", payload.get("kickoff")))
    status = str(payload.get("status", "FINISHED")).upper()

    full_time = (payload.get("score") or {}).get("fullTime") or {}
    raw_home_score = payload.get("home_score", full_time.get("home"))
    raw_away_score = payload.get("away_score", full_time.get("away"))
    if status == "FINISHED":
        home_score = _score(raw_home_score, "主队")
        away_score = _score(raw_away_score, "客队")
    else:
        home_score = None
        away_score = None

    return MatchRecord(
        match_id=match_id,
        kickoff=kickoff,
        status=status,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        home_score=home_score,
        away_score=away_score,
    )
