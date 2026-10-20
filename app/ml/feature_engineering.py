from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from statistics import median
from typing import Any, Iterable, Mapping

from app.data_service.match_records import (
    FeatureEngineeringError,
    MatchRecord,
    _parse_datetime,
    _positive_team_id,
    _score,
    normalize_match,
)






@dataclass(frozen=True)
class FeatureDataset:
    rows: tuple[dict[str, float], ...]
    targets: tuple[int, ...]
    match_ids: tuple[int, ...]
    team_states: dict[str, dict[str, Any]]
    feature_defaults: dict[str, float]
    kickoffs: tuple[str, ...] = ()


@dataclass
class _TeamState:
    elo: float = 1500.0
    matches_played: int = 0
    last_match_at: datetime | None = None
    recent: list[dict[str, float | str]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> _TeamState:
        if not isinstance(payload, Mapping):
            raise FeatureEngineeringError("球队状态必须是对象")

        def finite_number(value: Any, label: str, *, minimum: float | None = None) -> float:
            if isinstance(value, bool):
                raise FeatureEngineeringError(f"球队状态 {label} 非法")
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise FeatureEngineeringError(f"球队状态 {label} 非法") from exc
            if not math.isfinite(numeric) or (minimum is not None and numeric < minimum):
                raise FeatureEngineeringError(f"球队状态 {label} 非法")
            return numeric

        elo = finite_number(payload.get("elo", 1500.0), "elo")
        raw_matches = payload.get("matches_played", 0)
        if isinstance(raw_matches, bool) or not isinstance(raw_matches, int) or raw_matches < 0:
            raise FeatureEngineeringError("球队状态 matches_played 非法")
        recent_payload = payload.get("recent", [])
        if not isinstance(recent_payload, list):
            raise FeatureEngineeringError("球队状态 recent 必须是数组")
        recent: list[dict[str, float | str]] = []
        for item in recent_payload:
            if not isinstance(item, Mapping):
                raise FeatureEngineeringError("球队近期记录必须是对象")
            missing = {"points", "goals_for", "goals_against", "venue"} - set(item)
            if missing:
                raise FeatureEngineeringError("球队近期记录缺少字段: " + ", ".join(sorted(missing)))
            points = finite_number(item["points"], "points", minimum=0.0)
            if points not in (0.0, 1.0, 3.0):
                raise FeatureEngineeringError("球队状态 points 只能是 0、1、3")
            venue = item["venue"]
            if venue not in ("H", "A"):
                raise FeatureEngineeringError("球队状态 venue 只能是 H 或 A")
            recent.append(
                {
                    "points": points,
                    "goals_for": finite_number(item["goals_for"], "goals_for", minimum=0.0),
                    "goals_against": finite_number(item["goals_against"], "goals_against", minimum=0.0),
                    "venue": venue,
                }
            )
        last_match = payload.get("last_match_at")
        parsed_last_match = (
            None if last_match is None else _parse_datetime(last_match)
        )
        if len(recent) > raw_matches:
            raise FeatureEngineeringError("球队近期记录不能多于已比赛场次")
        if (raw_matches == 0) != (parsed_last_match is None):
            raise FeatureEngineeringError(
                "球队状态 matches_played 与 last_match_at 不一致"
            )
        return cls(
            elo=elo,
            matches_played=raw_matches,
            last_match_at=parsed_last_match,
            recent=recent,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "elo": self.elo,
            "matches_played": self.matches_played,
            "last_match_at": self.last_match_at.isoformat() if self.last_match_at else None,
            "recent": [dict(item) for item in self.recent],
        }








def validate_team_states(
    team_states: Mapping[str, Mapping[str, Any]],
) -> dict[int, _TeamState]:
    """把持久化球队状态严格还原，供资产加载和实时特征共用。"""
    if not isinstance(team_states, Mapping):
        raise FeatureEngineeringError("team_states 必须是对象")
    states: dict[int, _TeamState] = {}
    required_fields = {"elo", "matches_played", "last_match_at", "recent"}
    for team_id, payload in team_states.items():
        normalized_id = _positive_team_id(team_id, "状态")
        if not isinstance(team_id, str) or team_id != str(normalized_id):
            raise FeatureEngineeringError("球队状态 ID 必须是规范正整数字符串")
        if not isinstance(payload, Mapping):
            raise FeatureEngineeringError("球队状态必须是对象")
        missing = required_fields - set(payload)
        if missing:
            raise FeatureEngineeringError(
                "球队状态缺少字段: " + ", ".join(sorted(missing))
            )
        if normalized_id in states:
            raise FeatureEngineeringError("球队状态 ID 不能重复")
        states[normalized_id] = _TeamState.from_dict(payload)
    return states




class FeatureEngineer:
    BASE_ELO = 1500.0
    ELO_K = 20.0
    HOME_ELO_ADVANTAGE = 80.0
    DEFAULT_REST_DAYS = 7.0
    MAX_REST_DAYS = 21.0

    FEATURES = (
        "home_elo",
        "away_elo",
        "elo_diff",
        "home_form_points",
        "away_form_points",
        "form_points_diff",
        "home_goals_for",
        "away_goals_for",
        "attack_diff",
        "home_goals_against",
        "away_goals_against",
        "defense_diff",
        "home_goal_difference",
        "away_goal_difference",
        "goal_difference_diff",
        "home_win_rate",
        "away_win_rate",
        "win_rate_diff",
        "home_clean_sheet_rate",
        "away_clean_sheet_rate",
        "home_rest_days",
        "away_rest_days",
        "rest_diff",
        "home_venue_points",
        "away_venue_points",
        "venue_points_diff",
        "home_matches_seen",
        "away_matches_seen",
        "experience_diff",
        "home_advantage",
    )

    def __init__(self, window: int = 5) -> None:
        if window <= 0:
            raise FeatureEngineeringError("滚动窗口必须大于 0")
        self.window = window
        self.features = self.FEATURES

    def build_training_dataset(
        self,
        matches: Iterable[Mapping[str, Any] | MatchRecord],
    ) -> FeatureDataset:
        unique: dict[int, MatchRecord] = {}
        for payload in matches:
            record = normalize_match(payload)
            existing = unique.get(record.match_id)
            if existing is not None and existing != record:
                raise FeatureEngineeringError(f"比赛 {record.match_id} 存在冲突重复记录")
            unique[record.match_id] = record

        records = sorted(
            (record for record in unique.values() if record.status == "FINISHED"),
            key=lambda record: (record.kickoff, record.match_id),
        )
        states: dict[int, _TeamState] = {}
        rows: list[dict[str, float]] = []
        targets: list[int] = []
        match_ids: list[int] = []

        bucket_start = 0
        while bucket_start < len(records):
            bucket_end = bucket_start + 1
            while (
                bucket_end < len(records)
                and records[bucket_end].kickoff == records[bucket_start].kickoff
            ):
                bucket_end += 1
            bucket = records[bucket_start:bucket_end]
            # 同一开球时间桶必须共享完全相同的赛前状态，ID 顺序不能泄漏赛果。
            for record in bucket:
                rows.append(
                    self._build_features(
                        record.home_team_id,
                        record.away_team_id,
                        record.kickoff,
                        states,
                    )
                )
                targets.append(self._target(record))
                match_ids.append(record.match_id)
            for record in bucket:
                self._update_states(record, states)
            bucket_start = bucket_end

        defaults = {
            name: float(median([row[name] for row in rows])) if rows else self._neutral_default(name)
            for name in self.features
        }
        return FeatureDataset(
            rows=tuple(rows),
            targets=tuple(targets),
            match_ids=tuple(match_ids),
            team_states={str(team_id): state.to_dict() for team_id, state in states.items()},
            feature_defaults=defaults,
            kickoffs=tuple(record.kickoff.isoformat() for record in records),
        )

    def build_fixture_features(
        self,
        home_team_id: int,
        away_team_id: int,
        kickoff: datetime | str,
        team_states: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, float]:
        home_id = _positive_team_id(home_team_id, "主队")
        away_id = _positive_team_id(away_team_id, "客队")
        if home_id == away_id:
            raise FeatureEngineeringError("主客队不能相同")
        states = validate_team_states(team_states)
        return self._build_features(home_id, away_id, _parse_datetime(kickoff), states)

    def _state(self, states: dict[int, _TeamState], team_id: int) -> _TeamState:
        return states.get(team_id, _TeamState())

    def _aggregates(self, state: _TeamState, venue: str) -> dict[str, float]:
        recent = state.recent[-self.window :]
        if not recent:
            return {
                "points": 1.0,
                "goals_for": 1.25,
                "goals_against": 1.25,
                "goal_difference": 0.0,
                "win_rate": 1.0 / 3.0,
                "clean_sheet_rate": 0.2,
                "venue_points": 1.0,
            }

        count = float(len(recent))
        venue_rows = [item for item in recent if item["venue"] == venue]
        return {
            "points": sum(float(item["points"]) for item in recent) / count,
            "goals_for": sum(float(item["goals_for"]) for item in recent) / count,
            "goals_against": sum(float(item["goals_against"]) for item in recent) / count,
            "goal_difference": sum(
                float(item["goals_for"]) - float(item["goals_against"])
                for item in recent
            )
            / count,
            "win_rate": sum(float(item["points"]) == 3.0 for item in recent) / count,
            "clean_sheet_rate": sum(float(item["goals_against"]) == 0.0 for item in recent)
            / count,
            "venue_points": (
                sum(float(item["points"]) for item in venue_rows) / len(venue_rows)
                if venue_rows
                else 1.0
            ),
        }

    def _rest_days(self, state: _TeamState, kickoff: datetime) -> float:
        if state.last_match_at is None:
            return self.DEFAULT_REST_DAYS
        elapsed = (kickoff - state.last_match_at).total_seconds() / 86400.0
        if elapsed < 0:
            raise FeatureEngineeringError("模型球队状态晚于待预测比赛时间")
        return min(self.MAX_REST_DAYS, elapsed)

    def _build_features(
        self,
        home_id: int,
        away_id: int,
        kickoff: datetime,
        states: dict[int, _TeamState],
    ) -> dict[str, float]:
        home = self._state(states, home_id)
        away = self._state(states, away_id)
        home_form = self._aggregates(home, "H")
        away_form = self._aggregates(away, "A")
        home_rest = self._rest_days(home, kickoff)
        away_rest = self._rest_days(away, kickoff)

        values = {
            "home_elo": home.elo,
            "away_elo": away.elo,
            "elo_diff": home.elo - away.elo,
            "home_form_points": home_form["points"],
            "away_form_points": away_form["points"],
            "form_points_diff": home_form["points"] - away_form["points"],
            "home_goals_for": home_form["goals_for"],
            "away_goals_for": away_form["goals_for"],
            "attack_diff": home_form["goals_for"] - away_form["goals_for"],
            "home_goals_against": home_form["goals_against"],
            "away_goals_against": away_form["goals_against"],
            "defense_diff": away_form["goals_against"] - home_form["goals_against"],
            "home_goal_difference": home_form["goal_difference"],
            "away_goal_difference": away_form["goal_difference"],
            "goal_difference_diff": home_form["goal_difference"] - away_form["goal_difference"],
            "home_win_rate": home_form["win_rate"],
            "away_win_rate": away_form["win_rate"],
            "win_rate_diff": home_form["win_rate"] - away_form["win_rate"],
            "home_clean_sheet_rate": home_form["clean_sheet_rate"],
            "away_clean_sheet_rate": away_form["clean_sheet_rate"],
            "home_rest_days": home_rest,
            "away_rest_days": away_rest,
            "rest_diff": home_rest - away_rest,
            "home_venue_points": home_form["venue_points"],
            "away_venue_points": away_form["venue_points"],
            "venue_points_diff": home_form["venue_points"] - away_form["venue_points"],
            "home_matches_seen": float(home.matches_played),
            "away_matches_seen": float(away.matches_played),
            "experience_diff": float(home.matches_played - away.matches_played),
            "home_advantage": 1.0,
        }
        ordered = {name: float(values[name]) for name in self.features}
        if any(not math.isfinite(value) for value in ordered.values()):
            raise FeatureEngineeringError("特征包含非有限数值")
        return ordered

    def _target(self, record: MatchRecord) -> int:
        assert record.home_score is not None and record.away_score is not None
        if record.home_score > record.away_score:
            return 2
        if record.home_score == record.away_score:
            return 1
        return 0

    def _update_states(
        self,
        record: MatchRecord,
        states: dict[int, _TeamState],
    ) -> None:
        assert record.home_score is not None and record.away_score is not None
        home = states.setdefault(record.home_team_id, _TeamState())
        away = states.setdefault(record.away_team_id, _TeamState())
        if record.home_score > record.away_score:
            home_points, away_points, actual_home = 3.0, 0.0, 1.0
        elif record.home_score == record.away_score:
            home_points, away_points, actual_home = 1.0, 1.0, 0.5
        else:
            home_points, away_points, actual_home = 0.0, 3.0, 0.0

        expected_home = 1.0 / (
            1.0
            + 10.0
            ** ((away.elo - (home.elo + self.HOME_ELO_ADVANTAGE)) / 400.0)
        )
        change = self.ELO_K * (actual_home - expected_home)
        home.elo += change
        away.elo -= change

        home.recent.append(
            {
                "points": home_points,
                "goals_for": float(record.home_score),
                "goals_against": float(record.away_score),
                "venue": "H",
            }
        )
        away.recent.append(
            {
                "points": away_points,
                "goals_for": float(record.away_score),
                "goals_against": float(record.home_score),
                "venue": "A",
            }
        )
        # 模型包只保留计算窗口所需的最小历史，避免资产随赛季无限膨胀。
        home.recent = home.recent[-self.window :]
        away.recent = away.recent[-self.window :]
        home.matches_played += 1
        away.matches_played += 1
        home.last_match_at = record.kickoff
        away.last_match_at = record.kickoff

    def _neutral_default(self, name: str) -> float:
        if name in {"home_elo", "away_elo"}:
            return self.BASE_ELO
        if name in {"home_form_points", "away_form_points", "home_venue_points", "away_venue_points"}:
            return 1.0
        if name in {"home_goals_for", "away_goals_for", "home_goals_against", "away_goals_against"}:
            return 1.25
        if name in {"home_win_rate", "away_win_rate"}:
            return 1.0 / 3.0
        if name in {"home_clean_sheet_rate", "away_clean_sheet_rate"}:
            return 0.2
        if name in {"home_rest_days", "away_rest_days"}:
            return self.DEFAULT_REST_DAYS
        if name == "home_advantage":
            return 1.0
        return 0.0
