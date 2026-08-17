from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from typing import Any, Mapping, Sequence


class ForecastIdentityError(ValueError):
    """预测血缘字段无法构成可重放的规范摘要。"""


def _sha256(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ForecastIdentityError("SHA-256 必须是 64 位十六进制字符串")
    normalized = value.lower()
    if any(character not in "0123456789abcdef" for character in normalized):
        raise ForecastIdentityError("SHA-256 包含非法字符")
    return normalized


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ForecastIdentityError(f"{label} 必须是正整数")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ForecastIdentityError(f"{label} 必须是有限数值")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ForecastIdentityError(f"{label} 必须是有限数值") from exc
    if not math.isfinite(numeric):
        raise ForecastIdentityError(f"{label} 必须是有限数值")
    return numeric


def _timestamp(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ForecastIdentityError(f"{label} 必须是带时区时间")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ForecastIdentityError(f"{label} 必须是带时区时间") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ForecastIdentityError(f"{label} 必须是带时区时间")
    return parsed.astimezone(UTC).isoformat()


def _digest(payload: Mapping[str, Any]) -> str:
    material = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def feature_fingerprint(
    feature_names: Sequence[str],
    features: Mapping[str, Any],
) -> str:
    names = tuple(feature_names)
    if (
        not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
        or set(features) != set(names)
    ):
        raise ForecastIdentityError("特征 schema 与特征值不一致")
    return _digest(
        {
            "feature_schema": list(names),
            "features": [
                _finite_number(features[name], f"特征 {name}") for name in names
            ],
        }
    )


def cold_start_evidence(prediction: Mapping[str, Any]) -> bool:
    quality = prediction.get("data_quality")
    if not isinstance(quality, Mapping):
        raise ForecastIdentityError("data_quality 必须是对象")
    cold_start = quality.get("is_cold_start")
    if not isinstance(cold_start, bool):
        raise ForecastIdentityError("is_cold_start 必须是布尔值")
    history_values: dict[str, int] = {}
    for name in ("home_history_matches", "away_history_matches"):
        value = quality.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ForecastIdentityError(f"{name} 必须是非负整数")
        history_values[name] = value
    expected = (
        history_values["home_history_matches"] == 0
        or history_values["away_history_matches"] == 0
    )
    if cold_start is not expected:
        raise ForecastIdentityError("is_cold_start 与球队历史场次不一致")
    return cold_start


def forecast_id_for_prediction(prediction: Mapping[str, Any]) -> str:
    competition = prediction.get("competition")
    model = prediction.get("model")
    probabilities = prediction.get("probabilities")
    if (
        not isinstance(competition, str)
        or not competition
        or competition != competition.strip().upper()
        or not isinstance(model, Mapping)
        or isinstance(model.get("schema_version"), bool)
        or model.get("schema_version") != 3
        or not isinstance(probabilities, Mapping)
        or set(probabilities) != {"home_win", "draw", "away_win"}
    ):
        raise ForecastIdentityError("预测缺少规范联赛、模型或概率字段")
    home_team_id = _positive_int(prediction.get("home_team_id"), "主队 ID")
    away_team_id = _positive_int(prediction.get("away_team_id"), "客队 ID")
    if home_team_id == away_team_id:
        raise ForecastIdentityError("主客队 ID 不能相同")
    probability_values = {
        name: _finite_number(probabilities.get(name), f"概率 {name}")
        for name in ("home_win", "draw", "away_win")
    }
    confidence = _finite_number(prediction.get("confidence"), "confidence")
    outcome = prediction.get("prediction")
    if outcome not in ("AWAY_WIN", "DRAW", "HOME_WIN"):
        raise ForecastIdentityError("prediction 类别非法")
    cold_start = cold_start_evidence(prediction)
    quality = prediction["data_quality"]
    return _digest(
        {
            "schema": "SoccerAnalytics.ForecastIdentity.v2",
            "competition": competition,
            "match_id": _positive_int(prediction.get("match_id"), "比赛 ID"),
            "kickoff": _timestamp(prediction.get("utc_date"), "开球时间"),
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "trained_at": _timestamp(model.get("trained_at"), "模型训练时间"),
            "model_schema_version": 3,
            "model_sha256": _sha256(model.get("model_sha256")),
            "feature_fingerprint": _sha256(
                prediction.get("feature_fingerprint")
            ),
            "probabilities": probability_values,
            "confidence": confidence,
            "prediction": outcome,
            "data_quality": {
                "is_cold_start": cold_start,
                "home_history_matches": quality["home_history_matches"],
                "away_history_matches": quality["away_history_matches"],
            },
        }
    )


def forecast_id_matches(prediction: Mapping[str, Any]) -> bool:
    try:
        supplied = _sha256(prediction.get("forecast_id"))
        expected = forecast_id_for_prediction(prediction)
    except ForecastIdentityError:
        return False
    return supplied == expected
