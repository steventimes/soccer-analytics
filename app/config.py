from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

DEFAULT_COMPETITIONS_MAP = {
    "CL": 2001,   # 欧冠
    "PL": 2021,   # 英超
    "PPL": 2017,  # 葡超
    "DED": 2003,  # 荷甲
    "BL1": 2002,  # 德甲
    "FL1": 2015,  # 法甲
    "SA": 2019,   # 意甲
    "PD": 2014,   # 西甲
    "BSA": 2013,  # 巴甲
    "ELC": 2016,  # 英冠
    "EC": 2018,   # 欧洲杯
    "WC": 2000,   # 世界杯
}

class ConfigurationError(ValueError):
    """配置值不合法，继续运行会让管线产生不可预测结果。"""


def default_training_seasons(today: date | None = None) -> list[str]:
    """返回最近四个可用于训练的欧洲赛季起始年，避免默认年份长期过期。"""
    current = today or date.today()
    latest_start_year = current.year - 1 if current.month >= 7 else current.year - 2
    return [str(year) for year in range(latest_start_year - 3, latest_start_year + 1)]


DEFAULT_SEASONS = default_training_seasons()


@dataclass(frozen=True, kw_only=True)
class PipelineSettings:
    competitions_map: dict[str, int]
    training_seasons: list[str]
    prediction_days: int
    site_export_days: int
    request_timeout_seconds: float = 15.0
    request_retries: int = 3
    model_dir: Path = Path("models")
    site_data_dir: Path = Path("docs/data")


def _parts(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_seasons(raw: str | None) -> list[str]:
    if raw is None or not raw.strip():
        return list(DEFAULT_SEASONS)

    seasons = _parts(raw)
    invalid = [season for season in seasons if re.fullmatch(r"\d{4}", season) is None]
    if invalid:
        raise ConfigurationError(
            "SOCCER_ANALYTICS_TRAINING_SEASONS 包含非法赛季: " + ", ".join(invalid)
        )

    # dict 能去重且保持用户声明顺序，便于重现训练范围。
    return list(dict.fromkeys(seasons)) or list(DEFAULT_SEASONS)


def _parse_competitions(raw: str | None) -> dict[str, int]:
    if raw is None or not raw.strip():
        return dict(DEFAULT_COMPETITIONS_MAP)

    requested = list(dict.fromkeys(part.upper() for part in _parts(raw)))
    if not requested:
        return dict(DEFAULT_COMPETITIONS_MAP)
    unknown = [code for code in requested if code not in DEFAULT_COMPETITIONS_MAP]
    if unknown:
        raise ConfigurationError(
            "SOCCER_ANALYTICS_COMPETITIONS 包含未知联赛代码: " + ", ".join(unknown)
        )
    return {code: DEFAULT_COMPETITIONS_MAP[code] for code in requested}


def _parse_positive_int(raw: str | None, default: int, variable: str) -> int:
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{variable} 必须是正整数，实际为 {raw!r}") from exc
    if value <= 0:
        raise ConfigurationError(f"{variable} 必须大于 0，实际为 {value}")
    return value


def _parse_positive_float(raw: str | None, default: float, variable: str) -> float:
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{variable} 必须是正数，实际为 {raw!r}") from exc
    if not math.isfinite(value) or value <= 0:
        raise ConfigurationError(f"{variable} 必须是有限正数，实际为 {raw!r}")
    return value


def _parse_path(raw: str | None, default: str) -> Path:
    value = raw.strip() if raw and raw.strip() else default
    return Path(value).expanduser()


def load_settings() -> PipelineSettings:
    """每次调用都读取环境变量，测试和多进程任务不会共享陈旧配置。"""
    return PipelineSettings(
        competitions_map=_parse_competitions(os.getenv("SOCCER_ANALYTICS_COMPETITIONS")),
        training_seasons=_parse_seasons(os.getenv("SOCCER_ANALYTICS_TRAINING_SEASONS")),
        prediction_days=_parse_positive_int(
            os.getenv("SOCCER_ANALYTICS_PREDICTION_DAYS"),
            3,
            "SOCCER_ANALYTICS_PREDICTION_DAYS",
        ),
        site_export_days=_parse_positive_int(
            os.getenv("SOCCER_ANALYTICS_SITE_EXPORT_DAYS"),
            1,
            "SOCCER_ANALYTICS_SITE_EXPORT_DAYS",
        ),
        request_timeout_seconds=_parse_positive_float(
            os.getenv("SOCCER_ANALYTICS_REQUEST_TIMEOUT"),
            15.0,
            "SOCCER_ANALYTICS_REQUEST_TIMEOUT",
        ),
        request_retries=_parse_positive_int(
            os.getenv("SOCCER_ANALYTICS_REQUEST_RETRIES"),
            3,
            "SOCCER_ANALYTICS_REQUEST_RETRIES",
        ),
        model_dir=_parse_path(os.getenv("SOCCER_ANALYTICS_MODEL_DIR"), "models"),
        site_data_dir=_parse_path(
            os.getenv("SOCCER_ANALYTICS_SITE_DATA_DIR"),
            "docs/data",
        ),
    )


def resolve_competitions(
    raw: str | None,
    settings: PipelineSettings | None = None,
) -> dict[str, int]:
    active_settings = settings or load_settings()
    if raw is None or not raw.strip():
        return dict(active_settings.competitions_map)

    requested = list(dict.fromkeys(part.upper() for part in _parts(raw)))
    if not requested:
        return dict(active_settings.competitions_map)
    unknown = [code for code in requested if code not in active_settings.competitions_map]
    if unknown:
        raise ConfigurationError("未知或未启用的联赛代码: " + ", ".join(unknown))
    return {code: active_settings.competitions_map[code] for code in requested}
