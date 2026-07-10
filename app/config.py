from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_COMPETITIONS_MAP = {
    "CL": 2001,  # Champions League
    "PL": 2021,  # Premier League
    "PPL": 2017, # Primeira Liga
    "DED": 2003, # Eredivisie
    "BL1": 2002, # Bundesliga
    "FL1": 2015, # Ligue 1
    "SA": 2019,  # Serie A
    "PD": 2014,  # La Liga
    "BSA": 2013, # Série A (Brazil)
    "ELC": 2016, # Championship
    "EC": 2018,  # European Championship
    "WC": 2000   # World Cup
}

UNDERSTAT_LEAGUE_MAP = {
    "PL": "EPL",      # Premier League -> English Premier League
    "PD": "La_liga",  # La Liga
    "BL1": "Bundesliga",
    "SA": "Serie_A",
    "FL1": "Ligue_1",
    "PPL": "RFPL"     # Russian Premier League
}

DEFAULT_SEASONS = [str(x) for x in range(2021, 2025)]


@dataclass(frozen=True)
class PipelineSettings:
    competitions_map: dict[str, int]
    training_seasons: list[str]
    prediction_days: int
    site_export_days: int


def _parse_seasons(raw: str | None) -> list[str]:
    if not raw:
        return list(DEFAULT_SEASONS)
    seasons = [part.strip() for part in raw.split(",") if part.strip()]
    return seasons or list(DEFAULT_SEASONS)


def _parse_competitions(raw: str | None) -> dict[str, int]:
    if not raw:
        return dict(DEFAULT_COMPETITIONS_MAP)
    requested_codes = [part.strip().upper() for part in raw.split(",") if part.strip()]
    if not requested_codes:
        return dict(DEFAULT_COMPETITIONS_MAP)
    return {
        code: comp_id
        for code, comp_id in DEFAULT_COMPETITIONS_MAP.items()
        if code in requested_codes
    }


def _parse_positive_int(raw: str | None, default: int) -> int:
    if raw is None or raw.strip() == "":
        return default
    value = int(raw)
    return value if value > 0 else default


def load_settings() -> PipelineSettings:
    return PipelineSettings(
        competitions_map=_parse_competitions(os.getenv("SOCCER_ANALYTICS_COMPETITIONS")),
        training_seasons=_parse_seasons(os.getenv("SOCCER_ANALYTICS_TRAINING_SEASONS")),
        prediction_days=_parse_positive_int(os.getenv("SOCCER_ANALYTICS_PREDICTION_DAYS"), 3),
        site_export_days=_parse_positive_int(os.getenv("SOCCER_ANALYTICS_SITE_EXPORT_DAYS"), 1),
    )


def resolve_competitions(raw: str | None, settings: PipelineSettings | None = None) -> dict[str, int]:
    active_settings = settings or load_settings()
    if not raw:
        return dict(active_settings.competitions_map)

    requested_codes = [part.strip().upper() for part in raw.split(",") if part.strip()]
    resolved = {
        code: comp_id
        for code, comp_id in active_settings.competitions_map.items()
        if code in requested_codes
    }
    return resolved or dict(active_settings.competitions_map)


_settings = load_settings()
COMPETITIONS_MAP = _settings.competitions_map
SEASONS = list(_settings.training_seasons)
TRAINING_SEASONS = list(_settings.training_seasons)
