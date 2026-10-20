"""Key-free sources with provider-scoped, stable identities (never merge providers)."""

from __future__ import annotations

import csv
import hashlib
import io
import math
import os
import re
from datetime import UTC, date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import requests

from app.config import default_training_seasons
from app.data_service.fetch.fetcher import (
    DataSourceError,
    FootballDataClient,
    SlidingWindowRateLimiter,
)
from .retry import close_response

# Source identifiers and OpenFootball local time zones.
LEAGUES = {
    "PL": ("E0", "en.1", "Europe/London"),
    "ELC": ("E1", "en.2", "Europe/London"),
    "BL1": ("D1", "de.1", "Europe/Berlin"),
    "PD": ("SP1", "es.1", "Europe/Madrid"),
    "SA": ("I1", "it.1", "Europe/Rome"),
    "FL1": ("F1", "fr.1", "Europe/Paris"),
    "PPL": ("P1", "pt.1", "Europe/Lisbon"),
    "DED": ("N1", "nl.1", "Europe/Amsterdam"),
    "EL1": ("E2", "en.3", "Europe/London"),
    "EL2": ("E3", "en.4", "Europe/London"),
    "ENL": ("EC", None, "Europe/London"),
    "SPL": ("SC0", "sco.1", "Europe/London"),
    "SCC": ("SC1", None, "Europe/London"),
    "SL1": ("SC2", None, "Europe/London"),
    "SL2": ("SC3", None, "Europe/London"),
    "BL2": ("D2", "de.2", "Europe/Berlin"),
    "SB": ("I2", "it.2", "Europe/Rome"),
    "SD": ("SP2", "es.2", "Europe/Madrid"),
    "FL2": ("F2", "fr.2", "Europe/Paris"),
    "BPL": ("B1", "be.1", "Europe/Brussels"),
    "TSL": ("T1", "tr.1", "Europe/Istanbul"),
    "GSL": ("G1", "gr.1", "Europe/Athens"),
}


def _identity(*parts: str) -> int:
    # 52 bits remain exactly representable in dashboard JavaScript numbers.
    return int(hashlib.sha256("\0".join(parts).encode()).hexdigest()[:13], 16) or 1


class PublicMatchClient(FootballDataClient):
    def __init__(self, *, limiter=None, **kwargs):
        # Static files have no API subscription quota. Still bound each provider to 60/min.
        super().__init__(
            api_key="", limiter=limiter or SlidingWindowRateLimiter(limit=60), **kwargs
        )

    def _download(self, url: str) -> Any:
        for attempt in range(self.max_retries + 1):
            self.limiter.wait_if_needed()
            try:
                response = self.session.get(url, timeout=self.timeout)
            except requests.RequestException as exc:
                self._record_request()
                if attempt < self.max_retries:
                    self._wait_to_retry(None, attempt, url)
                    continue
                raise DataSourceError(
                    f"{self.source_name}: {type(exc).__name__}",
                    endpoint=url,
                    retryable=True,
                ) from exc
            self._record_request()
            if 200 <= response.status_code < 300:
                return response
            retryable = response.status_code in self.RETRYABLE_STATUS_CODES
            close_response(response)
            if retryable and attempt < self.max_retries:
                self._wait_to_retry(response, attempt, url)
                continue
            raise DataSourceError(
                f"{self.source_name}: HTTP {response.status_code}",
                endpoint=url,
                status_code=response.status_code,
                retryable=retryable,
                retry_after_seconds=self._retry_policy.retry_after(response) if retryable else None,
            )
        raise AssertionError("unreachable")

    def _league(self, code: str, season: str) -> tuple[str, str, str]:
        if code not in LEAGUES:
            raise DataSourceError(
                f'{self.source_name} 不支持 {code}，支持 {",".join(LEAGUES)}',
                endpoint=code,
            )
        if len(season) != 4 or not season.isdigit() or not 1900 <= int(season) <= 2100:
            raise ValueError("season 必须是 1900 到 2100 的赛季起始年")
        return LEAGUES[code]

    def _match(
        self,
        code: str,
        day: date,
        time: str | None,
        home: str,
        away: str,
        scores: tuple[int, int] | None,
        timezone: str,
        url: str,
    ) -> dict[str, Any]:
        home, away = home.strip(), away.strip()
        if not home or not away or home == away:
            raise ValueError("invalid teams")
        if scores is not None and any(
            isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in scores
        ):
            raise ValueError("invalid scores")
        if time:
            kickoff = (
                datetime.fromisoformat(f"{day.isoformat()}T{time}")
                .replace(tzinfo=ZoneInfo(timezone))
                .astimezone(UTC)
            )
        else:
            # Date-only history shares one time bucket; no invented within-day ordering.
            kickoff = datetime.combine(day, datetime.min.time(), UTC)
        return {
            "id": _identity(
                self.source_name,
                code,
                day.isoformat(),
                home.casefold(),
                away.casefold(),
            ),
            "utcDate": kickoff.isoformat(),
            "status": (
                "FINISHED" if scores is not None else "TIMED" if time else "SCHEDULED"
            ),
            "homeTeam": {
                "id": _identity(self.source_name, code, home.casefold()),
                "name": home,
            },
            "awayTeam": {
                "id": _identity(self.source_name, code, away.casefold()),
                "name": away,
            },
            "score": {
                "fullTime": {
                    "home": scores[0] if scores else None,
                    "away": scores[1] if scores else None,
                }
            },
            "source": self.source_name,
            "source_url": url,
            "source_date": day.isoformat(),
            "time_precision": "minute" if time else "date",
            "time_zone_assumption": timezone if time else None,
        }

    def fetch_scheduled_matches(
        self, competition_code: str, date_from: str, date_to: str
    ) -> list[dict[str, Any]]:
        start, end = date.fromisoformat(date_from), date.fromisoformat(date_to)
        if end < start:
            raise ValueError("date_to 不能早于 date_from")
        first = start.year if start.month >= 7 else start.year - 1
        last = end.year if end.month >= 7 else end.year - 1
        years = range(first, last + 1)
        matches = [
            match
            for year in years
            for match in self.fetch_matches(competition_code, season=str(year))
        ]
        return [
            match
            for match in matches
            if match["status"] == "TIMED"
            and start <= datetime.fromisoformat(match["utcDate"]).date() <= end
        ]


class FootballCSVClient(PublicMatchClient):
    source_name = "football-data.co.uk"

    def _parse(
        self, code: str, url: str, *, division_filter: bool = False
    ) -> list[dict[str, Any]]:
        text = self._download(url).text.lstrip("\ufeff")
        return self.parse_csv_text(code, text, url, division_filter=division_filter)

    def parse_csv_text(
        self, code: str, text: str, url: str, *, division_filter: bool = False
    ) -> list[dict[str, Any]]:
        reader = csv.DictReader(io.StringIO(text))
        if reader.fieldnames:
            reader.fieldnames = [
                {"HT": "HomeTeam", "AT": "AwayTeam"}.get(name, name)
                for name in reader.fieldnames
            ]
        required = {"Date", "HomeTeam", "AwayTeam"}
        if not required.issubset(reader.fieldnames or []):
            raise DataSourceError("CSV 缺少比赛字段", endpoint=url)
        matches = []
        for index, row in enumerate(reader, 2):
            fixture_fields = (
                "Date",
                "HomeTeam",
                "AwayTeam",
                "FTHG",
                "FTAG",
                "HG",
                "AG",
            )
            if not any(
                row.get(field) is not None and str(row.get(field)).strip()
                for field in fixture_fields
            ):
                continue
            if division_filter and row.get("Div") != LEAGUES[code][0]:
                continue
            try:
                day = None
                for fmt in ("%d/%m/%Y", "%d/%m/%y"):
                    try:
                        day = datetime.strptime(row["Date"], fmt).date()
                        break
                    except ValueError:
                        pass
                if day is None:
                    raise ValueError("invalid date")
                home_score, away_score = row.get("FTHG", row.get("HG", "")), row.get(
                    "FTAG", row.get("AG", "")
                )
                if bool(home_score) != bool(away_score):
                    raise ValueError("incomplete score")
                scores = (int(home_score), int(away_score)) if home_score else None
                # Football-Data CSV times use UK clock time across European leagues.
                match = self._match(
                    code,
                    day,
                    row.get("Time") or None,
                    row["HomeTeam"],
                    row["AwayTeam"],
                    scores,
                    "Europe/London",
                    url,
                )
                fields = {
                    "home_shots": "HS",
                    "away_shots": "AS",
                    "home_shots_on_target": "HST",
                    "away_shots_on_target": "AST",
                    "home_corners": "HC",
                    "away_corners": "AC",
                    "home_fouls": "HF",
                    "away_fouls": "AF",
                    "home_yellow_cards": "HY",
                    "away_yellow_cards": "AY",
                    "home_red_cards": "HR",
                    "away_red_cards": "AR",
                    "attendance": "Attendance",
                    "half_time_home_goals": "HTHG",
                    "half_time_away_goals": "HTAG",
                }

                def optional_number(field, minimum=0.0):
                    try:
                        value = float(row.get(field, ""))
                        return (
                            value if math.isfinite(value) and value >= minimum else None
                        )
                    except (ValueError, TypeError):
                        return None

                match["statistics"] = {
                    name: optional_number(field) for name, field in fields.items()
                }
                if {"HxG", "AxG"}.intersection(reader.fieldnames):
                    # Preserve provider-labelled totals without assuming a shot model,
                    # penalty convention, or a pre-match availability timestamp.
                    metadata = {
                        "source_fields": {"home": "HxG", "away": "AxG"},
                        "definition_status": "unverified",
                        "usage_policy": "post_match_only",
                    }
                    for side, field in metadata["source_fields"].items():
                        raw = row.get(field)
                        value = optional_number(field)
                        match["statistics"][f"{side}_xg"] = value
                        if field not in reader.fieldnames:
                            state = "column_absent"
                        elif raw is None or not raw.strip():
                            state = "missing"
                        else:
                            state = "valid" if value is not None else "invalid"
                        metadata[f"{side}_status"] = state
                    metadata["complete_pair"] = all(
                        metadata[f"{side}_status"] == "valid"
                        for side in ("home", "away")
                    )
                    match["statistics_metadata"] = {"xg": metadata}
                match["referee"] = row.get("Referee") or None
                match["odds"] = {
                    stage: {
                        book: {
                            outcome: optional_number(prefix + suffix + key, 1.0)
                            for outcome, key in [
                                ("home", "H"),
                                ("draw", "D"),
                                ("away", "A"),
                            ]
                        }
                        for book, prefix in [
                            ("bet365", "B365"),
                            ("pinnacle", "PS"),
                            ("average", "Avg"),
                        ]
                    }
                    for stage, suffix in [("pre_close", ""), ("closing", "C")]
                }
                matches.append(match)
            except (TypeError, ValueError, KeyError) as exc:
                raise DataSourceError(
                    f"CSV 第 {index} 行非法: {exc}", endpoint=url
                ) from exc
        return matches

    def fetch_matches(
        self,
        competition_code: str,
        *,
        season: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        code, year = competition_code.upper(), season or default_training_seasons()[-1]
        division, _, _ = self._league(code, year)
        tag = f"{int(year) % 100:02}{(int(year) + 1) % 100:02}"
        matches = self._parse(
            code, f"https://www.football-data.co.uk/mmz4281/{tag}/{division}.csv"
        )
        return [m for m in matches if status is None or m["status"] == status.upper()]

    def fetch_scheduled_matches(
        self, competition_code: str, date_from: str, date_to: str
    ) -> list[dict[str, Any]]:
        code = competition_code.upper()
        self._league(code, default_training_seasons()[-1])
        start, end = date.fromisoformat(date_from), date.fromisoformat(date_to)
        if end < start:
            raise ValueError("date_to 不能早于 date_from")
        matches = self._parse(
            code, "https://www.football-data.co.uk/fixtures.csv", division_filter=True
        )
        return [
            m
            for m in matches
            if m["status"] == "TIMED"
            and start <= datetime.fromisoformat(m["utcDate"]).date() <= end
        ]


class OpenFootballClient(PublicMatchClient):
    source_name = "openfootball"

    def __init__(self, *, revision: str | None = None, **kwargs):
        if revision is not None and (
            not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision)
        ):
            raise ValueError("revision 必须是完整的40位Git提交SHA")
        self.revision = revision.lower() if revision is not None else "master"
        super().__init__(**kwargs)

    def fetch_matches(
        self,
        competition_code: str,
        *,
        season: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        code, year = competition_code.upper(), season or default_training_seasons()[-1]
        _, slug, timezone = self._league(code, year)
        if slug is None:
            raise DataSourceError(
                f"OpenFootball 不提供 {code} 对应的公开 JSON", endpoint=code
            )
        tag = f"{year}-{(int(year) + 1) % 100:02}"
        url = f"https://raw.githubusercontent.com/openfootball/football.json/{self.revision}/{tag}/{slug}.json"
        response = None
        try:
            response = self._download(url)
            payload = response.json()
            rows = self._matches(payload, endpoint=url)
            matches = []
            for row in rows:
                raw_score = row.get("score")
                scores = (
                    raw_score
                    if isinstance(raw_score, list)
                    else (raw_score or {}).get("ft")
                )
                if scores is not None and (
                    not isinstance(scores, list) or len(scores) != 2
                ):
                    raise ValueError("invalid score.ft")
                match = self._match(
                    code,
                    date.fromisoformat(row["date"]),
                    row.get("time"),
                    row["team1"],
                    row["team2"],
                    tuple(scores) if scores is not None else None,
                    timezone,
                    url,
                )
                if "round" in row:
                    match["round"] = row["round"]
                if isinstance(raw_score, dict):
                    details = {key: value for key, value in raw_score.items() if key != "ft"}
                    if details:
                        # Keep additional provider score codes without treating
                        # extra time or penalty shootouts as regulation scores.
                        match["source_score"] = details
                    if "ht" in raw_score:
                        half = raw_score["ht"]
                        valid = isinstance(half, list) and len(half) == 2 and all(
                            isinstance(value, int) and not isinstance(value, bool) and value >= 0
                            for value in half
                        )
                        match["score"]["halfTime"] = {
                            "home": half[0] if valid else None,
                            "away": half[1] if valid else None,
                        }
                        match["score_metadata"] = {
                            "half_time_status": "valid" if valid else "missing" if half is None else "invalid",
                            "half_time_consistent_with_full_time": (
                                all(h <= f for h, f in zip(half, scores))
                                if valid and scores is not None else None
                            ),
                        }
                matches.append(match)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise DataSourceError(
                f"OpenFootball JSON 非法: {exc}", endpoint=url
            ) from exc
        finally:
            close_response(response)
        return [m for m in matches if status is None or m["status"] == status.upper()]


def create_data_client(source: str | None = None) -> Any:
    selected = source or os.getenv("SOCCER_ANALYTICS_DATA_SOURCE", "auto")
    if selected == "auto":
        selected = (
            "football-data" if os.getenv("FOOTBALL_DATA_API_KEY") else "football-csv"
        )
    clients = {
        "football-data": FootballDataClient,
        "football-csv": FootballCSVClient,
        "openfootball": OpenFootballClient,
    }
    if selected not in clients:
        raise ValueError(f'未知数据源 {selected}，可选 auto,{",".join(clients)}')
    return clients[selected]()
