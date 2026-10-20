"""Descriptive team analysis; retrospective totals are never model features."""

from __future__ import annotations

from collections import defaultdict
from statistics import median
from time import perf_counter
from typing import Any, Iterable

from app.ml.feature_engineering import normalize_match


def summarize_matches(
    matches: Iterable[dict[str, Any]], *, backend: str = "python"
) -> dict[str, Any]:
    if backend not in {"python", "fpstreams", "compare"}:
        raise ValueError("backend 必须是 python、fpstreams 或 compare")
    unique = {}
    names = {}
    for payload in matches:
        record = normalize_match(payload)
        if record.match_id in unique and unique[record.match_id] != record:
            raise ValueError(f"冲突比赛 ID {record.match_id}")
        if record.status == "FINISHED":
            unique[record.match_id] = record
            for key in ("homeTeam", "awayTeam"):
                team = payload.get(key, {})
                if team.get("id") and team.get("name"):
                    names[team["id"]] = team["name"]
    rows = []
    home_wins = draws = away_wins = 0
    for match in sorted(unique.values(), key=lambda m: (m.kickoff, m.match_id)):
        h, a = match.home_score, match.away_score
        home_wins += h > a
        draws += h == a
        away_wins += h < a
        for team, venue, gf, ga in [
            (match.home_team_id, "H", h, a),
            (match.away_team_id, "A", a, h),
        ]:
            rows.append(
                {
                    "team_id": team,
                    "venue": venue,
                    "points": 3 if gf > ga else 1 if gf == ga else 0,
                    "goals_for": gf,
                    "goals_against": ga,
                    "clean_sheets": int(ga == 0),
                }
            )
    totals = ("points", "goals_for", "goals_against", "clean_sheets")
    timings = {}
    results = {}
    if backend in {"python", "compare"}:
        start = perf_counter()
        groups = defaultdict(lambda: {"matches": 0, **dict.fromkeys(totals, 0)})
        for row in rows:
            group = groups[row["team_id"]]
            group["matches"] += 1
            for key in totals:
                group[key] += row[key]
        results["python"] = [
            {"team_id": key, **value} for key, value in sorted(groups.items())
        ]
        timings["python"] = perf_counter() - start
    if backend in {"fpstreams", "compare"}:
        from fpstreams import agg, flow

        start = perf_counter()
        results["fpstreams"] = (
            flow(rows)
            .group_by("team_id")
            .aggregate(matches=agg.count(), **{key: agg.sum(key) for key in totals})
            .sort_by("team_id")
            .to_list()
        )
        timings["fpstreams"] = perf_counter() - start
    parity = (
        results.get("python") == results.get("fpstreams")
        if backend == "compare"
        else None
    )
    if parity is False:
        raise ValueError("fpstreams 与 Python 结果不一致")
    teams = results["python" if backend in {"python", "compare"} else "fpstreams"]
    for team in teams:
        team["name"] = names.get(team["team_id"], str(team["team_id"]))
        team["points_per_match"] = team["points"] / team["matches"]
        team["goal_difference"] = team["goals_for"] - team["goals_against"]
    return {
        "matches": len(unique),
        "outcomes": {"home_win": home_wins, "draw": draws, "away_win": away_wins},
        "median_total_goals": (
            median([m.home_score + m.away_score for m in unique.values()])
            if unique
            else None
        ),
        "teams": teams,
        "backend": backend,
        "parity": parity,
        "aggregation_seconds": timings,
        "timing_note": "Single-run aggregation only; excludes download and normalization. Not a general benchmark.",
    }
