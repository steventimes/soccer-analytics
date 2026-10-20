"""Observed player panels and team windows with explicit exposure and time boundaries."""

from __future__ import annotations
from collections import Counter, defaultdict
from datetime import date, timedelta
import json
import math
import sqlite3
from .event_validation import LEAGUES, event_metrics

PLAYER_CUTS = ("2017-10-31", "2017-12-31", "2018-02-28", "2018-05-31")
COUNT_FIELDS = (
    "recorded_actions",
    "shots",
    "goals",
    "duels",
    "duel_wins",
    "defensive_duels",
    "interceptions",
    "keeper_attempts",
    "keeper_success",
    "passes",
    "accurate_passes",
    "long_passes",
    "accurate_long_passes",
    "pass_length_sum",
    "passes_with_length",
    "progressive_passes",
)


def beta_estimate(wins, attempts, prior_rate, strength=50):
    if (
        attempts < 0
        or wins < 0
        or wins > attempts
        or not 0 < prior_rate < 1
        or strength <= 0
    ):
        raise ValueError("Invalid beta observation/prior")
    alpha = wins + prior_rate * strength
    beta = attempts - wins + (1 - prior_rate) * strength
    return {
        "observed": wins / attempts if attempts else None,
        "posterior_mean": alpha / (alpha + beta),
        "alpha": alpha,
        "beta": beta,
        "prior_only": attempts == 0,
        "attempts": attempts,
    }


def gamma_estimate(count, minutes, prior_rate, prior_minutes=270):
    if count < 0 or minutes < 0 or prior_rate <= 0 or prior_minutes <= 0:
        raise ValueError("Invalid Poisson observation/prior")
    alpha = count + prior_rate * prior_minutes / 90
    rate = (minutes + prior_minutes) / 90
    return {
        "observed": count * 90 / minutes if minutes else None,
        "posterior_mean": alpha / rate,
        "alpha": alpha,
        "rate": rate,
        "prior_only": minutes == 0,
        "minutes": minutes,
    }


def lineup_minutes(match, red_cards):
    intervals = {}
    for team in match["teamsData"].values():
        formation = team.get("formation") or {}
        for player in formation.get("lineup", []):
            intervals[player["playerId"]] = [0.0, 90.0]
        for sub in formation.get("substitutions", []) or []:
            try:
                minute = min(90.0, max(0.0, float(sub["minute"])))
            except (TypeError, ValueError):
                continue
            if sub["playerOut"] in intervals:
                intervals[sub["playerOut"]][1] = minute
            intervals[sub["playerIn"]] = [minute, 90.0]
    for player, minute in red_cards.items():
        if player in intervals:
            intervals[player][1] = min(intervals[player][1], minute)
    return {pid: max(0.0, end - start) for pid, (start, end) in intervals.items()}


def load_players(connection):
    people = {}
    matches = {}
    ranks = {}
    duplicates = set()
    audit = Counter()
    for kind, payload in connection.execute(
        "SELECT entity_type,payload FROM wyscout_entities WHERE entity_type='players' OR entity_type='playerank' OR entity_type LIKE 'matches_%' ORDER BY asset_id,row_number"
    ):
        p = json.loads(payload)
        if kind == "players":
            people[p["wyId"]] = p
        elif kind in LEAGUES and p["status"] == "Played" and p["duration"] == "Regular":
            matches[p["wyId"]] = {**p, "league": kind.removeprefix("matches_")}
        elif kind == "playerank":
            key = (p["matchId"], p["playerId"])
            if key in ranks:
                duplicates.add(key)
            else:
                ranks[key] = p
    for key in duplicates:
        ranks.pop(key)
    audit["duplicate_minutes_records"] = len(duplicates)
    reds = defaultdict(dict)
    extra = defaultdict(lambda: dict.fromkeys(COUNT_FIELDS, 0))
    team_styles = defaultdict(lambda: dict.fromkeys(COUNT_FIELDS, 0))

    def events():
        for (payload,) in connection.execute(
            "SELECT payload FROM wyscout_events WHERE period IN ('1H','2H') ORDER BY asset_id,row_number"
        ):
            e = json.loads(payload)
            if e["matchId"] not in matches:
                continue
            if not e.get("playerId"):
                audit["unknown_event_player"] += 1
                continue
            key = (e["matchId"], e["playerId"])
            s = extra[key]
            s["recorded_actions"] += 1
            tags = {t["id"] for t in e.get("tags", [])}
            kind = e["eventName"]
            if tags & {1701, 1703}:
                clock = min(45.0, max(0.0, float(e["eventSec"]) / 60)) + (
                    45 if e["matchPeriod"] == "2H" else 0
                )
                reds[e["matchId"]][e["playerId"]] = min(
                    clock, reds[e["matchId"]].get(e["playerId"], 90.0)
                )
            if kind == "Shot" or (
                kind == "Free Kick" and e.get("subEventName") == "Free kick shot"
            ):
                s["shots"] += 1
                s["goals"] += int(101 in tags and 102 not in tags)
            if kind == "Duel":
                if (701 in tags) != (703 in tags):
                    s["duels"] += 1
                    s["duel_wins"] += 703 in tags
                else:
                    audit["neutral_or_unknown_duel_outcome"] += 1
                s["defensive_duels"] += e.get("subEventName") == "Ground defending duel"
            s["interceptions"] += 1401 in tags
            if kind == "Save attempt":
                if (1801 in tags) != (1802 in tags):
                    s["keeper_attempts"] += 1
                    s["keeper_success"] += 1801 in tags
                else:
                    audit["unknown_keeper_outcome"] += 1
            yield e

    passes, pass_audit = event_metrics(events())
    audit.update(pass_audit)
    for key, values in passes.items():
        extra[key].update(values)
    # The published PlayeRank file omits goalkeepers. Reconstruct nominal
    # regular-time exposure for them; raw event times and formations stay intact.
    for mid, match in matches.items():
        for pid, minutes in lineup_minutes(match, reds[mid]).items():
            if (
                pid in people
                and people[pid]["role"]["code2"] == "GK"
                and (mid, pid) not in ranks
                and minutes > 0
            ):
                ranks[(mid, pid)] = {
                    "minutesPlayed": minutes,
                    "minutes_source": "nominal_lineup_intervals",
                }
                audit["goalkeeper_nominal_exposure_rows"] += 1
    observations = []
    for (mid, pid), rank in ranks.items():
        if mid not in matches or pid not in people:
            continue
        minutes = rank.get("minutesPlayed")
        if not isinstance(minutes, (float, int)) or not 0 < minutes <= 110:
            audit["invalid_minutes"] += 1
            continue
        m = matches[mid]
        team_ids = []
        for tid, team in m["teamsData"].items():
            f = team.get("formation") or {}
            if pid in {p["playerId"] for p in f.get("lineup", []) + f.get("bench", [])}:
                team_ids.append(str(tid))
        if len(team_ids) != 1:
            audit["ambiguous_match_team"] += 1
            continue
        row = {
            "entity_id": str(pid),
            "match_id": str(mid),
            "date": m["dateutc"][:10],
            "league": m["league"],
            "team": team_ids[0],
            "position": people[pid]["role"]["name"],
            "minutes": minutes,
            "minutes_source": rank.get("minutes_source", "supplied_playerank_minutes"),
            **extra.get((mid, pid), dict.fromkeys(COUNT_FIELDS, 0)),
        }
        observations.append(row)
        style = team_styles[(mid, team_ids[0])]
        for field in COUNT_FIELDS:
            style[field] += row[field]
    return people, matches, observations, team_styles, dict(audit)


def player_panels(observations, people):
    groups = {}
    for row in observations:
        block = next(
            (i for i, cut in enumerate(PLAYER_CUTS) if row["date"] <= cut), None
        )
        if block is None:
            continue
        key = (row["entity_id"], row["league"], block)
        if key not in groups:
            groups[key] = {
                "entity_id": row["entity_id"],
                "league": row["league"],
                "position": row["position"],
                "block": block,
                "as_of": PLAYER_CUTS[block],
                "minutes": 0.0,
                "matches": 0,
                "teams": defaultdict(float),
                **dict.fromkeys(COUNT_FIELDS, 0),
            }
        g = groups[key]
        g["minutes"] += row["minutes"]
        g["matches"] += 1
        g["teams"][row["team"]] += row["minutes"]
        for field in COUNT_FIELDS:
            g[field] += row[field]
    result = []
    for row in groups.values():
        p = people[int(row["entity_id"])]
        row["name"] = " ".join(
            filter(None, [p.get("firstName"), p.get("middleName"), p.get("lastName")])
        )
        try:
            row["age"] = (
                date.fromisoformat(row["as_of"]) - date.fromisoformat(p["birthDate"])
            ).days / 365.2425
        except (TypeError, ValueError):
            row["age"] = None
        teams = row.pop("teams")
        row["team"] = max(teams, key=teams.get)
        result.append(row)
    return sorted(result, key=lambda r: (r["block"], r["league"], r["entity_id"]))


def rates(row):
    minutes = row["minutes"]
    result = {
        field + "_per90": row[field] * 90 / minutes if minutes else None
        for field in COUNT_FIELDS
        if field not in {"pass_length_sum"}
    }
    for name, wins, attempts in [
        ("pass_completion", "accurate_passes", "passes"),
        ("duel_win_rate", "duel_wins", "duels"),
        ("keeper_success_rate", "keeper_success", "keeper_attempts"),
    ]:
        result[name] = row[wins] / row[attempts] if row[attempts] else None
    result["mean_pass_length"] = (
        row["pass_length_sum"] / row["passes_with_length"]
        if row["passes_with_length"]
        else None
    )
    return result


def fit_priors(panels):
    groups = defaultdict(lambda: dict.fromkeys(COUNT_FIELDS + ("minutes",), 0))
    for row in panels:
        for key in [
            (row["league"], row["position"]),
            ("*", row["position"]),
            ("*", "*"),
        ]:
            for field in COUNT_FIELDS + ("minutes",):
                groups[key][field] += row[field]
    return {key: rates(value) for key, value in groups.items()}


def player_features(row, priors):
    prior = priors.get(
        (row["league"], row["position"]),
        priors.get(("*", row["position"]), priors[("*", "*")]),
    )
    features = {
        "age": row["age"],
        "minutes": row["minutes"],
        "matches": row["matches"],
        "position": row["position"],
        "league": row["league"],
        "team": row["team"],
    }
    for field in (
        "recorded_actions",
        "shots",
        "goals",
        "progressive_passes",
        "defensive_duels",
        "interceptions",
        "keeper_attempts",
        "passes",
        "duels",
    ):
        value = max(prior.get(field + "_per90") or 0.001, 0.001)
        features[field + "_per90"] = gamma_estimate(row[field], row["minutes"], value)[
            "posterior_mean"
        ]
    for name, wins, attempts in [
        ("pass_completion", "accurate_passes", "passes"),
        ("duel_win_rate", "duel_wins", "duels"),
        ("keeper_success_rate", "keeper_success", "keeper_attempts"),
    ]:
        p = min(0.999, max(0.001, prior.get(name) or 0.5))
        features[name] = beta_estimate(row[wins], row[attempts], p)["posterior_mean"]
        features[name + "_known"] = float(row[attempts] > 0)
    return features


def player_examples(panels, priors):
    index = {(r["entity_id"], r["league"], r["block"]): r for r in panels}
    result = []
    for row in panels:
        future = index.get((row["entity_id"], row["league"], row["block"] + 1))
        if not future or row["minutes"] < 270 or future["minutes"] < 270:
            continue
        target = rates(future)
        values = {
            "goals_per90": target["goals_per90"],
            "progressive_passes_per90": target["progressive_passes_per90"],
            "pass_completion": (
                target["pass_completion"] if future["passes"] >= 20 else None
            ),
            "duel_win_rate": target["duel_win_rate"] if future["duels"] >= 10 else None,
            "keeper_success_rate": (
                target["keeper_success_rate"]
                if row["position"] == "Goalkeeper" and future["keeper_attempts"] >= 10
                else None
            ),
        }
        result.append(
            {
                "entity_id": row["entity_id"],
                "league": row["league"],
                "name": row["name"],
                "position": row["position"],
                "phase": row["block"],
                "feature_asof": row["as_of"],
                "target_start": (
                    date.fromisoformat(PLAYER_CUTS[row["block"]]) + timedelta(days=1)
                ).isoformat(),
                "target_end": future["as_of"],
                "features": player_features(row, priors),
                "targets": values,
                "carry_forward": {k: rates(row)[k] for k in values},
            }
        )
    return result


def load_team_matches(connection, as_of):
    seen = {}
    audit = Counter()
    query = "SELECT d.payload FROM documents d JOIN assets a ON a.id=d.asset_id JOIN datasets s ON s.id=a.dataset_id WHERE s.name='football-csv' AND a.path LIKE 'history/football-csv/%' AND json_type(d.payload,'$.matches')='array' ORDER BY a.id"
    for (payload,) in connection.execute(query):
        snapshot = json.loads(payload)
        code = snapshot["competition"]
        season = int(snapshot["season"])
        for match in snapshot["matches"]:
            when = match["utcDate"][:10]
            if when >= as_of:
                audit["same_day_or_future_match"] += 1
                continue
            score = match["score"]["fullTime"]
            if (
                match["status"] != "FINISHED"
                or score["home"] is None
                or score["away"] is None
            ):
                continue
            row = {
                "id": match["id"],
                "competition": code,
                "season": season,
                "date": when,
                "home_id": str(match["homeTeam"]["id"]),
                "away_id": str(match["awayTeam"]["id"]),
                "home": match["homeTeam"]["name"],
                "away": match["awayTeam"]["name"],
                "home_goals": score["home"],
                "away_goals": score["away"],
            }
            key = (code, row["id"])
            if key in seen and seen[key] != row:
                raise ValueError("Conflicting historical match snapshot")
            seen[key] = row
    return sorted(
        seen.values(), key=lambda r: (r["date"], r["competition"], r["id"])
    ), dict(audit)


def team_rows(matches):
    groups = defaultdict(list)
    for m in matches:
        for home in [True, False]:
            tid = m["home_id" if home else "away_id"]
            gf = m["home_goals" if home else "away_goals"]
            ga = m["away_goals" if home else "home_goals"]
            groups[(m["competition"], m["season"], tid)].append(
                {
                    "entity_id": tid,
                    "name": m["home" if home else "away"],
                    "league": m["competition"],
                    "season": m["season"],
                    "date": m["date"],
                    "match_id": m["id"],
                    "gf": gf,
                    "ga": ga,
                    "home": int(home),
                    "opponent": m["away_id" if home else "home_id"],
                    "points": 3 if gf > ga else 1 if gf == ga else 0,
                }
            )
    for rows in groups.values():
        rows.sort(key=lambda r: (r["date"], r["match_id"]))
    return groups


def rolling_team_features(rows):
    n = len(rows)
    return {
        "league": rows[-1]["league"],
        "season": rows[-1]["season"],
        "goals_for": sum(r["gf"] for r in rows) / n,
        "goals_against": sum(r["ga"] for r in rows) / n,
        "points": sum(r["points"] for r in rows) / n,
        "home_fraction": sum(r["home"] for r in rows) / n,
        "win_fraction": sum(r["points"] == 3 for r in rows) / n,
        "clean_sheet_fraction": sum(r["ga"] == 0 for r in rows) / n,
        "scoring_fraction": sum(r["gf"] > 0 for r in rows) / n,
        "goal_balance": sum(r["gf"] - r["ga"] for r in rows) / n,
        "rest_days": (
            (
                date.fromisoformat(rows[-1]["date"])
                - date.fromisoformat(rows[-2]["date"])
            ).days
            if n > 1
            else None
        ),
    }


def team_examples(matches, history=8, horizon=5):
    result = []
    for (league, season, tid), rows in team_rows(matches).items():
        for i in range(history, len(rows) - horizon + 1, horizon):
            past = rows[i - history : i]
            future = rows[i : i + horizon]
            if past[-1]["date"] >= future[0]["date"]:
                continue
            targets = {
                k: sum(r[field] for r in future) / horizon
                for k, field in [
                    ("goals_for", "gf"),
                    ("goals_against", "ga"),
                    ("points", "points"),
                ]
            }
            result.append(
                {
                    "entity_id": tid,
                    "name": past[-1]["name"],
                    "league": league,
                    "season": season,
                    "phase": season,
                    "feature_asof": past[-1]["date"],
                    "target_start": future[0]["date"],
                    "target_end": future[-1]["date"],
                    "features": rolling_team_features(past),
                    "targets": targets,
                    "carry_forward": {
                        k: rolling_team_features(past)[k] for k in targets
                    },
                }
            )
    return sorted(
        result, key=lambda r: (r["feature_asof"], r["league"], r["entity_id"])
    )
