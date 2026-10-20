"""Football-first observation cards, youth follow-up and opponent preparation.

Observed habits and current performance are separate from future potential. This
module fits no predictive model and preserves traceable match/event evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date
import hashlib
import inspect
import json
import math
from pathlib import Path
import sqlite3

from .assessment_data import PLAYER_CUTS, fit_priors
from .assessment_profiles import season_profiles, profile_scores
from .artifacts import cached_json, file_digest, source_signature, archive_outputs

SOURCES = [
    {
        "title": "FIFA talent identification, Module3",
        "url": "https://www.fifatrainingcentre.com/en/environment/guide/high-performance/find/talent-identification-guide-module-3.php",
        "application": "Separate current performance from future potential; retain structured observations, relative age and missing maturation information.",
    },
    {
        "title": "Tom Bergkamp, How scouts assess players",
        "url": "https://www.fifatrainingcentre.com/en/environment/research-brief/high-performance/find/bergkamp---how-scouts-assess-players.php",
        "application": "Use explicit evidence and repeated observations; measured physical output alone is not a talent label.",
    },
]


def player_age(birthday, as_of):
    try:
        born, cut = date.fromisoformat(birthday), date.fromisoformat(as_of)
        return cut.year - born.year - ((cut.month, cut.day) < (born.month, born.day))
    except (TypeError, ValueError):
        return None


def observation_tier(minutes, matches, positive_signal):
    if minutes < 450 or matches < 5:
        return "limited_observation"
    return "priority_for_review" if positive_signal else "routine_followup"


def cohort_summary(profiles, backend="python"):
    def records():
        for profile in profiles:
            habit = profile.get("habits") or {}
            count = habit.get("located_actions", 0)
            origin = habit.get("mean_ball_action_origin")
            yield {
                "cohort": profile["league"]
                + "/"
                + profile["position"]
                + "/"
                + profile["age_band"],
                "minutes": profile["minutes"],
                "origin_x_numerator": origin[0] * count if origin else 0.0,
                "located_actions": count if origin else 0.0,
                "origin_unknown": int(origin is None),
            }

    result = {}
    for row in records():
        group = result.setdefault(
            row["cohort"],
            {
                "players": 0,
                "minutes": 0.0,
                "origin_x_numerator": 0.0,
                "located_actions": 0.0,
                "origin_unknown": 0,
            },
        )
        group["players"] += 1
        for key in [
            "minutes",
            "origin_x_numerator",
            "located_actions",
            "origin_unknown",
        ]:
            group[key] += row[key]
    parity = None
    if backend in {"fpstreams", "compare"}:
        from fpstreams import flow, agg

        streamed = (
            flow(records())
            .group_by("cohort")
            .aggregate(
                players=agg.count(),
                minutes=agg.sum("minutes"),
                origin_x_numerator=agg.sum("origin_x_numerator"),
                located_actions=agg.sum("located_actions"),
                origin_unknown=agg.sum("origin_unknown"),
            )
            .to_list()
        )
        fp = {
            row["cohort"]: {key: value for key, value in row.items() if key != "cohort"}
            for row in streamed
        }
        parity = set(fp) == set(result) and all(
            math.isclose(fp[key][field], value, rel_tol=1e-10, abs_tol=1e-8)
            for key, row in result.items()
            for field, value in row.items()
        )
        if not parity:
            raise ValueError("Cohort aggregation differs between Python and fpstreams")
        if backend == "fpstreams":
            result = fp
    for row in result.values():
        row["action_weighted_origin_x"] = (
            row["origin_x_numerator"] / row["located_actions"]
            if row["located_actions"]
            else None
        )
    return {
        "backend": backend,
        "fpstreams_parity": parity,
        "cohorts": [{"cohort": key, **row} for key, row in sorted(result.items())],
        "note": "Generator-fed cohort aggregation; weights are located ball actions, not minutes or measured standing time. No performance claim.",
    }


def location(value):
    try:
        x, y = float(value["x"]), float(value["y"])
        return (
            (x, y) if all(math.isfinite(v) and 0 <= v <= 100 for v in (x, y)) else None
        )
    except (TypeError, KeyError, ValueError):
        return None


def summarize_habits(events, matches, as_of):
    groups = {}
    for event in events:
        match = matches.get(event["matchId"], matches.get(str(event["matchId"])))
        if (
            not match
            or match["dateutc"][:10] > as_of
            or event.get("matchPeriod") not in {"1H", "2H"}
            or not event.get("playerId")
        ):
            continue
        key = (str(event["playerId"]), match["league"])
        if key not in groups:
            groups[key] = {
                "actions": 0,
                "passes": 0,
                "crosses": 0,
                "shots": 0,
                "key_pass_tags": 0,
                "assist_tags": 0,
                "counterattack_tagged_actions": 0,
                "pass_direction_counts": Counter(
                    {"forward": 0, "backward": 0, "lateral": 0}
                ),
                "shot_body_part_counts": Counter(
                    {"left": 0, "right": 0, "head_or_body": 0, "unknown": 0}
                ),
                "event_zone_grid": [[0] * 4 for _ in range(3)],
                "location_x_sum": 0.0,
                "location_y_sum": 0.0,
                "located_actions": 0,
                "direction_by_match": defaultdict(Counter),
                "evidence_examples": defaultdict(list),
            }
        row = groups[key]
        row["actions"] += 1
        kind = event["eventName"]
        tags = {t["id"] for t in event.get("tags", [])}
        row["key_pass_tags"] += 302 in tags
        row["assist_tags"] += 301 in tags
        row["counterattack_tagged_actions"] += 1901 in tags
        points = event.get("positions") or []
        start = location(points[0]) if points else None
        # Action locations are deliberately restricted to ball actions. They
        # are not average standing positions or off-ball occupation.
        if start and kind in {"Pass", "Shot", "Free Kick", "Others on the ball"}:
            row["event_zone_grid"][min(2, int(start[1] * 3 / 100))][
                min(3, int(start[0] * 4 / 100))
            ] += 1
            row["location_x_sum"] += start[0]
            row["location_y_sum"] += start[1]
            row["located_actions"] += 1
        label = None
        if kind == "Pass":
            row["passes"] += 1
            row["crosses"] += event.get("subEventName") == "Cross"
            end = location(points[-1]) if len(points) == 2 else None
            if (
                start
                and end
                and event.get("subEventName") not in {"Head pass", "Hand pass"}
            ):
                dx = end[0] - start[0]
                label = "forward" if dx >= 5 else "backward" if dx <= -5 else "lateral"
                row["pass_direction_counts"][label] += 1
                row["direction_by_match"][str(event["matchId"])][label] += 1
        if kind == "Shot" or (
            kind == "Free Kick" and event.get("subEventName") == "Free kick shot"
        ):
            row["shots"] += 1
            parts = [
                name
                for code, name in [(401, "left"), (402, "right"), (403, "head_or_body")]
                if code in tags
            ]
            row["shot_body_part_counts"][
                parts[0] if len(parts) == 1 else "unknown"
            ] += 1
            label = "shot"
        if label and len(row["evidence_examples"][label]) < 3:
            row["evidence_examples"][label].append(
                {
                    "provider": "wyscout",
                    "match_id": event["matchId"],
                    "event_id": event["id"],
                    "date": match["dateutc"][:10],
                    "period": event["matchPeriod"],
                    "seconds": event["eventSec"],
                    "subtype": event.get("subEventName"),
                    "positions": points,
                }
            )
    for row in groups.values():
        count = sum(row["pass_direction_counts"].values())
        row["pass_direction_share"] = {
            k: v / count if count else None
            for k, v in row["pass_direction_counts"].items()
        }
        row["cross_share_all_passes"] = (
            row["crosses"] / row["passes"] if row["passes"] else None
        )
        known = (
            row["shot_body_part_counts"]["left"] + row["shot_body_part_counts"]["right"]
        )
        row["observed_left_foot_shot_share"] = (
            row["shot_body_part_counts"]["left"] / known if known >= 5 else None
        )
        row["footed_shots_with_known_foot"] = known
        n = row["located_actions"]
        row["mean_ball_action_origin"] = (
            [row["location_x_sum"] / n, row["location_y_sum"] / n] if n else None
        )
        qualifying = {
            mid: counts
            for mid, counts in row.pop("direction_by_match").items()
            if sum(counts.values()) >= 5
        }
        dominant = (
            max(row["pass_direction_counts"], key=row["pass_direction_counts"].get)
            if count
            else None
        )
        supporting = (
            sum(
                counts[dominant] == max(counts.values())
                for counts in qualifying.values()
            )
            if dominant
            else 0
        )
        row["pass_direction_observed_matches"] = len(qualifying)
        row["dominant_direction_match_fraction"] = (
            supporting / len(qualifying) if qualifying else None
        )
        row["persistent_pass_direction"] = (
            dominant
            if len(qualifying) >= 8 and supporting / len(qualifying) >= 0.65
            else None
        )
        row["evidence_examples"] = dict(row["evidence_examples"])
    return groups


def development_profiles(data, as_of, habits):
    panels = [p for p in data["panels"] if p["as_of"] <= as_of]
    priors = fit_priors([p for p in panels if p["block"] == 0])
    profiles = profile_scores(season_profiles(panels, minimum_minutes=90), priors)
    by_player = defaultdict(list)
    for p in panels:
        by_player[(p["entity_id"], p["league"])].append(p)
    cohorts = defaultdict(list)
    for profile in profiles:
        person = data["people"][profile["entity_id"]]
        age = player_age(person.get("birthDate"), as_of)
        band = (
            "16_to18"
            if age is not None and 16 <= age <= 18
            else "19_to21" if age is not None and 19 <= age <= 21 else "other"
        )
        profile["age_years_at_cutoff"] = age
        profile["age_band"] = band
        profile["reported_preferred_foot"] = person.get("foot") or None
        profile["birth_quarter"] = (
            (int(person["birthDate"][5:7]) - 1) // 3 + 1
            if person.get("birthDate")
            else None
        )
        profile["habits"] = habits.get((profile["entity_id"], profile["league"]))
        trajectory = []
        for panel in by_player[(profile["entity_id"], profile["league"])]:
            trajectory.append(
                {
                    "window_end": panel["as_of"],
                    "minutes": panel["minutes"],
                    "matches": panel["matches"],
                    "rate_exposure_ge270min": panel["minutes"] >= 270,
                    "team_id": panel["team"],
                    "metrics": {
                        "goals_per90": panel["goals"] * 90 / panel["minutes"],
                        "progressive_passes_per90": panel["progressive_passes"]
                        * 90
                        / panel["minutes"],
                        "pass_completion": (
                            panel["accurate_passes"] / panel["passes"]
                            if panel["passes"] >= 20
                            else None
                        ),
                    },
                }
            )
        profile["covered_window_trajectory"] = trajectory
        profile["trajectory_interpretation"] = (
            "Within-season changes in observed output and opportunity; not a maturation curve or a forecast."
        )
        cohorts[(profile["league"], profile["position"], band)].append(profile)
    watch = []
    for profile in profiles:
        if profile["age_band"] == "other":
            continue
        cohort = cohorts[(profile["league"], profile["position"], profile["age_band"])]
        fields = [
            "goals_per90",
            "progressive_passes_per90",
            "pass_completion",
            "duel_win_rate",
        ]
        if profile["position"] == "Goalkeeper":
            fields = ["pass_completion", "keeper_success_rate"]
        peers = [p for p in cohort if p["minutes"] >= 450]
        percentiles = {}
        for field in fields:

            def known(p):
                return (
                    p["passes"] >= 20
                    if field == "pass_completion"
                    else (
                        p["duels"] >= 10
                        if field == "duel_win_rate"
                        else (
                            p["keeper_attempts"] >= 10
                            if field == "keeper_success_rate"
                            else True
                        )
                    )
                )

            values = [p["posterior_features"][field] for p in peers if known(p)]
            value = profile["posterior_features"][field]
            percentiles[field] = (
                100
                * (
                    sum(v < value for v in values)
                    + 0.5 * sum(v == value for v in values)
                )
                / len(values)
                if len(values) >= 5 and profile["minutes"] >= 450 and known(profile)
                else None
            )
        dimensions = profile["evaluation"]["dimensions"]
        signals = [k for k, v in dimensions.items() if v is not None and v >= 75]
        tier = observation_tier(profile["minutes"], profile["matches"], bool(signals))
        watch.append(
            {
                "provider": "wyscout",
                "player_id": profile["entity_id"],
                "name": profile["name"],
                "league": profile["league"],
                "position": profile["position"],
                "team_id": profile["team"],
                "age_years": profile["age_years_at_cutoff"],
                "as_of": as_of,
                "minutes": profile["minutes"],
                "matches": profile["matches"],
                "observation_tier": tier,
                "observed_strength_dimensions": signals,
                "role_relative_dimensions": dimensions,
                "age_role_cohort_players": len(cohort),
                "adequate_exposure_peers": len(peers),
                "age_role_percentiles": percentiles,
                "birth_quarter": profile["birth_quarter"],
                "reported_preferred_foot": profile["reported_preferred_foot"],
                "habits": profile["habits"],
                "trajectory": profile["covered_window_trajectory"],
                "future_potential_estimate": None,
                "biological_maturation": None,
                "availability_prediction": None,
                "next_observations": [
                    "Observe receiving and decisions under pressure.",
                    "Review off-ball support and recovery positioning.",
                    "Compare additional opponents and a later real season.",
                    "Check weaker-foot actions; missing tags are not evidence of one-footedness.",
                ],
                "interpretation": "Manual follow-up criteria on covered historical observations; not a ranking of future stars or a current transfer recommendation.",
            }
        )
    return profiles, watch


def game_context(connection, watch):
    lookup = defaultdict(list)
    for link in connection.execute(
        "SELECT real_player_id,game_asset_id,game_row_number FROM game_real_links WHERE real_provider='wyscout' AND game_provider='fifa'"
    ):
        profile = connection.execute(
            "SELECT effective_date,overall,edition FROM game_profiles WHERE asset_id=? AND row_number=?",
            (link[1], link[2]),
        ).fetchone()
        if not profile or not profile["effective_date"]:
            continue
        table = connection.execute(
            "SELECT kind FROM imports WHERE asset_id=? AND kind LIKE '%players'",
            (link[1],),
        ).fetchone()
        potential = None
        if table:
            columns = {
                r[1]
                for r in connection.execute(
                    'PRAGMA table_info("' + table[0].replace('"', '""') + '")'
                )
            }
            if "potential" in columns:
                row = connection.execute(
                    'SELECT potential FROM "'
                    + table[0].replace('"', '""')
                    + '" WHERE asset_id=? AND row_number=?',
                    (link[1], link[2]),
                ).fetchone()
                try:
                    potential = float(row[0]) if row and row[0] is not None else None
                except ValueError:
                    pass
        lookup[str(link[0])].append(
            {
                "provider": "fifa",
                "edition": profile["edition"],
                "effective_date": profile["effective_date"],
                "overall": profile["overall"],
                "game_potential": potential,
                "evidence_asset_id": link[1],
                "evidence_row_number": link[2],
            }
        )
    for row in watch:
        row["dated_game_context"] = [
            p
            for p in lookup[row["player_id"]]
            if p["effective_date"][:10] < row["as_of"]
        ]
        row["game_interpretation"] = (
            "Game potential is a subjective game attribute, not observed real-world future ability; it does not select this watchlist."
        )


def opponent_preparation(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["competition_id"], row["season_id"], row["team_id"])].append(row)
    metrics = [
        "possession_clock_share_proxy",
        "central_attack_share",
        "low_y_attack_share",
        "high_y_attack_share",
        "npxg_per100_possessions",
        "pressure_per100_opponent_passes",
    ]
    dossiers = []
    for key, sample in groups.items():
        opponents = defaultdict(list)
        for row in sample:
            opponents[row["opponent_id"]].append(row)
        for opponent, observed in opponents.items():
            remainder = [r for r in sample if r["opponent_id"] != opponent]
            if len(observed) < 2 or len(remainder) < 5:
                continue
            differences = {}
            for metric in metrics:
                aligned = []
                eligible = 0
                for home in [False, True]:
                    own = [
                        r[metric]
                        for r in observed
                        if r["home"] == home and r.get(metric) is not None
                    ]
                    reference = [
                        r[metric]
                        for r in remainder
                        if r["home"] == home and r.get(metric) is not None
                    ]
                    if own and len(reference) >= 2:
                        aligned.extend(
                            (value - sum(reference) / len(reference)) for value in own
                        )
                        eligible += len(own)
                differences[metric] = {
                    "fixture_balanced_difference": (
                        sum(aligned) / len(aligned) if aligned else None
                    ),
                    "supported_opponent_fixtures": eligible,
                }
            coverage = [
                r["clock_coverage_fraction"]
                for r in observed
                if r.get("clock_coverage_fraction") is not None
            ]
            dossiers.append(
                {
                    "competition_id": key[0],
                    "season_id": key[1],
                    "season": sample[0]["season"],
                    "team_id": key[2],
                    "team": sample[0]["team"],
                    "opponent_id": opponent,
                    "opponent": observed[0]["opponent"],
                    "opponent_matches": len(observed),
                    "other_opponent_matches": len(remainder),
                    "home_matches": sum(r["home"] for r in observed),
                    "formation_choices": dict(
                        Counter(r["starting_formation"] or "unknown" for r in observed)
                    ),
                    "home_away_aligned_differences_vs_other_opponents": differences,
                    "mean_clock_coverage": (
                        sum(coverage) / len(coverage) if coverage else None
                    ),
                    "evidence_matches": [
                        {
                            "match_id": r["match_id"],
                            "date": r["date"],
                            "home": r["home"],
                        }
                        for r in observed
                    ],
                    "interpretation": "Descriptive observed adjustments within one competition-season, aligned by home/away. Different opponent strength, score and selection remain confounders.",
                }
            )
    return dossiers


def run(
    database="data/football.sqlite3",
    output=".tmp/scouting-intelligence",
    assessment=".tmp/assessments",
    player_as_of="2018-05-31",
    backend="python",
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    player_cache = Path(assessment) / "player-data.json"
    cache_receipt = json.loads(player_cache.with_suffix(".receipt.json").read_text())
    if cache_receipt["content_sha256"] != file_digest(player_cache):
        raise ValueError("Player cache hash mismatch; rerun assess")
    data = json.loads(player_cache.read_text())
    connection = sqlite3.connect(f"file:{Path(database).resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    inputs = {
        "sources": source_signature(connection, ("wyscout",)),
        "player_as_of": player_as_of,
        "extractor_sha256": hashlib.sha256(
            (inspect.getsource(location) + inspect.getsource(summarize_habits)).encode()
        ).hexdigest(),
        "signature_scope": "location and summarize_habits functions",
    }
    if cache_receipt["inputs"]["sources"] != inputs["sources"]:
        raise ValueError("Player cache source revision changed; rerun assess")

    def build_habits():
        events = (
            json.loads(row[0])
            for row in connection.execute(
                "SELECT payload FROM wyscout_events WHERE period IN ('1H','2H') ORDER BY asset_id,row_number"
            )
        )
        result = summarize_habits(events, data["matches"], player_as_of)
        return [
            {"player_id": key[0], "league": key[1], **value}
            for key, value in result.items()
        ]

    habits = cached_json(output / "habits.json", inputs, build_habits)
    profiles, watch = development_profiles(
        data, player_as_of, {(r["player_id"], r["league"]): r for r in habits}
    )
    game_context(connection, watch)
    run_id = "tactics-" + file_digest(".tmp/tactical-relations/report.json")[:16]
    tactical_rows = [
        json.loads(r[0])
        for r in connection.execute(
            "SELECT payload FROM tactical_team_matches WHERE run_id=?", (run_id,)
        )
    ]
    if not tactical_rows:
        raise ValueError("Run the tactics command before scouting intelligence")
    dossiers = opponent_preparation(tactical_rows)
    cohorts = cohort_summary(profiles, backend)
    connection.close()
    report = {
        "player_as_of": player_as_of,
        "player_cards": len(profiles),
        "habits_profiles": len(habits),
        "youth_watchlist": len(watch),
        "youth_tiers": dict(Counter(r["observation_tier"] for r in watch)),
        "young_players_with_dated_game_context": sum(
            bool(r["dated_game_context"]) for r in watch
        ),
        "opponent_preparation_dossiers": len(dossiers),
        "sources": SOURCES,
        "cohort_aggregation": cohorts,
        "input_receipt": inputs,
        "definitions": {
            "metadata_timestamp": "Roster role and reported preferred foot lack dated historical snapshots; early-cutoff cards remain retrospective observations.",
            "ages": "Exact calendar ages16-21 at the observation cutoff, not competition U21 eligibility.",
            "opportunity": "At least90 observed minutes enter the longlist; <450min or <5matches retain limited-observation status.",
            "priority": "Manual review flag: sufficient exposure and >=75th role/league dimension percentile. Current performance, not future talent.",
            "age_peers": "Age bands16-18 and19-21 within league and broad role; percentiles require>=5peers with>=450min.",
            "habits": "Wyscout normalized0-100 ball-action locations, pass direction>=5x units, crosses and tagged shot foot. Sparse/unknown fields remain unknown.",
            "persistent": "Dominant passing direction recurs in>=65% of at least8matches with>=5located foot passes; ties count as co-dominant.",
            "potential": "Real future ceiling and maturation remain unknown; a single season cannot validate long-term growth.",
            "birth_quarter": "Descriptive birth-month groups only; no population denominator or competition cutoff to estimate relative-age bias.",
            "game": "Previously verified identity links and snapshots strictly before cutoff; game potential is context only.",
            "opponents": "Competition-season and home/away aligned, comparing the opponent with the same team facing other opponents. Observational, not an instruction to change tactics.",
        },
    }
    for name, value in [
        ("report", report),
        ("player-cards", profiles),
        ("youth-watchlist", watch),
        ("opponent-preparation", dossiers),
    ]:
        (output / f"{name}.json").write_text(
            json.dumps(value, indent=2, allow_nan=False)
        )
    run_id = file_digest(output / "report.json")[:16]
    from app.data_service.warehouse import Warehouse

    with Warehouse(database) as db:
        db.connection.execute(
            "CREATE TABLE IF NOT EXISTS scouting_insight_runs(run_id TEXT PRIMARY KEY,payload TEXT)"
        )
        db.connection.execute(
            "INSERT INTO scouting_insight_runs VALUES(?,?) ON CONFLICT(run_id) DO UPDATE SET payload=excluded.payload",
            (run_id, json.dumps(report)),
        )
        for table, rows in [
            ("player_habit_cards", profiles),
            ("youth_observation_watchlist", watch),
            ("opponent_preparation_cards", dossiers),
        ]:
            db.connection.execute(
                f"CREATE TABLE IF NOT EXISTS {table}(run_id TEXT REFERENCES scouting_insight_runs(run_id),row_number INTEGER,payload TEXT,PRIMARY KEY(run_id,row_number))"
            )
            db.connection.execute(f"DELETE FROM {table} WHERE run_id=?", (run_id,))
            db.connection.executemany(
                f"INSERT INTO {table} VALUES(?,?,?)",
                [(run_id, i, json.dumps(r)) for i, r in enumerate(rows, 1)],
            )
        archive_outputs(
            db,
            "scouting_intelligence_artifacts",
            output,
            "scouting_intelligence_artifact",
            run_id,
        )
        db.connection.commit()
        db.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in {"definitions", "sources", "input_receipt"}
            },
            indent=2,
        )
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="data/football.sqlite3")
    parser.add_argument("--output", default=".tmp/scouting-intelligence")
    parser.add_argument("--assessment", default=".tmp/assessments")
    parser.add_argument("--player-as-of", choices=PLAYER_CUTS, default=PLAYER_CUTS[-1])
    parser.add_argument(
        "--backend", choices=("python", "fpstreams", "compare"), default="python"
    )
    args = parser.parse_args()
    run(args.database, args.output, args.assessment, args.player_as_of, args.backend)
