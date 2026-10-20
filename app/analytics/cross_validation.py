"""Audit game/real-world identities and test pre-match ratings against observations.

Associations are exploratory. Game ratings and provider-derived tracking metrics
are independent sources of information, not independent measurements of ability.
"""

from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import date
import json
import math
from pathlib import Path
import sqlite3
import unicodedata


def name_tokens(value):
    value = unicodedata.normalize("NFKD", value or "").casefold()
    value = "".join(c for c in value if not unicodedata.combining(c))
    return tuple("".join(c if c.isalnum() else " " for c in value).split())


def same_name(real, candidate):
    """Full real names may omit middle names, but initials alone never match."""
    left, right = name_tokens(real), name_tokens(candidate)
    if len(left) < 2 or any(len(t) == 1 for t in left):
        return False
    if left == right:
        return True
    return (
        bool(right)
        and left[0] == right[0]
        and left[-1] == right[-1]
        and all(t in right for t in left)
    )


def quantile_histogram(histogram, q):
    total = sum(histogram)
    if not total:
        return None
    threshold = math.ceil(total * q)
    running = 0
    for index, count in enumerate(histogram):
        running += count
        if running >= threshold:
            return (index + 0.5) / 10


def position_group(value):
    value = (value or "").lower()
    if "goalkeeper" in value:
        return "goalkeeper"
    if "back" in value or "defend" in value:
        return "defender"
    if "forward" in value or "wing" in value:
        return "forward"
    return "midfielder" if "midfield" in value else "unknown"


def choose_snapshot(rows, cutoff):
    cutoff_date = date.fromisoformat(cutoff[:10])
    eligible = []
    for row in rows:
        try:
            effective = date.fromisoformat(row["effective_date"][:10])
        except (ValueError, TypeError):
            continue
        if effective < cutoff_date:
            eligible.append(row)
    return (
        max(
            eligible,
            key=lambda r: (r["effective_date"], r["asset_id"], r["row_number"]),
        )
        if eligible
        else None
    )


def extract_observations(root):
    root = Path(root)
    observations = []
    people = {}
    exclusions = Counter()
    for metadata_file in sorted(
        (root / "open-data/skillcorner/data/matches").glob("*/*_match.json")
    ):
        metadata = json.loads(metadata_file.read_text())
        mid = str(metadata["id"])
        report_file = root / "scouting/movement" / f"skillcorner-{mid}.json"
        if not report_file.exists():
            exclusions["missing_tracking_report"] += 1
            continue
        report = json.loads(report_file.read_text())
        tracking = {str(p["player_id"]): p for p in report["players"]}
        runs = {
            str(p["player_id"]): p
            for p in report.get("dynamic_events", {}).get("players", [])
        }
        teams = {
            metadata["home_team"]["id"]: metadata["home_team"]["name"],
            metadata["away_team"]["id"]: metadata["away_team"]["name"],
        }
        for p in metadata["players"]:
            pid = str(p["id"])
            name = " ".join(filter(None, [p.get("first_name"), p.get("last_name")]))
            birthday = p.get("birthday")
            group = position_group((p.get("player_role") or {}).get("position_group"))
            people.setdefault(
                pid,
                {
                    "id": pid,
                    "name": name,
                    "birth_date": birthday,
                    "names": set(),
                    "birth_dates": set(),
                },
            )
            people[pid]["names"].add(name)
            if birthday:
                people[pid]["birth_dates"].add(birthday)
            if group in {"goalkeeper", "unknown"}:
                exclusions["goalkeeper_or_unknown_position"] += 1
                continue
            minutes = ((p.get("playing_time") or {}).get("total") or {}).get(
                "minutes_played"
            )
            t = tracking.get(pid)
            if (
                not minutes
                or minutes < 20
                or not t
                or t["observed_segment_seconds"] < 600
            ):
                exclusions["insufficient_minutes_or_observation"] += 1
                continue
            coverage = t["observed_segment_seconds"] / (minutes * 60)
            if not 0.5 <= coverage <= 1.05:
                exclusions["coverage_below_half_or_inconsistent"] += 1
                continue
            r = runs.get(pid)
            observations.append(
                {
                    "match_id": mid,
                    "player_id": pid,
                    "date": metadata["date_time"][:10],
                    "name": name,
                    "birth_date": birthday,
                    "position": group,
                    "team": teams.get(p["team_id"], str(p["team_id"])),
                    "minutes": minutes,
                    "observed_seconds": t["observed_segment_seconds"],
                    "coverage": coverage,
                    "p99_speed_mps": quantile_histogram(t["speed_histogram"], 0.99),
                    "distance_per_observed_minute": t["distance_m"]
                    / t["observed_segment_seconds"]
                    * 60,
                    "off_ball_runs_per90": (
                        r["off_ball_runs"] / minutes * 90 if r else None
                    ),
                    "off_ball_runs": r["off_ball_runs"] if r else None,
                    "unknown_run_extrapolation_status": (
                        r.get("unknown_extrapolation_status", 0) if r else None
                    ),
                    "tracking_report": str(report_file),
                    "metadata_file": str(metadata_file),
                }
            )
    return people, observations, dict(exclusions)


def group_players(observations, backend="python"):
    groups = defaultdict(list)
    for r in observations:
        groups[r["player_id"]].append(r)
    result = {}
    for pid, rows in groups.items():
        first = min(r["date"] for r in rows)
        total_minutes = sum(r["minutes"] for r in rows)
        observed = sum(r["observed_seconds"] for r in rows)
        counts = defaultdict(float)
        teams = defaultdict(float)
        for r in rows:
            counts[r["position"]] += r["minutes"]
            teams[r["team"]] += r["minutes"]
        runs = [r for r in rows if r["off_ball_runs"] is not None]
        result[pid] = {
            "player_id": pid,
            "first_match": first,
            "last_match": max(r["date"] for r in rows),
            "position": max(counts, key=counts.get),
            "team": max(teams, key=teams.get),
            "matches": len(rows),
            "minutes": total_minutes,
            "coverage": observed / (total_minutes * 60),
            "p99_speed_mps": sum(
                r["p99_speed_mps"] * r["observed_seconds"] for r in rows
            )
            / observed,
            "distance_per_observed_minute": sum(
                r["distance_per_observed_minute"] * r["observed_seconds"] for r in rows
            )
            / observed,
            "off_ball_runs_per90": (
                sum(r["off_ball_runs"] for r in runs)
                / sum(r["minutes"] for r in runs)
                * 90
                if runs
                else None
            ),
        }
    if backend not in {"python", "fpstreams", "compare"}:
        raise ValueError("Unknown aggregation backend")
    if backend in {"fpstreams", "compare"}:
        from fpstreams import flow, agg

        rows = [
            {
                "player_id": r["player_id"],
                "minutes": r["minutes"],
                "observed": r["observed_seconds"],
                "weighted_speed": r["p99_speed_mps"] * r["observed_seconds"],
                "weighted_distance": r["distance_per_observed_minute"]
                * r["observed_seconds"],
                "known_runs": (
                    r["off_ball_runs"] if r["off_ball_runs"] is not None else 0
                ),
                "known_run_minutes": (
                    r["minutes"] if r["off_ball_runs"] is not None else 0
                ),
            }
            for r in observations
        ]
        totals = (
            flow(rows)
            .group_by("player_id")
            .aggregate(
                matches=agg.count(),
                **{
                    k: agg.sum(k)
                    for k in [
                        "minutes",
                        "observed",
                        "weighted_speed",
                        "weighted_distance",
                        "known_runs",
                        "known_run_minutes",
                    ]
                },
            )
            .to_list()
        )
        if {r["player_id"] for r in totals} != set(result):
            raise ValueError("fpstreams group-key mismatch")
        for row in totals:
            metrics = {
                "minutes": row["minutes"],
                "coverage": row["observed"] / (row["minutes"] * 60),
                "p99_speed_mps": row["weighted_speed"] / row["observed"],
                "distance_per_observed_minute": row["weighted_distance"]
                / row["observed"],
                "off_ball_runs_per90": (
                    row["known_runs"] / row["known_run_minutes"] * 90
                    if row["known_run_minutes"]
                    else None
                ),
            }
            expected = result[row["player_id"]]
            if row["matches"] != expected["matches"]:
                raise ValueError("fpstreams match-count mismatch")
            for key, value in metrics.items():
                if value is None or expected[key] is None:
                    if value != expected[key]:
                        raise ValueError("fpstreams missing-value mismatch")
                elif not math.isclose(
                    value, expected[key], rel_tol=1e-10, abs_tol=1e-8
                ):
                    raise ValueError("fpstreams weighted aggregate mismatch")
            if backend == "fpstreams":
                expected.update(metrics)
    return result


def associations(pairs, bootstrap=500, additional_controls=()):
    if len(pairs) < 20:
        return {"players": len(pairs), "status": "insufficient_matched_players"}
    import numpy as np
    from scipy.stats import rankdata
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupShuffleSplit

    def correlation(x, y):
        if np.std(x) < 1e-10 or np.std(y) < 1e-10:
            return None
        return float(np.corrcoef(x, y)[0, 1])

    x = np.array([p["game_value"] for p in pairs], float)
    y = np.array([p["real_value"] for p in pairs], float)
    ages = np.array([p["age"] for p in pairs])
    positions = [p["position"] for p in pairs]
    teams = [p["team"] for p in pairs]
    columns = [ages] + [
        np.array([p[key] for p in pairs], float) for key in additional_controls
    ]
    for category in sorted(set(positions))[1:]:
        columns.append(np.array([float(v == category) for v in positions]))
    controls = np.column_stack(columns)
    team_codes = np.unique(teams, return_inverse=True)[1]

    def demean(values, codes):
        counts = np.bincount(codes)
        if values.ndim == 1:
            sums = np.bincount(codes, weights=values)
            means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
            return values - means[codes]
        return np.column_stack(
            [demean(values[:, j], codes) for j in range(values.shape[1])]
        )

    def adjusted(x, y, c, codes):
        rx = demean(rankdata(x), codes)
        ry = demean(rankdata(y), codes)
        c = demean(c, codes)
        rx = rx - c @ np.linalg.lstsq(c, rx, rcond=None)[0]
        ry = ry - c @ np.linalg.lstsq(c, ry, rcond=None)[0]
        return correlation(rx, ry)

    degrees = np.linalg.matrix_rank(demean(controls, team_codes)) + len(set(teams))
    if len(pairs) <= degrees + 5:
        return {"players": len(pairs), "status": "insufficient_degrees_of_freedom"}
    rho = adjusted(x, y, controls, team_codes)
    raw = correlation(rankdata(x), rankdata(y))
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(bootstrap):
        ids = rng.integers(0, len(pairs), len(pairs))
        value = adjusted(x[ids], y[ids], controls[ids], team_codes[ids])
        if value is not None:
            samples.append(value)
    result = {
        "players": len(pairs),
        "status": "exploratory",
        "spearman": raw,
        "adjusted_rank_correlation": rho,
        "adjusted_controls": ["position", "age", "team", *additional_controls],
        "player_bootstrap_samples": len(samples),
        "player_bootstrap_ci95": (
            np.quantile(samples, [0.025, 0.975]).tolist() if samples else None
        ),
    }
    if len(pairs) >= 40:
        # Outcome-derived pass geometry/exposure covariates are descriptive controls
        # only; the predictive comparison uses position, age, team and game score.
        features = np.array(
            [[p["position"], p["team"], p["age"], p["game_value"]] for p in pairs],
            dtype=object,
        )
        game_column = 3
        split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train, test = next(
            split.split(features, y, groups=[p["player_id"] for p in pairs])
        )
        predictions = {}
        for mode in ["baseline", "with_game_attribute"]:
            transformer = ColumnTransformer(
                [
                    (
                        "category",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        [0, 1],
                    ),
                    (
                        "numeric",
                        StandardScaler(),
                        (
                            list(range(2, game_column))
                            if mode == "baseline"
                            else list(range(2, game_column + 1))
                        ),
                    ),
                ]
            )
            model = make_pipeline(transformer, Ridge(alpha=1.0))
            model.fit(features[train], y[train])
            predictions[mode] = model.predict(features[test])
        losses = {k: np.abs(v - y[test]) for k, v in predictions.items()}
        difference = losses["with_game_attribute"] - losses["baseline"]
        ci = np.quantile(
            [
                np.mean(difference[rng.integers(0, len(test), len(test))])
                for _ in range(bootstrap)
            ],
            [0.025, 0.975],
        ).tolist()
        result["player_disjoint_holdout"] = {
            "train_players": len(train),
            "test_players": len(test),
            "baseline_mae": float(np.mean(losses["baseline"])),
            "with_game_attribute_mae": float(np.mean(losses["with_game_attribute"])),
            "prediction_controls": ["position", "age", "team"],
            "mae_difference": float(np.mean(difference)),
            "conditional_test_player_bootstrap_ci95": ci,
            "interpretation": "Negative difference favors game feature. Retrospective within-season player holdout; not prospective temporal validation.",
        }
        if pairs and all("league" in p for p in pairs):
            folds = []
            for league in sorted({p["league"] for p in pairs}):
                held = np.array([p["league"] == league for p in pairs])
                training = ~held
                if np.sum(held) < 20 or np.sum(training) < 40:
                    continue
                losses_by_mode = {}
                for mode in ["baseline", "with_game_attribute"]:
                    transformer = ColumnTransformer(
                        [
                            (
                                "category",
                                OneHotEncoder(
                                    handle_unknown="ignore", sparse_output=False
                                ),
                                [0, 1],
                            ),
                            (
                                "numeric",
                                StandardScaler(),
                                [2] if mode == "baseline" else [2, 3],
                            ),
                        ]
                    )
                    model = make_pipeline(transformer, Ridge(alpha=1.0))
                    model.fit(features[training], y[training])
                    losses_by_mode[mode] = float(
                        np.mean(np.abs(model.predict(features[held]) - y[held]))
                    )
                folds.append(
                    {
                        "held_out_league": league,
                        "train_players": int(np.sum(training)),
                        "test_players": int(np.sum(held)),
                        **losses_by_mode,
                        "mae_difference": losses_by_mode["with_game_attribute"]
                        - losses_by_mode["baseline"],
                    }
                )
            result["leave_one_league_out"] = folds
    return result


def validate(
    database="data/football.sqlite3",
    root=".tmp",
    output=".tmp/cross-source-validation.json",
    backend="python",
):
    root = Path(root)
    people, observations, exclusions = extract_observations(root)
    aggregates = group_players(observations, backend=backend)
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript("""
    CREATE INDEX IF NOT EXISTS profiles_birth_lookup ON game_profiles(birth_date);
    CREATE TABLE IF NOT EXISTS real_players(provider TEXT,player_id TEXT,name TEXT,birth_date TEXT,payload TEXT,PRIMARY KEY(provider,player_id));
    CREATE TABLE IF NOT EXISTS real_observations(provider TEXT,player_id TEXT,match_id TEXT,match_date TEXT,payload TEXT,PRIMARY KEY(provider,player_id,match_id),FOREIGN KEY(provider,player_id) REFERENCES real_players(provider,player_id));
    CREATE TABLE IF NOT EXISTS game_real_links(real_provider TEXT,real_player_id TEXT,game_provider TEXT,game_asset_id INTEGER,game_row_number INTEGER,evidence TEXT,PRIMARY KEY(real_provider,real_player_id,game_provider),FOREIGN KEY(game_asset_id,game_row_number) REFERENCES game_profiles(asset_id,row_number));
    CREATE TABLE IF NOT EXISTS cross_source_validation(test_name TEXT PRIMARY KEY,payload TEXT);
    """)
    for pid, p in people.items():
        payload = {
            **p,
            "names": sorted(p["names"]),
            "birth_dates": sorted(p["birth_dates"]),
        }
        connection.execute(
            "INSERT OR REPLACE INTO real_players VALUES(?,?,?,?,?)",
            ("skillcorner", pid, p["name"], p["birth_date"], json.dumps(payload)),
        )
    for r in observations:
        connection.execute(
            "INSERT OR REPLACE INTO real_observations VALUES(?,?,?,?,?)",
            ("skillcorner", r["player_id"], r["match_id"], r["date"], json.dumps(r)),
        )
    connection.execute(
        "DELETE FROM game_real_links WHERE real_provider='skillcorner' AND game_provider IN ('fm','fifa') AND json_extract(evidence,'$.method')='full-name tokens with optional middle names + exact DOB; unique game ID; stable game DOB; no initials-only match'"
    )
    linked = {}
    link_stats = Counter()
    cache = {}
    for pid, aggregate in aggregates.items():
        person = people[pid]
        if len(person["birth_dates"]) != 1:
            link_stats["unknown_or_conflicting_real_birth_date"] += 1
            continue
        candidates = connection.execute(
            "SELECT * FROM game_profiles WHERE birth_date=? AND provider IN ('fm','fifa')",
            (person["birth_date"],),
        ).fetchall()
        candidates = [
            dict(r)
            for r in candidates
            if any(same_name(n, r["name"]) for n in person["names"])
        ]
        for provider in ["fm", "fifa"]:
            rows = [r for r in candidates if r["provider"] == provider]
            ids = {r["player_id"] for r in rows}
            if len(ids) != 1:
                link_stats[provider + "_unmatched_or_ambiguous"] += 1
                continue
            external_id = next(iter(ids))
            stable = (
                connection.execute(
                    "SELECT COUNT(DISTINCT birth_date) FROM game_profiles WHERE provider=? AND player_id=?",
                    (provider, external_id),
                ).fetchone()[0]
                == 1
            )
            if not stable:
                link_stats[provider + "_conflicting_game_birth_date"] += 1
                continue
            snapshot = choose_snapshot(rows, aggregate["first_match"])
            if not snapshot:
                link_stats[provider + "_no_dated_pre_match_snapshot"] += 1
                continue
            source = connection.execute(
                "SELECT i.kind,d.edition FROM imports i JOIN assets a ON a.id=i.asset_id JOIN datasets d ON d.id=a.dataset_id WHERE i.asset_id=? AND i.kind!='document'",
                (snapshot["asset_id"],),
            ).fetchone()
            if not source:
                continue
            table = source["kind"]
            key = (table, snapshot["asset_id"], snapshot["row_number"])
            if key not in cache:
                cache[key] = dict(
                    connection.execute(
                        'SELECT * FROM "'
                        + table.replace('"', '""')
                        + '" WHERE asset_id=? AND row_number=?',
                        key[1:],
                    ).fetchone()
                )
            attributes = cache[key]
            evidence = {
                "method": "full-name tokens with optional middle names + exact DOB; unique game ID; stable game DOB; no initials-only match",
                "real_name": person["name"],
                "game_name": snapshot["name"],
                "birth_date": person["birth_date"],
                "effective_date": snapshot["effective_date"],
                "first_real_match": aggregate["first_match"],
                "game_player_id": external_id,
                "game_table": table,
            }
            connection.execute(
                "INSERT OR REPLACE INTO game_real_links VALUES(?,?,?,?,?,?)",
                (
                    "skillcorner",
                    pid,
                    provider,
                    snapshot["asset_id"],
                    snapshot["row_number"],
                    json.dumps(evidence),
                ),
            )
            linked[(pid, provider)] = {
                "snapshot": snapshot,
                "attributes": attributes,
                "evidence": evidence,
            }
            link_stats[provider + "_linked"] += 1
    hypotheses = [
        ("fm", "pace", "p99_speed_mps", 1, 20),
        ("fm", "stamina", "distance_per_observed_minute", 1, 20),
        ("fm", "off_the_ball", "off_ball_runs_per90", 1, 20),
        ("fifa", "movement_sprint_speed", "p99_speed_mps", 1, 99),
        ("fifa", "power_stamina", "distance_per_observed_minute", 1, 99),
        ("fifa", "mentality_positioning", "off_ball_runs_per90", 1, 99),
    ]
    results = []
    for provider, attribute, target, minimum, maximum in hypotheses:
        pairs = []
        for pid, aggregate in aggregates.items():
            link = linked.get((pid, provider))
            if not link or aggregate[target] is None:
                continue
            try:
                rating = float(link["attributes"][attribute])
            except (KeyError, TypeError, ValueError):
                continue
            if not minimum <= rating <= maximum:
                continue
            birthday = date.fromisoformat(people[pid]["birth_date"])
            age = (
                date.fromisoformat(aggregate["first_match"]) - birthday
            ).days / 365.2425
            pairs.append(
                {
                    "player_id": pid,
                    "game_value": rating,
                    "real_value": aggregate[target],
                    "age": age,
                    "position": aggregate["position"],
                    "team": aggregate["team"],
                }
            )
        result = {
            "provider": provider,
            "game_attribute": attribute,
            "real_metric": target,
            **associations(pairs),
        }
        results.append(result)
        connection.execute(
            "INSERT OR REPLACE INTO cross_source_validation VALUES(?,?)",
            (provider + "_" + attribute + "_" + target, json.dumps(result)),
        )
    report = {
        "aggregation_backend": backend,
        "fpstreams_parity": True if backend == "compare" else None,
        "scope": "20 public A-League 2024/25 matches; detected-only physical segments; provider-classified off-ball events",
        "real_players_in_rosters": len(people),
        "eligible_real_players": len(aggregates),
        "eligible_player_match_rows": len(observations),
        "identity_audit": dict(link_stats),
        "observation_exclusions": exclusions,
        "tests": results,
        "protocol": {
            "minimum_played_minutes": 20,
            "minimum_observed_seconds": 600,
            "minimum_observation_fraction": 0.5,
            "maximum_observation_fraction": 1.05,
            "snapshot_cutoff": "strictly before earliest eligible match date; unknown same-day timing excluded",
            "aggregation": "one row per player; observation-duration weights for physical metrics; minutes weights for run rates",
            "selection": "all six predeclared pairs; no selection by favorable result",
            "uncertainty": "500 resamples of players; descriptive associations; no causal claim or multiplicity-adjusted significance claim",
            "identity": "unknown or contradictory birth dates and ambiguous game IDs excluded",
        },
        "limits": [
            "Small single-league public sample and selective camera detection limit transportability.",
            "Game scores are subjective estimates; tracking and dynamic events also contain measurement/model error.",
            "Observed speed percentile is not maximum athletic capacity.",
            "Off-ball events can include unknown extrapolation status; rates do not describe measured running distance.",
            "FMInside and undated EA/FC25/FC26 exports cannot pass the historical snapshot cutoff.",
            "Player-disjoint holdout does not establish future-season or unseen-league performance.",
        ],
    }
    connection.commit()
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    connection.close()
    Path(output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="data/football.sqlite3")
    parser.add_argument("--sources", default=".tmp")
    parser.add_argument("--output", default=".tmp/cross-source-validation.json")
    parser.add_argument(
        "--backend", choices=["python", "fpstreams", "compare"], default="compare"
    )
    args = parser.parse_args()
    validate(args.database, args.sources, args.output, backend=args.backend)
