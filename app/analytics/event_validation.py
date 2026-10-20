"""Cross-league validation of dated FIFA attributes against public Wyscout events."""

import argparse
from collections import Counter, defaultdict
from datetime import date
import json
import math
from pathlib import Path
import sqlite3
from .cross_validation import same_name, name_tokens, choose_snapshot, associations

LEAGUES = {
    "matches_England",
    "matches_France",
    "matches_Germany",
    "matches_Italy",
    "matches_Spain",
}


def event_metrics(events):
    stats = defaultdict(
        lambda: {
            "passes": 0,
            "accurate_passes": 0,
            "long_passes": 0,
            "accurate_long_passes": 0,
            "pass_length_sum": 0.0,
            "passes_with_length": 0,
            "progressive_passes": 0,
        }
    )
    errors = Counter()
    for event in events:
        if event.get("eventName") != "Pass" or event.get("subEventName") in {
            "Cross",
            "Head pass",
            "Hand pass",
        }:
            continue
        if not event.get("playerId"):
            errors["unknown_pass_player"] += 1
            continue
        if event.get("matchPeriod") not in {"1H", "2H"}:
            continue
        tags = {t["id"] for t in event.get("tags", [])}
        if (1801 in tags) == (1802 in tags):
            errors["unknown_or_conflicting_accuracy"] += 1
            continue
        key = (event["matchId"], event["playerId"])
        s = stats[key]
        s["passes"] += 1
        s["accurate_passes"] += 1801 in tags
        positions = event.get("positions", [])
        if len(positions) != 2:
            errors["missing_pass_geometry"] += 1
            continue
        a, b = positions
        try:
            coordinates = [float(point[key]) for point in (a, b) for key in ["x", "y"]]
            if not all(
                math.isfinite(value) and 0 <= value <= 100 for value in coordinates
            ):
                raise ValueError("Invalid geometry")
            length = math.hypot(b["x"] - a["x"], b["y"] - a["y"])
        except (KeyError, TypeError, ValueError):
            errors["invalid_pass_geometry"] += 1
            continue
        s["pass_length_sum"] += length
        s["passes_with_length"] += 1
        if length >= 30:
            s["long_passes"] += 1
            s["accurate_long_passes"] += 1801 in tags
        prior = math.hypot(100 - a["x"], 50 - a["y"])
        after = math.hypot(100 - b["x"], 50 - b["y"])
        if 1801 in tags and prior - after >= 10 and prior - after >= 0.25 * prior:
            s["progressive_passes"] += 1
    return stats, dict(errors)


def validate(
    database="data/football.sqlite3", output=".tmp/event-game-validation.json"
):
    db = sqlite3.connect(database)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    players = {}
    matches = {}
    ranks = {}
    duplicate_ranks = set()
    audit = Counter()
    for row in db.execute(
        "SELECT entity_type,payload FROM wyscout_entities WHERE entity_type='players' OR entity_type='playerank' OR entity_type LIKE 'matches_%'"
    ):
        p = json.loads(row["payload"])
        if row["entity_type"] == "players":
            players[p["wyId"]] = p
        elif row["entity_type"] in LEAGUES:
            if p["status"] == "Played" and p["duration"] == "Regular":
                matches[p["wyId"]] = {
                    **p,
                    "league": row["entity_type"].removeprefix("matches_"),
                }
        elif row["entity_type"] == "playerank":
            key = (p["matchId"], p["playerId"])
            if key in ranks:
                duplicate_ranks.add(key)
            else:
                ranks[key] = p
    for key in duplicate_ranks:
        ranks.pop(key, None)
    audit["ambiguous_playerank_rows"] = len(duplicate_ranks)
    selected_ids = set(matches)
    events = (
        json.loads(r["payload"])
        for r in db.execute(
            "SELECT payload FROM wyscout_events WHERE event_type='Pass' AND period IN ('1H','2H')"
        )
    )
    stats, pass_errors = event_metrics(
        event for event in events if event["matchId"] in selected_ids
    )
    aggregates = defaultdict(
        lambda: {
            "minutes": 0.0,
            "rank_sum": 0.0,
            "rank_minutes": 0.0,
            "matches": 0,
            "teams": defaultdict(float),
            "first_match": None,
            "leagues": set(),
            "passes": 0,
            "accurate_passes": 0,
            "long_passes": 0,
            "accurate_long_passes": 0,
            "pass_length_sum": 0.0,
            "passes_with_length": 0,
            "progressive_passes": 0,
        }
    )
    db.executescript("""
    CREATE TABLE IF NOT EXISTS real_players(provider TEXT,player_id TEXT,name TEXT,birth_date TEXT,payload TEXT,PRIMARY KEY(provider,player_id));
    CREATE TABLE IF NOT EXISTS real_observations(provider TEXT,player_id TEXT,match_id TEXT,match_date TEXT,payload TEXT,PRIMARY KEY(provider,player_id,match_id),FOREIGN KEY(provider,player_id) REFERENCES real_players(provider,player_id));
    CREATE TABLE IF NOT EXISTS game_real_links(real_provider TEXT,real_player_id TEXT,game_provider TEXT,game_asset_id INTEGER,game_row_number INTEGER,evidence TEXT,PRIMARY KEY(real_provider,real_player_id,game_provider),FOREIGN KEY(game_asset_id,game_row_number) REFERENCES game_profiles(asset_id,row_number));
    CREATE TABLE IF NOT EXISTS cross_source_validation(test_name TEXT PRIMARY KEY,payload TEXT);
    """)
    for pid, p in players.items():
        name = " ".join(
            filter(None, [p.get("firstName"), p.get("middleName"), p.get("lastName")])
        )
        db.execute(
            "INSERT OR REPLACE INTO real_players VALUES(?,?,?,?,?)",
            ("wyscout", str(pid), name, p.get("birthDate"), json.dumps(p)),
        )
    for (mid, pid), rank in ranks.items():
        if mid not in matches or pid not in players:
            continue
        player = players[pid]
        if player["role"]["code2"] == "GK":
            continue
        minutes = rank.get("minutesPlayed")
        if not isinstance(minutes, (int, float)) or not 0 < minutes <= 110:
            audit["invalid_playerank_minutes"] += 1
            continue
        match = matches[mid]
        team_id = None
        for tid, t in match["teamsData"].items():
            formation = t.get("formation") or {}
            listed = {
                p["playerId"]
                for p in formation.get("lineup", []) + formation.get("bench", [])
            }
            if pid in listed:
                if team_id is not None:
                    team_id = None
                    break
                team_id = str(tid)
        if team_id is None:
            audit["unknown_or_ambiguous_match_team"] += 1
            continue
        s = stats.get((mid, pid), {})
        g = aggregates[pid]
        g["minutes"] += minutes
        g["matches"] += 1
        g["teams"][team_id] += minutes
        g["leagues"].add(match["league"])
        when = match["dateutc"][:10]
        g["first_match"] = min(g["first_match"] or when, when)
        score = rank.get("playerankScore")
        if isinstance(score, (float, int)) and math.isfinite(score):
            g["rank_sum"] += score * minutes
            g["rank_minutes"] += minutes
        for key in [
            "passes",
            "accurate_passes",
            "long_passes",
            "accurate_long_passes",
            "pass_length_sum",
            "passes_with_length",
            "progressive_passes",
        ]:
            g[key] += s.get(key, 0)
        db.execute(
            "INSERT OR REPLACE INTO real_observations VALUES(?,?,?,?,?)",
            (
                "wyscout",
                str(pid),
                str(mid),
                when,
                json.dumps(
                    {
                        "minutes": minutes,
                        "team_id": team_id,
                        "league": match["league"],
                        "playerank_score": score,
                        **s,
                    }
                ),
            ),
        )
    db.execute(
        "DELETE FROM game_real_links WHERE real_provider='wyscout' AND game_provider='fifa' AND json_extract(evidence,'$.method')='name tokens + exact birthday + unique stable FIFA ID'"
    )
    links = {}
    stable_cache = {}
    for pid, g in aggregates.items():
        if g["minutes"] < 450 or len(g["leagues"]) != 1:
            continue
        player = players[pid]
        name = " ".join(
            filter(
                None,
                [
                    player.get("firstName"),
                    player.get("middleName"),
                    player.get("lastName"),
                ],
            )
        )
        birthday = player.get("birthDate")
        try:
            date.fromisoformat(birthday)
        except (ValueError, TypeError):
            audit["invalid_real_birth_date"] += 1
            continue
        # Prefix filtering reduces millions of historical updates; final matching
        # still requires full name tokens, exact DOB and one stable game ID.
        prefix = (
            (player.get("firstName") or "").split()[0]
            if player.get("firstName")
            else ""
        )
        normalized = name_tokens(prefix)
        alternate = normalized[0] if normalized else prefix
        candidates = [
            dict(r)
            for r in db.execute(
                "SELECT * FROM game_profiles WHERE birth_date=? AND provider='fifa' AND gender='male' AND effective_date<? AND (name LIKE ? OR name LIKE ?)",
                (birthday, g["first_match"], prefix + "%", alternate + "%"),
            )
            if same_name(name, r["name"])
        ]
        ids = {r["player_id"] for r in candidates}
        if len(ids) != 1:
            audit["unmatched_or_ambiguous_fifa_identity"] += 1
            continue
        identity = next(iter(ids))
        if identity not in stable_cache:
            stable_cache[identity] = (
                db.execute(
                    "SELECT COUNT(DISTINCT birth_date) FROM game_profiles WHERE provider='fifa' AND player_id=?",
                    (identity,),
                ).fetchone()[0]
                == 1
            )
        if not stable_cache[identity]:
            audit["conflicting_game_birth_date"] += 1
            continue
        snapshot = choose_snapshot(candidates, g["first_match"])
        if snapshot is None:
            continue
        table = db.execute(
            "SELECT kind FROM imports WHERE asset_id=? AND kind LIKE '%players'",
            (snapshot["asset_id"],),
        ).fetchone()
        if not table:
            continue
        attrs = dict(
            db.execute(
                'SELECT * FROM "'
                + table[0].replace('"', '""')
                + '" WHERE asset_id=? AND row_number=?',
                (snapshot["asset_id"], snapshot["row_number"]),
            ).fetchone()
        )
        evidence = {
            "method": "name tokens + exact birthday + unique stable FIFA ID",
            "real_name": name,
            "game_name": snapshot["name"],
            "birth_date": birthday,
            "effective_date": snapshot["effective_date"],
            "first_real_match": g["first_match"],
            "game_player_id": identity,
            "game_table": table[0],
        }
        db.execute(
            "INSERT OR REPLACE INTO game_real_links VALUES(?,?,?,?,?,?)",
            (
                "wyscout",
                str(pid),
                "fifa",
                snapshot["asset_id"],
                snapshot["row_number"],
                json.dumps(evidence),
            ),
        )
        links[pid] = {"attributes": attrs, "snapshot": snapshot}
        audit["fifa_linked"] += 1
    hypotheses = [
        ("attacking_short_passing", "pass_completion", 100),
        ("skill_long_passing", "long_pass_completion", 50),
        ("mentality_vision", "progressive_passes_per90", 100),
        ("overall", "supplied_playerank", 100),
    ]
    results = []
    for attribute, target, min_passes in hypotheses:
        pairs = []
        for pid, link in links.items():
            g = aggregates[pid]
            p = players[pid]
            if g["passes"] < min_passes or g["passes_with_length"] < 0.8 * g["passes"]:
                continue
            if target == "long_pass_completion" and g["long_passes"] < 50:
                continue
            try:
                value = float(link["attributes"][attribute])
            except (KeyError, TypeError, ValueError):
                continue
            if not 1 <= value <= 99:
                continue
            y = {
                "pass_completion": g["accurate_passes"] / g["passes"],
                "long_pass_completion": (
                    g["accurate_long_passes"] / g["long_passes"]
                    if g["long_passes"]
                    else None
                ),
                "progressive_passes_per90": g["progressive_passes"] / g["minutes"] * 90,
                "supplied_playerank": (
                    g["rank_sum"] / g["rank_minutes"] if g["rank_minutes"] else None
                ),
            }[target]
            if y is None:
                continue
            pairs.append(
                {
                    "player_id": str(pid),
                    "game_value": value,
                    "real_value": y,
                    "position": p["role"]["name"],
                    "age": (
                        date.fromisoformat(g["first_match"])
                        - date.fromisoformat(p["birthDate"])
                    ).days
                    / 365.2425,
                    "team": max(g["teams"], key=g["teams"].get),
                    "league": next(iter(g["leagues"])),
                    "mean_pass_length": g["pass_length_sum"] / g["passes_with_length"],
                    "long_pass_share": g["long_passes"] / g["passes"],
                    "minutes": g["minutes"],
                }
            )
        controls = ("mean_pass_length", "long_pass_share", "minutes")
        result = {
            "provider": "fifa",
            "game_attribute": attribute,
            "real_metric": target,
            **associations(pairs, additional_controls=controls),
        }
        result["league_counts"] = dict(Counter(p["league"] for p in pairs))
        results.append(result)
        db.execute(
            "INSERT OR REPLACE INTO cross_source_validation VALUES(?,?)",
            ("wyscout_fifa_" + attribute + "_" + target, json.dumps(result)),
        )
    report = {
        "scope": "Wyscout 2017/18 England, France, Germany, Italy, Spain; regular-time outfield players",
        "real_player_records": len(players),
        "matches": len(matches),
        "identity_audit": dict(audit),
        "event_quality": pass_errors,
        "tests": results,
        "protocol": {
            "minimum_season_minutes": 450,
            "short_pass_minimum_attempts": 100,
            "long_pass_minimum_attempts": 50,
            "cutoff": "strictly before earliest appearance date; same-day, future and undated snapshots excluded",
            "passing": "Pass events excluding crosses and head passes; accurate/inaccurate tag required; first/second half only",
            "long_pass": "endpoint distance >=30 in provider 0-100 coordinate units; a proxy, not measured meters",
            "progression": "successful pass reduces provider-unit goal distance >=10 and >=25%",
            "playerank": "minutes-weighted mean of supplied per-match score; no probability or ability-scale interpretation",
            "descriptive_controls": [
                "position",
                "age",
                "team",
                "mean pass length",
                "long pass share",
                "minutes",
            ],
            "prediction_controls": ["position", "age", "team"],
            "uncertainty": "500 player resamples; exploratory tests, no multiplicity-adjusted significance claims",
        },
        "limits": [
            "Selection requires 450 minutes and identifiable dated game records; not all players are represented.",
            "Provider roster position is a broad static label, not an observed per-action role.",
            "Per-match minutes and performance score use supplied PlayeRank definitions.",
            "Coordinate-derived pass lengths assume the provider direction convention and retain native units.",
            "Retrospective player-disjoint holdout does not demonstrate future-season or unseen-league prediction.",
        ],
    }
    db.commit()
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    db.close()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="data/football.sqlite3")
    parser.add_argument("--output", default=".tmp/event-game-validation.json")
    args = parser.parse_args()
    validate(args.database, args.output)
