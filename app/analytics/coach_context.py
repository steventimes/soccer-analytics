"""Match-day coaches, observed spells and within-team tactical comparisons.

Observed spell endpoints are not appointment dates. Coach changes are selected
events and before/after differences are not causal coaching effects.
"""

from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

from .artifacts import archive_outputs, file_digest
from app.data_service.warehouse import Warehouse


def match_manager(managers):
    unique = {str(m["id"]): m for m in managers if m.get("id") is not None}
    if len(unique) != 1:
        return {
            "coach_id": None,
            "coach_name": None,
            "coach_candidates": list(unique.values()),
        }
    cid, manager = next(iter(unique.items()))
    return {
        "coach_id": cid,
        "coach_name": manager.get("name"),
        "coach_candidates": list(unique.values()),
    }


def supported_tenure(match, side, tenures):
    choices = [
        r
        for r in tenures
        if r["team"] == match[side + "_team"]
        and r["gender"] == match["gender"]
        and r["age_group"] == match["age_group"]
        and r["start"] <= match["date"] <= r["supported_until"]
    ]
    return choices[0] if len(choices) == 1 else None


def split_spells(rows):
    spells = []
    for row in sorted(rows, key=lambda r: r["date"]):
        if not spells or spells[-1][-1]["coach_id"] != row["coach_id"]:
            spells.append([])
        spells[-1].append(row)
    return spells


def collect_observations(connection, tenures):
    observations = {}
    for asset, payload in connection.execute(
        "SELECT asset_id,payload FROM documents WHERE kind='statsbomb_matches' ORDER BY asset_id"
    ):
        for match in json.loads(payload):
            for side in ["home", "away"]:
                team = match[side + "_team"]
                tid = str(team[side + "_team_id"])
                record = {
                    "provider": "statsbomb",
                    "match_id": str(match["match_id"]),
                    "team_id": tid,
                    "team": team[side + "_team_name"],
                    "date": match["match_date"],
                    "competition_id": str(match["competition"]["competition_id"]),
                    "season_id": str(match["season"]["season_id"]),
                    "evidence_type": "match_metadata",
                    "asset_id": asset,
                    **match_manager(team.get("managers") or []),
                }
                observations[("statsbomb", record["match_id"], tid)] = record
    coaches = {
        str(cid): json.loads(p)
        for cid, p in connection.execute(
            "SELECT entity_id,payload FROM wyscout_entities WHERE entity_type='coaches'"
        )
    }
    teams = {
        str(cid): json.loads(p)
        for cid, p in connection.execute(
            "SELECT entity_id,payload FROM wyscout_entities WHERE entity_type='teams'"
        )
    }
    for asset, payload in connection.execute(
        "SELECT asset_id,payload FROM wyscout_entities WHERE entity_type LIKE 'matches_%' ORDER BY asset_id"
    ):
        match = json.loads(payload)
        for tid, team in match["teamsData"].items():
            cid = str(team["coachId"]) if team.get("coachId") else None
            coach = coaches.get(cid, {})
            record = {
                "provider": "wyscout",
                "match_id": str(match["wyId"]),
                "team_id": tid,
                "team": teams.get(tid, {}).get("name", tid),
                "date": match["dateutc"][:10],
                "competition_id": str(match["competitionId"]),
                "season_id": str(match["seasonId"]),
                "evidence_type": "match_coach_id",
                "asset_id": asset,
                "coach_id": cid,
                "coach_name": (
                    (
                        " ".join(
                            [coach.get("firstName", ""), coach.get("lastName", "")]
                        ).strip()
                        or coach.get("shortName")
                    )
                    if cid
                    else None
                ),
            }
            observations[("wyscout", record["match_id"], tid)] = record
    for payload, asset in connection.execute(
        "SELECT payload,asset_id FROM latest_match_observations WHERE provider='espn'"
    ):
        match = json.loads(payload)
        for side in ["home", "away"]:
            tenure = supported_tenure(match, side, tenures)
            record = {
                "provider": "espn",
                "match_id": match["match_id"],
                "team_id": match[side + "_team_id"],
                "team": match[side + "_team"],
                "date": match["date"],
                "competition_id": match["competition"],
                "season_id": match["date"][:4],
                "asset_id": asset,
                "coach_id": tenure["coach_id"] if tenure else None,
                "coach_name": tenure["coach_name"] if tenure else None,
                "evidence_type": (
                    "dated_official_tenure" if tenure else "unknown_match_coach"
                ),
                "tenure_source": tenure["source_url"] if tenure else None,
            }
            observations[("espn", record["match_id"], record["team_id"])] = record
    return observations


def tactical_panels(connection, observations):
    run = connection.execute(
        "SELECT run_id FROM tactical_assessment_runs ORDER BY rowid DESC LIMIT 1"
    ).fetchone()[0]
    rows = []
    for payload in connection.execute(
        "SELECT payload FROM tactical_team_matches WHERE run_id=?", (run,)
    ):
        r = json.loads(payload[0])
        coach = observations[("statsbomb", str(r["match_id"]), str(r["team_id"]))]
        rows.append(
            {
                **r,
                **coach,
                "xg_difference": r["npxg"] - r["opponent_npxg"],
                "pass_share": r["on_ball_action_share_proxy"],
                "forward_pass_share": None,
            }
        )
    # Wyscout carries per-match coach IDs but no pressure events or tracking.
    aggregate = connection.execute(
        """SELECT match_id,team_id,
        SUM(event_type='Pass') passes,
        SUM(event_type='Shot' OR (event_type='Free Kick' AND json_extract(payload,'$.subEventName')='Free kick shot')) shots,
        SUM(CASE WHEN event_type='Pass' AND json_array_length(payload,'$.positions')=2 AND json_extract(payload,'$.positions[1].x')-json_extract(payload,'$.positions[0].x')>=15 THEN 1 ELSE 0 END) forward_passes
        FROM wyscout_events WHERE period IN ('1H','2H') GROUP BY match_id,team_id"""
    ).fetchall()
    by_match = defaultdict(list)
    for mid, tid, passes, shots, forward in aggregate:
        by_match[str(mid)].append(
            {
                "team_id": str(tid),
                "passes": passes,
                "shots": shots,
                "forward_passes": forward,
            }
        )
    for (provider, mid, tid), coach in observations.items():
        if provider != "wyscout":
            continue
        games = by_match.get(mid, [])
        own = next((r for r in games if r["team_id"] == tid), None)
        if len(games) != 2 or own is None:
            continue
        other = next(r for r in games if r["team_id"] != tid)
        rows.append(
            {
                **coach,
                "shots": own["shots"],
                "shot_difference": own["shots"] - other["shots"],
                "pass_share": (
                    own["passes"] / (own["passes"] + other["passes"])
                    if own["passes"] + other["passes"]
                    else None
                ),
                "forward_pass_share": (
                    own["forward_passes"] / own["passes"] if own["passes"] else None
                ),
                "starting_formation": None,
                "pressure_per100_opponent_passes": None,
                "npxg_per100_possessions": None,
            }
        )
    return rows, run


def summarize(rows):
    import numpy as np

    contexts = defaultdict(list)
    for row in rows:
        contexts[
            (row["provider"], row["competition_id"], row["season_id"], row["team_id"])
        ].append(row)
    profiles = []
    changes = []
    metrics = [
        "pass_share",
        "forward_pass_share",
        "shots",
        "shot_difference",
        "pressure_per100_opponent_passes",
        "npxg_per100_possessions",
        "xg_difference",
    ]
    for key, context in sorted(contexts.items()):
        spells = split_spells(context)
        for number, spell in enumerate(spells):
            coach = spell[0]
            forms = Counter(
                str(r["starting_formation"])
                for r in spell
                if r.get("starting_formation")
            )
            summary = {
                m: (
                    float(np.mean([r[m] for r in spell if r.get(m) is not None]))
                    if any(r.get(m) is not None for r in spell)
                    else None
                )
                for m in metrics
            }
            profiles.append(
                {
                    "context": list(key),
                    "team": coach["team"],
                    "coach_id": coach["coach_id"],
                    "coach_name": coach["coach_name"],
                    "observed_first_match": spell[0]["date"],
                    "observed_last_match": spell[-1]["date"],
                    "matches": len(spell),
                    "formation_counts": dict(forms),
                    "metrics": summary,
                    "boundary": "observed match sequence; not verified appointment/resignation dates",
                }
            )
            if (
                number == 0
                or coach["coach_id"] is None
                or spells[number - 1][0]["coach_id"] is None
            ):
                continue
            before = spells[number - 1][-8:]
            after = spell[:8]
            if min(len(before), len(after)) < 5:
                continue
            shifts = {}
            for metric in metrics:
                a = [r[metric] for r in before if r.get(metric) is not None]
                b = [r[metric] for r in after if r.get(metric) is not None]
                if min(len(a), len(b)) < 5:
                    continue
                rng = np.random.default_rng(20261011)
                draws = np.mean(rng.choice(b, (1000, len(b))), axis=1) - np.mean(
                    rng.choice(a, (1000, len(a))), axis=1
                )
                shifts[metric] = {
                    "before": float(np.mean(a)),
                    "after": float(np.mean(b)),
                    "difference": float(np.mean(b) - np.mean(a)),
                    "exploratory_match_bootstrap_ci95": np.quantile(
                        draws, [0.025, 0.975]
                    ).tolist(),
                }
            changes.append(
                {
                    "context": list(key),
                    "team": coach["team"],
                    "old_coach": before[0]["coach_name"],
                    "new_coach": coach["coach_name"],
                    "old_coach_id": before[0]["coach_id"],
                    "new_coach_id": coach["coach_id"],
                    "first_observed_new_coach_match": after[0]["date"],
                    "before_matches": len(before),
                    "after_matches": len(after),
                    "metrics": shifts,
                    "before_formation_counts": dict(
                        Counter(
                            str(r["starting_formation"])
                            for r in before
                            if r.get("starting_formation")
                        )
                    ),
                    "after_formation_counts": dict(
                        Counter(
                            str(r["starting_formation"])
                            for r in after
                            if r.get("starting_formation")
                        )
                    ),
                    "interpretation": "same team/competition/season; opponents, injuries, lineup selection and regression to the mean remain uncontrolled; no coach ranking or causal effect",
                }
            )
    return profiles, changes


def run(
    database="data/football.sqlite3", output=".tmp/coach-context", tenure_file=None
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tenures = json.loads(Path(tenure_file).read_text()) if tenure_file else []
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as c:
        observations = collect_observations(c, tenures)
        panels, tactical_run = tactical_panels(c, observations)
    profiles, changes = summarize(panels)
    report = {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "coach_observation_coverage": dict(
            Counter(
                r["provider"] + "/" + ("known" if r["coach_id"] else "unknown")
                for r in observations.values()
            )
        ),
        "tactical_run": tactical_run,
        "tactical_team_matches": len(panels),
        "observed_spell_profiles": profiles,
        "within_season_coach_changes": changes,
        "official_tenure_rules": tenures,
        "extractor_sha256": file_digest(__file__),
        "limits": [
            "No current-roster coach is assigned to historical fixtures.",
            "Match-day IDs remain scoped to each provider.",
            "Unknown coaches interrupt observed spells.",
            "Before/after samples are up to8 matches per side, minimum5; bootstrap intervals are exploratory and do not adjust fixture difficulty or serial dependence.",
            "Coach changes often follow poor results, so improvements can reflect regression to the mean.",
            "Sparse national-team matches and club fixtures remain separate contexts.",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    run_id = file_digest(output / "report.json")[:16]
    with Warehouse(database) as db:
        db.connection.executescript(
            """CREATE TABLE IF NOT EXISTS match_coach_observations(provider TEXT,match_id TEXT,team_id TEXT,match_date TEXT,coach_id TEXT,coach_name TEXT,evidence_type TEXT,asset_id INTEGER REFERENCES assets(id),payload TEXT,PRIMARY KEY(provider,match_id,team_id));
            CREATE TABLE IF NOT EXISTS coach_tactical_runs(run_id TEXT PRIMARY KEY,payload TEXT);
            CREATE VIEW IF NOT EXISTS coach_tactical_team_matches AS SELECT t.run_id,t.match_id,t.team_id,c.coach_id,c.coach_name,c.evidence_type,t.payload tactical_payload,c.payload coach_payload FROM tactical_team_matches t LEFT JOIN match_coach_observations c ON c.provider='statsbomb' AND c.match_id=CAST(t.match_id AS TEXT) AND c.team_id=CAST(t.team_id AS TEXT);"""
        )
        db.connection.executemany(
            "INSERT OR REPLACE INTO match_coach_observations VALUES(?,?,?,?,?,?,?,?,?)",
            [
                (
                    r["provider"],
                    r["match_id"],
                    r["team_id"],
                    r["date"],
                    r["coach_id"],
                    r["coach_name"],
                    r["evidence_type"],
                    r["asset_id"],
                    json.dumps(r),
                )
                for r in observations.values()
            ],
        )
        db.connection.execute(
            "INSERT OR REPLACE INTO coach_tactical_runs VALUES(?,?)",
            (run_id, json.dumps(report)),
        )
        archive_outputs(db, "coach_context_artifacts", output, "coach_context", run_id)
        db.connection.commit()
        db.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    print(
        json.dumps(
            {
                "run_id": run_id,
                "coach_coverage": report["coach_observation_coverage"],
                "tactical_team_matches": len(panels),
                "coach_change_windows": len(changes),
            },
            indent=2,
        )
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--database", default="data/football.sqlite3")
    p.add_argument("--output", default=".tmp/coach-context")
    p.add_argument("--tenures")
    a = p.parse_args()
    run(a.database, a.output, a.tenures)
