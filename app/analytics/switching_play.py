"""Descriptive cross-field pass windows with match-day coach provenance.

This ball-movement proxy does not identify weak-side space or a causal tactic.
Pass windows overlap; rates describe passes, not independent possessions.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sqlite3

from .artifacts import archive_outputs, file_digest
from .tactical_relations import clock, point
from .switch_context import snapshot_context
from app.data_service.warehouse import Warehouse


def pass_windows(events):
    possessions = defaultdict(list)
    for index, event in enumerate(events):
        owner = (event.get("possession_team") or {}).get("id")
        if (
            event.get("period") in {1, 2}
            and owner is not None
            and event.get("possession") is not None
        ):
            possessions[(event["period"], event["possession"], owner)].append(
                (index, event)
            )
    rows = []
    exclusions = Counter()
    for (period, possession, owner), sequence in possessions.items():
        shots = [
            (i, e)
            for i, e in sequence
            if (e.get("type") or {}).get("name") == "Shot"
            and (e.get("team") or {}).get("id") == owner
        ]
        for index, event in sequence:
            if (event.get("type") or {}).get("name") != "Pass" or (
                event.get("team") or {}
            ).get("id") != owner:
                continue
            details = event.get("pass") or {}
            if details.get("outcome") or details.get("type"):
                exclusions["incomplete_or_special_pass"] += 1
                continue
            start, end = point(event.get("location")), point(
                details.get("end_location")
            )
            t, duration = clock(event), event.get("duration")
            if (
                start is None
                or end is None
                or t is None
                or not math.isfinite(t)
                or not isinstance(duration, (int, float))
                or isinstance(duration, bool)
                or not 0 <= duration <= 30
            ):
                exclusions["unknown_geometry_or_end_time"] += 1
                continue
            decision = t + duration
            if any(
                i <= index or (clock(s) is not None and clock(s) <= decision)
                for i, s in shots
            ):
                exclusions["shot_already_observed"] += 1
                continue
            later = [
                s
                for i, s in shots
                if i > index
                and clock(s) is not None
                and 0 < clock(s) - decision <= 10
                and (s.get("shot", {}).get("type") or {}).get("name") != "Penalty"
            ]
            cross_field = (start[1] < 80 / 3 and end[1] >= 160 / 3) or (
                end[1] < 80 / 3 and start[1] >= 160 / 3
            )
            rows.append(
                {
                    "event_id": event.get("id"),
                    "team_id": str(owner),
                    "period": period,
                    "possession_id": possession,
                    "cross_field": cross_field,
                    "shot_within10s": int(bool(later)),
                    "decision_clock": decision,
                    "start_x_zone": min(2, int(start[0] // 40)),
                    "start": start,
                    "end": end,
                    "pressure_marked": event.get("under_pressure") is True,
                }
            )
    return rows, dict(exclusions)


def comparison(rows):
    groups = {}
    for label, flag in [("cross_field", True), ("other_completed_passes", False)]:
        eligible = [r for r in rows if r["cross_field"] == flag]
        groups[label] = {
            "passes": len(eligible),
            "shot_windows": sum(r["shot_within10s"] for r in eligible),
            "shot_window_rate": (
                sum(r["shot_within10s"] for r in eligible) / len(eligible)
                if eligible
                else None
            ),
        }
    a, b = (
        groups[k]["shot_window_rate"] for k in ["cross_field", "other_completed_passes"]
    )
    groups["raw_rate_difference"] = a - b if a is not None and b is not None else None
    return groups


def run(
    database="data/football.sqlite3",
    output=".tmp/switching-play",
    competition=55,
    season="2024",
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as c:
        matches = {}
        for aid, payload in c.execute(
            "SELECT asset_id,payload FROM documents WHERE kind='statsbomb_matches' ORDER BY asset_id"
        ):
            for match in json.loads(payload):
                if (
                    match["competition"]["competition_id"] == competition
                    and match["season"]["season_name"] == season
                ):
                    matches[match["match_id"]] = (match, aid)
        assets = {
            int(Path(path).stem): (aid, digest, size)
            for aid, path, digest, size in c.execute(
                "SELECT d.asset_id,a.path,a.sha256,a.bytes FROM documents d JOIN assets a ON a.id=d.asset_id WHERE d.kind='statsbomb_events'"
            )
            if Path(path).stem.isdigit() and int(Path(path).stem) in matches
        }
        coaches = {
            (mid, tid): json.loads(payload)
            for mid, tid, payload in c.execute(
                "SELECT match_id,team_id,payload FROM match_coach_observations WHERE provider='statsbomb'"
            )
        }
        rows, exclusions = [], Counter()
        frame_assets = {
            int(Path(path).stem): (aid, digest, size)
            for aid, path, digest, size in c.execute(
                "SELECT d.asset_id,a.path,a.sha256,a.bytes FROM documents d JOIN assets a ON a.id=d.asset_id WHERE d.kind='statsbomb_360'"
            )
            if Path(path).stem.isdigit() and int(Path(path).stem) in matches
        }
        for mid, (meta, _) in sorted(matches.items()):
            if mid not in assets:
                continue
            events = json.loads(
                c.execute(
                    "SELECT payload FROM documents WHERE asset_id=?", (assets[mid][0],)
                ).fetchone()[0]
            )
            windows, audit = pass_windows(events)
            frames = (
                {
                    r["event_uuid"]: r
                    for r in json.loads(
                        c.execute(
                            "SELECT payload FROM documents WHERE asset_id=?",
                            (frame_assets[mid][0],),
                        ).fetchone()[0]
                    )
                }
                if mid in frame_assets
                else {}
            )
            exclusions.update(audit)
            for row in windows:
                row["spatial_context"] = snapshot_context(
                    row, frames.get(row["event_id"])
                )
                side = (
                    "home"
                    if row["team_id"] == str(meta["home_team"]["home_team_id"])
                    else "away"
                )
                opponent = "away" if side == "home" else "home"
                coach = coaches.get((str(mid), row["team_id"]), {})
                row.update(
                    {
                        "match_id": mid,
                        "date": meta["match_date"],
                        "team": meta[side + "_team"][side + "_team_name"],
                        "opponent": meta[opponent + "_team"][opponent + "_team_name"],
                        "coach_id": coach.get("coach_id"),
                        "coach_name": coach.get("coach_name"),
                        "coach_evidence_type": coach.get("evidence_type", "unknown"),
                    }
                )
            rows.extend(windows)
        input_assets = sorted(
            set(
                [r[1] for r in matches.values()]
                + [r[0] for r in assets.values()]
                + [r[0] for r in frame_assets.values()]
            )
        )
        provenance = [
            list(
                c.execute(
                    "SELECT id,sha256,bytes FROM assets WHERE id=?", (aid,)
                ).fetchone()
            )
            for aid in input_assets
        ]
        cards = [
            json.loads(payload)
            for payload, in c.execute(
                "SELECT payload FROM tactical_evidence_claims WHERE claim_id='switch-draw'"
            )
        ]
    if not rows:
        raise ValueError("No eligible pass windows in this competition/season")
    by_team, by_opponent = defaultdict(list), defaultdict(list)
    for row in rows:
        by_team[
            (row["team_id"], row["team"], row["coach_id"], row["coach_name"])
        ].append(row)
        by_opponent[(row["team"], row["opponent"], row["coach_id"])].append(row)
    report = {
        "competition_id": competition,
        "season": season,
        "metadata_matches": len(matches),
        "event_matches": len(assets),
        "eligible_passes": len(rows),
        "overall": comparison(rows),
        "spatial_context_coverage": {
            "snapshot_pass_windows": sum(
                r["spatial_context"]["snapshot_available"] for r in rows
            ),
            "cross_field_with_snapshot": sum(
                r["cross_field"] and r["spatial_context"]["snapshot_available"]
                for r in rows
            ),
            "cross_field_target_local_visible": sum(
                r["cross_field"]
                and r["spatial_context"].get("target_local_visible", False)
                for r in rows
            ),
        },
        "pressure_flag_comparisons": [
            {
                "event_under_pressure_marked": flag,
                **comparison([r for r in rows if r["pressure_marked"] == flag]),
            }
            for flag in [False, True]
        ],
        "target_density_comparisons": [
            {
                "visible_opponents_within5_units": label,
                **comparison(
                    [
                        r
                        for r in rows
                        if r["spatial_context"].get("target_local_visible")
                        and (
                            r["spatial_context"]["target_opponents_within5_units"] <= 1
                        )
                        == low
                    ]
                ),
            }
            for label, low in [("0_to1", True), ("2_or_more", False)]
        ],
        "start_zone_comparisons": [
            {
                "start_x_zone": zone,
                **comparison([r for r in rows if r["start_x_zone"] == zone]),
            }
            for zone in sorted({r["start_x_zone"] for r in rows})
        ],
        "team_coach_profiles": [
            {
                "team_id": key[0],
                "team": key[1],
                "coach_id": key[2],
                "coach_name": key[3],
                "matches": len({r["match_id"] for r in group}),
                **comparison(group),
            }
            for key, group in sorted(by_team.items(), key=lambda x: str(x[0]))
        ],
        "opponent_profiles": [
            {
                "team": key[0],
                "opponent": key[1],
                "coach_id": key[2],
                "matches": len({r["match_id"] for r in group}),
                **comparison(group),
            }
            for key, group in sorted(by_opponent.items(), key=lambda x: str(x[0]))
        ],
        "exclusions": dict(exclusions),
        "source_asset_fingerprint": hashlib.sha256(
            json.dumps(provenance).encode()
        ).hexdigest(),
        "source_assets": len(provenance),
        "extractor_sha256": file_digest(__file__),
        "source_claim_ids": [card["id"] for card in cards],
        "method": "Completed non-special passes in regulation periods, from one outer y-third to the opposite on StatsBomb120x80 coordinates. Decision is pass end; outcome is a later non-penalty shot within10s by the same possession owner. Passes after an already observed shot are excluded.",
        "limits": [
            "A single cross-field pass is a ball-movement candidate, not proof of defensive displacement, weak-side exploitation or coach intent.",
            "360 context reports visible outfield players at pass time; shape spans are partial. Ball side/far side counts do not establish overloads or off-ball runs. Local visibility is polygon coverage, not proof that every player was detected.",
            "Distances use StatsBomb normalized pitch units, not metres. The binary event pressure flag is not pressing intensity; absent marking is not a measured low-pressure condition.",
            "Other completed passes are descriptive controls, not matched tactical alternatives. Starting locations, pressure, score, personnel and opponents differ.",
            "Pass windows overlap within possessions; no independent-sample significance or causal coaching effect is claimed.",
            "Coach IDs remain StatsBomb-specific. Opponent profiles can have one match; rates are not coach rankings.",
            "2024 tournament findings do not validate2026 source statements or describe current national-team styles.",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    run_id = file_digest(output / "report.json")[:16]
    with Warehouse(database) as db:
        db.connection.execute(
            "CREATE TABLE IF NOT EXISTS tactical_switch_runs(run_id TEXT PRIMARY KEY,payload TEXT)"
        )
        db.connection.execute(
            "CREATE TABLE IF NOT EXISTS tactical_switch_windows(run_id TEXT REFERENCES tactical_switch_runs(run_id),match_id INTEGER,event_id TEXT,team_id TEXT,payload TEXT,PRIMARY KEY(run_id,match_id,event_id))"
        )
        db.connection.execute(
            "INSERT OR REPLACE INTO tactical_switch_runs VALUES(?,?)",
            (run_id, json.dumps(report)),
        )
        db.connection.executemany(
            "INSERT OR REPLACE INTO tactical_switch_windows VALUES(?,?,?,?,?)",
            [
                (
                    run_id,
                    row["match_id"],
                    row["event_id"],
                    row["team_id"],
                    json.dumps(row),
                )
                for row in rows
            ],
        )
        archive_outputs(
            db, "switching_play_artifacts", output, "switching_play", run_id
        )
        db.connection.commit()
        db.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    print(
        json.dumps(
            {
                "run_id": run_id,
                "matches": len(assets),
                "eligible_passes": len(rows),
                "overall": report["overall"],
            },
            indent=2,
        )
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--database", default="data/football.sqlite3")
    p.add_argument("--output", default=".tmp/switching-play")
    p.add_argument("--competition", type=int, default=55)
    p.add_argument("--season", default="2024")
    a = p.parse_args()
    run(a.database, a.output, a.competition, a.season)
