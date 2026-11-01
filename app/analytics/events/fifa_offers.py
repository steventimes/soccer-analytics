"""Verified offers/receptions from official FIFA reports; no trajectory inference."""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import subprocess

from app.analytics.artifacts import archive_outputs, file_digest
from app.data_service import fifa_reports
from app.data_service.warehouse import Warehouse
from app.file_io import write_json_atomic

FIELDS = ("total_offers", "in_front", "in_between", "out_to_in", "in_to_out",
          "in_behind", "no_movement", "offers_received")
HEADERS = ("Total Offers", "In Front", "In Between", "Out to In", "In to Out",
           "In Behind", "No Movement", "Offers Received")
SCHEMA = "SoccerAnalytics.FifaOffers.v1"


def parse_offers(pages, teams, rosters):
    """Require explicit cells, player identities and independent overview totals."""
    if len(teams) != 2 or len(set(teams)) != 2 or set(rosters) != {0, 1}:
        raise ValueError("Offers require two identified team rosters")
    rows, summaries, overviews = [], {}, {}
    for page, text in enumerate(pages, 1):
        title = re.search(r"^\s*In Possession - Offers & Receptions[ \t]+([^\n]+)$", text, re.M)
        overview = re.search(r"^\s*Offering to Receive[ \t]+([^\n]+)$", text, re.M)
        if not title and not overview:
            continue
        team = (title or overview)[1].strip()
        if team not in teams:
            raise ValueError("Offers team identity mismatch")
        side = teams.index(team)
        if overview:
            if side in overviews or "Total Offers Made" not in text or "Total Offers Received" not in text:
                raise ValueError("Duplicate or unrecognized offers overview")
            # First numeric pair precedes the labelled totals, above the player list.
            pair = re.search(r"^[ \t]*(\d+)[ \t]+(\d+)[ \t]*$", text.split("Total Offers Made")[0], re.M)
            if not pair:
                raise ValueError("Offers overview totals unreadable")
            overviews[side] = (page, tuple(map(int, pair.groups())))
            continue
        if side in summaries or not rosters[side]:
            raise ValueError("Duplicate offers page or missing player roster")
        lines = text.splitlines()
        start = next((i for i, line in enumerate(lines) if re.match(r"\s*#\s+Player\b", line)), None)
        if start is None or " ".join(lines[start].split()) != "# Player " + " ".join(HEADERS):
            raise ValueError("Offers column meanings changed")
        seen, totals = set(), dict.fromkeys(FIELDS, 0)
        for line in lines[start + 1:]:
            if not line.strip():
                continue
            row = re.fullmatch(r"\s*(\d{1,2})\s+(.+?)\s+((?:\d+\s+){7}\d+)\s*", line)
            if not row:
                raise ValueError("Offers row lacks eight explicit count cells")
            shirt, name = int(row[1]), " ".join(row[2].split())
            values = list(map(int, row[3].split()))
            if shirt in seen or rosters[side].get(shirt) != name:
                raise ValueError("Offers player identity missing, duplicated or inconsistent")
            if values[0] != sum(values[1:7]) or values[-1] > values[0]:
                raise ValueError("Offers categories or received counts inconsistent")
            seen.add(shirt)
            record = dict(zip(FIELDS, values))
            rows.append({"team_slot": side, "shirt_number": shirt, "player_name": name,
                         "page": page, **record})
            for field in FIELDS:
                totals[field] += record[field]
        if seen != set(rosters[side]):
            raise ValueError("Offers table does not cover the verified physical roster")
        summaries[side] = {"team_slot": side, "team": team, "players": len(seen),
                           "player_table_page": page, **totals}
    if set(summaries) != {0, 1} or set(overviews) != {0, 1}:
        raise ValueError("Offers require both player tables and both overviews")
    for side, summary in summaries.items():
        page, values = overviews[side]
        if values != (summary["total_offers"], summary["offers_received"]):
            raise ValueError("Offers player sums disagree with independent team overview")
        summary["overview_page"] = page
    return rows, [summaries[side] for side in (0, 1)]


def build(database="data/football.sqlite3", source_root=".tmp/open-data/fifa-reports",
          output="results/fifa-offers.json", private_dir=".tmp/fifa-offers"):
    source_root, output, private_dir = Path(source_root).resolve(), Path(output), Path(private_dir)
    with closing(sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("BEGIN")
        sources = connection.execute(
            "SELECT m.series,m.match_number,m.asset_id,m.payload,a.sha256,a.bytes,a.complete,d.provider "
            "FROM fifa_report_matches m JOIN assets a ON a.id=m.asset_id "
            "JOIN datasets d ON d.id=a.dataset_id ORDER BY m.series,m.match_number"
        ).fetchall()
    if not sources:
        raise ValueError("No archived FIFA reports")
    matches, players, inputs, manifests = [], [], {}, {}
    for series, number, asset, stored_payload, sha, size, complete, provider in sources:
        if series not in fifa_reports.SERIES or not complete or provider != "fifa_training_centre":
            raise ValueError("FIFA source edition/provider/archive state mismatch")
        if series not in manifests:
            manifest = source_root / series / "manifest.json"
            manifests[series] = json.loads(manifest.read_text())
        records = [r for r in manifests[series]["files"] if r.get("match_number") == number]
        if len(records) != 1:
            raise ValueError("Missing or duplicate report in source manifest")
        source = records[0]
        candidate = source_root / series / source["path"]
        path = candidate.resolve()
        stored = json.loads(stored_payload)
        if (candidate.is_symlink() or not path.is_relative_to(source_root / series)
                or path.suffix.lower() != ".pdf" or source["sha256"] != sha
                or stored.get("source_sha256") != sha or source["url"] != stored.get("source_url")
                or path.stat().st_size != size or file_digest(path) != sha):
            raise ValueError("FIFA PDF path/hash/source mismatch")
        layout = subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True,
                                check=True, text=True, timeout=45).stdout.split("\f")
        metadata, _ = fifa_reports.parse_report(layout, number, fifa_reports.SERIES[series]["edition"])
        if metadata["teams"] != stored["teams"] or metadata["goals"] != stored["goals"]:
            raise ValueError("FIFA source identity changed")
        rosters = {side: {} for side in (0, 1)}
        for row in fifa_reports.physical_rows(layout, metadata["teams"]):
            rosters[row["team_slot"]][row["shirt_number"]] = row["player_name"]
        try:
            rows, summaries = parse_offers(layout, metadata["teams"], rosters)
        except ValueError as exc:
            raise ValueError(f"{series} match {number}: {exc}") from exc
        match_id = f"fifa-{series}-{number}"
        matches.append({"id": match_id, "series": series, "match_number": number,
                        "date": metadata["date"], "stage": metadata["stage"], "goals": metadata["goals"],
                        "source_url": source["url"], "source_sha256": sha, "teams": summaries})
        players.extend({"match_id": match_id, **row} for row in rows)
        inputs[asset] = {"series": series, "match_number": number, "sha256": sha}
    signature = {"inputs": inputs, "code_sha256": {
        "offers": file_digest(__file__), "fifa_reports": file_digest(fifa_reports.__file__)}}
    run_id = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:20]
    coverage = [{"series": series, "reports": sum(m["series"] == series for m in matches),
                 "expected_reports": fifa_reports.SERIES[series]["matches"]} for series in sorted(manifests)]
    report = {"schema": SCHEMA, "run_id": run_id, "coverage": coverage, "matches": matches,
              "evidence": {"reports": len(matches), "team_pages": 2 * len(matches),
                           "player_rows": len(players), "overview_totals_reconciled": 2 * len(matches)},
              "scope": "FIFA official whole-match offers/receptions, including any extra time as reported. Counts are not per 90 minutes or event trajectories.",
              "limits": "Offers are observed opportunities, not player quality. Counts depend on playing time, possession, role and match state. Received/offered is descriptive, not a calibrated probability."}
    # Every source is checked before publishing or archiving any new derivative.
    write_json_atomic(private_dir / "report.json", report)
    write_json_atomic(private_dir / "player-rows.json", {"run_id": run_id, "rows": players})
    write_json_atomic(private_dir / "receipt.json", signature)
    with Warehouse(database) as db:
        archive_outputs(db, "fifa-offers", private_dir, "fifa_offers", run_id,
                        filenames=["report.json", "player-rows.json", "receipt.json"])
        db.record_analysis_inputs("fifa_offers", run_id, inputs)
        db.connection.commit()
    write_json_atomic(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default="data/football.sqlite3")
    parser.add_argument("--source-root", default=".tmp/open-data/fifa-reports")
    parser.add_argument("--output", default="results/fifa-offers.json")
    parser.add_argument("--private-dir", default=".tmp/fifa-offers")
    args = parser.parse_args()
    result = build(args.database, args.source_root, args.output, args.private_dir)
    print(json.dumps({"run_id": result["run_id"], **result["evidence"]}))
