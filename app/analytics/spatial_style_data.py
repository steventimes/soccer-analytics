"""Verified Wyscout observations and open-play spatial maps for style retrieval."""

from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import zipfile

import numpy as np
from scipy.ndimage import gaussian_filter

from .artifacts import file_digest, source_signature
from app.data_service.wyscout import json_items

KINDS = ("shots", "crosses", "take_ons", "passes")
CUT = "2018-01-01"
LEAGUES = ("England", "France", "Germany", "Italy", "Spain")


def normalize_heatmaps(counts, minutes, sigma=1):
    counts, minutes = np.asarray(counts), np.asarray(minutes)
    if (counts.ndim != 4 or minutes.shape != (len(counts),) or not np.isfinite(minutes).all()
            or np.any(minutes <= 0) or not np.isfinite(counts).all() or np.any(counts < 0)
            or not np.isfinite(sigma) or sigma < 0):
        raise ValueError("Invalid spatial counts or exposure")
    rates = counts * (90 / minutes[:, None, None, None])
    return gaussian_filter(rates, sigma=(0, 0, sigma, sigma), mode="reflect")


def build_heatmaps(events, observations, minimum_minutes=900, bins=50, sigma=1):
    if not np.isfinite(minimum_minutes) or minimum_minutes <= 0 or type(bins) is not int or bins < 2:
        raise ValueError("Invalid spatial cohort/grid")
    groups, match_players = {}, {}
    for row in observations:
        if row["position"] == "Goalkeeper" or row["league"] not in LEAGUES or not "2017-07-01" <= row["date"] < "2018-07-01":
            continue
        key = (str(row["match_id"]), str(row["entity_id"]))
        if key in match_players or type(row["minutes"]) not in (int, float) or not np.isfinite(row["minutes"]) or not 0 < row["minutes"] <= 110:
            raise ValueError("Duplicate or invalid player-match exposure")
        match_players[key] = row
        identity = (row["league"], str(row["entity_id"]))
        group = groups.setdefault(identity, {"minutes": [0., 0.], "teams": set(), "positions": set()})
        group["minutes"][int(row["date"] >= CUT)] += row["minutes"]
        group["teams"].add(str(row["team"])); group["positions"].add(row["position"])
    players = [{"league": league, "id": pid, "position": next(iter(g["positions"])), "team": next(iter(g["teams"]))}
               for (league, pid), g in sorted(groups.items())
               if min(g["minutes"]) >= minimum_minutes and len(g["teams"]) == len(g["positions"]) == 1]
    if not players:
        raise ValueError("No eligible spatial style cohort")
    identities = {(p["league"], p["id"]): i for i, p in enumerate(players)}
    minutes = np.array([groups[(p["league"], p["id"])]["minutes"] for p in players])
    maps = {block: {kind: np.zeros((len(players), 2 if kind == "passes" else 1, bins, bins), dtype=np.float32)
                    for kind in KINDS} for block in ("early", "late")}
    audit, seen = Counter(), set()

    def cell(point):
        xy = [point.get(axis) for axis in ("x", "y")] if isinstance(point, dict) else []
        if len(xy) != 2 or any(type(v) not in (int, float) or not np.isfinite(v) or not 0 <= v <= 100 for v in xy):
            raise ValueError("Invalid Wyscout coordinates")
        return min(bins - 1, int(xy[1] * bins / 100)), min(bins - 1, int(xy[0] * bins / 100))

    dates = {block: [] for block in maps}
    for event in events:
        audit["events_seen"] += 1
        if event.get("matchPeriod") not in ("1H", "2H"):
            continue
        row = match_players.get((str(event.get("matchId")), str(event.get("playerId"))))
        if row is None or (row["league"], str(row["entity_id"])) not in identities:
            continue
        if str(event.get("teamId")) != str(row["team"]):
            audit["team_mismatch"] += 1; continue
        event_key = (row["league"], event.get("id"))
        if type(event.get("id")) is not int:
            audit["invalid_event_identity"] += 1; continue
        if event_key in seen:
            audit["duplicate_events"] += 1; continue
        seen.add(event_key)
        name, sub = event.get("eventName"), event.get("subEventName")
        kind = ("crosses" if sub == "Cross" else "passes") if name == "Pass" else "shots" if name == "Shot" else "take_ons" if name == "Duel" and sub == "Ground attacking duel" else None
        if kind is None:
            continue
        points = event.get("positions") or []
        try:
            cells = [cell(points[0])]
            if kind == "passes":
                cells.append(cell(points[1]))
        except (ValueError, IndexError):
            audit["invalid_coordinates"] += 1; continue
        block = "early" if row["date"] < CUT else "late"
        index = identities[(row["league"], str(row["entity_id"]))]
        for channel, (y, x) in enumerate(cells):
            maps[block][kind][index, channel, y, x] += 1
        dates[block].append(row["date"])
        audit["retained_" + kind] += 1
        audit[block + "_" + kind] += 1
    if not all(dates.values()):
        raise ValueError("Both style periods require retained events")
    for block, column in (("early", 0), ("late", 1)):
        for kind in KINDS:
            maps[block][kind] = normalize_heatmaps(maps[block][kind], minutes[:, column], sigma)
    return {"players": players, "minutes": minutes, "maps": maps, "audit": dict(audit),
            "dates": {block: {"start": min(d), "end": max(d)} for block, d in dates.items()},
            "grid": bins, "sigma": sigma, "minimum_minutes": minimum_minutes}


def load_data(database, source_dir, cache_path):
    source_dir, cache_path = Path(source_dir), Path(cache_path)
    receipt = json.loads(cache_path.with_suffix(".receipt.json").read_text())
    if file_digest(cache_path) != receipt["content_sha256"] or file_digest(Path(__file__).with_name("assessment_data.py")) != receipt["inputs"]["extractor_sha256"]:
        raise ValueError("Player exposure cache failed receipt verification; rebuild assessments first")
    with sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True) as db:
        if source_signature(db, ("wyscout",)) != receipt["inputs"]["sources"]:
            raise ValueError("Player exposure cache source signature is stale")
    manifest = json.loads((source_dir / "manifest.json").read_text())
    descriptions = [f for article in manifest for f in article["files"] if f["name"] == "events.zip"]
    event_path = source_dir / "events.zip"
    if len(descriptions) != 1 or event_path.stat().st_size != descriptions[0]["size"]:
        raise ValueError("Wyscout event archive does not match inventory")
    md5 = hashlib.md5(usedforsecurity=False)
    with event_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            md5.update(block)
    if md5.hexdigest() != descriptions[0]["computed_md5"]:
        raise ValueError("Wyscout event archive failed publisher checksum")
    cached = json.loads(cache_path.read_text())

    def events():
        with zipfile.ZipFile(event_path) as archive:
            for league in LEAGUES:
                print(f"Reading verified Wyscout {league} events", flush=True)
                with archive.open(f"events_{league}.json") as raw:
                    yield from json_items(io.TextIOWrapper(raw, encoding="utf-8"))

    data = build_heatmaps(events(), cached["observations"])
    data["source"] = {"provider": "Wyscout public research data", "source_url": "https://doi.org/10.6084/m9.figshare.c.4415000",
                      "event_zip_sha256": file_digest(event_path), "exposure_cache_sha256": receipt["content_sha256"],
                      "inventory_sha256": file_digest(source_dir / "manifest.json"), "warehouse_signature": receipt["inputs"]["sources"]["sha256"]}
    return data
