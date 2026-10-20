"""Refresh dated club/national observations without pooling source identities."""

from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile
from threading import Lock
import requests

from .archive import _request, verify_asset_file
from .build_warehouse import document
from .fetch.fetcher import SlidingWindowRateLimiter
from .fetch.public import LEAGUES, FootballCSVClient, _identity
from .warehouse import Warehouse

COMPETITIONS = {
    **{
        k: ("club", "male")
        for k in [
            "eng.1",
            "eng.2",
            "ger.1",
            "esp.1",
            "ita.1",
            "fra.1",
            "por.1",
            "ned.1",
            "uefa.champions",
        ]
    },
    **{
        k: ("national", "male")
        for k in [
            "uefa.nations",
            "fifa.friendly",
            "concacaf.nations.league",
            "caf.nations_qual",
            "fifa.worldq.afc",
            "fifa.worldq.conmebol",
            "fifa.worldq.caf",
            "uefa.euro_u21_qual",
        ]
    },
    "fifa.friendly.w": ("national", "female"),
}


def calendar_days(calendar, start, end):
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if first > last:
        raise ValueError("Invalid date range")
    if calendar and all(isinstance(item, str) for item in calendar):
        return sorted(
            {
                item[:10].replace("-", "")
                for item in calendar
                if start <= item[:10] <= end
            }
        )
    # International competitions may provide phase blocks rather than match days.
    return [
        (first + timedelta(days=i)).strftime("%Y%m%d")
        for i in range((last - first).days + 1)
    ]


def normalize_event(event, competition, team_type, gender):
    match_id = str(event["id"])
    if re.fullmatch(r"[0-9]+", match_id) is None:
        raise ValueError("Invalid ESPN event identifier")
    match = event["competitions"][0]
    teams = {t["homeAway"]: t for t in match["competitors"]}
    if set(teams) != {"home", "away"} or len(match["competitors"]) != 2:
        raise ValueError("Ambiguous competitor roles")
    status = event.get("status", match.get("status", {})).get("type", {})
    finished = status.get("state") == "post" and status.get("completed") is True
    scores = {side: int(t["score"]) if finished else None for side, t in teams.items()}
    if any(value is not None and value < 0 for value in scores.values()):
        raise ValueError("Invalid final score")
    return {
        "provider": "espn",
        "competition": competition,
        "match_id": match_id,
        "date": event["date"][:10],
        "kickoff_utc": event["date"],
        "status": (
            "FINISHED"
            if finished
            else status.get("name", status.get("state", "unknown"))
        ),
        "home_team": teams["home"]["team"]["displayName"],
        "away_team": teams["away"]["team"]["displayName"],
        "home_team_id": str(teams["home"]["team"]["id"]),
        "away_team_id": str(teams["away"]["team"]["id"]),
        "home_goals": scores["home"],
        "away_goals": scores["away"],
        "neutral": match.get("neutralSite"),
        "team_type": team_type,
        "gender": gender,
        "age_group": "u21" if competition == "uefa.euro_u21_qual" else "senior",
        "score_duration": status.get("name", "provider-final-including-extra-time"),
        "source_season": event.get("season"),
        "source_url": f"https://www.espn.com/soccer/match/_/gameId/{event['id']}",
    }


def historical_result(row):
    if row["neutral"].lower() not in {"true", "false"}:
        raise ValueError("Unknown neutral-ground flag")
    scores = [int(row[k]) for k in ["home_score", "away_score"]]
    if min(scores) < 0:
        raise ValueError("Negative final score")
    mid = _identity(
        "international_results", row["date"], row["home_team"], row["away_team"]
    )
    return {
        "provider": "international_results",
        "competition": row["tournament"],
        "match_id": str(mid),
        "date": row["date"],
        "kickoff_utc": None,
        "status": "FINISHED",
        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "home_team_id": str(_identity("international_results", row["home_team"])),
        "away_team_id": str(_identity("international_results", row["away_team"])),
        "home_goals": scores[0],
        "away_goals": scores[1],
        "neutral": row["neutral"].lower() == "true",
        "team_type": "national",
        "gender": "male",
        "age_group": "senior",
        "score_duration": "includes-extra-time; shootout-winner-separate",
        "city": row["city"],
        "venue_country": row["country"],
        "source_url": "https://github.com/martj42/international_results",
    }


def _observation_time(value):
    observed = datetime.fromisoformat(value)
    # Earlier manifests sometimes recorded a UTC date without an offset.
    if observed.tzinfo is None:
        return observed.replace(tzinfo=timezone.utc)
    return observed.astimezone(timezone.utc)


def _validate_summary(data, match):
    header = data.get("header", {})
    if str(header.get("id")) != match["match_id"]:
        raise ValueError("Summary event identity mismatch")
    competitions = header.get("competitions", [])
    if len(competitions) != 1:
        raise ValueError("Summary competition is missing or ambiguous")
    competition = competitions[0]
    competitors = competition.get("competitors", [])
    teams = {team.get("homeAway"): team for team in competitors}
    if len(competitors) != 2 or set(teams) != {"home", "away"}:
        raise ValueError("Summary competitor roles are ambiguous")
    status = competition.get("status", {}).get("type", {})
    if (match["status"] != "FINISHED" or status.get("state") != "post"
            or status.get("completed") is not True):
        raise ValueError("Summary final status mismatch")
    for side, team in teams.items():
        if str(team.get("id")) != match[f"{side}_team_id"]:
            raise ValueError("Summary team identity mismatch")
        if int(team["score"]) != match[f"{side}_goals"]:
            raise ValueError("Summary final score mismatch")


def _result_signature(match):
    keys = ("date", "kickoff_utc", "status", "home_team_id", "away_team_id",
            "home_goals", "away_goals")
    return tuple(match.get(key) for key in keys)


def collect(root, start, end, competitions=COMPETITIONS, *, resume=False, csv_competitions=None):
    if date.fromisoformat(start) > date.fromisoformat(end):
        raise ValueError("Invalid date range")
    csv_codes = list(LEAGUES) if csv_competitions is None else list(csv_competitions)
    if any(code not in LEAGUES for code in csv_codes):
        raise ValueError("Unknown Football-Data competition")
    csv_leagues = {code: LEAGUES[code] for code in csv_codes}
    root = Path(root).resolve()
    request = {
        "start": start,
        "end": end,
        "competitions": {key: list(value) for key, value in competitions.items()},
        "football_csv": {code: value[0] for code, value in csv_leagues.items()},
    }
    request_path = root / "request.json"
    if root.exists() and any(root.iterdir()):
        if not resume:
            raise ValueError("Collection root is a saved snapshot; choose a new root or explicitly resume")
        if request_path.is_symlink() or not request_path.is_file():
            raise ValueError("Cannot resume without a collection request; choose a new root")
        capture = json.loads(request_path.read_text())
        if capture["request"] != request:
            raise ValueError("Cannot resume a different collection request; choose a new root")
    else:
        capture = {"request": request, "started_at": datetime.now(timezone.utc).isoformat()}
        root.mkdir(parents=True, exist_ok=True)
        request_path.write_text(json.dumps(capture, indent=2))
    limiter = SlidingWindowRateLimiter(limit=60)
    request_lock = Lock()
    counts = {"attempts": 0, "downloaded": 0, "cached": 0}
    files, errors, matches, summaries = [], [], {}, []

    def reserve_request():
        with request_lock:
            limiter.wait_if_needed()
            limiter.add_request()
            counts["attempts"] += 1

    def get(url, relative):
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Refresh cache path escapes collection root")
        receipt = path.with_suffix(path.suffix + ".receipt.json")
        if receipt.is_symlink():
            raise ValueError("Refresh cache receipt cannot be an alias")
        if path.exists() != receipt.exists():
            raise ValueError("Cached response is missing its payload or receipt; choose a new root")
        if path.exists() and receipt.exists():
            record = json.loads(receipt.read_text())
            content = path.read_bytes()
            if record["url"] != url or record["path"] != relative:
                raise ValueError("Cached response request identity mismatch")
            if len(content) != record["bytes"]:
                raise ValueError("Cached response size mismatch")
            if hashlib.sha256(content).hexdigest() != record["sha256"]:
                raise ValueError("Cached response checksum mismatch")
            _observation_time(record["observed_at"])
            with request_lock:
                counts["cached"] += 1
            return path, record
        with requests.Session() as session:
            session.trust_env = False
            response = _request(session, url, before_request=reserve_request)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
        record = {
            "path": relative,
            "url": url,
            "sha256": hashlib.sha256(response.content).hexdigest(),
            "bytes": len(response.content),
            "observed_at": datetime.now(timezone.utc).isoformat(),
        }
        receipt.write_text(json.dumps(record, indent=2))
        with request_lock:
            counts["downloaded"] += 1
        return path, record

    jobs = []
    for league, (team_type, gender) in competitions.items():
        if team_type == "club":
            # Supported European club seasons use their July starting year.
            first = int(start[:4]) - int(int(start[5:7]) < 7)
            last = int(end[:4]) - int(int(end[5:7]) < 7)
            windows = [(year, max(start, f"{year}-07-01"), min(end, f"{year+1}-06-30"))
                       for year in range(first, last + 1)]
        else:
            # National competitions retain their existing calendar/phase policy.
            windows = [(int(end[:4]), start, end)]
        days = set()
        for year, window_start, window_end in windows:
            url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard?dates={year}0701"
            suffix = f"_{year}" if len(windows) > 1 else ""
            try:
                path, record = get(url, f"espn/calendar_{league}{suffix}.json")
                files.append(record)
                data = json.loads(path.read_text())
                calendar = data["leagues"][0].get("calendar", [])
                days.update(calendar_days(calendar, window_start, window_end))
            except Exception as exc:
                errors.append(
                    {"stage": "calendar", "competition": league, "season_start_year": year,
                     "error": str(exc)}
                )
        jobs.extend((league, day, team_type, gender) for day in sorted(days))

    def scoreboard(job):
        league, day, kind, gender = job
        path, record = get(
            f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard?dates={day}&limit=100",
            f"espn/scoreboard_{league}_{day}.json",
        )
        events = json.loads(path.read_text())["events"]
        if len(events) >= 100:
            raise ValueError(
                "Possible truncated scoreboard; refuse silently limited results"
            )
        return record, [
            normalize_event(e, league, kind, gender)
            for e in events
            if start <= e["date"][:10] <= end
        ]

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [(job, pool.submit(scoreboard, job)) for job in jobs]
        for job, future in futures:
            try:
                record, events = future.result()
                files.append(record)
                for event in events:
                    key = (event["competition"], event["match_id"])
                    if (
                        key in matches
                        and {k: v for k, v in matches[key].items() if k != "raw_path"}
                        != event
                    ):
                        raise ValueError("Conflicting repeated fixture")
                    matches[key] = {**event, "raw_path": record["path"]}
            except Exception as exc:
                errors.append(
                    {
                        "stage": "scoreboard",
                        "competition": job[0],
                        "day": job[1],
                        "error": str(exc),
                    }
                )

    def summary(match):
        league, mid = match["competition"], match["match_id"]
        path, record = get(
            f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/summary?event={mid}",
            f"espn/summary_{league}_{mid}.json",
        )
        data = json.loads(path.read_text())
        _validate_summary(data, match)
        return record

    completed = [m for m in matches.values() if m["status"] == "FINISHED"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [(m, pool.submit(summary, m)) for m in completed]
        for match, future in futures:
            try:
                record = future.result()
                files.append(record)
                summaries.append(
                    {
                        "competition": match["competition"],
                        "match_id": match["match_id"],
                        "path": record["path"],
                    }
                )
            except Exception as exc:
                errors.append(
                    {
                        "stage": "summary",
                        "competition": match["competition"],
                        "match_id": match["match_id"],
                        "error": str(exc),
                    }
                )
    csv_records = []
    client = FootballCSVClient()
    for code, (division, _, _) in csv_leagues.items():
        year = int(end[:4]) - int(int(end[5:7]) < 7)
        season = f"{year%100:02}{(year+1)%100:02}"
        try:
            url = f"https://www.football-data.co.uk/mmz4281/{season}/{division}.csv"
            path, record = get(url, f"football-csv/{code}_{year}.csv")
            files.append(record)
            parsed = client.parse_csv_text(
                code, path.read_text(encoding="utf-8-sig"), url
            )
            csv_records.append(
                {
                    "competition": code,
                    "year": year,
                    "path": record["path"],
                    "matches": parsed,
                }
            )
        except Exception as exc:
            errors.append(
                {"stage": "football_csv", "competition": code, "error": str(exc)}
            )
    observed = [_observation_time(file["observed_at"]) for file in files]
    report = {
        "request": request,
        "start": start,
        "end": end,
        "observed_at": max(observed).isoformat() if observed else capture["started_at"],
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "source_observation_window": {
            "earliest": min(observed).isoformat() if observed else None,
            "latest": max(observed).isoformat() if observed else None,
        },
        "requests": counts,
        "mode": "resume" if resume else "fresh",
        "files": files,
        "matches": list(matches.values()),
        "summaries": summaries,
        "football_csv": csv_records,
        "errors": errors,
    }
    (root / "collection.json").write_text(json.dumps(report, indent=2))
    return report


def ingest(database, root, international_root):
    root = Path(root)
    report = json.loads((root / "collection.json").read_text())
    added = Counter()
    updated = Counter()
    skipped = Counter()
    excluded = Counter()
    verified = []
    for file in report["files"]:
        path = verify_asset_file(root, file["path"], {file["path"]: file})
        observed_at = _observation_time(file.get("observed_at", report["observed_at"]))
        verified.append((file, path, observed_at.isoformat()))
    verified_paths = {file["path"]: path for file, path, _ in verified}
    match_index = {(match["competition"], match["match_id"]): match for match in report["matches"]}
    for summary in report["summaries"]:
        match = match_index[(summary["competition"], summary["match_id"])]
        _validate_summary(json.loads(verified_paths[summary["path"]].read_text()), match)
    with Warehouse(database) as db:
        db.connection.executescript(
            """CREATE TABLE IF NOT EXISTS latest_match_observations(provider TEXT,competition TEXT,match_id TEXT,match_date TEXT,status TEXT,team_type TEXT,gender TEXT,asset_id INTEGER REFERENCES assets(id),observed_at TEXT,payload TEXT,PRIMARY KEY(provider,competition,match_id));
            CREATE INDEX IF NOT EXISTS latest_match_dates ON latest_match_observations(match_date,team_type);
            CREATE TABLE IF NOT EXISTS fixture_identity_conflicts(provider TEXT,competition TEXT,match_id TEXT,asset_id INTEGER REFERENCES assets(id),source_rows TEXT,reason TEXT,PRIMARY KEY(provider,competition,match_id,asset_id));
            CREATE TABLE IF NOT EXISTS recent_match_summaries(provider TEXT,competition TEXT,match_id TEXT,asset_id INTEGER REFERENCES assets(id),PRIMARY KEY(provider,competition,match_id));
            CREATE TABLE IF NOT EXISTS recent_source_observation_times(asset_id INTEGER PRIMARY KEY REFERENCES assets(id),observed_at TEXT NOT NULL);"""
        )
        ds = db.dataset("recent_match_sources", "football", "public_recent")
        asset_ids = {}
        asset_observations = {}

        def observe_asset(aid, observed_at):
            observed_at = _observation_time(observed_at).isoformat()
            asset_observations[aid] = observed_at
            prior = db.connection.execute(
                "SELECT observed_at FROM recent_source_observation_times WHERE asset_id=?",
                (aid,),
            ).fetchone()
            if prior is None or _observation_time(prior[0]) < _observation_time(observed_at):
                db.connection.execute(
                    "INSERT OR REPLACE INTO recent_source_observation_times VALUES(?,?)",
                    (aid, observed_at),
                )

        for file, path, observed_at in verified:
            aid = db.asset(ds, path, file["path"])
            asset_ids[file["path"]] = aid
            observe_asset(aid, observed_at)
            if path.suffix == ".json":
                document(db, aid, path, "recent_provider_json")
            else:
                db.csv_table(aid, path, "football_results_raw")

        revised = set()

        def store(match, aid):
            key = (match["provider"], match["competition"], match["match_id"])
            observed_at = asset_observations.get(aid, report["observed_at"])
            prior = db.connection.execute(
                "SELECT payload,observed_at FROM latest_match_observations WHERE provider=? AND competition=? AND match_id=?",
                key,
            ).fetchone()
            if prior and _observation_time(prior[1]) > _observation_time(observed_at):
                skipped[match["provider"]] += 1
                return False
            conflicts = db.connection.execute(
                "SELECT t.observed_at FROM fixture_identity_conflicts c "
                "LEFT JOIN recent_source_observation_times t ON t.asset_id=c.asset_id "
                "WHERE c.provider=? AND c.competition=? AND c.match_id=?", key,
            )
            if any(row[0] is None or _observation_time(row[0]) >= _observation_time(observed_at)
                   for row in conflicts):
                excluded[match["provider"]] += 1
                return False
            if (prior and match["provider"] == "espn"
                    and _result_signature(json.loads(prior[0])) != _result_signature(match)):
                revised.add((match["competition"], match["match_id"]))
            payload = json.dumps(match, sort_keys=True)
            if prior is None:
                added[match["provider"]] += 1
            elif prior[0] != payload:
                updated[match["provider"]] += 1
            db.connection.execute(
                "INSERT INTO latest_match_observations VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(provider,competition,match_id) DO UPDATE SET match_date=excluded.match_date,status=excluded.status,asset_id=excluded.asset_id,observed_at=excluded.observed_at,payload=excluded.payload",
                (
                    *key,
                    match["date"],
                    match["status"],
                    match["team_type"],
                    match["gender"],
                    aid,
                    observed_at,
                    payload,
                ),
            )
            return True

        accepted = set()
        for match in report["matches"]:
            if store(match, asset_ids[match["raw_path"]]):
                accepted.add((match["competition"], match["match_id"]))
        refreshed_summaries = set()
        for summary in report["summaries"]:
            key = (summary["competition"], summary["match_id"])
            if key not in accepted:
                continue
            aid = asset_ids[summary["path"]]
            prior = db.connection.execute(
                "SELECT t.observed_at FROM recent_match_summaries s "
                "LEFT JOIN recent_source_observation_times t ON t.asset_id=s.asset_id "
                "WHERE s.provider='espn' AND s.competition=? AND s.match_id=?", key,
            ).fetchone()
            if (prior and prior[0] is not None
                    and _observation_time(prior[0]) > _observation_time(asset_observations[aid])):
                continue
            db.connection.execute(
                "INSERT OR REPLACE INTO recent_match_summaries VALUES(?,?,?,?)",
                (
                    "espn",
                    summary["competition"],
                    summary["match_id"],
                    aid,
                ),
            )
            refreshed_summaries.add(key)
        for key in revised - refreshed_summaries:
            db.connection.execute(
                "DELETE FROM recent_match_summaries WHERE provider='espn' AND competition=? AND match_id=?", key,
            )
        football_new_to_previous = 0
        for group in report["football_csv"]:
            known = set()
            for r in db.connection.execute(
                "SELECT d.payload FROM documents d JOIN assets a ON a.id=d.asset_id WHERE a.path LIKE ?",
                (f"history/football-csv/{group['competition']}/{group['year']}.json%",),
            ):
                known.update(str(m["id"]) for m in json.loads(r[0]).get("matches", []))
            for m in group["matches"]:
                if m["status"] != "FINISHED":
                    continue
                football_new_to_previous += str(m["id"]) not in known
                match = {
                    "provider": "football-data.co.uk",
                    "competition": group["competition"],
                    "match_id": str(m["id"]),
                    "date": m["source_date"],
                    "kickoff_utc": m["utcDate"],
                    "status": m["status"],
                    "home_team": m["homeTeam"]["name"],
                    "away_team": m["awayTeam"]["name"],
                    "home_team_id": str(m["homeTeam"]["id"]),
                    "away_team_id": str(m["awayTeam"]["id"]),
                    "home_goals": m["score"]["fullTime"]["home"],
                    "away_goals": m["score"]["fullTime"]["away"],
                    "neutral": False,
                    "team_type": "club",
                    "gender": "male",
                    "age_group": "senior",
                    "score_duration": "regular-time",
                    "source_url": m["source_url"],
                }
                store(match, asset_ids[group["path"]])
        idir = Path(international_root)
        manifest = json.loads((idir / "manifest.json").read_text())
        index = {r["path"]: r for r in manifest["files"]}
        ds = db.dataset(
            "international-results",
            "football",
            "international_results",
            manifest["revision"],
        )
        for relative in index:
            path = verify_asset_file(idir, relative, index)
            aid = db.asset(ds, path, relative)
            observe_asset(aid, manifest.get("fetched_at", report["observed_at"]))
            if path.suffix == ".csv":
                db.csv_table(aid, path, "international_results_csv_" + path.stem)
            if relative == "results.csv":
                with path.open(newline="", encoding="utf-8-sig") as f:
                    grouped = {}
                    for row_number, row in enumerate(csv.DictReader(f), 1):
                        if row["date"] > report["end"]:
                            continue
                        match = historical_result(row)
                        key = (
                            match["provider"],
                            match["competition"],
                            match["match_id"],
                        )
                        grouped.setdefault(key, []).append((row_number, match))
                    for key, group in grouped.items():
                        if any(m != group[0][1] for _, m in group):
                            prior = db.connection.execute(
                                "SELECT observed_at FROM latest_match_observations "
                                "WHERE provider=? AND competition=? AND match_id=?", key,
                            ).fetchone()
                            if prior and _observation_time(prior[0]) > _observation_time(asset_observations[aid]):
                                skipped["international_results"] += 1
                                continue
                            db.connection.execute(
                                "DELETE FROM latest_match_observations WHERE provider=? AND competition=? AND match_id=?",
                                key,
                            )
                            db.connection.execute(
                                "INSERT OR REPLACE INTO fixture_identity_conflicts VALUES(?,?,?,?,?,?)",
                                (
                                    *key,
                                    aid,
                                    json.dumps([n for n, _ in group]),
                                    "Different rows share date/teams/tournament; preserve originals and exclude ambiguous canonical fixture pending review",
                                ),
                            )
                        else:
                            store(group[0][1], aid)
        aid = db.asset(ds, idir / "manifest.json", "manifest.json")
        document(db, aid, idir / "manifest.json", "international_archive_manifest")
        aid = db.asset(
            db.dataset("recent_match_sources", "football", "public_recent"),
            root / "collection.json",
            "collection.json",
        )
        document(db, aid, root / "collection.json", "recent_collection_receipt")
        db.connection.commit()
        db.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        counts = [
            dict(row)
            for row in db.connection.execute(
                "SELECT provider,team_type,gender,COUNT(*) rows,MAX(match_date) maximum_date FROM latest_match_observations GROUP BY provider,team_type,gender"
            )
        ]
        conflicts = db.connection.execute(
            "SELECT COUNT(*) FROM fixture_identity_conflicts"
        ).fetchone()[0]
    return {
        "added_to_latest_view": dict(added),
        "updated_latest_records": dict(updated),
        "skipped_older_observations": dict(skipped),
        "excluded_by_conflict": dict(excluded),
        "football_csv_new_to_previous_archive": football_new_to_previous,
        "coverage": counts,
        "identity_conflicts": conflicts,
        "errors": report["errors"],
        "note": "Different providers may overlap. Latest-view additions are not globally unique new matches; current-day scheduled/live events are retained but not treated as final results.",
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--database", default="data/football.sqlite3")
    p.add_argument("--root", help="Snapshot directory; defaults to a new unique directory for each collection")
    p.add_argument("--resume", action="store_true", help="Resume the same request in --root using its saved responses; does not refresh cached responses")
    p.add_argument(
        "--international-root", default=".tmp/open-data/international-results"
    )
    p.add_argument("--start")
    p.add_argument("--end", default=datetime.now(timezone.utc).date().isoformat())
    p.add_argument("--competitions", help="Comma-separated ESPN codes, or none; defaults to all registered competitions")
    p.add_argument("--csv-competitions", help="Comma-separated Football-Data codes, or none; defaults to all registered leagues")
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--collect-only", action="store_true")
    modes.add_argument("--ingest-only", action="store_true")
    a = p.parse_args(argv)
    def selection(value, available, option):
        if value is None:
            return list(available)
        if value == "none":
            return []
        codes = list(dict.fromkeys(code.strip() for code in value.split(",")))
        if any(code not in available for code in codes):
            p.error(f"{option} contains an unknown or empty competition code")
        return codes
    espn_codes = selection(a.competitions, COMPETITIONS, "--competitions")
    csv_codes = selection(a.csv_competitions, LEAGUES, "--csv-competitions")
    if not a.root and (a.resume or a.ingest_only):
        p.error("--root is required for --resume or --ingest-only")
    a.start = a.start or (date.fromisoformat(a.end) - timedelta(days=21)).isoformat()
    if not a.root:
        Path(".tmp").mkdir(exist_ok=True)
        a.root = tempfile.mkdtemp(prefix=f"refresh-{a.end}-", dir=".tmp")
    print(f"Collection root: {a.root}")
    if not a.ingest_only:
        collect(a.root, a.start, a.end,
                {code: COMPETITIONS[code] for code in espn_codes},
                resume=a.resume, csv_competitions=csv_codes)
    if not a.collect_only:
        result = ingest(a.database, a.root, a.international_root)
        Path(a.root, "ingest.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
