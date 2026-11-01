"""Bounded anonymous SoccerData reads, retaining raw caches and parsed frames.

Install SoccerData in an optional environment. Each provider is independent;
blocked requests, human verification and empty frames are recorded explicitly.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import inspect
import hashlib
import html
import json
import os
from pathlib import Path
import re
import signal
import time

from app.file_io import write_bytes_atomic, write_json_atomic

RESPONSE_CAPTURE_SCHEMA = "SoccerAnalytics.SourceResponseCapture.v1"

PROVIDERS = (
    "MatchHistory",
    "ClubElo",
    "ESPN",
    "Sofascore",
    "Understat",
    "FBref",
    "SoFIFA",
    "WhoScored",
)

UNDERSTAT_SITUATIONS = {
    "OpenPlay": "Open Play",
    "FromCorner": "From Corner",
    "SetPiece": "Set Piece",
    "DirectFreekick": "Direct Freekick",
    "Penalty": "Penalty",
}


class SourceStopped(BaseException):
    """Stop upstream retry/captcha loops while preserving the access evidence."""


class RequestCapture:
    """Retain each adapter call and its content without recording headers or cookies."""

    def __init__(self, report, output, request_budget):
        self.report = report
        self.output = Path(output).resolve()
        self.request_budget = request_budget
        report["response_capture"] = {
            "schema": RESPONSE_CAPTURE_SCHEMA,
            "body_directory": "cache/_responses",
            "request_unit": "SDK GET or browser navigation; excludes redirects and browser subresources",
            "content_scope": "SDK response.content, browser DOM and extracted payloads; not wire bytes",
            "duration_scope": "adapter call and response archiving",
            "cache_hits": "not new network observations",
        }

    @contextmanager
    def request(self, url, transport):
        requests = self.report["requests"]
        if len(requests) >= self.request_budget:
            raise SourceStopped("request_budget_reached")
        entry = {"ordinal": len(requests) + 1, "url": url, "transport": transport,
                 "started_at": datetime.now(timezone.utc).isoformat()}
        requests.append(entry)
        started = time.monotonic()
        try:
            yield entry
        except BaseException as exc:
            entry.setdefault("error", type(exc).__name__)
            raise
        finally:
            entry["completed_at"] = datetime.now(timezone.utc).isoformat()
            entry["elapsed_seconds"] = time.monotonic() - started

    def record(self, entry, payload, kind, field="response"):
        try:
            if not isinstance(payload, bytes):
                raise TypeError("Captured response content must be bytes")
            if field == "response":
                entry["response_observed_at"] = datetime.now(timezone.utc).isoformat()
            digest = hashlib.sha256(payload).hexdigest()
            relative = Path("cache") / "_responses" / (digest + ".bin")
            path = self.output / relative
            if path.resolve() != path:
                raise ValueError("Response capture cannot use a path alias")
            if path.exists():
                if path.stat().st_nlink != 1 or path.read_bytes() != payload:
                    raise ValueError("Existing response capture differs from its content identity")
            else:
                write_bytes_atomic(path, payload)
            entry[field] = {"path": relative.as_posix(), "sha256": digest,
                            "bytes": len(payload), "kind": kind}
        except Exception as exc:
            entry["capture_error"] = type(exc).__name__
            raise SourceStopped("response_capture_error:" + str(exc)) from exc

    def wrap_get(self, original):
        def tracked_get(url, *args, **kwargs):
            with self.request(url, "soccerdata_native_tls_client") as entry:
                try:
                    response = original(url, *args, **kwargs)
                except Exception as exc:
                    entry["error"] = type(exc).__name__
                    raise SourceStopped("transport_error:" + str(exc)) from exc
                try:
                    entry["http_status"] = response.status_code
                    payload = response.content
                except Exception as exc:
                    entry["capture_error"] = type(exc).__name__
                    raise SourceStopped("response_capture_error:" + str(exc)) from exc
                self.record(entry, payload, "sdk_response_content")
                entry["bytes"] = len(payload)
                if type(response.status_code) is not int or response.status_code < 100:
                    raise SourceStopped("non_http_transport_response")
                if response.status_code >= 400:
                    raise SourceStopped("http_" + str(response.status_code))
                return response
        return tracked_get

    def browser_download(self, reader, url, filepath=None, var=None):
        with self.request(url, "normal_isolated_chrome") as entry:
            reader._driver.get(url)
            time.sleep(1)
            text = reader._driver.page_source
            self.record(entry, text.encode("utf-8"), "browser_page_source_utf8")
            if any(s in text for s in (
                "Verify you are human", "Checking your browser", "Just a moment...", "cf-challenge",
            )):
                entry["status"] = "human_verification"
                raise SourceStopped("browser_human_verification_required")
            if '"reason":"Forbidden"' in text or '"reason": "Forbidden"' in text:
                entry["status"] = "forbidden"
                raise SourceStopped("browser_forbidden")
            try:
                payload = (reader._validate_page(url).encode() if var is None
                           else json.dumps(reader._driver.execute_script("return " + var)).encode())
            except Exception as exc:
                entry["status"] = "invalid_page"
                raise SourceStopped("invalid_browser_page:" + str(exc)) from exc
            self.record(entry, payload, "browser_extracted_payload_utf8", "parser_payload")
            entry.update(status="page_read", bytes=len(payload))
            if filepath is not None:
                write_bytes_atomic(Path(filepath), payload)
            import io

            return io.BytesIO(payload)


def tracked_requests_reader(base, capture):
    def init_session(self, *args, **kwargs):
        session = base._init_session(self, *args, **kwargs)
        session.get = capture.wrap_get(session.get)
        return session

    # Preserve SoccerData's class-name key for league and season configuration.
    return type(base.__name__, (base,), {"_init_session": init_session})


def export_frame(frame, path):
    if len(frame) == 0:
        raise ValueError("Provider returned an empty dataframe")
    frame = frame.copy()
    schema = {
        "index_names": list(frame.index.names),
        "columns": [list(c) if isinstance(c, tuple) else str(c) for c in frame.columns],
        "dtypes": [str(dtype) for dtype in frame.dtypes],
    }
    # Explicit ordinal labels preserve duplicate/multi-level column identities.
    frame.columns = [
        f"field_{i:04}_" + ("/".join(map(str, c)) if isinstance(c, tuple) else str(c))
        for i, c in enumerate(frame.columns)
    ]
    frame.index.names = [
        f"index_{i:02}_" + str(name or "unnamed")
        for i, name in enumerate(frame.index.names)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=True)
    path.with_suffix(".schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2)
    )
    return {
        "path": str(path),
        "rows": len(frame),
        "columns": len(frame.columns),
        "schema": schema,
    }


def explicit_seasons(years):
    result = []
    for year in years:
        year = int(year)
        if not 1900 <= year <= 2100:
            raise ValueError("Season start year outside1900-2100")
        result.append(f"{year}-{year+1}")
    return result


def restore_situations(shots, raw):
    result, recovered = [], 0
    for shot in shots:
        expected = UNDERSTAT_SITUATIONS.get(raw.get(shot["shot_id"]))
        if expected is None or (shot["situation"] and shot["situation"] != expected):
            raise ValueError("Shot situation cannot be reconciled by source shot ID")
        recovered += not bool(shot["situation"])
        result.append({**shot, "situation": expected})
    return result, recovered


def normalize_understat_shots(frame, cache):
    """Reconcile SDK fields with exact raw identities, keeping native row order/types."""
    required = {"game_id", "shot_id", "team_id", "player_id", "season_id",
                "situation", "assist_player", "assist_player_id"}
    if not frame.columns.is_unique or not required <= set(frame.columns):
        raise ValueError("Understat shot identity/field columns are missing or ambiguous")
    assist_dtype = frame["assist_player_id"].dtype
    if str(assist_dtype) != "Int64" and assist_dtype.kind not in ("O", "U", "S"):
        raise ValueError("Understat assist IDs require nullable Int64 or string storage")

    def identity(value):
        token = str(value)
        if re.fullmatch(r"[1-9][0-9]*", token) is None:
            raise ValueError("Understat shot identities must be positive integer IDs")
        return token

    game_ids = sorted({identity(value) for value in frame["game_id"]})
    raw, teams, names, roster_ids, legacy_ids, fingerprints = {}, {}, {}, {}, {}, []
    for game in game_ids:
        path = Path(cache) / f"match_{game}.json"
        try:
            content = path.read_bytes()
            payload = json.loads(content)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Understat original match {game} is unavailable") from exc
        if set(payload.get("rosters", {})) != {"h", "a"} or set(payload.get("shots", {})) != {"h", "a"}:
            raise ValueError("Understat original match roles are ambiguous")
        fingerprints.append({"game_id": game, "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)})
        for side in ("h", "a"):
            roster = payload["rosters"][side]
            team_ids = {identity(player["team_id"]) for player in roster.values()}
            if len(team_ids) != 1:
                raise ValueError("Understat original roster team identity is ambiguous")
            teams[game, side] = next(iter(team_ids))
            players = names[game, side] = {}
            roster_ids[game, side] = set()
            for player in roster.values():
                name = html.unescape(player["player"])
                player_id = identity(player["player_id"])
                players.setdefault(name, set()).add(player_id)
                roster_ids[game, side].add(player_id)
                legacy_ids.setdefault((game, name), set()).add(identity(player["id"]))
        if teams[game, "h"] == teams[game, "a"]:
            raise ValueError("Understat original opponents share a team ID")
        for side, events in payload["shots"].items():
            for event in events:
                shot = identity(event["id"])
                if (shot in raw or identity(event["match_id"]) != game or event["h_a"] != side):
                    raise ValueError("Duplicate or mismatched Understat original shot identity")
                raw[shot] = event

    situations, assists = [], []
    seen = set()
    present = {name: frame[name].notna().tolist() for name in ("situation", "assist_player", "assist_player_id")}
    counts = {"corrected_assist_ids": 0, "recovered_missing_assist_ids": 0,
              "source_association_unspecified": 0, "ambiguous_assist_names": 0,
              "unmatched_assist_names": 0, "cleared_unverifiable_assist_ids": 0}
    for position, row in enumerate(frame.to_dict("records")):
        game, shot = identity(row["game_id"]), identity(row["shot_id"])
        event = raw.get(shot)
        if (shot in seen or event is None or identity(event["match_id"]) != game
                or identity(event["player_id"]) != identity(row["player_id"])
                or identity(row["player_id"]) not in roster_ids[game, event["h_a"]]
                or identity(event["season"]) != identity(row["season_id"])
                or teams[game, event["h_a"]] != identity(row["team_id"])):
            raise ValueError("Understat frame and original shot identity disagree")
        seen.add(shot)
        situation = row["situation"] if present["situation"][position] else ""
        if not isinstance(situation, str):
            raise ValueError("Understat situation must be a label or missing")
        situations.append({"shot_id": shot, "situation": situation})
        raw_name = event["player_assisted"]
        if raw_name is not None and not isinstance(raw_name, str):
            raise ValueError("Understat original assist association must be a name or null")
        name = html.unescape(raw_name) if isinstance(raw_name, str) and raw_name else None
        displayed = row["assist_player"] if present["assist_player"][position] else None
        displayed = None if displayed == "" else displayed
        if displayed != name:
            raise ValueError("Understat assist name disagrees with original shot")
        existing = (identity(row["assist_player_id"])
                    if present["assist_player_id"][position] and row["assist_player_id"] != "" else None)
        if name is None:
            if existing is not None:
                raise ValueError("Understat assist ID lacks an original association")
            counts["source_association_unspecified"] += 1
            expected = None
        else:
            candidates = names[game, event["h_a"]].get(name, set())
            if len(candidates) != 1:
                counts["ambiguous_assist_names" if candidates else "unmatched_assist_names"] += 1
                counts["cleared_unverifiable_assist_ids"] += existing is not None
                expected = None
            else:
                expected = next(iter(candidates))
                if existing is not None and existing != expected and existing not in legacy_ids[game, name]:
                    raise ValueError("Understat assist ID is neither stable nor an original roster-entry ID")
                counts["corrected_assist_ids"] += existing is not None and existing != expected
                counts["recovered_missing_assist_ids"] += existing is None
        assists.append(expected)

    restored, recovered = restore_situations(situations, {shot: event["situation"] for shot, event in raw.items()})
    result = frame.copy()
    labels, identifiers = frame["situation"].copy(), frame["assist_player_id"].copy()
    for position, (shot, expected) in enumerate(zip(restored, assists)):
        labels.iloc[position] = shot["situation"]
        if expected is None:
            if present["assist_player_id"][position] and identifiers.iloc[position] != "":
                identifiers.iloc[position] = None
        elif identifiers.dtype.kind in ("O", "U", "S"):
            identifiers.iloc[position] = expected
        else:
            identifiers.iloc[position] = int(expected)
    # SDK MultiIndex labels repeat across a player's shots: assign by position.
    result["situation"], result["assist_player_id"] = labels.array, identifiers.array
    return result, {"rows": len(frame), "restored_situations": recovered,
                    "assist_id_semantics": "stable same-side roster player_id, never roster-entry id",
                    "name_normalization": "HTML entity decoding only; stable target uses same-side roster; no fuzzy matching",
                    "raw_sources": fingerprints, **counts}


def unique_understat_player_matches(frame, *, quarantine_conflicts=False):
    """Retain unique identities; quarantining conflicts requires an explicit caller policy."""
    flat = frame.reset_index()
    keys = ["league", "season", "game_id", "team_id", "player_id"]
    if (not set(keys) <= set(flat.columns) or flat[keys].isna().any().any()
            or flat[keys].astype(str).eq("").any().any()):
        raise ValueError("Understat player-match identity is missing")
    keep = ~flat.duplicated()
    unique = flat.loc[keep]
    conflicting = unique.duplicated(subset=keys, keep=False)
    if conflicting.any() and not quarantine_conflicts:
        raise ValueError("Conflicting Understat player-match identity")
    conflicting_keys = list(unique.loc[conflicting, keys].drop_duplicates().itertuples(index=False, name=None))
    excluded = flat[keys].apply(tuple, axis=1).isin(set(conflicting_keys))
    identical_duplicates = int((~keep).sum())
    keep &= ~excluded
    result = frame.iloc[keep.to_numpy()].copy()
    return result, {
        "key_fields": keys,
        "raw_sdk_rows": len(frame),
        "unique_rows": len(result),
        "identical_duplicates_removed": identical_duplicates,
        "quarantined_unique_rows": int(conflicting.sum()),
        "quarantined_raw_rows": int(excluded.sum()),
        "conflicting_identities": [dict(zip(keys, map(str, key))) for key in conflicting_keys],
        "policy": "Remove fully identical records; conflicting identities are rejected by default or explicitly quarantined in full. Original match responses remain in the cache.",
    }


def anonymous_browser_reader(base, driver, download):
    # SoccerData uses cls.__name__ as the league-configuration key.
    return type(
        base.__name__,
        (base,),
        {"_init_webdriver": driver, "_download_and_save": download},
    )


def espn_legacy_summary(payload):
    if "form" in payload.get("boxscore", {}):
        return payload
    rosters = payload.get("rosters") or []
    if len(rosters) != 2 or {r.get("homeAway") for r in rosters} != {"home", "away"}:
        raise ValueError("ESPN roster home/away mapping is ambiguous")
    result = copy.deepcopy(payload)
    rosters = sorted(result["rosters"], key=lambda r: r["homeAway"] != "home")
    teams = {str(r["team"]["id"]): r for r in result["boxscore"]["teams"]}
    if any(str(r["team"]["id"]) not in teams for r in rosters):
        raise ValueError("ESPN roster and boxscore team IDs disagree")
    result["rosters"] = rosters
    result["boxscore"]["teams"] = [teams[str(r["team"]["id"])] for r in rosters]
    result["boxscore"]["form"] = [{"team": r["team"]} for r in rosters]
    return result


def run(
    provider,
    output,
    leagues=("ENG-Premier League",),
    seasons=(2024,),
    request_budget=20,
    deadline=120,
    browser=False,
    details=False,
    all_matches=False,
):
    if provider not in PROVIDERS:
        raise ValueError("Unknown SoccerData reader")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["SOCCERDATA_DIR"] = str(output / "runtime")
    os.environ["SOCCERDATA_LOGLEVEL"] = "ERROR"
    import soccerdata as sd
    from soccerdata._common import BaseSeleniumReader

    cls = getattr(sd, provider)
    report = {
        "provider": provider,
        "soccerdata_version": sd.__version__,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "leagues": list(leagues),
        "seasons": list(seasons),
        "requests": [],
        "tables": [],
        "errors": [],
        "details_requested": details,
        "all_matches_requested": all_matches,
        "public_read_methods": [name for name in dir(cls) if name.startswith("read_")],
        "controls": {
            "request_budget": request_budget,
            "deadline_seconds": deadline,
            "anonymous": True,
            "browser": browser,
            "human_verification": "stop; no solver",
            "mirror": "not automatically pooled with existing sources",
        },
    }
    sdk_root = Path(inspect.getfile(sd)).parent
    report["sdk_source_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [
            sdk_root / "_common.py",
            sdk_root / "_config.py",
            Path(inspect.getfile(cls)),
        ]
    }
    capture = RequestCapture(report, output, request_budget)
    start = time.monotonic()
    reader = None
    created_drivers = []

    def stop_deadline(signum, frame):
        raise SourceStopped("provider_deadline_reached")

    previous_handler = signal.signal(signal.SIGALRM, stop_deadline)
    signal.alarm(deadline)
    try:
        is_browser = issubclass(cls, BaseSeleniumReader)
        if is_browser and not browser:
            raise SourceStopped("browser_backend_not_requested")
        if is_browser:
            import seleniumbase as sb

            def normal_driver(self):
                driver = sb.Driver(
                    browser="chrome",
                    uc=False,
                    headless=True,
                    binary_location="/usr/bin/google-chrome",
                    user_data_dir=str(output / "isolated-browser-profile"),
                )
                created_drivers.append(driver)
                return driver

            def download(self, url, filepath=None, var=None):
                return capture.browser_download(self, url, filepath=filepath, var=var)

            cls = anonymous_browser_reader(cls, normal_driver, download)
        else:
            cls = tracked_requests_reader(cls, capture)
        kwargs = {"data_dir": output / "cache"}
        parameters = inspect.signature(getattr(sd, provider)).parameters
        if "leagues" in parameters:
            kwargs["leagues"] = list(leagues)
        if "seasons" in parameters:
            kwargs["seasons"] = explicit_seasons(seasons)
        if is_browser:
            kwargs.update(
                {"headless": True, "path_to_browser": Path("/usr/bin/google-chrome")}
            )
        reader = cls(**kwargs)
        if "seasons" in parameters:
            report["resolved_seasons"] = list(reader.seasons)
            if len(set(reader.seasons)) != len(set(seasons)):
                raise SourceStopped("ambiguous_or_duplicate_resolved_seasons")
        if is_browser and not hasattr(reader, "_driver"):
            raise SourceStopped("browser_driver_unavailable")
        if provider == "ESPN":
            original_get = reader.get

            def legacy_get(url, *args, **kwargs):
                result = original_get(url, *args, **kwargs)
                if "/summary?" not in url:
                    return result
                payload = json.load(result)
                converted = espn_legacy_summary(payload)
                report["parser_compatibility"] = (
                    "ESPN boxscore.form derived from current roster/team IDs; home/away aligned. Original cached JSON is unchanged."
                )
                import io

                return io.BytesIO(json.dumps(converted).encode())

            reader.get = legacy_get

        def save(label, frame):
            report["tables"].append(
                {
                    "method": label,
                    **export_frame(frame, output / "tables" / (label + ".csv")),
                }
            )

        if provider == "MatchHistory":
            save("games", reader.read_games())
        elif provider == "ClubElo":
            save("ratings_2024-07-01", reader.read_by_date("2024-07-01"))
            save("ManchesterCity_history", reader.read_team_history("ManCity"))
        elif provider == "ESPN":
            schedule = reader.read_schedule()
            save("schedule", schedule)
            if details and len(schedule):
                game_id = None if all_matches else int(schedule.iloc[0]["game_id"])
                save(
                    "matchsheets" if all_matches else "sample_matchsheet",
                    reader.read_matchsheet(match_id=game_id),
                )
                save(
                    "lineups" if all_matches else "sample_lineup",
                    reader.read_lineup(match_id=game_id),
                )
        elif provider == "Sofascore":
            save("leagues", reader.read_leagues())
            save("seasons", reader.read_seasons())
            save("league_table", reader.read_league_table())
            save("schedule", reader.read_schedule())
        elif provider == "Understat":
            save("schedule", reader.read_schedule())
            save("player_season_stats", reader.read_player_season_stats())
            save("team_match_stats", reader.read_team_match_stats())
            if details:
                shots, shot_audit = normalize_understat_shots(reader.read_shot_events(), output / "cache")
                report.setdefault("normalization", {})["shot_events"] = shot_audit
                save("shot_events", shots)
                players, audit = unique_understat_player_matches(
                    reader.read_player_match_stats(), quarantine_conflicts=True)
                report["normalization"]["player_match_stats"] = audit
                report["quality_status"] = (
                    "shot_and_player_fields_normalized_with_exclusions"
                    if (audit["conflicting_identities"] or shot_audit["ambiguous_assist_names"]
                        or shot_audit["unmatched_assist_names"])
                    else "shot_links_and_player_keys_verified")
                save("player_match_stats", players)
            else:
                schedule = reader.read_schedule()
                finished = (
                    schedule.dropna(subset=["home_score", "away_score"])
                    if {"home_score", "away_score"} <= set(schedule.columns)
                    else schedule
                )
                if len(finished) and "game_id" in finished.columns:
                    shots, shot_audit = normalize_understat_shots(
                        reader.read_shot_events(match_id=int(finished.iloc[0]["game_id"])), output / "cache")
                    report.setdefault("normalization", {})["sample_shot_events"] = shot_audit
                    report["quality_status"] = (
                        "unresolved_shot_associations"
                        if shot_audit["ambiguous_assist_names"] or shot_audit["unmatched_assist_names"]
                        else "shot_fields_reconciled")
                    save(
                        "sample_shot_events",
                        shots,
                    )
        elif provider == "FBref":
            save(
                "player_season_standard",
                reader.read_player_season_stats(stat_type="standard"),
            )
        elif provider == "SoFIFA":
            save("leagues", reader.read_leagues())
            save("versions", reader.read_versions())
        elif provider == "WhoScored":
            save("leagues", reader.read_leagues())
    except (SourceStopped, Exception) as exc:
        report["errors"].append({"type": type(exc).__name__, "reason": str(exc)})
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        for driver in created_drivers:
            try:
                driver.quit()
            except Exception:
                pass
    report["elapsed_seconds"] = time.monotonic() - start
    report["status"] = (
        "collected"
        if report["tables"] and not report["errors"]
        else "partial" if report["tables"] else "unavailable_in_probe"
    )
    write_json_atomic(output / "report.json", report)
    print(
        json.dumps(
            {
                "provider": provider,
                "status": report["status"],
                "tables": len(report["tables"]),
                "rows": sum(t["rows"] for t in report["tables"]),
                "requests": len(report["requests"]),
                "errors": report["errors"],
            },
            ensure_ascii=False,
        )
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("provider", choices=PROVIDERS)
    p.add_argument("--output", required=True)
    p.add_argument("--leagues", default="ENG-Premier League")
    p.add_argument("--seasons", default="2024")
    p.add_argument("--request-budget", type=int, default=20)
    p.add_argument("--deadline", type=int, default=120)
    p.add_argument("--browser", action="store_true")
    p.add_argument("--details", action="store_true")
    p.add_argument("--all-matches", action="store_true")
    a = p.parse_args()
    run(
        a.provider,
        a.output,
        tuple(a.leagues.split(",")),
        tuple(map(int, a.seasons.split(","))),
        a.request_budget,
        a.deadline,
        a.browser,
        a.details,
        a.all_matches,
    )
