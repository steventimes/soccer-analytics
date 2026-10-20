"""Bounded anonymous SoccerData reads, retaining raw caches and parsed frames.

Install SoccerData in an optional environment. Each provider is independent;
blocked requests, human verification and empty frames are recorded explicitly.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import inspect
import hashlib
import json
import os
from pathlib import Path
import signal
import time

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


class SourceStopped(BaseException):
    """Stop upstream retry/captcha loops while preserving the access evidence."""


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
                if len(report["requests"]) >= request_budget:
                    raise SourceStopped("request_budget_reached")
                entry = {"url": url, "transport": "normal_isolated_chrome"}
                report["requests"].append(entry)
                self._driver.get(url)
                time.sleep(1)
                text = self._driver.page_source
                if any(
                    s in text
                    for s in [
                        "Verify you are human",
                        "Checking your browser",
                        "Just a moment...",
                        "cf-challenge",
                    ]
                ):
                    entry["status"] = "human_verification"
                    raise SourceStopped("browser_human_verification_required")
                if '"reason":"Forbidden"' in text or '"reason": "Forbidden"' in text:
                    entry["status"] = "forbidden"
                    raise SourceStopped("browser_forbidden")
                try:
                    payload = (
                        self._validate_page(url).encode()
                        if var is None
                        else json.dumps(
                            self._driver.execute_script("return " + var)
                        ).encode()
                    )
                except Exception as exc:
                    entry["status"] = "invalid_page"
                    raise SourceStopped("invalid_browser_page:" + str(exc)) from exc
                entry.update({"status": "page_read", "bytes": len(payload)})
                if filepath is not None:
                    filepath.parent.mkdir(parents=True, exist_ok=True)
                    filepath.write_bytes(payload)
                import io

                return io.BytesIO(payload)

            cls = anonymous_browser_reader(cls, normal_driver, download)
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
        if not is_browser:
            original = reader._session.get

            def tracked_get(url, *args, **kwargs):
                if len(report["requests"]) >= request_budget:
                    raise SourceStopped("request_budget_reached")
                entry = {"url": url, "transport": "soccerdata_native_tls_client"}
                report["requests"].append(entry)
                try:
                    response = original(url, *args, **kwargs)
                except Exception as exc:
                    entry["error"] = type(exc).__name__
                    raise SourceStopped("transport_error:" + str(exc)) from exc
                entry.update(
                    {
                        "http_status": response.status_code,
                        "bytes": len(response.content),
                    }
                )
                if (
                    not isinstance(response.status_code, int)
                    or response.status_code < 100
                ):
                    raise SourceStopped("non_http_transport_response")
                if response.status_code >= 400:
                    raise SourceStopped("http_" + str(response.status_code))
                return response

            reader._session.get = tracked_get
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
                save("shot_events", reader.read_shot_events())
                save("player_match_stats", reader.read_player_match_stats())
            else:
                schedule = reader.read_schedule()
                finished = (
                    schedule.dropna(subset=["home_score", "away_score"])
                    if {"home_score", "away_score"} <= set(schedule.columns)
                    else schedule
                )
                if len(finished) and "game_id" in finished.columns:
                    save(
                        "sample_shot_events",
                        reader.read_shot_events(
                            match_id=int(finished.iloc[0]["game_id"])
                        ),
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
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
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
