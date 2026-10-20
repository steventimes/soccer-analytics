import unittest
import tempfile
from pathlib import Path
import hashlib
import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
import requests
from app.data_service import archive
from app.data_service import recent_matches as recent
from app.data_service.recent_matches import (
    calendar_days,
    normalize_event,
    historical_result,
)
from app.data_service.recent_matches import ingest


class TestRecentMatches(unittest.TestCase):
    def test_archive_retry_closes_failed_streams_and_respects_dates_and_permanent_statuses(self):
        import io
        from datetime import datetime, timedelta, timezone
        from email.utils import format_datetime
        from app.data_service.fetch.retry import RetryAfterLimitError, RetryPolicy

        def response(status, retry_after=None):
            r = requests.Response(); r.status_code = status; r.url = 'https://public.invalid/data'
            r.raw = io.BytesIO(b'data')
            if retry_after is not None: r.headers['Retry-After'] = retry_after
            return r

        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        policy = RetryPolicy(backoff_factor=1., wall_clock=lambda: now.timestamp(), random_value=lambda: .5)
        limited = response(429, format_datetime(now + timedelta(seconds=12), usegmt=True))
        successful = response(200); session = MagicMock()
        session.get.side_effect = [requests.ConnectionError('offline'), limited, successful]
        reserve, waits = MagicMock(), []
        result = archive._request(session, 'https://public.invalid/data', stream=True,
                                  before_request=reserve, retry_policy=policy, sleep=waits.append)
        self.assertIs(result, successful)
        self.assertEqual(waits, [.5, 12.])
        self.assertEqual(reserve.call_count, 3)
        self.assertEqual(session.get.call_count, 3)
        self.assertTrue(limited.raw.closed)
        self.assertFalse(successful.raw.closed)
        for status in [403, 404, 501]:
            failed = response(status, '0'); session.get.side_effect = [failed, successful]
            session.reset_mock(); waits.clear()
            with self.assertRaises(requests.HTTPError):
                archive._request(session, 'https://public.invalid/data', stream=True,
                                 retry_policy=policy, sleep=waits.append)
            self.assertEqual(session.get.call_count, 1)
            self.assertEqual(waits, [])
            self.assertTrue(failed.raw.closed)
        deferred = response(503, '120'); session.get.side_effect = [deferred, successful]
        session.reset_mock()
        with self.assertRaises(RetryAfterLimitError) as error:
            archive._request(session, 'https://public.invalid/data', stream=True,
                             retry_policy=policy, sleep=waits.append)
        self.assertIs(error.exception.response, deferred)
        self.assertEqual(error.exception.retry_after_seconds, 120.)
        self.assertEqual(session.get.call_count, 1)
        self.assertTrue(deferred.raw.closed)
        exhausted = response(503, '120')
        session.get.side_effect = [requests.ConnectionError('offline')] * 3 + [exhausted]
        session.reset_mock(); waits.clear()
        with self.assertRaises(requests.HTTPError) as error:
            archive._request(session, 'https://public.invalid/data', stream=True,
                             retry_policy=policy, sleep=waits.append)
        self.assertIs(error.exception.response, exhausted)
        self.assertEqual(error.exception.retry_after_seconds, 120.)
        self.assertEqual(session.get.call_count, 4)
        self.assertEqual(waits, [.5, 1., 2.])
        self.assertTrue(exhausted.raw.closed)
        successful.close()

    def test_retry_attempts_each_reserve_rate_budget(self):
        session = MagicMock()
        response = MagicMock()
        session.get.side_effect = [requests.RequestException("temporary"), response]
        reserve = MagicMock()
        with patch("time.sleep"):
            self.assertIs(
                archive._request(
                    session, "https://public.invalid", before_request=reserve
                ),
                response,
            )
        self.assertEqual(reserve.call_count, 2)
        self.assertEqual(session.get.call_count, 2)

    def event(self, identifier="732497"):
        return {
            "id": identifier,
            "date": "2026-10-10T12:00Z",
            "status": {"type": {"state": "post", "completed": True}},
            "competitions": [
                {
                    "competitors": [
                        {
                            "homeAway": "home",
                            "team": {"id": "1", "displayName": "A"},
                            "score": "1",
                        },
                        {
                            "homeAway": "away",
                            "team": {"id": "2", "displayName": "B"},
                            "score": "0",
                        },
                    ]
                }
            ],
        }

    def summary_header(self, event):
        return {
            "id": event["id"],
            "competitions": [{
                "status": event["status"],
                "competitors": [
                    {"id": team["team"]["id"], "homeAway": team["homeAway"], "score": team["score"]}
                    for team in event["competitions"][0]["competitors"]
                ],
            }],
        }

    def snapshot(self, root, timestamp, goals, *, summary_time=None, summary=True):
        root.mkdir()
        event = self.event()
        event["competitions"][0]["competitors"][0]["score"] = goals
        match = normalize_event(event, "eng.1", "club", "male")
        match["raw_path"] = "scoreboard.json"
        files, summaries = [], []
        payloads = [("scoreboard.json", {"events": [event]}, timestamp)]
        if summary:
            payloads.append(("summary.json", {"header": self.summary_header(event), "note": root.name}, summary_time or timestamp))
            summaries.append({"competition": "eng.1", "match_id": event["id"], "path": "summary.json"})
        for relative, payload, observed in payloads:
            content = json.dumps(payload).encode()
            (root / relative).write_bytes(content)
            files.append({"path": relative, "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content), "observed_at": observed})
        (root / "collection.json").write_text(json.dumps({
            "end": "2026-10-10", "observed_at": "2026-10-16T12:00:00Z", "files": files,
            "matches": [match], "summaries": summaries, "football_csv": [], "errors": [],
        }))
        return root

    def test_provider_id_rejects_path_query_and_unicode_preserving_numeric_ids(self):
        for identifier in [
            "x/../../../escaped",
            "..\\escaped",
            "1?x=2",
            "1%2f2",
            "１",
            "-1",
            "1\n",
            "",
        ]:
            with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                normalize_event(self.event(identifier), "eng.1", "club", "male")
        for identifier in [1, "732497", "401879276", "0001"]:
            self.assertEqual(
                normalize_event(self.event(identifier), "eng.1", "club", "male")[
                    "match_id"
                ],
                str(identifier),
            )

    def test_collection_blocks_outside_write_and_counts_uncached_requests(self):
        for identifier, expected in [("x/../../../escaped", 0), ("732497", 1)]:
            with self.subTest(
                identifier=identifier
            ), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                root = base / "a" / "cache"

                def request(session, url, *, before_request):
                    before_request()
                    if "summary?" in url:
                        payload = {"header": self.summary_header(self.event(identifier))}
                    elif "dates=20260701" in url:
                        payload = {"leagues": [{"calendar": ["2026-10-10T07:00Z"]}]}
                    else:
                        payload = {"events": [self.event(identifier)]}
                    return SimpleNamespace(content=json.dumps(payload).encode())

                with patch.object(
                    recent, "_request", side_effect=request
                ) as http, patch.object(recent, "LEAGUES", {}), patch.object(
                    recent, "SlidingWindowRateLimiter"
                ) as factory:
                    report = recent.collect(
                        root, "2026-10-10", "2026-10-10", {"eng.1": ("club", "male")}
                    )
                    self.assertEqual(len(report["summaries"]), expected)
                    self.assertFalse((base / "a" / "escaped.json").exists())
                    self.assertEqual(
                        factory.return_value.add_request.call_count, 2 + expected
                    )
                    self.assertEqual(
                        factory.return_value.wait_if_needed.call_count, http.call_count
                    )
                    if expected:
                        factory.return_value.reset_mock()
                        http.reset_mock()
                        repeated = recent.collect(
                            root,
                            "2026-10-10",
                            "2026-10-10",
                            {"eng.1": ("club", "male")},
                            resume=True,
                        )
                        self.assertEqual(len(repeated["summaries"]), 1)
                        self.assertEqual(repeated["observed_at"], report["observed_at"])
                        self.assertEqual(repeated["requests"]["downloaded"], 0)
                        self.assertEqual(repeated["requests"]["cached"], 3)
                        http.assert_not_called()
                        factory.return_value.add_request.assert_not_called()
                        before = (root / "collection.json").read_bytes()
                        with self.assertRaisesRegex(ValueError, "resume"):
                            recent.collect(root, "2026-10-10", "2026-10-10", {})
                        with self.assertRaisesRegex(ValueError, "request"):
                            recent.collect(root, "2026-10-10", "2026-10-11", {}, resume=True)
                        self.assertEqual((root / "collection.json").read_bytes(), before)

    def test_resume_checks_cached_request_identity_and_payload_size(self):
        for field, replacement in [("url", "https://wrong.invalid"), ("bytes", 1)]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "snapshot"
                payload = json.dumps({"leagues": [{"calendar": ["2027-01-01"]}]}).encode()
                response = SimpleNamespace(content=payload)
                with patch.object(recent, "_request", return_value=response), patch.object(recent, "LEAGUES", {}):
                    recent.collect(root, "2026-10-10", "2026-10-10", {"eng.1": ("club", "male")})
                receipt = root / "espn/calendar_eng.1.json.receipt.json"
                record = json.loads(receipt.read_text())
                record[field] = replacement
                receipt.write_text(json.dumps(record))
                with patch.object(recent, "_request") as http, patch.object(recent, "LEAGUES", {}):
                    report = recent.collect(root, "2026-10-10", "2026-10-10", {"eng.1": ("club", "male")}, resume=True)
                http.assert_not_called()
                self.assertEqual(len(report["errors"]), 1)
                self.assertEqual(report["errors"][0]["stage"], "calendar")
                self.assertIn("Cached response", report["errors"][0]["error"])
                self.assertEqual((root / "espn/calendar_eng.1.json").read_bytes(), payload)

    def test_cli_creates_distinct_snapshots_and_requires_root_for_resume_or_ingest(self):
        make_directory = tempfile.mkdtemp
        with tempfile.TemporaryDirectory() as directory:
            def new_snapshot(**kwargs):
                return make_directory(prefix=kwargs["prefix"], dir=directory)
            with patch.object(recent, "collect") as collect, patch.object(recent.tempfile, "mkdtemp", side_effect=new_snapshot):
                recent.main(["--collect-only", "--end", "2026-10-10"])
                recent.main(["--collect-only", "--end", "2026-10-10"])
                roots = [Path(call.args[0]) for call in collect.call_args_list]
                self.assertNotEqual(*roots)
                self.assertTrue(all(root.is_dir() for root in roots))
                recent.main(["--collect-only", "--end", "2026-10-10",
                             "--competitions", "uefa.champions,eng.1,uefa.champions",
                             "--csv-competitions", "PL,FL1,PL"])
                selected = collect.call_args
                self.assertEqual(list(selected.args[3]), ["uefa.champions", "eng.1"])
                self.assertEqual(selected.kwargs["csv_competitions"], ["PL", "FL1"])
                recent.main(["--collect-only", "--end", "2026-10-10",
                             "--competitions", "none", "--csv-competitions", "none"])
                self.assertEqual(collect.call_args.args[3], {})
                self.assertEqual(collect.call_args.kwargs["csv_competitions"], [])
            for args in [["--competitions", "eng.1,unknown"],
                         ["--csv-competitions", "PL,"],
                         ["--csv-competitions", "unknown"]]:
                with self.subTest(args=args), patch.object(recent, "collect") as collect, patch.object(recent.tempfile, "mkdtemp") as create, patch("sys.stderr"), self.assertRaises(SystemExit) as error:
                    recent.main(["--collect-only", *args])
                self.assertEqual(error.exception.code, 2)
                collect.assert_not_called()
                create.assert_not_called()
            for flag in ["--resume", "--ingest-only"]:
                with self.subTest(flag=flag), patch.object(recent, "collect") as collect, patch("sys.stderr"), self.assertRaises(SystemExit) as error:
                    recent.main([flag])
                self.assertEqual(error.exception.code, 2)
                collect.assert_not_called()

    def test_ingest_preserves_source_time_and_does_not_roll_back_newer_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "national"
            source.mkdir()
            (source / "manifest.json").write_text(json.dumps({"revision": "abc", "files": []}))
            database = base / "archive.sqlite3"
            for name, timestamp, goals in [("new", "2026-10-15T00:00:00Z", "2"), ("old", "2026-10-14T00:00:00Z", "1")]:
                root = self.snapshot(base / name, timestamp, goals)
                result = ingest(database, root, source)
                with sqlite3.connect(database) as connection:
                    observation = connection.execute("SELECT observed_at,payload,asset_id FROM latest_match_observations").fetchone()
                    summary = connection.execute("SELECT asset_id FROM recent_match_summaries").fetchone()
                self.assertEqual(observation[0], "2026-10-15T00:00:00+00:00")
                self.assertEqual(json.loads(observation[1])["home_goals"], 2)
                if name == "new":
                    first_asset, first_summary = observation[2], summary[0]
                else:
                    self.assertEqual(observation[2], first_asset)
                    self.assertEqual(summary[0], first_summary)
                    self.assertEqual(result["skipped_older_observations"], {"espn": 1})

    def test_summary_has_independent_time_and_result_revision_invalidates_old_link(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "national"
            source.mkdir()
            (source / "manifest.json").write_text(json.dumps({"revision": "abc", "files": []}))
            database = base / "archive.sqlite3"
            first = self.snapshot(base / "later-summary", "2026-10-14T00:00:00Z", "1", summary_time="2026-10-16T00:00:00Z")
            second = self.snapshot(base / "earlier-summary", "2026-10-15T00:00:00Z", "1")
            third = self.snapshot(base / "revised-result", "2026-10-17T00:00:00Z", "2", summary=False)
            ingest(database, first, source)
            with sqlite3.connect(database) as connection:
                first_summary = connection.execute("SELECT asset_id FROM recent_match_summaries").fetchone()[0]
            ingest(database, second, source)
            with sqlite3.connect(database) as connection:
                self.assertEqual(connection.execute("SELECT asset_id FROM recent_match_summaries").fetchone()[0], first_summary)
                self.assertEqual(connection.execute("SELECT observed_at FROM latest_match_observations").fetchone()[0], "2026-10-15T00:00:00+00:00")
            ingest(database, third, source)
            with sqlite3.connect(database) as connection:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM recent_match_summaries").fetchone()[0], 0)
                self.assertEqual(json.loads(connection.execute("SELECT payload FROM latest_match_observations").fetchone()[0])["home_goals"], 2)
                self.assertIsNotNone(connection.execute("SELECT id FROM assets WHERE id=? AND complete=1", (first_summary,)).fetchone())

    def test_summary_score_conflict_is_rejected_before_database_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = self.snapshot(base / "mismatch", "2026-10-15T00:00:00Z", "2")
            path = root / "summary.json"
            data = json.loads(path.read_text())
            data["header"]["competitions"][0]["competitors"][0]["score"] = "1"
            path.write_text(json.dumps(data))
            report = json.loads((root / "collection.json").read_text())
            record = report["files"][1]
            record.update(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            (root / "collection.json").write_text(json.dumps(report))
            database = base / "archive.sqlite3"
            with self.assertRaisesRegex(ValueError, "Summary.*score"):
                ingest(database, root, base / "unused-national")
            self.assertFalse(database.exists())

    def test_saved_manifest_cannot_read_outside_payload_even_with_matching_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "cache"
            root.mkdir()
            outside = base / "outside.json"
            outside.write_text('{"sentinel": "public-test"}')
            (root / "alias.json").symlink_to(outside)
            for name in ["../outside.json", str(outside), "alias.json"]:
                (root / "collection.json").write_text(
                    json.dumps(
                        {
                            "files": [
                                {
                                    "path": name,
                                    "sha256": hashlib.sha256(
                                        outside.read_bytes()
                                    ).hexdigest(),
                                }
                            ]
                        }
                    )
                )
                with self.subTest(name=name), self.assertRaisesRegex(
                    ValueError, "outside inventory"
                ):
                    ingest(base / "database.sqlite3", root, base / "unused")

    def test_calendar_handles_day_lists_and_uses_requested_days_for_phase_blocks(self):
        self.assertEqual(
            calendar_days(
                ["2026-10-10T07:00Z", "2026-11-01T07:00Z"], "2026-10-09", "2026-10-11"
            ),
            ["20261010"],
        )
        self.assertEqual(
            calendar_days(
                [{"entries": [{"startDate": "2026-09-24"}]}], "2026-10-09", "2026-10-11"
            ),
            ["20261009", "20261010", "20261011"],
        )

    def test_collection_covers_intermediate_and_first_half_seasons_without_duplicate_day_fetches(self):
        calendars = {
            "20240701": ["2025-06-30T07:00Z", "2025-07-01T07:00Z"],
            "20250701": ["2025-07-01T07:00Z", "2026-01-06T07:00Z", "2026-07-01T07:00Z"],
            "20260701": ["2026-07-01T07:00Z"],
        }
        def request(session, url, *, before_request):
            before_request()
            day = url.split("dates=")[1].split("&")[0]
            if "limit=" not in url:
                payload = {"leagues": [{"calendar": calendars[day]}]}
            else:
                event = self.event(day)
                event["date"] = f"{day[:4]}-{day[4:6]}-{day[6:]}T12:00Z"
                event["status"] = {"type": {"state": "pre", "completed": False}}
                payload = {"events": [event]}
            return SimpleNamespace(content=json.dumps(payload).encode())

        with tempfile.TemporaryDirectory() as directory, patch.object(recent, "_request", side_effect=request) as http, patch.object(recent, "SlidingWindowRateLimiter"):
            root = Path(directory) / "multi"
            report = recent.collect(root, "2025-06-30", "2026-07-01",
                                    {"eng.1": ("club", "male")}, csv_competitions=[])
            self.assertEqual(report["errors"], [])
            self.assertEqual({m["match_id"] for m in report["matches"]},
                             {"20250630", "20250701", "20260106", "20260701"})
            urls = [call.args[1] for call in http.call_args_list]
            self.assertEqual(len(urls), len(set(urls)))
            self.assertEqual(len(urls), 7)
            self.assertEqual({Path(r["path"]).name for r in report["files"] if "calendar_" in r["path"]},
                             {"calendar_eng.1_2024.json", "calendar_eng.1_2025.json", "calendar_eng.1_2026.json"})
            http.reset_mock()
            resumed = recent.collect(root, "2025-06-30", "2026-07-01",
                                     {"eng.1": ("club", "male")}, csv_competitions=[], resume=True)
            http.assert_not_called()
            self.assertEqual(resumed["requests"], {"attempts": 0, "downloaded": 0, "cached": 7})
            with self.assertRaisesRegex(ValueError, "different collection request"):
                recent.collect(root, "2025-06-30", "2026-07-01",
                               {"eng.1": ("club", "male")}, csv_competitions=["PL"], resume=True)
            first_half = recent.collect(Path(directory) / "first-half", "2026-01-06", "2026-01-06",
                                        {"eng.1": ("club", "male")}, csv_competitions=[])
            self.assertEqual([m["match_id"] for m in first_half["matches"]], ["20260106"])
            self.assertIn("dates=20250701", first_half["files"][0]["url"])
            with self.assertRaisesRegex(ValueError, "Unknown Football-Data"):
                recent.collect(Path(directory) / "invalid", "2026-01-06", "2026-01-06", csv_competitions=["BAD"])
            self.assertFalse((Path(directory) / "invalid").exists())

            def phase_request(session, url, *, before_request):
                before_request()
                payload = ({"events": []} if "limit=" in url else
                           {"leagues": [{"calendar": [{"entries": []}]}]})
                return SimpleNamespace(content=json.dumps(payload).encode())
            http.reset_mock(); http.side_effect = phase_request
            phase = recent.collect(Path(directory) / "phase", "2026-06-30", "2026-07-01",
                                   {"uefa.champions": ("club", "male")}, csv_competitions=[])
            self.assertEqual(phase["errors"], [])
            self.assertEqual(len(http.call_args_list), 4)
            self.assertEqual(len({call.args[1] for call in http.call_args_list}), 4)

    def test_scheduled_zero_scores_are_unknown_and_completed_draw_is_retained(self):
        event = {
            "id": "1",
            "date": "2026-10-11T12:00Z",
            "status": {"type": {"state": "pre", "completed": False}},
            "competitions": [
                {
                    "competitors": [
                        {
                            "homeAway": "away",
                            "team": {"id": "2", "displayName": "B"},
                            "score": "0",
                        },
                        {
                            "homeAway": "home",
                            "team": {"id": "1", "displayName": "A"},
                            "score": "0",
                        },
                    ]
                }
            ],
        }
        parsed = normalize_event(event, "fifa.friendly", "national", "male")
        self.assertIsNone(parsed["home_goals"])
        self.assertEqual(parsed["home_team"], "A")
        event["status"]["type"] = {"state": "post", "completed": True}
        self.assertEqual(
            normalize_event(event, "fifa.friendly", "national", "male")["home_goals"], 0
        )
        event["competitions"][0]["competitors"][1]["score"] = ""
        with self.assertRaises(ValueError):
            normalize_event(event, "fifa.friendly", "national", "male")

    def test_national_neutral_and_extra_time_are_not_club_home_advantage(self):
        row = {
            "date": "2026-10-06",
            "home_team": "A",
            "away_team": "B",
            "home_score": "1",
            "away_score": "1",
            "tournament": "Friendly",
            "neutral": "False",
            "city": "X",
            "country": "Y",
        }
        parsed = historical_result(row)
        self.assertFalse(parsed["neutral"])
        self.assertEqual(parsed["team_type"], "national")
        self.assertIn("extra", parsed["score_duration"])

    def test_conflicting_date_team_fixture_is_preserved_raw_and_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "refresh"
            root.mkdir()
            (root / "collection.json").write_text(
                json.dumps(
                    {
                        "start": "2026-10-01",
                        "end": "2026-10-11",
                        "observed_at": "2026-10-11",
                        "files": [],
                        "matches": [],
                        "summaries": [],
                        "football_csv": [],
                        "errors": [],
                    }
                )
            )
            source = Path(directory) / "national"
            source.mkdir()
            payload = b"date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n1974-02-17,A,B,2,1,Friendly,X,Y,FALSE\n1974-02-17,A,B,1,2,Friendly,X,Y,FALSE\n"
            (source / "results.csv").write_bytes(payload)
            (source / "manifest.json").write_text(
                json.dumps(
                    {
                        "revision": "abc",
                        "fetched_at": "2026-10-15T00:00:00Z",
                        "complete": True,
                        "files": [
                            {
                                "path": "results.csv",
                                "bytes": len(payload),
                                "sha256": hashlib.sha256(payload).hexdigest(),
                            }
                        ],
                    }
                )
            )
            database = Path(directory) / "data.sqlite3"
            report = ingest(database, root, source)
            self.assertEqual(report["identity_conflicts"], 1)
            with sqlite3.connect(database) as c:
                self.assertEqual(
                    c.execute(
                        "SELECT COUNT(*) FROM latest_match_observations"
                    ).fetchone()[0],
                    0,
                )
                self.assertEqual(
                    c.execute(
                        "SELECT COUNT(*) FROM international_results_csv_results"
                    ).fetchone()[0],
                    2,
                )
            for name, fetched, expected in [("old-valid", "2026-10-14T00:00:00Z", 0), ("tied-valid", "2026-10-15T00:00:00Z", 0), ("new-valid", "2026-10-16T00:00:00Z", 1)]:
                other = Path(directory) / name
                other.mkdir()
                content = b"\n".join(payload.splitlines()[:2]) + b"\n"
                (other / "results.csv").write_bytes(content)
                (other / "manifest.json").write_text(json.dumps({
                    "revision": name, "fetched_at": fetched, "files": [{
                        "path": "results.csv", "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                    }],
                }))
                ingest(database, root, other)
                with sqlite3.connect(database) as c:
                    self.assertEqual(c.execute("SELECT COUNT(*) FROM latest_match_observations").fetchone()[0], expected)
            ingest(database, root, source)
            with sqlite3.connect(database) as c:
                self.assertEqual(c.execute("SELECT COUNT(*) FROM latest_match_observations").fetchone()[0], 1)
