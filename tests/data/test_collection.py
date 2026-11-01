from pathlib import Path
from unittest.mock import Mock, patch
import copy
import csv
import hashlib
import importlib.util
import json
import os
import tempfile
import unittest

from app.data_service.soccerdata_collect import explicit_seasons, export_frame, normalize_understat_shots
from app.data_service.wyscout import collect


class TestSoccerdataExport(unittest.TestCase):
    def _assert_capture_receipt(self, output, entry, ordinal):
        from datetime import datetime

        self.assertEqual(entry["ordinal"], ordinal)
        started = datetime.fromisoformat(entry["started_at"])
        completed = datetime.fromisoformat(entry["completed_at"])
        self.assertIsNotNone(started.utcoffset())
        self.assertIsNotNone(completed.utcoffset())
        self.assertGreaterEqual(completed, started)
        self.assertGreaterEqual(entry["elapsed_seconds"], 0)
        for field in ("response", "parser_payload"):
            if field not in entry:
                continue
            receipt = entry[field]
            path = Path(receipt["path"])
            self.assertFalse(path.is_absolute())
            content = (output / path).read_bytes()
            self.assertEqual(len(content), receipt["bytes"])
            self.assertEqual(hashlib.sha256(content).hexdigest(), receipt["sha256"])
            self.assertTrue(receipt["kind"])
        if "response" in entry:
            observed = datetime.fromisoformat(entry["response_observed_at"])
            self.assertLessEqual(started, observed)
            self.assertLessEqual(observed, completed)

    def test_native_capture_keeps_versions_and_budget_across_sessions_without_refetching_cache(self):
        from types import SimpleNamespace
        from app.data_service.soccerdata_collect import RequestCapture, SourceStopped, tracked_requests_reader

        responses = [SimpleNamespace(status_code=200, content=body)
                     for body in (b"first version", b"second version", b"second version")]
        pending = iter(responses)
        original_gets = []

        class Reader:
            def __init__(self, headers=None):
                self._session = self._init_session(headers)

            def _init_session(self, headers=None):
                original = Mock(side_effect=lambda *a, **kw: next(pending))
                original_gets.append(original)
                return SimpleNamespace(get=original)

            def cached_get(self, path):
                # The SDK's cache-hit path opens the existing file directly.
                return path.open("rb")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report = {"requests": []}
            capture = RequestCapture(report, output, 3)
            wrapped = tracked_requests_reader(Reader, capture)
            self.assertEqual(wrapped.__name__, Reader.__name__)
            reader = wrapped(headers={"Accept": "application/json"})
            url = "https://example.invalid/match/10"
            response = reader._session.get(url, timeout=3)
            self.assertIs(response, responses[0])
            self.assertEqual(response.content, b"first version")
            original_gets[0].assert_called_once_with(url, timeout=3)
            cached = output / "cached.json"
            cached.write_bytes(b"previously observed cache")
            with reader.cached_get(cached) as stream:
                self.assertEqual(stream.read(), b"previously observed cache")
            self.assertEqual(len(report["requests"]), 1)
            reader._session = reader._init_session()
            self.assertIs(reader._session.get(url), responses[1])
            self.assertIs(reader._session.get(url), responses[2])
            reader._session = reader._init_session()
            with self.assertRaisesRegex(SourceStopped, "request_budget_reached"):
                reader._session.get(url)
            original_gets[-1].assert_not_called()
            self.assertEqual(sum(original.call_count for original in original_gets), 3)
            entries = report["requests"]
            self.assertEqual(len(entries), 3)
            for ordinal, entry in enumerate(entries, 1):
                self._assert_capture_receipt(output, entry, ordinal)
                self.assertEqual(entry["url"], url)
                self.assertEqual((output / entry["response"]["path"]).read_bytes(), responses[ordinal - 1].content)
            self.assertNotEqual(entries[0]["response"]["path"], entries[1]["response"]["path"])
            self.assertEqual(entries[1]["response"], entries[2]["response"])
            self.assertEqual(len(list((output / "cache" / "_responses").glob("*.bin"))), 2)

    def test_capture_failure_receipts_preserve_received_bodies_and_complete_interrupted_attempts(self):
        from types import SimpleNamespace
        from app.data_service.soccerdata_collect import RequestCapture, SourceStopped

        cases = [
            ("non_http", SimpleNamespace(status_code=0, content=b"transport diagnostics"), "non_http"),
            ("forbidden", SimpleNamespace(status_code=403, content=b"access denied"), "http_403"),
            ("transport", OSError("synthetic connection failure"), "transport_error"),
            ("deadline", SourceStopped("provider_deadline_reached"), "provider_deadline_reached"),
        ]
        for label, outcome, reason in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                report = {"requests": []}
                capture = RequestCapture(report, output, 1)
                original = Mock(side_effect=outcome) if isinstance(outcome, BaseException) else Mock(return_value=outcome)
                with self.assertRaisesRegex(SourceStopped, reason):
                    capture.wrap_get(original)("https://example.invalid/source")
                original.assert_called_once()
                self.assertEqual(len(report["requests"]), 1)
                entry = report["requests"][0]
                self._assert_capture_receipt(output, entry, 1)
                if isinstance(outcome, BaseException):
                    self.assertNotIn("response", entry)
                else:
                    self.assertEqual(entry["http_status"], outcome.status_code)
                    self.assertEqual((output / entry["response"]["path"]).read_bytes(), outcome.content)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report = {"requests": []}
            capture = RequestCapture(report, output, 1)
            blocked = output / "cache" / "_responses"
            blocked.parent.mkdir(parents=True, exist_ok=True)
            if blocked.is_dir():
                blocked.rmdir()
            blocked.write_bytes(b"occupied capture destination")
            original = Mock(return_value=SimpleNamespace(status_code=200, content=b"received response"))
            with self.assertRaises(SourceStopped):
                capture.wrap_get(original)("https://example.invalid/source")
            entry = report["requests"][0]
            self._assert_capture_receipt(output, entry, 1)
            self.assertNotIn("response", entry)
            self.assertEqual(blocked.read_bytes(), b"occupied capture destination")

    def test_browser_capture_separates_dom_and_parser_output_and_keeps_blocked_pages(self):
        from types import SimpleNamespace
        from app.data_service.soccerdata_collect import RequestCapture, SourceStopped

        with tempfile.TemporaryDirectory() as directory, patch("app.data_service.soccerdata_collect.time.sleep"):
            output = Path(directory)
            report = {"requests": []}
            capture = RequestCapture(report, output, 8)
            driver = SimpleNamespace(get=Mock(), page_source="<html>original DOM</html>",
                                     execute_script=Mock(return_value={"shots": [1, 2]}))
            reader = SimpleNamespace(_driver=driver, _validate_page=Mock(return_value="<html>validated body</html>"))
            target = output / "cache" / "page.html"
            with capture.browser_download(reader, "https://example.invalid/page", filepath=target) as stream:
                first = stream.read()
            self.assertEqual(first, b"<html>validated body</html>")
            self.assertEqual(target.read_bytes(), first)
            with capture.browser_download(reader, "https://example.invalid/page", var="matchCentreData") as stream:
                derived = stream.read()
            self.assertEqual(json.loads(derived), {"shots": [1, 2]})
            reader._validate_page.assert_called_once()
            driver.execute_script.assert_called_once_with("return matchCentreData")
            for ordinal, entry in enumerate(report["requests"], 1):
                self._assert_capture_receipt(output, entry, ordinal)
                self.assertNotIn("http_status", entry)
                self.assertEqual((output / entry["response"]["path"]).read_bytes(), b"<html>original DOM</html>")
            self.assertEqual((output / report["requests"][0]["parser_payload"]["path"]).read_bytes(), first)
            self.assertEqual((output / report["requests"][1]["parser_payload"]["path"]).read_bytes(), derived)

            for label, page, reason in [
                ("verification", "<html>Just a moment...</html>", "human_verification"),
                ("forbidden", '{"reason":"Forbidden"}', "browser_forbidden"),
                ("invalid", "<html>unusable page</html>", "invalid_browser_page"),
            ]:
                with self.subTest(label=label):
                    driver.page_source = page
                    reader._validate_page.side_effect = ValueError("synthetic invalid page")
                    destination = output / "cache" / (label + ".json")
                    with self.assertRaisesRegex(SourceStopped, reason):
                        capture.browser_download(reader, "https://example.invalid/" + label, filepath=destination)
                    entry = report["requests"][-1]
                    self._assert_capture_receipt(output, entry, len(report["requests"]))
                    self.assertEqual((output / entry["response"]["path"]).read_bytes(), page.encode())
                    self.assertNotIn("parser_payload", entry)
                    self.assertNotIn("http_status", entry)
                    self.assertFalse(destination.exists())

    @staticmethod
    def _shot_fixture():
        import pandas as pd

        name = "Alex O&#039;Neil"
        payload = {
            "rosters": {
                "h": {
                    "7000": {"id": "7000", "player_id": "101", "team_id": "1", "player": "Shooter"},
                    "7001": {"id": "7001", "player_id": "201", "team_id": "1", "player": name},
                    "7002": {"id": "7002", "player_id": "201", "team_id": "1", "player": name},
                },
                "a": {"8001": {"id": "8001", "player_id": "301", "team_id": "2", "player": name}},
            },
            "shots": {"h": [
                {"id": "11", "match_id": "10", "h_a": "h", "season": "2026", "player_id": "101", "situation": "Penalty", "player_assisted": name},
                {"id": "12", "match_id": "10", "h_a": "h", "season": "2026", "player_id": "101", "situation": "OpenPlay", "player_assisted": None},
            ], "a": []},
        }
        frame = pd.DataFrame({
            "game_id": pd.array([10, 10], dtype="Int64"),
            "shot_id": pd.array([11, 12], dtype="Int64"),
            "team_id": pd.array([1, 1], dtype="Int64"),
            "player_id": pd.array([101, 101], dtype="Int64"),
            "season_id": pd.array([2026, 2026], dtype="Int64"),
            "situation": pd.array([pd.NA, "Open Play"], dtype="string"),
            "assist_player": pd.array(["Alex O'Neil", pd.NA], dtype="string"),
            "assist_player_id": pd.array([8001, pd.NA], dtype="Int64"),
            "xg": pd.array([0.76, 0.12], dtype="Float64"),
        })
        frame.index = pd.MultiIndex.from_tuples(
            [("league", "2627", "game", "home", "Shooter")] * 2,
            names=["league", "season", "game", "team", "player"])
        return frame, payload

    @classmethod
    def setUpClass(cls):
        directory = cls.enterClassContext(tempfile.TemporaryDirectory())
        cls.enterClassContext(
            patch.dict(
                os.environ,
                {"SOCCERDATA_DIR": directory, "SOCCERDATA_LOGLEVEL": "ERROR"},
            )
        )

    def test_espn_compatibility_aligns_team_ids_without_rewriting_original_payload(
        self,
    ):
        from app.data_service.soccerdata_collect import espn_legacy_summary

        payload = {
            "boxscore": {"teams": [{"team": {"id": "1"}}, {"team": {"id": "2"}}]},
            "rosters": [
                {"team": {"id": "2"}, "homeAway": "away"},
                {"team": {"id": "1"}, "homeAway": "home"},
            ],
        }
        converted = espn_legacy_summary(payload)
        self.assertEqual(converted["boxscore"]["form"][0]["team"]["id"], "1")
        self.assertEqual(converted["rosters"][0]["homeAway"], "home")
        self.assertNotIn("form", payload["boxscore"])
        payload["rosters"][0]["homeAway"] = "home"
        with self.assertRaises(ValueError):
            espn_legacy_summary(payload)

    @unittest.skipUnless(
        importlib.util.find_spec("soccerdata"), "optional soccerdata package"
    )
    def test_browser_wrapper_preserves_source_configuration_identity(self):
        import soccerdata as sd
        from app.data_service.soccerdata_collect import anonymous_browser_reader

        for name in ["FBref", "SoFIFA", "WhoScored"]:
            original = getattr(sd, name)
            wrapped = anonymous_browser_reader(
                original, lambda self: None, lambda self, *a, **k: None
            )
            self.assertEqual(wrapped.available_leagues(), original.available_leagues())
            self.assertEqual(wrapped.__name__, original.__name__)

    @unittest.skipUnless(
        importlib.util.find_spec("soccerdata"), "optional soccerdata package"
    )
    def test_explicit_start_years_do_not_duplicate_2020_21_or_drop_2021_22(self):
        from soccerdata._common import SeasonCode

        actual = [
            SeasonCode.MULTI_YEAR.parse(s) for s in explicit_seasons([2020, 2021, 2022])
        ]
        self.assertEqual(actual, ["2021", "2122", "2223"])
        self.assertEqual(len(set(actual)), 3)

    @unittest.skipUnless(importlib.util.find_spec("pandas"), "optional pandas package")
    def test_export_preserves_duplicate_multilevel_columns_indices_and_missing_values(
        self,
    ):
        import pandas as pd

        frame = pd.DataFrame(
            [[1, None]],
            columns=pd.MultiIndex.from_tuples([("pass", "count"), ("pass", "count")]),
            index=pd.Index(["player-1"], name="player_id"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.csv"
            export_frame(frame, path)
            with path.open(newline="") as f:
                rows = list(csv.reader(f))
            self.assertEqual(len(rows[0]), len(set(rows[0])))
            self.assertEqual(rows[1][0], "player-1")
            self.assertEqual(rows[1][-1], "")
            schema = json.loads(path.with_suffix(".schema.json").read_text())
            self.assertEqual(schema["columns"], [["pass", "count"], ["pass", "count"]])

    @unittest.skipUnless(importlib.util.find_spec("pandas"), "optional pandas package")
    def test_understat_player_matches_keep_distinct_games_and_reject_conflicts(self):
        import pandas as pd
        from app.data_service.soccerdata_collect import unique_understat_player_matches

        frame = pd.DataFrame([
            ["league", "2627", "game-1", "team", "player", 1, 2, 3, 40, 0.25],
            ["league", "2627", "game-1", "team", "player", 1, 2, 3, 40, 0.25],
            ["league", "2627", "game-2", "team", "player", 4, 2, 3, 40, 0.25],
        ], columns=["league", "season", "game", "team", "player", "game_id", "team_id", "player_id", "minutes", "xg"])
        frame = frame.set_index(["league", "season", "game", "team", "player"])
        unique, audit = unique_understat_player_matches(frame)
        pd.testing.assert_frame_equal(unique, frame.iloc[[0, 2]])
        self.assertEqual(len(frame), 3)
        self.assertEqual(audit["identical_duplicates_removed"], 1)
        self.assertEqual(audit["unique_rows"], 2)
        conflict = frame.copy()
        conflict.iloc[1, conflict.columns.get_loc("minutes")] = 41
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            unique_understat_player_matches(conflict)
        quarantined, audit = unique_understat_player_matches(conflict, quarantine_conflicts=True)
        pd.testing.assert_frame_equal(quarantined, conflict.iloc[[2]])
        self.assertEqual(audit["quarantined_raw_rows"], 2)
        self.assertEqual(audit["quarantined_unique_rows"], 2)
        self.assertEqual(len(audit["conflicting_identities"]), 1)
        missing = frame.copy()
        missing.iloc[0, missing.columns.get_loc("player_id")] = None
        with self.assertRaisesRegex(ValueError, "identity is missing"):
            unique_understat_player_matches(missing)

    @unittest.skipUnless(importlib.util.find_spec("pandas"), "optional pandas package")
    def test_understat_shot_fields_preserve_native_rows_and_stable_assist_identity(self):
        import pandas as pd

        frame, payload = self._shot_fixture()
        original = frame.copy(deep=True)
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            path = cache / "match_10.json"
            path.write_text(json.dumps(payload))
            raw = path.read_bytes()
            fixed, audit = normalize_understat_shots(frame, cache)
            self.assertEqual(fixed["situation"].tolist(), ["Penalty", "Open Play"])
            self.assertEqual(fixed["assist_player_id"].iloc[0], 201)
            self.assertTrue(pd.isna(fixed["assist_player_id"].iloc[1]))
            self.assertEqual(audit["restored_situations"], 1)
            self.assertEqual(audit["corrected_assist_ids"], 1)
            self.assertEqual(audit["source_association_unspecified"], 1)
            pd.testing.assert_frame_equal(frame, original)
            pd.testing.assert_frame_equal(fixed.drop(columns=["situation", "assist_player_id"]),
                                          original.drop(columns=["situation", "assist_player_id"]))
            self.assertEqual(fixed.dtypes.tolist(), frame.dtypes.tolist())
            self.assertEqual(path.read_bytes(), raw)
            repeated, repeated_audit = normalize_understat_shots(fixed, cache)
            pd.testing.assert_frame_equal(repeated, fixed)
            self.assertEqual(repeated_audit["restored_situations"], 0)
            self.assertEqual(repeated_audit["corrected_assist_ids"], 0)

            for kind in ("ambiguous", "unmatched"):
                with self.subTest(kind=kind):
                    source, sample = copy.deepcopy(payload), frame.copy()
                    if kind == "ambiguous":
                        source["rosters"]["h"]["7003"] = {
                            "id": "7003", "player_id": "202", "team_id": "1", "player": "Alex O&#039;Neil"}
                    else:
                        source["shots"]["h"][0]["player_assisted"] = "Unknown name"
                        sample.iloc[0, sample.columns.get_loc("assist_player")] = "Unknown name"
                    path.write_text(json.dumps(source))
                    unknown, evidence = normalize_understat_shots(sample, cache)
                    self.assertTrue(pd.isna(unknown["assist_player_id"].iloc[0]))
                    self.assertEqual(evidence[f"{kind}_assist_names"], 1)
                    self.assertEqual(evidence["cleared_unverifiable_assist_ids"], 1)

    @unittest.skipUnless(importlib.util.find_spec("pandas"), "optional pandas package")
    def test_understat_shot_mismatches_are_not_silently_repaired(self):
        import pandas as pd

        frame, payload = self._shot_fixture()
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            (cache / "match_10.json").write_text(json.dumps(payload))
            floating = frame.copy()
            floating["assist_player_id"] = pd.array([pd.NA, pd.NA], dtype="Float64")
            with self.assertRaisesRegex(ValueError, "nullable Int64"):
                normalize_understat_shots(floating, cache)
            for field in ("game_id", "team_id", "player_id", "season_id"):
                with self.subTest(frame_identity=field):
                    sample = frame.copy()
                    sample.iloc[0, sample.columns.get_loc(field)] = 99
                    (cache / "match_10.json").write_text(json.dumps(payload))
                    with self.assertRaises(ValueError):
                        normalize_understat_shots(sample, cache)
            for kind in ("side", "game", "duplicate", "unknown-category", "mixed-roster", "shooter-roster", "label", "assist-name", "assist-id", "null-association"):
                with self.subTest(mismatch=kind):
                    source, sample = copy.deepcopy(payload), frame.copy()
                    if kind == "side": source["shots"]["h"][0]["h_a"] = "a"
                    if kind == "game": source["shots"]["h"][0]["match_id"] = "99"
                    if kind == "duplicate": source["shots"]["h"][1]["id"] = "11"
                    if kind == "unknown-category": source["shots"]["h"][0]["situation"] = "UndocumentedCategory"
                    if kind == "mixed-roster": source["rosters"]["h"]["7001"]["team_id"] = "2"
                    if kind == "shooter-roster":
                        source["shots"]["h"][0]["player_id"] = "999"
                        sample.iloc[0, sample.columns.get_loc("player_id")] = 999
                    if kind == "label": sample.iloc[0, sample.columns.get_loc("situation")] = "Open Play"
                    if kind == "assist-name": sample.iloc[0, sample.columns.get_loc("assist_player")] = "Different name"
                    if kind == "assist-id": sample.iloc[0, sample.columns.get_loc("assist_player_id")] = 9999
                    if kind == "null-association": sample.iloc[1, sample.columns.get_loc("assist_player_id")] = 7001
                    (cache / "match_10.json").write_text(json.dumps(source))
                    with self.assertRaises(ValueError):
                        normalize_understat_shots(sample, cache)


class TestWyscoutCollection(unittest.TestCase):
    def session(self, name, body=b"[]", digest=None, cached=False):
        article = Mock()
        article.json.return_value = [{"id": 1}]
        details = Mock()
        details.json.return_value = {
            "files": [
                {
                    "id": 2,
                    "name": name,
                    "computed_md5": digest or hashlib.md5(body).hexdigest(),
                }
            ]
        }
        empty = Mock()
        empty.json.return_value = []
        download = Mock()
        download.iter_content.return_value = [body]
        download.__enter__ = Mock(return_value=download)
        download.__exit__ = Mock(return_value=False)
        session = Mock()
        session.get.side_effect = (
            [article, details, empty] if cached else [article, details, download, empty]
        )
        session.__enter__ = Mock(return_value=session)
        session.__exit__ = Mock(return_value=False)
        return session

    def test_untrusted_names_are_rejected_before_download(self):
        for name in [
            "../escape.json",
            "ABSOLUTE",
            "..\\escape.json",
            "C:\\escape.json",
            "nested/file.json",
            ".",
            "..",
            "",
            None,
            "a\x00.json",
            "manifest.json",
            "NUL",
            "file.json.",
        ]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "source"
                session = self.session(
                    str(Path(tmp) / "escape.json") if name == "ABSOLUTE" else name
                )
                with patch(
                    "app.data_service.wyscout.requests.Session", return_value=session
                ):
                    with self.assertRaises(ValueError):
                        collect(root)
                self.assertEqual(session.get.call_count, 2)
                self.assertEqual(list(root.iterdir()), [])
                self.assertFalse((Path(tmp) / "escape.json").exists())

    def test_download_ignores_old_part_symlink_and_reuses_valid_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            root.mkdir()
            sentinel = Path(tmp) / "sentinel"
            sentinel.write_bytes(b"unchanged")
            (root / "events.json.part").symlink_to(sentinel)
            session = self.session("events.json")
            with patch(
                "app.data_service.wyscout.requests.Session", return_value=session
            ):
                collect(root)
            self.assertEqual(sentinel.read_bytes(), b"unchanged")
            self.assertEqual((root / "events.json").read_bytes(), b"[]")
            session = self.session("events.json", cached=True)
            with patch(
                "app.data_service.wyscout.requests.Session", return_value=session
            ):
                collect(root)
            self.assertEqual(session.get.call_count, 3)

    def test_bad_digest_leaves_no_completed_or_temporary_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            with patch(
                "app.data_service.wyscout.requests.Session",
                return_value=self.session("events.json", digest="0" * 32),
            ):
                with self.assertRaises(ValueError):
                    collect(root)
            self.assertEqual(list(root.iterdir()), [])

    def test_destination_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            root.mkdir()
            (root / "events.json").symlink_to(Path(tmp) / "outside.json")
            with patch(
                "app.data_service.wyscout.requests.Session",
                return_value=self.session("events.json"),
            ):
                with self.assertRaises(ValueError):
                    collect(root)
            self.assertFalse((Path(tmp) / "outside.json").exists())


if __name__ == "__main__":
    unittest.main()
