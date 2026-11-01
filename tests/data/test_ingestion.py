from pathlib import Path
from unittest.mock import MagicMock, patch
import hashlib
import importlib.util
import json
import os
import sqlite3
import tempfile
import unittest

from app.analytics.scouting.source_findings import consecutive_pairs, correlation, possession_eligible, restore_situations
from app.data_service import fifa_reports
from app.data_service.build_warehouse import build
from app.data_service.fifa_reports import parse_report, report_links, shape_metrics
from app.data_service.repository_data import ingest as ingest_repository
from app.data_service.soccerdata_ingest import ingest as ingest_soccerdata
from app.data_service.tactical_evidence import validate_card


class TestSoccerdataIngest(unittest.TestCase):
    def test_gzip_logical_sources_keep_checksums_and_resume_without_network(self):
        import gzip
        from app.data_service.archive import (
            asset_file_size, collect_repository, open_asset_file, read_asset_json,
            resolve_asset_path, verify_asset_file,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logical = root / 'events.json'
            raw = '{"label":"足球", "events":[1,2]}\r\n'.encode()
            compressed = root / 'events.json.gz'
            logical.write_bytes(raw)
            compressed.write_bytes(gzip.compress(b'stale', mtime=0))
            index = {'events.json': {'path': 'events.json', 'bytes': len(raw),
                                    'sha256': hashlib.sha256(raw).hexdigest()}}
            self.assertEqual(resolve_asset_path(logical), logical)
            self.assertEqual(read_asset_json(root, 'events.json', index), json.loads(raw))
            with open_asset_file(compressed, 'rb') as handle:
                self.assertEqual(handle.read(), compressed.read_bytes())
            compressed.write_bytes(gzip.compress(raw, mtime=0))
            logical.unlink()
            self.assertEqual(resolve_asset_path(logical), compressed)
            self.assertEqual(verify_asset_file(root, 'events.json', index), logical)
            self.assertEqual(read_asset_json(root, 'events.json', index), json.loads(raw))
            self.assertEqual(asset_file_size(logical), len(raw))
            self.assertEqual(asset_file_size(logical, expected_bytes=len(raw)), len(raw))
            manifest = {'complete': True, 'files': list(index.values())}
            (root / 'manifest.json').write_text(json.dumps(manifest))
            with patch('app.data_service.archive.requests.Session', side_effect=AssertionError('must resume offline')):
                self.assertEqual(collect_repository('statsbomb', root), manifest)
            wrong_size = {'events.json': {**index['events.json'], 'bytes': len(raw)+1}}
            with self.assertRaisesRegex(ValueError, 'checksum'):
                verify_asset_file(root, 'events.json', wrong_size)
            split = len(raw)//2
            compressed.write_bytes(gzip.compress(raw[:split], mtime=0) + gzip.compress(raw[split:], mtime=0))
            self.assertEqual(asset_file_size(logical), len(raw))
            self.assertEqual(read_asset_json(root, 'events.json', index), json.loads(raw))
            compressed.write_bytes(compressed.read_bytes()[:-4])
            with self.assertRaises((EOFError, OSError)):
                read_asset_json(root, 'events.json', index)
            compressed.unlink()
            target = root / 'source.gz'
            target.write_bytes(gzip.compress(raw, mtime=0))
            compressed.symlink_to(target)
            with self.assertRaisesRegex(ValueError, 'symlink'):
                resolve_asset_path(logical)
            folder = root / 'actual'
            folder.mkdir()
            (folder / 'sample.json.gz').write_bytes(gzip.compress(raw, mtime=0))
            (root / 'alias').symlink_to(folder, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'symlink'):
                resolve_asset_path(root / 'alias/sample.json')
            missing = root / 'absent.json'
            self.assertEqual(resolve_asset_path(missing), missing)
            with self.assertRaises(FileNotFoundError):
                open_asset_file(missing)

    def test_gzip_sources_rebuild_identical_raw_assets_sql_views_and_tracking(self):
        import gzip
        import io
        import zlib
        from contextlib import redirect_stdout
        from app.analytics.tactics.receiving_space import _skillcorner_prefix
        from app.data_service.archive import read_asset_json, verify_asset_file
        from app.data_service.tracking_frames import skillcorner_frames

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = root / 'sources'
            skillcorner = sources / 'skillcorner'
            relative = 'data/matches/1874553/1874553'
            metadata = {'id': 1874553, 'players': [{'id': 1, 'team_id': 10, 'short_name': 'A'}],
                        'pitch_length': 105, 'pitch_width': 68}
            frame = {'frame': 1, 'period': 1, 'timestamp': '00:00:01.0',
                     'player_data': [{'player_id': 1, 'x': 1.5, 'y': -2.5, 'is_detected': True}],
                     'ball_data': {'x': 0, 'y': 0, 'z': 0, 'is_detected': True}}
            bodies = {
                sources / 'statsbomb/data/events/1.json': json.dumps([{'id': 'event-1', 'type': {'name': 'Pass'}}]).encode(),
                sources / 'custom/rows.csv': b'name,value\r\n"Andr\xe9, B",1\r\n',
                skillcorner / (relative+'_match.json'): json.dumps(metadata).encode(),
                skillcorner / (relative+'_tracking_extrapolated.jsonl'): (json.dumps(frame)+'\n').encode(),
                skillcorner / 'data/bodypose/sample_1.jsonl.gz': gzip.compress(b'{"native":true}\n', mtime=0),
                root / 'scouting/assessments/team-data.json': b'{"teams":[{"id":1}]}',
            }
            entries = [{'path': str(path.relative_to(skillcorner)), 'bytes': len(body),
                        'sha256': hashlib.sha256(body).hexdigest()}
                       for path, body in bodies.items() if path.is_relative_to(skillcorner)]
            manifest = {'complete': True, 'files': entries}
            bodies[skillcorner / 'manifest.json'] = json.dumps(manifest).encode()
            for path, body in bodies.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(body)
                if path.name != 'manifest.json' and path.suffix != '.gz':
                    Path(str(path)+'.gz').write_bytes(gzip.compress(body, mtime=0))
            # A stale transport is ignored while the authoritative original exists.
            csv_path = sources / 'custom/rows.csv'
            Path(str(csv_path)+'.gz').write_bytes(b'not a gzip stream')
            tracking = skillcorner / (relative+'_tracking_extrapolated.jsonl')
            original_frames = list(skillcorner_frames(tracking, metadata))
            prefix_before = _skillcorner_prefix(tracking.parent)
            before, after = root / 'plain.sqlite3', root / 'gzip.sqlite3'
            with redirect_stdout(io.StringIO()):
                build(before, sources)
            for path, body in bodies.items():
                if path.name != 'manifest.json' and path.suffix != '.gz':
                    Path(str(path)+'.gz').write_bytes(gzip.compress(body, mtime=0))
                    path.unlink()
            index = {entry['path']: entry for entry in entries}
            verified = verify_asset_file(skillcorner, relative+'_tracking_extrapolated.jsonl', index)
            self.assertEqual(verified, tracking)
            self.assertEqual(read_asset_json(skillcorner, relative+'_match.json', index), metadata)
            self.assertEqual(list(skillcorner_frames(verified, metadata)), original_frames)
            self.assertEqual(_skillcorner_prefix(tracking.parent), prefix_before)
            with redirect_stdout(io.StringIO()):
                build(after, sources)
            inventories = []
            for database in (before, after):
                with sqlite3.connect(database) as connection:
                    assets = connection.execute('SELECT d.name,a.id,a.path,a.sha256,a.bytes FROM assets a JOIN datasets d ON d.id=a.dataset_id ORDER BY d.name,a.path').fetchall()
                    self.assertEqual(len(assets), len(bodies))
                    self.assertEqual(sum(row[4] for row in assets), sum(len(body) for body in bodies.values()))
                    inventories.append([(dataset, path, sha, size) for dataset, _, path, sha, size in assets])
                    for dataset, aid, path, sha, size in assets:
                        raw = b''.join(zlib.decompress(row[0]) for row in connection.execute('SELECT content FROM raw_chunks WHERE asset_id=? ORDER BY ordinal', (aid,)))
                        source_root = root if dataset == 'scouting' else sources
                        self.assertEqual(raw, bodies[source_root / dataset / path])
                        self.assertEqual(hashlib.sha256(raw).hexdigest(), sha)
                        self.assertEqual(len(raw), size)
                    self.assertEqual(connection.execute('SELECT event_id FROM statsbomb_events').fetchall(), [('event-1',)])
                    self.assertEqual(connection.execute('SELECT frame,player_id,x_m FROM skillcorner_tracking').fetchall(), [(1, 1, 1.5)])
                    self.assertEqual(connection.execute('SELECT name,value FROM custom_csv_rows').fetchall(), [('André, B', '1')])
            self.assertEqual(inventories[0], inventories[1])

    @staticmethod
    def _sdk_run(root, name, *, rows=1, first_header="value"):
        folder = root / name
        tables = folder / "tables"
        tables.mkdir(parents=True)
        frames = []
        for method, header, count in [("schedule", first_header, rows), ("details", "detail", 1)]:
            path = tables / (method + ".csv")
            path.write_text("id," + header + "\n" + "".join(f"{i},old\n" for i in range(count)))
            path.with_suffix(".schema.json").write_text("{}")
            frames.append({"path": str(path.resolve()), "method": method, "rows": count})
        cache = folder / "cache"
        cache.mkdir()
        (cache / "payload.json").write_text('{"version": "old"}')
        report = {
            "provider": "Understat", "soccerdata_version": "test", "status": "collected",
            "observed_at": "2026-10-09", "leagues": ["ENG-Premier League"],
            "seasons": ["2026-2027"], "controls": {}, "errors": [], "tables": frames,
        }
        (folder / "report.json").write_text(json.dumps(report))
        return folder, report

    @staticmethod
    def _warehouse_state(database):
        # Compare contents and schema, including archived bytes and visibility to
        # readers that query raw documents without consulting the SDK ledger.
        with sqlite3.connect(database) as connection:
            schema = connection.execute(
                "SELECT type,name,sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name"
            ).fetchall()
            tables = {
                name: connection.execute('SELECT * FROM "' + name.replace('"', '""') + '" ORDER BY rowid').fetchall()
                for kind, name, _ in schema if kind == "table"
            }
        return schema, tables

    @staticmethod
    def _ingest_source(database, root, general):
        return build(database, root.parent) if general else ingest_soccerdata(database, root)

    @staticmethod
    def _response_capture(folder, report):
        report["response_capture"] = {
            "schema": "SoccerAnalytics.SourceResponseCapture.v1",
            "body_directory": "cache/_responses",
        }
        payloads = [b"response version one", b"response version two"]
        bodies = {}
        for payload in payloads:
            digest = hashlib.sha256(payload).hexdigest()
            relative = "cache/_responses/" + digest + ".bin"
            path = folder / relative
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(payload)
            bodies[relative] = payload
        report["requests"] = []
        for ordinal, payload in enumerate([payloads[0], payloads[1], payloads[0]], 1):
            digest = hashlib.sha256(payload).hexdigest()
            report["requests"].append({
                "ordinal": ordinal, "url": "https://example.test/match/10",
                "transport": "soccerdata_native_tls_client", "http_status": 200,
                "started_at": "2026-10-23T10:00:02Z", "completed_at": "2026-10-23T10:00:01Z",
                "response_observed_at": "2026-10-23T10:00:01Z", "elapsed_seconds": .1,
                "response": {"path": "cache/_responses/" + digest + ".bin",
                             "sha256": digest, "bytes": len(payload), "kind": "sdk_response_content"},
            })
        (folder / "report.json").write_text(json.dumps(report))
        return bodies

    def test_response_versions_are_archived_once_without_becoming_sdk_documents(self):
        import zlib

        for general in (False, True):
            with self.subTest(general=general), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "sources" / "soccerdata"
                database = Path(tmp) / "warehouse.sqlite3"
                folder, report = self._sdk_run(root, "Understat-current")
                bodies = self._response_capture(folder, report)
                self._ingest_source(database, root, general)
                with sqlite3.connect(database) as connection:
                    assets = connection.execute(
                        "SELECT a.id,a.path,a.sha256,a.bytes,d.kind FROM assets a LEFT JOIN documents d ON d.asset_id=a.id WHERE a.path LIKE '%/_responses/%' ORDER BY a.path"
                    ).fetchall()
                    self.assertEqual(len(assets), 2)
                    for aid, path, digest, size, kind in assets:
                        payload = bodies[path.removeprefix(folder.name + "/")]
                        self.assertEqual((digest, size, kind), (hashlib.sha256(payload).hexdigest(), len(payload), None))
                        chunks = connection.execute(
                            "SELECT content FROM raw_chunks WHERE asset_id=? ORDER BY ordinal", (aid,)
                        ).fetchall()
                        self.assertEqual(b"".join(zlib.decompress(chunk[0]) for chunk in chunks), payload)
                    self.assertEqual(connection.execute(
                        "SELECT count(*) FROM documents WHERE kind='soccerdata_original_json'"
                    ).fetchone()[0], 1)
                    archived = json.loads(connection.execute(
                        "SELECT payload FROM documents WHERE kind='soccerdata_access_report'"
                    ).fetchone()[0])
                    self.assertEqual(archived["requests"], report["requests"])
                previous = self._warehouse_state(database)
                self._ingest_source(database, root, general)
                self.assertEqual(self._warehouse_state(database), previous)

    def test_invalid_response_receipts_leave_the_previous_database_unchanged(self):
        original_rglob = Path.rglob
        for general in (False, True):
            for failure in ("missing", "tampered", "size", "path", "timestamp", "duration", "ordinal",
                            "kind", "schema", "directory", "missing_body", "not_archived"):
                with self.subTest(general=general, failure=failure), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp) / "sources" / "soccerdata"
                    database = Path(tmp) / "warehouse.sqlite3"
                    self._sdk_run(root, "A-old")
                    self._ingest_source(database, root, general)
                    previous = self._warehouse_state(database)
                    folder, report = self._sdk_run(root, "B-new")
                    self._response_capture(folder, report)
                    entry = report["requests"][0]
                    body = entry["response"]
                    if failure == "missing":
                        (folder / body["path"]).unlink()
                    elif failure == "tampered":
                        (folder / body["path"]).write_bytes(b"changed after capture")
                    elif failure == "size":
                        for request in report["requests"]:
                            if request["response"]["path"] == body["path"]:
                                request["response"]["bytes"] += 1
                    elif failure == "path":
                        body["path"] = "cache/_responses/../" + Path(body["path"]).name
                    elif failure == "timestamp":
                        entry["started_at"] = "2026-10-23T10:00:02"
                    elif failure == "duration":
                        entry["elapsed_seconds"] = float("nan")
                    elif failure == "ordinal":
                        report["requests"][1]["ordinal"] = True
                    elif failure == "kind":
                        body["kind"] = "browser_extracted_payload_utf8"
                    elif failure == "schema":
                        report["response_capture"]["schema"] = "unrecognized"
                    elif failure == "directory":
                        report["response_capture"]["body_directory"] = "cache"
                    elif failure == "missing_body":
                        del entry["response"]
                    (folder / "report.json").write_text(json.dumps(report))

                    def incomplete_scan(path, pattern):
                        return (item for item in original_rglob(path, pattern)
                                if failure != "not_archived" or item.suffix != ".bin")

                    with patch.object(Path, "rglob", incomplete_scan):
                        with self.assertRaises(ValueError):
                            self._ingest_source(database, root, general)
                    self.assertEqual(self._warehouse_state(database), previous)

    def test_response_diagnostics_and_empty_cache_replay_keep_their_original_meaning(self):
        for outcome in ("transport_error", "capture_error", "browser_error", "browser_success", "cache_only"):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "soccerdata"
                database = Path(tmp) / "warehouse.sqlite3"
                folder, report = self._sdk_run(root, "Understat-current")
                self._response_capture(folder, report)
                entry = report["requests"][0]
                report["requests"] = [entry]
                if outcome == "cache_only":
                    report["requests"] = []
                    for path in (folder / "cache" / "_responses").iterdir():
                        path.unlink()
                elif outcome in ("transport_error", "capture_error"):
                    del entry["response"]
                    entry[outcome] = "OSError"
                    if outcome == "transport_error":
                        entry["error"] = entry.pop("transport_error")
                        del entry["response_observed_at"]
                        del entry["http_status"]
                    report.update(tables=[], status="unavailable_in_probe", errors=[{"type": "SourceStopped"}])
                else:
                    entry["transport"] = "normal_isolated_chrome"
                    entry.pop("http_status")
                    entry["response"]["kind"] = "browser_page_source_utf8"
                    if outcome == "browser_error":
                        entry["error"] = "SourceStopped"
                        report.update(tables=[], status="unavailable_in_probe", errors=[{"type": "SourceStopped"}])
                    else:
                        entry["parser_payload"] = {**report["requests"][0]["response"],
                                                   "kind": "browser_extracted_payload_utf8"}
                (folder / "report.json").write_text(json.dumps(report))
                ingest_soccerdata(database, root)
                with sqlite3.connect(database) as connection:
                    archived = json.loads(connection.execute(
                        "SELECT payload FROM documents WHERE kind='soccerdata_access_report'"
                    ).fetchone()[0])
                    self.assertEqual(archived["requests"], report["requests"])
                    family = connection.execute("SELECT family FROM datasets").fetchone()[0]
                    self.assertEqual(family, "real_source_adapter" if report["tables"] else "access_diagnostics")

    def test_failed_sdk_run_rolls_back_frames_and_can_retry(self):
        from app.data_service.warehouse import Warehouse

        original_asset = Warehouse.asset
        for general in (False, True):
            for failure in ("row_count", "cache_read"):
                with self.subTest(general=general, failure=failure), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp) / "sources" / "soccerdata"
                    database = Path(tmp) / "warehouse.sqlite3"
                    self._sdk_run(root, "A-old")
                    self._ingest_source(database, root, general)
                    previous = self._warehouse_state(database)
                    # Cross the CSV importer's 20,000-row commit boundary before
                    # the second table fails; the new header also creates DDL.
                    folder, report = self._sdk_run(
                        root, "B-new", rows=20001 if failure == "row_count" else 2,
                        first_header="new_value",
                    )
                    if failure == "row_count":
                        report["tables"][1]["rows"] = 2
                        (folder / "report.json").write_text(json.dumps(report))
                    else:
                        (folder / "cache" / "z_late.json").write_text('{"late": true}')

                    def archive(db, dataset, path, relative_path):
                        if failure == "cache_read" and Path(path).name == "z_late.json":
                            raise OSError("synthetic late cache read failure")
                        return original_asset(db, dataset, path, relative_path)

                    with patch.object(Warehouse, "asset", archive):
                        with self.assertRaises(ValueError if failure == "row_count" else OSError):
                            self._ingest_source(database, root, general)
                    self.assertEqual(self._warehouse_state(database), previous)
                    self.assertTrue((folder / "tables" / "schedule.csv").is_file())
                    report["tables"][1]["rows"] = 1
                    (folder / "report.json").write_text(json.dumps(report))
                    self._ingest_source(database, root, general)
                    recovered = self._warehouse_state(database)
                    self.assertGreater(len(recovered[1]["assets"]), len(previous[1]["assets"]))
                    self.assertEqual(len(recovered[1]["soccerdata_imports"]), 4)
                    self._ingest_source(database, root, general)
                    self.assertEqual(self._warehouse_state(database), recovered)

    def test_failed_sdk_revision_keeps_previous_frames_and_caches(self):
        from app.data_service.warehouse import Warehouse

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "soccerdata"
            database = Path(tmp) / "warehouse.sqlite3"
            folder, report = self._sdk_run(root, "Understat-current")
            ingest_soccerdata(database, root)
            previous = self._warehouse_state(database)
            for frame in report["tables"]:
                path = Path(frame["path"])
                path.write_text(path.read_text().replace(",old\n", ",revised\n"))
            (folder / "cache" / "payload.json").write_text('{"version": "revised"}')
            (folder / "cache" / "z_late.json").write_text('{"late": true}')
            report["observed_at"] = "2026-10-10"
            (folder / "report.json").write_text(json.dumps(report))
            original_asset = Warehouse.asset

            def archive(db, dataset, path, relative_path):
                if Path(path).name == "z_late.json":
                    raise OSError("synthetic late revision failure")
                return original_asset(db, dataset, path, relative_path)

            with patch.object(Warehouse, "asset", archive):
                with self.assertRaisesRegex(OSError, "late revision"):
                    ingest_soccerdata(database, root)
            self.assertEqual(self._warehouse_state(database), previous)
            ingest_soccerdata(database, root)
            revised = self._warehouse_state(database)
            self.assertEqual(len(revised[1]["soccerdata_imports"]), 4)
            self.assertTrue(set(previous[1]["assets"]).issubset(revised[1]["assets"]))
            self.assertTrue(set(previous[1]["documents"]).issubset(revised[1]["documents"]))
            ingest_soccerdata(database, root)
            self.assertEqual(self._warehouse_state(database), revised)

    def test_sdk_rejects_incomplete_frames_and_invalid_receipt_counts(self):
        for failure in ("irregular_row", "boolean_count", "negative_count"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "soccerdata"
                database = Path(tmp) / "warehouse.sqlite3"
                self._sdk_run(root, "A-old")
                ingest_soccerdata(database, root)
                previous = self._warehouse_state(database)
                folder, report = self._sdk_run(root, "B-new")
                if failure == "irregular_row":
                    Path(report["tables"][0]["path"]).write_text("id,value\n1,old\n2,extra,column\n")
                    report["tables"][0]["rows"] = 2
                else:
                    report["tables"][0]["rows"] = True if failure == "boolean_count" else -1
                (folder / "report.json").write_text(json.dumps(report))
                with self.assertRaises(ValueError):
                    ingest_soccerdata(database, root)
                self.assertEqual(self._warehouse_state(database), previous)

    def test_sdk_ingest_keeps_public_cache_and_excludes_browser_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.check_public_cache(Path(tmp))

    def test_general_builder_uses_the_same_public_source_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.check_public_cache(Path(tmp), general=True)

    def test_private_symlink_aliases_are_rejected(self):
        for alias in [
            "frame",
            "schema",
            "cache",
            "report",
            "cache_directory",
            "run_directory",
            "hard_link",
        ]:
            with self.subTest(alias=alias), tempfile.TemporaryDirectory() as tmp:
                self.check_public_cache(Path(tmp), alias=alias)

    def test_general_custom_family_excludes_private_directories_and_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.check_public_cache(Path(tmp), general=True, custom=True)

    def test_general_builder_rejects_family_alias_to_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sources"
            private = root / "provider" / "runtime"
            private.mkdir(parents=True)
            (private / "private.log").write_text("test sentinel")
            (root / "alias").symlink_to(private, target_is_directory=True)
            database = Path(tmp) / "warehouse.sqlite3"
            with self.assertRaises(ValueError):
                build(database, root)
            with sqlite3.connect(database) as c:
                self.assertEqual(
                    c.execute("SELECT COUNT(*) FROM assets").fetchone()[0], 0
                )

    def test_general_builder_rejects_hard_link_to_private_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sources"
            private = root / "custom" / "runtime"
            private.mkdir(parents=True)
            original = private / "Cookies"
            original.write_text("test sentinel")
            os.link(original, root / "custom" / "cookies.bin")
            database = Path(tmp) / "warehouse.sqlite3"
            with self.assertRaises(ValueError):
                build(database, root)
            with sqlite3.connect(database) as c:
                self.assertEqual(
                    c.execute("SELECT COUNT(*) FROM assets").fetchone()[0], 0
                )

    def check_public_cache(self, tmp_path, general=False, alias=None, custom=False):
        root = tmp_path / "sources" / ("custom" if custom else "soccerdata")
        run = root / "ESPN"
        tables = run / "tables"
        tables.mkdir(parents=True)
        frame = tables / "schedule.csv"
        frame.write_text("id,value\n1,\n2,3\n")
        frame.with_suffix(".schema.json").write_text("{}")
        cache = run / "cache"
        cache.mkdir()
        (cache / "payload.json").write_text('{"events": []}')
        (cache / "invalid.json").write_text('{"events":')
        for parent, name in [
            ("isolated-browser-profile", "Cookies"),
            ("runtime", "private.log"),
        ]:
            private = run / parent
            private.mkdir()
            (private / name).write_text("private sentinel")
        report = {
            "provider": "ESPN",
            "soccerdata_version": "test",
            "status": "collected",
            "observed_at": "2026-10-09",
            "leagues": ["ENG-Premier League"],
            "seasons": ["2024-2025"],
            "controls": {},
            "errors": [],
            "tables": [
                {"path": str(frame.resolve()), "method": "read_schedule", "rows": 2}
            ],
        }
        (run / "report.json").write_text(json.dumps(report))
        if custom:
            (run / "cache" / "private-alias.txt").symlink_to(
                run / "runtime" / "private.log"
            )
        if alias == "hard_link":
            os.link(run / "isolated-browser-profile" / "Cookies", cache / "cookies.bin")
        elif alias == "cache_directory":
            for item in cache.iterdir():
                item.unlink()
            cache.rmdir()
            cache.symlink_to(run / "runtime", target_is_directory=True)
        elif alias == "run_directory":
            moved = tmp_path / "outside-run"
            run.rename(moved)
            run.symlink_to(moved, target_is_directory=True)
        elif alias:
            original, target = {
                "frame": (frame, run / "runtime" / "private.log"),
                "schema": (
                    frame.with_suffix(".schema.json"),
                    run / "runtime" / "private.log",
                ),
                "cache": (cache / "payload.json", run / "runtime" / "private.log"),
                "report": (run / "report.json", run / "runtime" / "report.json"),
            }[alias]
            if alias == "report":
                target.write_text(json.dumps(report))
            original.unlink()
            original.symlink_to(target)
        database = tmp_path / "warehouse.sqlite3"
        if alias:
            with self.assertRaises(ValueError):
                ingest_soccerdata(database, root)
            with sqlite3.connect(database) as connection:
                paths = [r[0] for r in connection.execute("SELECT path FROM assets")]
                self.assertFalse(any("private.log" in p for p in paths))
            return
        if general:
            build(database, root.parent)
        else:
            assert ingest_soccerdata(database, root)["frame_rows_by_provider"] == {"ESPN": 2}
        with sqlite3.connect(database) as connection:
            paths = [row[0] for row in connection.execute("SELECT path FROM assets")]
            assert "ESPN/cache/payload.json" in paths
            assert not any("Cookies" in path or "private.log" in path for path in paths)
            assert not any("private-alias" in path for path in paths)
            assert (
                connection.execute("SELECT COUNT(*) FROM quarantine").fetchone()[0] == 1
            )
            table = (
                "custom_csv_schedule"
                if custom
                else connection.execute(
                    "SELECT table_name FROM soccerdata_imports"
                ).fetchone()[0]
            )
            assert (
                connection.execute(
                    f"SELECT value FROM \"{table}\" WHERE id='1'"
                ).fetchone()[0]
                is None
            )


class TestRepositoryData(unittest.TestCase):
    def test_duplicate_mirror_uses_one_payload_and_one_csv_population(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "jokecamp"
            root.mkdir()
            raw = b"id,value\n1,3\n2,\n"
            entries = []
            for name in ["first.csv", "copy.csv"]:
                (root / name).write_bytes(raw)
                entries.append(
                    {
                        "path": name,
                        "bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                )
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "repository": "https://github.com/jokecamp/FootballData",
                        "revision": "abc",
                        "complete": True,
                        "files": entries,
                    }
                )
            )
            database = Path(tmp) / "data.sqlite3"
            report = ingest_repository(database, [root])
            self.assertEqual(report["unique_csv_assets"], 1)
            self.assertEqual(report["mirrored_files"], 1)
            with sqlite3.connect(database) as c:
                self.assertEqual(
                    c.execute(
                        "SELECT count(DISTINCT asset_id) FROM repository_files"
                    ).fetchone()[0],
                    1,
                )
                table = c.execute(
                    "SELECT table_name FROM repository_files WHERE table_name IS NOT NULL LIMIT 1"
                ).fetchone()[0]
                self.assertEqual(
                    c.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0], 2
                )
                self.assertIsNone(
                    c.execute(
                        f'SELECT value FROM "{table}" WHERE id=?', ("2",)
                    ).fetchone()[0]
                )

    def test_tampered_asset_fails_before_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "eddwebster"
            root.mkdir()
            (root / "data.csv").write_text("id\n1\n")
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "repository": "https://github.com/eddwebster/football_analytics",
                        "revision": "abc",
                        "complete": True,
                        "files": [{"path": "data.csv", "bytes": 5, "sha256": "0" * 64}],
                    }
                )
            )
            with self.assertRaises(ValueError):
                ingest_repository(Path(tmp) / "data.sqlite3", [root])

    def test_invalid_json_is_preserved_and_reported_as_quarantined(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "jokecamp"
            root.mkdir()
            raw = b'{"events":'
            (root / "invalid.json").write_bytes(raw)
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "repository": "https://github.com/jokecamp/FootballData",
                        "revision": "abc",
                        "complete": True,
                        "files": [
                            {
                                "path": "invalid.json",
                                "bytes": len(raw),
                                "sha256": hashlib.sha256(raw).hexdigest(),
                            }
                        ],
                    }
                )
            )
            database = Path(tmp) / "data.sqlite3"
            report = ingest_repository(database, [root])
            self.assertEqual(len(report["issues"]), 1)
            with sqlite3.connect(database) as connection:
                self.assertEqual(
                    connection.execute(
                        "SELECT status FROM repository_files"
                    ).fetchone()[0],
                    "quarantined_json",
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT complete FROM assets WHERE path='invalid.json'"
                    ).fetchone()[0],
                    1,
                )


class TestNewSourceFindings(unittest.TestCase):
    def test_pairs_require_same_source_id_league_adjacent_season_and_exposure(self):
        rows = [
            {"player_id": "1", "league": "A", "season_id": "2020", "minutes": "900"},
            {"player_id": "1", "league": "A", "season_id": "2021", "minutes": "950"},
            {"player_id": "1", "league": "A", "season_id": "2023", "minutes": "950"},
            {"player_id": "2", "league": "B", "season_id": "2020", "minutes": "950"},
            {"player_id": "2", "league": "A", "season_id": "2021", "minutes": "950"},
            {"player_id": "3", "league": "A", "season_id": "2020", "minutes": "899"},
            {"player_id": "3", "league": "A", "season_id": "2021", "minutes": "950"},
        ]
        pairs = consecutive_pairs(rows, 900)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][0]["season_id"], "2020")
        with self.assertRaises(ValueError):
            consecutive_pairs(rows + [rows[0]], 900)

    def test_zero_placeholder_possession_is_excluded_without_erasing_real_zero(self):
        self.assertFalse(possession_eligible([0, 0]))
        self.assertFalse(possession_eligible([None, 100]))
        self.assertFalse(possession_eligible([20, 50]))
        self.assertTrue(possession_eligible([0, 100]))
        self.assertTrue(possession_eligible([49.9, 50.1]))

    def test_penalty_label_recovered_by_shot_id_and_original_is_preserved(self):
        shots = [
            {"shot_id": "1", "situation": ""},
            {"shot_id": "2", "situation": "Open Play"},
        ]
        restored, count = restore_situations(shots, {"1": "Penalty", "2": "OpenPlay"})
        self.assertEqual(count, 1)
        self.assertEqual(restored[0]["situation"], "Penalty")
        self.assertEqual(shots[0]["situation"], "")
        with self.assertRaises(ValueError):
            restore_situations(shots, {"2": "FromCorner"})

    @unittest.skipUnless(
        importlib.util.find_spec("numpy"), "optional numerical analysis runtime"
    )
    def test_cluster_correlation_matches_independent_numpy_and_keeps_constant_unknown(
        self,
    ):
        import numpy as np

        x, y = [1, 3, 5, 2, 9], [2, 4, 2, 3, 8]
        result = correlation(x, y, ["A", "A", "B", "C", "C"], draws=50)
        self.assertAlmostEqual(result["r"], float(np.corrcoef(x, y)[0, 1]))
        self.assertEqual(result["clusters"], 3)
        constant = correlation([1, 1, 1], [2, 3, 4], ["A", "B", "C"], draws=20)
        self.assertIsNone(constant["r"])
        self.assertIsNone(constant["cluster_bootstrap_ci95"])


class TestFifaReports(unittest.TestCase):
    def offers_fixture(self):
        header = "# Player Total Offers In Front In Between Out to In In to Out In Behind No Movement Offers Received\n"
        pages = [
            "13 July 2025 - Stadium - 15:00\nOffering to Receive Alpha FC\n 10 4\nTotal Offers Made Total Offers Received\n",
            "Offering to Receive Beta FC\n 5 2\nTotal Offers Made Total Offers Received\n",
            "Divider\n",
            "In Possession - Offers & Receptions Alpha FC\n" + header + "1 Alpha PLAYER 10 2 3 0 0 1 4 4\n",
            "In Possession - Offers & Receptions Beta FC\n" + header + "9 Beta PLAYER 5 0 2 1 0 2 0 2\n",
        ]
        rosters = {0: {1: "Alpha PLAYER"}, 1: {9: "Beta PLAYER"}}
        return pages, rosters

    def test_offers_reconcile_categories_rosters_and_separate_overview_pages(self):
        from app.analytics.events.fifa_offers import parse_offers

        pages, rosters = self.offers_fixture()
        rows, teams = parse_offers(pages, ["Alpha FC", "Beta FC"], rosters)
        self.assertEqual(teams[0]["total_offers"], 10)
        self.assertEqual(teams[1]["offers_received"], 2)
        self.assertEqual(teams[0]["player_table_page"], 4)
        self.assertEqual(teams[0]["overview_page"], 1)
        self.assertEqual(rows[0]["out_to_in"], 0)
        self.assertEqual(rows[0]["player_name"], "Alpha PLAYER")
        self.assertEqual(teams[0]["players"], 1)

    def test_offers_reject_missing_cells_wrong_identities_and_unreconciled_counts(self):
        from app.analytics.events.fifa_offers import parse_offers

        pages, rosters = self.offers_fixture()
        changes = [
            (3, pages[3].replace("10 2 3 0 0 1 4 4", "10 2 3 0 1 4 4")),
            (3, pages[3].replace("10 2 3", "11 2 3")),
            (3, pages[3].replace("Alpha FC", "Other FC")),
            (3, pages[3].replace("Alpha PLAYER", "Other PLAYER")),
            (3, pages[3] + "1 Alpha PLAYER 10 2 3 0 0 1 4 4\n"),
            (3, pages[3].replace("1 Alpha PLAYER", "2 Alpha PLAYER")),
            (3, pages[3].replace("Offers Received", "Other Meaning")),
            (0, pages[0].replace("10 4", "10 5")),
            (3, "Divider"),
        ]
        for index, value in changes:
            with self.subTest(index=index, value=value):
                changed = pages.copy(); changed[index] = value
                with self.assertRaises(ValueError):
                    parse_offers(changed, ["Alpha FC", "Beta FC"], rosters)

    def test_physical_zones_preserve_provider_labels_and_reject_inconsistent_totals(
        self,
    ):
        text = "Physical Data Alpha FC\nZone 1: 0-7 km/h Zone 2: 7-15 km/h Zone 3: 15-20 km/h Zone 4: 20-25 km/h Zone 5: 25+ km/h High Speed Runs Sprints Top Speed\n(Zone 3) (Zone 4 & 5)\n10 Player NAME 10000.0 3000.0 4000.0 2000.0 700.0 300.0 150.0 45.0 33.1\n"
        rows = fifa_reports.physical_rows([text], ["Alpha FC", "Beta FC"])
        self.assertEqual(rows[0]["zone_5_above_25_kmh_m"], 300)
        self.assertEqual(rows[0]["sprints_zone_4_5_count"], 45)
        earlier = fifa_reports.physical_rows(
            [text.replace("(Zone 3) (Zone 4 & 5)", "(Zone 4) (Zone 5)")],
            ["Alpha FC", "Beta FC"],
        )[0]
        self.assertEqual(earlier["sprints_zone_5_count"], 45)
        self.assertNotIn("sprints_zone_4_5_count", earlier)
        self.assertEqual(rows[0]["player_name"], "Player NAME")
        for changed in [
            text.replace("10000.0", "11000.0"),
            text.replace("Alpha FC", "Other FC"),
            text.replace("25+", "24+"),
        ]:
            with self.assertRaises(ValueError):
                fifa_reports.physical_rows([changed], ["Alpha FC", "Beta FC"])

    def test_official_collection_limits_requests_and_reuses_verified_cache(self):
        settings = {
            "edition": 2025,
            "matches": 1,
            "competition": "Test",
            "indices": ["https://www.fifatrainingcentre.com/hub.php"],
        }

        def response(url, **kwargs):
            body = (
                b"%PDF-1.7\npublic-test"
                if url.endswith(".pdf")
                else b'<a href="/media/native/M01 A V B.pdf">Report</a>'
            )
            result = MagicMock()
            result.__enter__.return_value = result
            result.iter_content.return_value = iter([body])
            return result

        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            fifa_reports.SERIES, {"test": settings}
        ), patch.object(fifa_reports.requests, "Session") as session, patch.object(
            fifa_reports, "SlidingWindowRateLimiter"
        ) as limiter:
            session.return_value.__enter__.return_value.get.side_effect = response
            manifest = fifa_reports.collect(temporary, "test")
            self.assertEqual(manifest["errors"], [])
            self.assertEqual(len(manifest["files"]), 2)
            self.assertTrue(
                (Path(temporary) / "test" / "M001.pdf")
                .read_bytes()
                .startswith(b"%PDF-")
            )
            self.assertEqual(limiter.return_value.add_request.call_count, 2)
            self.assertEqual(limiter.return_value.wait_if_needed.call_count, 2)
            session.reset_mock()
            limiter.return_value.reset_mock()
            fifa_reports.collect(temporary, "test")
            session.assert_not_called()
            limiter.return_value.add_request.assert_not_called()

    def valid_pages(self):
        return [
            "Alpha FC 3 - 0 Beta FC\nFinal - Match 63\n13 July 2025\n",
            "teams",
            "Total 32.3% 6.7% 61% Total\n3 Goals 0\n9(5) Attempts at Goal (On Target) 9(6)\n327(263) Total Passes (Complete) 620(567)\n287(44) Defensive Pressures Applied (Direct Pressures) 199(49)\n",
            "9% Long Ball 1%\n6% High Press 10%\n",
        ]

    def test_report_retains_three_state_possession_and_team_sides(self):
        metadata, rows = parse_report(self.valid_pages(), 63, 2025)
        self.assertEqual(metadata["goals"], [3, 0])
        self.assertEqual(metadata["date"], "2025-07-13")
        self.assertEqual(metadata["in_contest_pct"], 6.7)
        values = {(r["team_slot"], r["metric"]): r["value"] for r in rows}
        self.assertEqual(values[0, "controlled_possession_pct"], 32.3)
        self.assertEqual(values[1, "controlled_possession_pct"], 61)
        self.assertEqual(values[0, "direct_pressures"], 44)
        self.assertEqual(values[1, "in_possession_long_ball"], 1)
        self.assertNotIn((0, "ball_recovery_time"), values)

    def test_wrapped_cover_uses_explicit_team_summary_sides(self):
        pages = self.valid_pages()
        pages[0] = (
            "Alpha       3-   Beta\nFC          0    FC\nFinal - Match 63\n13 July 2025\n"
        )
        pages[1] = "Match Summary - Teams\nAlpha FC   3     0    Beta FC\nSTARTING\n"
        parsed, _ = parse_report(pages, 63, 2025)
        self.assertEqual(parsed["teams"], ["Alpha FC", "Beta FC"])
        self.assertEqual(parsed["goals"], [3, 0])
        self.assertEqual(parsed["stage"], "Final")

    def test_wrong_identity_edition_or_missing_core_metrics_is_rejected(self):
        for number, edition in [(62, 2025), (63, 2026)]:
            with self.assertRaises(ValueError):
                parse_report(self.valid_pages(), number, edition)
        p = self.valid_pages()
        p[2] = ""
        with self.assertRaises(ValueError):
            parse_report(p, 63, 2025)

    def test_index_accepts_official_practice_paths_and_rejects_other_hosts(self):
        html = '<a href="/media/native/practice/M61 FLU V CHE.pdf">61</a><a href="https://evil.invalid/M63 CHE V PSG.pdf">63</a><a href="/media/native/M64 invalid.pdf">bad</a>'
        with self.assertRaises(ValueError):
            report_links(html, "https://www.fifatrainingcentre.com/hub.php", 63)
        self.assertEqual(
            set(
                report_links(
                    html.split('<a href="/media/native/M64')[0],
                    "https://www.fifatrainingcentre.com/hub.php",
                    63,
                )
            ),
            {61},
        )

    def test_dimensions_follow_annotation_position_not_text_reading_order(self):
        words = []
        for offset, values in [
            (0, [34, 43, 48]),
            (300, [26, 42, 40]),
            (600, [18, 36, 19]),
        ]:
            for x, v in zip([40, 150, 270], values):
                words.append(
                    f'<word xMin="{offset+x}" xMax="{offset+x+10}">{v}m</word>'
                )
        xml = (
            '<html xmlns="http://www.w3.org/1999/xhtml"><body><page width="900">'
            + "".join(reversed(words))
            + "</page></body></html>"
        )
        rows = shape_metrics(
            xml,
            ["Defensive Line Height & Team Length Alpha FC"],
            ["Alpha FC", "Beta FC"],
        )
        values = {r["metric"]: r["value"] for r in rows}
        self.assertEqual(values["defense_high_block_press_team_width"], 43)
        self.assertEqual(values["defense_high_block_press_line_height"], 48)
        self.assertEqual(values["defense_low_block_team_length"], 18)


class TestTacticalEvidence(unittest.TestCase):
    def test_metadata_and_show_notes_cannot_become_coach_statements(self):
        for level in ["metadata_only", "blocked", "show_notes"]:
            with self.assertRaises(ValueError):
                validate_card(
                    {"content_level": level}, {"claim_type": "coach_self_report"}, ""
                )
        validate_card(
            {"content_level": "show_notes"},
            {"claim_type": "topic_only", "anchor": "rest defence"},
            "rest defence is a programme topic",
        )

    def test_statement_requires_an_anchor_in_the_inspected_text(self):
        source = {"content_level": "automatic_captions"}
        card = {
            "claim_type": "coach_self_report",
            "anchor": "we wanted Gomez on the ball",
        }
        validate_card(
            source, card, "So we wanted Gomez on the ball and waited for the next pass"
        )
        validate_card(
            source, card, "So we wanted Gomez\non the ball and waited for the next pass"
        )
        with self.assertRaises(ValueError):
            validate_card(source, card, "No such statement")


if __name__ == "__main__":
    unittest.main()


class TestIdentityBridgeAudit(unittest.TestCase):
    def _identity_fixture(self, root):
        import csv
        import gzip
        import io
        from app.data_service.identity_bridge import TABLES
        data = {
            'entities': [
                {'reep_id':rid, 'entity_type':kind, 'canonical_name':'Same Name',
                 'metadata_json':json.dumps({'status':'withdrawn'}) if rid == 'gone' else '{}'}
                for rid,kind in [('a','player'),('b','player'),('team','team'),('gone','player'),('middle','team')]
            ],
            'redirects': [dict(zip(TABLES['redirects'], r)) for r in [
                ('old','step','merge',''),('step','a','merge',''),
                ('cycle','cycle2','merge',''),('cycle2','cycle','merge',''),
                ('retired','','withdrawn',''),('conflict','a','merge',''),('conflict','b','merge',''),
                ('mixed','middle','merge',''),('middle','a','merge','')]],
            'bridges': [dict(zip(TABLES['bridges'], r)) for r in [
                ('a','espn','athlete','1','player'),('a','espn','athlete','1','player'),
                ('b','espn','athlete','001','player'),('old','statsbomb','player','2','player'),
                ('a','understat','player','3','player'),('b','understat','player','3','player'),
                ('team','espn','team','4','team'),('gone','espn','athlete','5','player'),
                ('retired','espn','athlete','6','player'),('cycle','espn','athlete','7','player'),
                ('conflict','espn','athlete','8','player'),('absent','espn','athlete','9','player'),
                ('mixed','espn','athlete','10','player')]],
        }
        release = {'schema_version':'bridge-register-v1', 'publishable':True,
                   'licence':{'spdx':'CC0-1.0'},'stamp':'fixture-1','release_tag':'fixture-1','files':{}}
        (root/'csv').mkdir()
        for name, records in data.items():
            text=io.StringIO(newline='');writer=csv.DictWriter(text,fieldnames=TABLES[name]);writer.writeheader();writer.writerows(records)
            raw=gzip.compress(text.getvalue().encode(),mtime=0); path=f'csv/{name}.csv.gz';(root/path).write_bytes(raw)
            release['files'][path]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'rows':len(records)}
        (root/'release.json').write_text(json.dumps(release))
        queries = root.parent/(root.name+'-queries.csv')
        queries.write_text('provider,namespace,external_id,entity_type,label\n'
                           'espn,athlete,1,player,Same Name\nespn,athlete,001,player,Same Name\n'
                           'statsbomb,player,2,player,x\nunderstat,player,3,player,x\n'
                           'espn,team,4,player,x\nespn,athlete,5,player,x\nespn,athlete,6,player,x\n'
                           'espn,athlete,7,player,x\nespn,athlete,8,player,x\nespn,athlete,9,player,x\n'
                           'espn,athlete,10,player,x\nespn,player,1,player,Same Name\n')
        return queries, hashlib.sha256((root/'release.json').read_bytes()).hexdigest()

    def test_exact_identity_redirects_conflicts_and_row_lineage(self):
        from app.data_service.identity_bridge import audit_identity_bridges
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'release';root.mkdir();queries,sha=self._identity_fixture(root)
            before={p:p.read_bytes() for p in [queries,root/'release.json',*list((root/'csv').iterdir())]}
            with patch('requests.Session',side_effect=AssertionError('offline audit')):
                result=audit_identity_bridges(root,queries,release_sha256=sha)
            self.assertEqual(result,audit_identity_bridges(root,queries,release_sha256=sha))
            self.assertEqual([r['status'] for r in result['results']],
                ['unique','unique','redirected','ambiguous','type_mismatch','retired','retired',
                 'invalid_redirect','ambiguous','missing_entity','type_mismatch','uncovered'])
            self.assertEqual(result['results'][0]['duplicate_bridge_rows'],1)
            self.assertEqual(result['results'][1]['resolved_reep_id'],'b')
            moved=result['results'][2]['matches'][0]
            self.assertEqual((moved['original_reep_id'],moved['resolved_reep_id']),('old','a'))
            self.assertEqual([e['record'] for e in moved['chain']],[1,2])
            self.assertEqual(moved['bridge']['record'],4)
            self.assertEqual(moved['entities'][0]['record'],1)
            self.assertEqual(sum(result['counts'].values()),12)
            self.assertEqual(sum(g['queries'] for g in result['groups']),12)
            self.assertEqual(result['usage']['rows'],27)
            self.assertEqual(result['adoption'],'not_adopted')
            self.assertFalse(result['historical_availability_verified'])
            self.assertEqual(result['mapping_status'],'review_required')
            for p,raw in before.items():self.assertEqual(p.read_bytes(),raw)
            queries.write_text('provider,namespace,external_id,entity_type\nespn,no-such-namespace,1,player\n')
            empty=audit_identity_bridges(root,queries,release_sha256=sha)
            self.assertEqual(empty['mapping_status'],'no_intersection')
            self.assertEqual(empty['counts'],{'uncovered':1})
            queries.write_text('provider,namespace,external_id,entity_type\nespn,athlete,1,player\n')
            self.assertEqual(audit_identity_bridges(root,queries,release_sha256=sha)['mapping_status'],'mapped_queries_only')

    def test_identity_budgets_checksums_schema_and_concurrent_change(self):
        from app.data_service import identity_bridge as bridge
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'release';root.mkdir();queries,sha=self._identity_fixture(root)
            for limits in ({'max_bytes':1},{'max_decoded_bytes':1},{'max_rows':1},{'max_queries':1},
                           {'max_rows':0},{'max_rows':True}):
                with self.subTest(limits=limits), self.assertRaises(ValueError):
                    bridge.audit_identity_bridges(root,queries,release_sha256=sha,**limits)
            with self.assertRaisesRegex(ValueError,'Release checksum'):
                bridge.audit_identity_bridges(root,queries,release_sha256='0'*64)
            path=root/'csv/bridges.csv.gz';raw=path.read_bytes();path.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
            with self.assertRaisesRegex(ValueError,'Source checksum'):
                bridge.audit_identity_bridges(root,queries,release_sha256=sha)
            path.write_bytes(raw)
            original_rows=bridge._rows
            def mutate(path,required,budget):
                yield from original_rows(path,required,budget)
                if path.name=='entities.csv.gz': queries.write_text(queries.read_text()+'\n')
            with patch.object(bridge,'_rows',side_effect=mutate), self.assertRaisesRegex(ValueError,'changed during'):
                bridge.audit_identity_bridges(root,queries,release_sha256=sha)
            release=json.loads((root/'release.json').read_text());release['files']['csv/entities.csv.gz']['rows']+=1
            (root/'release.json').write_text(json.dumps(release));sha=hashlib.sha256((root/'release.json').read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError,'row count mismatch'):
                bridge.audit_identity_bridges(root,queries,release_sha256=sha)
            release['files']['csv/entities.csv.gz']['rows']-=1
            (root/'release.json').write_text(json.dumps(release));sha=hashlib.sha256((root/'release.json').read_bytes()).hexdigest()
            queries.write_text('provider,namespace,external_id,entity_type\nwikidata,player,1,player\n')
            with self.assertRaisesRegex(ValueError,'supported provider'):
                bridge.audit_identity_bridges(root,queries,release_sha256=sha)
            queries.write_text('provider,external_id,entity_type\nespn,1,player\n')
            with self.assertRaisesRegex(ValueError,'schema'):
                bridge.audit_identity_bridges(root,queries,release_sha256=sha)
            queries.write_text('provider,namespace,external_id,entity_type\nespn,athlete,,player\n')
            with self.assertRaisesRegex(ValueError,'nonempty'):
                bridge.audit_identity_bridges(root,queries,release_sha256=sha)
