import json
import tempfile
import unittest
from pathlib import Path


class TestWarehouse(unittest.TestCase):
    def test_shared_read_database_pins_snapshot_closes_on_failure_and_encodes_paths(self):
        import sqlite3
        from unittest.mock import Mock, patch
        from app.data_service.database import read_database
        from contextlib import closing
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source?#.sqlite3'
            with closing(sqlite3.connect(path)) as writer:
                writer.execute('PRAGMA journal_mode=WAL')
                writer.execute('CREATE TABLE sample(value INTEGER)')
                writer.execute('INSERT INTO sample VALUES(1)');writer.commit()
                with read_database(path,row_factory=sqlite3.Row) as reader:
                    self.assertEqual(reader.execute('SELECT value FROM sample').fetchone()['value'],1)
                    writer.execute('UPDATE sample SET value=2');writer.commit()
                    self.assertEqual(reader.execute('SELECT value FROM sample').fetchone()['value'],1)
                with self.assertRaises(sqlite3.ProgrammingError):reader.execute('SELECT 1')
            with self.assertRaisesRegex(ValueError,'body failed'):
                with read_database(path) as failed:
                    with self.assertRaises(sqlite3.OperationalError):failed.execute('INSERT INTO sample VALUES(3)')
                    raise ValueError('body failed')
            with self.assertRaises(sqlite3.ProgrammingError):failed.execute('SELECT 1')
        connection=Mock();connection.execute.side_effect=sqlite3.OperationalError('begin failed')
        with patch('app.data_service.database.sqlite3.connect',return_value=connection),self.assertRaises(sqlite3.OperationalError):
            with read_database('unused.sqlite3'):self.fail('must not enter failed transaction')
        connection.close.assert_called_once_with()

    def test_connection_is_closed_when_transaction_finalization_fails(self):
        from unittest.mock import Mock
        from app.data_service.warehouse import Warehouse

        for operation, exception_type in (("commit", None), ("rollback", ValueError)):
            with self.subTest(operation=operation):
                warehouse = object.__new__(Warehouse)
                warehouse.connection = Mock()
                getattr(warehouse.connection, operation).side_effect = OSError(
                    f"synthetic {operation} failure"
                )
                error = ValueError("original body failure") if exception_type else None
                with self.assertRaisesRegex(OSError, f"{operation} failure"):
                    warehouse.__exit__(exception_type, error, None)
                getattr(warehouse.connection, operation).assert_called_once_with()
                other = "rollback" if operation == "commit" else "commit"
                getattr(warehouse.connection, other).assert_not_called()
                warehouse.connection.close.assert_called_once_with()

    def test_failed_context_rolls_back_semantic_tail_and_keeps_completed_raw_asset(self):
        import hashlib
        import sqlite3
        import zlib
        from app.data_service.warehouse import Warehouse

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "warehouse.sqlite3"
            source = root / "source.json"
            payload = b'{"source":"completed capture"}'
            source.write_bytes(payload)
            failure = ValueError("invalid second semantic record")
            with self.assertRaises(ValueError) as raised:
                with Warehouse(database) as warehouse:
                    warehouse.connection.execute(
                        "CREATE TABLE semantic_records(asset_id INTEGER REFERENCES assets(id),value TEXT)"
                    )
                    aid = warehouse.asset(
                        warehouse.dataset("source", "football", "test"), source, source.name
                    )
                    warehouse.connection.execute(
                        "INSERT INTO semantic_records VALUES(?,?)", (aid, "first record")
                    )
                    raise failure
            self.assertIs(raised.exception, failure)
            with self.assertRaises(sqlite3.ProgrammingError):
                warehouse.connection.execute("SELECT 1")
            with sqlite3.connect(database) as connection:
                self.assertEqual(connection.execute("SELECT * FROM semantic_records").fetchall(), [])
                self.assertEqual(
                    connection.execute("SELECT id,sha256,bytes,complete FROM assets").fetchall(),
                    [(aid, hashlib.sha256(payload).hexdigest(), len(payload), 1)],
                )
                chunks = connection.execute(
                    "SELECT content FROM raw_chunks WHERE asset_id=? ORDER BY ordinal", (aid,)
                ).fetchall()
                self.assertEqual(b"".join(zlib.decompress(row[0]) for row in chunks), payload)

    def test_atomic_scopes_restore_versions_nested_writes_and_csv_batches_before_retry(self):
        import hashlib
        import io
        import sqlite3
        from app.data_service.warehouse import Warehouse

        class FailingCSV(io.StringIO):
            def __init__(self, payload):
                super().__init__(payload)
                self.lines_read = 0

            def __next__(self):
                if self.lines_read == 21001:
                    raise OSError("source failed after the committed-batch threshold")
                line = super().__next__()
                self.lines_read += 1
                return line

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "warehouse.sqlite3"
            with Warehouse(database) as warehouse:
                dataset = warehouse.dataset("source", "football", "test")
                original = warehouse.asset_stream(dataset, io.BytesIO(b"original"), "source.txt")
                warehouse.connection.execute("CREATE TABLE transaction_journal(value TEXT UNIQUE)")
                with self.assertRaisesRegex(RuntimeError, "outer failure"):
                    with warehouse.atomic():
                        revision = warehouse.asset_stream(
                            dataset, io.BytesIO(b"rejected revision"), "source.txt"
                        )
                        with warehouse.atomic():
                            warehouse.connection.execute(
                                "INSERT INTO transaction_journal VALUES('inner success')"
                            )
                            warehouse.finish(revision, "nested_revision", 1)
                        with sqlite3.connect(database) as observer:
                            self.assertEqual(observer.execute("SELECT count(*) FROM assets").fetchone()[0], 1)
                            self.assertEqual(observer.execute("SELECT * FROM transaction_journal").fetchall(), [])
                        raise RuntimeError("outer failure")
                self.assertEqual(warehouse.read_raw(original), b"original")
                self.assertEqual(warehouse.connection.execute("SELECT count(*) FROM assets").fetchone()[0], 1)
                self.assertEqual(warehouse.connection.execute("SELECT * FROM imports").fetchall(), [])
                self.assertEqual(warehouse.connection.execute("SELECT * FROM transaction_journal").fetchall(), [])

                with warehouse.atomic():
                    warehouse.connection.execute("INSERT INTO transaction_journal VALUES('outer before')")
                    with self.assertRaisesRegex(ValueError, "inner failure"):
                        with warehouse.atomic():
                            warehouse.asset_stream(dataset, io.BytesIO(b"rejected inner revision"), "source.txt")
                            warehouse.connection.execute("INSERT INTO transaction_journal VALUES('rejected inner')")
                            raise ValueError("inner failure")
                    accepted = warehouse.asset_stream(dataset, io.BytesIO(b"accepted revision"), "source.txt")
                    warehouse.finish(accepted, "accepted_revision", 1)
                    warehouse.connection.execute("INSERT INTO transaction_journal VALUES('outer after')")
                with sqlite3.connect(database) as observer:
                    self.assertEqual(
                        observer.execute("SELECT value FROM transaction_journal ORDER BY rowid").fetchall(),
                        [("outer before",), ("outer after",)],
                    )
                    self.assertEqual(
                        observer.execute("SELECT path FROM assets ORDER BY id").fetchall(),
                        [("source.txt",), ("source.txt@" + hashlib.sha256(b"accepted revision").hexdigest(),)],
                    )
                self.assertEqual(warehouse.read_raw(original), b"original")
                self.assertEqual(warehouse.read_raw(accepted), b"accepted revision")

                # SQLite can end the whole transaction itself; preserve that
                # original error when both nested savepoints have disappeared.
                with self.assertRaisesRegex(sqlite3.IntegrityError, "UNIQUE"):
                    with warehouse.atomic():
                        with warehouse.atomic():
                            warehouse.connection.execute(
                                "INSERT INTO transaction_journal VALUES('automatically rolled back')"
                            )
                            warehouse.connection.execute(
                                "INSERT OR ROLLBACK INTO transaction_journal VALUES('outer before')"
                            )
                self.assertFalse(warehouse.connection.in_transaction)
                self.assertEqual(
                    [row[0] for row in warehouse.connection.execute("SELECT value FROM transaction_journal ORDER BY rowid")],
                    ["outer before", "outer after"],
                )

                csv_text = "id,value\n" + "".join(f"{index},row-{index}\n" for index in range(21001))
                with self.assertRaisesRegex(OSError, "source failed"):
                    with warehouse.atomic():
                        failed = warehouse.asset_stream(dataset, io.BytesIO(csv_text.encode()), "batch.csv")
                        warehouse.csv_stream(failed, FailingCSV(csv_text), "transactional_csv")
                with sqlite3.connect(database) as observer:
                    self.assertEqual(observer.execute("SELECT count(*) FROM assets").fetchone()[0], 2)
                    self.assertEqual(
                        observer.execute("SELECT name FROM sqlite_master WHERE name='transactional_csv'").fetchall(), []
                    )
                    for table in ("raw_chunks", "csv_columns", "csv_irregular_rows", "asset_encodings", "imports"):
                        self.assertEqual(
                            observer.execute(f"SELECT count(*) FROM {table} WHERE asset_id=?", (failed,)).fetchone()[0],
                            0,
                            table,
                        )
                with warehouse.atomic():
                    retried = warehouse.asset_stream(dataset, io.BytesIO(csv_text.encode()), "batch.csv")
                    warehouse.csv_stream(retried, io.StringIO(csv_text), "transactional_csv")
                with sqlite3.connect(database) as observer:
                    self.assertEqual(observer.execute("SELECT count(*) FROM transactional_csv").fetchone()[0], 21001)
                    self.assertEqual(
                        observer.execute("SELECT rows FROM imports WHERE asset_id=? AND kind='transactional_csv'", (retried,)).fetchone()[0],
                        21001,
                    )
                    self.assertEqual(observer.execute("SELECT complete FROM assets WHERE id=?", (retried,)).fetchone()[0], 1)
                self.assertEqual(warehouse.read_raw(retried), csv_text.encode())

    def test_xg_audit_keeps_asset_versions_excludes_bad_pairs_and_reads_without_writes(self):
        import copy
        import hashlib
        import sqlite3
        import subprocess
        import sys
        from unittest.mock import patch
        from app.data_service import database as module
        from app.data_service.build_warehouse import document
        from app.data_service.fetch.public import FootballCSVClient
        from app.data_service.warehouse import Warehouse

        rows = FootballCSVClient().parse_csv_text('PL',
            'Date,HomeTeam,AwayTeam,FTHG,FTAG,HxG,AxG\n'
            '01/08/2026,A,B,2,1,0,2.5\n02/08/2026,B,A,1,1,4.0,1.5\n'
            '03/08/2026,A,B,0,0,,1.0\n04/08/2026,B,A,0,1,1e309,1.0\n'
            '05/08/2026,A,B,,,5.0,2.0\n', 'x')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'warehouse.sqlite3'
            with Warehouse(path) as db:
                dataset = db.dataset('test', 'football', 'football-csv')
                for version in ('original', 'revision', 'legacy'):
                    matches = copy.deepcopy(rows)
                    if version == 'revision':
                        matches[0]['statistics']['home_xg'] = True
                        matches[1]['statistics_metadata']['xg']['complete_pair'] = False
                        matches[2]['statistics_metadata']['xg']['home_status'] = []
                        matches[3]['statistics']['home_xg'] = 10**400
                    elif version == 'legacy':
                        for match in matches:
                            del match['statistics_metadata']
                            match['statistics'].pop('home_xg')
                            match['statistics'].pop('away_xg')
                    source = root / f'{version}.json'
                    source.write_text(json.dumps({'source': 'football-csv', 'provider': 'football-data.co.uk',
                        'competition': 'PL', 'season': '2026', 'matches': matches}, allow_nan=False))
                    aid = db.asset(dataset, source, source.name)
                    document(db, aid, source, 'football-csv_document')
                # An identically shaped document from another provider must not be pooled.
                source = root / 'other.json'
                source.write_text(json.dumps({'source': 'other', 'provider': 'other', 'matches': rows}))
                document(db, db.asset(dataset, source, source.name), source, 'other')
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            connect = sqlite3.connect
            def enforce_read_only(*args, **kwargs):
                self.assertTrue(kwargs.get('uri'))
                self.assertTrue(args[0].endswith('?mode=ro'))
                connection = connect(*args, **kwargs)
                with self.assertRaisesRegex(sqlite3.OperationalError, 'readonly'):
                    connection.execute('DELETE FROM documents')
                connection.rollback()  # The failed write still opened an implicit transaction.
                return connection
            with patch.object(module.sqlite3, 'connect', side_effect=enforce_read_only):
                report = module.match_xg_coverage(path)
            snapshots = report['snapshots']
            self.assertEqual(len(snapshots), 3)
            original, revision, legacy = snapshots
            self.assertEqual([r['matches'] for r in snapshots], [5, 5, 5])
            self.assertEqual(original['complete_numeric_pairs'], 3)
            self.assertEqual(original['complete_finished_pairs'], 2)
            self.assertEqual(original['paired_xg_summary'], {
                'home': {'mean': 2.0, 'min': 0.0, 'max': 4.0},
                'away': {'mean': 2.0, 'min': 1.5, 'max': 2.5}})
            self.assertEqual(original['field_status_counts']['home'],
                {'valid': 3, 'missing': 1, 'invalid': 1, 'column_absent': 0, 'untracked': 0})
            self.assertEqual(revision['inconsistent_metadata_matches'], 4)
            self.assertEqual(revision['complete_finished_pairs'], 0)
            self.assertIsNone(revision['paired_xg_summary']['home']['mean'])
            self.assertEqual(legacy['field_status_counts']['home']['untracked'], 5)
            cli = subprocess.run([sys.executable, '-m', 'app.data_service.database',
                '--database', str(path), '--xg-coverage'], capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(cli.stdout), report)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
            with self.assertRaises(sqlite3.OperationalError):
                module.match_xg_coverage(root / 'missing.sqlite3')
            self.assertFalse((root / 'missing.sqlite3').exists())

    def test_failed_export_never_publishes_corrupt_or_partial_bytes(self):
        import sqlite3
        import zlib
        from app.data_service.warehouse import Warehouse
        from app.data_service.database import export_asset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.write_bytes(b"verified original")
            database = root / "warehouse.sqlite3"
            with Warehouse(database) as db:
                asset = db.asset(db.dataset("source", "football", "test"), source, "source")
            for index, damaged in enumerate((zlib.compress(b"wrong bytes"), b"invalid zlib")):
                with self.subTest(damaged=damaged):
                    with sqlite3.connect(database) as db:
                        db.execute("UPDATE raw_chunks SET content=? WHERE asset_id=?", (damaged, asset))
                    output = root / f"export-{index}"
                    with self.assertRaises((ValueError, zlib.error)):
                        export_asset(database, asset, output)
                    self.assertFalse(output.exists())
                    self.assertEqual(list(root.glob(".export-*.tmp")), [])

    def test_export_keeps_one_snapshot_during_a_concurrent_source_update(self):
        import hashlib
        import sqlite3
        import zlib
        from unittest.mock import patch
        from app.data_service import database as module
        from app.data_service.warehouse import Warehouse

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            original, replacement = b"before update", b"after update with new length"
            source.write_bytes(original)
            path = root / "warehouse.sqlite3"
            with Warehouse(path) as db:
                asset = db.asset(db.dataset("source", "football", "test"), source, "source")
            connect = sqlite3.connect

            class UpdatingConnection(sqlite3.Connection):
                def execute(self, sql, parameters=()):
                    if "SELECT content FROM raw_chunks" in sql:
                        with connect(path) as writer:
                            writer.execute("UPDATE assets SET sha256=?,bytes=? WHERE id=?", (hashlib.sha256(replacement).hexdigest(), len(replacement), asset))
                            writer.execute("UPDATE raw_chunks SET content=? WHERE asset_id=?", (zlib.compress(replacement), asset))
                    return super().execute(sql, parameters)

            with patch.object(module.sqlite3, "connect", side_effect=lambda *a, **kw: connect(*a, **kw, factory=UpdatingConnection)):
                module.export_asset(path, asset, root / "export")
            self.assertEqual((root / "export").read_bytes(), original)

    def test_export_does_not_overwrite_a_destination_created_while_streaming(self):
        from unittest.mock import patch
        from app.data_service import database as module
        from app.data_service.warehouse import Warehouse

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.write_bytes(b"verified source")
            path = root / "warehouse.sqlite3"
            with Warehouse(path) as db:
                asset = db.asset(db.dataset("source", "football", "test"), source, "source")
            link = module.os.link

            def race(source, destination):
                Path(destination).write_bytes(b"created by another caller")
                return link(source, destination)

            with patch.object(module.os, "link", side_effect=race):
                with self.assertRaises(FileExistsError):
                    module.export_asset(path, asset, root / "export")
            self.assertEqual((root / "export").read_bytes(), b"created by another caller")
            self.assertEqual(list(root.glob(".export.*.tmp")), [])

    def test_lineage_scopes_identical_run_ids_and_reports_unknown_inputs(self):
        from app.data_service.database import analysis_lineage
        from app.data_service.warehouse import Warehouse

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.write_bytes(b"test")
            path = root / "warehouse.sqlite3"
            with Warehouse(path) as db:
                dataset = db.dataset("public source", "football", "statsbomb", "2024")
                inputs = [db.asset(dataset, source, name) for name in ("input-a", "input-b")]
                output = db.asset(db.dataset("derived", "derived", "soccer_analytics"), source, "figure.svg")
                db.connection.execute("CREATE TABLE analysis_run_assets(run_kind TEXT,run_id TEXT,asset_id INTEGER REFERENCES assets(id),PRIMARY KEY(run_kind,run_id,asset_id))")
                db.connection.execute("CREATE TABLE analysis_run_inputs(run_kind TEXT,run_id TEXT,asset_id INTEGER REFERENCES assets(id),PRIMARY KEY(run_kind,run_id,asset_id))")
                db.connection.executemany("INSERT INTO analysis_run_assets VALUES(?,?,?)", [(kind, "shared-id", output) for kind in ("first", "second")])
                db.connection.executemany("INSERT INTO analysis_run_inputs VALUES(?,?,?)", [("first", "shared-id", aid) for aid in inputs])
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                analysis_lineage(path, "shared-id")
            first = analysis_lineage(path, "shared-id", run_kind="first", limit=1)
            self.assertEqual(first["inputs"]["total"], 2)
            self.assertEqual(len(first["inputs"]["assets"]), 1)
            self.assertTrue(first["inputs"]["truncated"])
            self.assertEqual(first["inputs"]["assets"][0]["provider"], "statsbomb")
            self.assertEqual(first["outputs"]["assets"][0]["path"], "figure.svg")
            second = analysis_lineage(path, "shared-id", run_kind="second")
            self.assertFalse(second["inputs_recorded"])
            self.assertEqual(second["inputs"]["total"], 0)
            with self.assertRaises(ValueError):
                analysis_lineage(path, "shared-id", run_kind="unrelated")
            with self.assertRaises(ValueError):
                analysis_lineage(path, "shared-id", run_kind="first", limit=True)

    def test_fm_ui_without_birthdate_column_keeps_edition(self):
        import sqlite3
        import zipfile
        from contextlib import redirect_stdout
        from io import StringIO
        from app.data_service.build_warehouse import build

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = root / "sources"
            ratings = sources / "game-ratings"
            ratings.mkdir(parents=True)
            with zipfile.ZipFile(ratings / "fm22.zip", "w") as archive:
                archive.writestr("fm20data.csv", "UID,Name\n1,Missing birthday\n")
            database = root / "football.sqlite3"
            for _ in range(2):
                with redirect_stdout(StringIO()):
                    build(database, sources)
                with sqlite3.connect(database) as db:
                    self.assertEqual(
                        db.execute(
                            "SELECT birth_date,edition FROM game_profiles"
                        ).fetchall(),
                        [(None, "2020")],
                    )

    def test_fm_birthdates_survive_rebuilds_and_recover_from_raw_sources(self):
        import sqlite3
        import zipfile
        from contextlib import redirect_stdout
        from io import StringIO
        from app.data_service.build_warehouse import build

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = root / "sources"
            ratings = sources / "game-ratings"
            ratings.mkdir(parents=True)
            with zipfile.ZipFile(ratings / "fm17.zip", "w") as archive:
                archive.writestr(
                    "players.csv",
                    "UID,Name,Born\n1,Valid,01-02-1995\n2,Invalid,31-02-1995\n3,Missing,\n",
                )
            editions = {
                "fm20data": "2020",
                "fm21data": "2021",
                "fm22data": "2022",
                "fm2023": "2023",
            }
            with zipfile.ZipFile(ratings / "fm22.zip", "w") as archive:
                for member in editions:
                    archive.writestr(
                        member + ".csv",
                        " UID , Name , DoB \n1,Valid, 01/02/1995 (age 31) \n2,Invalid,31/02/1995\n",
                    )
            database = root / "football.sqlite3"
            expected = sorted(
                [
                    ("fm17", "1", "1995-02-01", "2017"),
                    ("fm17", "2", None, "2017"),
                    ("fm17", "3", None, "2017"),
                ]
                + [
                    ("fm_export", identity, birth, edition)
                    for edition in editions.values()
                    for identity, birth in [("1", "1995-02-01"), ("2", None)]
                ]
            )
            for iteration in range(3):
                with redirect_stdout(StringIO()):
                    build(database, sources)
                with sqlite3.connect(database) as db:
                    self.assertEqual(
                        db.execute(
                            "SELECT provider,player_id,birth_date,edition FROM game_profiles "
                            "ORDER BY provider,player_id,edition"
                        ).fetchall(),
                        expected,
                    )
                    self.assertEqual(
                        db.execute("SELECT COUNT(*) FROM assets").fetchone()[0], 7
                    )
                    if iteration == 1:
                        # Recover dates erased by earlier versions of the importer.
                        db.execute("UPDATE game_profiles SET birth_date=NULL")

    def test_end_to_end_preserves_corrupt_event_source_and_valid_events(self):
        from app.data_service.build_warehouse import build

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = root / "open-data" / "statsbomb" / "data" / "events"
            events.mkdir(parents=True)
            (events / "123.json").write_text(
                json.dumps(
                    [{"id": "event1", "type": {"name": "Pass"}, "player": {"id": 7}}]
                )
            )
            corrupted = b"[\x00]"
            (events / "124.json").write_bytes(corrupted)
            build(root / "football.sqlite3", root / "open-data")
            import sqlite3

            with sqlite3.connect(root / "football.sqlite3") as db:
                self.assertEqual(
                    db.execute(
                        "SELECT match_id,event_id,event_type,player_id FROM statsbomb_events"
                    ).fetchone(),
                    ("123", "event1", "Pass", 7),
                )
                self.assertEqual(
                    db.execute("SELECT COUNT(*) FROM quarantine").fetchone()[0], 1
                )
                self.assertEqual(
                    db.execute(
                        "SELECT COUNT(*) FROM assets WHERE complete=1"
                    ).fetchone()[0],
                    2,
                )

    def test_raw_file_roundtrip_and_duplicate_ingest(self):
        from app.data_service.warehouse import Warehouse

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "players.csv"
            source.write_bytes(b"id,name\n1,A\n")
            with Warehouse(root / "football.sqlite3") as db:
                dataset = db.dataset("fifa-test", "game", "test", edition="FIFA test")
                first = db.asset(dataset, source, "players.csv")
                second = db.asset(dataset, source, "players.csv")
                self.assertEqual(first, second)
                self.assertEqual(db.read_raw(first), source.read_bytes())
                self.assertEqual(
                    db.connection.execute("SELECT COUNT(*) FROM assets").fetchone()[0],
                    1,
                )

    def test_changed_file_keeps_both_source_revisions(self):
        from app.data_service.warehouse import Warehouse

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            source.write_bytes(b'{"rating":80}')
            with Warehouse(root / "football.sqlite3") as db:
                dataset = db.dataset("ratings", "game", "test")
                first = db.asset(dataset, source, source.name)
                source.write_bytes(b'{"rating":81}')
                second = db.asset(dataset, source, source.name)
                self.assertNotEqual(first, second)
                self.assertEqual(db.read_raw(first), b'{"rating":80}')
                self.assertEqual(db.read_raw(second), b'{"rating":81}')
                self.assertEqual(db.asset(dataset, source, source.name), second)

    def test_duplicate_headers_and_irregular_rows_preserve_source_mapping(self):
        from app.data_service.warehouse import Warehouse

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            source.write_text("name,name,\nA,B,\nfooter\n")
            with Warehouse(root / "football.sqlite3") as db:
                dataset = db.dataset("csv", "football", "test")
                asset = db.asset(dataset, source, source.name)
                db.csv_table(asset, source, "results")
                self.assertEqual(
                    tuple(
                        db.connection.execute(
                            "SELECT name,name_2,unnamed_3 FROM results"
                        ).fetchone()
                    ),
                    ("A", "B", None),
                )
                self.assertEqual(
                    db.connection.execute(
                        "SELECT COUNT(*) FROM csv_columns"
                    ).fetchone()[0],
                    3,
                )
                self.assertEqual(
                    json.loads(
                        db.connection.execute(
                            "SELECT payload FROM csv_irregular_rows"
                        ).fetchone()[0]
                    ),
                    ["footer"],
                )

    def test_semicolon_legacy_csv_keeps_accents_and_fields(self):
        from app.data_service.warehouse import Warehouse

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "fm19.csv"
            source.write_bytes("Name;Club;Age\nRayane;Béziers;26\n".encode("latin-1"))
            with Warehouse(root / "football.sqlite3") as db:
                asset = db.asset(db.dataset("fm19", "game", "fm"), source, source.name)
                db.csv_table(asset, source, "fm19")
                self.assertEqual(
                    tuple(
                        db.connection.execute(
                            "SELECT Name,Club,Age FROM fm19"
                        ).fetchone()
                    ),
                    ("Rayane", "Béziers", "26"),
                )
                self.assertEqual(
                    db.connection.execute(
                        "SELECT encoding FROM asset_encodings"
                    ).fetchone()[0],
                    "latin-1",
                )

    def test_chunked_export_roundtrip(self):
        from app.data_service.warehouse import Warehouse
        from app.data_service.database import export_asset, verify_database

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            payload = bytes(range(256)) * 9000 + b"END"
            source.write_bytes(payload)
            database = root / "football.sqlite3"
            with Warehouse(database) as db:
                asset = db.asset(
                    db.dataset("source", "football", "test"), source, source.name
                )
                self.assertGreater(
                    db.connection.execute("SELECT COUNT(*) FROM raw_chunks").fetchone()[
                        0
                    ],
                    1,
                )
            destination = root / "export"
            export_asset(database, asset, destination)
            self.assertEqual(destination.read_bytes(), payload)
            self.assertEqual(verify_database(database)["quick_check"], ["ok"])
            with self.assertRaises(FileExistsError):
                export_asset(database, asset, destination)

    def test_metrica_preamble_is_metadata_and_coordinates_have_named_axes(self):
        from app.data_service.build_warehouse import ingest_metrica_csv
        from app.data_service.warehouse import Warehouse

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "tracking.csv"
            source.write_text(
                ",,,Home,,\n,,,1,,\nPeriod,Frame,Time [s],Player1,,Ball,\n1,1,0.04,0.5,0.2,0.3,0.4\n1,2,0.08,,,0.4,0.5\n"
            )
            with Warehouse(root / "football.sqlite3") as db:
                asset = db.asset(
                    db.dataset("metrica", "football", "metrica"), source, source.name
                )
                ingest_metrica_csv(db, asset, source)
                rows = db.connection.execute(
                    "SELECT Frame,Player1_x,Player1_y FROM metrica_tracking_samples ORDER BY row_number"
                ).fetchall()
                self.assertEqual(
                    [tuple(r) for r in rows], [("1", "0.5", "0.2"), ("2", None, None)]
                )
                self.assertEqual(
                    db.connection.execute(
                        "SELECT COUNT(*) FROM csv_irregular_rows"
                    ).fetchone()[0],
                    0,
                )
                self.assertEqual(
                    db.connection.execute(
                        "SELECT COUNT(*) FROM csv_preamble"
                    ).fetchone()[0],
                    2,
                )

    def test_csv_schema_preserves_missing_and_source_versions(self):
        from app.data_service.warehouse import Warehouse

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "players.csv"
            source.write_text(
                "person_id,name,current_ability,potential_ability\n1,A,,150\n"
            )
            with Warehouse(root / "football.sqlite3") as db:
                dataset = db.dataset("FM24.3", "game", "fm", edition="24.3.0")
                asset = db.asset(dataset, source, "players.csv")
                db.csv_table(asset, source, "fm_players")
                row = db.connection.execute(
                    "SELECT person_id,current_ability,potential_ability FROM fm_players"
                ).fetchone()
                self.assertEqual(tuple(row), ("1", None, "150"))
                self.assertEqual(
                    db.connection.execute("SELECT edition FROM datasets").fetchone()[0],
                    "24.3.0",
                )

    def test_pose_joints_are_queryable_from_database_frames(self):
        from app.data_service.warehouse import Warehouse

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "frame.json"
            payload = {
                "player_data": [
                    {
                        "player_id": 7,
                        "joints": {"lAnkle": {"xyz": [1, 2, 3], "p90_mae_cm": 4}},
                    }
                ]
            }
            source.write_text(json.dumps(payload))
            with Warehouse(root / "football.sqlite3") as db:
                dataset = db.dataset("pose", "tracking", "skillcorner")
                asset = db.asset(dataset, source, "frame.json")
                db.pose_frame(asset, "1", 1, 1, "00:00:00", payload)
                row = db.connection.execute(
                    "SELECT player_id,joint_name,x,y,z FROM pose_joints"
                ).fetchone()
                self.assertEqual(tuple(row), (7, "lAnkle", 1, 2, 3))

    def test_game_profiles_do_not_merge_equal_ids_across_providers(self):
        from app.data_service.warehouse import Warehouse

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "row.json"
            source.write_text("{}")
            with Warehouse(root / "football.sqlite3") as db:
                for provider, edition, rating in [
                    ("fm", "24.3", 160),
                    ("fifa", "23", 86),
                ]:
                    dataset = db.dataset(provider, "game", provider, edition=edition)
                    asset = db.asset(dataset, source, "row.json")
                    db.profile(
                        asset,
                        1,
                        provider,
                        "1",
                        name="A",
                        birth_date="1985-02-05",
                        overall=rating,
                        scale="0-200" if provider == "fm" else "0-99",
                    )
                rows = db.connection.execute(
                    "SELECT provider,player_id,overall,scale FROM game_profiles ORDER BY provider"
                ).fetchall()
                self.assertEqual(len(rows), 2)
                self.assertEqual({row["scale"] for row in rows}, {"0-200", "0-99"})


if __name__ == "__main__":
    unittest.main()
