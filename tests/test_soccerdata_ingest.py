import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest

from app.data_service.soccerdata_ingest import ingest
from app.data_service.build_warehouse import build


class TestSoccerdataIngest(unittest.TestCase):
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
                ingest(database, root)
            with sqlite3.connect(database) as connection:
                paths = [r[0] for r in connection.execute("SELECT path FROM assets")]
                self.assertFalse(any("private.log" in p for p in paths))
            return
        if general:
            build(database, root.parent)
        else:
            assert ingest(database, root)["frame_rows_by_provider"] == {"ESPN": 2}
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
