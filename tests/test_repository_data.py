import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from app.data_service.repository_data import ingest


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
            report = ingest(database, [root])
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
                ingest(Path(tmp) / "data.sqlite3", [root])

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
            report = ingest(database, [root])
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


if __name__ == "__main__":
    unittest.main()
