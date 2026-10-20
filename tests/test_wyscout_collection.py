import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from app.data_service.wyscout import collect


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
