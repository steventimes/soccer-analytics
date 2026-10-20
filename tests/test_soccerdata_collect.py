import csv
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.data_service.soccerdata_collect import explicit_seasons, export_frame


class TestSoccerdataExport(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
