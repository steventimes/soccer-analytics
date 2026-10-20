import json
from pathlib import Path
import tempfile
import unittest
from app.analytics.cross_validation import (
    same_name,
    choose_snapshot,
    quantile_histogram,
    validate,
)


class TestCrossValidation(unittest.TestCase):
    def test_name_matching_keeps_identity_boundaries(self):
        self.assertTrue(same_name("Álvaro García", "Alvaro José Garcia"))
        self.assertFalse(same_name("A. García", "Alvaro Garcia"))
        self.assertFalse(same_name("Alvaro Garcia", "Alvaro Smith"))

    def test_future_and_undated_snapshots_are_excluded(self):
        rows = [
            {"effective_date": None, "asset_id": 3, "row_number": 1},
            {"effective_date": "2025-01-01", "asset_id": 4, "row_number": 1},
            {"effective_date": "bad-date", "asset_id": 5, "row_number": 1},
            {"effective_date": "2026-01-01", "asset_id": 2, "row_number": 1},
            {"effective_date": "2024-03-15", "asset_id": 1, "row_number": 1},
        ]
        self.assertEqual(choose_snapshot(rows, "2025-01-01")["asset_id"], 1)
        self.assertIsNone(choose_snapshot(rows, "2023-01-01"))

    def test_histogram_quantile_preserves_empty_observation(self):
        self.assertIsNone(quantile_histogram([0] * 10, 0.99))
        self.assertEqual(quantile_histogram([9, 1], 0.99), 0.15)

    def test_database_link_uses_pre_match_snapshot_and_records_evidence(self):
        from app.data_service.warehouse import Warehouse

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "football.sqlite3"
            source = root / "players.csv"
            source.write_text(
                "person_id,full_name,birth_date,current_ability,pace,stamina,off_the_ball\n1,Alvaro Jose Garcia,1990-01-01,150,15,14,13\n"
            )
            with Warehouse(database) as db:
                for name, effective in [
                    ("old", "2024-03-15"),
                    ("future", "2026-01-01"),
                ]:
                    asset = db.asset(
                        db.dataset(name, "game", "fm"), source, source.name
                    )
                    db.csv_table(asset, source, "fm_players")
                    db.profile(
                        asset,
                        1,
                        "fm",
                        "1",
                        name="Alvaro Jose Garcia",
                        birth_date="1990-01-01",
                        overall=150,
                        scale="0-200",
                        effective_date=effective,
                    )
            meta = root / "open-data/skillcorner/data/matches/1/1_match.json"
            meta.parent.mkdir(parents=True)
            meta.write_text(
                json.dumps(
                    {
                        "id": 1,
                        "date_time": "2025-01-01T00:00:00Z",
                        "home_team": {"id": 1, "name": "A"},
                        "away_team": {"id": 2, "name": "B"},
                        "players": [
                            {
                                "id": 7,
                                "first_name": "Álvaro",
                                "last_name": "García",
                                "birthday": "1990-01-01",
                                "team_id": 1,
                                "player_role": {"position_group": "Center Forward"},
                                "playing_time": {"total": {"minutes_played": 60}},
                            }
                        ],
                    }
                )
            )
            report = root / "scouting/movement/skillcorner-1.json"
            report.parent.mkdir(parents=True)
            report.write_text(
                json.dumps(
                    {
                        "players": [
                            {
                                "player_id": 7,
                                "observed_segment_seconds": 2400,
                                "speed_histogram": [1, 9],
                                "distance_m": 4000,
                            }
                        ],
                        "dynamic_events": {
                            "players": [{"player_id": 7, "off_ball_runs": 20}]
                        },
                    }
                )
            )
            result = validate(database, root, root / "report.json")
            self.assertEqual(result["identity_audit"]["fm_linked"], 1)
            import sqlite3

            with sqlite3.connect(database) as db:
                evidence = json.loads(
                    db.execute("SELECT evidence FROM game_real_links").fetchone()[0]
                )
                self.assertEqual(evidence["effective_date"], "2024-03-15")
                self.assertEqual(
                    db.execute("SELECT COUNT(*) FROM real_observations").fetchone()[0],
                    1,
                )


class TestPublicEventValidation(unittest.TestCase):
    def test_json_array_stream_handles_boundaries_and_unicode(self):
        import io
        from app.data_service.wyscout import json_items

        payload = [{"name": "Álvaro", "content": "x" * 70000}, {"id": 2}, 42]
        self.assertEqual(list(json_items(io.StringIO(json.dumps(payload)))), payload)

    def test_json_array_stream_waits_for_complete_numeric_tokens(self):
        import io
        from app.data_service.wyscout import json_items

        class SingleCharacterStream(io.StringIO):
            def read(self, size=-1):
                return super().read(1)

        self.assertEqual(
            list(json_items(SingleCharacterStream("[1.25e30,-3.5,0.5]"))),
            [1.25e30, -3.5, 0.5],
        )

    def test_json_array_stream_rejects_truncated_or_invalid_source(self):
        import io
        from app.data_service.wyscout import json_items

        for payload in ['[{"id":1}', "[1,]", "[1] trailing", "[1 2]"]:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                list(json_items(io.StringIO(payload)))

    def test_pass_accuracy_preserves_unknown_labels_and_native_geometry(self):
        from app.analytics.event_validation import event_metrics

        rows = [
            {
                "eventName": "Pass",
                "subEventName": "Simple pass",
                "matchId": 1,
                "playerId": 7,
                "matchPeriod": "1H",
                "tags": [{"id": 1801}],
                "positions": [{"x": 20, "y": 50}, {"x": 60, "y": 50}],
            },
            {
                "eventName": "Pass",
                "subEventName": "Simple pass",
                "matchId": 1,
                "playerId": 7,
                "matchPeriod": "2H",
                "tags": [{"id": 1802}],
                "positions": [{"x": 60, "y": 50}, {"x": 90, "y": 50}],
            },
            {
                "eventName": "Pass",
                "subEventName": "Simple pass",
                "matchId": 1,
                "playerId": 7,
                "matchPeriod": "2H",
                "tags": [],
                "positions": [],
            },
        ]
        result, errors = event_metrics(rows)
        self.assertEqual(result[(1, 7)]["passes"], 2)
        self.assertEqual(result[(1, 7)]["accurate_passes"], 1)
        self.assertEqual(result[(1, 7)]["long_passes"], 2)
        self.assertEqual(result[(1, 7)]["progressive_passes"], 1)
        self.assertEqual(errors["unknown_or_conflicting_accuracy"], 1)


if __name__ == "__main__":
    unittest.main()
