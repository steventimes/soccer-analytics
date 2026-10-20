import unittest
import json
import tempfile
from pathlib import Path
from app.data_service.warehouse import Warehouse
from app.data_service.build_warehouse import document
from app.analytics.switch_context import run

from app.analytics.switch_context import (
    snapshot_context,
    continuous_motion,
    tracked_context,
)


class TestSwitchContext(unittest.TestCase):
    def test_metadata_identity_cannot_name_cache_paths_or_another_match(self):
        for identity in ["../escape", True, 0, 2]:
            with self.subTest(identity=identity), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / "match.json"
                source.write_text(json.dumps({"id": identity}))
                database = root / "warehouse.sqlite3"
                with Warehouse(database) as db:
                    ds = db.dataset("skillcorner", "football", "skillcorner")
                    aid = db.asset(ds, source, "data/matches/1/1_match.json")
                    document(db, aid, source, "skillcorner_document")
                with self.assertRaises(ValueError):
                    run(database, root / "output")
                self.assertFalse((root / "escape.json").exists())

    def test_snapshot_keeps_invisible_destination_and_missing_frames_explicit(self):
        row = {"start": [20, 10], "end": [50, 70]}
        frame = {
            "visible_area": [0, 0, 60, 0, 60, 30, 0, 30],
            "freeze_frame": [
                {"teammate": True, "actor": True, "location": [20, 10]},
                {"teammate": True, "location": [24, 12]},
                {"teammate": False, "location": [22, 11]},
            ],
        }
        context = snapshot_context(row, frame)
        self.assertTrue(context["origin_local_visible"])
        self.assertFalse(context["target_local_visible"])
        self.assertEqual(context["origin_opponents_within5_units"], 1)
        self.assertEqual(context["teammates_visible_excluding_passer"], 1)
        self.assertFalse(context["continuous_runs_available"])
        self.assertEqual(snapshot_context(row, None), {"snapshot_available": False})

    def frames(self):
        return [
            {
                "frame": i,
                "period": 1,
                "timestamp": i / 10,
                "ball_data": {"x": 0, "y": -20, "is_detected": True},
                "possession": {"group": "home team"},
                "player_data": [
                    {"player_id": 1, "x": 0, "y": -20, "is_detected": True},
                    {"player_id": 2, "x": i / 5, "y": 20, "is_detected": True},
                    {"player_id": 3, "x": 10 - i / 5, "y": -20, "is_detected": True},
                ],
            }
            for i in range(21)
        ]

    def test_continuous_motion_rejects_hidden_frames_gaps_and_spikes(self):
        frames = self.frames()
        self.assertAlmostEqual(continuous_motion(frames, 2, 1)["dx"], 4)
        frames[10]["player_data"][1]["is_detected"] = False
        self.assertIsNone(continuous_motion(frames, 2, 1))
        frames = self.frames()
        frames[10]["timestamp"] = 2
        self.assertIsNone(continuous_motion(frames, 2, 1))
        frames = self.frames()
        frames[10]["player_data"][1]["x"] = 40
        self.assertIsNone(continuous_motion(frames, 2, 1))

    def test_detected_far_side_motion_closing_and_attacking_direction(self):
        case = {
            "pass_frame": 10,
            "period": 1,
            "attacking_side": "left_to_right",
            "team_id": "10",
            "player_id": "1",
            "x": 0,
            "y": -20,
        }
        frames = self.frames()
        snapshots = {(1, r["frame"]): r for r in frames}
        metadata = {
            "home_team": {"id": 10},
            "pitch_width": 68,
            "players": [
                {"id": 1, "team_id": 10},
                {"id": 2, "team_id": 10},
                {"id": 3, "team_id": 20},
            ],
        }
        r = tracked_context(case, snapshots, metadata)
        self.assertTrue(r["usable_ball_anchor"])
        self.assertEqual(r["observed_movements"][0]["side_at_start"], "far_side")
        self.assertAlmostEqual(r["opponent_closing"][0]["closing_speed_mps_proxy"], 2)
        reverse = dict(case, attacking_side="right_to_left", y=20)
        r = tracked_context(reverse, snapshots, metadata)
        self.assertTrue(r["usable_ball_anchor"])
        self.assertAlmostEqual(continuous_motion(frames, 2, -1)["dx"], -4)
        self.assertEqual(r["observed_movements"], [])
        self.assertEqual(
            r["continuous_movement_counts"]["far_side_players_continuously_observed"], 1
        )

    def test_unobserved_or_misaligned_movement_is_not_zero_runs(self):
        case = {
            "pass_frame": 10,
            "period": 1,
            "attacking_side": "left_to_right",
            "team_id": "10",
            "player_id": "1",
            "x": 40,
            "y": -20,
        }
        metadata = {
            "home_team": {"id": 10},
            "pitch_width": 68,
            "players": [
                {"id": 1, "team_id": 10},
                {"id": 2, "team_id": 10},
                {"id": 3, "team_id": 20},
            ],
        }
        snapshots = {(1, r["frame"]): r for r in self.frames()}
        r = tracked_context(case, snapshots, metadata)
        self.assertFalse(r["usable_ball_anchor"])
        self.assertEqual(r["continuous_movement_counts"], {})
        self.assertEqual(r["movement_coverage"], "no_complete_teammate_window")

    def test_possession_change_keeps_recovery_distinct_from_attacking_support(self):
        case = {
            "pass_frame": 10,
            "period": 1,
            "attacking_side": "left_to_right",
            "team_id": "10",
            "player_id": "1",
            "x": 0,
            "y": -20,
        }
        metadata = {
            "home_team": {"id": 10},
            "pitch_width": 68,
            "players": [
                {"id": 1, "team_id": 10},
                {"id": 2, "team_id": 10},
                {"id": 3, "team_id": 20},
            ],
        }
        frames = self.frames()
        frames[15]["possession"]["group"] = "away team"
        r = tracked_context(case, {(1, f["frame"]): f for f in frames}, metadata)
        self.assertEqual(r["movement_window_team_context"], "possession_changed")
        self.assertAlmostEqual(r["observed_movements"][0]["pre_release_dx"], 2)
        self.assertAlmostEqual(r["observed_movements"][0]["post_release_dx"], 2)
