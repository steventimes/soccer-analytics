import unittest
from app.analytics.switching_play import pass_windows


class TestSwitchingPlay(unittest.TestCase):
    def event(self, kind, time, **fields):
        return {
            "type": {"name": kind},
            "timestamp": f"00:00:{time:06.3f}",
            "period": 1,
            "possession": 1,
            "possession_team": {"id": 1},
            "team": {"id": 1},
            **fields,
        }

    def test_outcome_starts_after_pass_end_and_stays_in_same_possession(self):
        ball = self.event(
            "Pass",
            1,
            id="p",
            location=[50, 10],
            duration=2,
            **{"pass": {"end_location": [60, 70]}},
        )
        shot = self.event("Shot", 13, shot={"type": {"name": "Open Play"}})
        rows, _ = pass_windows([ball, shot])
        self.assertTrue(rows[0]["cross_field"])
        self.assertEqual(rows[0]["shot_within10s"], 1)
        for changed in [
            dict(shot, possession=2),
            dict(shot, period=2),
            dict(shot, team={"id": 2}),
            dict(shot, timestamp="00:00:13.001"),
        ]:
            rows, _ = pass_windows([ball, changed])
            self.assertEqual(rows[0]["shot_within10s"], 0)
        rows, exclusions = pass_windows([ball, dict(shot, timestamp="00:00:02.000")])
        self.assertEqual(rows, [])
        self.assertEqual(exclusions["shot_already_observed"], 1)

    def test_incomplete_set_piece_and_penalty_are_not_positive_controls(self):
        ball = self.event(
            "Pass",
            1,
            location=[50, 10],
            duration=1,
            **{"pass": {"end_location": [60, 70]}},
        )
        penalty = self.event("Shot", 4, shot={"type": {"name": "Penalty"}})
        rows, _ = pass_windows([ball, penalty])
        self.assertEqual(rows[0]["shot_within10s"], 0)
        for detail in [
            {"outcome": {"name": "Incomplete"}},
            {"type": {"name": "Corner"}},
        ]:
            changed = dict(ball, **{"pass": {**ball["pass"], **detail}})
            self.assertEqual(pass_windows([changed])[0], [])
