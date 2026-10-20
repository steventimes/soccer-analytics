import unittest

from app.analytics.movement_lab import (
    rotation_candidates,
    smooth_segments,
    peak_windows,
    external_load,
)


class MovementLabTests(unittest.TestCase):
    def test_rotations_require_persistent_observed_exchange(self):
        snapshots = []
        for second, y in enumerate([-5, -4, 0, 4, 5]):
            snapshots.append(
                {
                    "period": 1,
                    "frame": second * 10,
                    "owner": 1,
                    "teams": {1: {10: (0, y), 11: (0, -y)}},
                }
            )
        candidates = rotation_candidates(snapshots)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["axis"], "lateral")
        self.assertEqual(candidates[0]["decision_frame"], 40)
        snapshots[2]["teams"][1].pop(10)
        self.assertEqual(rotation_candidates(snapshots), [])

    def test_halftime_and_collective_translation_do_not_make_rotation(self):
        snapshots = [
            {
                "period": 1 if i < 3 else 2,
                "frame": i * 10,
                "owner": 1,
                "teams": {1: {10: (i, -5), 11: (i, 5)}},
            }
            for i in range(5)
        ]
        self.assertEqual(rotation_candidates(snapshots), [])

    def test_smoothing_does_not_bridge_missing_detection(self):
        points = [(1, i, i / 10, i * 0.8, 0) for i in range(16)]
        result = smooth_segments(points)
        self.assertAlmostEqual(result[0]["speed_mps"], 8)
        missing = smooth_segments([p for p in points if p[1] != 3])
        self.assertTrue(all(p["frame"] > 5 for p in missing))

    def test_peak_window_requires_observed_exposure(self):
        segments = [
            {
                "period": 1,
                "frame": i * 5,
                "seconds": i * 0.5,
                "duration": 0.5,
                "speed_mps": 8.0,
                "acceleration_mps2": 0.0,
            }
            for i in range(1, 121)
        ]
        peak = peak_windows(segments, (60,))["60"]
        self.assertAlmostEqual(peak["distance_m_per_minute"], 480)
        self.assertEqual(peak["eligible_windows"], 1)
        self.assertIsNone(
            peak_windows(segments[::2], (60,))["60"]["distance_m_per_minute"]
        )

    def test_unknown_recovery_does_not_join_repeated_sprints(self):
        segments = [
            {
                "period": 1,
                "frame": i * 5,
                "seconds": i * 0.5,
                "duration": 0.5,
                "speed_mps": 8.0 if i % 6 in (1, 2) else 1.0,
                "acceleration_mps2": 0.0,
            }
            for i in range(1, 19)
        ]
        self.assertEqual(external_load(segments, 1)["repeated_sprint_clusters_ge3"], 1)
        only_sprints = [r for r in segments if r["speed_mps"] == 8]
        self.assertEqual(
            external_load(only_sprints, 1)["repeated_sprint_clusters_ge3"], 0
        )


if __name__ == "__main__":
    unittest.main()
