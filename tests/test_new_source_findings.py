import unittest
import importlib.util

from app.analytics.new_source_findings import (
    consecutive_pairs,
    possession_eligible,
    restore_situations,
    correlation,
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
