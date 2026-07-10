from __future__ import annotations

import unittest

from app.web.operations import build_operations_summary


class TestOperationsSummary(unittest.TestCase):
    def test_builds_healthy_summary_for_covered_prediction_run(self) -> None:
        summary = build_operations_summary(
            [
                {"competition": "PL", "confidence": 0.72},
                {"competition": "PL", "confidence": 0.61},
                {"competition": "SA", "confidence": 0.54},
            ],
            [{"league": "Premier League"}],
            generated_at="2026-07-09T00:00:00+00:00",
        )

        self.assertEqual(summary["status"], "healthy")
        self.assertEqual(summary["prediction_count"], 3)
        self.assertEqual(summary["score_count"], 1)
        self.assertEqual(summary["competition_count"], 2)
        self.assertEqual(summary["average_confidence"], 0.6233)
        self.assertEqual(summary["high_confidence_count"], 2)
        self.assertEqual(summary["low_confidence_count"], 0)
        self.assertEqual(
            summary["coverage_by_competition"],
            [
                {"competition": "PL", "prediction_count": 2},
                {"competition": "SA", "prediction_count": 1},
            ],
        )
        self.assertEqual(summary["alerts"], [])

    def test_flags_empty_and_low_quality_exports(self) -> None:
        empty = build_operations_summary([], [])
        self.assertEqual(empty["status"], "attention")
        self.assertEqual({alert["code"] for alert in empty["alerts"]}, {"NO_PREDICTIONS", "NO_SCORES"})

        weak = build_operations_summary(
            [
                {"home_team": "A", "away_team": "B", "confidence": 0.42},
                {"home_team": "C", "away_team": "D", "confidence": "bad"},
            ],
            [{"league": "Premier League"}],
        )
        self.assertEqual(weak["status"], "attention")
        self.assertIn("MISSING_CONFIDENCE", {alert["code"] for alert in weak["alerts"]})
        self.assertIn("NO_HIGH_CONFIDENCE_PICKS", {alert["code"] for alert in weak["alerts"]})
        self.assertIn("MISSING_COMPETITION_LABELS", {alert["code"] for alert in weak["alerts"]})


if __name__ == "__main__":
    unittest.main(verbosity=2)
