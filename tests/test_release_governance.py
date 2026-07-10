from __future__ import annotations

import unittest

from app.web.release_governance import build_release_governance


class TestReleaseGovernance(unittest.TestCase):
    def test_approves_healthy_prediction_release_with_disclosure(self) -> None:
        release = build_release_governance(
            [
                {"competition": "PL", "confidence": 0.72},
                {"competition": "PL", "confidence": 0.61},
                {"competition": "SA", "confidence": 0.56},
            ],
            {
                "status": "healthy",
                "average_confidence": 0.63,
                "high_confidence_count": 2,
                "alerts": [],
            },
            generated_at="2026-07-09T00:00:00+00:00",
        )

        self.assertEqual(release["schema"], "SoccerAnalytics.ReleaseGovernance.v1")
        self.assertEqual(release["release_decision"], "approved")
        self.assertTrue(release["can_publish_recommendations"])
        self.assertEqual(release["blockers"], [])
        self.assertIn("not guaranteed outcomes", release["risk_disclosure"]["summary"])
        self.assertIn("node --check docs/app.js", release["verification_commands"])

    def test_blocks_weak_or_unhealthy_prediction_release(self) -> None:
        release = build_release_governance(
            [{"competition": "PL", "confidence": 0.42}],
            {
                "status": "attention",
                "average_confidence": 0.42,
                "high_confidence_count": 0,
                "alerts": [
                    {"level": "critical", "code": "NO_PREDICTIONS", "message": "No predictions exported."},
                    {"level": "warning", "code": "NO_HIGH_CONFIDENCE_PICKS", "message": "No strong picks."},
                ],
            },
        )

        codes = {blocker["code"] for blocker in release["blockers"]}
        self.assertEqual(release["release_decision"], "blocked")
        self.assertFalse(release["can_publish_recommendations"])
        self.assertIn("INSUFFICIENT_FIXTURE_COVERAGE", codes)
        self.assertIn("OPERATIONS_NOT_HEALTHY", codes)
        self.assertIn("NO_PUBLISHABLE_PICKS", codes)
        self.assertIn("LOW_AVERAGE_CONFIDENCE", codes)
        self.assertIn("CRITICAL_ALERT_NO_PREDICTIONS", codes)
        self.assertEqual(release["warnings"][0]["code"], "NO_HIGH_CONFIDENCE_PICKS")


if __name__ == "__main__":
    unittest.main(verbosity=2)
