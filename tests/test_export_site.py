from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.web import export_site


class TestExportSiteData(unittest.TestCase):
    def test_export_writes_dashboard_payloads_and_manifest(self) -> None:
        predictions = [
            {
                "competition": "PL",
                "utc_date": "2026-07-10T19:00:00Z",
                "home_team": "North FC",
                "away_team": "South FC",
                "prediction": "Win",
                "confidence": 0.72,
                "probabilities": {"Win": 0.72, "Draw": 0.18, "Loss": 0.10},
            }
        ]
        scores = [
            {
                "league": "Premier League",
                "date": "2026-07-09",
                "time": "20:00",
                "home_team": "East FC",
                "away_team": "West FC",
                "home_score": 2,
                "away_score": 1,
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "docs" / "data"
            repo_root = Path(tmpdir)

            with patch.object(export_site, "data_dir", return_value=data_dir), patch.object(
                export_site, "repo_root", return_value=repo_root
            ), patch.object(
                export_site, "_generate_predictions", return_value=predictions
            ) as generate_predictions, patch.object(
                export_site, "fetch_daily_scores", return_value=scores
            ):
                outputs = export_site.export_site_data(days=4)

            self.assertEqual(
                set(outputs),
                {"manifest", "predictions", "preset_questions", "scores", "operations", "release_governance"},
            )
            generate_predictions.assert_called_once_with(
                models_dir=repo_root / "models",
                days=4,
            )

            manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
            predictions_payload = json.loads(outputs["predictions"].read_text(encoding="utf-8"))
            preset_payload = json.loads(outputs["preset_questions"].read_text(encoding="utf-8"))
            scores_payload = json.loads(outputs["scores"].read_text(encoding="utf-8"))
            operations_payload = json.loads(outputs["operations"].read_text(encoding="utf-8"))
            release_payload = json.loads(outputs["release_governance"].read_text(encoding="utf-8"))

            self.assertEqual(manifest["prediction_count"], 1)
            self.assertIn("generated_at", manifest)
            self.assertEqual(manifest["files"]["predictions"], "docs/data/predictions.json")
            self.assertEqual(manifest["files"]["operations"], "docs/data/operations.json")
            self.assertEqual(manifest["files"]["release_governance"], "docs/data/release.json")
            self.assertEqual(predictions_payload["predictions"], predictions)
            self.assertEqual(scores_payload["scores"], scores)
            self.assertEqual(operations_payload["status"], "healthy")
            self.assertEqual(operations_payload["prediction_count"], 1)
            self.assertEqual(operations_payload["score_count"], 1)
            self.assertEqual(release_payload["schema"], "SoccerAnalytics.ReleaseGovernance.v1")
            self.assertEqual(release_payload["release_decision"], "blocked")
            self.assertFalse(release_payload["can_publish_recommendations"])
            self.assertIn("INSUFFICIENT_FIXTURE_COVERAGE", {blocker["code"] for blocker in release_payload["blockers"]})

            recommended = next(
                question
                for question in preset_payload["questions"]
                if question["id"] == "recommended_bets_today"
            )
            self.assertEqual(recommended["items"][0]["match"], "North FC vs South FC")
            self.assertEqual(recommended["items"][0]["confidence"], 0.72)

    def test_export_preserves_same_day_preset_cache(self) -> None:
        cached_payload = {
            "date": export_site.build_preset_questions([])["date"],
            "generated_at": "2026-07-09T00:00:00+00:00",
            "questions": [
                {
                    "id": "recommended_bets_today",
                    "title": "Recommended picks today",
                    "answer": "cached",
                    "items": [],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "docs" / "data"
            data_dir.mkdir(parents=True)
            preset_path = data_dir / "preset_questions.json"
            preset_path.write_text(json.dumps(cached_payload), encoding="utf-8")

            with patch.object(export_site, "data_dir", return_value=data_dir), patch.object(
                export_site, "repo_root", return_value=Path(tmpdir)
            ), patch.object(
                export_site, "_generate_predictions", return_value=[]
            ), patch.object(
                export_site, "fetch_daily_scores", return_value=[]
            ):
                export_site.export_site_data()

            self.assertEqual(
                json.loads(preset_path.read_text(encoding="utf-8")),
                cached_payload,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
