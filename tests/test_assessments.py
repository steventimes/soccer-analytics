import unittest
import json
import tempfile
from pathlib import Path


class TestAssessments(unittest.TestCase):
    def test_spatial_style_keeps_axes_endpoints_and_exposure_distinct(self):
        from app.analytics.spatial_style_data import build_heatmaps

        observations = [{"entity_id": "1", "match_id": "10", "date": "2017-10-01", "league": "England",
                         "team": "A", "position": "Defender", "minutes": 90},
                        {"entity_id": "1", "match_id": "20", "date": "2018-04-01", "league": "England",
                         "team": "A", "position": "Defender", "minutes": 90},
                        {"entity_id": "1", "match_id": "21", "date": "2018-05-01", "league": "England",
                         "team": "A", "position": "Defender", "minutes": 90}]
        def event(identity, match, start, end, **extra):
            return {"id": identity, "matchId": match, "playerId": 1, "teamId": "A", "matchPeriod": "1H",
                    "eventName": "Pass", "subEventName": "Simple pass", "positions": [start, end], **extra}
        start, end = {"x": 0, "y": 100}, {"x": 100, "y": 0}
        early = event(1, 10, start, end)
        data = build_heatmaps([early, early, event(2, 20, start, end),
                               event(3, 10, start, {"x": float("nan"), "y": 0}),
                               event(4, 10, start, end, eventName="Free Kick"),
                               event(5, 10, start, end, teamId="Other")], observations, minimum_minutes=90, bins=2, sigma=0)
        self.assertEqual(data["maps"]["early"]["passes"].shape, (1, 2, 2, 2))
        self.assertEqual(data["maps"]["early"]["passes"][0, 0, 1, 0], 1)
        self.assertEqual(data["maps"]["early"]["passes"][0, 1, 0, 1], 1)
        self.assertEqual(data["maps"]["late"]["passes"][0, 0, 1, 0], .5)
        self.assertEqual(data["audit"]["duplicate_events"], 1)
        self.assertEqual(data["audit"]["invalid_coordinates"], 1)
        self.assertEqual(data["audit"]["team_mismatch"], 1)
        self.assertEqual(data["audit"]["retained_passes"], 2)

    def test_spatial_style_smoothing_keeps_mass_and_rejects_zero_exposure(self):
        import numpy as np
        from app.analytics.spatial_style_data import normalize_heatmaps

        counts = np.zeros((2, 1, 5, 5)); counts[:, 0, 0, 4] = [2, 4]
        smoothed = normalize_heatmaps(counts, np.array([90, 180]), sigma=1)
        np.testing.assert_allclose(smoothed.sum(axis=(1, 2, 3)), [2, 2])
        np.testing.assert_allclose(smoothed[0], smoothed[1])
        self.assertGreater(smoothed[0, 0, 0, 3], 0)
        with self.assertRaisesRegex(ValueError, "exposure"):
            normalize_heatmaps(counts, np.array([0, 180]), sigma=1)

    def test_spatial_retrieval_shares_ties_and_respects_candidate_pools(self):
        import numpy as np
        from app.analytics.spatial_style import retrieval_scores

        players = [{"league": "England", "position": "Defender"},
                   {"league": "England", "position": "Defender"},
                   {"league": "Spain", "position": "Forward"}]
        scores = retrieval_scores(np.zeros((3, 2)), np.zeros((3, 2)), players)
        self.assertAlmostEqual(scores["same_league"]["top1"], 2 / 3)
        self.assertAlmostEqual(scores["same_league"]["mrr"], 5 / 6)
        self.assertAlmostEqual(scores["same_league"]["random_top1"], 2 / 3)
        self.assertEqual(scores["same_league"]["queries"], 3)
        self.assertEqual(scores["same_position"]["candidate_min"], 1)

    def test_spatial_nmf_basis_never_fits_later_period(self):
        import numpy as np
        from app.analytics.spatial_style import fit_style_vectors

        rng = np.random.default_rng(7)
        early = {kind: rng.uniform(.1, 2, (6, 2 if kind == "passes" else 1, 3, 3))
                 for kind in ("shots", "crosses", "take_ons", "passes")}
        late = {kind: values * 1.2 for kind, values in early.items()}
        first = fit_style_vectors(early, late, seed=7)
        changed = fit_style_vectors(early, {kind: values + 10 for kind, values in late.items()}, seed=7)
        np.testing.assert_array_equal(first["early"], changed["early"])
        for kind in early:
            np.testing.assert_array_equal(first["basis"][kind], changed["basis"][kind])
        self.assertEqual(first["early"].shape, (6, 18))
        self.assertFalse(np.allclose(first["late"], changed["late"]))

    def test_cache_rebuilds_for_changed_sources_and_tampered_content(self):
        from app.analytics.artifacts import cached_json

        built = []

        def build():
            built.append(1)
            return {"version": len(built)}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.json"
            self.assertEqual(cached_json(path, "a", build)["version"], 1)
            self.assertEqual(cached_json(path, "a", build)["version"], 1)
            self.assertEqual(cached_json(path, "b", build)["version"], 2)
            path.write_text(json.dumps({"version": 999}))
            self.assertEqual(cached_json(path, "b", build)["version"], 3)

    def test_offline_research_commands_do_not_choose_live_match_sources(self):
        from app.pipeline import build_parser

        for command in ("assess", "tactics", "movement-lab"):
            self.assertIsNone(build_parser().parse_args([command]).source)

    def test_beta_shrinkage_keeps_missing_observation_visible(self):
        from app.analytics.assessment_data import beta_estimate

        result = beta_estimate(0, 0, 0.6, 50)
        self.assertIsNone(result["observed"])
        self.assertTrue(result["prior_only"])
        self.assertAlmostEqual(result["posterior_mean"], 0.6)
        self.assertLess(beta_estimate(1, 1, 0.6, 50)["posterior_mean"], 0.7)

    def test_gamma_shrinkage_respects_minutes_and_unknown_exposure(self):
        from app.analytics.assessment_data import gamma_estimate

        self.assertAlmostEqual(gamma_estimate(3, 270, 0.3, 270)["posterior_mean"], 0.65)
        self.assertIsNone(gamma_estimate(0, 0, 0.3, 270)["observed"])

    def test_future_results_cannot_change_team_features(self):
        from app.analytics.assessment_data import team_examples

        matches = [
            {
                "id": i,
                "competition": "PL",
                "season": 2024,
                "date": f"2024-09-{i:02d}",
                "home_id": "a",
                "away_id": "b",
                "home": "A",
                "away": "B",
                "home_goals": 1,
                "away_goals": 0,
            }
            for i in range(1, 14)
        ]
        before = team_examples(matches, history=8, horizon=5)
        matches[-1]["home_goals"] = 20
        after = team_examples(matches, history=8, horizon=5)
        a = next(r for r in before if r["entity_id"] == "a")
        b = next(r for r in after if r["entity_id"] == "a")
        self.assertEqual(a["features"], b["features"])
        self.assertNotEqual(a["targets"], b["targets"])
        self.assertLess(a["feature_asof"], a["target_start"])

    def test_team_windows_do_not_cross_seasons(self):
        from app.analytics.assessment_data import team_examples

        matches = [
            {
                "id": i,
                "competition": "PL",
                "season": 2023 if i < 9 else 2024,
                "date": f"2024-09-{i:02d}",
                "home_id": "a",
                "away_id": "b",
                "home": "A",
                "away": "B",
                "home_goals": 1,
                "away_goals": 0,
            }
            for i in range(1, 14)
        ]
        self.assertEqual(team_examples(matches, history=8, horizon=5), [])

    def test_goalkeeper_minutes_use_lineups_and_substitutions(self):
        from app.analytics.assessment_data import lineup_minutes

        match = {
            "teamsData": {
                "a": {
                    "formation": {
                        "lineup": [{"playerId": 1}],
                        "bench": [{"playerId": 2}],
                        "substitutions": [
                            {"playerOut": 1, "playerIn": 2, "minute": 70}
                        ],
                    }
                }
            }
        }
        self.assertEqual(lineup_minutes(match, {}), {1: 70, 2: 20})
        self.assertEqual(lineup_minutes(match, {1: 50}), {1: 50, 2: 20})

    def test_assessment_cli_is_offline_and_keeps_source_unset(self):
        from app.pipeline import build_parser

        args = build_parser().parse_args(["assess", "--train-cap", "1000"])
        self.assertEqual(args.command, "assess")
        self.assertEqual(args.train_cap, 1000)
        self.assertIsNone(args.source)

    def test_player_prior_and_features_use_only_earlier_window(self):
        from app.analytics.assessment_data import (
            COUNT_FIELDS,
            player_panels,
            fit_priors,
            player_examples,
        )

        people = {1: {"firstName": "A", "lastName": "B", "birthDate": "1990-01-01"}}
        rows = []
        for when, goals in [("2017-09-01", 1), ("2017-11-01", 2)]:
            rows.append(
                {
                    "entity_id": "1",
                    "league": "England",
                    "position": "Forward",
                    "team": "a",
                    "date": when,
                    "minutes": 300,
                    **dict.fromkeys(COUNT_FIELDS, 0),
                }
            )
            rows[-1]["goals"] = goals
        panels = player_panels(rows, people)
        priors = fit_priors([p for p in panels if p["block"] == 0])
        before = player_examples(panels, priors)[0]
        rows[-1]["goals"] = 99
        panels = player_panels(rows, people)
        after = player_examples(
            panels, fit_priors([p for p in panels if p["block"] == 0])
        )[0]
        self.assertEqual(before["features"], after["features"])
        self.assertNotEqual(before["targets"], after["targets"])
        self.assertLess(before["feature_asof"], before["target_start"])

    def test_model_selection_uses_validation_only(self):
        from app.analytics.assessment_models import choose_model

        self.assertEqual(
            choose_model(
                [
                    {"model": "a", "validation_mae": 1, "test_mae": 0.1},
                    {"model": "b", "validation_mae": 0.5, "test_mae": 9},
                ]
            ),
            "b",
        )


if __name__ == "__main__":
    unittest.main()
