import unittest

from app.analytics.scouting_intelligence import (
    summarize_habits,
    player_age,
    observation_tier,
)


class TestScoutingIntelligence(unittest.TestCase):
    def test_opponent_comparison_aligns_home_away_and_preserves_unknown_coverage(self):
        from app.analytics.scouting_intelligence import opponent_preparation

        rows = []
        for i in range(8):
            own = i < 2
            home = i % 2 == 0
            rows.append(
                {
                    "competition_id": 1,
                    "season_id": 1,
                    "season": "A",
                    "team_id": 1,
                    "team": "A",
                    "opponent_id": 2 if own else 3,
                    "opponent": "B" if own else "C",
                    "match_id": i,
                    "date": "2024-01-01",
                    "home": home,
                    "starting_formation": "433",
                    "clock_coverage_fraction": None,
                    "possession_clock_share_proxy": (
                        (0.7 if home else 0.4) if own else (0.5 if home else 0.3)
                    ),
                }
            )
        result = next(r for r in opponent_preparation(rows) if r["opponent_id"] == 2)
        self.assertAlmostEqual(
            result["home_away_aligned_differences_vs_other_opponents"][
                "possession_clock_share_proxy"
            ]["fixture_balanced_difference"],
            0.15,
        )
        self.assertIsNone(result["mean_clock_coverage"])
        self.assertIsNone(
            result["home_away_aligned_differences_vs_other_opponents"][
                "central_attack_share"
            ]["fixture_balanced_difference"]
        )

    def test_cohort_aggregation_keeps_unknown_origins_out_of_denominator(self):
        from app.analytics.scouting_intelligence import cohort_summary

        profiles = [
            {
                "league": "A",
                "position": "Midfielder",
                "age_band": "19_to21",
                "minutes": 90,
                "habits": {"located_actions": 10, "mean_ball_action_origin": [20, 30]},
            },
            {
                "league": "A",
                "position": "Midfielder",
                "age_band": "19_to21",
                "minutes": 180,
                "habits": {"located_actions": 30, "mean_ball_action_origin": [60, 40]},
            },
            {
                "league": "A",
                "position": "Midfielder",
                "age_band": "19_to21",
                "minutes": 90,
                "habits": None,
            },
        ]
        result = cohort_summary(profiles)["cohorts"][0]
        self.assertEqual(result["action_weighted_origin_x"], 50)
        self.assertEqual(result["origin_unknown"], 1)
        self.assertEqual(result["minutes"], 360)

    def test_future_matches_cannot_change_habits_at_earlier_cutoff(self):
        metadata = {
            1: {"dateutc": "2017-09-01 00:00:00", "league": "England"},
            2: {"dateutc": "2018-01-01 00:00:00", "league": "England"},
        }
        events = [
            {
                "id": 1,
                "matchId": 1,
                "playerId": 10,
                "teamId": 20,
                "matchPeriod": "1H",
                "eventSec": 1,
                "eventName": "Pass",
                "subEventName": "Simple pass",
                "tags": [{"id": 1801}],
                "positions": [{"x": 30, "y": 20}, {"x": 50, "y": 20}],
            },
            {
                "id": 2,
                "matchId": 2,
                "playerId": 10,
                "teamId": 20,
                "matchPeriod": "1H",
                "eventSec": 1,
                "eventName": "Pass",
                "subEventName": "Cross",
                "tags": [],
                "positions": [{"x": 80, "y": 20}, {"x": 40, "y": 50}],
            },
        ]
        early = summarize_habits(events, metadata, "2017-10-31")
        events[-1]["positions"][-1]["x"] = 99
        self.assertEqual(early, summarize_habits(events, metadata, "2017-10-31"))
        card = early[("10", "England")]
        self.assertEqual(card["pass_direction_counts"]["forward"], 1)
        self.assertEqual(card["pass_direction_counts"]["backward"], 0)

    def test_unknown_shot_body_part_and_sparse_matches_remain_unknown(self):
        metadata = {1: {"dateutc": "2017-09-01 00:00:00", "league": "England"}}
        row = {
            "id": 1,
            "matchId": 1,
            "playerId": 10,
            "teamId": 20,
            "matchPeriod": "1H",
            "eventSec": 1,
            "eventName": "Shot",
            "subEventName": "Shot",
            "tags": [],
            "positions": [{"x": 90, "y": 50}],
        }
        card = summarize_habits([row], metadata, "2018-05-31")[("10", "England")]
        self.assertIsNone(card["observed_left_foot_shot_share"])
        self.assertIsNone(card["persistent_pass_direction"])
        self.assertEqual(card["shot_body_part_counts"]["unknown"], 1)

    def test_age_uses_birthday_and_low_exposure_does_not_become_bad_potential(self):
        self.assertEqual(player_age("1996-06-01", "2018-05-31"), 21)
        self.assertEqual(player_age("1996-05-31", "2018-05-31"), 22)
        self.assertEqual(observation_tier(120, 2, True), "limited_observation")
        self.assertEqual(observation_tier(900, 15, True), "priority_for_review")
        self.assertEqual(observation_tier(900, 15, False), "routine_followup")


if __name__ == "__main__":
    unittest.main()
