import unittest
from app.analytics.coach_context import match_manager, supported_tenure, split_spells


class TestCoachContext(unittest.TestCase):
    def test_match_metadata_does_not_pick_one_of_multiple_managers(self):
        self.assertEqual(match_manager([{"id": 1, "name": "A"}])["coach_id"], "1")
        self.assertIsNone(
            match_manager([{"id": 1, "name": "A"}, {"id": 2, "name": "B"}])["coach_id"]
        )
        self.assertIsNone(match_manager([])["coach_id"])

    def test_tenure_requires_supported_dates_and_exact_team_population(self):
        tenure = {
            "team": "England",
            "gender": "male",
            "age_group": "senior",
            "start": "2025-01-01",
            "supported_until": "2026-10-06",
            "coach_name": "Thomas Tuchel",
        }
        match = {
            "home_team": "England",
            "gender": "male",
            "age_group": "senior",
            "date": "2026-10-06",
        }
        self.assertEqual(supported_tenure(match, "home", [tenure]), tenure)
        self.assertIsNone(
            supported_tenure({**match, "date": "2026-10-07"}, "home", [tenure])
        )
        self.assertIsNone(
            supported_tenure({**match, "gender": "female"}, "home", [tenure])
        )
        self.assertIsNone(supported_tenure(match, "home", [tenure, tenure]))

    def test_unknown_coach_breaks_observed_spell_instead_of_filling_it(self):
        rows = [
            {"date": "2020-01-01", "coach_id": "A"},
            {"date": "2020-01-02", "coach_id": None},
            {"date": "2020-01-03", "coach_id": "A"},
            {"date": "2020-01-04", "coach_id": "B"},
        ]
        spells = split_spells(rows)
        self.assertEqual([s[0]["coach_id"] for s in spells], ["A", None, "A", "B"])
