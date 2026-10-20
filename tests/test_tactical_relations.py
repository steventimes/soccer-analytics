import unittest
import tempfile
import json
import sqlite3
from pathlib import Path


def events(shot=True):
    rows = [
        {
            "index": 1,
            "period": 1,
            "timestamp": "00:00:00",
            "team": {"id": 1},
            "type": {"name": "Starting XI"},
            "tactics": {"formation": 433},
        },
        {
            "index": 2,
            "period": 1,
            "timestamp": "00:00:00",
            "team": {"id": 2},
            "type": {"name": "Starting XI"},
            "tactics": {"formation": 442},
        },
    ]
    for i in range(3):
        rows.append(
            {
                "id": str(i),
                "index": i + 3,
                "period": 1,
                "timestamp": f"00:00:0{i*2}",
                "duration": 1.0,
                "team": {"id": 1},
                "possession": 3,
                "possession_team": {"id": 1},
                "type": {"name": "Pass"},
                "location": [20 + i * 15, 10],
                "pass": {"end_location": [35 + i * 15, 10]},
            }
        )
    if shot:
        rows.append(
            {
                "index": 6,
                "period": 1,
                "timestamp": "00:00:06",
                "team": {"id": 1},
                "possession": 3,
                "possession_team": {"id": 1},
                "type": {"name": "Shot"},
                "location": [100, 40],
                "shot": {"statsbomb_xg": 0.2, "outcome": {"name": "Saved"}},
            }
        )
    return rows


class TestTacticalRelations(unittest.TestCase):
    def test_repeated_save_keeps_opponent_dossiers(self):
        from app.analytics.tactical_models import save_results

        report = {
            "relations": {
                "opponent_dossiers": [{"season_id": 1, "team_id": 1, "opponent_id": 2}]
            }
        }
        data = {"team_match_rows": [], "tracking_context": []}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out"
            output.mkdir()
            (output / "report.json").write_text(json.dumps(report))
            database = Path(tmp) / "football.sqlite3"
            save_results(database, output, report, data)
            save_results(database, output, report, data)
            with sqlite3.connect(database) as connection:
                self.assertEqual(
                    connection.execute(
                        "SELECT count(*) FROM tactical_opponent_dossiers"
                    ).fetchone()[0],
                    1,
                )

    def test_shot_at_decision_clock_and_later_formation_are_not_features(self):
        from app.analytics.tactical_relations import analyze_match

        meta = {
            "match_id": 1,
            "match_date": "2024-01-01",
            "home_team": {"home_team_id": 1, "home_team_name": "A"},
            "away_team": {"away_team_id": 2, "away_team_name": "B"},
            "competition": {"competition_id": 1, "competition_name": "C"},
            "season": {"season_id": 1, "season_name": "2023/24"},
        }
        sample = events()
        sample[-1]["timestamp"] = "00:00:05"
        self.assertEqual(analyze_match(sample, meta)["prefixes"], [])
        sample = events()
        sample.append(
            {
                "index": 7,
                "period": 1,
                "timestamp": "00:00:07",
                "team": {"id": 1},
                "type": {"name": "Tactical Shift"},
                "tactics": {"formation": 352},
            }
        )
        self.assertEqual(
            analyze_match(sample, meta)["prefixes"][0]["features"]["formation"], "433"
        )

    def test_future_shot_changes_label_but_not_action_prefix(self):
        from app.analytics.tactical_relations import analyze_match

        meta = {
            "match_id": 1,
            "match_date": "2024-01-01",
            "home_team": {"home_team_id": 1, "home_team_name": "A"},
            "away_team": {"away_team_id": 2, "away_team_name": "B"},
            "competition": {"competition_id": 1, "competition_name": "C"},
            "season": {"season_id": 1, "season_name": "2023/24"},
        }
        a = analyze_match(events(), meta)
        b = analyze_match(events(False), meta)
        self.assertEqual(a["prefixes"][0]["features"], b["prefixes"][0]["features"])
        self.assertEqual(a["prefixes"][0]["shot_created"], 1)
        self.assertEqual(b["prefixes"][0]["shot_created"], 0)
        self.assertEqual(a["prefixes"][0]["features"]["formation"], "433")
        self.assertEqual(a["prefixes"][0]["features"]["opponent_formation"], "442")

    def test_unknown_formations_and_zero_possession_denominators_stay_unknown(self):
        from app.analytics.tactical_relations import analyze_match

        meta = {
            "match_id": 1,
            "match_date": "2024-01-01",
            "home_team": {"home_team_id": 1, "home_team_name": "A"},
            "away_team": {"away_team_id": 2, "away_team_name": "B"},
            "competition": {"competition_id": 1, "competition_name": "C"},
            "season": {"season_id": 1, "season_name": "2023/24"},
        }
        report = analyze_match([], meta)
        self.assertIsNone(report["teams"][0]["starting_formation"])
        self.assertIsNone(report["teams"][0]["on_ball_action_share_proxy"])


if __name__ == "__main__":
    unittest.main()
