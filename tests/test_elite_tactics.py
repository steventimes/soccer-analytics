import unittest
import sqlite3
import json
import tempfile
from pathlib import Path
from app.analytics.elite_tactics import team_comparison, core_competition, coverage


def coverage_connection():
    c = sqlite3.connect(":memory:")
    c.executescript(
        "CREATE TABLE documents(asset_id PRIMARY KEY,kind,payload);"
        "CREATE TABLE assets(id PRIMARY KEY,path,dataset_id,complete);"
        "CREATE TABLE datasets(id,name);"
        "CREATE TABLE core_statsbomb_matches(match_id,match_date,payload);"
        "CREATE TABLE core_wyscout_matches(entity_id,entity_type,payload);"
        "CREATE TABLE football_matches(match_id,payload,asset_id);"
        "CREATE TABLE soccerdata_imports(provider,method,table_name,asset_id);"
        "CREATE TABLE schedules(asset_id,index_00_league,index_01_season,field_0002_game_id,field_0003_date);"
        "CREATE TABLE core_recent_matches(match_id,competition,match_date,status,provider,payload);"
        "CREATE TABLE recent_match_summaries(provider,competition,match_id,asset_id,PRIMARY KEY(provider,competition,match_id));"
    )
    return c


class TestEliteTactics(unittest.TestCase):
    def test_coverage_deduplicates_source_ids_and_does_not_relabel_other_providers(
        self,
    ):
        c = coverage_connection()
        c.executemany(
            "INSERT INTO soccerdata_imports VALUES(?,?,?,?)",
            [
                ("Understat", "schedule", "schedules", 1),
                ("Understat", "schedule", "schedules", 2),
            ],
        )
        c.executemany(
            "INSERT INTO schedules VALUES(?,?,?,?,?)",
            [(a, "ENG-Premier League", "2425", "123", "2024-08-20") for a in [1, 2]],
        )
        c.executemany(
            "INSERT INTO core_recent_matches VALUES(?,?,?,?,?,?)",
            [
                ("1", "eng.1", "2026-10-01", "FINISHED", "espn", "{}"),
                (
                    "2",
                    "FIFA World Cup",
                    "2022-11-22",
                    "FINISHED",
                    "international_results",
                    "{}",
                ),
            ],
        )
        result = coverage(c)["coverage"]
        self.assertEqual(len(result), 2)
        self.assertEqual(
            {r["provider"]: r["source_match_records"] for r in result},
            {"understat": 1, "espn_recent": 1},
        )
        c.close()

    def test_coverage_requires_a_complete_summary_matching_the_latest_result(self):
        # Ten results: one valid header-only summary, three invalid payloads,
        # six unavailable summaries. Cross-provider/competition links do not match.
        c = coverage_connection()
        self.addCleanup(c.close)
        for index in range(1, 12):
            mid = str(index)
            status = "SCHEDULED" if index == 11 else "FINISHED"
            match = {"match_id": mid, "status": status, "home_team_id": "A",
                     "away_team_id": "B", "home_goals": 2, "away_goals": 0}
            c.execute("INSERT INTO core_recent_matches VALUES(?,?,?,?,?,?)",
                      (mid, "eng.1", "2026-10-01", status, "espn", json.dumps(match)))
            if index == 3:
                continue
            provider = "international_results" if index == 5 else "espn"
            competition = "esp.1" if index == 6 else "eng.1"
            c.execute("INSERT INTO recent_match_summaries VALUES(?,?,?,?)",
                      (provider, competition, mid, index))
            c.execute("INSERT INTO assets VALUES(?,?,?,?)",
                      (index, f"summaries/{mid}.json", 1, int(index != 8)))
            if index == 4:
                continue
            summary = {"header": {"id": mid, "competitions": [{
                "status": {"type": {"state": "post", "completed": True}},
                "competitors": [
                    {"homeAway": "home", "id": "A", "score": "1" if index == 2 else "2"},
                    {"homeAway": "away", "id": "B", "score": "0"},
                ],
            }]}}
            serialized = json.dumps(summary)
            if index == 10:
                serialized = serialized.replace('"score": "2"', '"score": 1e309')
            c.execute("INSERT INTO documents VALUES(?,?,?)", (
                index, "other_document" if index == 9 else "recent_provider_json",
                "invalid JSON" if index == 7 else serialized,
            ))
        row, = coverage(c)["coverage"]
        self.assertEqual(row["grain"], "result")
        self.assertEqual(row["source_match_records"], 10)
        self.assertEqual(row["matched_summary_documents"], 1)
        self.assertEqual(row["missing_summary_documents"], 6)
        self.assertEqual(row["invalid_summary_documents"], 3)

    def test_coverage_export_is_read_only_and_rejects_a_missing_database(self):
        from app.analytics.elite_tactics import export_coverage

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "source.sqlite3"
            c = coverage_connection()
            c.execute("INSERT INTO core_recent_matches VALUES(?,?,?,?,?,?)",
                      ("1", "eng.1", "2026-10-01", "FINISHED", "espn", "{}"))
            c.commit()
            with sqlite3.connect(database) as target:
                c.backup(target)
            c.close()
            before = database.read_bytes()
            result = export_coverage(database, root / "output")
            self.assertEqual(database.read_bytes(), before)
            report = json.loads((root / "output" / "core-coverage.json").read_text())
            self.assertEqual(report["coverage"][0]["matched_summary_documents"], 0)
            self.assertEqual(report["coverage"][0]["missing_summary_documents"], 1)
            self.assertEqual(result["coverage_rows"], 1)
            first_run = result["run_id"]
            with sqlite3.connect(database) as target:
                target.execute("DELETE FROM core_recent_matches")
            self.assertNotEqual(export_coverage(database, root / "output")["run_id"], first_run)
            with self.assertRaises(sqlite3.OperationalError):
                export_coverage(root / "missing.sqlite3", root / "missing-output")
            self.assertFalse((root / "missing.sqlite3").exists())
            self.assertFalse((root / "missing-output").exists())

    def test_scope_keeps_primary_mens_competitions_separate_from_validation(self):
        self.assertEqual(core_competition("1. Bundesliga"), "bundesliga")
        self.assertEqual(core_competition("uefa.champions"), "champions_league")
        self.assertIsNone(core_competition("A-League"))
        self.assertIsNone(core_competition("UEFA Women's Euro"))
        self.assertIsNone(core_competition("UEFA Euro", "female"))

    def test_pre_final_baseline_excludes_final_future_other_team_and_missing_values(
        self,
    ):
        final = {
            "match_number": 63,
            "date": "2025-07-13",
            "teams": ["A", "B"],
            "goals": [3, 0],
            "metrics_by_team": [{"possession": 32}, {"possession": 61}],
        }
        matches = [
            final,
            {
                "match_number": 1,
                "date": "2025-06-15",
                "teams": ["X", "A"],
                "goals": [0, 2],
                "metrics_by_team": [{}, {"possession": 50}],
            },
            {
                "match_number": 2,
                "date": "2025-06-20",
                "teams": ["A", "Y"],
                "goals": [1, 3],
                "metrics_by_team": [{}, {}],
            },
            {
                "match_number": 64,
                "date": "2025-07-14",
                "teams": ["A", "B"],
                "goals": [9, 0],
                "metrics_by_team": [{"possession": 100}, {}],
            },
        ]
        result = team_comparison(matches, final, 0)
        self.assertEqual(result["pre_final_form"]["matches"], 2)
        self.assertEqual(result["pre_final_form"]["goals_for"], 3)
        self.assertEqual(result["pre_final_form"]["goals_against"], 3)
        self.assertEqual(
            result["metrics"]["possession"]["pre_final_available_matches"], 1
        )
        self.assertEqual(result["metrics"]["possession"]["pre_final_mean"], 50)
        self.assertEqual(result["metrics"]["possession"]["final_minus_pre_final"], -18)
        self.assertEqual(result["pre_final_matches"][0]["opponent"], "X")
