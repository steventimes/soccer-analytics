from pathlib import Path
import json
import sqlite3
import tempfile
import unittest

from app.analytics.tactics.coach_context import match_manager, split_spells, supported_tenure
from app.analytics.tactics.elite_tactics import core_competition, coverage, team_comparison
from app.analytics.tactics.switching_play import pass_windows
from app.analytics.tracking.movement_lab import external_load, peak_windows, rotation_candidates, smooth_segments


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
    def test_extraction_is_pure_and_standalone_collection_retains_output(self):
        import tempfile
        from pathlib import Path
        from app.data_service.warehouse import Warehouse
        from app.analytics.tactics.tactical_relations import extract, collect
        from app.analytics.artifacts import cached_json
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);database=root/'source.sqlite3'
            with Warehouse(database):
                pass
            output=root/'output'
            data=extract(database)
            self.assertEqual(data['matches'],0)
            self.assertFalse(output.exists())
            self.assertEqual(collect(database,output),data)
            self.assertEqual(json.loads((output/'data.json').read_text()),data)
            target=root/'managed/data.json'
            from app.file_io import write_bytes_atomic
            with patch('app.file_io.write_bytes_atomic',wraps=write_bytes_atomic) as writes:
                self.assertEqual(cached_json(target,{'source':'fixture'},lambda:extract(database)),data)
                self.assertEqual(writes.call_count,2)  # One cache payload and its receipt.
                cached_json(target,{'source':'fixture'},lambda:self.fail('cache hit must not extract'))
                self.assertEqual(writes.call_count,2)

    def test_repeated_save_keeps_opponent_dossiers(self):
        from app.analytics.tactics.tactical_models import save_results

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
        from app.analytics.tactics.tactical_relations import analyze_match

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
        from app.analytics.tactics.tactical_relations import analyze_match

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
        from app.analytics.tactics.tactical_relations import analyze_match

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
    def test_coverage_merges_latest_csv_by_source_identity_and_declared_season(self):
        c = coverage_connection()
        self.addCleanup(c.close)
        c.execute(
            "CREATE TABLE latest_match_observations(provider,competition,match_id,match_date,status,"
            "team_type,gender,asset_id,observed_at,payload,PRIMARY KEY(provider,competition,match_id))"
        )
        c.executemany("INSERT INTO datasets VALUES(?,?)", [(1, "football-csv"), (2, "recent_match_sources")])
        c.executemany("INSERT INTO assets VALUES(?,?,?,?)", [
            (1, "history/football-csv/PL/2026.json", 1, 1),
            (2, "history/football-csv/PL/2026.json@" + "a" * 64, 1, 1),
            (3, "history/football-csv/SA/2026.json", 1, 1),
            (4, "history/football-csv/PL/2025.json", 1, 1),
            (5, "football-csv/PL_2026.csv@" + "b" * 64, 2, 1),
            (6, "football-csv/SA_2026.csv", 2, 1),
            (7, "football-csv/PL_2025.csv", 2, 1),
            (8, "football-csv/PL_2019.csv", 2, 1),
            (9, "football-csv/PL_2026.csv@" + "c" * 64, 2, 0),
        ])
        c.executemany("INSERT INTO football_matches VALUES(?,?,?)", [
            ("shared", json.dumps({"source_date": "2026-08-01"}), 1),
            ("shared", json.dumps({"source_date": "2026-10-23"}), 2),
            ("legacy-only", json.dumps({"source_date": "2026-09-01"}), 1),
            ("shared", json.dumps({"source_date": "2026-08-22"}), 3),
            ("shared", json.dumps({"source_date": "2025-09-01"}), 4),
        ])
        observations = [
            ("shared", "PL", "2026-10-12", 5),
            ("new", "PL", "2026-10-19", 5),
            ("shared", "SA", "2026-10-18", 6),
            ("old-refresh", "PL", "2026-06-12", 7),
            ("covid-delayed", "PL", "2020-08-01", 8),
        ]
        rows = [("football-data.co.uk", comp, mid, date, "FINISHED", "club", "male", aid,
                 "2026-10-23T00:00:00Z", "{}") for mid, comp, date, aid in observations]
        for mid, provider, status, team_type, gender, aid in [
            ("pending", "football-data.co.uk", "STATUS_SCHEDULED", "club", "male", 5),
            ("other-source", "espn", "FINISHED", "club", "male", 5),
            ("women", "football-data.co.uk", "FINISHED", "club", "female", 5),
            ("national", "football-data.co.uk", "FINISHED", "national", "male", 5),
            ("incomplete", "football-data.co.uk", "FINISHED", "club", "male", 9),
        ]:
            rows.append((provider, "PL", mid, "2027-06-01", status, team_type, gender, aid,
                         "2026-10-23T00:00:00Z", "{}"))
        c.executemany("INSERT INTO latest_match_observations VALUES(?,?,?,?,?,?,?,?,?,?)", rows)
        result = {(row["competition"], row["season"]): row for row in coverage(c)["coverage"]}
        self.assertEqual(set(result), {
            ("premier_league", "2019/2020"), ("premier_league", "2025/2026"),
            ("premier_league", "2026/2027"), ("serie_a", "2026/2027"),
        })
        self.assertEqual(
            [(key, row["source_match_records"], row["date_start"], row["date_end"])
             for key, row in sorted(result.items())],
            [(("premier_league", "2019/2020"), 1, "2020-08-01", "2020-08-01"),
             (("premier_league", "2025/2026"), 2, "2025-09-01", "2026-06-12"),
             (("premier_league", "2026/2027"), 3, "2026-09-01", "2026-10-19"),
             (("serie_a", "2026/2027"), 1, "2026-10-18", "2026-10-18")],
        )
        self.assertTrue(all(row["provider"] == "football_csv" for row in result.values()))
        self.assertTrue(all(row["grain"] == "result_and_basic_statistics" for row in result.values()))
        self.assertEqual(coverage(c)["coverage"], list(result.values()))
        c.execute("UPDATE assets SET path='football-csv/SA_2026.csv' WHERE id=5")
        with self.assertRaisesRegex(ValueError, "identity missing or inconsistent"):
            coverage(c)

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
        from app.analytics.tactics.elite_tactics import export_coverage

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


class TestSwitchingPlay(unittest.TestCase):
    def event(self, kind, time, **fields):
        return {
            "type": {"name": kind},
            "timestamp": f"00:00:{time:06.3f}",
            "period": 1,
            "possession": 1,
            "possession_team": {"id": 1},
            "team": {"id": 1},
            **fields,
        }

    def test_outcome_starts_after_pass_end_and_stays_in_same_possession(self):
        ball = self.event(
            "Pass",
            1,
            id="p",
            location=[50, 10],
            duration=2,
            **{"pass": {"end_location": [60, 70]}},
        )
        shot = self.event("Shot", 13, shot={"type": {"name": "Open Play"}})
        rows, _ = pass_windows([ball, shot])
        self.assertTrue(rows[0]["cross_field"])
        self.assertEqual(rows[0]["shot_within10s"], 1)
        for changed in [
            dict(shot, possession=2),
            dict(shot, period=2),
            dict(shot, team={"id": 2}),
            dict(shot, timestamp="00:00:13.001"),
        ]:
            rows, _ = pass_windows([ball, changed])
            self.assertEqual(rows[0]["shot_within10s"], 0)
        rows, exclusions = pass_windows([ball, dict(shot, timestamp="00:00:02.000")])
        self.assertEqual(rows, [])
        self.assertEqual(exclusions["shot_already_observed"], 1)

    def test_incomplete_set_piece_and_penalty_are_not_positive_controls(self):
        ball = self.event(
            "Pass",
            1,
            location=[50, 10],
            duration=1,
            **{"pass": {"end_location": [60, 70]}},
        )
        penalty = self.event("Shot", 4, shot={"type": {"name": "Penalty"}})
        rows, _ = pass_windows([ball, penalty])
        self.assertEqual(rows[0]["shot_within10s"], 0)
        for detail in [
            {"outcome": {"name": "Incomplete"}},
            {"type": {"name": "Corner"}},
        ]:
            changed = dict(ball, **{"pass": {**ball["pass"], **detail}})
            self.assertEqual(pass_windows([changed])[0], [])


if __name__ == "__main__":
    unittest.main()
