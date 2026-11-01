from pathlib import Path
import hashlib
import io
import json
import os
import tarfile
import tempfile
import unittest
import zipfile

from app.analytics.scouting.scouting_intelligence import observation_tier, player_age, summarize_habits

from tests.support import requires_modules


class TestPublicArchives(unittest.TestCase):
    def test_community_archives_select_data_without_executing_tools(self):
        from app.data_service.archive import _selected

        self.assertTrue(_selected("jokecamp", "MLS/1996/Teams/CLB.csv"))
        self.assertTrue(
            _selected("jokecamp", "World Cups/world-cup-1930/world_cup_1930.json")
        )
        self.assertFalse(_selected("jokecamp", "Parsers/epl.js"))
        self.assertTrue(_selected("eddwebster", "data/fbref/raw/players.csv"))
        self.assertTrue(_selected("eddwebster", "data/stats-perform/players.xlsx"))
        self.assertFalse(_selected("eddwebster", "notebooks/analysis.ipynb"))
        self.assertFalse(_selected("eddwebster", "scripts/fetch.py"))

    def test_old_csv_headers_and_blank_spreadsheet_footer(self):
        from app.data_service.fetch.public import FootballCSVClient

        text = "Div,Date,HT,AT,HG,AG,Odds\nG1,16/09/00,A,B,2,1,1.5\n,,,,,,#REF!\n"
        rows = FootballCSVClient().parse_csv_text(
            "GSL", text, "https://example.org/archive"
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["score"]["fullTime"], {"home": 2, "away": 1})

    def archive(self, files):
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w:gz") as archive:
            for name, body in files.items():
                member = tarfile.TarInfo("repository-main/" + name)
                member.size = len(body)
                archive.addfile(member, io.BytesIO(body))
        stream.seek(0)
        return stream

    def test_archive_checks_blob_identity_and_ignores_code(self):
        from app.data_service.archive import extract_archive

        body = b'[{"match_id":1}]'
        tree = {
            "data/matches.json": {
                "size": len(body),
                "sha": hashlib.sha1(
                    b"blob " + str(len(body)).encode() + b"\0" + body
                ).hexdigest(),
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            report = extract_archive(
                self.archive(
                    {"data/matches.json": body, "setup.py": b"raise Exception()"}
                ),
                Path(directory),
                tree,
            )
            self.assertEqual(len(report), 1)
            self.assertEqual(
                json.loads((Path(directory) / "data/matches.json").read_text()),
                [{"match_id": 1}],
            )
            self.assertFalse((Path(directory) / "setup.py").exists())

    def test_archive_path_traversal_and_bad_digest_fail(self):
        from app.data_service.archive import extract_archive

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                extract_archive(
                    self.archive({"data/../../escape.json": b"[]"}), Path(directory), {}
                )
            with self.assertRaises(ValueError):
                extract_archive(
                    self.archive({"data/matches.json": b"[]"}),
                    Path(directory),
                    {"data/matches.json": {"sha": "0" * 40, "size": 2}},
                )

    def test_lfs_pointer_is_replaced_by_verified_payload(self):
        from app.data_service.archive import extract_archive

        data = b'{"frame":1}\n'
        pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{hashlib.sha256(data).hexdigest()}\nsize {len(data)}\n".encode()
        git_sha = hashlib.sha1(
            b"blob " + str(len(pointer)).encode() + b"\0" + pointer
        ).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = extract_archive(
                self.archive({"data/tracking.jsonl": pointer}),
                root,
                {"data/tracking.jsonl": {"size": len(pointer), "sha": git_sha}},
                lfs_fetch=lambda _: iter([data]),
            )
            self.assertEqual((root / "data/tracking.jsonl").read_bytes(), data)
            self.assertEqual(report[0]["bytes"], len(data))


class TestTacticalMetrics(unittest.TestCase):
    def test_xt_solves_rare_shots_without_false_iteration_convergence(self):
        import numpy as np
        from app.analytics.tactics.metrics import fit_expected_threat

        moves, attempts, shots, goals = np.zeros((96, 96)), np.zeros(96), np.zeros(96), np.zeros(96)
        attempts[0] = moves[0, 0] = 1_000_000_000_000
        shots[0] = goals[0] = 1
        attempts[1] = moves[1, 0] = 1
        report = fit_expected_threat(moves, attempts, shots, goals)
        # Every trajectory eventually takes the sole scoring exit, even though
        # the first iteration changes by less than the old stopping threshold.
        np.testing.assert_allclose(np.asarray(report['grid']).ravel()[:2], [1., 1.], atol=1e-12, rtol=0)
        self.assertTrue(report['converged'])
        # An ill-conditioned near-closed cycle may be refused, but downstream
        # callers must never receive a grid that failed its numerical check.
        for n, shot_pair, goal_pair in ((10 ** 8, [1, 1], [1, 0]), (10 ** 12, [1, 1], [1, 0]), (10 ** 11, [3, 1], [2, 0])):
            moves[:] = 0; attempts[:] = 0; shots[:] = 0; goals[:] = 0
            attempts[:2] = n; moves[0, 1] = moves[1, 0] = n
            shots[:2], goals[:2] = shot_pair, goal_pair
            try:
                checked = fit_expected_threat(moves, attempts, shots, goals)
            except ValueError as error:
                self.assertIn('xT', str(error))
            else:
                self.assertTrue(checked['converged'])
                # Rational solution of the two-state count equations. A zero
                # floating-point residual alone can hide cancellation error.
                s0, s1 = shot_pair; g0, g1 = goal_pair
                denominator = n * (s0 + s1) + s0 * s1
                expected = [(g0 * (n + s1) + n * g1) / denominator,
                            (g1 * (n + s0) + n * g0) / denominator]
                np.testing.assert_allclose(np.asarray(checked['grid']).ravel()[:2], expected, atol=1e-9, rtol=0)

    def test_xt_uses_minimal_goal_probability_for_closed_non_scoring_classes(self):
        import numpy as np
        from app.analytics.tactics.metrics import fit_expected_threat

        moves, attempts, shots, goals = np.zeros((96, 96)), np.zeros(96), np.zeros(96), np.zeros(96)
        attempts[[0, 1, 3]] = 1
        moves[0, 1] = moves[1, 0] = moves[3, 2] = 1
        shots[2], goals[2] = 2, 1
        attempts[4], shots[4], goals[4] = 2, 2, 1
        moves[4, 4] = moves[4, 5] = 1
        attempts[5], moves[5, 4] = 2, 1  # One failed move terminates.
        report = fit_expected_threat(moves, attempts, shots, goals)
        np.testing.assert_allclose(np.asarray(report['grid']).ravel()[:6], [0., 0., .5, .5, .4, .2])
        self.assertEqual(report['diagnostics']['goal_reachable_cells'], 4)
        self.assertEqual(report['diagnostics']['observed_cells'], 6)
        self.assertEqual(report['diagnostics']['unobserved_cells'], 90)
        self.assertEqual(report['diagnostics']['zero_shot_cells'], 94)
        self.assertLessEqual(report['diagnostics']['bellman_residual_max'], 1e-12)
        self.assertTrue(report['converged'])

    def test_xt_rejects_nonfinite_negative_and_nonconserving_counts(self):
        import numpy as np
        from app.analytics.tactics.metrics import fit_expected_threat

        base = [np.zeros((96, 96)), np.zeros(96), np.zeros(96), np.zeros(96)]
        for invalid in ('negative', 'nan', 'infinite', 'extra_moves', 'extra_goals', 'shape', 'nonnumeric', 'boolean'):
            arrays = [a.copy() for a in base]
            if invalid == 'negative': arrays[1][0] = -1
            elif invalid == 'nan': arrays[2][0] = np.nan
            elif invalid == 'infinite': arrays[0][0, 0] = np.inf
            elif invalid == 'extra_moves': arrays[0][0, 1] = 1
            elif invalid == 'extra_goals': arrays[3][0] = 1
            elif invalid == 'shape': arrays[0] = np.zeros((95, 96))
            elif invalid == 'nonnumeric': arrays[1] = np.full(96, 'missing')
            else: arrays[1] = np.zeros(96, dtype=bool)
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'xT'):
                fit_expected_threat(*arrays)

    def test_bad_optional_360_keeps_verified_event_analysis(self):
        from app.analytics.scouting.metrics import analyze_statsbomb

        match = {
            "match_id": 1,
            "match_date": "2024-01-01",
            "competition": {"competition_id": 55, "competition_name": "Test"},
            "season": {"season_id": 1, "season_name": "2024"},
            "home_team": {"home_team_name": "Home", "home_team_gender": "male"},
            "away_team": {"away_team_name": "Away"},
            "home_score": 1,
            "away_score": 0,
        }
        event = {
            "id": "a",
            "type": {"name": "Shot"},
            "team": {"id": 1, "name": "Home"},
            "player": {"id": 11, "name": "A"},
            "location": [110, 40],
            "shot": {"statsbomb_xg": 0.3, "outcome": {"name": "Goal"}},
        }
        files = {
            "data/matches/55/1.json": json.dumps([match]).encode(),
            "data/events/1.json": json.dumps([event]).encode(),
            "data/lineups/1.json": b"[]",
            "data/three-sixty/1.json": b"[{broken",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            root.mkdir()
            entries = []
            for relative, body in files.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(body)
                entries.append(
                    {"path": relative, "sha256": hashlib.sha256(body).hexdigest()}
                )
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "files": entries,
                        "repository": "https://example.org",
                        "revision": "test",
                    }
                )
            )
            report = analyze_statsbomb(root, Path(directory) / "report")
        self.assertEqual(report["matches_analyzed"], 1)
        self.assertEqual(report["errors"], [])
        self.assertEqual(len(report["quality_exclusions"]), 1)

    def test_event_metrics_use_successful_actions_and_spatial_endpoints(self):
        from app.analytics.tactics.metrics import analyze_event_match

        events = [
            {
                "id": "a",
                "type": {"name": "Pass"},
                "team": {"id": 1, "name": "Home"},
                "player": {"id": 11, "name": "A"},
                "location": [30, 40],
                "pass": {
                    "end_location": [105, 40],
                    "recipient": {"id": 12, "name": "B"},
                },
            },
            {
                "id": "b",
                "type": {"name": "Pass"},
                "team": {"id": 1, "name": "Home"},
                "player": {"id": 11, "name": "A"},
                "location": [30, 40],
                "pass": {"end_location": [110, 40], "outcome": {"name": "Incomplete"}},
            },
            {
                "id": "c",
                "type": {"name": "Shot"},
                "team": {"id": 1, "name": "Home"},
                "player": {"id": 12, "name": "B"},
                "location": [110, 40],
                "shot": {"statsbomb_xg": 0.3, "outcome": {"name": "Goal"}},
            },
        ]
        report = analyze_event_match(events, [], match_id=1)
        team = report["teams"][0]
        self.assertEqual(team["passes"], 2)
        self.assertEqual(team["completed_passes"], 1)
        self.assertEqual(team["progressive_passes"], 1)
        self.assertEqual(team["box_entries"], 1)
        self.assertAlmostEqual(team["xg"], 0.3)
        self.assertEqual(report["passing_network"][0]["completed_passes"], 1)

    def test_360_reports_visible_context_without_inventing_tracking(self):
        from app.analytics.tactics.metrics import analyze_event_match

        events = [
            {
                "id": "a",
                "type": {"name": "Pass"},
                "team": {"id": 1, "name": "Home"},
                "location": [60, 40],
                "pass": {"end_location": [80, 40]},
            }
        ]
        frames = [
            {
                "event_uuid": "a",
                "visible_area": [0, 0, 120, 0, 120, 80, 0, 80],
                "freeze_frame": [
                    {"teammate": False, "location": [63, 44]},
                    {"teammate": True, "location": [80, 40]},
                ],
            }
        ]
        report = analyze_event_match(events, frames, match_id=1)
        self.assertAlmostEqual(
            report["spatial_context"][0]["nearest_visible_opponent_units"], 5.0
        )
        self.assertFalse(report["spatial_context"][0]["continuous_tracking"])


class TestTrackingMetrics(unittest.TestCase):
    def test_speed_spikes_preserve_completed_sprints_and_separate_later_bouts(self):
        from app.analytics.tracking.metrics import summarize_tracking

        cases = [
            ([0, 2, 4, 6, 20], 0, 6.0),
            ([0, 2, 4, 6, 8, 20], 1, 8.0),
            ([0, 2, 4, 6, 8, 20, 22, 24, 26, 28], 2, 16.0),
        ]
        for positions, expected_bouts, expected_distance in cases:
            with self.subTest(positions=positions):
                frames = (
                    {
                        "period": 1,
                        "time": i * 0.25,
                        "players": [
                            {"id": 1, "team": "home", "x": x, "y": 0, "detected": True}
                        ],
                    }
                    for i, x in enumerate(positions)
                )
                player = summarize_tracking(frames)["players"][0]
                self.assertEqual(player["sprint_bouts_ge_1s"], expected_bouts)
                self.assertEqual(player["excluded_speed_spikes"], 1)
                self.assertAlmostEqual(player["sprint_distance_m"], expected_distance)

    def test_phase_shape_is_duration_weighted(self):
        from app.analytics.tracking.movement import analyze_phases

        text = "team_in_possession_id,duration,team_in_possession_phase_type,team_out_of_possession_phase_type,team_in_possession_width_start,team_in_possession_length_start,team_possession_lead_to_shot\n1,10,create,medium_block,40,30,True\n1,30,create,medium_block,60,40,False\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "phase.csv"
            path.write_text(text)
            rows = analyze_phases(path, [1, 2])["profiles"]
        attack = next(
            row
            for row in rows
            if row["team_id"] == 1 and row["state"] == "in_possession"
        )
        self.assertEqual(attack["duration_seconds"], 40.0)
        self.assertEqual(attack["provided_mean_width_m"], 55.0)
        self.assertEqual(attack["shot_ending_fraction_known"], 0.5)

    def test_bodypose_full_match_handles_two_periods_and_missing_joints(self):
        from app.data_service.bodypose import analyze_bodypose

        frames = [
            {
                "period": 1,
                "player_data": [
                    {"player_id": 1, "joints": {"lAnkle": {}, "rAnkle": {}}}
                ],
            },
            {"period": 2, "player_data": [{"player_id": 2, "joints": None}]},
            {
                "period": 2,
                "player_data": [
                    {"player_id": 1, "joints": {"lAnkle": {}}},
                    {"player_id": 2, "joints": None},
                ],
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            external = root / "external-bodypose"
            external.mkdir()
            path = external / "sample.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(
                    "sample.jsonl", "\n".join(json.dumps(frame) for frame in frames)
                )
                archive.writestr("__MACOSX/._sample.jsonl", b"\x00resource fork")
            (external / "manifest.json").write_text(
                json.dumps(
                    {
                        "files": [
                            {
                                "path": "sample.zip",
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            }
                        ]
                    }
                )
            )
            report = analyze_bodypose(root, root / "output")
        row = report["matches"][0]
        self.assertEqual(row["frames_by_period"], {"1": 1, "2": 2})
        self.assertEqual(row["joint_observations"], 3)
        self.assertEqual(row["pose_coverage_fraction"], 0.5)

    def test_distance_gaps_and_extrapolated_points_are_not_counted(self):
        from app.analytics.tracking.metrics import summarize_tracking

        frames = [
            {
                "period": 1,
                "time": i * 0.1,
                "players": [
                    {"id": 1, "team": "home", "x": x, "y": 0.0, "detected": detected}
                ],
            }
            for i, (x, detected) in enumerate(
                [(0.0, True), (0.5, True), (1.0, False), (1.5, True), (2.0, True)]
            )
        ]
        report = summarize_tracking(frames)
        player = report["players"][0]
        self.assertAlmostEqual(player["distance_m"], 1.0)
        self.assertAlmostEqual(player["observed_segment_seconds"], 0.2)
        self.assertEqual(player["extrapolated_samples"], 1)

    def test_team_width_and_depth_use_real_meter_coordinates(self):
        from app.analytics.tracking.metrics import summarize_tracking

        players = [
            {"id": i, "team": "home", "x": x, "y": y, "detected": True}
            for i, (x, y) in enumerate([(0, 0), (10, 0), (0, 5), (10, 5)])
        ]
        report = summarize_tracking(
            [{"period": 1, "time": 0.0, "players": players}], minimum_shape_players=4
        )
        self.assertEqual(report["team_shape"][0]["mean_depth_m"], 10.0)
        self.assertEqual(report["team_shape"][0]["mean_width_m"], 5.0)


class TestFootballFigures(unittest.TestCase):
    def test_pass_network_respects_direction_failed_passes_and_half_open_time_window(
        self,
    ):
        from app.analytics.visualization import figures

        prepare = getattr(figures, "completed_pass_network", None)
        self.assertIsNotNone(
            prepare,
            "A data-only pass network must be available without constructing a figure.",
        )

        def event(player, recipient, x, timestamp, *, period=1, failed=False, team=1):
            return {
                "type": {"name": "Pass"},
                "team": {"id": team},
                "player": {"id": player},
                "location": [x, 20],
                "period": period,
                "timestamp": timestamp,
                "pass": {
                    "recipient": {"id": recipient},
                    "outcome": {"name": "Incomplete"} if failed else None,
                },
            }

        events = [
            event(10, 20, 30, "00:10:00.000"),
            event(10, 20, 50, "00:39:59.999"),
            event(20, 10, 80, "00:20:00.000"),
            event(10, 20, 120, "00:40:00.000"),
            event(10, 20, 1, "00:09:59.999"),
            event(10, 20, 2, "00:20:00.000", failed=True),
            event(10, 20, 3, "00:20:00.000", period=2),
            event(10, 20, 4, "00:20:00.000", team=2),
        ]
        positions, edges, centroids = prepare(events, 1, (10, 40))
        self.assertEqual(dict(edges), {(10, 20): 2, (20, 10): 1})
        self.assertEqual(len(positions[10]), 2)
        self.assertEqual(centroids[10].tolist(), [40, 20])
        self.assertEqual(centroids[20].tolist(), [80, 20])

    @requires_modules("matplotlib")
    def test_real_network_figure_renders_both_pass_directions(self):
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/soccer-refactor-test-matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        from app.analytics.visualization.football import passing_network

        events = []
        for player, receiver, x in [(10, 20, 30), (20, 10, 80)]:
            events.extend(
                {
                    "team": {"id": 1},
                    "type": {"name": "Pass"},
                    "player": {"id": player},
                    "location": [x, 20],
                    "period": 1,
                    "timestamp": "00:20:00.000",
                    "pass": {"recipient": {"id": receiver}},
                }
                for _ in range(3)
            )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            chosen = {
                "teams": [
                    {"team_id": 1, "team": "Alpha"},
                    {"team_id": 2, "team": "Beta"},
                ]
            }
            paths = passing_network(
                events,
                chosen,
                {10: ("First", 10), 20: ("Second", 20)},
                (10, 40),
                scouting_root=root,
                output_dir=root,
            )
            self.assertEqual(
                [p.name for p in paths], ["passing-network.png", "passing-network.svg"]
            )
            self.assertTrue(all(p.stat().st_size > 1000 for p in paths))
            import xml.etree.ElementTree as ET

            svg = ET.parse(root / "passing-network.svg").getroot()
            self.assertEqual(svg.get("role"), "img")
            labelled_by = svg.get("aria-labelledby").split()
            self.assertEqual(len(labelled_by), 2)
            labels = {element.get("id"): element for element in svg[:2]}
            self.assertEqual(set(labelled_by), set(labels))
            self.assertTrue(all(element.text.strip() for element in labels.values()))
            self.assertIn("Source:", svg[1].text)


class TestScoutingIntelligence(unittest.TestCase):
    def test_opponent_comparison_aligns_home_away_and_preserves_unknown_coverage(self):
        from app.analytics.scouting.scouting_intelligence import opponent_preparation

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
        from app.analytics.scouting.scouting_intelligence import cohort_summary

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


class TestAssessments(unittest.TestCase):
    def test_spatial_style_keeps_axes_endpoints_and_exposure_distinct(self):
        from app.analytics.representation.spatial_style_data import build_heatmaps

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
        from app.analytics.representation.spatial_style_data import normalize_heatmaps

        counts = np.zeros((2, 1, 5, 5)); counts[:, 0, 0, 4] = [2, 4]
        smoothed = normalize_heatmaps(counts, np.array([90, 180]), sigma=1)
        np.testing.assert_allclose(smoothed.sum(axis=(1, 2, 3)), [2, 2])
        np.testing.assert_allclose(smoothed[0], smoothed[1])
        self.assertGreater(smoothed[0, 0, 0, 3], 0)
        with self.assertRaisesRegex(ValueError, "exposure"):
            normalize_heatmaps(counts, np.array([0, 180]), sigma=1)

    def test_spatial_retrieval_shares_ties_and_respects_candidate_pools(self):
        import numpy as np
        from app.analytics.representation.spatial_style import retrieval_scores

        players = [{"league": "England", "position": "Defender"},
                   {"league": "England", "position": "Defender"},
                   {"league": "Spain", "position": "Forward"}]
        scores = retrieval_scores(np.zeros((3, 2)), np.zeros((3, 2)), players)
        self.assertAlmostEqual(scores["same_league"]["top1"], 2 / 3)
        self.assertAlmostEqual(scores["same_league"]["mrr"], 5 / 6)
        self.assertAlmostEqual(scores["same_league"]["random_top1"], 2 / 3)
        self.assertEqual(scores["same_league"]["queries"], 3)
        self.assertEqual(scores["same_position"]["candidate_min"], 1)

    @requires_modules("sklearn")
    def test_spatial_nmf_basis_never_fits_later_period(self):
        import numpy as np
        from app.analytics.representation.spatial_style import fit_style_vectors

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
        import gzip
        from app.analytics.artifacts import cached_json, file_digest

        built = []

        def build():
            built.append(1)
            return {"version": len(built)}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.json"
            self.assertEqual(cached_json(path, "a", build)["version"], 1)
            original = path.read_bytes()
            original_digest = file_digest(path)
            packed = gzip.compress(original, mtime=0)
            compressed = path.with_suffix('.json.gz')
            compressed.write_bytes(packed)
            path.unlink()
            self.assertEqual(cached_json(path, "a", build)["version"], 1)
            self.assertEqual(len(built), 1)
            self.assertFalse(path.exists())
            self.assertEqual(file_digest(path), original_digest)
            self.assertEqual(gzip.decompress(compressed.read_bytes()), original)
            self.assertEqual(file_digest(compressed), hashlib.sha256(packed).hexdigest())
            self.assertEqual(cached_json(path, "b", build)["version"], 2)
            self.assertEqual(cached_json(path, "b", build)["version"], 2)
            self.assertEqual(len(built), 2)
            self.assertTrue(path.exists())
            self.assertEqual(compressed.read_bytes(), packed)
            path.write_text(json.dumps({"version": 999}))
            self.assertEqual(cached_json(path, "b", build)["version"], 3)
            self.assertEqual(compressed.read_bytes(), packed)

    def test_player_cache_invalidates_delegated_event_extractor_and_legacy_style_receipts(self):
        import sqlite3
        from unittest.mock import patch
        from app.analytics import artifacts
        from app.analytics.scouting import assessments
        from app.analytics.scouting import scouting_intelligence
        from app.analytics.representation import spatial_style_data

        class PlayerCacheCaptured(Exception):
            pass

        original_digest, original_cache = artifacts.file_digest, artifacts.cached_json
        revision, built, captured = ["first-event-extractor"], [], []

        def digest(path):
            path = Path(path)
            if path.name == "assessment_data.py":
                return "unchanged-player-extractor"
            if path.name == "event_validation.py":
                return revision[0]
            return original_digest(path)

        def capture_player_cache(path, inputs, build):
            def synthetic_build():
                built.append(1)
                return {"version": len(built)}
            captured.append(original_cache(path, inputs, synthetic_build))
            raise PlayerCacheCaptured

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            connection = sqlite3.connect(":memory:")
            try:
                with patch.object(assessments.sqlite3, "connect", return_value=connection), \
                        patch.object(artifacts, "source_signature", return_value={"sha256": "same-source"}), \
                        patch.object(artifacts, "file_digest", side_effect=digest), \
                        patch.object(artifacts, "cached_json", side_effect=capture_player_cache):
                    for version in ("first-event-extractor", "first-event-extractor", "corrected-event-extractor"):
                        revision[0] = version
                        with self.assertRaises(PlayerCacheCaptured):
                            assessments.run(database=":memory:", output=root)
            finally:
                connection.close()
            self.assertEqual([row["version"] for row in captured], [1, 1, 2])
            receipt_path = root / "player-data.receipt.json"
            receipt = json.loads(receipt_path.read_text())
            self.assertEqual(receipt["inputs"]["event_validation_sha256"], revision[0])
            # Consumers must reject old/mismatched extraction receipts before
            # opening a warehouse or an archive, rather than blessing stale data.
            for value in (None, "first-event-extractor"):
                with self.subTest(event_validation_sha256=value):
                    if value is None:
                        receipt["inputs"].pop("event_validation_sha256", None)
                    else:
                        receipt["inputs"]["event_validation_sha256"] = value
                    receipt_path.write_text(json.dumps(receipt))
                    consumers = (
                        (spatial_style_data, lambda: spatial_style_data.load_data(root / "unused.sqlite3", root, root / "player-data.json")),
                        (scouting_intelligence, lambda: scouting_intelligence.run(root / "unused.sqlite3", root / "scouting", root)),
                    )
                    for module, consume in consumers:
                        with self.subTest(consumer=module.__name__), \
                                patch.object(module, "file_digest", side_effect=digest), \
                                patch.object(module.sqlite3, "connect") as connect:
                            with self.assertRaisesRegex(ValueError, "receipt verification"):
                                consume()
                            connect.assert_not_called()

    def test_offline_research_commands_do_not_choose_live_match_sources(self):
        from app.pipeline import build_parser

        for command in ("assess", "tactics", "movement-lab"):
            self.assertIsNone(build_parser().parse_args([command]).source)

    def test_beta_shrinkage_keeps_missing_observation_visible(self):
        from app.analytics.scouting.assessment_data import beta_estimate

        result = beta_estimate(0, 0, 0.6, 50)
        self.assertIsNone(result["observed"])
        self.assertTrue(result["prior_only"])
        self.assertAlmostEqual(result["posterior_mean"], 0.6)
        self.assertLess(beta_estimate(1, 1, 0.6, 50)["posterior_mean"], 0.7)

    def test_gamma_shrinkage_respects_minutes_and_unknown_exposure(self):
        from app.analytics.scouting.assessment_data import gamma_estimate

        self.assertAlmostEqual(gamma_estimate(3, 270, 0.3, 270)["posterior_mean"], 0.65)
        self.assertIsNone(gamma_estimate(0, 0, 0.3, 270)["observed"])

    def test_future_results_cannot_change_team_features(self):
        from app.analytics.scouting.assessment_data import team_examples

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
        from app.analytics.scouting.assessment_data import team_examples

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
        from app.analytics.scouting.assessment_data import lineup_minutes

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
        from app.analytics.scouting.assessment_data import COUNT_FIELDS, player_panels, fit_priors, player_examples

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
        from app.analytics.scouting.assessment_models import choose_model

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
