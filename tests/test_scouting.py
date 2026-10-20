import hashlib
import io
import json
import os
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path


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
        from app.analytics.tactics import fit_expected_threat

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
        from app.analytics.tactics import fit_expected_threat

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
        from app.analytics.tactics import fit_expected_threat

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
        from app.analytics.scouting import analyze_statsbomb

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
        from app.analytics.tactics import analyze_event_match

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
        from app.analytics.tactics import analyze_event_match

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
        from app.analytics.tracking import summarize_tracking

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
        from app.analytics.movement import analyze_phases

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
        from app.analytics.tracking import summarize_tracking

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
        from app.analytics.tracking import summarize_tracking

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
        from app.analytics import figures

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

if __name__ == "__main__":
    unittest.main()
