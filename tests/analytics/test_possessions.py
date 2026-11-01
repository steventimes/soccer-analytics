from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch
import copy
import csv
import json
import sqlite3
import tempfile
import unittest

import numpy as np

from app.analytics.tactics import buildup_roles as br
from app.analytics.tactics.passing_context import build_pass_contexts, run as run_passing_context, summarize_passes, summarize_pressing_chains
from app.analytics.tactics.receiving_space import CausalVelocity, complete_frame_quality, control_surfaces, synchronization_experiment
from app.analytics.tactics.switch_context import continuous_motion, run as run_switch_context, snapshot_context, tracked_context
from app.analytics.tracking.metrics import metrica_csv_frames, metrica_epts_frames, skillcorner_frames, summarize_tracking
from app.data_service.build_warehouse import document
from app.data_service.warehouse import Warehouse


def event(event_id, event_type, **fields):
    return {
        "event_id": event_id,
        "event_type": event_type,
        "match_id": "1",
        "period": "1",
        "team_id": "10",
        "frame_start": "10",
        "frame_end": "20",
        **fields,
    }


def possession(**fields):
    return event("p", "player_possession", end_type="pass", **fields)


def option(event_id="o", **fields):
    return event(
        event_id,
        "passing_option",
        **{"associated_player_possession_event_id": "p",
           "passing_option_at_pass_moment": "True", **fields},
    )


def chain(event_id, **fields):
    return event(
        event_id,
        "on_ball_engagement",
        **{"pressing_chain_index": "1", "pressing_chain": "True", **fields},
    )


class TestPassingContext(unittest.TestCase):
    def test_selected_receiver_unavailable_is_preserved_outside_denominator(self):
        rows = [
            possession(targeted_passing_option_event_id="chosen"),
            option("available"),
            option("chosen", passing_option_at_pass_moment="False", targeted="True"),
            option("missing", passing_option_at_pass_moment=""),
        ]
        cases, joins = build_pass_contexts(rows, "1")
        case = cases[0]
        self.assertEqual(case["release_option_ids"], ["available"])
        self.assertEqual(case["release_options"], 1)
        self.assertEqual(case["selected_receiver_availability"], "false")
        self.assertTrue(case["selected_receiver_unavailable"])
        self.assertEqual(joins["passing_options_joined"], 3)
        self.assertEqual(summarize_passes(cases)["release_available_options"], 1)

    def test_flag_defines_release_denominator_interval_discrepancy_is_flagged(self):
        rows = [possession(), option(frame_end="18", peak_passing_option_frame="18")]
        cases, joins = build_pass_contexts(rows, "1")
        self.assertEqual(cases[0]["release_options"], 1)
        self.assertEqual(cases[0]["release_flag_outside_option_interval"], 1)
        self.assertEqual(joins["passing_options_joined"], 1)

    def test_join_rejects_cross_period_team_future_interval_and_missing_reference(self):
        rows = [
            possession(), option("ok"), option("period", period="2"),
            option("team", team_id="20"), option("future", frame_start="21", frame_end="30"),
            option("orphan", associated_player_possession_event_id="absent"),
        ]
        cases, joins = build_pass_contexts(rows, "1")
        self.assertEqual(cases[0]["release_option_ids"], ["ok"])
        self.assertEqual(joins["passing_options_joined"], 1)
        self.assertEqual(sum(joins["rejected_option_reasons"].values()), 4)

    def test_unknown_outcome_quality_and_raw_coordinates_are_not_zero_or_clipped(self):
        cases, _ = build_pass_contexts([possession(x_end="54", y_end="-35"), option()], "1")
        case = cases[0]
        self.assertEqual(case["coordinates"]["x_end"], 54)
        self.assertEqual(case["quality"]["fully_extrapolated"], "unknown")
        summary = summarize_passes(cases)
        self.assertEqual(summary["outcomes"], {"unknown": 1})
        self.assertIsNone(summary["completion_rate_known"])
        self.assertEqual(summary["quality"]["is_player_possession_end_matched"], {"unknown": 1})

    def test_duplicate_chain_members_never_inflate_regain_or_joined_value(self):
        pp = possession(possession_epv_delta_for="0.2", possession_epv_delta_against="0.1")
        rows = [pp] + [chain(str(i), pressing_chain_end_type="regain", team_id="20",
                             associated_player_possession_event_id="p") for i in range(5)]
        summary, records = summarize_pressing_chains(rows, "1")
        self.assertEqual(summary["chains"], 1)
        self.assertEqual(summary["outcomes"], {"regain": 1})
        self.assertEqual(records[0]["associated_possessions"], 1)
        self.assertEqual(records[0]["opponent_supplier_epv"]["possession_epv_delta_for"]["n"], 1)
        self.assertEqual(summary["opponent_supplier_epv_chain_means"]["possession_epv_delta_for"]["n"], 1)

    def test_chain_epv_average_weights_chains_after_deduplicating_spells(self):
        rows = [possession(possession_epv_delta_for="0.2"),
                dict(possession(possession_epv_delta_for="0.8"), event_id="q"),
                dict(possession(possession_epv_delta_for="0.2"), event_id="r")]
        rows += [chain("a", team_id="20", associated_player_possession_event_id="p"),
                 chain("b", team_id="20", associated_player_possession_event_id="q"),
                 chain("duplicate", team_id="20", associated_player_possession_event_id="q"),
                 chain("c", team_id="20", pressing_chain_index="2", associated_player_possession_event_id="r")]
        summary, _ = summarize_pressing_chains(rows, "1")
        value = summary["opponent_supplier_epv_chain_means"]["possession_epv_delta_for"]
        self.assertAlmostEqual(value["mean"], 0.35)
        self.assertEqual(value["n"], 2)
        self.assertNotIn("sum", value)
        self.assertEqual(summary["associated_possession_links"], 3)

    def test_period_team_and_match_chain_collisions_are_scoped(self):
        rows = [chain("a", pressing_chain_end_type="regain"),
                chain("b", period="2", pressing_chain_end_type="disruption"),
                chain("c", team_id="20")]
        summary, records = summarize_pressing_chains(rows, "1")
        self.assertEqual(summary["chains"], 3)
        self.assertEqual(summary["outcomes"], {"disruption": 1, "regain": 1, "unknown": 1})
        other, _ = summarize_pressing_chains([dict(row, match_id="2") for row in rows], "2")
        self.assertEqual(other["chains"], 3)
        self.assertEqual({r["match_id"] for r in records}, {"1"})

    def test_conflicting_and_missing_chain_outcomes_are_explicit(self):
        rows = [chain("a", pressing_chain_end_type="regain"),
                chain("b", pressing_chain_end_type="disruption"),
                chain("c", pressing_chain_index="2"),
                chain("d", pressing_chain_index="3", pressing_chain_end_type="regain")]
        summary, records = summarize_pressing_chains(rows, "1")
        self.assertEqual(summary["outcomes"], {"conflict": 1, "regain": 1, "unknown": 1})
        self.assertEqual(summary["known_outcome_chains"], 1)
        self.assertEqual(summary["regain_rate_known"], 1)
        self.assertEqual(summary["observed_regain_fraction_all_chains"], 1 / 3)
        self.assertTrue(records[0]["outcome_conflict"])
        self.assertEqual(records[1]["quality"]["fully_extrapolated"], {"unknown": 1})

    def test_chain_missing_identity_is_not_combined_into_fake_chain(self):
        summary, _ = summarize_pressing_chains([chain("a", period=""), chain("b", team_id="")], "1")
        self.assertEqual(summary["chains"], 0)
        self.assertEqual(summary["members_missing_chain_identity"], 2)
        self.assertIsNone(summary["regain_rate_known"])

    def test_supplier_components_and_option_peak_values_do_not_become_decision_scores(self):
        pp = possession(pass_epv_total="0.03", pass_epv_delta_for="0.05", pass_epv_delta_against="0.02",
                        possession_epv_total="0.1", possession_epv_delta_for="0.2", possession_epv_delta_against="0.1")
        cases, _ = build_pass_contexts([pp, option(xthreat="999", xpass_completion="999", peak_passing_option_frame="30")], "1")
        summary = summarize_passes(cases)
        epv = summary["supplier_epv"]
        self.assertEqual(epv["pass_epv_delta_for"]["mean"], 0.05)
        self.assertEqual(epv["pass_epv_delta_against"]["mean"], 0.02)
        self.assertNotIn("xthreat", cases[0])
        self.assertNotIn("regret", json.dumps(summary))

    def write_match(self, root, mid, date):
        folder = root / str(mid)
        folder.mkdir()
        (folder / f"{mid}_match.json").write_text(json.dumps({
            "id": mid, "date_time": date, "pitch_length": 105, "pitch_width": 68,
        }))
        rows = [dict(possession(pass_outcome="successful"), match_id=str(mid)),
                dict(option(), match_id=str(mid))]
        fields = sorted({key for row in rows for key in row})
        with (folder / f"{mid}_dynamic_events.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_run_uses_complete_chronological_matches_with_input_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "matches"
            root.mkdir()
            for mid, date in [(5, "2025-01-01T12:00:00Z"), (2, "2025-03-01T12:00:00Z"),
                              (9, "2025-02-01T12:00:00Z")]:
                self.write_match(root, mid, date)
            report = run_passing_context(root, Path(tmp) / "output")
            self.assertEqual(report["matches"], 3)
            self.assertEqual(report["passing"]["pass_attempts"], 3)
            self.assertEqual(report["chronological_split"]["earlier"]["match_ids"], ["5", "9"])
            self.assertEqual(report["chronological_split"]["later"]["match_ids"], ["2"])
            self.assertFalse(report["chronological_split"]["model_trained"])
            provenance = json.loads((Path(tmp) / "output/provenance.json").read_text())
            self.assertEqual(len(provenance["files"]), 6)
            self.assertTrue(all(len(r["sha256"]) == 64 for r in provenance["files"]))
            self.assertEqual(report["run_id"], run_passing_context(root, Path(tmp) / "repeat")["run_id"])
            json.dumps(report, allow_nan=False)

    def test_same_kickoff_matches_are_not_split_across_cohorts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "matches"
            root.mkdir()
            for mid in range(1, 4):
                self.write_match(root, mid, "2025-01-01T12:00:00Z")
            report = run_passing_context(root, Path(tmp) / "output")
            self.assertEqual(report["chronological_split"]["later"]["match_ids"], [])
            self.assertEqual(report["chronological_split"]["status"], "insufficient_distinct_match_dates")

    def test_no_source_matches_or_mismatched_metadata_fail_without_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                run_passing_context(root, root / "output")
            self.write_match(root, 1, "2025-01-01T12:00:00Z")
            (root / "1/1_match.json").write_text(json.dumps({"id": 2}))
            with self.assertRaises(ValueError):
                run_passing_context(root, root / "output")
            self.assertFalse((root / "output/report.json").exists())


class TestSwitchContext(unittest.TestCase):
    def test_metadata_identity_cannot_name_cache_paths_or_another_match(self):
        for identity in ["../escape", True, 0, 2]:
            with self.subTest(identity=identity), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / "match.json"
                source.write_text(json.dumps({"id": identity}))
                database = root / "warehouse.sqlite3"
                with Warehouse(database) as db:
                    ds = db.dataset("skillcorner", "football", "skillcorner")
                    aid = db.asset(ds, source, "data/matches/1/1_match.json")
                    document(db, aid, source, "skillcorner_document")
                with self.assertRaises(ValueError):
                    run_switch_context(database, root / "output")
                self.assertFalse((root / "escape.json").exists())

    def test_snapshot_keeps_invisible_destination_and_missing_frames_explicit(self):
        row = {"start": [20, 10], "end": [50, 70]}
        frame = {
            "visible_area": [0, 0, 60, 0, 60, 30, 0, 30],
            "freeze_frame": [
                {"teammate": True, "actor": True, "location": [20, 10]},
                {"teammate": True, "location": [24, 12]},
                {"teammate": False, "location": [22, 11]},
            ],
        }
        context = snapshot_context(row, frame)
        self.assertTrue(context["origin_local_visible"])
        self.assertFalse(context["target_local_visible"])
        self.assertEqual(context["origin_opponents_within5_units"], 1)
        self.assertEqual(context["teammates_visible_excluding_passer"], 1)
        self.assertFalse(context["continuous_runs_available"])
        self.assertEqual(snapshot_context(row, None), {"snapshot_available": False})

    def frames(self):
        return [
            {
                "frame": i,
                "period": 1,
                "timestamp": i / 10,
                "ball_data": {"x": 0, "y": -20, "is_detected": True},
                "possession": {"group": "home team"},
                "player_data": [
                    {"player_id": 1, "x": 0, "y": -20, "is_detected": True},
                    {"player_id": 2, "x": i / 5, "y": 20, "is_detected": True},
                    {"player_id": 3, "x": 10 - i / 5, "y": -20, "is_detected": True},
                ],
            }
            for i in range(21)
        ]

    def test_continuous_motion_rejects_hidden_frames_gaps_and_spikes(self):
        frames = self.frames()
        self.assertAlmostEqual(continuous_motion(frames, 2, 1)["dx"], 4)
        frames[10]["player_data"][1]["is_detected"] = False
        self.assertIsNone(continuous_motion(frames, 2, 1))
        frames = self.frames()
        frames[10]["timestamp"] = 2
        self.assertIsNone(continuous_motion(frames, 2, 1))
        frames = self.frames()
        frames[10]["player_data"][1]["x"] = 40
        self.assertIsNone(continuous_motion(frames, 2, 1))

    def test_detected_far_side_motion_closing_and_attacking_direction(self):
        case = {
            "pass_frame": 10,
            "period": 1,
            "attacking_side": "left_to_right",
            "team_id": "10",
            "player_id": "1",
            "x": 0,
            "y": -20,
        }
        frames = self.frames()
        snapshots = {(1, r["frame"]): r for r in frames}
        metadata = {
            "home_team": {"id": 10},
            "pitch_width": 68,
            "players": [
                {"id": 1, "team_id": 10},
                {"id": 2, "team_id": 10},
                {"id": 3, "team_id": 20},
            ],
        }
        r = tracked_context(case, snapshots, metadata)
        self.assertTrue(r["usable_ball_anchor"])
        self.assertEqual(r["observed_movements"][0]["side_at_start"], "far_side")
        self.assertAlmostEqual(r["opponent_closing"][0]["closing_speed_mps_proxy"], 2)
        reverse = dict(case, attacking_side="right_to_left", y=20)
        r = tracked_context(reverse, snapshots, metadata)
        self.assertTrue(r["usable_ball_anchor"])
        self.assertAlmostEqual(continuous_motion(frames, 2, -1)["dx"], -4)
        self.assertEqual(r["observed_movements"], [])
        self.assertEqual(
            r["continuous_movement_counts"]["far_side_players_continuously_observed"], 1
        )

    def test_unobserved_or_misaligned_movement_is_not_zero_runs(self):
        case = {
            "pass_frame": 10,
            "period": 1,
            "attacking_side": "left_to_right",
            "team_id": "10",
            "player_id": "1",
            "x": 40,
            "y": -20,
        }
        metadata = {
            "home_team": {"id": 10},
            "pitch_width": 68,
            "players": [
                {"id": 1, "team_id": 10},
                {"id": 2, "team_id": 10},
                {"id": 3, "team_id": 20},
            ],
        }
        snapshots = {(1, r["frame"]): r for r in self.frames()}
        r = tracked_context(case, snapshots, metadata)
        self.assertFalse(r["usable_ball_anchor"])
        self.assertEqual(r["continuous_movement_counts"], {})
        self.assertEqual(r["movement_coverage"], "no_complete_teammate_window")

    def test_possession_change_keeps_recovery_distinct_from_attacking_support(self):
        case = {
            "pass_frame": 10,
            "period": 1,
            "attacking_side": "left_to_right",
            "team_id": "10",
            "player_id": "1",
            "x": 0,
            "y": -20,
        }
        metadata = {
            "home_team": {"id": 10},
            "pitch_width": 68,
            "players": [
                {"id": 1, "team_id": 10},
                {"id": 2, "team_id": 10},
                {"id": 3, "team_id": 20},
            ],
        }
        frames = self.frames()
        frames[15]["possession"]["group"] = "away team"
        r = tracked_context(case, {(1, f["frame"]): f for f in frames}, metadata)
        self.assertEqual(r["movement_window_team_context"], "possession_changed")
        self.assertAlmostEqual(r["observed_movements"][0]["pre_release_dx"], 2)
        self.assertAlmostEqual(r["observed_movements"][0]["post_release_dx"], 2)


HAS_NUMPY = find_spec("numpy") is not None
HAS_SKLEARN = find_spec("sklearn") is not None
if HAS_NUMPY:
    import numpy as np



def player(year=2020, pid='1', **extra):
    return dict(provider='understat', league='L', league_id='1', season_id=year,
                player_id=pid, minutes=1000, position='D M S', xg_buildup=8,
                xg_chain=12, np_xg=3, np_goals=2, xa=4, goals=2,
                shots=20, key_passes=30, **extra)


def synthetic_pairs():
    rows=[]
    for year in range(2019,2026):
        for pid in range(12):
            r=player(year,str(pid));r['xg_buildup']=pid+year%3;r['np_xg']=pid/2
            rows.append(r)
    return br.consecutive_pairs(rows)



class BuildupRoleTests(unittest.TestCase):
    def test_rates_do_not_invent_zero_exposure(self):
        for minutes in (0, -1, None, float('nan')):
            row = player(); row['minutes'] = minutes
            assert br.per90(row, 'xg_buildup') is None
        row = player(); row['xg_buildup'] = None
        assert br.per90(row, 'xg_buildup') is None
        assert br.per90(player(), 'xg_buildup') == .72


    def test_duplicate_and_provider_identity_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            br.consecutive_pairs([player(), player()])
        row = player(); row['provider'] = 'wyscout'
        with self.assertRaisesRegex(ValueError, 'provider'):
            br.consecutive_pairs([row])


    def test_partial_never_a_target_or_input(self):
        rows = [player(y) for y in (2024, 2025, 2026, 2027)]
        assert [(a['season_id'], b['season_id']) for a,b in br.consecutive_pairs(rows)] == [(2024, 2025)]


    def test_multi_team_keeps_whole_league_and_composite_position(self):
        row = player(2026); row['team'] = 'First'
        raw = {('1',2026,'1'): {'team_title':'First,Second', 'position':'D M S', 'time':'1000', 'xGBuildup':'8', 'xGChain':'12'}}
        result = br.reconcile_players([row], raw)[0]
        assert result['team'] is None
        assert result['team_scope'] == 'whole_league_multiple_clubs'
        assert result['position'] == 'D M S'
        assert result['source_team_title'] == 'First,Second'


    @unittest.skipUnless(HAS_NUMPY, "numpy is an optional experiment dependency")
    def test_test_target_cannot_change_fitted_model_or_selection(self):
        pairs=synthetic_pairs()
        first, private=br.temporal_models(pairs, draws=20)
        altered=copy.deepcopy(pairs)
        for a,b in altered:
            if a['season_id'] in (2023,2024): b['xg_buildup'] += 1000
        second, private2=br.temporal_models(altered, draws=20)
        assert private['fitted_models'] == private2['fitted_models']
        assert first['validation_selected_model'] == second['validation_selected_model']
        assert first['models']['augmented']['test_mae'] != second['models']['augmented']['test_mae']


    def test_all_native_passes_ignore_tags_and_unknown_accuracy(self):
        c=sqlite3.connect(':memory:')
        c.execute('CREATE TABLE wyscout_events(match_id,player_id,period,event_type,payload)')
        for tag in ([],[{'id':1801}],[{'id':1802}],[{'id':1801},{'id':1802}]):
            c.execute('INSERT INTO wyscout_events VALUES(1,2,\'1H\',\'Pass\',?)',(json.dumps({'tags':tag}),))
        assert br.native_counts(c)[('1','2')]['Pass'] == 4
        c.execute("UPDATE wyscout_events SET payload='{}'")
        assert br.native_counts(c)[('1','2')]['Pass'] == 4


    def test_half_boundary_and_zero_exposure(self):
        rows=[dict(entity_id='2',match_id=str(i),date=d,league='L',position='M',minutes=m,
                   goals=0,passes=3,accurate_passes=2,duels=3,duel_wins=2,minutes_source='supplied')
              for i,d,m in [(1,'2017-12-31',90),(2,'2018-01-01',90),(3,'2018-01-02',0)]]
        panels,audit=br.half_panels(rows,{('1','2'):{'Pass':10},('2','2'):{'Pass':20}})
        assert [r['half'] for r in panels] == ['early','late']
        assert [r['minutes'] for r in panels] == [90,90]
        assert audit['invalid_exposure_rows'] == 1


    @unittest.skipUnless(HAS_NUMPY and HAS_SKLEARN, "numpy/sklearn are optional experiment dependencies")
    def test_neighbors_same_candidates_exclude_self_and_scaler_is_early_only(self):
        early=np.array([[i,i*i] for i in range(24)],float)
        late=early+3
        a=br.neighbor_stability(early,late,[str(i) for i in range(24)])
        b=br.neighbor_stability(early,late*100,[str(i) for i in range(24)])
        assert a['scaler'] == b['scaler']
        assert a['scaler']['mean'] == early.mean(axis=0).tolist()
        for row in a['neighbors']:
            assert row['player_id'] not in row['early']+row['late']
            assert len(row['early']) == len(row['late']) == 5


    def test_small_neighbor_cohort_has_explicit_missing_estimate(self):
        result=br.neighbor_stability([[0],[1]],[[1],[2]],['a','b'])
        assert result['status'] == 'insufficient_sample'
        assert result['mean_jaccard'] is None

    def test_small_repeatability_and_bootstrap_do_not_claim_precision(self):
        result = br.persistence([(player(2020), player(2021))])
        self.assertEqual(result['xg_buildup']['status'], 'insufficient_sample')
        self.assertIsNone(result['xg_buildup']['r'])
        result = br.paired_bootstrap([1, 2], ['one', 'two'])
        self.assertIsNone(result['player_cluster_ci95'])

    def test_invalid_threshold_rejected_before_database_access(self):
        for threshold in (0, -1, float('nan'), float('inf')):
            with self.assertRaisesRegex(ValueError, 'minimum_minutes'):
                br.run('/nonexistent/database', '/nonexistent/output', threshold)

    def test_observation_cache_invalidates_for_dependency_only_change_and_legacy_receipt(self):
        import gzip

        source = {'sha256': 'unchanged-source'}
        event_digest = ['events-v1']
        original_digest = br.file_digest
        def digest(path):
            if Path(path).name == 'assessment_data.py':
                return 'unchanged-extractor'
            if Path(path).name == 'event_validation.py':
                return event_digest[0]
            return original_digest(path)

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / 'players.json'
            cache.write_text(json.dumps({'matches': {}, 'observations': ['cached'], 'audit': {}}))
            original_bytes = cache.read_bytes()
            inputs = {'sources': source, 'extractor_sha256': 'unchanged-extractor',
                      'event_validation_sha256': 'events-v1'}
            receipt = cache.with_suffix('.receipt.json')
            def write_receipt():
                receipt.write_text(json.dumps({'inputs': inputs, 'content_sha256': original_digest(cache)}))
            write_receipt()
            with patch.object(br, 'source_signature', return_value=source), \
                 patch.object(br, 'file_digest', side_effect=digest), \
                 patch('app.analytics.scouting.assessment_data.load_players', return_value=({}, {}, ['rebuilt'], {}, {})) as rebuild:
                _, observations, _, provenance = br.load_wyscout_observations(None, cache)
                self.assertEqual(observations, ['cached'])
                self.assertTrue(provenance['verified_cache_used'])
                rebuild.assert_not_called()
                compressed = cache.with_suffix('.json.gz')
                packed = gzip.compress(original_bytes, mtime=0)
                compressed.write_bytes(packed)
                cache.unlink()
                _, observations, _, provenance = br.load_wyscout_observations(None, cache)
                self.assertEqual(observations, ['cached'])
                self.assertTrue(provenance['verified_cache_used'])
                rebuild.assert_not_called()
                event_digest[0] = 'events-v2'
                _, observations, _, provenance = br.load_wyscout_observations(None, cache)
                self.assertEqual(observations, ['rebuilt'])
                self.assertFalse(provenance['verified_cache_used'])
                self.assertEqual(provenance['inputs']['event_validation_sha256'], 'events-v2')
                rebuild.assert_called_once()
                rebuild.reset_mock()
                event_digest[0] = 'events-v1'
                del inputs['event_validation_sha256']
                write_receipt()
                _, observations, _, provenance = br.load_wyscout_observations(None, cache)
                self.assertEqual(observations, ['rebuilt'])
                self.assertFalse(provenance['verified_cache_used'])
                rebuild.assert_called_once()
            self.assertFalse(cache.exists())
            self.assertEqual(compressed.read_bytes(), packed)
            self.assertEqual(gzip.decompress(packed), original_bytes)


def frame(time=0.0, period=1):
    return {
        'provider': 'metrica', 'match_id': 'fixture', 'frame': round(time * 25),
        'period': period, 'time': time, 'pitch_length': 105.0, 'pitch_width': 68.0,
        'ball': {'x': 0.0, 'y': 0.0, 'supplier_detected': None},
        'players': [dict(id=f'{team}{i}', team=team, x=sign * (3 + i * 3),
                         y=(i % 5 - 2) * 7.0, detected=True, supplier_detected=None,
                         vx=0.0, vy=0.0, velocity_quality='valid')
                    for team, sign in [('home', -1), ('away', 1)] for i in range(11)],
    }


class ReceivingSpaceTests(unittest.TestCase):
    def test_completeness_does_not_invent_detection(self):
        sample = frame()
        self.assertTrue(complete_frame_quality(sample)['accepted'])
        self.assertFalse(complete_frame_quality(sample, require_detection=True)['accepted'])
        sample['provider'] = 'skillcorner'
        with self.assertRaisesRegex(ValueError, 'supplier_detected'):
            control_surfaces(sample, np.array([[0., 0.]]))
        sample['players'].pop()
        self.assertIn('incomplete_teams', complete_frame_quality(sample)['reasons'])
        sample['ball']['x'] = None
        self.assertIn('missing_ball', complete_frame_quality(sample)['reasons'])

    def test_outside_coordinates_preserved_and_flagged(self):
        sample = frame()
        sample['players'][0]['x'] = 70.0
        quality = complete_frame_quality(sample)
        self.assertTrue(quality['accepted'])
        self.assertEqual(quality['outside_player_count'], 1)
        self.assertEqual(sample['players'][0]['x'], 70.0)

    def test_velocities_are_causal_and_reset_for_missing_period_and_gap(self):
        velocity = CausalVelocity()
        a = velocity.update(frame())
        self.assertIsNone(a['players'][0]['vx'])
        b = frame(.04)
        b['players'][0]['x'] += .2
        self.assertAlmostEqual(velocity.update(b)['players'][0]['vx'], 5)
        c = frame(.08)
        c['players'][0]['x'] = None
        velocity.update(c)
        self.assertIsNone(velocity.update(frame(.12))['players'][0]['vx'])
        self.assertIsNone(velocity.update(frame(.16, 2))['players'][0]['vx'])
        self.assertIsNone(velocity.update(frame(2, 2))['players'][0]['vx'])

    def test_velocity_reset_after_absent_player_and_clock_reversal(self):
        velocity = CausalVelocity()
        velocity.update(frame())
        missing = frame(.04)
        missing['players'].pop(0)
        velocity.update(missing)
        self.assertIsNone(velocity.update(frame(.08))['players'][0]['vx'])
        self.assertIsNone(velocity.update(frame(.02))['players'][1]['vx'])

    def test_speed_spike_is_unknown_not_clipped(self):
        velocity = CausalVelocity()
        velocity.update(frame())
        sample = frame(.04)
        sample['players'][0]['x'] += 20
        point = velocity.update(sample)['players'][0]
        self.assertIsNone(point['vx'])
        self.assertEqual(point['velocity_quality'], 'speed_spike')

    def test_control_conserves_mass_and_is_team_symmetric(self):
        sample = frame()
        points = np.array([[-20., 0.], [0., 0.], [13., 7.]])
        result = control_surfaces(sample, points)
        swapped = copy.deepcopy(sample)
        for p in swapped['players']:
            p['team'] = 'away' if p['team'] == 'home' else 'home'
        other = control_surfaces(swapped, points)
        for method in ('nearest', 'gaussian', 'arrival'):
            values = result[method]
            self.assertTrue(np.all(np.isfinite(values)))
            self.assertTrue(np.all(values >= 0))
            self.assertTrue(np.all(values <= 1))
            np.testing.assert_allclose(values[0], other[method][1], atol=1e-12)
        np.testing.assert_allclose(result['arrival'].sum(axis=0) + result['uncontrolled'], 1, atol=1e-12)

    def test_rotation_equivariance_and_tie_symmetry(self):
        sample = frame()
        points = np.array([[-20., 0.], [0., 0.], [13., 7.]])
        for p in sample['players']:
            p['vx'], p['vy'] = 2., 1.
        rotated = copy.deepcopy(sample)
        for p in rotated['players'] + [rotated['ball']]:
            p['x'], p['y'] = -p['x'], -p['y']
            if 'vx' in p:
                p['vx'], p['vy'] = -p['vx'], -p['vy']
        for key in ('nearest', 'gaussian', 'arrival'):
            np.testing.assert_allclose(control_surfaces(sample, points)[key],
                                       control_surfaces(rotated, -points)[key], atol=1e-12)
        self.assertEqual(control_surfaces(frame(), np.array([[0., 0.]]))['nearest'][0, 0], .5)

    def test_velocity_and_ball_speed_really_change_model(self):
        sample = frame()
        points = np.array([[0., 0.], [2., 0.], [7., 0.], [15., 10.]])
        static = control_surfaces(sample, points)
        for p in sample['players'][:11]:
            p['vx'] = 5.0
        moving = control_surfaces(sample, points)
        self.assertGreater(np.max(abs(moving['gaussian'] - static['gaussian'])), .01)
        self.assertGreater(np.max(abs(moving['arrival'] - static['arrival'])), .01)
        slow = control_surfaces(sample, points, ball_speed_mps=3.0)
        self.assertGreater(np.max(abs(moving['arrival'] - slow['arrival'])), .01)
        np.testing.assert_array_equal(moving['nearest'], slow['nearest'])

    def test_extreme_hazard_conserves_mass_without_probability_clipping(self):
        result = control_surfaces(frame(), np.array([[0., 0.], [20., 2.]]),
                                  control_rate_per_second=1000., integration_dt_seconds=.9)
        self.assertTrue(np.all(result['arrival'] >= 0))
        self.assertTrue(np.all(result['arrival'] <= 1))
        np.testing.assert_allclose(result['arrival'].sum(axis=0) + result['uncontrolled'], 1, atol=1e-14)

    def test_systematic_sampling_covers_both_periods_and_skips_missing_ball(self):
        from unittest.mock import patch
        from app.analytics.tactics.receiving_space import _sample_match
        rows = [frame(i * .04, 1 if i < 50 else 2) for i in range(100)]
        for row in rows[30:40]:
            row['ball']['x'] = None
        # Two cold-start period frames plus ten missing-ball frames are ineligible.
        with patch('app.analytics.tactics.receiving_space.metrica_csv_frames', return_value=iter(rows)):
            selected, quality = _sample_match(None, None, 'fixture', 8, 88)
        self.assertEqual(len(selected), 8)
        self.assertEqual({f['period'] for f in selected}, {1, 2})
        self.assertEqual(quality['rejected_frame_reasons']['missing_ball'], 10)
        self.assertEqual(len({f['frame'] for f in selected}), 8)
        self.assertGreater(selected[-1]['source_ordinal'], 90)

    def test_invalid_model_inputs_reject_instead_of_filling(self):
        sample = frame()
        sample['players'][0]['vx'] = None
        with self.assertRaises(ValueError):
            control_surfaces(sample, np.array([[0., 0.]]))
        with self.assertRaises(ValueError):
            control_surfaces(frame(), np.array([[0., 0.]]), ball_speed_mps=0)

    def test_clock_alignment_fits_early_and_evaluates_later_same_period(self):
        clocks = {1: {'times': np.arange(100) * .04, 'frames': np.arange(100)},
                  2: {'times': np.arange(100) * .04, 'frames': np.arange(100) + 100}}
        events = [dict(event_id=f'{p}:{i}', period=p, frame=i + (p - 1) * 100,
                       time=i * .04, x=0., y=0.) for p in (1, 2) for i in range(5, 90, 5)]
        result = synchronization_experiment(events, clocks, offsets=(0., .5, -1.))
        self.assertEqual(len(result['cases']), 6)
        for case in result['cases']:
            self.assertLess(case['last_training_time'], case['first_evaluation_time'])
            self.assertAlmostEqual(case['estimated_offset_seconds'], case['injected_offset_seconds'])
            self.assertEqual(case['after']['exact_match_rate'], 1.)
            self.assertEqual(case['after']['mean_absolute_frame_error'], 0.)
            self.assertEqual(case['after']['mean_absolute_seconds_error'], 0.)
        self.assertTrue(any(c['before']['exact_match_rate'] < 1 for c in result['cases']))

    def test_missing_frame_truth_reported_not_nearest_relabelled(self):
        clocks = {1: {'times': np.arange(20) * .04, 'frames': np.arange(20)}}
        events = [dict(event_id=str(i), period=1, frame=i, time=i * .04) for i in range(20)]
        events.append(dict(event_id='missing', period=1, frame=300, time=12.))
        result = synchronization_experiment(events, clocks, offsets=(.5,))
        self.assertEqual(result['excluded']['missing_truth_frame'], 1)

    def test_csv_adapter_keeps_native_identity_ball_and_unknown_supplier_quality(self):
        with tempfile.TemporaryDirectory() as d:
            paths = [Path(d) / f'{team}.csv' for team in ('home', 'away')]
            for path in paths:
                with path.open('w', newline='') as out:
                    writer = csv.writer(out)
                    writer.writerows([[], [], ['Period','Frame','Time [s]','Player1','','Ball',''],
                                      [1,77,3.08,.2,.3,1.1,.4], [1,78,3.12,.201,.3,1.1,.4]])
            rows = list(metrica_csv_frames(*paths, match_id='fixture'))
            self.assertEqual(rows[0]['frame'], 77)
            self.assertEqual(rows[0]['provider'], 'metrica')
            self.assertAlmostEqual(rows[0]['ball']['x'], 63.)
            self.assertTrue(rows[0]['players'][0]['detected'])
            self.assertIsNone(rows[0]['players'][0]['supplier_detected'])
            self.assertIsNone(rows[0]['attacking_direction'])
            self.assertIn('assumed', rows[0]['coordinate_system']['pitch_dimensions_source'])
            self.assertGreater(summarize_tracking(rows)['players'][0]['distance_m'], 0)
            paths[1].write_text(paths[1].read_text().replace('1,78,3.12', '2,78,3.12'))
            with self.assertRaisesRegex(ValueError, 'clocks'):
                list(metrica_csv_frames(*paths))

    def test_epts_keeps_raw_clock_period_clock_and_metadata_direction(self):
        from datetime import timedelta
        from types import SimpleNamespace as NS
        import sys
        from unittest.mock import Mock, patch
        home, away = NS(name='Team A', ground=NS(name='HOME')), NS(name='Team B', ground=NS(name='AWAY'))
        record = NS(frame_id=250, period=NS(id=2), timestamp=timedelta(seconds=2),
                    ball_coordinates=NS(x=.5, y=.25, z=1.7), players_data={},
                    attacking_direction=NS(name='RTL'))
        dataset = NS(metadata=NS(pitch_dimensions=NS(pitch_length=105., pitch_width=68.),
                                 frame_rate=25., teams=[home, away]), records=[record])
        fake_kloppy = NS(metrica=NS(load_tracking_epts=Mock(return_value=dataset)))
        with patch.dict(sys.modules, {'kloppy': fake_kloppy}):
            result = next(metrica_epts_frames(Path('metadata.xml'), Path('raw.txt'), match_id=3))
        self.assertEqual(result['native_timestamp'], 10.)
        self.assertEqual(result['time'], 2.)
        self.assertEqual(result['attacking_direction'], {'Team A': -1, 'Team B': 1})
        self.assertEqual(result['ball']['z'], 1.7)
        self.assertIsNone(result['ball']['supplier_detected'])

    def test_run_identity_binds_skillcorner_prefix_metadata_and_availability(self):
        from app.analytics.tactics.receiving_space import _run_id, _skillcorner_prefix
        sources = [{'file': 'same-metrica.csv', 'sha256': 'unchanged'}]
        code = [{'file': '/same/receiving_space.py', 'sha256': 'unchanged'}]
        config = {'samples_total': 240}
        metadata = dict(id=1874553, pitch_length=105, pitch_width=68, players=[])
        raw = dict(frame=1, period=1, timestamp='00:00:00.1', player_data=[],
                   ball_data={'x': 3., 'y': 4., 'is_detected': True})
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            meta_path = base / '1874553_match.json'
            tracking_path = base / '1874553_tracking_extrapolated.jsonl'
            meta_path.write_text(json.dumps(metadata))
            tracking_path.write_text(json.dumps(raw) + '\n')
            baseline = _skillcorner_prefix(base)
            baseline_id = _run_id(sources, code, config, baseline)
            self.assertEqual(baseline_id, _run_id(sources, code, config, _skillcorner_prefix(base)))
            raw['ball_data']['x'] = 5.
            tracking_path.write_text(json.dumps(raw) + '\n')
            changed_prefix = _skillcorner_prefix(base)
            self.assertEqual(baseline['counts'], changed_prefix['counts'])
            prefix_id = _run_id(sources, code, config, changed_prefix)
            metadata['pitch_length'] = 106
            meta_path.write_text(json.dumps(metadata))
            metadata_id = _run_id(sources, code, config, _skillcorner_prefix(base))
            tracking_path.unlink()
            missing_tracking = _skillcorner_prefix(base)
            missing_tracking_id = _run_id(sources, code, config, missing_tracking)
            self.assertEqual(missing_tracking['availability'], {'metadata': True, 'tracking': False})
            meta_path.unlink()
            tracking_path.write_text(json.dumps(raw) + '\n')
            missing_metadata = _skillcorner_prefix(base)
            missing_metadata_id = _run_id(sources, code, config, missing_metadata)
            self.assertEqual(missing_metadata['availability'], {'metadata': False, 'tracking': True})
            tracking_path.unlink()
            missing_both = _skillcorner_prefix(base)
            missing_both_id = _run_id(sources, code, config, missing_both)
            self.assertEqual(missing_both['availability'], {'metadata': False, 'tracking': False})
        self.assertEqual(len({baseline_id, prefix_id, metadata_id, missing_tracking_id, missing_metadata_id, missing_both_id}), 6)

    def test_skillcorner_preserves_false_unknown_ball_z_and_orientation(self):
        metadata = dict(id=4, pitch_length=105, pitch_width=68,
                        home_team={'id': 10}, away_team={'id': 20},
                        home_team_side=['left_to_right', 'right_to_left'],
                        players=[{'id': 1, 'team_id': 10}, {'id': 2, 'team_id': 20}])
        data = dict(frame=111, period=2, timestamp='00:45:01.2',
                    ball_data={'x': 3., 'y': 4., 'z': 2., 'is_detected': False},
                    player_data=[{'player_id': 1, 'x': 60., 'y': 0., 'is_detected': False},
                                 {'player_id': 2, 'x': 2., 'y': 0., 'is_detected': None}])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'tracking.jsonl'
            path.write_text(json.dumps(data) + '\n')
            result = next(skillcorner_frames(path, metadata))
        self.assertEqual(result['frame'], 111)
        self.assertEqual(result['match_id'], 4)
        self.assertEqual(result['ball']['z'], 2.)
        self.assertFalse(result['ball']['supplier_detected'])
        self.assertFalse(result['players'][0]['supplier_detected'])
        self.assertIsNone(result['players'][1]['supplier_detected'])
        self.assertEqual(result['players'][0]['x'], 60.)
        self.assertEqual(result['attacking_direction'], {'10': -1, '20': 1})
        self.assertEqual(result['native_timestamp'], '00:45:01.2')


if __name__ == "__main__":
    unittest.main()
