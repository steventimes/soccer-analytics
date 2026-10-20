import copy
import unittest

import numpy as np


class TestRepresentationData(unittest.TestCase):
    def event(self, second, *, match=1, period="1H", team=10, accurate=True):
        return {
            "id": second + 100,
            "matchId": match,
            "matchPeriod": period,
            "teamId": team,
            "eventSec": second,
            "eventId": 8,
            "eventName": "Pass",
            "positions": [{"x": 40, "y": 30}, {"x": 60, "y": 50}],
            "tags": [{"id": 1801 if accurate else 1802}],
        }

    def test_current_outcome_and_endpoint_never_enter_representation_inputs(self):
        from app.analytics.representation.data import pass_sequences

        events = [self.event(2), self.event(5)]
        changed = copy.deepcopy(events)
        changed[-1]["positions"][1] = {"x": 1, "y": 2}
        changed[-1]["tags"] = [{"id": 1802}]
        a = pass_sequences(events)
        b = pass_sequences(changed)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        self.assertNotEqual(a[2][-1].tolist(), b[2][-1].tolist())

    def test_label_eligibility_audit_accounts_for_exclusions_without_removing_context(self):
        from app.analytics.representation.data import pass_sequences

        events = [self.event(i) for i in range(1, 9)]
        events[1]['positions'] = []
        events[2]['positions'][1]['x'] = 'invalid'
        events[3]['positions'] = events[3]['positions'][:1]
        events[4]['tags'] = []
        events[5]['tags'] = [{'id': 1801}, {'id': 1802}]
        events[6]['tags'] = [{'id': 1802}]
        events[7].update(eventName='Duel', eventId=1)
        audit = {}
        try:
            tokens, values, labels = pass_sequences(events, audit=audit)
        except TypeError as error:
            self.fail(f'Pass encoding must expose its eligibility counts: {error}')
        self.assertEqual(audit, {
            'event_rows': 8, 'pass_events': 7, 'eligible_passes': 2,
            'excluded_invalid_start': 1, 'excluded_endpoint_count': 1,
            'excluded_outcome_tags': 2, 'excluded_invalid_endpoint': 1,
            'known_outcome_valid_start_passes': 4,
            'known_outcome_valid_start_completed': 3, 'eligible_completed': 1,
        })
        np.testing.assert_array_equal(labels[:, 0], [1, 0])
        # Invalid labels/endpoints do not remove valid starts from later history.
        self.assertEqual(np.count_nonzero(tokens[1]), 6)
        self.assertEqual(values.shape, (2, 8, 4))
        empty = {}
        self.assertEqual(pass_sequences([], audit=empty)[0].shape, (0, 8))
        self.assertTrue(empty and all(v == 0 for v in empty.values()))

    def test_decreasing_clocks_cannot_put_future_starts_in_pass_context(self):
        from app.analytics.representation.data import pass_sequences

        later, earlier = self.event(5), self.event(2)
        later["positions"][0] = {"x": 90, "y": 10}
        with self.assertRaisesRegex(ValueError, "clock"):
            pass_sequences([later, earlier])
        # Equal source clocks retain the supplied row order; a new period resets.
        tokens, values, _ = pass_sequences([
            self.event(2), self.event(2), self.event(1, period="2H")
        ])
        self.assertEqual(np.count_nonzero(tokens[1]), 2)
        self.assertEqual(float(values[1, -2, 2]), 0)
        self.assertEqual(np.count_nonzero(tokens[2]), 1)

    def test_context_stops_at_match_period_and_uses_current_team_coordinates(self):
        from app.analytics.representation.data import pass_sequences

        events = [
            self.event(2),
            self.event(3, team=20),
            self.event(4),
            self.event(1, period="2H"),
            self.event(1, match=2),
        ]
        tokens, values, _ = pass_sequences(events)
        self.assertEqual(int(np.count_nonzero(tokens[-1])), 1)
        self.assertEqual(int(np.count_nonzero(tokens[-2])), 1)
        self.assertAlmostEqual(float(values[2, -2, 0]), 0.6, places=6)
        self.assertEqual(float(values[2, -2, 3]), 0)
        self.assertEqual(float(values[2, -1, 3]), 1)

    def test_source_pretraining_is_strictly_before_target_validation(self):
        from app.analytics.representation.data import split_matches

        records = [
            {"id": 1, "league": "England", "date": "2017-12-31"},
            {"id": 2, "league": "England", "date": "2018-01-01"},
            {"id": 3, "league": "England", "date": "2018-03-01"},
            {"id": 4, "league": "France", "date": "2017-12-30"},
            {"id": 5, "league": "France", "date": "2018-01-01"},
        ]
        split = split_matches(records, "England")
        self.assertEqual(
            {k: [r["id"] for r in v] for k, v in split.items()},
            {"source": [4], "train": [1], "validation": [2], "test": [3]},
        )

    def test_other_seasons_do_not_enter_the_fixed_2017_18_benchmark(self):
        from app.analytics.representation.data import split_matches

        records = [
            {"id": 1, "league": "England", "date": "2018-03-01"},
            {"id": 2, "league": "England", "date": "2026-09-01"},
            {"id": 3, "league": "France", "date": "2016-08-01"},
        ]
        split = split_matches(records, "England")
        self.assertEqual([r["id"] for r in split["test"]], [1])
        self.assertEqual(split["source"], [])

    def test_sampled_match_ids_stay_aligned_with_selected_pass_rows(self):
        import json
        from pathlib import Path
        import sqlite3
        import tempfile
        from app.analytics.representation.data import load_dataset

        fixtures = [(1, 'France', '2017-12-01'), (2, 'England', '2017-12-02'),
                    (3, 'England', '2018-01-02'), (4, 'England', '2018-03-02'),
                    (5, 'England', '2018-03-03')]
        starts = {1: .1, 2: .2, 3: .3, 4: .4, 5: .5}
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / 'events.db'
            with sqlite3.connect(database) as db:
                db.executescript('''CREATE TABLE wyscout_entities(entity_type TEXT,payload TEXT);
                    CREATE TABLE wyscout_events(match_id INTEGER,period TEXT,seconds REAL,row_number INTEGER,payload TEXT);
                    CREATE TABLE datasets(id INTEGER,name TEXT,provider TEXT);
                    CREATE TABLE assets(dataset_id INTEGER,path TEXT,sha256 TEXT,bytes INTEGER,complete INTEGER);''')
                for mid, league, date in fixtures:
                    match = dict(wyId=mid,dateutc=date,status='Played',duration='Regular')
                    db.execute('INSERT INTO wyscout_entities VALUES(?,?)', ('matches_' + league,json.dumps(match)))
                    for second in range(1, 4 if mid == 4 else 2):
                        event = self.event(second, match=mid)
                        event['positions'][0]['x'] = starts[mid] * 100
                        db.execute('INSERT INTO wyscout_events VALUES(?,?,?,?,?)',
                                   (mid,'1H',second,second,json.dumps(event)))
            data, receipt = load_dataset(database, 'England', 3)
        for key, arrays in data.items():
            self.assertIn('sample_match_ids', receipt['splits'][key])
            groups = receipt['splits'][key]['sample_match_ids']
            self.assertEqual(len(groups), len(arrays[0]))
            np.testing.assert_allclose(arrays[1][:, -1, 0], [starts[mid] for mid in groups])
            self.assertEqual(sorted(set(groups)), receipt['splits'][key]['match_ids'])
            self.assertIn('label_eligibility', receipt['splits'][key])
            eligibility = receipt['splits'][key]['label_eligibility']
            self.assertEqual(eligibility['eligible_passes'], receipt['splits'][key]['sequences_available'])
            self.assertEqual(eligibility['pass_events'], eligibility['eligible_passes'])
            self.assertEqual(eligibility['eligible_completed'], eligibility['eligible_passes'])


class TestRepresentationModels(unittest.TestCase):
    def test_league_comparison_keeps_seed_runs_and_excludes_source_and_test_labels_from_fit(self):
        import importlib.util

        self.assertIsNotNone(importlib.util.find_spec('app.analytics.representation.leagues'))
        try:
            import torch
        except ImportError:
            self.skipTest('Optional PyTorch not installed')
        from threadpoolctl import threadpool_limits
        from app.analytics.representation.leagues import evaluate_models

        rng = np.random.default_rng(56)
        data = {}
        for name, n in [('source', 40), ('train', 80), ('validation', 40), ('test', 40)]:
            values = rng.random((n, 8, 4), dtype=np.float32)
            data[name] = (np.full((n, 8), 8, dtype=np.int64), values,
                          np.column_stack((np.arange(n) % 2, values[:, -1, 0] - .5)).astype(np.float32))
        changed = copy.deepcopy(data)
        changed['source'][1][:] = -100
        changed['source'][2][:] = 100
        changed['test'][2][:, 0] = 1 - changed['test'][2][:, 0]
        changed['test'][2][:, 1] += .2
        torch.set_num_threads(2)
        with threadpool_limits(limits=2):
            first = evaluate_models(data, epochs=2, seeds=(11, 29))
            second = evaluate_models(changed, epochs=2, seeds=(11, 29))
        self.assertEqual(len(first['runs']), 7)
        self.assertEqual({name: len(runs) for name, runs in first['predictions'].items()}, {
            'constant_train_prior': 1, 'linear_context': 1, 'hist_gradient_boosting_context': 1,
            'mlp_context': 2, 'shared_multi_task': 2})
        self.assertEqual(first['curves'], second['curves'])
        for name in first['predictions']:
            for a, b in zip(first['predictions'][name], second['predictions'][name]):
                np.testing.assert_array_equal(a, b)
        for a, b in zip(first['runs'], second['runs']):
            self.assertEqual(a['validation'], b['validation'])
            self.assertNotEqual(a['test'], b['test'])
        for seeds in [(), (11, 11), (True,), (-1,)]:
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                evaluate_models(data, epochs=2, seeds=seeds)

    def test_match_resampling_pools_pass_counts_instead_of_averaging_matches(self):
        import importlib.util

        self.assertIsNotNone(importlib.util.find_spec('app.analytics.representation.comparison'))
        from app.analytics.representation.comparison import paired_match_ranges

        # Three zero-loss passes in one match, one loss-four pass in another.
        result = paired_match_ranges(np.array([[0., 2], [0, 2], [0, 2], [4, 2]]),
                                     np.array([10, 10, 10, 20]), replicates=2000, seed=7)
        np.testing.assert_allclose(result['difference'], [1., 2.])
        np.testing.assert_allclose(result['lower'], [0., 2.])
        np.testing.assert_allclose(result['upper'], [4., 2.])
        again = paired_match_ranges(np.array([[0., 2], [0, 2], [0, 2], [4, 2]]),
                                    np.array([10, 10, 10, 20]), replicates=2000, seed=7)
        self.assertEqual(result, again)
        for values, groups in [(np.ones((4, 2)), [1, 1, 1, 1]),
                               (np.ones((4, 2)), [1, 2]),
                               (np.full((4, 2), np.nan), [1, 1, 2, 2])]:
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                paired_match_ranges(values, groups)

    def test_comparison_averages_seed_losses_before_resampling(self):
        import importlib.util

        self.assertIsNotNone(importlib.util.find_spec('app.analytics.representation.comparison'))
        from app.analytics.representation.comparison import compare_test_predictions

        labels = np.tile([1., 0.], (4, 1))
        predictions = {'reference': [np.tile([.5, 0.], (4, 1))],
                       'two_seeds': [np.tile([.1, -.2], (4, 1)), np.tile([.9, .2], (4, 1))]}
        result = compare_test_predictions(labels, np.array([1, 1, 2, 2]), predictions,
                                          reference='reference', replicates=200, seed=7)
        row = result['comparisons'][0]
        self.assertEqual(row['runs'], 2)
        self.assertAlmostEqual(row['completion_log_loss']['difference'], .5108256237659907)
        self.assertAlmostEqual(row['completion_brier']['difference'], .16)
        self.assertAlmostEqual(row['forward_displacement_mae_pitch_pct']['difference'], 20.)
        for metric in ('completion_log_loss', 'completion_brier', 'forward_displacement_mae_pitch_pct'):
            self.assertAlmostEqual(row[metric]['lower'], row[metric]['difference'])
            self.assertAlmostEqual(row[metric]['upper'], row[metric]['difference'])
        predictions['two_seeds'][0][0, 0] = 1.1
        with self.assertRaises(ValueError):
            compare_test_predictions(labels, [1, 1, 2, 2], predictions, reference='reference')

    def test_tabular_baselines_use_training_data_and_ignore_test_outcomes(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Optional PyTorch not installed")
        from threadpoolctl import threadpool_limits
        from app.analytics.representation.experiment import baselines, train_seed

        rng = np.random.default_rng(18)
        data = {}
        for key, count in (("source", 80), ("train", 80), ("validation", 40), ("test", 40)):
            values = rng.random((count, 8, 4), dtype=np.float32)
            data[key] = (
                np.full((count, 8), 8, dtype=np.int64),
                values,
                np.column_stack((values[:, -1, 0] > .5, values[:, -1, 1] - .5)).astype(np.float32),
            )
        changed = copy.deepcopy(data)
        changed["test"][2][:, 0] = 1 - changed["test"][2][:, 0]
        changed["test"][2][:, 1] += 1
        torch.set_num_threads(2)
        with threadpool_limits(limits=2):
            fixed = baselines(data)
            self.assertIn("hist_gradient_boosting_context", [row["method"] for row in fixed])
            first, _ = train_seed(data, seed=11, epochs=2)
            self.assertIn("mlp_context", [row["method"] for row in first])
            repeated_fixed = baselines(changed)
            repeated, _ = train_seed(changed, seed=11, epochs=2)
        for before, after in zip(fixed + first, repeated_fixed + repeated):
            with self.subTest(method=before["method"]):
                self.assertEqual(before["method"], after["method"])
                self.assertEqual(before["validation"], after["validation"])
                self.assertNotEqual(before["test"], after["test"])

    def test_spatial_nmf_random_starts_are_reproducible_and_fit_early_only(self):
        from app.analytics.spatial_style import fit_style_vectors
        from app.analytics.spatial_style_data import KINDS

        rng = np.random.default_rng(13)
        early = {k: rng.random((8, 2 if k == 'passes' else 1, 4, 4)) for k in KINDS}
        late = {k: rng.random(v.shape) for k, v in early.items()}
        first = fit_style_vectors(early, late, 7)
        repeat = fit_style_vectors(early, {k: v * 3 for k, v in late.items()}, 7)
        other = fit_style_vectors(early, late, 17)
        for kind in KINDS:
            np.testing.assert_array_equal(first['basis'][kind], repeat['basis'][kind])
            np.testing.assert_allclose(first['basis'][kind].sum(axis=1), 1)
        np.testing.assert_array_equal(first['early'], repeat['early'])
        # These small full-rank inputs gave identical NNDSVDA starts across seeds.
        self.assertGreater(float(np.max(np.abs(first['basis']['shots'] - other['basis']['shots']))), .001)

    def test_spatial_basis_matching_handles_permutations_and_dead_components(self):
        from app.analytics.spatial_style import match_basis

        first = np.array([[1., 0, 0], [0, 1, 0], [0, 0, 0]])
        second = np.array([[0., 3, 0], [2, 0, 0], [0, 0, 0]])
        result = match_basis(first, second)
        self.assertEqual(result['matched_cosines'], [1., 1.])
        self.assertEqual(result['dead_components'], [1, 1])
        self.assertEqual(result['unassessed_pairs'], 1)
        self.assertEqual(result['mean_cosine'], 1.)
        self.assertEqual(result['min_cosine'], 1.)
        zero = match_basis(np.zeros((2, 3)), np.zeros((2, 3)))
        self.assertIsNone(zero['mean_cosine'])
        self.assertEqual(zero['unassessed_pairs'], 2)
        for bad in [np.full((3, 3), np.nan), -first, first[:2]]:
            with self.subTest(bad=bad.tolist()), self.assertRaises(ValueError):
                match_basis(first, bad)

    def test_self_supervised_encoder_is_independent_of_task_labels(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Optional PyTorch not installed")
        from app.analytics.representation.models import pretrain_masked

        tokens = np.full((4, 8), 8, dtype=np.int64)
        values = np.full((4, 8, 4), 0.4, dtype=np.float32)
        first, _ = pretrain_masked(
            (tokens, values, np.zeros((4, 2), dtype=np.float32)), epochs=2, seed=42
        )
        second, _ = pretrain_masked(
            (tokens, values, np.ones((4, 2), dtype=np.float32)), epochs=2, seed=42
        )
        for key, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[key]))

    def test_frozen_transfer_updates_heads_and_preserves_encoder(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Optional PyTorch not installed")
        from app.analytics.representation.models import PassModel, fit_supervised

        torch.manual_seed(9)
        model = PassModel()
        data = (
            np.full((8, 8), 8, dtype=np.int64),
            np.full((8, 8, 4), 0.4, dtype=np.float32),
            np.array([[i % 2, 0.1] for i in range(8)], dtype=np.float32),
        )
        before = {k: v.clone() for k, v in model.encoder.state_dict().items()}
        head_before = model.heads.weight.detach().clone()
        fit_supervised(model, data, epochs=2, seed=9, freeze=True)
        for key, value in model.encoder.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))
        self.assertFalse(torch.equal(model.heads.weight, head_before))
