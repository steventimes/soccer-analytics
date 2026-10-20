import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.analytics import action_value as av


def event(i, kind=8, team=10, tags=(), period='1H'):
    return dict(id=i, matchId=1, matchPeriod=period, eventSec=float(i),
                eventId=kind, eventName='Pass' if kind == 8 else 'Other',
                teamId=team, positions=[dict(x=40, y=50), dict(x=60, y=50)],
                tags=[dict(id=t) for t in tags])


class TestActionValue(unittest.TestCase):
    def test_calibration_ties_are_pooled_and_brier_decomposition_is_exact(self):
        from app.analytics.calibration import binary_diagnostics

        y, p = np.array([0, 1, 0, 1]), np.array([.1, .4, .4, .9])
        result = binary_diagnostics(y, p)
        self.assertEqual([b['count'] for b in result['pav']], [1, 2, 1])
        self.assertEqual([b['observed_rate'] for b in result['pav']], [0., .5, 1.])
        self.assertAlmostEqual(result['brier'], .135)
        self.assertAlmostEqual(result['pav_brier'], .125)
        self.assertAlmostEqual(result['mcb'], .01)
        self.assertAlmostEqual(result['dsc'], .125)
        self.assertAlmostEqual(result['unc'], .25)
        self.assertAlmostEqual(result['binning']['10']['ece'], .1)
        shuffled = binary_diagnostics(y[[2, 0, 3, 1]], p[[2, 0, 3, 1]])
        self.assertEqual(result['pav'], shuffled['pav'])
        constant = binary_diagnostics([0, 1, 0, 1], [.5] * 4)
        self.assertEqual((constant['mcb'], constant['dsc'], constant['unc']), (0., 0., .25))
        repeated = binary_diagnostics([0, 1, 0], [.1] * 3)
        self.assertEqual(repeated['pav'][0]['mean_probability'], .1)
        boundaries = binary_diagnostics([0, 1], [0., 1.])
        self.assertEqual([b['count'] for b in boundaries['binning']['10']['bins']], [1] + [0] * 8 + [1])

    def test_calibration_rejects_invalid_or_empty_probabilities(self):
        from app.analytics.calibration import binary_diagnostics

        for y, p in [([], []), ([0], [np.nan]), ([1], [1.1]), ([2], [.5]),
                     ([0, 1], [.5]), ([[0]], [[.5]])]:
            with self.subTest(y=y, p=p), self.assertRaises(ValueError):
                binary_diagnostics(y, p)

    def test_spadl_split_rejects_overlap_and_keeps_fixed_calendar_boundaries(self):
        from app.analytics.spadl_vaep import calendar_split

        rows = [{'id': 1, 'date': '2017-12-31'}, {'id': 2, 'date': '2018-01-01'},
                {'id': 3, 'date': '2018-03-01'}]
        split = calendar_split(rows)
        self.assertEqual({key: [r['id'] for r in value] for key, value in split.items()},
                         {'train': [1], 'validation': [2], 'test': [3]})
        with self.assertRaises(ValueError):
            calendar_split(rows + [rows[0]])

    def test_spadl_period_boundaries_horizon_and_perspective_values(self):
        import importlib.util
        if importlib.util.find_spec('socceraction') is None:
            self.skipTest('optional socceraction package; use isolated requirements-vaep.txt environment')
        import pandas as pd
        from app.analytics.spadl_vaep import period_states, action_values, convert_match

        native = [dict(event(i, tags=(1801,)), playerId=1, subEventId=85, subEventName='Simple pass') for i in range(15)]
        native[5]['tags'] = [{'id': 102}]
        duplicate = dict(event(100, kind=9, tags=(101, 1802)), playerId=1,
                         subEventId=90, subEventName='Reflexes', eventSec=5.1)
        native.insert(6, duplicate)
        native += [dict(event(i, period='2H', tags=(1801,)), playerId=1, subEventId=85, subEventName='Simple pass') for i in range(20, 35)]
        metadata = {'wyId': 1, 'teamsData': {'10': {'side': 'home', 'score': 0}, '20': {'side': 'away', 'score': 1}}}
        converted = convert_match(native, metadata)
        own = converted['actions'].result_name.eq('owngoal').to_numpy()
        self.assertEqual(converted['actions'].loc[own, 'type_name'].tolist(), ['bad_touch'])
        self.assertEqual(converted['labels'][own].tolist(), [[0., 1.]])
        self.assertTrue(converted['terminal'][own].all())
        self.assertEqual(converted['audit']['ignored_goalkeeper_goal_duplicates'], 1)
        self.assertNotIn(100, converted['actions'].original_event_id.dropna().tolist())

        def action(i, period, team=10, kind='pass', result='success'):
            from socceraction.spadl import config
            return dict(game_id=1, original_event_id=i, action_id=i, period_id=period,
                        time_seconds=float(i % 20), team_id=team, player_id=1,
                        start_x=30., start_y=20., end_x=31., end_y=20.,
                        bodypart_id=0, type_id=config.actiontypes.index(kind),
                        result_id=config.results.index(result))
        rows = [action(i, 1) for i in range(12)] + [action(i, 2) for i in range(20, 32)]
        rows[0] = action(0, 1, kind='shot', result='success')
        rows[11] = action(11, 1, kind='tackle')
        rows[12] = action(20, 2, team=20)
        rows[13] = action(21, 2, team=20, kind='bad_touch', result='owngoal')
        prepared = period_states(pd.DataFrame(rows), 10, horizon=10)
        self.assertEqual(prepared['eligible'].sum(), 6)
        self.assertEqual(prepared['labels'][11].tolist(), [0., 0.])
        self.assertEqual(prepared['labels'][13].tolist(), [0., 1.])
        context = prepared['features']['context']
        self.assertEqual(context.loc[12, 'actiontype_pass_a1'], True)
        self.assertEqual(context.loc[12, 'result_success_a1'], True)
        self.assertEqual(context.loc[14, 'goalscore_team'], 2)
        probabilities = np.full((24, 2), [.1, .01])
        probabilities[11] = [.9, .2]
        probabilities[13] = [.3, .05]
        values = action_values(prepared['actions'], probabilities)
        self.assertEqual(values.loc[0, 'vaep_value'], 0.)
        self.assertEqual(values.loc[12, 'vaep_value'], 0.)
        # Team switch: previous concede becomes current actor's score baseline.
        self.assertAlmostEqual(values.loc[14, 'offensive_value'], .1)
        self.assertAlmostEqual(values.loc[14, 'defensive_value'], -.01)

    def test_goalkeeper_duplicates_and_own_goal_direction(self):
        self.assertIsNone(av.goal_team(event(1, 9, tags=(101,)), (10, 20)))
        self.assertEqual(av.goal_team(event(2, 10, tags=(101,)), (10, 20)), 10)
        self.assertEqual(av.goal_team(event(3, 8, tags=(102,)), (10, 20)), 20)
        shot = event(4, 3, tags=(101,))
        shot['subEventName'] = 'Penalty'
        self.assertEqual(av.goal_team(shot, (10, 20)), 10)
        self.assertIsNone(av.goal_team(event(5, 8, tags=(101,)), (10, 20)))

    def test_horizon_includes_current_plus_nine_native_events(self):
        events = [event(i) for i in range(12)]
        events[9] = event(9, 10, tags=(101,))
        arrays = av.pass_states(events, (10, 20), horizon=10)
        self.assertEqual(arrays[2][0].tolist(), [1, 0])
        events[9] = event(9)
        events[10] = event(10, 10, tags=(101,))
        arrays = av.pass_states(events, (10, 20), horizon=10)
        self.assertEqual(arrays[2][0].tolist(), [0, 0])
        arrays = av.pass_states([event(0, tags=(102,))], (10, 20), horizon=10)
        self.assertEqual(arrays[2][0].tolist(), [0, 1])

    def test_stoppage_period_and_goal_reset(self):
        for boundary in (event(1, 2), event(1, 5), event(1, 6), event(1, 3)):
            events = [event(0), boundary, event(2), event(3, 10, tags=(101,))]
            arrays = av.pass_states(events, (10, 20), horizon=10)
            self.assertEqual(arrays[2][0].tolist(), [0, 0])
        events = [event(0), event(1, period='2H'), event(2, 10, tags=(101,), period='2H')]
        arrays = av.pass_states(events, (10, 20), horizon=10)
        self.assertEqual(arrays[2][0].tolist(), [0, 0])
        self.assertEqual(np.count_nonzero(arrays[0][1]), 1)
        arrays = av.pass_states([event(0, tags=(102,)), event(1)], (10, 20), horizon=10)
        self.assertEqual(np.count_nonzero(arrays[0][1]), 1)

    def test_inputs_do_not_use_end_accuracy_or_future(self):
        events = [event(0), event(1), event(2, 10, tags=(101,))]
        before = av.pass_states(events, (10, 20), horizon=10)
        changed = copy.deepcopy(events)
        changed[0]['positions'] = [dict(x=40, y=50)]
        changed[0]['tags'] = [dict(id=1802)]
        changed[1]['positions'][1] = dict(x=1, y=100)
        changed[2]['positions'][0] = dict(x=5, y=1)
        changed[2]['tags'] = []
        after = av.pass_states(changed, (10, 20), horizon=10)
        for i in (0, 1, 3):
            np.testing.assert_array_equal(before[i], after[i])
        self.assertNotEqual(before[2][0, 0], after[2][0, 0])
        self.assertEqual(before[3].tolist(), [0, 1])

    def test_goal_metadata_mismatch_is_excluded(self):
        metadata = {'teamsData': {'10': {'score': 1}, '20': {'score': 0}}}
        self.assertIsNotNone(av.audit_goals([event(0)], metadata))
        self.assertIsNone(av.audit_goals([event(0, 10, tags=(101,)), event(1, 9, tags=(101,))], metadata))

    def test_xt_training_dispatch_and_failed_move(self):
        counts = av.new_xt_counts()
        av.update_xt(counts, [event(1, tags=(1801,)), event(2, tags=(1802,)), event(3, 10, tags=(101,))])
        self.assertEqual(counts[0].sum(), 1)
        self.assertEqual(counts[1].sum(), 2)
        self.assertEqual(counts[2].sum(), 1)
        self.assertEqual(counts[3].sum(), 1)

    def test_numeric_weights_roundtrip_no_pickle(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Optional PyTorch not installed')
        from app.analytics.representation.models import PassModel
        model = PassModel()
        data = (np.full((3, 8), 8, dtype=np.int64), np.zeros((3, 8, 4), dtype=np.float32))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.npz'
            av.save_gru(model, path)
            restored = av.load_gru(path)
            np.testing.assert_array_equal(av.predict_gru(model, data), av.predict_gru(restored, data))
            with np.load(path, allow_pickle=False) as weights:
                self.assertTrue(all(weights[k].dtype.kind in 'fi' for k in weights.files))

    def test_setting_selection_uses_validation_only(self):
        candidates = [(0.1, np.array([0.1, .9])), (1.0, np.array([.8, .2]))]
        self.assertEqual(av.select_setting(np.array([0, 1]), candidates), 0)
        self.assertEqual(av.select_setting(np.array([1, 0]), candidates), 1)


class TestActionValueLoading(unittest.TestCase):
    def test_train_only_xt_fixed_sampling_native_ids_and_audit_exclusion(self):
        import json
        import sqlite3
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / 'native.sqlite3'
            with sqlite3.connect(database) as db:
                db.executescript('''CREATE TABLE wyscout_entities(entity_type TEXT,payload TEXT);
                    CREATE TABLE wyscout_events(match_id INTEGER,period TEXT,seconds REAL,row_number INTEGER,payload TEXT);
                    CREATE INDEX event_match ON wyscout_events(match_id);
                    CREATE TABLE datasets(id INTEGER,name TEXT,provider TEXT);
                    CREATE TABLE assets(dataset_id INTEGER,path TEXT,sha256 TEXT,bytes INTEGER,complete INTEGER);''')
                for mid, date in ((1, '2017-11-01'), (2, '2018-02-01'), (3, '2018-04-01'), (4, '2017-12-01')):
                    metadata = dict(wyId=mid, status='Played', duration='Regular', dateutc=date,
                                    teamsData={'10': {'score': 1}, '20': {'score': 0}})
                    db.execute('INSERT INTO wyscout_entities VALUES(?,?)', ('matches_England', json.dumps(metadata)))
                    events = [event(mid * 100 + i, tags=(1801,)) for i in range(5)]
                    if mid != 4:
                        events.append(event(mid * 100 + 5, 10, tags=(101,)))
                    for i, row in enumerate(events):
                        row['matchId'] = mid
                        db.execute('INSERT INTO wyscout_events VALUES(?,?,?,?,?)', (mid, '1H', i, i, json.dumps(row)))
            first, receipt, xt = av.load_data(database, 'England', 3, 10)
            self.assertEqual(xt['fit_match_ids'], [1])
            self.assertEqual(receipt['excluded_matches'][0]['match_id'], 4)
            self.assertEqual(xt['attempts'], 5)
            with sqlite3.connect(database) as db:
                for row_number, payload in db.execute('SELECT row_number,payload FROM wyscout_events WHERE match_id=3').fetchall():
                    row = json.loads(payload)
                    row['positions'][1] = dict(x=99, y=1)
                    db.execute('UPDATE wyscout_events SET payload=? WHERE match_id=3 AND row_number=?', (json.dumps(row), row_number))
            second, receipt2, xt2 = av.load_data(database, 'England', 3, 10)
            self.assertEqual(xt, xt2)
            from app.analytics.xt_numerics import audit_grid
            from app.analytics.artifacts import file_digest
            frozen = {'run_id': 'a' * 16, 'data': receipt, 'raw_xt': xt,
                      'code_sha256': {'action_value.py': file_digest(Path(av.__file__))}}
            audit = audit_grid(database, frozen)
            self.assertEqual(audit['counts'], {'matches': 1, 'attempts': 5, 'completed_moves': 5, 'shots': 1, 'goals': 1})
            self.assertEqual(audit['comparison']['max_grid_difference'], 0)
            self.assertEqual(sum(map(sum, audit['support']['attempts'])), 5)
            self.assertNotIn('match_ids', json.dumps(audit))
            for invalid in ('future_match', 'counts', 'source'):
                changed = copy.deepcopy(frozen)
                if invalid == 'future_match':
                    changed['raw_xt']['fit_match_ids'] = [3]
                    changed['data']['splits']['train']['match_ids'] = [3]
                elif invalid == 'counts': changed['raw_xt']['attempts'] += 1
                else: changed['data']['inputs']['sha256'] = '0' * 64
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    audit_grid(database, changed)
            for split in av.SPLITS:
                self.assertEqual(len(first[split][0]), 3)
                np.testing.assert_array_equal(first[split][3], second[split][3])
                np.testing.assert_array_equal(first[split][0], second[split][0])
                np.testing.assert_array_equal(first[split][1], second[split][1])
                self.assertTrue(np.all(first[split][3] // 100 == first[split][4]))


class TestActionValueMetrics(unittest.TestCase):
    def test_xt_resampling_keeps_whole_matches_and_pools_counts(self):
        from app.analytics.xt_numerics import resample_grids

        first, second = av.new_xt_counts(), av.new_xt_counts()
        first[2][[0, 2]] = 1; first[3][[0, 2]] = 1
        second[2][0] = 9
        result = resample_grids([first, second], replicates=200, seed=17)
        # The pooled estimate is1/10, whereas averaging match estimates gives1/2.
        self.assertEqual(result['median_grid'][0][0], .1)
        self.assertEqual(result['lower_grid'][0][0], 0)
        self.assertEqual(result['upper_grid'][0][0], 1)
        self.assertEqual(result['match_support']['shots'][0][0], 2)
        self.assertEqual(result['match_support']['goals'][0][0], 1)
        self.assertEqual(result['unobserved_replicates'][0][1], 200)
        self.assertTrue(0 < result['unobserved_replicates'][0][2] < 200)
        self.assertEqual(result['zero_value_replicates'][0][1], 200)
        self.assertEqual(result, resample_grids([first, second], replicates=200, seed=17))
        for kwargs in ({'replicates': 1}, {'replicates': True}, {'seed': -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                resample_grids([first, second], **kwargs)
        with self.assertRaisesRegex(ValueError, 'two'):
            resample_grids([first])

    def test_reliability_bins_partition_probability_edges_exactly(self):
        probability = np.column_stack((np.linspace(0, 1, 11), np.linspace(0, 1, 11)))
        report = av.metrics(np.zeros((11, 2)), probability)
        for head in av.HEADS:
            self.assertEqual(sum(b['count'] for b in report[head]['reliability']), 11)


if __name__ == '__main__':
    unittest.main()
