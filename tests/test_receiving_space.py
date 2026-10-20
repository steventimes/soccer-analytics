import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.analytics.receiving_space import (
    CausalVelocity, complete_frame_quality, control_surfaces, synchronization_experiment,
)
from app.analytics.tracking import metrica_csv_frames, metrica_epts_frames, skillcorner_frames, summarize_tracking


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
        from app.analytics.receiving_space import _sample_match
        rows = [frame(i * .04, 1 if i < 50 else 2) for i in range(100)]
        for row in rows[30:40]:
            row['ball']['x'] = None
        # Two cold-start period frames plus ten missing-ball frames are ineligible.
        with patch('app.analytics.receiving_space.metrica_csv_frames', return_value=iter(rows)):
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
        from app.analytics.receiving_space import _run_id, _skillcorner_prefix
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


if __name__ == '__main__':
    unittest.main()
