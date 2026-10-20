"""Identity, exposure and chronological guardrails for buildup/role experiments."""
import copy
import json
import sqlite3

from importlib.util import find_spec
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HAS_NUMPY = find_spec("numpy") is not None
HAS_SKLEARN = find_spec("sklearn") is not None
if HAS_NUMPY:
    import numpy as np

from app.analytics import buildup_roles as br


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
                 patch('app.analytics.assessment_data.load_players', return_value=({}, {}, ['rebuilt'], {}, {})) as rebuild:
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
            self.assertEqual(cache.read_bytes(), original_bytes)
