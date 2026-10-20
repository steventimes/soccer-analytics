"""Separate-provider buildup persistence and native event-frequency role stability.

All database access is read-only. Individual panels, predictions, weights and
neighbor identities are private artifacts; report.json contains aggregates only.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import sqlite3
import time


from .artifacts import file_digest, source_signature
from .new_source_findings import load_frame, correlation

SEED = 20261014
STYLE_FIELDS = ('Pass', 'Shot', 'Duel', 'Free Kick', 'Others on the ball')
PERFORMANCE_FIELDS = ('np_xg', 'xa', 'np_goals', 'shots', 'key_passes')


def finite(value):
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def per90(row, field):
    minutes = finite(row.get('minutes'))
    if minutes is None or minutes <= 0:
        return None
    if field == 'finishing_residual':
        goals, xg = finite(row.get('np_goals')), finite(row.get('np_xg'))
        value = goals - xg if goals is not None and xg is not None else None
    else:
        value = finite(row.get(field))
    if value is None or (value < 0 and field != 'finishing_residual'):
        return None
    return value * 90 / minutes


def consecutive_pairs(rows, minimum_minutes=900):
    index = {}
    for row in rows:
        if row.get('provider') != 'understat':
            raise ValueError('Expected understat provider identity')
        key = (row['league'], int(row['season_id']), str(row['player_id']))
        if key in index:
            raise ValueError('Duplicate league/season/player source identity')
        index[key] = row
    result = []
    for (league, year, pid), row in sorted(index.items()):
        nxt = index.get((league, year + 1, pid))
        if not nxt or year > 2024 or year < 2014:
            continue
        a, b = finite(row.get('minutes')), finite(nxt.get('minutes'))
        if a is not None and b is not None and min(a, b) >= minimum_minutes:
            result.append((row, nxt))
    return result


def raw_player_index(connection, receipt):
    latest = {}
    for aid, path, digest, payload in connection.execute(
        """SELECT a.id,a.path,a.sha256,d.payload FROM assets a
        JOIN datasets s ON s.id=a.dataset_id JOIN documents d ON d.asset_id=a.id
        WHERE s.name='soccerdata:Understat' AND a.complete=1 AND
        (a.path LIKE 'Understat-history/cache/league_%_season_%'
        OR a.path LIKE 'Understat-2026-27/cache/league_%_season_%') ORDER BY a.id"""
    ):
        latest[path.split('@')[0]] = (aid, path, digest, payload)
    index = {}
    for aid, path, digest, payload in latest.values():
        if hashlib.sha256(payload.encode()).hexdigest() != digest:
            raise ValueError('Raw Understat cache SHA mismatch')
        match = re.search(r'league_(\d+)_season_(\d+)\.json', path)
        if not match:
            raise ValueError('Unrecognized source cache identity')
        league_id, year = match.group(1), int(match.group(2))
        receipt[aid] = dict(asset_id=aid, path=path, sha256=digest)
        for row in json.loads(payload)['players']:
            key = (league_id, year, str(row['id']))
            if key in index:
                raise ValueError('Duplicate raw league/season/player identity')
            index[key] = row
    return index


def reconcile_players(rows, raw):
    result, seen = [], set()
    for row in rows:
        key = (str(row['league_id']), int(row['season_id']), str(row['player_id']))
        if key in seen:
            raise ValueError('Duplicate league/season/player source identity')
        seen.add(key)
        source = raw.get(key)
        if source is None:
            raise ValueError('Missing original Understat player record')
        for exported, original in [('minutes', 'time'), ('xg_buildup', 'xGBuildup'), ('xg_chain', 'xGChain')]:
            a, b = finite(row.get(exported)), finite(source.get(original))
            if a is None or b is None or not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError('Understat raw/export measure mismatch')
        clubs = [s.strip() for s in source['team_title'].split(',') if s.strip()]
        result.append({**row, 'provider': 'understat', 'season_id': int(row['season_id']),
                       'position': source['position'], 'source_team_title': source['team_title'],
                       'team': clubs[0] if len(clubs) == 1 else None,
                       'team_id': row.get('team_id') if len(clubs) == 1 else None,
                       'team_scope': 'single_club' if len(clubs) == 1 else 'whole_league_multiple_clubs',
                       'partial_season': int(row['season_id']) == 2026})
    return result


def paired_bootstrap(values, labels, draws=1000):
    groups = defaultdict(list)
    for value, label in zip(values, labels):
        groups[str(label)].append(float(value))
    if len(values) < 20 or len(groups) < 10:
        return {'status': 'insufficient_sample', 'mean': None, 'player_cluster_ci95': None,
                'rows': len(values), 'players': len(groups), 'bootstrap_draws': 0}
    import numpy as np

    ordered = [groups[k] for k in sorted(groups)]
    sums = np.array([sum(v) for v in ordered]); counts = np.array([len(v) for v in ordered])
    rng = np.random.default_rng(SEED)
    estimates = []
    for _ in range(draws):
        idx = rng.integers(len(groups), size=len(groups))
        estimates.append(float(sums[idx].sum() / counts[idx].sum()))
    return {'status': 'estimated', 'mean': float(np.mean(values)),
            'player_cluster_ci95': np.quantile(estimates, [.025, .975]).tolist(),
            'rows': len(values), 'players': len(groups), 'bootstrap_draws': draws}


def persistence(pairs, draws=1000):
    result = {}
    for name in ('xg_buildup', 'xg_chain', 'np_xg', 'finishing_residual'):
        selected = [(a,b) for a,b in pairs if per90(a,name) is not None and per90(b,name) is not None]
        if len(selected) < 20 or len({a['player_id'] for a,b in selected}) < 10:
            result[name] = {'status':'insufficient_sample','r':None,'rows':len(selected)}
        else:
            result[name] = {'status':'estimated', **correlation(
                [per90(a,name) for a,b in selected], [per90(b,name) for a,b in selected],
                [str(a['player_id']) for a,b in selected], draws=draws)}
    return result


def temporal_models(pairs, draws=1000):
    import numpy as np

    fields_all = PERFORMANCE_FIELDS + ('xg_buildup', 'xg_chain')
    valid = [(a,b) for a,b in pairs if int(a['season_id']) <= 2024
             and int(b['season_id']) == int(a['season_id'])+1
             and int(b['season_id']) <= 2025
             and all(per90(a,f) is not None for f in fields_all)
             and per90(b,'xg_buildup') is not None]
    splits = {'train':[(a,b) for a,b in valid if int(a['season_id']) <= 2021],
              'validation':[(a,b) for a,b in valid if int(a['season_id']) == 2022],
              'test':[(a,b) for a,b in valid if int(a['season_id']) in (2023,2024)]}
    report = {'target':'next-season xg_buildup per90', 'sample':{k:len(v) for k,v in splits.items()},
              'windows':{'train_input':[2014,2021], 'validation_input':[2022],
                         'test_input':[2023,2024], 'test_target':[2024,2025]},
              'partial_2026_excluded':True}
    if min(map(len,splits.values())) < 10:
        return {**report,'status':'insufficient_sample','models':{},'validation_selected_model':None}, {'fitted_models':{},'predictions':[]}
    categories = sorted({(a['league'],a['position']) for a,b in splits['train']})
    models, fitted, predictions, losses = {}, {}, defaultdict(dict), {}
    target = {s:np.array([per90(b,'xg_buildup') for a,b in rows]) for s,rows in splits.items()}
    for label, fields in [('performance', PERFORMANCE_FIELDS), ('augmented', fields_all)]:
        numeric = {s:np.array([[per90(a,f) for f in fields] for a,b in rows]) for s,rows in splits.items()}
        mean = numeric['train'].mean(axis=0); scale = numeric['train'].std(axis=0)
        scale[scale == 0] = 1
        design = {}
        for split, rows in splits.items():
            cats = np.array([[float((a['league'],a['position']) == c) for c in categories] for a,b in rows])
            design[split] = np.column_stack((np.ones(len(rows)), (numeric[split]-mean)/scale, cats))
        penalty = np.eye(design['train'].shape[1]); penalty[0,0] = 0
        candidates = []
        for alpha in (0.1, 1.0, 10.0, 100.0):
            weights = np.linalg.solve(design['train'].T @ design['train'] + alpha*penalty,
                                      design['train'].T @ target['train'])
            validation = np.maximum(0, design['validation'] @ weights)
            candidates.append((float(np.mean(abs(validation-target['validation']))), alpha, weights))
        val_mae, alpha, weights = min(candidates,key=lambda c:(c[0],c[1]))
        test_predictions = np.maximum(0, design['test'] @ weights)
        losses[label] = abs(test_predictions-target['test'])
        models[label] = {'features':[f+'_per90' for f in fields]+['league/source_position'],
                         'alpha':alpha,'validation_mae':val_mae,'test_mae':float(losses[label].mean())}
        fitted[label] = {'fields':list(fields),'mean':mean.tolist(),'scale':scale.tolist(),
                         'categories':[list(c) for c in categories],'weights':weights.tolist(),'alpha':alpha}
        for i,value in enumerate(test_predictions): predictions[i][label] = float(value)
    carry = np.array([per90(a,'xg_buildup') for a,b in splits['test']])
    losses['carry_forward'] = abs(carry-target['test'])
    models['carry_forward'] = {'features':['xg_buildup_per90'],
        'validation_mae':float(np.mean([abs(per90(a,'xg_buildup')-per90(b,'xg_buildup')) for a,b in splits['validation']])),
        'test_mae':float(losses['carry_forward'].mean())}
    private_rows = []
    for i,(a,b) in enumerate(splits['test']):
        private_rows.append({'provider':'understat','league':a['league'],'player_id':str(a['player_id']),
                             'input_season':int(a['season_id']),'target_season':int(b['season_id']),
                             'target':float(target['test'][i]),'carry_forward':float(carry[i]),**predictions[i]})
    labels = [a['player_id'] for a,b in splits['test']]
    comparisons = {a+'_minus_'+b:paired_bootstrap(losses[a]-losses[b],labels,draws)
                   for a,b in [('augmented','performance'),('augmented','carry_forward'),('performance','carry_forward')]}
    return {**report,'status':'estimated','models':models,
            'validation_selected_model':min(models,key=lambda m:models[m]['validation_mae']),
            'paired_test_mae_differences':comparisons,
            'estimator':'Train-only standardized ridge; alpha chosen on 2022 input validation; no refitting on test.'}, {'fitted_models':fitted,'predictions':private_rows}


def native_counts(connection):
    """All source-native types, independent of accuracy, duel outcome or goal tags."""
    counts = defaultdict(dict)
    for mid,pid,kind,n in connection.execute(
        "SELECT match_id,player_id,event_type,COUNT(*) FROM wyscout_events "
        "WHERE period IN ('1H','2H') AND player_id > 0 "
        "GROUP BY match_id,player_id,event_type"
    ):
        counts[(str(mid),str(pid))][kind] = n
    return counts


def half_panels(observations, counts):
    groups, seen, audit = {}, set(), Counter()
    for row in observations:
        key = (str(row['match_id']),str(row['entity_id']))
        if key in seen:
            raise ValueError('Duplicate Wyscout match/player exposure')
        seen.add(key)
        minutes = finite(row.get('minutes'))
        if minutes is None or minutes <= 0:
            audit['invalid_exposure_rows'] += 1
            continue
        if not '2017-07-01' <= row['date'] < '2018-07-01':
            audit['outside_season'] += 1
            continue
        half = 'early' if row['date'] < '2018-01-01' else 'late'
        group_key = (row['league'],row['position'],str(row['entity_id']),half)
        group = groups.setdefault(group_key, {'provider':'wyscout','league':row['league'],
            'position':row['position'],'player_id':str(row['entity_id']),'half':half,
            'minutes':0.0,'matches':0, 'minutes_sources':Counter(),
            **dict.fromkeys(STYLE_FIELDS+('goals','passes','accurate_passes','duels','duel_wins'),0)})
        group['minutes'] += minutes; group['matches'] += 1
        group['minutes_sources'][row['minutes_source']] += 1
        for field in STYLE_FIELDS: group[field] += counts.get(key,{}).get(field,0)
        for field in ('goals','passes','accurate_passes','duels','duel_wins'): group[field] += row[field]
    audit['native_event_match_players_without_valid_exposure'] = len(set(counts)-seen)
    return [groups[k] for k in sorted(groups)], dict(audit)


def neighbor_stability(early, late, ids, k=5):
    if len(ids) < max(20,k+2):
        return {'status':'insufficient_sample','players':len(ids),'mean_jaccard':None,'neighbors':[]}
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(early)
    a,b = scaler.transform(early),scaler.transform(late)
    neighbor_lists = []
    for matrix in (a,b):
        distance = ((matrix[:,None,:]-matrix[None,:,:])**2).sum(axis=2)
        np.fill_diagonal(distance,np.inf)
        neighbor_lists.append(np.argsort(distance,axis=1,kind='stable')[:,:k])
    rows=[]
    for i,pid in enumerate(ids):
        x,y = ({ids[j] for j in neighbors[i]} for neighbors in neighbor_lists)
        rows.append({'player_id':pid,'early':sorted(x),'late':sorted(y),
                     'jaccard':len(x&y)/len(x|y),'overlap':len(x&y)/k})
    return {'status':'estimated','players':len(ids),'k':k,
            'mean_jaccard':float(np.mean([r['jaccard'] for r in rows])),
            'mean_overlap':float(np.mean([r['overlap'] for r in rows])),
            'scaler':{'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist()},'neighbors':rows}


def load_wyscout_observations(connection, cache=Path('.tmp/assessments/player-data.json')):
    """Reuse only observations whose receipt covers all extraction dependencies.

    Legacy shared assessment caches lack event_validation.py's digest. Leave
    those files untouched and rebuild observations through the read-only source.
    """
    from .assessment_data import load_players

    cache = Path(cache)
    inputs = {
        'sources': source_signature(connection, ('wyscout',)),
        'extractor_sha256': file_digest(Path(__file__).with_name('assessment_data.py')),
        'event_validation_sha256': file_digest(Path(__file__).with_name('event_validation.py')),
    }
    receipt_path = cache.with_suffix('.receipt.json')
    verified = False
    receipt = None
    if cache.exists() and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        verified = receipt.get('inputs') == inputs and receipt.get('content_sha256') == file_digest(cache)
    if verified:
        data = json.loads(cache.read_text())
        observations, matches, audit = data['observations'], data['matches'], data['audit']
    else:
        _, matches, observations, _, audit = load_players(connection)
    provenance = {'inputs': inputs, 'verified_cache_used': verified,
                  'accepted_cache_receipt': receipt if verified else None}
    return matches, observations, audit, provenance


def wyscout_analysis(connection, minimum_minutes):
    matches, observations, audit, cache_provenance = load_wyscout_observations(connection)
    inputs = cache_provenance['inputs']
    source, extractor = inputs['sources'], inputs['extractor_sha256']
    verified = cache_provenance['verified_cache_used']
    counts = native_counts(connection)
    # Restrict native counts to the same five regular-time domestic league fixtures.
    match_ids = {str(mid) for mid in matches}
    counts = {key:value for key,value in counts.items() if key[0] in match_ids}
    if match_ids - {key[0] for key in counts}:
        raise ValueError('Missing native events for an eligible Wyscout match')
    panels,panel_audit = half_panels(observations,counts)
    index = {(r['league'],r['position'],r['player_id'],r['half']):r for r in panels}
    cohorts = defaultdict(list)
    exclusions = Counter()
    for (league,position,pid,half), a in index.items():
        if half != 'early': continue
        b = index.get((league,position,pid,'late'))
        if not b or min(a['minutes'],b['minutes']) < minimum_minutes/2:
            exclusions['missing_half_or_below_exposure'] += 1; continue
        if min(a['passes'],b['passes'],a['duels'],b['duels']) <= 0:
            exclusions['undefined_performance_denominator'] += 1; continue
        cohorts[(league,position)].append((a,b))
    summaries,private,all_deltas,all_ids=[],[],[],[]
    all_jaccard = {'frequency_only': [], 'performance_only': []}
    for (league,position), pairs in sorted(cohorts.items()):
        pairs.sort(key=lambda ab:ab[0]['player_id'])
        ids=[a['player_id'] for a,b in pairs]
        vectors={}
        for label in ('frequency_only','performance_only'):
            halves=[]
            for idx in (0,1):
                rows=[p[idx] for p in pairs]
                halves.append([[r[f]*90/r['minutes'] for f in STYLE_FIELDS] for r in rows] if label == 'frequency_only'
                              else [[r['goals']*90/r['minutes'],r['accurate_passes']/r['passes'],r['duel_wins']/r['duels']] for r in rows])
            result=neighbor_stability(*halves,ids)
            private.append({'league':league,'position':position,'model':label,**result})
            vectors[label]=result
        summary={'league':league,'position':position,'players':len(ids)}
        for label,result in vectors.items():
            summary[label]={k:v for k,v in result.items() if k not in ('neighbors','scaler')}
        if vectors['frequency_only']['status'] == 'estimated':
            deltas=[a['jaccard']-b['jaccard'] for a,b in zip(vectors['frequency_only']['neighbors'],vectors['performance_only']['neighbors'])]
            summary['paired_jaccard_difference']=paired_bootstrap(deltas,ids)
            all_deltas.extend(deltas);all_ids.extend(ids)
            for label in all_jaccard:
                all_jaccard[label].extend(r['jaccard'] for r in vectors[label]['neighbors'])
        summaries.append(summary)
    return {'status':'estimated' if all_deltas else 'insufficient_sample',
            'source':'wyscout','season':'2017/18','cut_date':'2018-01-01',
            'regular_matches':len(matches),'observation_rows':len(observations),'panel_rows':len(panels),
            'native_event_records_with_known_player':sum(sum(types.values()) for types in counts.values()),
            'match_window_counts':dict(Counter('early' if m['dateutc'][:10] < '2018-01-01' else 'late' for m in matches.values())),
            'leagues':sorted({r['league'] for r in observations}), 'minimum_minutes_each_half':minimum_minutes/2,
            'frequency_features':[f+' per90' for f in STYLE_FIELDS],
            'performance_features':['goals per90','known-outcome pass completion','known-outcome duel win rate'],
            'candidate_rule':'Same eligible player IDs in both halves within league/source role; self excluded; k=5; minimum cohort=20.',
            'scaling':'StandardScaler fitted separately to each early-half league/role cohort, reused unchanged on late-half.',
            'cohorts':summaries,
            'overall_jaccard':{label:paired_bootstrap(values,all_ids) for label,values in all_jaccard.items()},
            'paired_frequency_minus_performance_jaccard':paired_bootstrap(all_deltas,all_ids),
            'exclusions':dict(exclusions),'extractor_audit':audit,'panel_audit':panel_audit,
            'minutes':'Supplied PlayeRank exposure for outfield players; nominal formation/substitution/red-card regular-time intervals for omitted goalkeepers.',
            'source_receipt':{'signature':source,'assessment_extractor_sha256':extractor,
                              'event_validation_sha256':inputs['event_validation_sha256'],
                              'verified_cache_used':verified}}, {'panels':panels,'neighbors':private,
                              'cache_receipt':cache_provenance}


def summarize_profiles(rows, minimum_minutes):
    import numpy as np

    cohorts=defaultdict(list)
    for row in rows:
        if finite(row.get('minutes')) is not None and float(row['minutes']) >= minimum_minutes:
            cohorts[(row['league'],row['position'])].append(row)
    result=[]
    for (league,position), selected in sorted(cohorts.items()):
        summary={'league':league,'source_position':position,'player_season_rows':len(selected),
                 'minimum_minutes':minimum_minutes}
        for field in ('xg_buildup','xg_chain'):
            values=[per90(r,field) for r in selected if per90(r,field) is not None]
            summary[field+'_per90_median'] = float(np.median(values)) if len(values)>=10 else None
            summary[field+'_per90_iqr'] = np.quantile(values,[.25,.75]).tolist() if len(values)>=10 else None
        result.append(summary)
    return result


def run(database='data/football.sqlite3', output='.tmp/analysis-upgrades-20261014/buildup', minimum_minutes=900):
    if finite(minimum_minutes) is None or minimum_minutes <= 0:
        raise ValueError('minimum_minutes must be finite and positive')
    import numpy as np

    started=time.monotonic();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    receipt={}
    with sqlite3.connect(Path(database).resolve().as_uri()+'?mode=ro',uri=True) as connection:
        historical=load_frame(database,connection,'Understat-history','player_season_stats',receipt)
        current=load_frame(database,connection,'Understat-2026-27','player_season_stats',receipt)
        raw=raw_player_index(connection,receipt)
        panel=reconcile_players(historical+current,raw)
        if any(not 2014 <= int(r['season_id']) <= 2025 for r in historical) or any(int(r['season_id']) != 2026 for r in current):
            raise ValueError('Unexpected complete/current source season boundary')
        wyscout,wprivate=wyscout_analysis(connection,minimum_minutes)
    history=[r for r in panel if not r['partial_season']];latest=[r for r in panel if r['partial_season']]
    pairs=consecutive_pairs(panel,minimum_minutes)
    prediction,pprivate=temporal_models(pairs)
    config={'minimum_minutes':minimum_minutes,'current_descriptive_minimum_minutes':270,
            'bootstrap_draws':1000,'seed':SEED,'repeatability_bootstrap_seed':20261009,
            'ridge_alphas':[0.1,1.0,10.0,100.0],'neighbor_k':5,'minimum_neighbor_cohort':20,
            'source_providers_joined':False,
            'historical_seasons':[2014,2025],'current_partial_season':2026,
            'wyscout_cut':'2018-01-01','wyscout_minimum_half_minutes':minimum_minutes/2}
    codes={name:file_digest(Path(__file__).with_name(name)) for name in
           ('buildup_roles.py','new_source_findings.py','assessment_data.py','event_validation.py','artifacts.py')}
    codes['database.py']=file_digest(Path(__file__).parents[1]/'data_service/database.py')
    provenance={'understat_assets':sorted(receipt.values(),key=lambda r:r['asset_id']),
                'wyscout':{'signature':wyscout['source_receipt']['signature'],
                           'assessment_extractor_sha256':wyscout['source_receipt']['assessment_extractor_sha256'],
                           'event_validation_sha256':wyscout['source_receipt']['event_validation_sha256']},
                'code_sha256':codes,'configuration':config}
    run_id=hashlib.sha256(json.dumps(provenance,sort_keys=True,allow_nan=False).encode()).hexdigest()[:20]
    cohort_counts=Counter((a['league'],a['position']) for a,b in pairs)
    repeatability_cohorts = []
    for field in ('league', 'position'):
        for value in sorted({a[field] for a,b in pairs}):
            selected = [(a,b) for a,b in pairs if a[field] == value and b[field] == value]
            repeatability_cohorts.append({'dimension':field,'value':value,
                                         'same_label_in_both_years':True,
                                         'metrics':persistence(selected)})
    current_minutes=[float(r['minutes']) for r in latest]
    report={'schema_version':1,'run_kind':'buildup_roles','run_id':run_id,'configuration':config,
            'understat':{'historical_rows':len(history),'current_partial_rows':len(latest),
              'source_seasons':sorted({r['season_id'] for r in history}),
              'returner_pairs':len(pairs),'repeatability':persistence(pairs),
              'repeatability_cohorts':repeatability_cohorts,
              'higher_exposure_repeatability':persistence(consecutive_pairs(panel,max(1800,minimum_minutes))),
              'pair_cohorts':[{'league':k[0],'source_position':k[1],'pairs':v} for k,v in sorted(cohort_counts.items())],
              'historical_profiles':summarize_profiles(history,minimum_minutes),
              'next_season_prediction':prediction,
              'current_partial':{'status':'descriptive_partial_season','rows':len(latest),
                'minutes_quantiles':dict(zip(('min','p25','median','p75','max'),np.quantile(current_minutes,[0,.25,.5,.75,1]).tolist())),
                'rows_at_least270_minutes':sum(m>=270 for m in current_minutes),
                'rows_at_least_historical_threshold':sum(m>=minimum_minutes for m in current_minutes),
                'profiles':summarize_profiles(latest,270)},
              'multi_club_seasons':sum(r['team_scope']!='single_club' for r in panel),
              'current_multi_club_seasons':sum(r['team_scope']!='single_club' for r in latest)},
            'wyscout':wyscout,'provenance':{'source_asset_set_sha256':hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest(),
                                         'code_sha256':codes,'understat_asset_count':len(receipt),
                                         'wyscout_signature':wyscout['source_receipt']['signature']},
            'runtime':{'seconds':round(time.monotonic()-started,3),'python':platform.python_version(),'numpy':np.__version__},
            'limits':[
                'xGChain and xGBuildup are attributed possession involvement measures, never additive player causal credit or team totals.',
                'Same-league consecutive returners and two-half players must meet exposure thresholds; absent players and league switchers are excluded.',
                'Bootstrap intervals are conditional on selected player panels, fixed fitted models and fixed neighbor graphs; repeated rows are clustered by provider player ID. Neighbor graph dependence between different players is not captured.',
                'Understat composite source positions are retained. S denotes substitute; position-specific minutes and actual coaching duties are unavailable.',
                'Multiple-club Understat season totals stay at league/player grain; SDK first-listed club receives no attribution.',
                '2026/27 is partial descriptive coverage with a separate 270-minute threshold, never a completed training/evaluation target or growth estimate.',
                'Native Wyscout duels are event records, not SPADL actions. Frequency features include every outcome and unknown outcome.',
                'Wyscout performance ratios use only known-outcome passing/duel subsets from the existing extractor; missing denominators exclude a player from both models.',
                'Broad static Wyscout roles are source labels. Stable neighbors do not establish quality, development potential, transfer suitability or causal roles.',
                'Understat and Wyscout identities and analyses remain separate; no cross-provider player matching is attempted.' ]}
    def write(name,value):
        (output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    write('provenance.json',{**provenance,'cache_verification':wprivate['cache_receipt']})
    write('understat-panel.json',panel)
    write('current-profiles.json', [
        {'provider':'understat','league':r['league'],'season_id':r['season_id'],
         'player_id':r['player_id'],'source_position':r['position'],
         'team_scope':r['team_scope'],'minutes':finite(r['minutes']),
         'descriptive_minimum_minutes':270,'meets_threshold':float(r['minutes'])>=270,
         **{field+'_per90':per90(r,field) for field in PERFORMANCE_FIELDS+('xg_buildup','xg_chain')}}
        for r in latest])
    write('predictions.json',pprivate['predictions'])
    write('model-weights.json',pprivate['fitted_models'])
    write('wyscout-panels.json',wprivate['panels'])
    write('neighbors.json',wprivate['neighbors'])
    write('report.json',report)
    return report
