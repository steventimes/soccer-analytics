"""Retrospective SPADL state probabilities and incremental action values.

Run with requirements-vaep.txt in an isolated environment. Existing native
pre-pass experiments and production scores are not inputs to this experiment.
"""

from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sqlite3
import warnings

import numpy as np

from app.file_io import write_json_atomic
from .artifacts import file_digest, source_signature

SPLITS = ('train', 'validation', 'test')
METHODS = ('prior', 'single', 'context')
HEADS = ('score', 'concede')
HORIZON = 10


def calendar_split(records):
    """Whole matches, fixed dates shared with the existing native experiment."""
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('Duplicate match identity')
    groups = {key: [] for key in SPLITS}
    for row in sorted(records, key=lambda r: (r['date'], r['id'])):
        if not '2017-07-01' <= row['date'] < '2018-07-01':
            continue
        split = 'train' if row['date'] < '2018-01-01' else 'validation' if row['date'] < '2018-03-01' else 'test'
        groups[split].append(row)
    if any(not rows for rows in groups.values()):
        raise ValueError('Every calendar split needs matches')
    return groups


def scoring_teams(actions, teams):
    """Wyscout own goals are bad_touch, not necessarily shot actions."""
    if len(set(teams)) != 2 or not actions.team_id.isin(teams).all():
        raise ValueError('Exactly two known action teams required')
    own = actions.result_name.eq('owngoal').to_numpy()
    goal = (actions.type_name.str.contains('shot') & actions.result_name.eq('success')).to_numpy()
    credited = np.where(goal, actions.team_id.to_numpy(dtype=np.int64), 0)
    credited[own] = sum(teams) - actions.team_id.to_numpy(dtype=np.int64)[own]
    return credited


def period_states(actions, home_team, horizon=HORIZON, teams=None):
    """Reset lags and labels each period; preserve match-wide score context."""
    import pandas as pd
    from socceraction.spadl import add_names
    from socceraction.vaep import VAEP

    if horizon < 1:
        raise ValueError('Positive action horizon required')
    actions = add_names(actions).reset_index(drop=True)
    if actions.empty or actions.game_id.nunique() != 1 or not actions.period_id.isin([1, 2]).all():
        raise ValueError('One regular-time match required')
    teams = tuple(actions.team_id.unique()) if teams is None else teams
    credits = scoring_teams(actions, teams)
    before = {team: np.cumsum(credits == team) - (credits == team) for team in teams}
    actor = actions.team_id.to_numpy()
    own_score = np.where(actor == teams[0], before[teams[0]], before[teams[1]])
    other_score = np.where(actor == teams[0], before[teams[1]], before[teams[0]])
    score_context = pd.DataFrame({'goalscore_team': own_score, 'goalscore_opponent': other_score,
                                 'goalscore_diff': own_score - other_score})
    features = {'single': [], 'context': []}
    labels, eligible = [], []
    for _, part in actions.groupby('period_id', sort=False):
        part_credits = credits[part.index]
        part = part.reset_index(drop=True)
        for name, lags in [('single', 1), ('context', 3)]:
            block = VAEP(nb_prev_actions=lags).compute_features(pd.Series({'home_team_id': home_team}), part)
            features[name].append(block)
        scored, conceded = np.zeros(len(part), dtype=bool), np.zeros(len(part), dtype=bool)
        actor = part.team_id.to_numpy()
        for lag in range(min(horizon, len(part))):
            future = part_credits[lag:]; count = len(future)
            scored[:count] |= future == actor[:count]
            conceded[:count] |= (future != 0) & (future != actor[:count])
        labels.append(np.column_stack([scored, conceded]))
        # The official label helper pads the last action. Exclude incomplete
        # windows rather than interpreting padding as observed future actions.
        eligible.extend(np.arange(len(part)) + horizon <= len(part))
    features = {name: pd.concat(blocks, ignore_index=True) for name, blocks in features.items()}
    for block in features.values():
        block[score_context.columns] = score_context
        if not np.isfinite(block.to_numpy(dtype=np.float32)).all():
            raise ValueError('Nonfinite SPADL features')
    labels = np.concatenate(labels).astype(np.float32)
    return dict(actions=actions, features=features, labels=labels,
                eligible=np.asarray(eligible, dtype=bool), terminal=credits != 0)


def action_values(actions, probabilities):
    """Official perspective swaps/priors, period resets and own-goal reset fix."""
    import pandas as pd
    from socceraction.vaep.formula import value

    probabilities = np.asarray(probabilities)
    if probabilities.shape != (len(actions), 2) or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError('Aligned score/concede probabilities in [0,1] required')
    blocks = []
    for _, part in actions.groupby('period_id', sort=False):
        indices = part.index.to_numpy()
        part = part.reset_index(drop=True)
        scores = pd.Series(probabilities[indices, 0])
        concedes = pd.Series(probabilities[indices, 1])
        values = value(part, scores, concedes)
        # Upstream 1.5.3 resets after successful shots, but misses own goals.
        # A new post-goal phase must not inherit the opposing team's old value.
        own = part.result_name.shift().eq('owngoal')
        active = own & part.time_seconds.diff().abs().le(10)
        same = part.team_id.eq(part.team_id.shift())
        previous_score = scores.shift().where(same, concedes.shift()).fillna(0)
        previous_concede = concedes.shift().where(same, scores.shift()).fillna(0)
        fixed = part.type_name.isin(['shot_penalty', 'corner_crossed', 'corner_short'])
        values.loc[active & ~fixed, 'offensive_value'] += previous_score[active & ~fixed]
        values.loc[active, 'defensive_value'] -= previous_concede[active]
        values['vaep_value'] = values.offensive_value + values.defensive_value
        values.iloc[0] = 0.  # No observed previous state at the period boundary.
        values.index = indices
        blocks.append(values)
    return pd.concat(blocks).sort_index()


def convert_match(events, metadata):
    from socceraction.data.wyscout import WyscoutLoader
    from socceraction.spadl import SPADLSchema
    from socceraction.spadl.wyscout import convert_to_actions
    from .action_value import audit_goals

    if not events or len({e['id'] for e in events}) != len(events) or any(e['matchId'] != metadata['wyId'] for e in events):
        raise ValueError('Missing/duplicate/wrong-match native events')
    problem = audit_goals(events, metadata)
    if problem:
        raise ValueError('Native goal audit: ' + json.dumps(problem, sort_keys=True))
    home = next(int(team) for team, row in metadata['teamsData'].items() if row['side'] == 'home')
    # The public loader accepts a callable getter. No files, remote fetch or
    # credentials are needed to feed the already verified warehouse records.
    expected = {int(team): int(row['score']) for team, row in metadata['teamsData'].items()}
    filtered = [e for e in events if not (e['eventId'] == 9 and any(tag['id'] == 101 for tag in e['tags']))]
    loader = WyscoutLoader(root='', getter=lambda _: {'events': filtered}, creds={}, feeds={'events': '{game_id}'})
    actions = SPADLSchema.validate(convert_to_actions(loader.events(metadata['wyId']), home))
    prepared = period_states(actions, home, teams=tuple(expected))
    actions = prepared['actions']
    credits = scoring_teams(actions, tuple(expected))
    actual = Counter(credits[credits != 0])
    if {team: actual[team] for team in expected} != expected or not actions.team_id.isin(expected).all():
        raise ValueError('Converted SPADL goals/teams disagree with match metadata')
    prepared['audit'] = {'native_events': len(events), 'actions': len(actions),
                         'ignored_goalkeeper_goal_duplicates': len(events) - len(filtered),
                         'inferred_dribbles': int(actions.original_event_id.isna().sum()),
                         'incomplete_horizon_states': int((~prepared['eligible']).sum())}
    return prepared


def load_league(database, league, scratch):
    """Read one consistent warehouse snapshot, preserve actual exclusion reasons."""
    if league not in ('England', 'France'):
        raise ValueError('This benchmark covers England and France')
    pieces = {split: [] for split in SPLITS}
    excluded, audit, warning_counts = [], Counter(), Counter()
    payload_digest = hashlib.sha256()
    private = []
    with sqlite3.connect(Path(database).resolve().as_uri() + '?mode=ro', uri=True) as db:
        db.execute('BEGIN')
        matches = [json.loads(row[0]) for row in db.execute('SELECT payload FROM wyscout_entities WHERE entity_type=?', ('matches_' + league,))]
        matches = [m for m in matches if m['status'] == 'Played' and m['duration'] == 'Regular']
        by_id = {m['wyId']: m for m in matches}
        groups = calendar_split([dict(id=m['wyId'], date=m['dateutc'][:10]) for m in matches])
        signature = source_signature(db, ('wyscout',))
        for split, records in groups.items():
            for number, record in enumerate(records, 1):
                metadata = by_id[record['id']]
                payload_digest.update(json.dumps(metadata, sort_keys=True).encode())
                events = []
                for payload, in db.execute("SELECT payload FROM wyscout_events WHERE match_id=? AND period IN ('1H','2H') ORDER BY period,seconds,row_number", (record['id'],)):
                    payload_digest.update(payload.encode()); events.append(json.loads(payload))
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('default')
                        prepared = convert_match(events, metadata)
                    warning_counts.update(str(w.message) for w in caught)
                except ValueError as error:
                    excluded.append({**record, 'split': split, 'reason': str(error)})
                    continue
                audit.update(prepared['audit'])
                prepared['match'] = {**record, 'split': split}
                pieces[split].append(prepared)
                private.append({**record, 'split': split, **prepared['audit']})
                if number % 25 == 0:
                    print(json.dumps({'league': league, 'split': split, 'processed': number, 'total': len(records),
                                      'accepted': len(pieces[split]), 'excluded_so_far': len(excluded)}), flush=True)
            if not pieces[split]:
                raise ValueError('No accepted matches in ' + split)
            print(json.dumps({'league': league, 'split': split, 'accepted': len(pieces[split]), 'excluded_so_far': len(excluded)}), flush=True)
    write_json_atomic(scratch / 'source-receipt.json', {'warehouse': signature, 'payloads_sha256': payload_digest.hexdigest(),
                      'matches': private, 'excluded': excluded, 'warnings': dict(warning_counts), 'audit': dict(audit)})
    return pieces, {'warehouse': signature, 'payloads_sha256': payload_digest.hexdigest(), 'audit': dict(audit),
                    'excluded_matches': len(excluded), 'conversion_warning_counts': dict(warning_counts)}


def arrays(matches, method):
    return tuple(np.concatenate(values) for values in [
        [m['features'][method].to_numpy(dtype=np.float32) for m in matches],
        [m['labels'] for m in matches], [m['eligible'] for m in matches],
        [m['terminal'] for m in matches], [np.full(len(m['labels']), m['match']['id']) for m in matches]])


def measure(y, p):
    from .action_value import metrics
    result = metrics(y, p)
    return {head: {key: result[head][key] for key in ('log_loss', 'brier', 'observed_rate', 'predicted_rate', 'reliability')}
            for head in HEADS}


def loss_intervals(y, model, baseline, match_ids, seed=2026):
    """Paired resampling of whole matches, not independent action rows."""
    from .action_value import binary_loss
    ids = np.unique(match_ids)
    counts = np.array([np.count_nonzero(match_ids == mid) for mid in ids])
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(ids), size=(1000, len(ids)))
    result = {}
    for j, head in enumerate(HEADS):
        differences = {'log_loss': binary_loss(y[:, j], model[:, j]) - binary_loss(y[:, j], baseline[:, j]),
                       'brier': (model[:, j] - y[:, j]) ** 2 - (baseline[:, j] - y[:, j]) ** 2}
        result[head] = {}
        for name, values in differences.items():
            totals = np.array([values[match_ids == mid].sum() for mid in ids])
            bootstrap = totals[draws].sum(axis=1) / counts[draws].sum(axis=1)
            result[head][name] = {'difference': float(values.mean()), 'interval95': np.quantile(bootstrap, [.025, .975]).tolist()}
    return result


def run_league(data, league, scratch):
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.feature_selection import VarianceThreshold
    from threadpoolctl import threadpool_limits
    from .action_value import binary_loss

    outcomes, predictions, selected = [], {}, {}
    test_y = np.concatenate([m['labels'] for m in data['test']])
    eligible = np.concatenate([m['eligible'] for m in data['test']])
    terminal = np.concatenate([m['terminal'] for m in data['test']])
    ids = np.concatenate([np.full(len(m['labels']), m['match']['id']) for m in data['test']])
    train_y = np.concatenate([m['labels'][m['eligible']] for m in data['train']])
    prior = train_y.mean(axis=0)
    predictions['prior'] = np.tile(prior, (len(test_y), 1))
    with threadpool_limits(limits=1):
        for method in ('single', 'context'):
            train, valid, test = [arrays(data[split], method) for split in SPLITS]
            selector = VarianceThreshold().fit(train[0][train[2]])
            tx, vx, ex = [selector.transform(block[0]) for block in (train, valid, test)]
            probabilities, settings, models = [], [], []
            for j, head in enumerate(HEADS):
                candidates = []
                for leaves in (15, 31):
                    model = HistGradientBoostingClassifier(max_iter=150, learning_rate=.08, max_leaf_nodes=leaves,
                        min_samples_leaf=80, l2_regularization=1., max_bins=64, early_stopping=False, random_state=2026)
                    model.fit(tx[train[2]], train[1][train[2], j])
                    score = float(binary_loss(valid[1][valid[2], j], model.predict_proba(vx[valid[2]])[:, 1]).mean())
                    candidates.append((score, leaves, model))
                score, leaves, model = min(candidates, key=lambda row: row[0])
                probabilities.append(model.predict_proba(ex)[:, 1]); models.append(model)
                settings.append({'head': head, 'max_leaf_nodes': leaves, 'validation_log_loss': score,
                                 'candidates': [{'max_leaf_nodes': n, 'validation_log_loss': s} for s, n, _ in candidates]})
            predictions[method] = np.column_stack(probabilities)
            selected[method] = {'input_features': train[0].shape[1], 'training_variable_features': int(selector.get_support().sum()), 'settings': settings}
            joblib.dump({'selector': selector, 'models': models, 'feature_names': list(data['train'][0]['features'][method].columns)}, scratch / (method + '.joblib'))
            print(json.dumps({'league': league, 'method': method, 'test': measure(test_y[eligible], predictions[method][eligible])}), flush=True)
    for method in METHODS:
        for scope, mask in [('complete_horizon', eligible), ('nonterminal_complete_horizon', eligible & ~terminal)]:
            outcomes.append({'method': method, 'scope': scope, 'states': int(mask.sum()), 'goals_in_current_action': int(terminal[mask].sum()),
                             'positive_counts': test_y[mask].sum(axis=0).astype(int).tolist(), 'metrics': measure(test_y[mask], predictions[method][mask])})
    value_groups = {}
    offset = 0
    import pandas as pd
    all_values = []
    for match in data['test']:
        count = len(match['labels']); actions = match['actions']
        values = action_values(actions, predictions['context'][offset:offset + count])
        values['type'] = actions.type_name; values['inferred'] = actions.original_event_id.isna()
        values['eligible'] = match['eligible']; values.loc[actions.groupby('period_id').head(1).index, 'eligible'] = False
        values['game_id'] = match['match']['id']; values['action_id'] = actions.action_id
        all_values.append(values); offset += count
    values = pd.concat(all_values, ignore_index=True)
    for kind, group in values[values.eligible].groupby('type'):
        value_groups[kind] = {'actions': len(group), 'inferred': int(group.inferred.sum()),
                             **{key: float(group[key].mean()) for key in ('offensive_value', 'defensive_value', 'vaep_value')}}
    values.to_csv(scratch / 'action-values.csv', index=False)
    np.savez_compressed(scratch / 'test-predictions.npz', y=test_y, eligible=eligible, terminal=terminal, match_ids=ids, **predictions)
    mask = eligible & ~terminal
    intervals = {baseline: loss_intervals(test_y[mask], predictions['context'][mask], predictions[baseline][mask], ids[mask]) for baseline in ('prior', 'single')}
    splits = {split: {'matches': len(matches), 'actions': sum(len(m['labels']) for m in matches),
                     'eligible_states': sum(int(m['eligible'].sum()) for m in matches),
                     'start': matches[0]['match']['date'], 'end': matches[-1]['match']['date']} for split, matches in data.items()}
    return {'league': league, 'splits': splits, 'training_prior': prior.tolist(), 'models': selected, 'results': outcomes,
            'paired_nonterminal_intervals': intervals, 'test_action_values': value_groups}


def run(database, output, scratch):
    scratch = Path(scratch); scratch.mkdir(parents=True, exist_ok=True)
    versions = {name: importlib.metadata.version(name) for name in ('socceraction', 'numpy', 'pandas', 'scikit-learn', 'pandera', 'multimethod')}
    if versions['socceraction'] != '1.5.3':
        raise ValueError('The audited socceraction version is 1.5.3')
    reports = []
    for league in ('England', 'France'):
        target = scratch / league; target.mkdir(exist_ok=True)
        print('Compare post-action probabilities on fixed calendar holdouts; do not interpret values as causal effects.', flush=True)
        data, source = load_league(database, league, target)
        report = run_league(data, league, target); report['source'] = source; reports.append(report)
        write_json_atomic(target / 'summary.json', report)
    report = {'schema': 'SoccerAnalytics.SPADLVAEP.v1', 'source_url': 'https://doi.org/10.6084/m9.figshare.c.4415000',
              'paper_url': 'https://doi.org/10.1145/3292500.3330758', 'implementation_url': 'https://github.com/ML-KULeuven/socceraction/tree/v1.5.3',
              'versions': versions, 'extractor_sha256': file_digest(Path(__file__)), 'horizon': HORIZON, 'leagues': reports,
              'limits': ['Retrospective post-action states, not pre-pass forecasts or causal effects. Current action is included in the ten-action window.',
                         'Original Opta multi-season results are not reproduced by Wyscout same-season holdouts.',
                         'The official converter inserts inferred dribbles; these are not continuous observed carries.',
                         'Period lags/labels reset; incomplete horizons and period-first action values are excluded. Match score context continues across halftime.',
                         'Own goals use any action type in labels, score context and terminal detection; they reset the next value baseline. Native goal-tagged keeper duplicates are removed before conversion.',
                         'Official penalty/corner prior constants 0.792453/0.046500 are retained, not locally estimated.',
                         'Paired intervals resample held-out matches, conditional on the fitted model. Repeated teams and training uncertainty remain.',
                         'Action-type means are descriptive conditional on action mix and model calibration; no player ranking is published.']}
    report['run_id'] = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()[:20]
    write_json_atomic(Path(output), report)
    (scratch / 'notes.txt').write_text(json.dumps({'run_id': report['run_id'], 'league_summaries': reports}, ensure_ascii=False, indent=2) + '\n')
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', default='data/football.sqlite3')
    parser.add_argument('--output', default='results/vaep-spadl.json')
    parser.add_argument('--scratch', default='.tmp/vaep-spadl')
    args = parser.parse_args()
    result = run(args.database, args.output, args.scratch)
    print(json.dumps({'run_id': result['run_id'], 'leagues': len(result['leagues'])}))
