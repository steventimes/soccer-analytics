"""Offline native-Wyscout pre-pass scoring/conceding states, not incremental VAEP."""

import copy
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import numpy as np

from app.file_io import write_json_atomic
from .artifacts import file_digest, source_signature
from .representation.data import LEAGUES, pass_sequences, split_matches
from .representation.experiment import linear_inputs
from .tactics import fit_expected_threat, grid_index

SPLITS = ('train', 'validation', 'test')
HEADS = ('score', 'concede')
STOPPAGES = {2, 5, 6}  # foul, interruption, offside; event 3 starts a new spell


def goal_team(event, teams):
    """Count attacking goals once; keeper event 9 is a duplicated observation."""
    tags = {tag.get('id') for tag in event.get('tags', [])}
    if event.get('eventId') == 9:
        return None
    if 101 in tags and 102 in tags:
        raise ValueError('Ambiguous goal and own-goal tags')
    if 102 in tags:
        opponents = [team for team in teams if team != event.get('teamId')]
        if len(opponents) != 1:
            raise ValueError('Own goal requires two known teams')
        return opponents[0]
    shot = event.get('eventId') == 10 or (
        event.get('eventId') == 3 and event.get('subEventName') in {'Penalty', 'Free kick shot'}
    )
    return event.get('teamId') if shot and 101 in tags else None


def audit_goals(events, metadata):
    """Return an exclusion reason unless native scoring agrees with match metadata."""
    try:
        expected = {int(team): int(row['score']) for team, row in metadata['teamsData'].items()}
        if len(expected) != 2 or any(score < 0 for score in expected.values()):
            raise ValueError('Invalid team/score metadata')
        observed = Counter()
        for event in events:
            scorer = goal_team(event, expected)
            if scorer is not None:
                if scorer not in expected:
                    raise ValueError('Goal credited to unknown team')
                observed[scorer] += 1
        actual = {team: observed[team] for team in expected}
        if actual != expected:
            return {'reason': 'native goals disagree with teamsData score', 'observed': actual, 'expected': expected}
    except (KeyError, TypeError, ValueError) as exc:
        return {'reason': str(exc)}
    return None


def spells(events, teams):
    """Keep turnovers, but stop at goals, explicit stoppages, restarts and periods."""
    current, boundary = [], None
    for event in events:
        key = (event['matchId'], event['matchPeriod'])
        if key != boundary or event.get('eventId') == 3:
            if current:
                yield current
            current = []
        boundary = key
        current.append(event)
        if event.get('eventId') in STOPPAGES or goal_team(event, teams) is not None:
            yield current
            current = []
    if current:
        yield current


def usable_start(event):
    try:
        point = event['positions'][0]
        values = [float(point['x']), float(point['y']), float(event['eventSec'])]
        return (np.isfinite(values).all() and all(0 <= v <= 100 for v in values[:2])
                and 1 <= int(event['eventId']) <= 10)
    except (IndexError, KeyError, TypeError, ValueError):
        return False


def pass_states(events, teams, horizon=10):
    """Inputs use observed starts only; labels use current + next horizon-1 events.

    Native event counts include duels and keeper records. They are not SPADL
    action counts. Targets terminate at the first observed spell boundary.
    """
    if horizon <= 0:
        raise ValueError('horizon must be positive')
    pieces, targets, ids = [], [], []
    for spell in spells(events, teams):
        sanitized, selected = [], []
        for i, event in enumerate(spell):
            row = dict(event)
            # Reuse the existing start-only sequence encoder without allowing its
            # auxiliary completion/endpoint-label eligibility to select examples.
            row['eventName'] = 'Pass' if event.get('eventId') == 8 else 'Other'
            if event.get('eventId') == 8 and usable_start(event):
                row['positions'] = [event['positions'][0], event['positions'][0]]
                row['tags'] = [{'id': 1801}]
                selected.append(i)
            sanitized.append(row)
        encoded = pass_sequences(sanitized)
        if len(encoded[0]) != len(selected):
            raise ValueError('Native event IDs do not align with encoded pass states')
        if not selected:
            continue
        pieces.append(encoded[:2])
        for i in selected:
            scoring = {goal_team(e, teams) for e in spell[i:i + horizon]} - {None}
            team = spell[i]['teamId']
            targets.append([float(team in scoring), float(any(t != team for t in scoring))])
            ids.append(spell[i]['id'])
    return (
        np.concatenate([p[0] for p in pieces]) if pieces else np.zeros((0, 8), dtype=np.int64),
        np.concatenate([p[1] for p in pieces]) if pieces else np.zeros((0, 8, 4), dtype=np.float32),
        np.asarray(targets, dtype=np.float32).reshape(-1, 2),
        np.asarray(ids, dtype=np.int64),
    )


def new_xt_counts():
    return np.zeros((96, 96)), np.zeros(96), np.zeros(96), np.zeros(96)


def update_xt(counts, events):
    moves, attempts, shots, goals = counts
    for event in events:
        if not usable_start(event):
            continue
        tags = {tag.get('id') for tag in event.get('tags', [])}
        point = event['positions'][0]
        start = grid_index([float(point['x']) * 1.2, float(point['y']) * .8])
        shot = event.get('eventId') == 10 or (
            event.get('eventId') == 3 and event.get('subEventName') in {'Penalty', 'Free kick shot'}
        )
        if shot:
            shots[start] += 1
            goals[start] += 101 in tags and 102 not in tags
        elif event.get('eventId') == 8:
            attempts[start] += 1
            if 1801 in tags and 1802 not in tags and 102 not in tags:
                try:
                    end = event['positions'][1]
                    xy = [float(end['x']), float(end['y'])]
                    if np.isfinite(xy).all() and all(0 <= v <= 100 for v in xy):
                        moves[start, grid_index([xy[0] * 1.2, xy[1] * .8])] += 1
                except (IndexError, KeyError, TypeError, ValueError):
                    pass


def load_data(database, target_league, cap, horizon):
    if target_league not in LEAGUES or cap <= 0 or horizon <= 0:
        raise ValueError('Known league, positive cap and horizon required')
    counts = new_xt_counts()
    with sqlite3.connect(Path(database).resolve().as_uri() + '?mode=ro', uri=True) as db:
        db.execute('BEGIN')  # Hold one consistent read-only warehouse snapshot.
        records, metadata = [], {}
        digest = hashlib.sha256()
        for payload, in db.execute('SELECT payload FROM wyscout_entities WHERE entity_type=?', ('matches_' + target_league,)):
            match = json.loads(payload)
            if match.get('status') != 'Played' or match.get('duration') != 'Regular':
                continue
            mid = int(match['wyId'])
            if mid in metadata:
                raise ValueError('Duplicate native Wyscout match ID')
            metadata[mid] = match
            records.append(dict(id=mid, date=match['dateutc'][:10], league=target_league))
        groups = split_matches(records, target_league)
        data, receipt, excluded = {}, {}, []
        rng = np.random.default_rng(2026)
        for split in SPLITS:
            pieces, accepted = [], []
            for match in groups[split]:
                mid = match['id']
                digest.update(json.dumps(metadata[mid], sort_keys=True).encode())
                events = []
                for payload, in db.execute("SELECT payload FROM wyscout_events WHERE match_id=? AND period IN ('1H','2H') ORDER BY period,seconds,row_number", (mid,)):
                    digest.update(payload.encode())
                    events.append(json.loads(payload))
                problem = audit_goals(events, metadata[mid])
                if problem or not events:
                    excluded.append({'match_id': mid, 'split': split, **(problem or {'reason': 'No native events'})})
                    continue
                if split == 'train':
                    update_xt(counts, events)
                piece = pass_states(events, tuple(int(t) for t in metadata[mid]['teamsData']), horizon)
                if len(piece[0]):
                    pieces.append((*piece, np.full(len(piece[0]), mid, dtype=np.int64)))
                    accepted.append(match)
            if not pieces:
                raise ValueError(f'No validated pass states in {split}')
            arrays = tuple(np.concatenate([p[i] for p in pieces]) for i in range(5))
            chosen = np.sort(rng.choice(len(arrays[0]), min(cap, len(arrays[0])), replace=False))
            data[split] = tuple(a[chosen] for a in arrays)
            if len(np.unique(data[split][3])) != len(chosen):
                raise ValueError('Duplicate native event identifiers')
            receipt[split] = dict(matches_before_audit=len(groups[split]), matches_accepted=len(accepted),
                                  matches_used=len(np.unique(data[split][4])), sequences_available=len(arrays[0]),
                                  sequences_used=len(chosen), match_ids=[m['id'] for m in accepted],
                                  date_start=accepted[0]['date'], date_end=accepted[-1]['date'],
                                  positive_counts=data[split][2].sum(axis=0).astype(int).tolist(),
                                  selected_ids_sha256=hashlib.sha256(data[split][3].tobytes()).hexdigest())
        provenance = dict(provider='wyscout', season='2017/18', target_league=target_league,
                          splits=receipt, excluded_matches=excluded, inputs=source_signature(db, ('wyscout',)),
                          native_payloads_sha256=digest.hexdigest())
    xt = fit_expected_threat(*counts)
    xt.update(fit_split='train', fit_match_ids=receipt['train']['match_ids'],
              attempts=int(counts[1].sum()), shots=int(counts[2].sum()), goals=int(counts[3].sum()),
              interpretation='Training-only eventual attacking shot/move value; not calibrated to the fixed native-event horizon. Raw grid is descriptive only.')
    return data, provenance, xt


def binary_loss(labels, probabilities):
    p = np.clip(probabilities, 1e-7, 1 - 1e-7)
    return -(labels * np.log(p) + (1 - labels) * np.log1p(-p))


def select_setting(labels, candidates):
    return int(np.argmin([binary_loss(labels, p).mean() for _, p in candidates]))


def metrics(labels, prediction):
    result = {}
    for j, head in enumerate(HEADS):
        y, p = labels[:, j], prediction[:, j]
        bins = []
        edges = np.linspace(0, 1, 11)
        membership = np.minimum(np.searchsorted(edges, p, side='right') - 1, 9)
        for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
            mask = membership == index
            bins.append(dict(lower=float(lower), upper=float(upper), count=int(mask.sum()),
                             mean_probability=float(p[mask].mean()) if mask.any() else None,
                             observed_rate=float(y[mask].mean()) if mask.any() else None))
        result[head] = dict(log_loss=float(binary_loss(y, p).mean()), brier=float(np.mean((p - y) ** 2)),
                            observed_rate=float(y.mean()), predicted_rate=float(p.mean()), reliability=bins)
    result['mean_log_loss'] = float(binary_loss(labels, prediction).mean())
    return result


def xt_inputs(data, grid):
    xy = data[1][:, -1, :2]
    indices = np.minimum(7, (xy[:, 1] * 8).astype(int)) * 12 + np.minimum(11, (xy[:, 0] * 12).astype(int))
    return np.column_stack((np.asarray(grid).reshape(-1)[indices], xy))


def fit_classifiers(train_x, train_y, valid_x, valid_y, method, seed, path):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    models, settings, weights = [], [], {}
    for j, head in enumerate(HEADS):
        if len(np.unique(train_y[:, j])) != 2:
            raise ValueError(f'Training target {head} needs both classes')
        candidates, fitted = [], []
        choices = (3, 6) if method == 'tree_context' else ((.01, .1, 1.) if method == 'logistic_context' else (1.,))
        for setting in choices:
            model = (DecisionTreeClassifier(max_depth=setting, min_samples_leaf=60, random_state=seed)
                     if method == 'tree_context' else make_pipeline(StandardScaler(), LogisticRegression(C=setting, max_iter=1000, random_state=seed)))
            model.fit(train_x, train_y[:, j])
            candidates.append((setting, model.predict_proba(valid_x)[:, 1]))
            fitted.append(model)
        selected = select_setting(valid_y[:, j], candidates)
        model = fitted[selected]
        models.append(model)
        settings.append(dict(head=head, chosen=choices[selected], candidate_validation_log_loss=[float(binary_loss(valid_y[:, j], p).mean()) for _, p in candidates]))
        if method == 'tree_context':
            for key in ('children_left', 'children_right', 'feature', 'threshold', 'value'):
                weights[head + '_' + key] = getattr(model.tree_, key)
        else:
            scaler, classifier = model.steps[0][1], model.steps[1][1]
            for key, array in dict(mean=scaler.mean_, scale=scaler.scale_, coef=classifier.coef_, intercept=classifier.intercept_).items():
                weights[head + '_' + key] = array
    np.savez_compressed(path, **weights)
    return models, settings


def save_gru(model, path):
    np.savez_compressed(path, **{k: v.detach().cpu().numpy() for k, v in model.state_dict().items()})


def load_gru(path):
    from .representation.models import load_model
    return load_model(path)


def predict_gru(model, data):
    import torch
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(data[0][i:i + 2048]), torch.from_numpy(data[1][i:i + 2048])).sigmoid().numpy()
                               for i in range(0, len(data[0]), 2048)])


def fit_gru(train, validation, epochs, seed):
    import torch
    from .representation.models import PassModel, BATCH_SIZE, LEARNING_RATE
    torch.manual_seed(seed)
    model = PassModel()  # existing EventEncoder GRU and two logits; both use BCE here
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    tensors = tuple(torch.from_numpy(a) for a in train[:3])
    generator = torch.Generator().manual_seed(seed)
    best, state, history, stale = float('inf'), None, [], 0
    for epoch in range(epochs):
        model.train()
        total = 0.
        for indices in torch.randperm(len(train[0]), generator=generator).split(BATCH_SIZE):
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(tensors[0][indices], tensors[1][indices]), tensors[2][indices])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        score = float(binary_loss(validation[2], predict_gru(model, validation)).mean())
        history.append(dict(epoch=epoch + 1, training_loss=total / len(train[0]), validation_loss=score))
        if score < best:
            best, state, stale = score, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= 3:
            break
    model.load_state_dict(state)
    return model, dict(selected_epoch=min(history, key=lambda row: row['validation_loss'])['epoch'], patience=3, history=history)


def paired_bootstrap(labels, match_ids, first, second, replicates=1000):
    """Paired match bootstrap of per-event losses, averaging seeds before resampling."""
    difference = np.mean([binary_loss(labels, p) for p in first], axis=0) - np.mean([binary_loss(labels, p) for p in second], axis=0)
    ids = np.unique(match_ids)
    sums = np.array([difference[match_ids == mid].sum(axis=0) for mid in ids])
    counts = np.array([(match_ids == mid).sum() for mid in ids])
    chosen = np.random.default_rng(2026).integers(len(ids), size=(replicates, len(ids)))
    bootstrap = sums[chosen].sum(axis=1) / counts[chosen].sum(axis=1)[:, None]
    return {head: dict(log_loss_difference=float(difference[:, j].mean()), ci95=np.quantile(bootstrap[:, j], [.025, .975]).tolist()) for j, head in enumerate(HEADS)}


def run(database, output, *, target_league='England', train_cap=12000, epochs=12, seeds=(11, 29, 47), horizon=10):
    import torch
    import sklearn
    from threadpoolctl import threadpool_limits
    if epochs <= 0 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Positive epochs and unique nonempty seeds required')
    start = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    data, provenance, xt = load_data(database, target_league, train_cap, horizon)
    configuration = dict(target_league=target_league, train_cap=train_cap, cap_each_split=train_cap,
                         epochs=epochs, seeds=list(seeds), horizon=horizon, sampling_seed=2026,
                         threads=2, deterministic=True, torch_version=torch.__version__, numpy_version=np.__version__, sklearn_version=sklearn.__version__,
                         sequence_length=8, gru_hidden=32, batch_size=256, learning_rate=.003,
                         logistic_C=[.01, .1, 1.], tree_depth=[3, 6], tree_min_leaf=60,
                         early_stopping_patience=3, selection='validation mean binary log loss; separate classifier heads selected independently')
    sources = [Path(__file__), Path(__file__).parent / 'artifacts.py', Path(__file__).parent / 'tactics.py', Path(__file__).parent / 'representation' / 'data.py', Path(__file__).parent / 'representation' / 'models.py', Path(__file__).parent / 'representation' / 'experiment.py']
    code = {str(p.relative_to(Path(__file__).parent)): file_digest(p) for p in sources}
    identity = dict(configuration=configuration, data=provenance, code_sha256=code)
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()[:16]
    rows, predictions, selections, fit_seconds = [], {}, {}, {}
    produced_files = set()

    def artifact_path(name):
        produced_files.add(name)
        return output / name

    features = {split: linear_inputs(arrays) for split, arrays in data.items()}
    location_features = {split: xt_inputs(arrays, xt['grid']) for split, arrays in data.items()}
    prior = data['train'][2].mean(axis=0)
    np.savez_compressed(artifact_path('constant_train_prior.npz'), probability=prior)
    for split, arrays in data.items():
        np.savez_compressed(artifact_path(f'{split}-examples.npz'), event_id=arrays[3], match_id=arrays[4], labels=arrays[2], tokens=arrays[0], values=arrays[1])
    def record(name, seed, prediction):
        predictions.setdefault(name, []).append(prediction['test'])
        row = dict(method=name, seed=seed)
        for split in ('validation', 'test'):
            row[split] = metrics(data[split][2], prediction[split])
            np.savez_compressed(artifact_path(f'{name}-seed-{seed}-{split}-predictions.npz'), event_id=data[split][3], match_id=data[split][4], labels=data[split][2], probability=prediction[split], net_state=prediction[split][:, 0] - prediction[split][:, 1])
        rows.append(row)
    record('constant_train_prior', None, {s: np.tile(prior, (len(data[s][0]), 1)) for s in ('validation', 'test')})
    with threadpool_limits(limits=2):
        for seed in seeds:
            print(f'Native pre-pass value seed {seed}: validation selects classifiers and GRU checkpoint.', flush=True)
            for method in ('xt_location_calibrated', 'logistic_context', 'tree_context'):
                x = location_features if method == 'xt_location_calibrated' else features
                fit_start = time.monotonic()
                models, selected = fit_classifiers(x['train'], data['train'][2], x['validation'], data['validation'][2], method, seed, artifact_path(f'{method}-seed-{seed}.npz'))
                fit_seconds[f'{method}-{seed}'] = time.monotonic() - fit_start
                selections[f'{method}-{seed}'] = selected
                record(method, seed, {s: np.column_stack([m.predict_proba(x[s])[:, 1] for m in models]) for s in ('validation', 'test')})
            fit_start = time.monotonic()
            model, selected = fit_gru(data['train'], data['validation'], epochs, seed)
            fit_seconds[f'gru_context-{seed}'] = time.monotonic() - fit_start
            selections[f'gru_context-{seed}'] = selected
            save_gru(model, artifact_path(f'gru_context-seed-{seed}.npz'))
            record('gru_context', seed, {s: predict_gru(model, data[s]) for s in ('validation', 'test')})
    summaries = []
    for method in predictions:
        group = [row for row in rows if row['method'] == method]
        summary = dict(method=method, runs=len(group))
        for split in ('validation', 'test'):
            summary[split] = {head: {metric: dict(mean=float(np.mean(values := [r[split][head][metric] for r in group])), seed_sd=float(np.std(values, ddof=1)) if len(values) > 1 else 0.)
                                      for metric in ('log_loss', 'brier', 'observed_rate', 'predicted_rate')} for head in HEADS}
        summaries.append(summary)
    comparisons = {f'gru_context-minus-{method}': paired_bootstrap(data['test'][2], data['test'][4], predictions['gru_context'], values)
                   for method, values in predictions.items() if method != 'gru_context'}
    report = dict(run_id=run_id, observed_at=datetime.now(timezone.utc).isoformat(), **identity,
                  raw_xt=xt, runs=rows, summaries=summaries, validation_selection=selections,
                  compute=dict(fit_and_validation_selection_seconds=fit_seconds,
                               matched_elapsed_budgets=False,
                               timing_scope='Per-method fitting and validation selection; classifier timing also includes small numeric weight serialization. Excludes final heldout inference. Data loading/xT fitting are shared preprocessing.',
                               budgets_per_seed={'xt_location_calibrated': '2 binary logistic fits, C=1, max_iter=1000',
                                                 'logistic_context': '6 binary logistic fits: 3 C choices per head, max_iter=1000',
                                                 'tree_context': '4 trees: 2 depths per head, minimum leaf size 60',
                                                 'gru_context': f'1 shared encoder, 2 binary heads, maximum {epochs} epochs, patience 3'},
                               fairness='Identical sampled native states/targets and CPU thread cap. Logistic/tree/GRU share all input information. xT-location uses current location plus xT learned from ALL accepted training-match passes/shots, followed by calibration on the same sampled states. This hybrid baseline has extra training-event exposure. Search, parameter and elapsed compute budgets are not equal.'),
                  paired_match_bootstrap=dict(replicates=1000, seed=2026, estimand='Per-event mean log loss difference; seed losses averaged, test matches resampled jointly. Negative favors GRU. Conditional on fixed trained models and sampled examples.', comparisons=comparisons),
                  input_fields=['eight observed native event types', 'event starts in current-team frame', 'same-team indicators', 'within-period elapsed gaps clipped at 120s'],
                  targets=['same team scores', 'opponent scores'],
                  semantics=dict(horizon='Current pass plus next horizon-1 native events, truncated at spell boundary; not seconds or SPADL actions.',
                                 spell='Turnovers retained; terminate after goal/foul/interruption/offside, before free-kick restart, or at match/period boundary. No inferred time-gap or possession reset.',
                                 eligibility='Native eventId 8 pass with valid observed start/time; endpoint and completion do not determine inclusion.',
                                 state='P(score)-P(concede) before pass outcome; not realized incremental VAEP or offensive action gain.',
                                 baseline='Training-fitted logistic map of raw training-only xT plus current x/y to each shared fixed-horizon target; empirical calibration is assessed on heldout matches, not assumed.',
                                 goals='Attacking shot/free-kick-shot/penalty tag101, any non-keeper own-goal102 credited to opponent; keeper event9 duplicates ignored; metadata mismatch excludes whole match.'),
                  limits=['One historical league-season; no cross-season or production validation.',
                          'Native Wyscout event counts, duel duplication and stoppage conventions differ from SPADL and published VAEP.',
                          'Observed spell boundaries shorten the target horizon; missing goal metadata inconsistencies are excluded, not repaired.',
                          'Inputs use recorded event-start type/location and history, not tracking, pressure, intended outcomes, or unobserved alternatives.',
                          'Rare conceding labels limit reliability estimates; predictive association is not causal action value.',
                          'Methods use documented small search budgets, not matched elapsed training compute; accuracy alone is not an efficiency comparison.',
                          'The xT-location hybrid fits its grid on every accepted training match, then its logistic map on the capped training states. Its extra training-event exposure prevents architecture-only comparisons with capped-state classifiers.',
                          'Match bootstrap omits training-set resampling and hyperparameter-selection uncertainty; seed SD is not a confidence interval.',
                          'Raw xT uses only native pass transitions and attacking shots; own goals terminate moves and are not attacking-shot successes.'],
                  elapsed_seconds=time.monotonic() - start)
    write_json_atomic(artifact_path('provenance.json'), provenance)
    write_json_atomic(artifact_path('report.json'), report)
    write_json_atomic(output / 'artifact-manifest.json', {'run_id': run_id, 'files': {name: file_digest(output / name) for name in sorted(produced_files)}})
    return report
