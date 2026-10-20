"""Teaching control surfaces and supervised clock-offset validation on raw tracking.

Finite supplied Metrica coordinates are usable geometry, not detection evidence.
The Gaussian surface is an influence share; the arrival surface is an uncalibrated
competing-hazard model. Neither estimates validated real reception outcomes.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path

import numpy as np

from app.analytics.tracking import metrica_csv_frames, skillcorner_frames


DEFAULT_PARAMETERS = {
    'reaction_seconds': .7, 'player_speed_mps': 5., 'ball_speed_mps': 15.,
    'arrival_sigma_seconds': .45, 'control_rate_per_second': 4.3,
    'integration_dt_seconds': .04, 'max_integration_seconds': 10.,
    'gaussian_radius_m': 6., 'gaussian_speed_scale_mps': 12.,
}


def _finite(value):
    return isinstance(value, (float, int, np.number)) and not isinstance(value, bool) and math.isfinite(value)


def _xy(point):
    return bool(point) and _finite(point.get('x')) and _finite(point.get('y'))


def complete_frame_quality(frame, *, require_detection=False):
    """Require two complete 11-player teams; report outside positions without clipping."""
    reasons = []
    players = frame.get('players', [])
    valid = [p for p in players if _xy(p)]
    counts = Counter(str(p['team']) for p in valid)
    if len(counts) != 2 or any(n != 11 for n in counts.values()):
        reasons.append('incomplete_teams')
    if len({(str(p['team']), str(p['id'])) for p in valid}) != len(valid):
        reasons.append('duplicate_players')
    ball = frame.get('ball')
    if not _xy(ball):
        reasons.append('missing_ball')
    if frame.get('period') is None or not _finite(frame.get('time')):
        reasons.append('invalid_clock')
    if require_detection:
        if any(p.get('supplier_detected') is not True for p in valid):
            reasons.append('players_not_supplier_detected')
        if not ball or ball.get('supplier_detected') is not True:
            reasons.append('ball_not_supplier_detected')
    length, width = frame.get('pitch_length'), frame.get('pitch_width')
    dimensions_valid = _finite(length) and _finite(width) and length > 0 and width > 0
    if not dimensions_valid:
        reasons.append('invalid_pitch_dimensions')
    outside = lambda p: bool(dimensions_valid and _xy(p) and (abs(p['x']) > length / 2 or abs(p['y']) > width / 2))
    return {
        'accepted': not reasons, 'reasons': reasons, 'finite_players': len(valid),
        'finite_players_by_team': dict(counts), 'ball_finite': bool(_xy(ball)),
        'supplier_detected_players': sum(p.get('supplier_detected') is True for p in valid),
        'unknown_detection_players': sum(p.get('supplier_detected') is None for p in valid),
        'outside_player_count': sum(outside(p) for p in valid),
        'ball_outside_pitch': outside(ball),
    }


class CausalVelocity:
    """Backward differences only. Gaps, missing records, periods and spikes reset state."""
    def __init__(self, maximum_gap_seconds=.25, maximum_speed_mps=12.):
        self.maximum_gap_seconds = maximum_gap_seconds
        self.maximum_speed_mps = maximum_speed_mps
        self.previous = {}

    def update(self, frame):
        current = {}
        identity = (frame.get('provider'), frame.get('match_id'), frame.get('period'))
        timestamp = frame.get('time')
        players = []
        for point in frame.get('players', []):
            p = dict(point, vx=None, vy=None, velocity_quality='missing_position')
            key = (str(p['team']), str(p['id']))
            valid = _xy(p) and _finite(timestamp)
            if frame.get('provider') == 'skillcorner' and p.get('supplier_detected') is not True:
                valid = False
                p['velocity_quality'] = 'not_supplier_detected'
            if valid:
                prior = self.previous.get(key)
                p['velocity_quality'] = 'no_causal_predecessor'
                current[key] = (identity, timestamp, p['x'], p['y'])
                if prior is not None:
                    dt = timestamp - prior[1]
                    if prior[0] != identity or not 0 < dt <= self.maximum_gap_seconds:
                        p['velocity_quality'] = 'period_or_clock_gap'
                    else:
                        vx, vy = (p['x'] - prior[2]) / dt, (p['y'] - prior[3]) / dt
                        if math.hypot(vx, vy) <= self.maximum_speed_mps:
                            p.update(vx=vx, vy=vy, velocity_quality='valid')
                        else:
                            p['velocity_quality'] = 'speed_spike'
                            current.pop(key, None)
            players.append(p)
        self.previous = current  # absent players must not bridge a missing frame
        return {**frame, 'players': players}


def control_surfaces(frame, targets, **overrides):
    """Return team shares (team, target) and explicit unassigned arrival mass.

    Arrival time is reaction + distance from projected reaction position / speed.
    Each player's hazard is control_rate * logistic(arrival CDF). At ball arrival,
    integration starts and conservatively allocates exact exponential survival
    loss under the midpoint piecewise-constant total hazard. Teams are symmetric.
    """
    params = {**DEFAULT_PARAMETERS, **overrides}
    if set(overrides) - set(DEFAULT_PARAMETERS):
        raise ValueError('Unknown model parameters')
    if any(not _finite(v) or v <= 0 for v in params.values()):
        raise ValueError('Model parameters must be finite and positive')
    quality = complete_frame_quality(frame, require_detection=frame.get('provider') == 'skillcorner')
    if not quality['accepted']:
        raise ValueError('Incomplete frame: ' + ', '.join(quality['reasons']))
    players = [p for p in frame['players'] if _xy(p)]
    if any(not _finite(p.get('vx')) or not _finite(p.get('vy')) for p in players):
        raise ValueError('Causal finite velocities required; no zero imputation')
    targets = np.asarray(targets, dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 2 or not np.isfinite(targets).all():
        raise ValueError('Targets must be finite Nx2 coordinates')
    teams = sorted({str(p['team']) for p in players})
    membership = np.array([str(p['team']) == teams[0] for p in players])
    position = np.array([[p['x'], p['y']] for p in players])
    velocity = np.array([[p['vx'], p['vy']] for p in players])
    displacement = targets[None, :, :] - position[:, None, :]
    distance = np.linalg.norm(displacement, axis=2)
    nearest_distance = np.stack([distance[membership].min(axis=0), distance[~membership].min(axis=0)])
    ties = np.isclose(nearest_distance[0], nearest_distance[1], rtol=0, atol=1e-10)
    first = np.where(ties, .5, (nearest_distance[0] < nearest_distance[1]).astype(float))
    nearest = np.stack([first, 1 - first])

    # Positive anisotropic widths, aligned with velocity, shift the center forward.
    speed = np.linalg.norm(velocity, axis=1)
    direction = np.divide(velocity, speed[:, None], out=np.zeros_like(velocity), where=speed[:, None] > 0)
    direction[speed == 0] = [1., 0.]
    shifted = displacement - velocity[:, None, :] * params['reaction_seconds']
    longitudinal = np.sum(shifted * direction[:, None, :], axis=2)
    lateral = shifted[:, :, 0] * direction[:, None, 1] - shifted[:, :, 1] * direction[:, None, 0]
    anisotropy = speed / params['gaussian_speed_scale_mps']
    along = params['gaussian_radius_m'] * (1 + anisotropy)
    across = params['gaussian_radius_m'] / (1 + anisotropy)
    log_weight = -.5 * ((longitudinal / along[:, None])**2 + (lateral / across[:, None])**2)
    weights = np.exp(log_weight - log_weight.max(axis=0))
    first = weights[membership].sum(axis=0) / weights.sum(axis=0)
    gaussian = np.stack([first, 1 - first])

    arrival = params['reaction_seconds'] + np.linalg.norm(shifted, axis=2) / params['player_speed_mps']
    ball = np.array([frame['ball']['x'], frame['ball']['y']])
    ball_time = np.linalg.norm(targets - ball, axis=1) / params['ball_speed_mps']
    # Extended precision accumulation prevents float64 sums exceeding unity by
    # roundoff; final arrays use float64 without clipping or renormalization.
    remaining = np.ones(len(targets), dtype=np.longdouble)
    probability = np.zeros((2, len(targets)), dtype=np.longdouble)
    dt = params['integration_dt_seconds']
    horizon = params['max_integration_seconds']
    elapsed = 0.
    steps = 0
    scale = math.pi / (math.sqrt(3) * params['arrival_sigma_seconds'])
    while elapsed < horizon - 1e-12:
        step = min(dt, horizon - elapsed)
        logits = scale * (ball_time[None, :] + elapsed + step / 2 - arrival)
        # Bound exponent evaluation only; positions and probabilities are not clipped.
        cumulative = 1 / (1 + np.exp(-np.clip(logits, -700, 700)))
        hazard = params['control_rate_per_second'] * cumulative
        team_hazard = np.stack([hazard[membership].sum(axis=0), hazard[~membership].sum(axis=0)])
        total = team_hazard.sum(axis=0)
        consumed = remaining * -np.expm1(-total * step)
        probability += np.divide(team_hazard, total, out=np.zeros_like(team_hazard), where=total > 0) * consumed
        remaining -= consumed
        elapsed += step
        steps += 1
        if np.max(remaining, initial=0) < 1e-10:
            break
    return {'teams': teams, 'nearest': nearest, 'gaussian': gaussian, 'arrival': np.asarray(probability, dtype=float),
            'uncontrolled': np.asarray(remaining, dtype=float), 'integration_steps': steps}


def _nearest_indices(times, query):
    right = np.searchsorted(times, query)
    right = np.minimum(right, len(times) - 1)
    left = np.maximum(right - 1, 0)
    return np.where(abs(times[left] - query) <= abs(times[right] - query) + 1e-12, left, right)


def synchronization_experiment(events, clocks, *, offsets=(0., -.5, .5, -1., 1., -2., 2.)):
    """Supervised early native-frame clock anchors; later events are held out.

    This validates synthetic constant offsets using trusted native Start Frame IDs.
    It is not blind spatial alignment or a validation of heterogeneous clock drift.
    Event coordinates are retained for independent spatial residual diagnostics.
    """
    excluded = Counter()
    cases = []
    private = []
    for period, clock in sorted(clocks.items()):
        times, ids = np.asarray(clock['times']), np.asarray(clock['frames'])
        if len(times) == 0 or np.any(np.diff(times) <= 0):
            raise ValueError('Tracking clocks must strictly increase within each period')
        lookup = {int(fid): i for i, fid in enumerate(ids)}
        usable = []
        for event in events:
            if event['period'] != period:
                continue
            idx = lookup.get(event['frame'])
            if idx is None:
                excluded['missing_truth_frame'] += 1
            elif not _finite(event.get('time')):
                excluded['invalid_event_time'] += 1
            else:
                usable.append({**event, 'truth_time': float(times[idx]), 'truth_index': idx})
        usable.sort(key=lambda e: (e['time'], e['event_id']))
        unique_times = sorted({e['time'] for e in usable})
        if len(unique_times) < 3:
            excluded['insufficient_anchor_periods'] += 1
            continue
        cutoff = unique_times[max(1, int(len(unique_times) * .2)) - 1]
        training = [e for e in usable if e['time'] <= cutoff]
        evaluation = [e for e in usable if e['time'] > cutoff]
        truth_ids = np.array([e['frame'] for e in evaluation])
        truth_times = np.array([e['truth_time'] for e in evaluation])
        queries = np.array([e['time'] for e in evaluation])
        for offset in offsets:
            estimate = float(np.median([e['time'] + offset - e['truth_time'] for e in training]))
            stats = {}
            indices = {}
            for label, correction in (('before', 0.), ('after', estimate)):
                query = queries + offset - correction
                matched = _nearest_indices(times, query)
                indices[label] = matched
                frame_error = abs(ids[matched] - truth_ids)
                seconds_error = abs(times[matched] - truth_times)
                stats[label] = {
                    'count': len(evaluation), 'exact_matches': int(np.sum(frame_error == 0)),
                    'exact_match_rate': float(np.mean(frame_error == 0)),
                    'mean_absolute_frame_error': float(np.mean(frame_error)),
                    'median_absolute_frame_error': float(np.median(frame_error)),
                    'maximum_absolute_frame_error': int(np.max(frame_error)),
                    'mean_absolute_seconds_error': float(np.mean(seconds_error)),
                    'median_absolute_seconds_error': float(np.median(seconds_error)),
                    'maximum_absolute_seconds_error': float(np.max(seconds_error)),
                    'queries_outside_period_clock': int(np.sum((query < times[0]) | (query > times[-1]))),
                }
            cases.append({'period': period, 'injected_offset_seconds': offset,
                          'estimated_offset_seconds': estimate, 'training_count': len(training),
                          'last_training_time': float(max(e['time'] for e in training)),
                          'first_evaluation_time': float(min(e['time'] for e in evaluation)), **stats})
            private.append({'period': period, 'injected_offset_seconds': offset,
                            'training_event_ids': [e['event_id'] for e in training],
                            'evaluation': [dict(event_id=e['event_id'], truth_frame=e['frame'],
                                                before_frame=int(ids[indices['before'][i]]),
                                                after_frame=int(ids[indices['after'][i]]))
                                           for i, e in enumerate(evaluation)]})
    known_periods = set(clocks)
    excluded['missing_tracking_period'] = sum(e['period'] not in known_periods for e in events)
    return {'method': 'Supervised median constant-clock calibration from early native Start Frame anchors, per period',
            'split': 'First 20% of unique event timestamps for anchors; strictly later timestamps held out',
            'truth': 'Native Metrica event Start Frame; never replaced by a nearest timestamp match',
            'claim': 'Synthetic constant offset recovery only; not blind Needleman-Wunsch or heterogeneous drift validation',
            'cases': cases, 'excluded': dict(excluded), 'private_assignments': private}


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path, data):
    Path(path).write_text(json.dumps(_json_safe(data), indent=2, allow_nan=False) + '\n')


def _events(path, match_id):
    with Path(path).open(newline='') as handle:
        return [dict(event_id=f'metrica:{match_id}:{i}', period=int(r['Period']),
                     frame=int(r['Start Frame']), time=float(r['Start Time [s]']),
                     x=(float(r['Start X']) - .5) * 105,
                     y=(float(r['Start Y']) - .5) * 68, type=r['Type'])
                for i, r in enumerate(csv.DictReader(handle))]


def _scan_match(home, away, match_id, events):
    coverage = Counter()
    period_counts = Counter()
    clocks = defaultdict(lambda: {'times': [], 'frames': []})
    anchors = {(e['period'], e['frame']) for e in events}
    ball_at_anchor = {}
    player_counts = Counter()
    velocity = CausalVelocity()
    for raw in metrica_csv_frames(home, away, match_id=match_id):
        frame = velocity.update(raw)
        quality = complete_frame_quality(frame)
        coverage['frames'] += 1
        coverage['ball_finite_frames'] += quality['ball_finite']
        coverage['complete_frames'] += quality['accepted']
        coverage['model_eligible_frames'] += quality['accepted'] and all(
            p['velocity_quality'] == 'valid' for p in frame['players'] if _xy(p))
        coverage['finite_player_records'] += quality['finite_players']
        coverage['unknown_supplier_detection_player_records'] += quality['unknown_detection_players']
        coverage['outside_player_records'] += quality['outside_player_count']
        coverage['outside_ball_frames'] += quality['ball_outside_pitch']
        coverage['frames_22_finite_players'] += quality['finite_players'] == 22
        for reason in quality['reasons']:
            coverage['rejected_' + reason] += 1
        period_counts[frame['period']] += 1
        player_counts[quality['finite_players']] += 1
        clock = clocks[frame['period']]
        clock['times'].append(frame['time'])
        clock['frames'].append(frame['frame'])
        if (frame['period'], frame['frame']) in anchors:
            ball_at_anchor[(frame['period'], frame['frame'])] = frame['ball']
    for clock in clocks.values():
        clock['times'] = np.array(clock['times'])
        clock['frames'] = np.array(clock['frames'])
    residuals = [math.hypot(e['x'] - b['x'], e['y'] - b['y'])
                 for e in events if _xy(e) and _xy(b := ball_at_anchor.get((e['period'], e['frame'])))]
    return dict(coverage, period_frames=dict(period_counts), finite_player_count_histogram=dict(player_counts),
                anchor_position_diagnostic={'count': len(residuals),
                                            'median_event_ball_distance_m': float(np.median(residuals)) if residuals else None,
                                            'note': 'Spatial residual at native frame IDs only; coordinates do not train the supervised clock fit'}), clocks


def _sample_match(home, away, match_id, count, eligible_count):
    """Systematically sample equally spaced ranks of whole-match eligible frames."""
    requested = count
    count = min(count, eligible_count)
    ranks = {int((i + .5) * eligible_count / count) for i in range(count)} if count else set()
    frames = []
    velocity = CausalVelocity()
    rejection = Counter()
    velocity_quality = Counter()
    eligible_rank = 0
    for ordinal, raw in enumerate(metrica_csv_frames(home, away, match_id=match_id)):
        frame = velocity.update(raw)
        quality = complete_frame_quality(frame)
        velocity_quality.update(p['velocity_quality'] for p in frame['players'] if _xy(p))
        if not quality['accepted']:
            rejection.update(quality['reasons'])
            continue
        if any(p['velocity_quality'] != 'valid' for p in frame['players'] if _xy(p)):
            rejection['unknown_causal_velocity'] += 1
            continue
        if eligible_rank in ranks:
            frames.append({**frame, 'quality': quality, 'eligible_rank': eligible_rank,
                           'source_ordinal': ordinal})
        eligible_rank += 1
    if eligible_rank != eligible_count:
        raise ValueError('Eligible frame count changed between streaming passes')
    return frames, {'requested': requested, 'selected': len(frames),
                    'eligible_frames': eligible_count, 'insufficient_eligible_frames': count < requested,
                    'rejected_frame_reasons': dict(rejection), 'finite_player_velocity_quality': dict(velocity_quality),
                    'sampling': 'Systematic equally spaced ranks across the complete whole-match eligible-frame sequence; no prefix truncation',
                    'bias': 'Uniform over eligible frames, not uniform wall-clock time; missing ball/velocity intervals have no samples',
                    'first_selected_time': frames[0]['time'] if frames else None,
                    'last_selected_time': frames[-1]['time'] if frames else None,
                    'selected_period_counts': dict(Counter(f['period'] for f in frames))}


def _skillcorner_prefix(base):
    metadata_path = base / '1874553_match.json'
    tracking_path = base / '1874553_tracking_extrapolated.jsonl'
    availability = {'metadata': metadata_path.exists(), 'tracking': tracking_path.exists()}
    if not all(availability.values()):
        return {'status': 'data_unavailable', 'scope': 'first 1000 valid-clock frames only',
                'availability': availability,
                'metadata_sha256': _hash(metadata_path) if availability['metadata'] else None}
    metadata = json.loads(metadata_path.read_text())
    counts = Counter()
    first = last = None
    for frame in islice(skillcorner_frames(tracking_path, metadata), 1000):
        quality = complete_frame_quality(frame, require_detection=True)
        counts['frames'] += 1
        counts['accepted'] += quality['accepted']
        counts['ball_supplier_detected'] += frame['ball']['supplier_detected'] is True
        counts['players_ge20_supplier_detected'] += quality['supplier_detected_players'] >= 20
        counts['players_ge22_supplier_detected'] += quality['supplier_detected_players'] >= 22
        for reason in quality['reasons']:
            counts['rejected_' + reason] += 1
        if first is None:
            first = frame['frame']
        last = frame['frame']
    prefix_digest = hashlib.sha256()
    prefix_lines = 0
    with tracking_path.open('rb') as handle:
        for line in handle:
            prefix_digest.update(line)
            prefix_lines += 1
            if json.loads(line).get('frame') == last:
                break
    return {'status': 'complete', 'provider': 'skillcorner', 'match_id': metadata['id'],
            'availability': availability,
            'scope': 'first 1000 valid-clock frames only; not whole-match or provider-wide coverage',
            'first_frame': first, 'last_frame': last, 'counts': dict(counts),
            'metadata_sha256': _hash(metadata_path),
            'tracking_identity': {'path': str(tracking_path), 'size_bytes': tracking_path.stat().st_size,
                                  'hash_scope': 'raw bytes from file start through last inspected native frame, inclusive',
                                  'prefix_raw_lines': prefix_lines, 'prefix_sha256': prefix_digest.hexdigest()},
            'interpretation': 'Partial detected broadcast coverage rejected; extrapolated coordinates never treated as detections'}


def _run_id(sources, code, config, skillcorner_prefix):
    """Bind every audited input, including optional prefix availability, without paths."""
    prefix_identity = dict(skillcorner_prefix)
    if 'tracking_identity' in prefix_identity:
        prefix_identity['tracking_identity'] = {
            key: value for key, value in prefix_identity['tracking_identity'].items() if key != 'path'
        }
    identity = {'sources': sources,
                'code': [{**row, 'file': Path(row['file']).name} for row in code],
                'config': config, 'skillcorner_prefix': prefix_identity}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()[:24]


def run(data_dir='.tmp/open-data/metrica/data', output='.tmp/analysis-upgrades-20261014/space/run', samples=240):
    """Run two real games, without database writes, and return a strict-JSON report.

    Timer covers source hashing, both streaming passes, model evaluation, clock
    validation and numeric/context writes; excludes final report serialization.
    """
    started = time.perf_counter()
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 2:
        raise ValueError('samples must be an integer >= 2')
    data_dir, output = Path(data_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sources = []
    paths = {}
    for game in (1, 2):
        stem = f'Sample_Game_{game}'
        folder = data_dir / stem
        paths[game] = tuple(folder / f'{stem}_{name}.csv' for name in (
            'RawTrackingData_Home_Team', 'RawTrackingData_Away_Team', 'RawEventsData'))
        for path in paths[game]:
            sources.append({'provider': 'metrica', 'match_id': game, 'file': path.name,
                            'size_bytes': path.stat().st_size, 'sha256': _hash(path)})
    code = [{'file': str(p), 'sha256': _hash(p)} for p in (Path(__file__), Path(__file__).with_name('tracking.py'))]
    config = {'samples_total': samples, 'games': [1, 2], 'grid': {'x_cells': 20, 'y_cells': 13},
              'parameters': DEFAULT_PARAMETERS, 'velocity': {'max_gap_seconds': .25, 'max_speed_mps': 12., 'method': 'causal backward difference'},
              'sensitivity': {'reaction_1s': {'reaction_seconds': 1.}, 'player_speed_7': {'player_speed_mps': 7.},
                              'ball_speed_8': {'ball_speed_mps': 8.}},
              'nominal_pitch_m': [105., 68.], 'sampling': 'systematic uniformly spaced ranks over whole-match eligible frames'}
    prefix = _skillcorner_prefix(data_dir.parent.parent / 'skillcorner/data/matches/1874553')
    run_id = _run_id(sources, code, config, prefix)
    selected = []
    games = []
    assignments = []
    for game in (1, 2):
        home, away, event_path = paths[game]
        events = _events(event_path, game)
        coverage, clocks = _scan_match(home, away, game, events)
        frames, sampling = _sample_match(home, away, game, samples // 2 + (samples % 2 if game == 1 else 0), coverage['model_eligible_frames'])
        selected.extend(frames)
        sync = synchronization_experiment(events, clocks)
        assignments.append({'provider': 'metrica', 'match_id': game, 'cases': sync.pop('private_assignments')})
        games.append({'provider': 'metrica', 'match_id': game, 'coverage': coverage, 'sampling': sampling,
                      'events': len(events), 'synchronization': sync})
    x = (np.arange(20) + .5) * 105 / 20 - 105 / 2
    y = (np.arange(13) + .5) * 68 / 13 - 68 / 2
    gx, gy = np.meshgrid(x, y)
    targets = np.column_stack((gx.ravel(), gy.ravel()))
    surfaces = defaultdict(list)
    measurements = []
    sensitivity = defaultdict(list)
    for frame in selected:
        result = control_surfaces(frame, targets)
        for name in ('nearest', 'gaussian', 'arrival', 'uncontrolled'):
            surfaces[name].append(result[name])
        pairwise = {}
        for left, right in (('nearest', 'gaussian'), ('nearest', 'arrival'), ('gaussian', 'arrival')):
            pairwise[f'{left}_vs_{right}'] = {'mean_absolute_team_share_difference': float(np.mean(abs(result[left] - result[right]))),
                                            'dominant_team_disagreement_fraction': float(np.mean(np.argmax(result[left], axis=0) != np.argmax(result[right], axis=0)))}
        measurements.append({'provider': frame['provider'], 'match_id': frame['match_id'], 'frame': frame['frame'],
                             'period': frame['period'], 'time': frame['time'], 'differences': pairwise,
                             'max_uncontrolled_mass': float(result['uncontrolled'].max()),
                             'max_mass_conservation_error': float(np.max(abs(result['arrival'].sum(axis=0) + result['uncontrolled'] - 1))),
                             'integration_steps': result['integration_steps']})
        for name, overrides in config['sensitivity'].items():
            changed = control_surfaces(frame, targets, **overrides)
            sensitivity[name].append(float(np.mean(abs(changed['arrival'] - result['arrival']))))
    if not selected:
        raise ValueError('No complete frames with causal velocities available')
    np.savez_compressed(output / 'surfaces.npz', targets=targets,
                        match_ids=np.array([f['match_id'] for f in selected]),
                        frame_ids=np.array([f['frame'] for f in selected]),
                        periods=np.array([f['period'] for f in selected]),
                        times=np.array([f['time'] for f in selected]),
                        **{k: np.stack(v) for k, v in surfaces.items()})
    with (output / 'frame-context.jsonl').open('w') as handle:
        for frame in selected:
            handle.write(json.dumps(_json_safe(frame), allow_nan=False) + '\n')
    _write_json(output / 'clock-assignments.json', assignments)
    _write_json(output / 'frame-measurements.json', measurements)
    method_differences = {}
    for pair in measurements[0]['differences']:
        method_differences[pair] = {metric: float(np.mean([m['differences'][pair][metric] for m in measurements]))
                                    for metric in measurements[0]['differences'][pair]}
    method_sources = Path('.tmp/project-comparison-20261014/open-source/upstream-source-identities.json')
    references = []
    if method_sources.exists():
        for project in json.loads(method_sources.read_text())['projects']:
            if 'LaurieOnTracking' in project['repository'] or 'databallpy' in project['repository']:
                references.append(project)
    _write_json(output / 'provenance.json', {'run_id': run_id, 'sources': sources, 'code': code,
                                           'skillcorner_prefix': prefix,
                                           'method_references': references,
                                           'method_usage': 'Independent mathematical implementation; reference scripts read statically, never executed or imported'})
    report = {
        'schema_version': 1, 'run_id': run_id, 'status': 'complete' if len(selected) == samples else 'insufficient_eligible_frames', 'config': config,
        'sampled_frames': len(selected), 'games': games, 'method_differences': method_differences,
        'sensitivity': {name: {'samples': len(v), 'mean_absolute_arrival_share_change': float(np.mean(v)),
                               'maximum_frame_mean_absolute_change': float(np.max(v)),
                               'minimum_frame_mean_absolute_change': float(np.min(v))} for name, v in sensitivity.items()},
        'numerical_checks': {'max_conservation_error': max(m['max_mass_conservation_error'] for m in measurements),
                             'max_uncontrolled_mass': max(m['max_uncontrolled_mass'] for m in measurements),
                             'min_arrival_share': float(np.min(surfaces['arrival'])),
                             'max_arrival_share': float(np.max(surfaces['arrival']))},
        'skillcorner_prefix': prefix,
        'method': {'nearest': 'Nearest team distance, ties assigned equally',
                   'gaussian': 'Velocity-directed shifted anisotropic Gaussian influence, normalized team shares',
                   'arrival': 'Reaction projection + constant-speed arrival, logistic timing uncertainty, symmetric competing control hazards after ball travel',
                   'integration': 'Midpoint hazards with exact exponential survival loss, extended precision accumulation, explicit remaining mass; no probability clipping or renormalization',
                   'team_order': sorted({str(p['team']) for p in selected[0]['players'] if _xy(p)}),
                   'array_layout': 'nearest/gaussian/arrival: snapshot, team, flattened target; uncontrolled: snapshot,target; identity arrays and JSONL use same snapshot order',
                   'coordinates': selected[0]['coordinate_system'],
                   'attacking_direction': None, 'offside_filter': 'not applied because attacking orientation is unknown'},
        'limitations': ['Teaching parameters are uncalibrated; no validated real reception prediction or predictive score.',
                       'Gaussian influence, arrival control and nearest distance are different assumptions, not ground truth.',
                       'Metrica supplier detection/extrapolation provenance is unknown; complete finite supplied geometry only.',
                       'Metrica CSV pitch dimensions 105x68m are nominal assumptions, not measured metadata.',
                       'Causal velocities exclude missing history, clock gaps and speeds above 12m/s; no interpolation or centered filtering.',
                       'Two-pass streamed full-match coverage, 240 requested snapshots by default; eligible-only sampling may favor clean tracking.',
                       'Outside-pitch coordinates are retained and flagged; no offside, interception along pass path, aerial ball or player-specific calibration.',
                       'Clock experiment uses supervised early native-frame anchors with synthetic constant offsets, not blind alignment or clock drift validation.',
                       'SkillCorner inspection is only its first 1000 valid-clock frames; no whole-match/provider coverage inference.'],
        'artifacts': {'provenance': 'provenance.json', 'numeric_surfaces': 'surfaces.npz',
                      'private_frames': 'frame-context.jsonl', 'private_clock_assignments': 'clock-assignments.json',
                      'private_frame_measurements': 'frame-measurements.json'},
        'runtime_seconds': time.perf_counter() - started,
        'runtime_scope': 'Source hashing, both full streaming passes, models, sensitivity, synchronization, prefix audit and evidence writes; excludes final report serialization',
    }
    _write_json(output / 'report.json', report)
    return _json_safe(report)
