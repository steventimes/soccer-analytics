"""Audit a frozen native-Wyscout training xT grid without refitting predictors."""

import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

from app.file_io import write_json_atomic
from . import action_value as av
from .artifacts import file_digest, source_signature
from .tactics import fit_expected_threat


def resample_grids(match_counts, replicates=1000, seed=20261015):
    """Resample whole matches and refit pooled counts, without dropping failures.

    Percentiles describe sensitivity within the observed match collection, not
    simultaneous coverage, future-season accuracy or the model's true error.
    """
    if type(replicates) is not int or replicates < 200 or type(seed) is not int or seed < 0:
        raise ValueError('Invalid xT resampling replicates or seed')
    if len(match_counts) < 2:
        raise ValueError('xT resampling requires at least two matches')
    for counts in match_counts:
        fit_expected_threat(*counts)
    parts = [np.stack([counts[i] for counts in match_counts]) for i in range(4)]
    reference = np.asarray(fit_expected_threat(*(p.sum(axis=0) for p in parts))['grid'])
    draws = np.random.default_rng(seed).integers(0, len(match_counts), size=(replicates, len(match_counts)), dtype=np.int64)
    grids, unobserved, zeros = [], np.zeros(96, dtype=int), np.zeros(96, dtype=int)
    for index, selected in enumerate(draws):
        weights = np.bincount(selected, minlength=len(match_counts))
        pooled = [np.tensordot(weights, part, axes=(0, 0)) for part in parts]
        try:
            grid = np.asarray(fit_expected_threat(*pooled)['grid'])
        except ValueError as error:
            raise ValueError(f'xT resampling draw {index} failed; no failed draws omitted') from error
        grids.append(grid)
        unobserved += pooled[1] + pooled[2] == 0
        zeros += grid.ravel() == 0
    low, median, high = np.quantile(grids, [.025, .5, .975], axis=0, method='linear')
    deviation = np.max(np.abs(np.asarray(grids) - reference), axis=(1, 2))
    support = {'observations': (parts[1] + parts[2] > 0).sum(axis=0),
               'shots': (parts[2] > 0).sum(axis=0), 'goals': (parts[3] > 0).sum(axis=0)}
    return {
        'method': 'whole_match_case_bootstrap', 'clusters': len(match_counts),
        'replicates': replicates, 'seed': seed, 'rng': 'numpy.PCG64', 'numpy_version': np.__version__,
        'quantiles': [.025, .5, .975], 'quantile_method': 'linear', 'numerical_failures': 0,
        'draw_indices_sha256': hashlib.sha256(draws.astype('<i8', copy=False).tobytes()).hexdigest(),
        'lower_grid': low.tolist(), 'median_grid': median.tolist(), 'upper_grid': high.tolist(),
        'unobserved_replicates': unobserved.reshape(8, 12).tolist(),
        'zero_value_replicates': zeros.reshape(8, 12).tolist(),
        'match_support': {key: value.reshape(8, 12).tolist() for key, value in support.items()},
        'max_abs_deviation_quantiles': dict(zip(('p50', 'p95'), np.quantile(deviation, [.5, .95]).tolist())),
        'method_url': 'https://journal.r-project.org/articles/RJ-2023-015/',
        'scope': '每次有放回抽取相同数量的整场训练比赛，汇总计数后重新求解网格，不平均各场 xT。逐格范围为重采样的2.5%–97.5%百分位，不是已验证覆盖率的置信区间或同时置信带。最大绝对偏差以原样本汇总网格为参照，不是真值误差。',
        'limits': ['只描述这批历史训练比赛中的样本敏感性，不覆盖重复球队带来的跨场相关、时间漂移和建模错误。',
                   '重采样中没有观测或没有进球路径的格可能取零，这不代表现实中没有威胁。',
                   '任何数值求解失败都会终止检查，不删除失败抽样后再计算范围；原留出预测器未重新训练。']}


def audit_grid(database, reference, replicates=0, seed=20261015):
    """Read only the original training IDs in one read-only warehouse snapshot."""
    source, frozen = reference['data'], reference['raw_xt']
    train = source['splits']['train']
    ids = frozen['fit_match_ids']
    if type(replicates) is not int or replicates < 0 or 0 < replicates < 200 or type(seed) is not int or seed < 0:
        raise ValueError('Invalid xT resampling replicates or seed')
    if replicates and len(ids) < 2:
        raise ValueError('xT resampling requires at least two matches')
    if (source['provider'] != 'wyscout' or source['season'] != '2017/18'
            or frozen['fit_split'] != 'train' or ids != train['match_ids']
            or not ids or len(set(ids)) != len(ids)
            or any(type(i) is not int or i <= 0 for i in ids)
            or reference['code_sha256']['action_value.py'] != file_digest(Path(av.__file__))):
        raise ValueError('xT source training identities or count extractor differ')
    old = np.asarray(frozen['grid'], dtype=float)
    if old.shape != (8, 12) or not np.isfinite(old).all() or np.any((old < 0) | (old > 1)):
        raise ValueError('Invalid frozen xT grid')
    counts = av.new_xt_counts()
    digest, dates, per_match = hashlib.sha256(), [], []
    with sqlite3.connect(Path(database).resolve().as_uri() + '?mode=ro', uri=True) as db:
        db.execute('PRAGMA query_only=1')
        db.execute('BEGIN')
        signature = source_signature(db, ('wyscout',))
        if signature != source['inputs']:
            raise ValueError('xT source asset signature differs from the frozen experiment')
        metadata = {}
        for payload, in db.execute('SELECT payload FROM wyscout_entities WHERE entity_type=?', ('matches_' + source['target_league'],)):
            row = json.loads(payload)
            if row['wyId'] in metadata:
                raise ValueError('Duplicate native xT match metadata')
            metadata[row['wyId']] = row
        for mid in ids:
            match = metadata.get(mid)
            if (not match or match['status'] != 'Played' or match['duration'] != 'Regular'
                    or not '2017-07-01' <= match['dateutc'][:10] < '2018-01-01'
                    or not train['date_start'] <= match['dateutc'][:10] <= train['date_end']):
                raise ValueError('Invalid xT training date or match metadata')
            dates.append(match['dateutc'][:10])
            digest.update(json.dumps(match, sort_keys=True).encode())
            events = []
            for payload, in db.execute("SELECT payload FROM wyscout_events WHERE match_id=? AND period IN ('1H','2H') ORDER BY period,seconds,row_number", (mid,)):
                digest.update(payload.encode())
                event = json.loads(payload)
                if event['matchId'] != mid:
                    raise ValueError('xT event provider identity mismatch')
                events.append(event)
            if not events or av.audit_goals(events, match):
                raise ValueError('xT original training match no longer passes native score audit')
            current = av.new_xt_counts()
            av.update_xt(current, events)
            for aggregate, part in zip(counts, current):
                aggregate += part
            if replicates:
                per_match.append(current)
    moves, attempts, shots, goals = counts
    totals = {name: int(a.sum()) for name, a in zip(('attempts', 'shots', 'goals'), counts[1:])}
    if any(totals[key] != frozen[key] for key in totals) or len(ids) != train['matches_accepted']:
        raise ValueError('xT counts differ from the frozen training experiment')
    solved = fit_expected_threat(*counts)
    if not solved['converged']:
        raise ValueError('xT residual bound exceeds the numerical tolerance')
    support = {name: a.reshape(8, 12).astype(int).tolist() for name, a in (
        ('attempts', attempts), ('completed_moves', moves.sum(axis=1)), ('shots', shots), ('goals', goals))}
    off_diagonal = moves.copy()
    np.fill_diagonal(off_diagonal, 0)
    total = attempts + shots
    residual = goals - (((attempts - np.diag(moves)) + shots) * old.ravel() - off_diagonal @ old.ravel())
    report = {
        'schema': 'SoccerAnalytics.XTNumerics.v1', 'source_run_id': reference['run_id'],
        'scope': 'Wyscout 2017/18，原生传球前状态实验的同一批早段训练比赛；只重算空间网格，不重训概率预测器。',
        'source': {'provider': 'Wyscout', 'league': source['target_league'], 'season': '2017/18',
                   'url': 'https://doi.org/10.6084/m9.figshare.c.4415000', 'asset_signature_sha256': signature['sha256']},
        'window': {'start': min(dates), 'end': max(dates)},
        'counts': {'matches': len(ids), **totals, 'completed_moves': int(moves.sum())},
        'support': support, 'grid': solved['grid'],
        'solver': {key: solved[key] for key in ('solver', 'iterations', 'converged', 'diagnostics')},
        'comparison': {'max_grid_difference': float(np.max(np.abs(np.asarray(solved['grid']) - old))),
                       'frozen_bellman_residual_max': float(np.max(np.divide(np.abs(residual), total, out=np.zeros(96), where=total > 0)))},
        'provenance': {'native_training_payloads_sha256': digest.hexdigest(),
                       'reference_sha256': hashlib.sha256(json.dumps(reference, sort_keys=True, allow_nan=False).encode()).hexdigest(),
                       'extractor_sha256': file_digest(Path(av.__file__)), 'solver_sha256': file_digest(Path(__file__).with_name('tactics.py')),
                       'audit_sha256': file_digest(Path(__file__))},
        'limits': ['残差及其放大界只检查已拟合计数系统的数值解，不包括浮点舍入、采样和建模不确定性。',
                   '无射门、无进球或无观测格不代表现实中没有威胁；当前数据与网格不能据此获得统计可靠性认证。',
                   '沿用原生训练基线的传球及射门口径，包含点球和直接任意球，不等同于论文的四百万事件样本。',
                   '原时间留出预测、分类器、分数和原始实验报告保持冻结；没有套用论文的球员排名阈值。']}
    if replicates:
        report['resampling'] = resample_grids(per_match, replicates, seed)
    report['run_id'] = hashlib.sha256(json.dumps(report, sort_keys=True, allow_nan=False).encode()).hexdigest()[:20]
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=Path('data/football.sqlite3'))
    parser.add_argument('--reference', type=Path, default=Path('.tmp/visual-redesign-20261015/value-lab/report.json'))
    parser.add_argument('--output', type=Path, default=Path('results/xt-numerics.json'))
    parser.add_argument('--replicates', type=int, default=1000, help='Whole-match resamples;0 disables, otherwise at least200')
    parser.add_argument('--seed', type=int, default=20261015)
    args = parser.parse_args()
    result = audit_grid(args.database, json.loads(args.reference.read_text()), args.replicates, args.seed)
    write_json_atomic(args.output, result)
    print(json.dumps({'run_id': result['run_id'], 'counts': result['counts'], 'comparison': result['comparison'], 'diagnostics': result['solver']['diagnostics']}))
