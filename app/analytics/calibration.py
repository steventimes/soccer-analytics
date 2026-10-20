"""Retrospective CORP diagnostics for frozen SPADL probabilities, not a calibrator.

PAV is fitted on the diagnostic cohort itself. Its score decomposition describes
that cohort and must not be reported as held-out performance after recalibration.
See Dimitriadis, Gneiting and Jordan (2021), doi:10.1073/pnas.2016191118.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from app.file_io import write_json_atomic


def binary_diagnostics(labels, probabilities):
    from sklearn.isotonic import IsotonicRegression

    y, p = np.asarray(labels, dtype=np.float64), np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape or not len(y) or not np.isfinite(p).all() or not np.isin(y, [0, 1]).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid binary calibration inputs')
    unique, inverse, counts = np.unique(p, return_inverse=True, return_counts=True)
    hits = np.bincount(inverse, weights=y, minlength=len(unique))
    # Group identical probabilities before PAV, so ties cannot depend on row order.
    fitted = IsotonicRegression(increasing=True).fit_transform(unique, hits / counts, sample_weight=counts)
    boundaries = np.r_[0, np.flatnonzero(np.diff(fitted)) + 1, len(unique)]
    blocks = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        n, positive = int(counts[start:end].sum()), int(hits[start:end].sum())
        # Roundoff can move a constant-probability mean just outside its support.
        mean = np.clip(np.dot(unique[start:end], counts[start:end]) / n, unique[start], unique[end - 1])
        blocks.append({'lower': float(unique[start]), 'upper': float(unique[end - 1]),
                       'count': n, 'positives': positive,
                       'mean_probability': float(mean),
                       'observed_rate': positive / n})
    brier = float(np.mean((p - y) ** 2))
    pav_brier = float(np.mean((fitted[inverse] - y) ** 2))
    rate = float(y.mean()); unc = rate * (1 - rate)
    out = {'states': len(y), 'positives': int(y.sum()), 'observed_rate': rate,
           'predicted_rate': float(p.mean()), 'brier': brier, 'pav_brier': pav_brier,
           'mcb': brier - pav_brier, 'dsc': unc - pav_brier, 'unc': unc,
           'pav': blocks, 'binning': {}}
    for size in (9, 10, 11):
        indices = np.minimum(size - 1, (p * size).astype(int))
        bins = []
        for i in range(size):
            mask = indices == i; n = int(mask.sum())
            bins.append({'lower': i / size, 'upper': (i + 1) / size, 'count': n,
                         'positives': int(y[mask].sum()),
                         'mean_probability': float(p[mask].mean()) if n else None,
                         'observed_rate': float(y[mask].mean()) if n else None})
        ece = sum(b['count'] * abs(b['mean_probability'] - b['observed_rate']) for b in bins if b['count']) / len(y)
        out['binning'][str(size)] = {'ece': ece, 'bins': bins}
    return out


def build_spadl(report_path, predictions_root, output):
    """Read existing test arrays only; never load/refit the predictive models."""
    report_path, predictions_root = Path(report_path), Path(predictions_root)
    report_bytes = report_path.read_bytes(); source = json.loads(report_bytes)
    if source.get('schema') != 'SoccerAnalytics.SPADLVAEP.v1':
        raise ValueError('Invalid SPADL calibration source')
    groups, hashes = [], {}
    for league in source['leagues']:
        path = predictions_root / league['league'] / 'test-predictions.npz'
        hashes[league['league']] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as arrays:
            y, eligible, terminal = arrays['y'], arrays['eligible'], arrays['terminal']
            if y.shape != (league['splits']['test']['actions'], 2) or eligible.shape != y.shape[:1] or terminal.shape != eligible.shape or eligible.dtype != bool or terminal.dtype != bool:
                raise ValueError('Invalid SPADL calibration arrays')
            mask = eligible & ~terminal
            for method in ('prior', 'single', 'context'):
                expected = next(r for r in league['results'] if r['method'] == method and r['scope'] == 'nonterminal_complete_horizon')
                probabilities = arrays[method]
                if probabilities.shape != y.shape or int(mask.sum()) != expected['states']:
                    raise ValueError('Invalid SPADL calibration cohort')
                for i, head in enumerate(('score', 'concede')):
                    diagnostics = binary_diagnostics(y[mask, i], probabilities[mask, i])
                    metric = expected['metrics'][head]
                    if diagnostics['positives'] != expected['positive_counts'][i] or any(not np.isclose(diagnostics[k], metric[k], rtol=2e-7, atol=1e-7) for k in ('brier', 'observed_rate', 'predicted_rate')):
                        raise ValueError('SPADL calibration predictions do not match the frozen report')
                    groups.append({'league': league['league'], 'method': method, 'head': head, **diagnostics})
    report = {'schema': 'SoccerAnalytics.VAEPCalibration.v1', 'source_run_id': source['run_id'],
              'paper_url': 'https://doi.org/10.1073/pnas.2016191118',
              'scope': 'Wyscout 2017/18，英超与法甲。固定模型的时间留出预测，排除当前进球与不完整十动作窗口。',
              'method': '相同预测概率先合并，再做非递减 PAV 拟合。Brier = MCB − DSC + UNC。另列 9、10、11 个等宽分箱的描述性 ECE。',
              'limits': ['PAV 使用这批测试标签拟合，只作回顾性诊断；拟合后的分数不是校准模型的留出表现。',
                         'MCB 是同样本 PAV 拟合前后的 Brier 差，DSC 相对这批样本的事件频率，UNC 为该事件频率的二项方差。',
                         '低 MCB 不代表预测有区分力。ECE 依赖分箱，不能单独用于挑选模型。',
                         '状态及重复球队存在相关性。这里没有置信带，不据此声称显著性、跨赛季效果或可部署收益。'],
              'groups': groups,
              'provenance': {'source_report_sha256': hashlib.sha256(report_bytes).hexdigest(),
                             'prediction_files_sha256': hashes,
                             'extractor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    report['run_id'] = hashlib.sha256(json.dumps(report, sort_keys=True, allow_nan=False).encode()).hexdigest()[:20]
    write_json_atomic(Path(output), report)
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', default='results/vaep-spadl.json')
    parser.add_argument('--predictions', default='.tmp/vaep-spadl-20261015/run')
    parser.add_argument('--output', default='results/vaep-calibration.json')
    args = parser.parse_args()
    result = build_spadl(args.report, args.predictions, args.output)
    print(json.dumps({'run_id': result['run_id'], 'groups': len(result['groups'])}))
