"""Seal future study settings and historical inputs; a seal is not prospective accuracy evidence."""
from datetime import UTC, datetime

from app.ml.alternatives import METHODS
from app.ml.availability import aware_time, available_records
from app.run_receipts import build_run_receipt, content_digest


def register_forecasts(report, *, input_cutoff, input_sha256, now=None):
    """Seal an observed forecast batch before kickoff; timestamps are not authentication."""
    import re
    import json
    import math
    from app.ml.forecast_identity import forecast_id_matches

    if not isinstance(report, dict):
        raise ValueError('预测批次必须为对象')
    recorded = aware_time(now or datetime.now(UTC))
    generated, cutoff = aware_time(report['generated_at']), aware_time(input_cutoff)
    if not cutoff <= generated <= recorded:
        raise ValueError('输入截止、预测生成与实际登记时间顺序非法')
    if not re.fullmatch(r'[a-f0-9]{64}', input_sha256):
        raise ValueError('预测输入文件身份非法')
    predictions = report.get('predictions')
    if not isinstance(predictions, list) or not predictions or report.get('errors'):
        raise ValueError('仅登记完整、非空且无来源错误的预测批次')
    entries, seen = [], set()
    for row in predictions:
        if not isinstance(row, dict) or not forecast_id_matches(row):
            raise ValueError('预测身份或概率不一致')
        keys = ('away_win', 'draw', 'home_win')
        probabilities = [row['probabilities'][key] for key in keys]
        if (isinstance(row['confidence'], bool) or not isinstance(row['confidence'], (int, float))
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 1 for v in probabilities)
                or not math.isclose(sum(probabilities), 1, abs_tol=1e-6, rel_tol=0)
                or not math.isclose(row['confidence'], max(probabilities), abs_tol=1e-6, rel_tol=0)
                or row['prediction'] != ('AWAY_WIN', 'DRAW', 'HOME_WIN')[max(range(3), key=probabilities.__getitem__)]
                or aware_time(row['model']['trained_at']) > generated):
            raise ValueError('预测概率、选择或模型训练时点非法')
        key = (row['competition'], row['match_id'])
        if key in seen or aware_time(row['utc_date']) <= recorded:
            raise ValueError('比赛重复或已开球，不能登记为赛前记录')
        seen.add(key)
        entry = {k: row[k] for k in ('competition', 'match_id', 'utc_date', 'home_team_id',
                 'away_team_id', 'forecast_id', 'feature_fingerprint', 'probabilities',
                 'confidence', 'prediction', 'data_quality')}
        entry['model'] = {k: row['model'][k] for k in ('schema_version', 'model_sha256', 'trained_at')}
        entries.append(entry)
    body = {'schema': 'SoccerAnalytics.ForecastRegistration.v1',
            'recorded_at': recorded.isoformat(), 'generated_at': generated.isoformat(),
            'input_cutoff': cutoff.isoformat(), 'input_sha256': input_sha256,
            'predictions': entries, 'prospective_accuracy_verified': False,
            'run_receipt': build_run_receipt('record-forecasts',
                configuration={'input_cutoff': cutoff.isoformat()},
                inputs={'forecast_file_sha256': input_sha256}, stages={'registration': 'complete'}),
            'limits': ['Input cutoff is caller-declared, not independently verified.',
                       'Local clock and hashes do not authenticate a publication time.',
                       'No independent schedule completeness or outcome evaluation is provided.']}
    body = json.loads(json.dumps(body, allow_nan=False))
    return {**body, 'sha256': content_digest(body)}


def _history_identity(collections, test_season):
    return {'|'.join(key): content_digest(matches) for key, matches in sorted(collections.items())
            if key[2] < test_season}


def freeze_season_protocol(collections, *, test_season, evaluation_start, evaluation_end,
                           models=('logistic_l2',), windows=('1',), seed=42, now=None, training_as_of=None):
    created = aware_time(now or datetime.now(UTC))
    start, end = aware_time(evaluation_start), aware_time(evaluation_end)
    training_limit = aware_time(training_as_of) if training_as_of is not None else created
    if training_limit > created:
        raise ValueError("历史训练截止时间不能晚于协议注册")
    if not created < start < end:
        raise ValueError("前瞻协议必须在评估区间开始前封存，且结束晚于开始")
    if (not collections or not isinstance(test_season, str) or not test_season.isdigit()
            or not models or len(set(models)) != len(models) or any(m not in METHODS for m in models)
            or not windows or any(w != 'all' and (not w.isdigit() or int(w) < 1) for w in windows)
            or type(seed) is not int):
        raise ValueError("协议的赛季、模型、历史窗口或种子非法")
    for key, matches in collections.items():
        if key[2] >= test_season:
            raise ValueError("封存输入只能包含早于测试赛季的历史快照")
        records = available_records(matches)
        if not records or any(r.result_available_at >= training_limit for r in records):
            raise ValueError("封存训练结果必须在封存时点之前可用")
    body = {'schema': 'SoccerAnalytics.SeasonProtocol.v1', 'created_at': created.isoformat(),
            'test_season': test_season, 'evaluation_start': start.isoformat(), 'evaluation_end': end.isoformat(),
            'training_as_of': training_limit.isoformat(),
            'models': list(models), 'windows': list(windows), 'seed': seed,
            'sources_and_competitions': sorted({(k[0], k[1]) for k in collections}),
            'training_snapshots': _history_identity(collections, test_season),
            'timing_policy': 'observed_result', 'pooled': False,
            'selection': 'fixed_default_presets_no_holdout_selection',
            'run_receipt': build_run_receipt('freeze-season-protocol', configuration={
                'models': list(models), 'windows': list(windows), 'seed': seed},
                inputs={'history': _history_identity(collections, test_season)}, stages={'registration': 'complete'}),
            'limits': ['Registration and content hashes are not authentication, pre-match prediction logs, or completed future validation.']}
    # JSON canonicalization makes a persisted round trip identical to the in-memory protocol.
    import json
    body = json.loads(json.dumps(body))
    return {**body, 'sha256': content_digest(body)}


def validate_season_protocol(protocol, collections, *, as_of):
    body = {k: v for k, v in protocol.items() if k != 'sha256'}
    if protocol.get('schema') != 'SoccerAnalytics.SeasonProtocol.v1' or content_digest(body) != protocol.get('sha256'):
        raise ValueError("协议内容身份不一致")
    start, end = aware_time(body['evaluation_start']), aware_time(body['evaluation_end'])
    if not aware_time(body['created_at']) < start < end <= aware_time(as_of):
        raise ValueError("协议尚未到评估结束时点，或注册时间非法")
    if _history_identity(collections, body['test_season']) != body['training_snapshots']:
        raise ValueError("封存的训练快照已经改变")
    pairs = sorted([list(k) for k in {(k[0], k[1]) for k in collections}])
    if pairs != body['sources_and_competitions']:
        raise ValueError("来源或联赛与协议不一致")
    receipt = build_run_receipt('protocol-validation', configuration={}, inputs={}, stages={})
    if (receipt['code']['application_sha256'] != body['run_receipt']['code']['application_sha256']
            or receipt['environment'] != body['run_receipt']['environment']):
        raise ValueError("代码或研究环境与封存协议不一致")
    for pair in pairs:
        rows = collections.get((*pair, body['test_season']), [])
        if not rows:
            raise ValueError("协议要求的测试赛季缺失")
        records = available_records(rows)
        if not records:
            raise ValueError("协议测试赛季没有可评估的已完赛记录")
        for record in records:
            if not start <= record.kickoff < end or record.result_available_at > aware_time(as_of):
                raise ValueError("测试记录超出封存区间或结果尚不可用")
    return {'status': 'registration_integrity_verified', 'protocol_sha256': protocol['sha256'],
            'prospective_accuracy_verified': False,
            'coverage_limit': 'Observed snapshot fixtures only; completeness requires an independent schedule ledger.'}
