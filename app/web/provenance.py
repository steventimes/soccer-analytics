"""Public run identities without private inputs or inferred historical metadata."""
import copy
import hashlib
import re
import subprocess
from pathlib import Path

from app.run_receipts import content_digest


CONFIGURATION_FIELDS = {
    'command', 'source', 'seasons', 'competitions', 'model', 'models', 'methods',
    'window', 'windows', 'folds', 'seed', 'as_of', 'timing_policy', 'scope',
    'trials', 'optimizer', 'holdout_fraction', 'calibration_fraction', 'max_rows',
    'test_seasons', 'class_index', 'bins', 'features',
}


def result_artifact(path, source_root):
    """Identify the exact result bytes; link to Git only when its bytes match."""
    path, source_root = Path(path).resolve(), Path(source_root).resolve()
    raw = path.read_bytes()
    artifact = {'path': path.relative_to(source_root).as_posix(),
                'sha256': hashlib.sha256(raw).hexdigest(), 'repository_url': None}
    root = Path(__file__).resolve().parents[2]
    if source_root == root / 'results':
        try:
            name = path.relative_to(root).as_posix()
            head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root,
                                  check=True, capture_output=True, text=True, timeout=5).stdout.strip()
            committed = subprocess.run(['git', 'show', f'{head}:{name}'], cwd=root,
                                      check=True, capture_output=True, timeout=5).stdout
            if committed == raw:
                artifact['repository_url'] = f'https://github.com/steventimes/soccer-analytics/blob/{head}/{name}'
        except (OSError, subprocess.SubprocessError):
            pass
    return artifact


def public_analysis_provenance(report, *, artifact=None):
    provenance = {'schema': 'Touchline.AnalysisProvenance.v1',
                  'status': 'legacy_missing', 'result': None,
                  'run_id': None, 'code': None, 'environment': None,
                  'configuration': None, 'configuration_sha256': None,
                  'timing_policy': 'not_recorded'}
    if artifact is not None:
        path = artifact.get('path')
        if (not isinstance(path, str) or Path(path).is_absolute() or '..' in Path(path).parts
                or not re.fullmatch(r'[a-f0-9]{64}', artifact.get('sha256', ''))):
            raise ValueError('Invalid public result identity')
        provenance['result'] = {k: artifact.get(k) for k in ('path', 'sha256', 'repository_url')}
    receipt = report.get('run_receipt')
    if receipt is None:
        return provenance
    if (not isinstance(receipt, dict) or receipt.get('schema') != 'SoccerAnalytics.RunReceipt.v1'
            or receipt.get('run_id') != content_digest({k: v for k, v in receipt.items() if k != 'run_id'})):
        raise ValueError('Invalid analysis run receipt')
    code, environment, configuration = (receipt.get(k) for k in ('code', 'environment', 'configuration'))
    if (not isinstance(code, dict) or not re.fullmatch(r'[a-f0-9]{64}', code.get('application_sha256', ''))
            or (code.get('git_head') is not None and not re.fullmatch(r'[a-f0-9]{40}', code['git_head']))
            or not isinstance(environment, dict) or not isinstance(environment.get('python'), str)
            or not isinstance(environment.get('packages'), dict) or not isinstance(configuration, dict)):
        raise ValueError('Incomplete analysis run receipt')
    configuration_public = {k: copy.deepcopy(v) for k, v in configuration.items() if k in CONFIGURATION_FIELDS}
    policy = configuration.get('timing_policy')
    if policy is None and 'as_of' in configuration:
        policy = 'observed_result' if configuration['as_of'] else 'kickoff_order'
    policy = {'observed': 'observed_result'}.get(policy, policy)
    if policy not in (None, 'kickoff_order', 'observed_result'):
        raise ValueError('Unknown analysis timing policy')
    packages = environment['packages']
    if any(not isinstance(v, str) for v in packages.values()):
        raise ValueError('Invalid analysis environment versions')
    provenance.update(status='recorded', run_id=receipt['run_id'],
                      code={k: code.get(k) for k in ('git_head', 'application_sha256')},
                      environment={'python': environment['python'], 'packages': {
                          k: v for k, v in packages.items() if k in {
                              'numpy', 'scipy', 'xgboost-cpu', 'scikit-learn', 'torch',
                              'optuna', 'lightgbm', 'catboost', 'scikit-optimize',
                              'fpstreams', 'interpret-core'}}}, configuration=configuration_public,
                      configuration_sha256=content_digest(configuration),
                      timing_policy=policy or 'not_recorded')
    return provenance
