"""Action-value public schemas and numerical diagnostic checks."""

import math
import re
from datetime import date
from urllib.parse import urlparse


def public_xt_numerics(report):
    """Publish grid support and numerical checks, without training identities."""
    if (report.get('schema') != 'SoccerAnalytics.XTNumerics.v1'
            or not re.fullmatch(r'[a-f0-9]{20}', report.get('run_id', ''))
            or not re.fullmatch(r'[a-f0-9]{16,20}', report.get('source_run_id', ''))):
        raise ValueError('Invalid xT numerical report')
    public = {key: report[key] for key in ('schema', 'run_id', 'source_run_id', 'scope', 'limits')}
    if (not isinstance(public['scope'], str) or not isinstance(public['limits'], list)
            or not all(isinstance(s, str) for s in public['limits'])):
        raise ValueError('Invalid xT numerical scope')
    public['source'] = {key: report['source'][key] for key in ('provider', 'league', 'season', 'url', 'asset_signature_sha256')}
    source = public['source']; url = urlparse(source['url'])
    if (source['provider'] != 'Wyscout' or source['season'] != '2017/18'
            or source['league'] not in ('England', 'France', 'Germany', 'Italy', 'Spain')
            or url.scheme != 'https' or not url.hostname or url.username or url.password
            or not re.fullmatch(r'[a-f0-9]{64}', source['asset_signature_sha256'])):
        raise ValueError('Invalid xT numerical source')
    public['window'] = {key: date.fromisoformat(report['window'][key]).isoformat() for key in ('start', 'end')}
    if not '2017-07-01' <= public['window']['start'] <= public['window']['end'] < '2018-01-01':
        raise ValueError('Invalid xT training window')
    def number(value, high=1):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= high:
            raise ValueError('Invalid xT numerical value')
        return value
    public['grid'] = [[number(v) for v in row] for row in report['grid']]
    if len(public['grid']) != 8 or any(len(row) != 12 for row in public['grid']):
        raise ValueError('Invalid xT grid dimensions')
    public['counts'] = {key: report['counts'][key] for key in ('matches', 'attempts', 'completed_moves', 'shots', 'goals')}
    if any(type(v) is not int or v < (1 if key == 'matches' else 0) for key, v in public['counts'].items()):
        raise ValueError('Invalid xT event counts')
    public['support'] = {}
    for key in ('attempts', 'completed_moves', 'shots', 'goals'):
        grid = [list(row) for row in report['support'][key]]
        if (len(grid) != 8 or any(len(row) != 12 for row in grid)
                or any(type(v) is not int or v < 0 for row in grid for v in row)
                or sum(map(sum, grid)) != public['counts'][key]):
            raise ValueError('Invalid xT support accounting')
        public['support'][key] = grid
    flat = {key: [v for row in grid for v in row] for key, grid in public['support'].items()}
    if any(c > a or g > s for a, c, s, g in zip(flat['attempts'], flat['completed_moves'], flat['shots'], flat['goals'])):
        raise ValueError('Invalid xT support conservation')
    solver = report['solver']
    if (solver['solver'] != 'direct_count_system_on_goal_reachable_cells' or type(solver['iterations']) is not int
            or solver['iterations'] != 0 or solver['converged'] is not True):
        raise ValueError('Invalid xT numerical solver')
    keys = ('observed_cells', 'unobserved_cells', 'zero_shot_cells', 'zero_goal_cells', 'goal_reachable_cells')
    diagnostic = {key: solver['diagnostics'][key] for key in keys}
    if any(type(v) is not int or not 0 <= v <= 96 for v in diagnostic.values()):
        raise ValueError('Invalid xT support diagnostic')
    observed = sum(a + s > 0 for a, s in zip(flat['attempts'], flat['shots']))
    if (diagnostic['observed_cells'] != observed or diagnostic['unobserved_cells'] != 96 - observed
            or diagnostic['zero_shot_cells'] != flat['shots'].count(0) or diagnostic['zero_goal_cells'] != flat['goals'].count(0)
            or diagnostic['goal_reachable_cells'] > observed):
        raise ValueError('Invalid xT support diagnostic accounting')
    for key in ('bellman_residual_max', 'residual_amplification', 'residual_error_bound'):
        diagnostic[key] = number(solver['diagnostics'][key], float('inf'))
    if (diagnostic['residual_error_bound'] > 1e-9 or not math.isclose(diagnostic['residual_error_bound'], diagnostic['bellman_residual_max'] * diagnostic['residual_amplification'], rel_tol=1e-12, abs_tol=1e-30)):
        raise ValueError('Invalid xT residual accounting')
    if (not isinstance(solver['diagnostics']['scope'], str)
            or diagnostic['goal_reachable_cells'] and diagnostic['residual_amplification'] < 1 - 1e-12
            or not diagnostic['goal_reachable_cells'] and diagnostic['residual_amplification'] != 0):
        raise ValueError('Invalid xT residual scope or amplification')
    diagnostic['scope'] = solver['diagnostics']['scope']
    public['solver'] = {'solver': solver['solver'], 'iterations': 0, 'converged': True, 'diagnostics': diagnostic}
    public['comparison'] = {key: number(report['comparison'][key]) for key in ('max_grid_difference', 'frozen_bellman_residual_max')}
    public['provenance'] = {key: report['provenance'][key] for key in ('native_training_payloads_sha256', 'reference_sha256', 'extractor_sha256', 'solver_sha256', 'audit_sha256')}
    if any(not re.fullmatch(r'[a-f0-9]{64}', value) for value in public['provenance'].values()):
        raise ValueError('Invalid xT provenance digest')
    if 'resampling' in report:
        raw = report['resampling']
        keys = ('method', 'clusters', 'replicates', 'seed', 'rng', 'numpy_version', 'quantiles', 'quantile_method',
                'numerical_failures', 'draw_indices_sha256', 'method_url', 'scope', 'limits')
        sampled = {key: raw[key] for key in keys}
        if (sampled['method'] != 'whole_match_case_bootstrap' or sampled['rng'] != 'numpy.PCG64'
                or sampled['quantiles'] != [.025, .5, .975] or sampled['quantile_method'] != 'linear'
                or type(sampled['clusters']) is not int or sampled['clusters'] != public['counts']['matches'] or sampled['clusters'] < 2
                or type(sampled['replicates']) is not int or sampled['replicates'] < 200
                or type(sampled['seed']) is not int or sampled['seed'] < 0
                or type(sampled['numerical_failures']) is not int or sampled['numerical_failures'] != 0
                or not isinstance(sampled['numpy_version'], str) or not re.fullmatch(r'[a-f0-9]{64}', sampled['draw_indices_sha256'])
                or sampled['method_url'] != 'https://journal.r-project.org/articles/RJ-2023-015/'
                or not isinstance(sampled['scope'], str) or not isinstance(sampled['limits'], list)
                or not all(isinstance(v, str) for v in sampled['limits'])):
            raise ValueError('Invalid xT resampling method or draw accounting')
        def matrix(value, maximum=1, integer=False):
            if len(value) != 8 or any(len(row) != 12 for row in value):
                raise ValueError('Invalid xT resampling grid dimensions')
            if integer and any(type(v) is not int for row in value for v in row):
                raise ValueError('Invalid xT resampling counts')
            return [[number(v, maximum) for v in row] for row in value]
        for key in ('lower_grid', 'median_grid', 'upper_grid'):
            sampled[key] = matrix(raw[key])
        lo, mid, hi = (sum(sampled[key], []) for key in ('lower_grid', 'median_grid', 'upper_grid'))
        if any(not a <= b <= c for a, b, c in zip(lo, mid, hi)):
            raise ValueError('Invalid xT resampling percentile order')
        for key in ('unobserved_replicates', 'zero_value_replicates'):
            sampled[key] = matrix(raw[key], sampled['replicates'], integer=True)
        sampled['match_support'] = {key: matrix(raw['match_support'][key], sampled['clusters'], integer=True)
                                    for key in ('observations', 'shots', 'goals')}
        support = {key: sum(values, []) for key, values in sampled['match_support'].items()}
        unseen, zero = (sum(sampled[key], []) for key in ('unobserved_replicates', 'zero_value_replicates'))
        for i, (observations, shots, goals) in enumerate(zip(support['observations'], support['shots'], support['goals'])):
            if (not goals <= shots <= observations or observations > flat['attempts'][i] + flat['shots'][i]
                    or shots > flat['shots'][i] or goals > flat['goals'][i]
                    or (observations > 0) != (flat['attempts'][i] + flat['shots'][i] > 0)
                    or (shots > 0) != (flat['shots'][i] > 0) or (goals > 0) != (flat['goals'][i] > 0)
                    or unseen[i] > zero[i]
                    or observations == 0 and unseen[i] != sampled['replicates']
                    or observations == sampled['clusters'] and unseen[i] != 0
                    or zero[i] == sampled['replicates'] and hi[i] != 0):
                raise ValueError('Invalid xT resampling support accounting')
        sampled['max_abs_deviation_quantiles'] = {key: number(raw['max_abs_deviation_quantiles'][key]) for key in ('p50', 'p95')}
        if sampled['max_abs_deviation_quantiles']['p50'] > sampled['max_abs_deviation_quantiles']['p95']:
            raise ValueError('Invalid xT resampling deviation order')
        public['resampling'] = sampled
    return public


def public_vaep_calibration(report, spadl):
    """Validate diagnostic accounting and publish only grouped probabilities."""
    source = public_spadl_vaep(spadl)
    if report.get('schema') != 'SoccerAnalytics.VAEPCalibration.v1' or report.get('source_run_id') != source['run_id'] or not re.fullmatch(r'[a-f0-9]{20}', report.get('run_id', '')):
        raise ValueError('Invalid calibration source')
    public = {key: report[key] for key in ('schema', 'run_id', 'source_run_id', 'scope', 'method', 'limits')}
    public['paper_url'] = 'https://doi.org/10.1073/pnas.2016191118'
    public['groups'] = []

    def numeric(value, low=0, high=1):
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError('Invalid calibration number')
        return value

    def count(value, minimum=0):
        if type(value) is not int or value < minimum:
            raise ValueError('Invalid calibration count')
        return value

    def same(a, b, tolerance=1e-10):
        if not math.isclose(a, b, rel_tol=0, abs_tol=tolerance):
            raise ValueError('Invalid calibration accounting')

    expected = {(league['league'], row['method'], head): (row, row['metrics'][head], i)
                for league in source['leagues'] for row in league['results'] if row['scope'] == 'nonterminal_complete_horizon'
                for i, head in enumerate(('score', 'concede'))}
    seen = set()
    for group in report['groups']:
        key = tuple(group[k] for k in ('league', 'method', 'head'))
        if key not in expected or key in seen:
            raise ValueError('Invalid calibration group coverage')
        seen.add(key); row, metric, index = expected[key]
        out = {k: group[k] for k in ('league', 'method', 'head')}
        n = out['states'] = count(group['states'], 1)
        positives = out['positives'] = count(group['positives'])
        if n != row['states'] or positives != row['positive_counts'][index]:
            raise ValueError('Invalid calibration cohort')
        for k in ('observed_rate', 'predicted_rate', 'brier', 'pav_brier', 'mcb', 'dsc', 'unc'):
            out[k] = numeric(group[k], -1e-12 if k in ('mcb', 'dsc') else 0)
        for k in ('observed_rate', 'predicted_rate', 'brier'): same(out[k], metric[k], 1e-7)
        same(out['observed_rate'], positives / n)
        same(out['unc'], out['observed_rate'] * (1 - out['observed_rate']))
        same(out['mcb'], out['brier'] - out['pav_brier'])
        same(out['dsc'], out['unc'] - out['pav_brier'])

        def buckets(values, size=None):
            result, previous = [], None
            for i, b in enumerate(values):
                item = {k: numeric(b[k]) for k in ('lower', 'upper')}
                bn, hits = count(b['count'], 0 if size else 1), count(b['positives'])
                item.update(count=bn, positives=hits)
                if hits > bn or item['lower'] > item['upper']:
                    raise ValueError('Invalid calibration bucket')
                if size:
                    same(item['lower'], i / size); same(item['upper'], (i + 1) / size)
                elif previous and (item['lower'] <= previous['upper'] or b['observed_rate'] <= previous['observed_rate']):
                    raise ValueError('Invalid calibration PAV order')
                if bn:
                    item['mean_probability'] = numeric(b['mean_probability'], item['lower'], item['upper'])
                    item['observed_rate'] = numeric(b['observed_rate'])
                    same(item['observed_rate'], hits / bn)
                else:
                    if b['mean_probability'] is not None or b['observed_rate'] is not None:
                        raise ValueError('Invalid calibration empty bucket')
                    item.update(mean_probability=None, observed_rate=None)
                result.append(item); previous = item
            if (size and len(result) != size) or sum(b['count'] for b in result) != n or sum(b['positives'] for b in result) != positives:
                raise ValueError('Invalid calibration bucket totals')
            same(sum(b['count'] * b['mean_probability'] for b in result if b['count']) / n, out['predicted_rate'])
            return result

        out['pav'] = buckets(group['pav'])
        same(sum(b['count'] * b['observed_rate'] * (1 - b['observed_rate']) for b in out['pav']) / n, out['pav_brier'])
        out['binning'] = {}
        for size in (9, 10, 11):
            block = group['binning'][str(size)]
            values = buckets(block['bins'], size)
            ece = numeric(block['ece'])
            same(sum(b['count'] * abs(b['mean_probability'] - b['observed_rate']) for b in values if b['count']) / n, ece)
            out['binning'][str(size)] = {'ece': ece, 'bins': values}
        public['groups'].append(out)
    if seen != set(expected):
        raise ValueError('Invalid calibration missing group')
    return public


def public_spadl_vaep(report):
    """Publish temporal probability diagnostics and action-type means only."""
    if report.get('schema') != 'SoccerAnalytics.SPADLVAEP.v1' or report.get('horizon') != 10 or not re.fullmatch(r'[a-f0-9]{20}', report.get('run_id', '')):
        raise ValueError('Invalid SPADL report')
    public = {key: report[key] for key in ('schema', 'run_id', 'horizon', 'source_url', 'paper_url', 'implementation_url', 'limits')}
    for key in ('source_url', 'paper_url', 'implementation_url'):
        url = urlparse(public[key])
        if url.scheme != 'https' or not url.hostname or url.username or url.password:
            raise ValueError('Invalid SPADL source URL')
    def finite(value, lower=-math.inf, upper=math.inf):
        if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError('Invalid SPADL numeric metric')
        return value
    def count(value, minimum=0):
        if type(value) is not int or value < minimum:
            raise ValueError('Invalid SPADL count')
        return value
    public['leagues'], leagues = [], set()
    for source in report['leagues']:
        league = source['league']
        if league not in ('England', 'France') or league in leagues:
            raise ValueError('Invalid SPADL league coverage')
        leagues.add(league)
        row = {'league': league, 'splits': {}, 'results': []}
        for split in ('train', 'validation', 'test'):
            block = source['splits'][split]
            row['splits'][split] = {key: block[key] for key in ('matches', 'actions', 'eligible_states', 'start', 'end')}
            for key in ('matches', 'actions', 'eligible_states'): count(block[key], 1)
            if block['eligible_states'] != block['actions'] - 18 * block['matches']:
                raise ValueError('Invalid SPADL horizon accounting')
            date.fromisoformat(block['start']); date.fromisoformat(block['end'])
        s = row['splits']
        if not ('2017-07-01' <= s['train']['start'] <= s['train']['end'] < '2018-01-01' <= s['validation']['start'] <= s['validation']['end'] < '2018-03-01' <= s['test']['start'] <= s['test']['end'] < '2018-07-01'):
            raise ValueError('Invalid SPADL chronology')
        seen = set()
        for source_result in source['results']:
            result = {key: source_result[key] for key in ('method', 'scope', 'states', 'goals_in_current_action', 'positive_counts')}
            identity = (result['method'], result['scope'])
            if identity in seen or identity[0] not in ('prior', 'single', 'context') or identity[1] not in ('complete_horizon', 'nonterminal_complete_horizon'):
                raise ValueError('Invalid SPADL comparison coverage')
            seen.add(identity); n = count(result['states'], 1)
            if n > s['test']['eligible_states'] or count(result['goals_in_current_action']) > n or len(result['positive_counts']) != 2:
                raise ValueError('Invalid SPADL state accounting')
            result['metrics'] = {}
            for index, head in enumerate(('score', 'concede')):
                metric = source_result['metrics'][head]
                out = {key: finite(metric[key], 0, math.inf if key == 'log_loss' else 1) for key in ('log_loss', 'brier', 'observed_rate', 'predicted_rate')}
                positives = count(result['positive_counts'][index])
                # The shared metric routine stores label means at float32 precision.
                if positives > n or not math.isclose(out['observed_rate'], positives / n, rel_tol=2e-7, abs_tol=1e-9):
                    raise ValueError('Invalid SPADL positive counts')
                bins = []
                for i, bucket in enumerate(metric['reliability']):
                    b = {key: bucket[key] for key in ('lower', 'upper', 'count', 'mean_probability', 'observed_rate')}
                    if not math.isclose(b['lower'], i / 10) or not math.isclose(b['upper'], (i + 1) / 10):
                        raise ValueError('Invalid SPADL calibration boundaries')
                    if count(b['count']):
                        finite(b['mean_probability'], b['lower'], b['upper']); finite(b['observed_rate'], 0, 1)
                    elif b['mean_probability'] is not None or b['observed_rate'] is not None:
                        raise ValueError('Invalid SPADL empty calibration bin')
                    bins.append(b)
                if len(bins) != 10 or sum(b['count'] for b in bins) != n:
                    raise ValueError('Invalid SPADL calibration accounting')
                out['reliability'] = bins; result['metrics'][head] = out
            row['results'].append(result)
        if len(seen) != 6:
            raise ValueError('Invalid SPADL missing comparison')
        row['source'] = {'excluded_matches': count(source['source']['excluded_matches']),
                         'audit': {key: count(source['source']['audit'][key]) for key in ('native_events', 'actions', 'inferred_dribbles', 'ignored_goalkeeper_goal_duplicates', 'incomplete_horizon_states')}}
        row['paired_nonterminal_intervals'] = {}
        for comparator in ('prior', 'single'):
            row['paired_nonterminal_intervals'][comparator] = {}
            for head in ('score', 'concede'):
                row['paired_nonterminal_intervals'][comparator][head] = {}
                for metric in ('log_loss', 'brier'):
                    item = source['paired_nonterminal_intervals'][comparator][head][metric]
                    lo, hi = [finite(v) for v in item['interval95']]
                    if lo > hi: raise ValueError('Invalid SPADL interval order')
                    row['paired_nonterminal_intervals'][comparator][head][metric] = {'difference': finite(item['difference']), 'interval95': [lo, hi]}
        types = {'pass', 'cross', 'throw_in', 'freekick_crossed', 'freekick_short', 'corner_crossed', 'corner_short',
                 'take_on', 'foul', 'tackle', 'interception', 'shot', 'shot_penalty', 'shot_freekick', 'keeper_save',
                 'keeper_claim', 'keeper_punch', 'keeper_pick_up', 'clearance', 'bad_touch', 'non_action', 'dribble', 'goalkick'}
        row['test_action_values'] = {}
        for kind, values in source['test_action_values'].items():
            if kind not in types: raise ValueError('Invalid SPADL action type')
            group = {key: count(values[key], 1 if key == 'actions' else 0) for key in ('actions', 'inferred')}
            group.update({key: finite(values[key], -2 if key == 'vaep_value' else -1, 2 if key == 'vaep_value' else 1)
                          for key in ('offensive_value', 'defensive_value', 'vaep_value')})
            if group['inferred'] > group['actions'] or not math.isclose(group['vaep_value'], group['offensive_value'] + group['defensive_value'], abs_tol=1e-10):
                raise ValueError('Invalid SPADL action value accounting')
            row['test_action_values'][kind] = group
        if sum(g['actions'] for g in row['test_action_values'].values()) != s['test']['eligible_states'] - 2 * s['test']['matches']:
            raise ValueError('Invalid SPADL action value cohort')
        public['leagues'].append(row)
    if leagues != {'England', 'France'}:
        raise ValueError('Invalid SPADL missing league')
    public['limits'] = list(public['limits']) + ['进球数核对失败的比赛会排除，样本选择可能影响结论。公开结果只含汇总，个体动作值与模型保留在本地。']
    return public
