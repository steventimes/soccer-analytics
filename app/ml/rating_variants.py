"""Research-only goal-weighted Elo and venue pi-ratings with separate mappings."""

from collections import defaultdict
from dataclasses import replace
import math
import hashlib
import json
from time import perf_counter
from types import SimpleNamespace

from app.ml.alternatives import ModelConvergenceError, estimator_factory
from app.ml.evaluation import evaluate_expanding_window
from app.ml.experiments import chronological_split
from app.ml.feature_engineering import FeatureEngineer
from app.ml.model import (accuracy_score, estimator_runtime, fit_estimator,
                          multiclass_brier_score, multiclass_log_loss, ranked_probability_score)
from app.ml.score_models import prepare_score_matches


def rating_dataset(records, *, kind='basic', k=20., home_offset=80., power=.5,
                   learning_rate=.035, catchup=.7, state_sink=None):
    if kind not in ('basic', 'goal_weighted', 'pi'):
        raise ValueError('未知评级定义')
    for value in (k, home_offset, power, learning_rate, catchup):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('评级参数必须是有限非负数')
    if catchup > 1 or learning_rate > 1 or power > 2:
        raise ValueError('评级学习率、追赶率或净胜球指数超出研究边界')
    if state_sink is not None and not callable(state_sink):
        raise ValueError('state_sink 必须可调用')
    records, _ = prepare_score_matches(records)
    data = FeatureEngineer().build_training_dataset(records)
    rows = [dict(row) for row in data.rows]
    if kind == 'basic':
        for row in rows:
            row['rating_diff'] = row['elo_diff']
        if state_sink is not None:
            state_sink({key: [value['elo']] for key, value in data.team_states.items()})
        return replace(data, rows=tuple(rows))
    states = defaultdict(lambda: [0., 0.] if kind == 'pi' else [1500.])
    start = 0
    def goal_equivalent(value):
        return math.copysign(math.expm1(math.log(10) * abs(value) / 3), value)

    while start < len(records):
        end = start + 1
        while end < len(records) and records[end].kickoff == records[start].kickoff:
            end += 1
        deltas = defaultdict(lambda: [[], []] if kind == 'pi' else [[]])
        for i in range(start, end):
            match = records[i]
            home, away = states[match.home_team_id], states[match.away_team_id]
            observed = match.home_score - match.away_score
            if kind == 'pi':
                expected = goal_equivalent(home[0]) - goal_equivalent(away[1])
                rows[i].update(rating_diff=expected, rating_home=home[0], rating_away=away[1])
                error = observed - expected
                delta = learning_rate * math.copysign(3 * math.log10(1 + abs(error)), error)
                # Update the played venue first; opposite venue follows by catchup.
                deltas[match.home_team_id][0].append(delta)
                deltas[match.home_team_id][1].append(catchup * delta)
                deltas[match.away_team_id][1].append(-delta)
                deltas[match.away_team_id][0].append(-catchup * delta)
            else:
                difference = home[0] - away[0]
                rows[i].update(home_elo=home[0], away_elo=away[0], elo_diff=difference, rating_diff=difference)
                d = (difference + home_offset) / 400
                small = 10 ** -abs(d)
                expected = 1 / (1 + small) if d >= 0 else small / (1 + small)
                actual = 1. if observed > 0 else .5 if observed == 0 else 0.
                delta = k * (1 + abs(observed)) ** power * (actual - expected)
                deltas[match.home_team_id][0].append(delta)
                deltas[match.away_team_id][0].append(-delta)
        # Every row at the same kickoff observes the same pre-kickoff states.
        for team, values in deltas.items():
            for venue, changes in enumerate(values):
                states[team][venue] += math.fsum(changes)
                if not math.isfinite(states[team][venue]):
                    raise ValueError('评级状态出现非有限数值')
        start = end
    if state_sink is not None:
        state_sink({str(key): list(value) for key, value in states.items()})
    return replace(data, rows=tuple(rows))


def run_rating_variants(matches, *, variants=('basic', 'goal_weighted', 'pi'), warmup_matches=0,
                        ridge=.001, folds=3, seed=42, prediction_sink=None):
    if (not variants or len(set(variants)) != len(variants)
            or any(v not in ('basic', 'goal_weighted', 'pi') for v in variants)
            or type(warmup_matches) is not int or warmup_matches < 0):
        raise ValueError('评级选择必须非空、唯一且有效，预热场数必须为非负整数')
    if prediction_sink is not None and not callable(prediction_sink):
        raise ValueError('prediction_sink 必须可调用')
    records, counts = prepare_score_matches(matches)
    seen = defaultdict(int)
    eligible, start = [], 0
    while start < len(records):
        end = start + 1
        while end < len(records) and records[end].kickoff == records[start].kickoff:
            end += 1
        eligible.extend(i for i in range(start, end) if min(seen[records[i].home_team_id], seen[records[i].away_team_id]) >= warmup_matches)
        for row in records[start:end]:
            seen[row.home_team_id] += 1; seen[row.away_team_id] += 1
        start = end
    times = tuple(records[i].kickoff.isoformat() for i in eligible)
    split = chronological_split(SimpleNamespace(rows=eligible, kickoffs=times))
    cutoff = records[eligible[split]].kickoff
    development_records = tuple(r for r in records if r.kickoff < cutoff)
    development_positions = [i for i in eligible if i < len(development_records)]
    if len(development_positions) != split:
        raise ValueError('预热与时间分区不一致')
    def subset(data, positions):
        return replace(data, rows=tuple(data.rows[i] for i in positions),
                       targets=tuple(data.targets[i] for i in positions),
                       match_ids=tuple(data.match_ids[i] for i in positions),
                       kickoffs=tuple(data.kickoffs[i] for i in positions))

    specifications = []
    if 'basic' in variants:
        specifications += [('elo_ordered_logit', {'kind': 'basic'}, 'elo_ordered_logit', ('elo_diff',), {'ridge': ridge}),
                           ('elo_multinomial', {'kind': 'basic'}, 'logistic_l2', ('elo_diff',), {}),
                           ('logistic_l2', {'kind': 'basic'}, 'logistic_l2', FeatureEngineer.FEATURES, {}),
                           ('xgboost', {'kind': 'basic'}, 'xgboost', FeatureEngineer.FEATURES, {})]
    if 'goal_weighted' in variants:
        specifications += [(f'goal_elo_k{k:g}_p{power:g}', {'kind': 'goal_weighted', 'k': k, 'power': power,
                                                        'home_offset': 80.}, 'elo_ordered_logit', ('elo_diff',), {'ridge': ridge})
                           for k in (10., 20.) for power in (.5, 1.)]
    if 'pi' in variants:
        specifications += [(f'pi_lr{rate:g}_catchup{catchup:g}', {'kind': 'pi', 'learning_rate': rate, 'catchup': catchup},
                            'elo_ordered_logit', ('rating_diff',), {'ridge': ridge, 'feature_name': 'rating_diff', 'unit_scale': 1.})
                           for rate in (.035, .07) for catchup in (.5, .7)]
    candidates = []
    for name, recipe, method, features, params in specifications:
        state_summary = {}
        def capture_state(states):
            state_summary.update(teams=len(states),
                                 sha256=hashlib.sha256(json.dumps(states, sort_keys=True).encode()).hexdigest(),
                                 after_kickoff=development_records[-1].kickoff.isoformat())
        data = subset(rating_dataset(development_records, **recipe, state_sink=capture_state), development_positions)
        item = {'model': name, 'rating': recipe, 'estimator': method, 'params': params, 'feature_names': list(features),
                'rating_version': {'basic': 'basic_elo_v1', 'goal_weighted': 'goal_margin_elo_v1', 'pi': 'venue_pi_signed_log_v1'}[recipe['kind']],
                'mapping_version': 'proportional_odds_v1' if method == 'elo_ordered_logit' else 'multinomial_or_tree_v1'}
        item['rating_state_after_development'] = state_summary
        started = perf_counter()
        try:
            score = evaluate_expanding_window(data.rows, data.targets, features,
                       estimator_factory(method, seed=seed, params=params), folds=folds, kickoffs=data.kickoffs)
            item.update(selection_log_loss=score.log_loss, development_folds=list(score.fold_metrics))
        except ModelConvergenceError as exc:
            item.update(selection_log_loss=None, selection_error=str(exc))
        item['development_objective_seconds'] = perf_counter() - started
        candidates.append(item)
    valid = [c for c in candidates if c['selection_log_loss'] is not None]
    if not valid:
        raise ModelConvergenceError('评级与映射候选均未收敛')
    selected = min(valid, key=lambda c: c['selection_log_loss'])['model']
    for item in valid:
        data = subset(rating_dataset(records, **item['rating']), eligible)
        model = estimator_factory(item['estimator'], seed=seed, params=item['params'])(tuple(item['feature_names']))
        fit_started = perf_counter()
        try:
            fit_estimator(model, data.rows[:split], data.targets[:split], kickoffs=times[:split])
        except ModelConvergenceError as exc:
            item['holdout_error'] = str(exc)
            item['runtime'] = {**estimator_runtime(model), 'final_fit_seconds': perf_counter() - fit_started,
                               'holdout_inference_seconds': None}
            continue
        fit_seconds = perf_counter() - fit_started
        started = perf_counter()
        probabilities = model.predict_proba(data.rows[split:])
        item['runtime'] = {**estimator_runtime(model), 'final_fit_seconds': fit_seconds,
                           'holdout_inference_seconds': perf_counter() - started}
        targets = data.targets[split:]
        item.update(holdout_rows=len(targets), holdout_accuracy=accuracy_score(targets, [max(range(3), key=p.__getitem__) for p in probabilities]),
                    holdout_log_loss=multiclass_log_loss(targets, probabilities),
                    holdout_brier_score=multiclass_brier_score(targets, probabilities), holdout_rps=ranked_probability_score(targets, probabilities),
                    mapping_fit_interval={'first': times[0], 'last': times[split - 1]},
                    rating_state_policy='advance_after_earlier_kickoff_results;mapping_frozen_before_holdout')
        if prediction_sink is not None:
            prediction_sink({'model': item['model'], 'match_ids': data.match_ids[split:], 'kickoffs': times[split:],
                             'targets': targets, 'probabilities': probabilities})
    return {'seed': seed, 'counts': counts, 'variants': list(variants), 'warmup_matches_per_team': warmup_matches,
            'warmup_excluded': len(records) - len(eligible), 'selection_rows': split, 'holdout_rows': len(eligible) - split,
            'selected': selected, 'selected_on': 'development_expanding_window_log_loss',
            'development_last_kickoff': times[split - 1], 'holdout_first_kickoff': times[split], 'holdout_last_kickoff': times[-1],
            'parameter_policy': 'fixed_small_local_grid_selected_on_past_development_only;not_paper_parameter_reproduction',
            'definitions': {'basic': {'initial': 1500., 'k': 20., 'home_offset': 80., 'scale': 400.},
                            'goal_weighted': 'K=k*(1+abs(observed_goal_difference))**power;home_offset80;scale400',
                            'pi': 'two_venue_states_start0;c3;signed(10**(abs(r)/3)-1);error=observed-predicted;played_venue_step=lambda*sign(error)*3*log10(1+abs(error));opposite_step=gamma*played_step'},
            'mapping_policy': 'independent_local_fitted_probability_mapping;not_the_original_28_bin_pi_mapping',
            'cohort_policy': 'all_configurations_share_the_same_pre_kickoff_warmup_eligibility_and_time_partitions',
            'limits': ['No neutral-venue flag is available; home/away recipes assume the supplied venue labels.',
                       'Kickoff order does not prove final-whistle or source-arrival availability.',
                       'Ratings describe this connected competition pool; no cross-league comparability claim.',
                       'Warmup is completed matches per team, not the papers two/five season protocols; no historical return replication.'],
            'results': candidates}
