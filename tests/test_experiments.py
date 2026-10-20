import math
import importlib.util
import unittest
from unittest.mock import Mock, patch

from tests.support import requires_modules


def without_elapsed_times(value):
    """Compare deterministic outcomes without asserting identical wall clocks."""
    if isinstance(value, dict):
        return {key: without_elapsed_times(item) for key, item in value.items()
                if not key.endswith('_seconds')}
    if isinstance(value, (list, tuple)):
        return [without_elapsed_times(item) for item in value]
    return value


class TestPairedOutcomeComparison(unittest.TestCase):
    @staticmethod
    def forecasts():
        from datetime import UTC, datetime, timedelta
        import numpy as np

        labels, times, probabilities = [], [], []
        for day in range(32):
            for match in range(1 + day % 5):
                label = (day + match) % 3
                correct = .8 if day < 16 else .15
                row = [(1 - correct) / 2] * 3; row[label] = correct
                labels.append(label); probabilities.append(row)
                times.append((datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=day, hours=match)).isoformat())
        n = len(labels)
        return dict(targets=np.array(labels), match_ids=np.arange(1, n + 1), kickoffs=np.array(times),
                    probabilities=np.array([np.full((n, 3), 1 / 3), probabilities]),
                    model_names=np.array(['reference', 'candidate']))

    def test_weighted_losses_and_hac_match_dense_covariance_with_serial_sensitivity(self):
        import numpy as np
        from app.ml.comparison import paired_outcome_comparison

        data = self.forecasts()
        report = paired_outcome_comparison(**data, reference='reference')
        expected = []
        for label, first, second in zip(data['targets'], *data['probabilities']):
            a, b = list(first), list(second)
            expected.append([math.log(a[label] / b[label]),
                             sum((b[k] - (label == k)) ** 2 - (a[k] - (label == k)) ** 2 for k in range(3)),
                             sum((sum(b[:k + 1]) - (label <= k)) ** 2 - (sum(a[:k + 1]) - (label <= k)) ** 2 for k in range(2)) / 2])
        values = np.array(expected); days = [x[:10] for x in data['kickoffs']]
        unique = sorted(set(days)); counts = np.array([days.count(day) for day in unique])
        day_totals = np.array([values[np.array(days) == day].sum(axis=0) for day in unique])
        for k, row in enumerate(report['comparisons'][0]['metrics'].values()):
            self.assertAlmostEqual(row['difference'], values[:, k].mean(), places=14)
            self.assertGreater(abs(row['difference'] - (day_totals[:, k] / counts).mean()), .001)
            influence = day_totals[:, k] - counts * values[:, k].mean()
            for entry in row['day_influence_acf']:
                lag = entry['lag']
                product = influence @ influence if lag == 0 else influence[lag:] @ influence[:-lag]
                self.assertAlmostEqual(entry['correlation'], product / (influence @ influence), places=14)
            for setting in row['hac']:
                distance = np.abs(np.arange(32)[:, None] - np.arange(32)[None, :])
                kernel = np.maximum(1 - distance / (setting['lag'] + 1), 0)
                se = math.sqrt(float(influence @ kernel @ influence)) / len(values)
                self.assertAlmostEqual(se, setting['standard_error'], places=14)
            short, _, long = row['stationary_bootstrap']
            self.assertGreater(long['upper'] - long['lower'], 1.4 * (short['upper'] - short['lower']))
        data['probabilities'][1] = data['probabilities'][0]
        zero = paired_outcome_comparison(**data, reference='reference')
        self.assertTrue(all(entry['correlation'] is None for row in zero['comparisons'][0]['metrics'].values()
                            for entry in row['day_influence_acf']))
        for i,(label,time) in enumerate(zip(data['targets'],data['kickoffs'])):
            day=unique.index(time[:10]); correct=.8 if day%2 else .2
            data['probabilities'][1,i]=[(correct if k==label else (1-correct)/2) for k in range(3)]
        alternating=paired_outcome_comparison(**data,reference='reference')
        self.assertTrue(all(row['day_influence_acf'][1]['correlation']<-.7
                            for row in alternating['comparisons'][0]['metrics'].values()))

    def test_canonical_pairing_timezone_invariance_and_invalid_inputs(self):
        from datetime import datetime, timedelta, timezone
        import numpy as np
        from app.ml.comparison import paired_outcome_comparison

        data = self.forecasts(); options = dict(reference='reference', replicates=200)
        first = paired_outcome_comparison(**data, **options)
        order = np.random.default_rng(9).permutation(len(data['targets']))
        other = {k: v.copy() for k, v in data.items()}
        for key in ('targets', 'match_ids', 'kickoffs'): other[key] = other[key][order]
        other['probabilities'] = other['probabilities'][:, order]
        # Equivalent +02 times must still form the same UTC days and canonical receipt.
        other['kickoffs'] = np.array([datetime.fromisoformat(s).astimezone(timezone(timedelta(hours=2))).isoformat() for s in other['kickoffs']])
        self.assertEqual(first, paired_outcome_comparison(**other, **options))
        for field, change in [('match_ids', lambda a: a.__setitem__(0, a[1])),
                              ('targets', lambda a: a.__setitem__(0, 3)),
                              ('probabilities', lambda a: a.__setitem__((0, 0, 0), -.1)),
                              ('kickoffs', lambda a: a.__setitem__(0, '2024-01-01T00:00:00'))]:
            invalid = {k: v.copy() for k, v in data.items()}; change(invalid[field])
            with self.assertRaises(ValueError): paired_outcome_comparison(**invalid, **options)

    def test_real_cli_numeric_file_and_zero_probability_boundaries(self):
        import json
        import tempfile
        from pathlib import Path
        import numpy as np
        from app.research import main

        data = self.forecasts(); data['probabilities'][1, 0] = [0., 0., 1.]
        # Cached float32 forecasts can be upcast into a shared float64 container.
        data['probabilities'] = data['probabilities'].astype('float32').astype('float64')
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / 'paired.npz', Path(folder) / 'report.json'
            np.savez_compressed(source, **data)
            self.assertEqual(main(['compare-outcomes', '--predictions', str(source), '--reference', 'reference',
                                   '--output', str(target), '--replicates', '200']), 0)
            report = json.loads(target.read_text())
            self.assertEqual(report['matches'], len(data['targets']))
            self.assertTrue(math.isfinite(report['scores'][1]['log_loss']))
            self.assertGreater(report['max_input_row_mass_error'], 1e-10)
            from app.ml.model import multiclass_log_loss
            self.assertAlmostEqual(report['scores'][0]['log_loss'], multiclass_log_loss(data['targets'].tolist(), data['probabilities'][0].tolist()), places=14)
            self.assertFalse({'match_ids', 'targets', 'kickoffs', 'probabilities'} & report.keys())
            original = target.read_bytes()
            data['targets'][0] = 4; np.savez_compressed(source, **data)
            with self.assertRaises(ValueError):
                main(['compare-outcomes', '--predictions', str(source), '--reference', 'reference', '--output', str(target)])
            self.assertEqual(original, target.read_bytes())


class TestScoreModels(unittest.TestCase):
    def test_shared_component_law_gradients_marginals_and_boundary_fits(self):
        from dataclasses import replace
        import numpy as np
        from scipy.stats import poisson
        from app.ml.score_models import FootballScoreModel, _bivariate_goal_objective, _bivariate_log_probabilities, _goal_count_diagnostics

        home, away = np.array([0, 0, 1, 1, 2, 3, 2, 3]), np.array([1, 2, 2, 3, 3, 0, 0, 1])
        x, y = np.array([0, 0, 1, 1, 2, 3, 0, 2]), np.array([0, 1, 0, 1, 3, 0, 2, 1])
        weights = np.arange(1, 9, dtype=float); weights /= weights.sum()
        for shared in (0., .4):
            theta = np.r_[.2, .3, np.linspace(-.2, .2, 8), shared]
            objective = lambda t: _bivariate_goal_objective(t, home, away, x, y, weights, 4, .01)
            value, gradient = objective(theta)
            numerical = []
            for k in range(len(theta)):
                delta = np.zeros(len(theta)); delta[k] = 1e-6
                numerical.append((objective(theta + delta)[0] - value) / 1e-6 if k == len(theta) - 1 and shared == 0
                                 else (objective(theta + delta)[0] - objective(theta - delta)[0]) / 2e-6)
            np.testing.assert_allclose(gradient, numerical, atol=3e-6, rtol=3e-5)
            h, a = 1.3, .8
            grid = np.array([[sum(poisson.pmf(k, shared) * poisson.pmf(i-k, h) * poisson.pmf(j-k, a)
                                  for k in range(min(i, j) + 1)) for j in range(18)] for i in range(18)])
            xx, yy = np.indices(grid.shape)
            actual = np.exp(_bivariate_log_probabilities(xx, yy, h, a, shared)[0])
            np.testing.assert_allclose(actual, grid, atol=1e-15, rtol=2e-13)
            from decimal import Decimal, localcontext
            with localcontext() as context:
                context.prec=60
                dh,da,dc=Decimal('1.3'),Decimal('.8'),Decimal(str(shared))
                for i,j in ((0,0),(1,1),(3,2),(20,18)):
                    exact=(-(dh+da+dc)).exp()*sum(dh**(i-k)*da**(j-k)*(Decimal(1) if k==0 else dc**k)
                        /Decimal(math.factorial(i-k)*math.factorial(j-k)*math.factorial(k)) for k in range(min(i,j)+1))
                    logp=_bivariate_log_probabilities(np.array([i]),np.array([j]),dh.__float__(),da.__float__(),shared)[0][0]
                    self.assertAlmostEqual(logp,float(exact.ln()),places=12)
            np.testing.assert_allclose(grid.sum(axis=1), poisson.pmf(np.arange(18), h+shared), atol=1e-13)
            np.testing.assert_allclose(grid.sum(axis=0), poisson.pmf(np.arange(18), a+shared), atol=1e-13)
            model = FootballScoreModel('bivariate_poisson').fit(self.matches())
            model.theta_[:] = 0.; model.theta_[0] = math.log(a); model.theta_[1] = math.log(h/a); model.theta_[-1] = shared
            np.testing.assert_allclose(model.predict_proba(self.matches()[:1])[0],
                                       [np.triu(grid,1).sum(), np.trace(grid), np.tril(grid,-1).sum()], atol=1e-13)
            cohort = self.matches()[:8]
            diagnostics = _goal_count_diagnostics(cohort, [(h+shared,a+shared)]*8, 0., shared)
            np.testing.assert_allclose([r['expected_count'] for r in diagnostics['low_scores']],
                                       [8*grid[i,j] for i,j in ((0,0),(0,1),(1,0),(1,1))] + [8*(1-grid[:2,:2].sum())], atol=1e-13)
            for side, rate in (('home',h+shared),('away',a+shared)):
                np.testing.assert_allclose([r['expected_count'] for r in diagnostics['marginals'][side]],
                                           list(8*poisson.pmf(np.arange(4),rate))+[8*poisson.sf(3,rate)], atol=1e-13)
        cohort = self.matches()
        constant = FootballScoreModel('poisson_constant').fit(cohort)
        np.testing.assert_allclose(constant.expected_goals(1,2),
                                   [np.mean([r.home_score for r in cohort]),np.mean([r.away_score for r in cohort])], rtol=1e-5)
        np.testing.assert_array_equal(constant.predict_proba(cohort), constant.predict_proba([replace(r,home_team_id=101,away_team_id=202) for r in cohort]))
        boundary = FootballScoreModel('poisson_constant').fit([replace(r,home_score=200,away_score=0) for r in cohort])
        np.testing.assert_allclose(boundary.theta_[:2], [2.,1.], atol=1e-7)
        # Strong positive covariance exercises an interior shared-count optimum.
        rng = np.random.default_rng(192)
        correlated = []
        for r in cohort:
            common = int(rng.poisson(1.2))
            correlated.append(replace(r,home_score=int(rng.poisson(.6))+common,away_score=int(rng.poisson(.5))+common))
        fitted = FootballScoreModel('bivariate_poisson').fit(correlated)
        self.assertGreater(fitted._shared_rate(), .3)
        summary=fitted.summary()
        np.testing.assert_allclose(fitted.expected_goals(999,1000),
                                   [math.exp(summary['intercept']+summary['home_log_advantage'])+fitted._shared_rate(),
                                    math.exp(summary['intercept'])+fitted._shared_rate()])
        independent = FootballScoreModel('bivariate_poisson').fit([replace(r,home_score=i%2,away_score=(i+1)%2) for i,r in enumerate(cohort)])
        self.assertAlmostEqual(independent._shared_rate(), 0., places=7)

    @staticmethod
    def matches():
        from datetime import UTC, datetime, timedelta
        import numpy as np
        from app.ml.feature_engineering import MatchRecord

        rng = np.random.default_rng(137)
        rows = []
        for i in range(240):
            home, away = i % 8 + 1, (3 * i + 1) % 8 + 1
            home_rate = math.exp(.2 + .25 + .08 * (home - away))
            away_rate = math.exp(.2 + .08 * (away - home))
            rows.append(MatchRecord(i + 1, datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i // 4),
                                    'FINISHED', home, away, int(rng.poisson(home_rate)), int(rng.poisson(away_rate))))
        return rows

    def test_joint_score_likelihood_gradient_matches_independent_finite_differences(self):
        import numpy as np
        from app.ml.score_models import _goal_objective

        rng = np.random.default_rng(18)
        home, away = np.array([0, 0, 1, 1, 2, 3, 2, 3]), np.array([1, 2, 2, 3, 3, 0, 0, 1])
        x, y = np.array([0, 0, 1, 1, 2, 3, 0, 2]), np.array([0, 1, 0, 1, 3, 0, 2, 1])
        weights = np.arange(1, 9, dtype=float); weights /= weights.sum()
        for correlated in (False, True):
            for intercept in (-1.2, .3):
                for home_advantage in (-.4, .5):
                    theta = rng.normal(0, .2, 10 + correlated)
                    theta[:2] = [intercept, home_advantage]
                    objective = lambda p: _goal_objective(p, home, away, x, y, weights, 4, .01, correlated)
                    _, analytic = objective(theta)
                    numeric = []
                    for k in range(len(theta)):
                        delta = np.zeros(len(theta)); delta[k] = 1e-6
                        numeric.append((objective(theta + delta)[0] - objective(theta - delta)[0]) / 2e-6)
                    np.testing.assert_allclose(analytic, numeric, rtol=3e-6, atol=3e-7)

    def test_score_mass_marginals_unseen_teams_and_serialized_models(self):
        import pickle
        from dataclasses import replace
        import numpy as np
        from scipy.stats import poisson
        from app.ml.score_models import FootballScoreModel, ScoreModelError

        matches = self.matches()
        for method in ('poisson', 'dixon_coles'):
            model = FootballScoreModel(method, half_life_days=30).fit(matches)
            restored = pickle.loads(pickle.dumps(model))
            fixtures = [replace(matches[0], home_team_id=h, away_team_id=a, status='SCHEDULED', home_score=None, away_score=None)
                        for h, a in ((1, 2), (2, 1), (3, 6), (9, 1), (9, 10))]
            actual = np.asarray(restored.predict_proba(fixtures))
            np.testing.assert_array_equal(actual, model.predict_proba(fixtures))
            summary = model.summary()
            self.assertAlmostEqual(summary['attack_sum'], 0., places=12)
            self.assertAlmostEqual(summary['defence_sum'], 0., places=12)
            for fixture, probabilities in zip(fixtures, actual):
                home, away = model.expected_goals(fixture.home_team_id, fixture.away_team_id)
                size = int(max(poisson.ppf(1 - 1e-13, home), poisson.ppf(1 - 1e-13, away))) + 1
                ph, pa = poisson.pmf(np.arange(size), home), poisson.pmf(np.arange(size), away)
                grid = np.outer(ph, pa); rho = summary['rho']
                grid[0, 0] *= 1 - home * away * rho
                grid[0, 1] *= 1 + home * rho
                grid[1, 0] *= 1 + away * rho
                grid[1, 1] *= 1 - rho
                self.assertGreaterEqual(grid.min(), 0.)
                np.testing.assert_allclose(grid.sum(axis=1), ph, atol=1e-12, rtol=0)
                np.testing.assert_allclose(grid.sum(axis=0), pa, atol=1e-12, rtol=0)
                np.testing.assert_allclose(probabilities, [np.triu(grid, 1).sum(), np.trace(grid), np.tril(grid, -1).sum()], atol=1e-12, rtol=0)
                self.assertAlmostEqual(sum(probabilities), 1., places=12)
                for h, a in ((0, 0), (0, 1), (1, 0), (1, 1), (3, 2)):
                    self.assertAlmostEqual(math.exp(model.score_log_probability(fixture.home_team_id, fixture.away_team_id, h, a)), grid[h, a], places=13)
            self.assertEqual(len(model.team_index_), 8)
            self.assertAlmostEqual(model.expected_goals(9, 10)[0], math.exp(summary['intercept'] + summary['home_log_advantage']))
            self.assertAlmostEqual(model.expected_goals(9, 10)[1], math.exp(summary['intercept']))
            self.assertTrue(math.isfinite(model.score_log_probability(9, 10, 1000, 1000)))
            with self.assertRaises(ValueError): model.fit([])
            with self.assertRaises(ScoreModelError): model.predict_proba(fixtures)

    def test_score_backtest_isolates_holdout_scores_and_preserves_failed_selected_fit(self):
        from dataclasses import replace
        from app.ml.score_models import FootballScoreModel, ScoreModelError, run_score_experiments

        matches = self.matches()
        models = ('poisson', 'poisson_decay', 'dixon_coles', 'poisson_constant', 'bivariate_poisson')
        report = run_score_experiments(matches, models=models, half_life_days=30, include_predictions=True)
        split = report['development_rows']
        changed = matches[:split] + [replace(r, home_score=(r.home_score + 1) % 4, away_score=(r.away_score + 2) % 3) for r in matches[split:]]
        second = run_score_experiments(changed, models=models, half_life_days=30, include_predictions=True)
        self.assertEqual(report['selected'], second['selected'])
        for name, first in report['private_predictions'].items():
            other = second['private_predictions'][name]
            self.assertEqual(first['probabilities'], other['probabilities'])
            self.assertEqual(first['theta'], other['theta'])
            self.assertEqual(first['team_ids'], other['team_ids'])
            self.assertNotEqual(first['targets'], other['targets'])
        self.assertTrue(any(a['holdout_metrics']['log_loss'] != b['holdout_metrics']['log_loss'] for a, b in zip(report['results'], second['results'])))
        for first, other in zip(report['results'], second['results']):
            self.assertEqual(first['folds'], other['folds'])
            self.assertEqual(first['selection_log_loss'], other['selection_log_loss'])
            self.assertEqual(first['final_fit'], other['final_fit'])
            for fold in first['folds']:
                self.assertLess(fold['training_last_kickoff'], fold['validation_first_kickoff'])
                self.assertEqual(fold['training_rows'] % 4, 0)
                self.assertEqual(fold['validation_rows'] % 4, 0)
        original_fit = FootballScoreModel.fit
        selected = report['selected']

        def fail_final(model, cohort):
            key = model.method if model.method != 'poisson' else 'poisson_decay' if model.half_life_days else 'poisson'
            if len(cohort) == split and key == selected:
                model.is_fitted_ = False
                raise ScoreModelError('Unavailable final fit')
            return original_fit(model, cohort)

        with patch.object(FootballScoreModel, 'fit', fail_final):
            failed = run_score_experiments(matches, models=models, half_life_days=30)
        self.assertEqual(failed['selected'], selected)
        final = next(c for c in failed['results'] if c['model'] == selected)
        self.assertIn('holdout_error', final)
        self.assertNotIn('holdout_metrics', final)
        for method in ('dixon_coles','bivariate_poisson'):
            with self.assertRaises(ScoreModelError): FootballScoreModel(method, max_iter=1).fit(matches)




class TestOptionalDependencyRouting(unittest.TestCase):
    def test_missing_search_and_stream_packages_do_not_hide_numeric_checks(self):
        import runpy
        from pathlib import Path

        find_spec = importlib.util.find_spec
        def available(name, *args, **kwargs):
            return None if name in {"skopt", "optuna", "fpstreams", "matplotlib"} else find_spec(name, *args, **kwargs)
        with patch("importlib.util.find_spec", side_effect=available):
            definitions = runpy.run_path(str(Path(__file__)))
        for name in ["TestScoreModels", "TestCalibration", "TestProbabilityDiagnostics", "TestRatings"]:
            self.assertFalse(getattr(definitions[name], "__unittest_skip__", False), name)
        self.assertFalse(getattr(definitions["TestCalibration"].test_odir_objective_gradient_independent_optimizer_and_degenerate_inputs, "__unittest_skip__", False))
        for group, methods in {
            'TestFeatureSearch': ['test_joint_search_uses_real_optimizers_with_distinct_model_and_feature_parameters'],
            'TestExplanations': ['test_fixed_output_shap_closes_independently_and_exports_exact_values'],
            'TestModelSearch': ['test_search_records_nonconverged_trials_without_selecting_them',
                                'test_all_nonconverged_searches_fail_with_the_complete_trial_history'],
        }.items():
            for method in methods:
                self.assertFalse(getattr(getattr(definitions[group], method), '__unittest_skip__', False), method)
        self.assertTrue(definitions["TestStatistics"].__unittest_skip__)


class TestCalibration(unittest.TestCase):
    def test_odir_objective_gradient_independent_optimizer_and_degenerate_inputs(self):
        import numpy as np
        from scipy.optimize import minimize, approx_fprime
        from scipy.special import softmax
        from app.ml.calibration import ProbabilityCalibrator, CalibrationError

        rng = np.random.default_rng(31)
        p = rng.dirichlet([1., 2., 1.], size=180)
        y = np.array([rng.choice(3,p=row) for row in softmax(np.log(p) @ np.array([[.8,.2,-.1],[.1,1.4,.2],[-.2,.1,.7]]).T,axis=1)])
        x = np.log(p); mask = 1-np.eye(3)
        def independent(theta):
            weights,bias = theta[:9].reshape(3,3),theta[9:]
            probabilities = softmax(x @ weights.T + bias,axis=1)
            return (-np.log(probabilities[np.arange(len(y)),y]).mean()
                    + .001/6*np.sum((weights*mask)**2)+.001/3*np.sum(bias**2))
        captured = []
        def inspect(fun, initial, **kwargs):
            for theta in (initial, rng.normal(0,.4,size=12)):
                value,gradient = fun(theta)
                self.assertAlmostEqual(value,independent(theta),places=13)
                np.testing.assert_allclose(gradient,approx_fprime(theta,independent,1e-7),atol=3e-7,rtol=3e-5)
            captured.append(fun)
            return minimize(fun,initial,**kwargs)
        with patch('scipy.optimize.minimize',side_effect=inspect):
            fitted = ProbabilityCalibrator('dirichlet_odir').fit(p.tolist(),y.tolist())
        alternative = minimize(independent,np.r_[np.eye(3).ravel(),np.zeros(3)],method='BFGS',options={'gtol':1e-7,'maxiter':1500})
        theta = np.r_[fitted.odir_coefficients_.ravel(),fitted.odir_intercept_]
        self.assertLessEqual(np.max(np.abs(captured[0](theta)[1])),1e-6)
        self.assertAlmostEqual(independent(theta),alternative.fun,places=9)
        np.testing.assert_allclose(fitted.predict_proba(p),softmax(x @ alternative.x[:9].reshape(3,3).T+alternative.x[9:],axis=1),atol=3e-5)
        with self.assertRaises(CalibrationError): fitted.fit(p[:20].tolist(),[0]*20)
        self.assertFalse(fitted.is_fitted_)
        flat = ProbabilityCalibrator('dirichlet_odir').fit([[1/3]*3]*30,[0,1,2]*10)
        np.testing.assert_allclose(flat.predict_proba([[1/3]*3]),[[1/3]*3],atol=1e-12)

    def test_calibrator_recovers_known_temperature_and_invalidates_failed_refit(self):
        import pickle
        from app.ml.calibration import CalibrationError, ProbabilityCalibrator
        from app.ml.model import multiclass_log_loss

        truth = [0.55, 0.30, 0.15]
        raw = [value ** 2 / sum(p ** 2 for p in truth) for value in truth]
        labels = [0] * 110 + [1] * 60 + [2] * 30
        probabilities = [raw] * len(labels)
        scaler = ProbabilityCalibrator('temperature').fit(probabilities, labels)
        self.assertAlmostEqual(scaler.temperature_, 2.0, places=5)
        calibrated = scaler.predict_proba(probabilities)
        for actual, expected in zip(calibrated[0], truth):
            self.assertAlmostEqual(actual, expected, places=8)
        self.assertLess(multiclass_log_loss(labels, calibrated), multiclass_log_loss(labels, probabilities))
        self.assertEqual([max(range(3), key=lambda i: p[i]) for p in calibrated], [0] * len(labels))
        for method in ('temperature', 'dirichlet_l2', 'draw_constant', 'draw_logistic', 'draw_beta'):
            fitted = ProbabilityCalibrator(method).fit(probabilities, labels)
            restored = pickle.loads(pickle.dumps(fitted))
            edges = [[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]]
            self.assertEqual(restored.predict_proba(edges), fitted.predict_proba(edges))
            self.assertTrue(all(math.isfinite(v) for row in restored.predict_proba(edges) for v in row))
        with self.assertRaises(CalibrationError):
            fitted.fit(probabilities, [0] * len(labels))
        self.assertFalse(fitted.is_fitted_)
        with self.assertRaises(CalibrationError):
            fitted.predict_proba(probabilities)

    @staticmethod
    def calibration_dataset():
        from datetime import date, timedelta
        from app.ml.feature_engineering import FeatureDataset

        count = 180
        return FeatureDataset(
            rows=tuple({'x': float((i * 7) % 11), 'row_id': i} for i in range(count)),
            targets=tuple((i + i // 11) % 3 for i in range(count)),
            match_ids=tuple(range(count)), team_states={}, feature_defaults={},
            kickoffs=tuple((date(2024, 1, 1) + timedelta(days=i // 4)).isoformat() for i in range(count)),
        )

    @requires_modules('sklearn')
    def test_nested_calibration_keeps_kickoff_groups_and_holdout_labels_out_of_selection(self):
        from dataclasses import replace
        from app.ml.calibration import CALIBRATORS, run_calibration_experiment

        dataset = self.calibration_dataset()
        first = run_calibration_experiment(dataset, ('x',), model='logistic_l2', calibrators=CALIBRATORS, include_predictions=True)
        split = first['development_rows']
        changed = replace(dataset, targets=dataset.targets[:split] + tuple((y + 1) % 3 for y in dataset.targets[split:]))
        second = run_calibration_experiment(changed, ('x',), model='logistic_l2', calibrators=CALIBRATORS, include_predictions=True)
        self.assertEqual(first['folds'], second['folds'])
        self.assertEqual(first['selected'], second['selected'])
        self.assertEqual(first['private_predictions']['probabilities'], second['private_predictions']['probabilities'])
        self.assertTrue(any(a['holdout_metrics']['log_loss'] != b['holdout_metrics']['log_loss']
                            for a, b in zip(first['results'], second['results'])))
        for window in [*first['folds'], first['final_windows']]:
            self.assertLess(window['training_last_kickoff'], window['calibration_first_kickoff'])
            self.assertLess(window['calibration_last_kickoff'], window['evaluation_first_kickoff'])
            self.assertEqual(window['training_rows'] + window['calibration_rows'] + window['evaluation_rows'],
                             dataset.kickoffs.index(window['evaluation_last_kickoff']) + 4)
        for a, b in zip(first['results'], second['results']):
            self.assertEqual(a['selection_log_loss'], b['selection_log_loss'])
            self.assertEqual(a['final_fit']['calibrator'], b['final_fit']['calibrator'])
        for bad in (replace(dataset, kickoffs=()), replace(dataset, kickoffs=tuple(reversed(dataset.kickoffs)))):
            with self.assertRaises(ValueError):
                run_calibration_experiment(bad, ('x',), model='logistic_l2')

    def test_draw_beta_recovers_duplicated_evidence_distortion_with_monotone_coupling(self):
        import numpy as np
        from app.ml.calibration import ProbabilityCalibrator, _metrics

        probabilities, targets = [], []
        # Three identical independent-feature assumptions multiply the true odds cubically.
        for positives in range(10, 100, 10):
            truth = positives / 100
            score = truth**3 / (truth**3 + (1 - truth)**3)
            probabilities.extend([[.25 * (1 - score), score, .75 * (1 - score)]] * 100)
            targets.extend([1] * positives + [0] * ((100 - positives) // 2)
                           + [2] * ((100 - positives) // 2))
        beta = ProbabilityCalibrator('draw_beta').fit(probabilities, targets)
        logistic = ProbabilityCalibrator('draw_logistic').fit(probabilities, targets)
        self.assertTrue(np.allclose(beta.draw_parameters_[:2], [1/3, 1/3], atol=.02))
        transformed = beta.predict_proba(probabilities)
        self.assertLess(_metrics(targets, transformed)['draw_metrics']['log_loss'],
                        _metrics(targets, logistic.predict_proba(probabilities))['draw_metrics']['log_loss'])
        for raw, row in zip(probabilities, transformed):
            self.assertAlmostEqual(sum(row), 1)
            self.assertAlmostEqual(row[0] / row[2], raw[0] / raw[2])
        grid = [[.25 * (1-s), s, .75 * (1-s)] for s in np.linspace(0, 1, 101)]
        calibrated = np.array(beta.predict_proba(grid))
        self.assertTrue(np.all(np.diff(calibrated[:, 1]) >= 0))
        self.assertTrue(np.isfinite(calibrated).all())
        self.assertTrue(np.all(calibrated >= 0))
        self.assertTrue(np.allclose(calibrated.sum(axis=1), 1))
        swapped = beta.predict_proba([[row[2], row[1], row[0]] for row in grid])
        self.assertTrue(np.allclose(swapped, calibrated[:, ::-1]))
        self.assertEqual(calibrated[-1, 0], calibrated[-1, 2])  # Undefined residual split is symmetric.

    def test_draw_maps_recover_constant_prevalence_and_require_both_binary_outcomes(self):
        import numpy as np
        from app.ml.calibration import CalibrationError, ProbabilityCalibrator

        targets = [1] * 30 + [0] * 35 + [2] * 35
        raw = [[.25 * (1-s), s, .75 * (1-s)] for s in (.05,.2,.5,.8,.95) for _ in targets]
        for method in ('draw_constant', 'draw_logistic', 'draw_beta'):
            calibrator = ProbabilityCalibrator(method).fit(raw, targets * 5)
            self.assertTrue(np.allclose(np.array(calibrator.predict_proba(raw))[:, 1], .3, atol=1e-5))
            self.assertEqual(calibrator.predict_proba([]), [])
            # Swapping home and away labels must leave the learned binary map unchanged.
            changed = ProbabilityCalibrator(method).fit(raw, [2 if y == 0 else 0 if y == 2 else 1 for y in targets] * 5)
            self.assertTrue(np.array_equal(calibrator.draw_parameters_, changed.draw_parameters_))
            with self.assertRaises(CalibrationError):
                calibrator.fit(raw, [0] * len(raw))
            with self.assertRaises(CalibrationError):
                calibrator.predict_proba(raw)

        from app.ml.calibration import run_calibration_experiment
        class ConstantModel:
            def __init__(self, names): pass
            def fit(self, rows, labels): return self
            def predict_proba(self, rows): return [[.45, .1, .45] for _ in rows]
        with patch('app.ml.calibration.estimator_factory', return_value=ConstantModel):
            for methods in (('identity', 'draw_beta', 'draw_logistic', 'draw_constant'),
                            ('draw_constant', 'draw_logistic', 'draw_beta', 'identity')):
                result = run_calibration_experiment(self.calibration_dataset(), ('x',), calibrators=methods)
                self.assertEqual(result['selected'], 'draw_constant')
                self.assertEqual(result['selection_tolerance'], 1e-9)

    def test_draw_fit_uses_stationarity_instead_of_an_unreliable_solver_flag(self):
        import numpy as np
        from scipy.optimize import OptimizeResult, minimize
        from app.ml.calibration import CalibrationError, ProbabilityCalibrator

        scores = np.random.default_rng(54).beta(.2, .2, 10)
        probabilities = np.column_stack(((1-scores)*.25, scores, (1-scores)*.75)).tolist()
        targets = [0, 1] * 5
        def reported_failure(*args, **kwargs):
            result = minimize(*args, **kwargs)
            result.success, result.status, result.message = False, 2, 'ABNORMAL: line search'
            return result
        with patch('scipy.optimize.minimize', side_effect=reported_failure):
            fitted = ProbabilityCalibrator('draw_beta').fit(probabilities, targets)
        self.assertLess(fitted.summary()['max_projected_gradient'], 1e-6)
        self.assertFalse(fitted.summary()['optimizer']['reported_success'])
        self.assertEqual(fitted.summary()['optimizer']['status'], 2)
        fake_success = OptimizeResult(x=np.array([1., 1., 0.]), success=True, status=0, message='synthetic success')
        with patch('scipy.optimize.minimize', return_value=fake_success), self.assertRaises(CalibrationError):
            fitted.fit(probabilities, targets)
        self.assertFalse(fitted.is_fitted_)
        with self.assertRaises(CalibrationError): fitted.predict_proba(probabilities)

    def test_final_calibration_failure_does_not_reselect_using_holdout(self):
        from app.ml.calibration import CalibrationError, ProbabilityCalibrator, run_calibration_experiment

        class OverconfidentModel:
            def __init__(self, names): pass
            def fit(self, rows, labels): return self
            def predict_proba(self, rows): return [[0.96, 0.02, 0.02] for _ in rows]

        dataset = self.calibration_dataset()
        final_calibration_rows = 32
        original_fit = ProbabilityCalibrator.fit

        def bounded_fit(calibrator, probabilities, labels):
            if calibrator.method == 'temperature' and len(labels) == final_calibration_rows:
                raise CalibrationError('Final calibration fit is unavailable')
            if calibrator.method == 'dirichlet_l2':
                raise CalibrationError('Unavailable candidate')
            return original_fit(calibrator, probabilities, labels)

        with patch('app.ml.calibration.estimator_factory', return_value=OverconfidentModel), \
             patch.object(ProbabilityCalibrator, 'fit', bounded_fit):
            report = run_calibration_experiment(dataset, ('x',))
        self.assertEqual(report['final_windows']['calibration_rows'], final_calibration_rows)
        self.assertEqual(report['selected'], 'temperature')
        results = {row['method']: row for row in report['results']}
        self.assertEqual(results['temperature']['final_fit']['state'], 'failed')
        self.assertNotIn('holdout_metrics', results['temperature'])
        self.assertIn('holdout_metrics', results['identity'])
        self.assertIsNone(results['dirichlet_l2']['selection_log_loss'])


class TestProbabilityDiagnostics(unittest.TestCase):
    def test_brier_partition_identity_signed_remainders_and_display_independence(self):
        import numpy as np
        from app.ml.calibration_diagnostics import calibration_diagnostics, compact_diagnostics

        labels = np.array([0,2]*12)
        aligned = np.array([[.39,.33,.28],[.27,.33,.40]]*12)
        cases = [(labels,aligned),(labels,aligned[::-1]),(labels,np.full((24,3),1/3)),(labels,np.eye(3)[labels])]
        signs = []
        for y,p in cases:
            report = calibration_diagnostics(y,p,bins=(2,5,10),minimum_bin_count=2)
            hidden = calibration_diagnostics(y,p,bins=(2,5,10),minimum_bin_count=1000)
            for row, suppressed in zip(report['results'],hidden['results']):
                resolution = row['resolution']; indices = np.minimum(resolution-1,(p*resolution).astype(int))
                means = np.array([p[(indices==key).all(axis=1)].mean(axis=0) for key in indices])
                binned = ((means-np.eye(3)[y])**2).sum(axis=1).mean()
                d = row['joint_tv']['brier_decomposition']
                self.assertAlmostEqual(d['binned_forecast_brier'],binned,places=14)
                self.assertAlmostEqual(d['uncertainty']-d['resolution']+d['reliability'],binned,places=14)
                self.assertAlmostEqual(binned+d['within_bin_remainder'],d['raw_brier'],places=14)
                self.assertEqual(d,suppressed['joint_tv']['brier_decomposition'])
                perm = [2,0,1]
                transformed = calibration_diagnostics([perm.index(k) for k in y],p[:,perm],bins=(resolution,))
                for key in ('raw_brier','uncertainty','resolution','reliability','within_bin_remainder'):
                    self.assertAlmostEqual(d[key],transformed['results'][0]['joint_tv']['brier_decomposition'][key],places=14)
            signs.append(report['results'][0]['joint_tv']['brier_decomposition']['within_bin_remainder'])
            for row in report['results']: del row['joint_tv']['brier_decomposition']
            compact_diagnostics(report)  # Legacy reports remain consumable.
        self.assertLess(signs[0],0); self.assertGreater(signs[1],0)

    def test_multiclass_calibration_counterexample_reveals_joint_bias_hidden_by_marginals(self):
        import numpy as np
        from app.ml.calibration_diagnostics import calibration_diagnostics

        # Published Example1/Table1: all scalar views calibrated, joint TV=.1.
        predictions = [[.1,.3,.6],[.1,.6,.3],[.3,.1,.6],[.3,.6,.1],[.6,.1,.3],[.6,.3,.1]]
        counts = [[2,2,6],[0,7,3],[2,2,6],[4,5,1],[7,0,3],[5,4,1]]
        p, y = [], []
        for row, observed in zip(predictions, counts):
            for k, n in enumerate(observed):
                p.extend([row] * n); y.extend([k] * n)
        report = calibration_diagnostics(y, p)
        for row in report['results']:
            self.assertAlmostEqual(row['joint_tv']['ece'], .1, places=14)
            self.assertEqual(row['joint_tv']['occupied_bins'], 6)
            for strategy in ['equal_width', 'equal_mass']:
                self.assertAlmostEqual(row[strategy]['top_label']['ece'], 0., places=14)
                self.assertAlmostEqual(row[strategy]['classwise_ece'], 0., places=14)
        permutation = [2, 0, 1]
        transformed = calibration_diagnostics([permutation.index(k) for k in y], np.array(p)[:, permutation])
        for a, b in zip(report['results'], transformed['results']):
            self.assertAlmostEqual(a['joint_tv']['ece'], b['joint_tv']['ece'], places=14)
        hidden = calibration_diagnostics(y, p, minimum_bin_count=11)
        for a, b in zip(report['results'], hidden['results']):
            self.assertEqual(a['joint_tv']['ece'], b['joint_tv']['ece'])
            self.assertEqual(b['joint_tv']['cells'], [])
            self.assertEqual(b['joint_tv']['suppressed_rows'], 60)

    def test_diagnostic_refinement_ties_fixed_inputs_and_atomic_cli_rejection(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        import numpy as np
        from app.ml.calibration_diagnostics import calibration_diagnostics, _equal_mass_indices
        from app.ml.evaluation import classwise_reliability, top_label_calibration_error
        from app.research import main

        rng = np.random.default_rng(73)
        p = rng.dirichlet([2., 2., 2.], 240); original = p.copy()
        y = np.arange(240) % 3
        report = calibration_diagnostics(y, p)
        self.assertTrue(np.array_equal(p, original))
        self.assertTrue(all(a['joint_tv']['ece'] <= b['joint_tv']['ece'] + 1e-14
                            for a,b in zip(report['results'], report['results'][1:])))
        ten = report['results'][1]
        self.assertAlmostEqual(ten['equal_width']['top_label']['ece'], top_label_calibration_error(y.tolist(), p.tolist()), places=14)
        expected = sum(c['calibration_error'] for c in classwise_reliability(y.tolist(), p.tolist())) / 3
        self.assertAlmostEqual(ten['equal_width']['classwise_ece'], expected, places=14)
        values = np.array([.1] * 80 + [.4] * 20 + [.8] * 15)
        indices, effective = _equal_mass_indices(values, 10)
        self.assertEqual(effective, 3)
        self.assertEqual(len(set(indices[:80])), 1)
        self.assertEqual(len(set(indices[80:100])), 1)
        order = rng.permutation(len(values))
        self.assertTrue(np.array_equal(_equal_mass_indices(values[order], 10)[0], indices[order]))
        constant = _equal_mass_indices(np.full(20, .6), 10)
        self.assertEqual(constant[1], 1)
        self.assertTrue((constant[0] == 0).all())
        for labels, matrix, options in [(y, p, {'bins':(10,10)}), (y,p,{'minimum_bin_count':1}),
                                       (y.astype(bool),p,{}), (y,p*1.1,{})]:
            with self.assertRaises(ValueError): calibration_diagnostics(labels, matrix, **options)
        with TemporaryDirectory() as temp:
            source, output = Path(temp)/'fixed.npz', Path(temp)/'aggregate.json'
            np.savez(source,targets=y,probabilities=np.array([p]),model_names=np.array(['fixed']))
            self.assertEqual(main(['calibration-diagnostics','--predictions',str(source),'--output',str(output)]),0)
            saved = output.read_bytes()
            self.assertNotIn(b'match_ids',saved);self.assertNotIn(b'targets',saved)
            for matrices, names in [(np.array([p,p]), np.array(['duplicate','duplicate'])),
                                    (np.array([p]), np.array([b'']))]:
                np.savez(source,targets=y,probabilities=matrices,model_names=names)
                with self.assertRaises(ValueError): main(['calibration-diagnostics','--predictions',str(source),'--output',str(output)])
                self.assertEqual(output.read_bytes(),saved)


class TestRatings(unittest.TestCase):
    def test_rating_variants_match_hand_updates_and_isolate_future_outcomes(self):
        from dataclasses import replace
        from datetime import UTC, datetime, timedelta
        from app.ml.feature_engineering import MatchRecord
        from app.ml.rating_variants import rating_dataset, run_rating_variants

        first = MatchRecord(1, datetime(2024, 1, 1, tzinfo=UTC), 'FINISHED', 1, 2, 2, 0)
        second = replace(first, match_id=2, kickoff=first.kickoff + timedelta(days=1), home_team_id=2, away_team_id=1)
        state = {}
        pi = rating_dataset([first, second], kind='pi', learning_rate=.035, catchup=.7, state_sink=state.update)
        step = .035 * 3 * math.log10(3)
        expected = -2 * (10 ** (.7 * step / 3) - 1)
        self.assertAlmostEqual(pi.rows[1]['rating_diff'], expected, places=12)
        self.assertAlmostEqual(sum(sum(s) for s in state.values()), 0., places=12)
        goal = rating_dataset([first, second], kind='goal_weighted', k=20., home_offset=0., power=.5)
        self.assertAlmostEqual(goal.rows[1]['elo_diff'], -20 * math.sqrt(3), places=10)
        tied = replace(second, kickoff=first.kickoff)
        self.assertEqual([r['rating_diff'] for r in rating_dataset([first, tied], kind='pi').rows], [0., 0.])
        matches = TestScoreModels.matches()
        original = run_rating_variants(matches, variants=('goal_weighted', 'pi'), warmup_matches=3, folds=2)
        boundary = original['holdout_first_kickoff']
        changed = [replace(r, home_score=r.away_score + 4, away_score=0) if r.kickoff.isoformat() >= boundary else r for r in matches]
        revised = run_rating_variants(changed, variants=('goal_weighted', 'pi'), warmup_matches=3, folds=2)
        self.assertEqual(original['selected'], revised['selected'])
        self.assertGreater(original['warmup_excluded'], 0)
        self.assertLess(original['development_last_kickoff'], boundary)
        for before, after in zip(original['results'], revised['results']):
            self.assertEqual(before['selection_log_loss'], after['selection_log_loss'])
            self.assertEqual(before['rating_state_after_development'], after['rating_state_after_development'])
            self.assertEqual(without_elapsed_times(before.get('development_folds')), without_elapsed_times(after.get('development_folds')))
            if 'mapping_fit_interval' in before:
                self.assertLess(before['mapping_fit_interval']['last'], boundary)

    def test_ordered_elo_likelihood_gradient_recovery_extremes_and_invalidated_refit(self):
        import pickle
        import numpy as np
        from scipy.special import expit
        from app.ml.alternatives import ModelConvergenceError, estimator_factory
        from app.ml.model import ModelTrainingError, NotFittedError
        from app.ml.ordinal import EloOrderedLogit, _log_probabilities, _objective

        x = np.linspace(-3., 3., 27); targets = np.arange(27) % 3
        for theta in [np.array([1.4, -.7, .2]), np.array([-2., 3., -10.]), np.array([.1, -4., 6.])]:
            loss, gradient = _objective(theta, x, targets, .003)
            numerical = []
            for i in range(3):
                step = np.eye(3)[i] * 1e-5
                numerical.append((_objective(theta + step, x, targets, .003)[0]
                                 - _objective(theta - step, x, targets, .003)[0]) / 2e-5)
            np.testing.assert_allclose(gradient, numerical, atol=2e-7, rtol=2e-5)
            self.assertTrue(np.isfinite(loss))
        self.assertTrue(np.isfinite(_log_probabilities(np.array([3., -.5, -10.]), np.array([-100., 100.]))).all())
        rows, labels = [], []
        for value in np.linspace(-2., 2., 25):
            a, b = expit(-.6 - 1.3 * value), expit(.8 - 1.3 * value)
            counts = np.rint(np.array([a, b - a, 1 - b]) * 240).astype(int)
            counts[-1] += 240 - counts.sum()
            for label, count in enumerate(counts):
                rows.extend([{'elo_diff': float(value * 400)}] * int(count)); labels.extend([label] * int(count))
        model = estimator_factory('elo_ordered_logit', params={'ridge': 0.})(('elo_diff',)).fit(rows, labels)
        self.assertAlmostEqual(model.fit_diagnostics_['coefficient_per400elo'], 1.3, delta=.015)
        np.testing.assert_allclose(model.fit_diagnostics_['thresholds'], [-.6, .8], atol=.015)
        grid = [{'elo_diff': float(v)} for v in np.linspace(-800, 800, 51)]
        p = np.asarray(model.predict_proba(grid))
        self.assertTrue((np.diff(p[:, 0]) < 0).all())
        self.assertTrue((np.diff(p[:, 2]) > 0).all())
        extreme = [{'elo_diff': -1e308}, {'elo_diff': 1e308}]
        self.assertEqual(model.predict_proba(extreme), [[1., 0., 0.], [0., 0., 1.]])
        self.assertEqual(pickle.loads(pickle.dumps(model)).predict_proba(grid), p.tolist())
        legacy = pickle.loads(pickle.dumps(model))
        del legacy.rating_feature
        del legacy.unit_scale
        np.testing.assert_allclose(legacy.predict_proba(grid), p)
        with self.assertRaises(ModelTrainingError): model.fit(rows, [0] * len(rows))
        self.assertFalse(model.is_fitted_)
        with self.assertRaises(NotFittedError): model.predict_proba(grid)
        with self.assertRaises(ModelConvergenceError): EloOrderedLogit(('elo_diff',), max_iter=1).fit(rows, labels)

    def test_rating_controls_preserve_future_independence_identity_and_ordinal_search_scope(self):
        from dataclasses import replace
        from app.ml.alternatives import estimator_factory
        from app.ml.experiments import search_parameters
        from app.ml.ordinal import run_rating_experiments

        matches = TestScoreModels.matches(); forecasts = []
        def bounded_factory(method, *, seed=42, params=None):
            params = dict(params or {})
            if method == 'xgboost': params.update(rounds=3, early_stopping_rounds=2)
            return estimator_factory(method, seed=seed, params=params)
        with patch('app.ml.alternatives.estimator_factory', bounded_factory):
            first = run_rating_experiments(matches, folds=2, prediction_sink=forecasts.append)
            split = first['selection_rows']
            changed = matches[:split] + [replace(r, home_score=r.away_score + 4, away_score=0) for r in matches[split:]]
            second = run_rating_experiments(changed, folds=2)
        self.assertEqual(first['selected'], second['selected'])
        self.assertEqual(len(forecasts), 4)
        for a, b, saved in zip(first['results'], second['results'], forecasts):
            self.assertEqual(a['selection_log_loss'], b['selection_log_loss'])
            self.assertEqual(without_elapsed_times(a['development_folds']), without_elapsed_times(b['development_folds']))
            self.assertEqual(saved['match_ids'], tuple(r.match_id for r in matches[split:]))
            self.assertLess(first['development_last_kickoff'], first['holdout_first_kickoff'])
        self.assertEqual([len(c['feature_names']) for c in first['results']], [1, 1, 30, 30])
        self.assertNotIn('match_ids', str(first))
        proposals = []
        def objective(params):
            self.assertNotIn('feature_window', params)
            self.assertIn('feature_elo_k', params)
            self.assertIn('feature_home_advantage', params)
            proposals.append(params)
            return params['ridge']
        result = search_parameters('elo_ordered_logit', 'genetic', objective,
                                    trials=4, seed=7, include_features=True)
        self.assertEqual(result['completed_trials'], 4)
        self.assertEqual(len(proposals), 4)


class TestFeatureSearch(unittest.TestCase):
    @requires_modules('sklearn')
    def test_joint_features_preserve_causal_search_and_holdout_identity_under_future_score_changes(self):
        from dataclasses import replace
        from app.ml.feature_engineering import FeatureEngineer
        from app.ml.feature_experiments import feature_dataset, run_feature_experiments

        matches = TestScoreModels.matches()
        captured = []
        with patch('app.ml.feature_experiments.feature_dataset', wraps=feature_dataset) as builder:
            original = run_feature_experiments(matches, methods=('gaussian_nb',), optimizer='genetic',
                                               trials=4, folds=2, seed=19, prediction_sink=captured.append)
        split = original['selection_rows']
        lengths = [len(call.args[0]) for call in builder.call_args_list]
        self.assertEqual(lengths[-3:], [len(matches)] * 3)
        self.assertTrue(all(n == split for n in lengths[:-3]))
        changed = [r if i < split else replace(r, home_score=r.away_score + 3, away_score=0)
                   for i, r in enumerate(matches)]
        revised = run_feature_experiments(changed, methods=('gaussian_nb',), optimizer='genetic',
                                          trials=4, folds=2, seed=19)
        self.assertEqual(original['selected'], revised['selected'])
        for a, b in zip(original['results'], revised['results']):
            self.assertEqual(a['selection_log_loss'], b['selection_log_loss'])
            self.assertEqual(without_elapsed_times(a.get('search')), without_elapsed_times(b.get('search')))
        self.assertEqual(len(captured), 3)
        for item in captured:
            self.assertEqual(item['match_ids'], tuple(r.match_id for r in matches[split:]))
        self.assertNotEqual([r['holdout_log_loss'] for r in original['results']],
                            [r['holdout_log_loss'] for r in revised['results']])
        self.assertLess(original['development_last_kickoff'], original['holdout_first_kickoff'])
        for fold in original['folds']:
            self.assertLess(fold['training_last_kickoff'], fold['validation_first_kickoff'])
        self.assertNotIn('match_ids', str(original))
        self.assertEqual(feature_dataset(matches), FeatureEngineer().build_training_dataset(matches))
        features = dict(window=3, elo_k=37., home_advantage=15.)
        before = feature_dataset(matches, **features)
        after = feature_dataset(changed, **features)
        self.assertEqual(before.rows[:split + 1], after.rows[:split + 1])
        self.assertNotEqual(before.rows[-1], after.rows[-1])
        self.assertEqual(FeatureEngineer.ELO_K, 20.)
        import io
        import json
        import tempfile
        from contextlib import redirect_stderr
        from pathlib import Path
        from app.cli.arguments import build_research_parser
        from app.cli.research import feature_experiment

        conflict = [replace(matches[0], match_id=981234567),
                    replace(matches[0], match_id=981234567, home_score=matches[0].home_score + 1)]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'aggregate.json'
            args = build_research_parser().parse_args(['feature-experiment', '--source', 'football-csv',
                    '--competitions', 'PL', '--seasons', '2022', '--output', str(output)])
            diagnostics = io.StringIO()
            with patch('app.cli.model_research._match_research_source', return_value=(None, 'football-csv')), \
                 patch('app.cli.model_research._match_research_matches', return_value=(conflict, [])), redirect_stderr(diagnostics):
                self.assertEqual(feature_experiment(args), 1)
            self.assertEqual(json.loads(output.read_text())['errors'][0]['code'], 'ValueError')
            self.assertNotIn('981234567', output.read_text())
            self.assertIn('981234567', diagnostics.getvalue())

    def test_joint_search_uses_real_optimizers_with_distinct_model_and_feature_parameters(self):
        from app.ml.experiments import search_parameters
        from app.ml.feature_experiments import feature_dataset

        for optimizer, dependency in [('genetic', None), ('bayesian', 'skopt'), ('optuna', 'optuna')]:
            with self.subTest(optimizer=optimizer):
                if dependency and importlib.util.find_spec(dependency) is None:
                    self.skipTest(f'Optional optimizer requires {dependency}')
                calls = []
                def objective(params):
                    calls.append(params.copy())
                    self.assertIs(type(params['feature_window']), int)
                    self.assertTrue(2 <= params['feature_window'] <= 12)
                    self.assertTrue(5. <= params['feature_elo_k'] <= 50.)
                    self.assertTrue(0. <= params['feature_home_advantage'] <= 150.)
                    self.assertIn('var_smoothing', params)
                    return (params['feature_window'] - 5) ** 2 + params['var_smoothing']
                first = search_parameters('gaussian_nb', optimizer, objective, trials=4, seed=23, include_features=True)
                second = search_parameters('gaussian_nb', optimizer, objective, trials=4, seed=23, include_features=True)
                self.assertEqual(without_elapsed_times(first), without_elapsed_times(second))
                self.assertEqual(len(calls), 8)
                self.assertEqual(first['completed_trials'], 4)
        for options in [dict(window=True), dict(window=2.5), dict(elo_k=float('nan')),
                        dict(home_advantage=-1), dict(elo_k=True)]:
            with self.assertRaises(ValueError):
                feature_dataset([], **options)


class TestExplanations(unittest.TestCase):
    def test_fixed_output_shap_closes_independently_and_exports_exact_values(self):
        import csv
        import tempfile
        from pathlib import Path
        import numpy as np
        import xgboost as xgb
        from app.ml.explanations import explain_cohort
        from app.ml.model import XGBoostClassifier
        from app.analytics.visualization.explanations import figures

        features = ('elo_diff', 'form_points_diff')
        rows = [{'elo_diff': float(i - 30), 'form_points_diff': float(i % 3 - 1)} for i in range(72)]
        model = XGBoostClassifier(features, rounds=18, early_stopping_rounds=4).fit(rows[:60], [i % 3 for i in range(60)])
        probe = rows[60:68]; times = [f'2024-03-{i + 1:02d}T00:00:00+00:00' for i in range(len(probe))]
        matrix = xgb.DMatrix([[r[name] for name in features] for r in probe], feature_names=list(features))
        native = model._booster().predict(matrix, output_margin=True, strict_shape=True)
        interactions = model._booster().predict(matrix, pred_interactions=True, strict_shape=True)
        for fixed, against in ((2, None), (2, 0), (1, 2)):
            report = explain_cohort(model, probe, times, class_index=fixed, against=against)
            expected = native[:, fixed] if against is None else native[:, fixed] - native[:, against]
            total = np.asarray(report['baseline']) + sum(np.asarray(f['contributions']) for f in report['features'])
            np.testing.assert_allclose(report['raw_outputs'], expected, atol=1e-5)
            np.testing.assert_allclose(total, expected, atol=1e-5)
            pair = interactions[:, fixed, 0, 1] if against is None else interactions[:, fixed, 0, 1] - interactions[:, against, 0, 1]
            self.assertAlmostEqual(report['interaction_pairs'][0]['mean_absolute_pair_total'], float(np.mean(np.abs(2 * pair))), places=6)
        with self.subTest(part='figure and CSV export'):
            if importlib.util.find_spec('matplotlib') is None:
                self.skipTest('Optional rendering requires matplotlib')
            with tempfile.TemporaryDirectory() as folder:
                figures(report, Path(folder))
                with (Path(folder) / 'contributions.csv').open(encoding='utf-8-sig', newline='') as handle:
                    table = list(csv.DictReader(handle))
                self.assertEqual(len(table), len(probe) * len(features))
                self.assertTrue(all(r['model_sha256'] == report['identity']['model_sha256'] for r in table))
                for r in table:
                    feature = next(f for f in report['features'] if f['feature'] == r['feature'])
                    self.assertEqual(float(r['contribution']), feature['contributions'][int(r['row'])])
                for stem in ('global-contributions', 'feature-dependence', 'feature-interactions'):
                    svg = (Path(folder) / f'{stem}.svg').read_text()
                    self.assertIn(report['run_id'], svg)
                    self.assertIn('role="img"', svg)
        with self.assertRaises(ValueError):
            explain_cohort(model, probe * 30, times * 30)
        with self.assertRaises(ValueError):
            explain_cohort(model, probe, times, class_index=2, against=2)


@requires_modules("sklearn")
class TestAlternativeEstimators(unittest.TestCase):
    def test_logistic_nonconvergence_does_not_return_a_fitted_model(self):
        from app.ml.alternatives import SklearnClassifier
        from app.ml.model import ModelTrainingError

        rows = [{"x": float(i)} for i in range(90)]
        for method in ("logistic_l1", "logistic_l2", "logistic_elasticnet"):
            with self.subTest(method=method):
                model = SklearnClassifier(("x",), method=method)
                model.fit(rows, [i // 30 for i in range(90)])
                model.pipeline.steps[-1][1].set_params(max_iter=1)
                with self.assertRaisesRegex(ModelTrainingError, "收敛"):
                    model.fit(rows, [i // 30 for i in range(90)])
                self.assertFalse(model.is_fitted_)

    def test_fixed_logistic_penalties_reject_conflicting_mixing_parameters(self):
        from app.ml.alternatives import estimator_factory
        from app.ml.model import ModelTrainingError

        for method, ratio in (("logistic_l1", 0.0), ("logistic_l2", 1.0)):
            with self.subTest(method=method):
                for params in ({"l1_ratio": ratio}, {"penalty": "l2"}):
                    original = params.copy()
                    with self.assertRaisesRegex(ModelTrainingError, "l1_ratio"):
                        estimator_factory(method, params=params)(("x",))
                    self.assertEqual(params, original)

    def test_regularized_logistic_models_preserve_sparse_signal_and_training_scaling(self):
        import pickle
        import numpy as np
        from app.ml.alternatives import estimator_factory

        rng = np.random.default_rng(18)
        matrix = rng.normal(size=(300, 9))
        names = tuple(f"x{i}" for i in range(9))
        rows = [dict(zip(names, row)) for row in matrix]
        targets = np.digitize(matrix[:, 0], [-0.5, 0.5]).tolist()
        for method in ("logistic_l1", "logistic_l2", "logistic_elasticnet"):
            with self.subTest(method=method):
                model = estimator_factory(method, seed=17, params={"C": 0.1})(names)
                model.fit(rows, targets)
                self.assertGreater(sum(a == b for a, b in zip(model.predict(rows), targets)) / 300, 0.85)
                scaler = model.pipeline.steps[0][1]
                np.testing.assert_allclose(scaler.mean_, matrix.mean(axis=0))
                restored = pickle.loads(pickle.dumps(model))
                restored_scaler = restored.pipeline.steps[0][1]
                before = restored_scaler.mean_.copy()
                probabilities = restored.predict_proba(
                    [dict.fromkeys(names, 1000.0)]
                )[0]
                np.testing.assert_array_equal(restored_scaler.mean_, before)
                self.assertEqual(len(probabilities), 3)
                self.assertAlmostEqual(sum(probabilities), 1.0)
                self.assertTrue(all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities))
                if method == "logistic_l1":
                    coefficients = model.pipeline.steps[-1][1].coef_
                    self.assertTrue((coefficients[:, 1:] == 0).any())

    def test_user_parameters_override_estimator_defaults_and_allow_training(self):
        from app.ml.alternatives import estimator_factory

        rows = [{"x": float(i)} for i in range(30)]
        targets = [i // 10 for i in range(30)]
        cases = [
            ("random_forest", {"n_estimators": 7, "n_jobs": 2, "random_state": 21}),
            (
                "sgd",
                {
                    "loss": "modified_huber",
                    "max_iter": 5000,
                    "tol": 1e-4,
                    "random_state": 21,
                },
            ),
        ]
        for method, params in cases:
            with self.subTest(method=method):
                original_params = params.copy()
                model = estimator_factory(method, seed=42, params=params)(("x",))
                actual = model.pipeline.steps[-1][1].get_params()
                for name, value in params.items():
                    self.assertEqual(actual[name], value)
                self.assertEqual(params, original_params)
                model.fit(rows, targets)
                for probabilities in model.predict_proba(rows):
                    self.assertEqual(len(probabilities), 3)
                    self.assertAlmostEqual(sum(probabilities), 1.0)

    def test_gradient_descent_probabilities_and_serialization(self):
        import pickle
        from app.ml.alternatives import SklearnClassifier

        rows = [{"x": float(i)} for i in range(90)]
        targets = [i // 30 for i in range(90)]
        model = SklearnClassifier(("x",), method="sgd").fit(rows, targets)
        probabilities = pickle.loads(pickle.dumps(model)).predict_proba(rows)
        self.assertGreater(
            sum(a == b for a, b in zip(model.predict(rows), targets)) / 90, 0.85
        )
        for row in probabilities:
            self.assertEqual(len(row), 3)
            self.assertTrue(all(math.isfinite(v) and 0 <= v <= 1 for v in row))
            self.assertAlmostEqual(sum(row), 1)

    def test_missing_class_keeps_ldw_order(self):
        from app.ml.alternatives import SklearnClassifier

        for method in ["sgd", "random_forest", "gaussian_nb", "logistic_l1", "logistic_l2", "logistic_elasticnet"]:
            model = SklearnClassifier(("x",), method=method).fit(
                [{"x": 0.0}, {"x": 1.0}] * 10, [0, 2] * 10
            )
            row = model.predict_proba([{"x": 0.5}])[0]
            self.assertEqual(row[1], 0.0)
            self.assertAlmostEqual(sum(row), 1.0)

    def test_naive_bayes_normalizes_extrapolated_probabilities(self):
        from app.ml.alternatives import SklearnClassifier

        rows = [{"x": float(i // 3)} for i in range(30)]
        model = SklearnClassifier(("x",), method="gaussian_nb").fit(
            rows, [i % 3 for i in range(30)]
        )
        for row in model.predict_proba([{"x": 1e7}, {"x": -1e7}]):
            self.assertAlmostEqual(sum(row), 1.0, places=10)


@requires_modules("fpstreams")
class TestStatistics(unittest.TestCase):
    def test_fpstreams_statistics_match_python(self):
        from app.analysis import summarize_matches

        matches = [
            {
                "id": i + 1,
                "utcDate": f"2024-08-{10+i:02}T14:00:00+00:00",
                "status": "FINISHED",
                "homeTeam": {"id": 1, "name": "A"},
                "awayTeam": {"id": 2, "name": "B"},
                "score": {"fullTime": {"home": i, "away": 1}},
            }
            for i in range(3)
        ]
        report = summarize_matches(matches, backend="compare")
        self.assertTrue(report["parity"])
        self.assertEqual(report["teams"][0]["matches"], 3)
        self.assertEqual(report["teams"][0]["points"], 4)


class TestModelSearch(unittest.TestCase):
    @requires_modules('sklearn')
    @unittest.skipUnless(
        all(importlib.util.find_spec(name) for name in ('lightgbm', 'catboost')),
        'Install requirements-tree-models.txt for optional tree backends',
    )
    def test_optional_tree_backends_temporal_search_and_invalidated_refits(self):
        from datetime import UTC, datetime, timedelta
        import numpy as np
        from app.ml.alternatives import estimator_factory
        from app.ml.experiments import run_experiments
        from app.ml.feature_engineering import FeatureDataset
        from app.ml.model import ModelTrainingError, NotFittedError, fit_estimator

        rows = tuple({'elo_diff': float((i % 9 - 4) * 30), 'form_points_diff': math.sin(i)} for i in range(96))
        targets = tuple(i % 3 for i in range(96))
        start = datetime(2024, 1, 1, tzinfo=UTC)
        times = tuple((start + timedelta(days=i // 2)).isoformat() for i in range(96))
        features = ('elo_diff', 'form_points_diff')
        data = FeatureDataset(rows, targets, tuple(range(96)), {}, {}, times)
        report = run_experiments(data, features, methods=('lightgbm', 'lightgbm_goss', 'catboost'),
                                 optimizers=('genetic',), trials=4, seed=13)
        self.assertEqual(len(report['results']), 6)
        for candidate in report['results']:
            self.assertTrue(math.isfinite(candidate['holdout_log_loss']))
            runtime = candidate['runtime']
            self.assertFalse(runtime['timing_used_for_selection'])
            self.assertEqual(runtime['execution_policy']['threads'], 1)
            self.assertGreaterEqual(runtime['final_fit_seconds'], 0)
            self.assertGreaterEqual(runtime['holdout_inference_seconds'], 0)
            evaluations = [candidate] if candidate['optimizer'] == 'default' else candidate['search']['trials']
            for evaluation in evaluations:
                self.assertEqual(len(evaluation['development_folds']), 3)
                for fold in evaluation['development_folds']:
                    self.assertEqual(fold['runtime']['execution_policy']['threads'], 1)
                    self.assertGreaterEqual(fold['runtime']['fit_seconds'], 0)
                    self.assertGreaterEqual(fold['runtime']['inference_seconds'], 0)
            if candidate['optimizer'] != 'default':
                search = candidate['search']
                self.assertEqual(search['budget']['attempted_trials'], 4)
                self.assertEqual(search['completed_trials'] + search['failed_trials'], 4)
                self.assertLessEqual(sum(t['objective_seconds'] for t in search['trials']), search['elapsed_seconds'])
        for method in ('xgboost', 'gaussian_nb', 'lightgbm', 'lightgbm_goss', 'catboost'):
            params = {'rounds': 8} if method == 'xgboost' else {'iterations': 8} if method == 'catboost' else {'n_estimators': 8} if method.startswith('lightgbm') else {}
            model = estimator_factory(method, seed=13, params=params)(features)
            fit_estimator(model, rows, targets, kickoffs=times)
            probabilities = np.asarray(model.predict_proba(rows[:4]))
            self.assertEqual(probabilities.shape, (4, 3))
            np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-7)
            with self.subTest(method=method), self.assertRaises(ModelTrainingError):
                fit_estimator(model, rows, [True] * len(rows), kickoffs=times)
            with self.assertRaises(NotFittedError):
                model.predict_proba(rows[:1])
        cat = estimator_factory('catboost', params={'iterations': 8})(features)
        fit_estimator(cat, rows, targets, kickoffs=times)
        self.assertFalse(cat.pipeline[-1].get_params()['allow_writing_files'])
        with self.assertRaises(ModelTrainingError):
            fit_estimator(cat, rows, targets, kickoffs=times[::-1])
        with self.assertRaises(NotFittedError):
            cat.predict_proba(rows[:1])
        for method, override in [('catboost', {'thread_count': 2}), ('lightgbm', {'n_jobs': 2})]:
            with self.assertRaises(ModelTrainingError):
                estimator_factory(method, params=override)(features)

    def test_search_records_nonconverged_trials_without_selecting_them(self):
        from app.ml.alternatives import ModelConvergenceError
        from app.ml.experiments import search_parameters

        for optimizer, dependency in [('genetic', None), ('bayesian', 'skopt'), ('optuna', 'optuna')]:
            with self.subTest(optimizer=optimizer):
                if dependency and importlib.util.find_spec(dependency) is None:
                    self.skipTest(f'Optional optimizer requires {dependency}')
                calls = []
                def objective(params):
                    calls.append(params.copy())
                    if params["alpha"] > 1e-3:
                        raise ModelConvergenceError("deliberate infeasible configuration")
                    return (math.log10(params["alpha"]) + 4) ** 2
                report = search_parameters("sgd", optimizer, objective, trials=8, seed=42)
                self.assertEqual(len(calls), 8)
                self.assertEqual(len(report["trials"]), 8)
                failed = [r for r in report["trials"] if r.get("state") == "failed"]
                valid = [r for r in report["trials"] if r.get("state") != "failed"]
                self.assertTrue(failed)
                self.assertTrue(valid)
                self.assertTrue(all(r["log_loss"] is None for r in failed))
                self.assertEqual(report["best_loss"], min(r["log_loss"] for r in valid))
                self.assertLessEqual(report["best_params"]["alpha"], 1e-3)

    def test_all_nonconverged_searches_fail_with_the_complete_trial_history(self):
        from app.ml.alternatives import ModelConvergenceError
        from app.ml.experiments import search_parameters

        for optimizer, dependency in [('genetic', None), ('bayesian', 'skopt'), ('optuna', 'optuna')]:
            with self.subTest(optimizer=optimizer):
                if dependency and importlib.util.find_spec(dependency) is None:
                    self.skipTest(f'Optional optimizer requires {dependency}')
                def objective(_):
                    raise ModelConvergenceError("failed fit")
                with self.assertRaises(ModelConvergenceError) as caught:
                    search_parameters("sgd", optimizer, objective, trials=4, seed=42)
                self.assertEqual(len(caught.exception.trials), 4)
                self.assertEqual(caught.exception.search['budget']['attempted_trials'], 4)
                self.assertGreaterEqual(caught.exception.search['elapsed_seconds'],
                                        sum(t['objective_seconds'] for t in caught.exception.trials))

    @requires_modules('sklearn')
    def test_experiment_preserves_failed_searches_and_selects_a_converged_model(self):
        from app.ml.alternatives import estimator_factory
        from app.ml.experiments import run_experiments
        from app.ml.feature_engineering import FeatureDataset

        data = FeatureDataset(tuple({'x': float(i)} for i in range(90)),
                              tuple(i // 30 for i in range(90)), tuple(range(90)), {}, {})
        def bounded_factory(method, *, seed, params):
            settings = dict(params)
            if method == 'logistic_l1':
                settings['max_iter'] = 1
            return estimator_factory(method, seed=seed, params=settings)
        with patch('app.ml.experiments.estimator_factory', side_effect=bounded_factory):
            report = run_experiments(data, ('x',), methods=('logistic_l1', 'logistic_l2'),
                                     optimizers=('genetic',), trials=4)
        self.assertEqual(report['selected']['model'], 'logistic_l2')
        failed = [r for r in report['results'] if r['model'] == 'logistic_l1']
        self.assertEqual(len(failed), 2)
        self.assertTrue(all(r['selection_log_loss'] is None and 'holdout_log_loss' not in r for r in failed))
        self.assertEqual(len(failed[1]['search']['trials']), 4)
        self.assertEqual(failed[1]['search']['failed_trials'], 4)

    @requires_modules('sklearn')
    def test_failed_final_refit_does_not_change_the_locked_selection(self):
        from app.ml.alternatives import SklearnClassifier
        from app.ml.experiments import run_experiments
        from app.ml.feature_engineering import FeatureDataset

        class LimitedFinalFit(SklearnClassifier):
            def fit(self, rows, targets):
                if len(rows) == 72:
                    self.pipeline.steps[-1][1].set_params(max_iter=1)
                return super().fit(rows, targets)

        data = FeatureDataset(tuple({'x': float(i)} for i in range(90)),
                              tuple(i // 30 for i in range(90)), tuple(range(90)), {}, {})
        with patch('app.ml.experiments.estimator_factory',
                   return_value=lambda names: LimitedFinalFit(names, method='logistic_l2')):
            report = run_experiments(data, ('x',), methods=('logistic_l2',), optimizers=())
        self.assertEqual(report['selected'], {'model': 'logistic_l2', 'optimizer': 'default', 'params': {}})
        self.assertIn('holdout_error', report['results'][0])
        self.assertNotIn('holdout_log_loss', report['results'][0])

    @requires_modules('sklearn')
    @unittest.skipUnless(importlib.util.find_spec("optuna"), "Optuna is optional")
    def test_optuna_selection_is_invariant_to_final_holdout_labels(self):
        from datetime import UTC, datetime, timedelta
        from app.ml.experiments import run_experiments
        from app.ml.feature_engineering import FeatureDataset

        rows = tuple({"x": math.sin(i), "elo_diff": float(i % 5)} for i in range(90))
        origin = datetime(2024, 1, 1, tzinfo=UTC)
        times = tuple((origin + timedelta(days=i // 3)).isoformat() for i in range(90))
        reports = []
        for holdout_label in (0, 1):
            data = FeatureDataset(rows, tuple(i % 3 for i in range(72)) + (holdout_label,) * 18,
                                  tuple(range(90)), {}, {}, times)
            reports.append(run_experiments(data, ("x", "elo_diff"),
                methods=("logistic_l1", "logistic_l2", "logistic_elasticnet"),
                optimizers=("optuna",), trials=4, seed=4))
        self.assertEqual(reports[0]["selection_rows"], 72)
        self.assertEqual(reports[0]["selected"], reports[1]["selected"])
        for first, second in zip(reports[0]["results"], reports[1]["results"]):
            self.assertEqual(first["selection_log_loss"], second["selection_log_loss"])
            self.assertEqual(without_elapsed_times(first.get("search")), without_elapsed_times(second.get("search")))
        self.assertTrue(any(a["holdout_log_loss"] != b["holdout_log_loss"]
                            for a, b in zip(reports[0]["results"], reports[1]["results"])))

    def test_xgboost_search_and_final_fit_receive_matching_times(self):
        from datetime import UTC, datetime, timedelta
        from app.ml.experiments import run_experiments
        from app.ml.feature_engineering import FeatureDataset
        from app.ml.model import XGBoostClassifier

        fits = []
        class CapturingClassifier(XGBoostClassifier):
            def __init__(self, names):
                super().__init__(names, rounds=2)

            def fit(self, rows, targets, *, kickoffs=()):
                fits.append((tuple(int(row['x']) for row in rows), tuple(kickoffs)))
                return super().fit(rows, targets, **({'kickoffs': kickoffs} if kickoffs else {}))

        origin = datetime(2025, 1, 1, tzinfo=UTC)
        times = tuple((origin + timedelta(days=i // 5)).isoformat() for i in range(60))
        dataset = FeatureDataset(tuple({'x': float(i)} for i in range(60)),
                                 tuple(i % 3 for i in range(60)), tuple(range(60)), {}, {}, times)
        with patch('app.ml.experiments.estimator_factory', return_value=CapturingClassifier):
            report = run_experiments(dataset, ('x',), methods=('xgboost',), optimizers=())
        self.assertEqual(len(fits), 4)
        self.assertEqual(report['selection_rows'], 45)
        for indexes, observed in fits:
            self.assertEqual(observed, tuple(times[i] for i in indexes))
            self.assertTrue(all(i < 45 for i in indexes))

    def test_time_buckets_do_not_cross_evaluation_boundaries(self):
        from app.ml.evaluation import evaluate_expanding_window
        from app.ml.experiments import chronological_split
        from app.ml.feature_engineering import FeatureDataset

        rows = tuple({"x": float(i // 5)} for i in range(60))
        times = tuple(f"{i // 5:03}" for i in range(60))
        dataset = FeatureDataset(
            rows,
            tuple(i % 3 for i in range(60)),
            tuple(range(60)),
            {},
            {},
            kickoffs=times,
        )
        self.assertEqual(chronological_split(dataset), 45)

        class BucketEstimator:
            def fit(self, rows, targets):
                self.last_time = rows[-1]["x"]

            def predict_proba(self, rows):
                if rows[0]["x"] <= self.last_time:
                    raise AssertionError("same-time matches crossed split")
                return [[1 / 3] * 3 for _ in rows]

            def predict(self, rows):
                return [0 for _ in rows]

        evaluation = evaluate_expanding_window(
            rows,
            dataset.targets,
            ("x",),
            lambda _: BucketEstimator(),
            folds=4,
            kickoffs=times,
        )
        self.assertEqual(evaluation.evaluated_rows, 30)

    @requires_modules('sklearn')
    def test_selection_never_sees_final_holdout(self):
        from app.ml.experiments import run_experiments
        from app.ml.feature_engineering import FeatureEngineer

        matches = [
            {
                "id": i + 1,
                "utcDate": f"2024-{1+i//28:02}-{1+i%28:02}T14:00:00+00:00",
                "status": "FINISHED",
                "homeTeam": {"id": 1},
                "awayTeam": {"id": 2},
                "score": {"fullTime": {"home": i % 3, "away": 1}},
            }
            for i in range(84)
        ]
        dataset = FeatureEngineer().build_training_dataset(matches)
        report = run_experiments(
            dataset,
            FeatureEngineer().features,
            methods=["sgd"],
            optimizers=["genetic"],
            trials=4,
        )
        self.assertEqual(report["selection_rows"] + report["holdout_rows"], 84)
        self.assertEqual(len(report["results"]), 2)
        self.assertEqual(report["results"][0]["holdout_rows"], report["holdout_rows"])
        self.assertEqual(report["selected_on"], "expanding_window_log_loss")

    def test_search_budgets_reproducibility_and_nonfinite_failures(self):
        from app.ml.experiments import search_parameters

        for model, optimizer, dependency in [("sgd", "bayesian", "skopt"),
                                             ("sgd", "genetic", None),
                                             ("xgboost", "optuna", "optuna")]:
            with self.subTest(model=model, optimizer=optimizer):
                if dependency and importlib.util.find_spec(dependency) is None:
                    self.skipTest(f"Optional dependency missing: {dependency}")
                calls = []
                def objective(params):
                    if model == "sgd":
                        self.assertTrue(1e-5 <= params["alpha"] <= .1)
                        loss = (params["alpha"] - .02) ** 2
                    else:
                        self.assertTrue(2 <= params["max_depth"] <= 6)
                        self.assertTrue(.01 <= params["learning_rate"] <= .2)
                        self.assertTrue(.1 <= params["l2"] <= 10)
                        loss = (params["max_depth"] - 4) ** 2 + params["learning_rate"]
                    calls.append(params.copy())
                    return loss
                first = search_parameters(model, optimizer, objective, trials=6, seed=42)
                self.assertEqual(len(calls), 6)
                second = search_parameters(model, optimizer, objective, trials=6, seed=42)
                self.assertEqual(without_elapsed_times(first), without_elapsed_times(second))
                self.assertEqual(len(first["trials"]), 6)
                self.assertEqual(first["best_loss"], min(trial["log_loss"] for trial in first["trials"]))
                self.assertLessEqual(first["best_loss"], first["trials"][0]["log_loss"])
                with self.assertRaisesRegex(ValueError, "有限"):
                    search_parameters(model, optimizer, lambda _: math.nan, trials=4)



if __name__ == "__main__":
    unittest.main()
