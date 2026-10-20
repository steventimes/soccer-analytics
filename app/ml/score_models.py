"""Regularized football score models and chronological research comparisons."""

from __future__ import annotations

from dataclasses import asdict
import math
from types import SimpleNamespace

from app.ml.feature_engineering import MatchRecord, normalize_match

DEFAULT_SCORE_MODELS = ("poisson", "poisson_decay", "dixon_coles")
SCORE_MODELS = DEFAULT_SCORE_MODELS + ("poisson_constant", "bivariate_poisson")


class ScoreModelError(ValueError):
    """A fitted score model has no usable numerical solution."""


def prepare_score_matches(matches):
    unique = {}
    count = 0
    for payload in matches:
        count += 1
        record = normalize_match(asdict(payload) if isinstance(payload, MatchRecord) else payload)
        if record.match_id in unique and unique[record.match_id] != record:
            raise ValueError(f"比赛 {record.match_id} 存在冲突重复记录")
        unique[record.match_id] = record
    records = tuple(sorted((r for r in unique.values() if r.status == "FINISHED"),
                           key=lambda r: (r.kickoff, r.match_id)))
    return records, {"input_rows": count, "duplicate_rows": count - len(unique),
                     "unfinished_excluded": len(unique) - len(records), "completed_rows": len(records)}


def _unpack(theta, team_count, correlated):
    import numpy as np
    from scipy.special import expit

    attack_raw = theta[2:2 + team_count]
    defence_raw = theta[2 + team_count:2 + 2 * team_count]
    attack = attack_raw - attack_raw.mean()
    defence = defence_raw - defence_raw.mean()
    if not correlated:
        return theta[0], theta[1], attack, defence, 0.0, (0., 0., 0., 0., 0.)
    # The envelope includes every known pairing and the neutral unseen-team prior.
    q = theta[0] + max(0., float(attack.max())) + max(0., float(defence.max()))
    lower = -math.exp(-q - max(theta[1], 0.)) * (1 - 1e-8)
    upper = math.exp(min(0., -2 * q - theta[1])) * (1 - 1e-8)
    s = float(expit(theta[-1]))
    rho = (1 - s) * lower + s * upper
    upper_slope = 1. if -2 * q - theta[1] < 0 else (0. if -2 * q - theta[1] > 0 else .5)
    home_slope = 1. if theta[1] > 0 else (0. if theta[1] < 0 else .5)
    dq = -(1 - s) * lower - 2 * s * upper * upper_slope
    dh = -(1 - s) * lower * home_slope - s * upper * upper_slope
    dr = (upper - lower) * s * (1 - s)
    return theta[0], theta[1], attack, defence, rho, (dq, dh, dr, lower, upper)


def _goal_objective(theta, home, away, goals_home, goals_away, weights, team_count, ridge, correlated):
    import numpy as np
    from scipy.special import gammaln

    intercept, advantage, attack, defence, rho, derivatives = _unpack(theta, team_count, correlated)
    log_home = intercept + advantage + attack[home] + defence[away]
    log_away = intercept + attack[away] + defence[home]
    home_rate, away_rate = np.exp(log_home), np.exp(log_away)
    loss = home_rate + away_rate - goals_home * log_home - goals_away * log_away
    loss += gammaln(goals_home + 1) + gammaln(goals_away + 1)
    grad_home, grad_away = home_rate - goals_home, away_rate - goals_away
    rho_gradient = 0.
    if correlated:
        both_zero = (goals_home == 0) & (goals_away == 0)
        zero_one = (goals_home == 0) & (goals_away == 1)
        one_zero = (goals_home == 1) & (goals_away == 0)
        both_one = (goals_home == 1) & (goals_away == 1)
        product = home_rate * away_rate
        tau = np.ones(len(home))
        tau[both_zero] = 1 - product[both_zero] * rho
        tau[zero_one] = 1 + home_rate[zero_one] * rho
        tau[one_zero] = 1 + away_rate[one_zero] * rho
        tau[both_one] = 1 - rho
        if (tau <= 0).any() or not np.isfinite(tau).all():
            raise ScoreModelError("Dixon–Coles 修正产生非法概率")
        loss -= np.log(tau)
        grad_home[both_zero] += product[both_zero] * rho / tau[both_zero]
        grad_away[both_zero] += product[both_zero] * rho / tau[both_zero]
        grad_home[zero_one] -= home_rate[zero_one] * rho / tau[zero_one]
        grad_away[one_zero] -= away_rate[one_zero] * rho / tau[one_zero]
        rho_terms = np.zeros(len(home))
        rho_terms[both_zero] = product[both_zero] / tau[both_zero]
        rho_terms[zero_one] = -home_rate[zero_one] / tau[zero_one]
        rho_terms[one_zero] = -away_rate[one_zero] / tau[one_zero]
        rho_terms[both_one] = 1 / tau[both_one]
        rho_gradient = float(weights @ rho_terms)
    grad_home, grad_away = weights * grad_home, weights * grad_away
    ga = np.bincount(home, weights=grad_home, minlength=team_count) + np.bincount(away, weights=grad_away, minlength=team_count)
    gd = np.bincount(away, weights=grad_home, minlength=team_count) + np.bincount(home, weights=grad_away, minlength=team_count)
    dq, dh, dr, _, _ = derivatives
    if correlated:
        for values, gradient in ((attack, ga), (defence, gd)):
            active = values == values.max()
            gradient[active] += rho_gradient * dq / active.sum()
    raw = theta[2:2 + 2 * team_count]
    gradient = np.concatenate(([grad_home.sum() + grad_away.sum() + rho_gradient * dq,
                                grad_home.sum() + rho_gradient * dh],
                               ga - ga.mean(), gd - gd.mean()))
    gradient[2:] += 2 * ridge * raw
    if correlated:
        gradient = np.append(gradient, rho_gradient * dr)
    return float(weights @ loss + ridge * (raw @ raw)), gradient


def _bivariate_log_probabilities(goals_home, goals_away, home, away, shared):
    """Exact shared-component log mass and latent-count derivatives, including c=0."""
    import numpy as np
    from scipy.special import gammaln

    x, y, home, away = np.broadcast_arrays(goals_home, goals_away, home, away)
    log_home, log_away = np.log(home), np.log(away)
    base = -home - away - shared + x * log_home + y * log_away - gammaln(x + 1) - gammaln(y + 1)
    if shared == 0:
        return base, np.zeros_like(base), x * y / (home * away)
    # Sum all feasible shared counts in log space. No score-grid cutoff is used.
    term, total = np.zeros_like(base), np.zeros_like(base)
    moment = np.full_like(base, -np.inf)
    for k in range(1, int(np.minimum(x, y).max(initial=0)) + 1):
        active = (x >= k) & (y >= k)
        term[active] += (np.log(x[active] - k + 1) + np.log(y[active] - k + 1)
                         + math.log(shared) - log_home[active] - log_away[active] - math.log(k))
        total[active] = np.logaddexp(total[active], term[active])
        moment[active] = np.logaddexp(moment[active], term[active] + math.log(k))
    return base + total, np.exp(moment - total), np.exp(moment - total - math.log(shared))


def _bivariate_goal_objective(theta, home, away, goals_home, goals_away, weights, team_count, ridge):
    import numpy as np

    intercept, advantage, attack, defence, _, _ = _unpack(theta, team_count, False)
    home_rate = np.exp(intercept + advantage + attack[home] + defence[away])
    away_rate = np.exp(intercept + attack[away] + defence[home])
    logp, latent, shared_ratio = _bivariate_log_probabilities(goals_home, goals_away, home_rate, away_rate, theta[-1])
    gh = weights * (home_rate - goals_home + latent)
    ga = weights * (away_rate - goals_away + latent)
    attack_gradient = (np.bincount(home, weights=gh, minlength=team_count)
                       + np.bincount(away, weights=ga, minlength=team_count))
    defence_gradient = (np.bincount(away, weights=gh, minlength=team_count)
                        + np.bincount(home, weights=ga, minlength=team_count))
    raw = theta[2:2 + 2 * team_count]
    gradient = np.r_[gh.sum() + ga.sum(), gh.sum(),
                      attack_gradient - attack_gradient.mean(), defence_gradient - defence_gradient.mean()]
    gradient[2:] += 2 * ridge * raw
    gradient = np.r_[gradient, weights @ (1 - shared_ratio)]
    return float(-weights @ logp + ridge * (raw @ raw)), gradient


def _goal_count_diagnostics(cohort, rates, rho, shared=0.):
    """Aggregate point-fit count expectations with complete marginal and joint tails."""
    import numpy as np
    from scipy.stats import poisson

    goals = np.asarray([(row.home_score, row.away_score) for row in cohort])
    rates = np.asarray(rates, dtype=float)
    marginal_bins = ("0", "1", "2", "3", "4+")
    marginals, pmfs = {}, []
    for column, side in enumerate(("home", "away")):
        observed = np.bincount(np.minimum(goals[:, column], 4), minlength=5)
        pmf = poisson.pmf(np.arange(4)[None, :], rates[:, column, None])
        pmfs.append(pmf)
        expected = np.r_[pmf.sum(axis=0), poisson.sf(3, rates[:, column]).sum()]
        marginals[side] = [{"bin": label, "observed_count": int(count), "expected_count": float(mean)}
                           for label, count, mean in zip(marginal_bins, observed, expected)]
    home, away = pmfs
    joint = np.column_stack((home[:, 0] * away[:, 0], home[:, 0] * away[:, 1],
                             home[:, 1] * away[:, 0], home[:, 1] * away[:, 1]))
    cells = ((0, 0), (0, 1), (1, 0), (1, 1))
    if shared:
        # Marginals use total means; the joint law uses the independent components.
        h, a = rates[:, 0] - shared, rates[:, 1] - shared
        joint = np.column_stack([np.exp(_bivariate_log_probabilities(
            np.full(len(cohort), x), np.full(len(cohort), y), h, a, shared)[0]) for x, y in cells])
        # Given shared count>=2, every score lies outside the displayed 2x2 block.
        other = (poisson.sf(1, shared)
                 + poisson.pmf(0, shared) * (poisson.sf(1, h) + poisson.cdf(1, h) * poisson.sf(1, a))
                 + poisson.pmf(1, shared) * (-np.expm1(-h - a)))
    else:
        correction = rho * joint[:, 3]
        joint += correction[:, None] * np.array([-1., 1., 1., -1.])
        # DC adjustments cancel within this block; compute its complement with tails.
        other = poisson.sf(1, rates[:, 0]) + home[:, :2].sum(axis=1) * poisson.sf(1, rates[:, 1])
    low_scores = [{"bin": f"{h}:{a}",
                   "observed_count": int(np.count_nonzero((goals == (h, a)).all(axis=1))),
                   "expected_count": float(joint[:, index].sum())}
                  for index, (h, a) in enumerate(cells)]
    low_scores.append({"bin": "other", "observed_count": len(cohort) - sum(row["observed_count"] for row in low_scores),
                       "expected_count": float(other.sum())})
    return {"schema": "SoccerAnalytics.ScoreCountDiagnostics.v1", "matches": len(cohort),
            "marginals": marginals, "low_scores": low_scores,
            "scope": "Observed counts versus sums of point-fit fixture probabilities. Marginals include 4+; joint low-score cells include all other scores. No truncated-grid normalization, posterior uncertainty, goodness-of-fit p-value or selection criterion."}


class FootballScoreModel:
    """Attack/defence rates with independent, DC or shared-component goal counts."""

    def __init__(self, method="poisson", *, half_life_days=0., ridge=.01, max_iter=1000):
        if method not in ("poisson", "poisson_constant", "dixon_coles", "bivariate_poisson"):
            raise ValueError("未知比分模型")
        for value, label in ((half_life_days, "half_life_days"), (ridge, "ridge")):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{label} 必须是非负有限数")
        if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter 必须是正整数")
        self.method, self.half_life_days, self.ridge, self.max_iter = method, half_life_days, ridge, max_iter
        self.is_fitted_ = False

    def fit(self, matches):
        self.is_fitted_ = False
        import numpy as np
        from scipy.optimize import minimize

        records, self.input_audit_ = prepare_score_matches(matches)
        if len(records) < 10:
            raise ValueError("比分模型至少需要 10 场已结束比赛")
        teams = sorted({team for r in records for team in (r.home_team_id, r.away_team_id)})
        self.team_index_ = {team: index for index, team in enumerate(teams)}
        count = len(teams)
        home = np.array([self.team_index_[r.home_team_id] for r in records])
        away = np.array([self.team_index_[r.away_team_id] for r in records])
        x = np.array([r.home_score for r in records], dtype=float)
        y = np.array([r.away_score for r in records], dtype=float)
        reference = records[-1].kickoff
        ages = np.array([(reference - r.kickoff).total_seconds() / 86400 for r in records])
        weights = np.ones(len(records)) if self.half_life_days == 0 else np.exp(-math.log(2) * ages / self.half_life_days)
        self.effective_rows_ = float(weights.sum() ** 2 / (weights @ weights))
        weights /= weights.sum()
        initial = np.zeros(2 + 2 * count + (self.method in ("dixon_coles", "bivariate_poisson")))
        initial[0] = np.clip(math.log(max(float(weights @ y), .05)), -2, 2)
        initial[1] = np.clip(math.log(max(float(weights @ x), .05)) - initial[0], -1, 1)
        if self.method == "dixon_coles":
            *_, (_, _, _, lo, hi) = _unpack(initial, count, True)
            initial[-1] = math.log(-lo / hi)  # Start at physical rho=0.
        team_bounds = (0., 0.) if self.method == "poisson_constant" else (-1.5, 1.5)
        bounds = [(-2., 2.), (-1., 1.)] + [team_bounds] * (2 * count)
        if self.method == "dixon_coles": bounds.append((-8., 8.))
        if self.method == "bivariate_poisson": bounds.append((0., 3.))
        objective = _bivariate_goal_objective if self.method == "bivariate_poisson" else _goal_objective
        args = (home, away, x, y, weights, count, self.ridge)
        if self.method != "bivariate_poisson": args += (self.method == "dixon_coles",)
        result = minimize(objective, initial, args=args,
                          jac=True, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": self.max_iter, "ftol": 1e-10, "gtol": 1e-6, "maxls": 40})
        if not result.success or not math.isfinite(result.fun) or not np.isfinite(result.x).all():
            raise ScoreModelError(f"比分模型未收敛: {result.message}")
        self.theta_ = result.x
        self.training_first_kickoff_, self.training_last_kickoff_ = records[0].kickoff.isoformat(), reference.isoformat()
        self.optimizer_iterations_, self.penalized_loss_ = int(result.nit), float(result.fun)
        self.is_fitted_ = True
        return self

    def _parameters(self):
        if not self.is_fitted_:
            raise ScoreModelError("比分模型尚未成功拟合")
        return _unpack(self.theta_, len(self.team_index_), self.method == "dixon_coles")

    def _component_rates(self, home_team_id, away_team_id):
        intercept, advantage, attack, defence, _, _ = self._parameters()
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (home_team_id, away_team_id)) or home_team_id == away_team_id:
            raise ValueError("主客队需要不同的正整数 ID")
        hi, ai = self.team_index_.get(home_team_id), self.team_index_.get(away_team_id)
        ha, hd = (0., 0.) if hi is None else (attack[hi], defence[hi])
        aa, ad = (0., 0.) if ai is None else (attack[ai], defence[ai])
        return math.exp(intercept + advantage + ha + ad), math.exp(intercept + aa + hd)

    def _shared_rate(self):
        self._parameters()
        return float(self.theta_[-1]) if self.method == "bivariate_poisson" else 0.

    def expected_goals(self, home_team_id, away_team_id):
        home, away = self._component_rates(home_team_id, away_team_id)
        shared = self._shared_rate()
        return home + shared, away + shared

    def predict_proba(self, fixtures):
        import numpy as np
        from scipy.stats import skellam

        self._parameters()
        rates = [self._component_rates(r.home_team_id, r.away_team_id)
                 for r in (normalize_match(asdict(p) if isinstance(p, MatchRecord) else p) for p in fixtures)]
        if not rates: return []
        home, away = np.asarray(rates).T
        rho = self._parameters()[4]
        correction = rho * np.exp(np.log(home) + np.log(away) - home - away)
        probabilities = np.column_stack((skellam.cdf(-1, home, away), skellam.pmf(0, home, away), skellam.sf(0, home, away)))
        probabilities += correction[:, None] * np.array([1., -2., 1.])
        if not np.isfinite(probabilities).all() or probabilities.min() < -1e-12 or probabilities.max() > 1 + 1e-12:
            raise ScoreModelError("比分模型产生非法胜平负概率")
        return np.clip(probabilities, 0., 1.).tolist()

    def score_log_probability(self, home_team_id, away_team_id, home_goals, away_goals):
        from scipy.special import gammaln

        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (home_goals, away_goals)):
            raise ValueError("比分必须为非负整数")
        home, away = self._component_rates(home_team_id, away_team_id)
        if self.method == "bivariate_poisson":
            return float(_bivariate_log_probabilities([home_goals], [away_goals], [home], [away], self._shared_rate())[0][0])
        rho = self._parameters()[4]
        tau = 1.
        if home_goals == away_goals == 0: tau = 1 - home * away * rho
        elif home_goals == 0 and away_goals == 1: tau = 1 + home * rho
        elif home_goals == 1 and away_goals == 0: tau = 1 + away * rho
        elif home_goals == away_goals == 1: tau = 1 - rho
        return float(math.log(tau) - home - away + home_goals * math.log(home) + away_goals * math.log(away)
                     - gammaln(home_goals + 1) - gammaln(away_goals + 1))

    def team_strengths(self):
        _, _, attack, defence, _, _ = self._parameters()
        return {team: {"attack_log_strength": float(attack[index]), "defence_log_weakness": float(defence[index])}
                for team, index in self.team_index_.items()}

    def summary(self):
        intercept, advantage, attack, defence, rho, derivatives = self._parameters()
        return {"method": self.method, "half_life_days": self.half_life_days, "ridge": self.ridge,
                "probability_class_order": ["away_win", "draw", "home_win"],
                "outcome_probability_method": "Infinite-support Skellam on independent-component rates with analytic DC correction where applicable; no truncated-grid renormalization.",
                "shared_goal_rate": self._shared_rate(),
                "goal_rate_interpretation": ("Log predictors give independent-component rates; marginal expected goals add the common shared rate. The shared count cancels from the score difference."
                                             if self.method == "bivariate_poisson" else "Log predictors give marginal expected goals."),
                "estimation_scope": "Bounded penalized point fit with fixed ridge; no learned hierarchical scale, mixture prior, posterior predictive samples or uncertainty intervals.",
                "time_weighting": {"unit": "elapsed 24-hour days", "reference": "last training kickoff",
                                   "decay_rate_formula": "log(2)/half_life_days for positive half-life; 0 when disabled",
                                   "fit_normalization": "weights sum to one before the penalized objective"},
                "neutral_venue_policy": "Home effect always applies to the listed home team; no neutral-venue flag is consumed.",
                "team_strengths_policy": ("All team log strengths fixed at zero; common home/away rates."
                                          if self.method == "poisson_constant" else "Centered fitted attack and defensive weakness."),
                "team_count": len(self.team_index_), "training_rows": self.input_audit_["completed_rows"],
                "effective_training_rows": self.effective_rows_, "intercept": float(intercept),
                "home_log_advantage": float(advantage), "rho": float(rho), "rho_bounds": list(derivatives[3:]),
                "attack_sum": float(attack.sum()), "defence_sum": float(defence.sum()),
                "training_first_kickoff": self.training_first_kickoff_, "training_last_kickoff": self.training_last_kickoff_,
                "optimizer_iterations": self.optimizer_iterations_, "penalized_training_loss": self.penalized_loss_,
                "unseen_team_policy": "Neutral log attack/defence=0, corresponding to the fitted geometric-average team; no future-team fitting.",
                "rate_parameter_bounds": {"intercept": [-2, 2], "home_effect": [-1, 1],
                                          "raw_team_log_strengths": [0., 0.] if self.method == "poisson_constant" else [-1.5, 1.5],
                                          "rho_coordinate": [-8, 8] if self.method == "dixon_coles" else None,
                                          "shared_goal_rate": [0., 3.] if self.method == "bivariate_poisson" else None}}


def run_score_experiments(matches, *, models=DEFAULT_SCORE_MODELS, half_life_days=365., ridge=.01, folds=3, max_iter=1000, include_predictions=False):
    from app.ml.calibration import _metrics
    from app.ml.evaluation import expanding_window_boundaries
    from app.ml.experiments import chronological_split

    selected_models = tuple(dict.fromkeys(models))
    if not selected_models or any(model not in SCORE_MODELS for model in selected_models):
        raise ValueError("未知或空的比分模型列表")
    records, audit = prepare_score_matches(matches)
    kickoffs = tuple(r.kickoff.isoformat() for r in records)
    split = chronological_split(SimpleNamespace(rows=records, kickoffs=kickoffs))
    boundaries = expanding_window_boundaries(split, folds=folds, kickoffs=kickoffs[:split])

    def factory(name):
        method = name if name in ("dixon_coles", "poisson_constant", "bivariate_poisson") else "poisson"
        return FootballScoreModel(method, half_life_days=0. if name in ("poisson", "poisson_constant", "bivariate_poisson") else half_life_days,
                                  ridge=ridge, max_iter=max_iter)

    def score(model, cohort):
        targets = [0 if r.home_score < r.away_score else 2 if r.home_score > r.away_score else 1 for r in cohort]
        probabilities = model.predict_proba(cohort)
        metrics = _metrics(targets, probabilities)
        metrics["score_log_loss"] = -math.fsum(model.score_log_probability(r.home_team_id, r.away_team_id, r.home_score, r.away_score) for r in cohort) / len(cohort)
        rates = [model.expected_goals(r.home_team_id, r.away_team_id) for r in cohort]
        metrics["goal_count_diagnostics"] = _goal_count_diagnostics(cohort, rates, model._parameters()[4], model._shared_rate())
        metrics["goals_mae"] = math.fsum(abs(h - r.home_score) + abs(a - r.away_score) for r, (h, a) in zip(cohort, rates)) / (2 * len(cohort))
        metrics["unseen_team_matches"] = sum(r.home_team_id not in model.team_index_ or r.away_team_id not in model.team_index_ for r in cohort)
        return metrics, targets, probabilities

    candidates = []
    for name in selected_models:
        fold_rows, all_targets, all_probabilities = [], [], []
        for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]), 1):
            entry = {"fold": index, "training_rows": start, "validation_rows": end - start,
                     "training_last_kickoff": kickoffs[start - 1], "validation_first_kickoff": kickoffs[start],
                     "validation_last_kickoff": kickoffs[end - 1]}
            try:
                model = factory(name).fit(records[:start])
                metrics, targets, probabilities = score(model, records[start:end])
                entry.update(state="completed", metrics=metrics, fit=model.summary())
                all_targets.extend(targets); all_probabilities.extend(probabilities)
            except ScoreModelError as exc:
                entry.update(state="failed", error=str(exc))
            fold_rows.append(entry)
        candidate = {"model": name, "folds": fold_rows, "selection_log_loss": None}
        if all(row['state'] == 'completed' for row in fold_rows):
            candidate["development_metrics"] = _metrics(all_targets, all_probabilities)
            candidate["selection_log_loss"] = candidate["development_metrics"]["log_loss"]
        candidates.append(candidate)
    valid = [c for c in candidates if c["selection_log_loss"] is not None]
    if not valid: raise ScoreModelError("所有比分模型均未完成开发回测")
    selected = min(valid, key=lambda c: c["selection_log_loss"])["model"]
    private = {}
    for candidate in valid:
        try:
            model = factory(candidate["model"]).fit(records[:split])
            metrics, targets, probabilities = score(model, records[split:])
            candidate.update(holdout_metrics=metrics, final_fit=model.summary())
            if include_predictions:
                private[candidate["model"]] = {"targets": targets, "probabilities": probabilities,
                                               "theta": model.theta_.tolist(), "team_ids": list(model.team_index_)}
        except ScoreModelError as exc:
            candidate["holdout_error"] = str(exc)
    return {"schema": "SoccerAnalytics.ScoreModels.v1", "audit": audit, "development_rows": split,
            "holdout_rows": len(records) - split, "holdout_first_kickoff": kickoffs[split],
            "holdout_last_kickoff": kickoffs[-1], "selected": selected,
            "selected_on": "expanding_window_log_loss", "results": candidates,
            **({"private_predictions": private, "private_match_ids": [r.match_id for r in records[split:]]} if include_predictions else {}),
            "note": "Joint penalized score likelihood, fixed half-life/ridge, development-only selection; no production model saved. Outcome probabilities use full-support Skellam on independent-component rates plus exact four-cell Dixon-Coles corrections where applicable; shared-component counts cancel from the difference."}
