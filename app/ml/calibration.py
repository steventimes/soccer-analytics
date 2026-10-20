"""Post-hoc match calibration with separate chronological fit and score windows."""

from __future__ import annotations

import math
import warnings
from typing import Sequence

from app.ml.alternatives import estimator_factory
from app.ml.evaluation import (
    classwise_reliability,
    expanding_window_boundaries,
    probability_sharpness,
    top_label_calibration_error,
)
from app.ml.experiments import chronological_split
from app.ml.model import (
    ModelTrainingError,
    _validate_probability_rows,
    accuracy_score,
    fit_estimator,
    multiclass_brier_score,
    multiclass_log_loss,
    ranked_probability_score,
)

DEFAULT_CALIBRATORS = ("identity", "temperature", "dirichlet_l2")
DRAW_CALIBRATORS = ("draw_constant", "draw_logistic", "draw_beta")
CALIBRATORS = DEFAULT_CALIBRATORS + DRAW_CALIBRATORS + ("dirichlet_odir",)


class CalibrationError(ModelTrainingError):
    """A calibration fit has no valid solution under the stated configuration."""


class ProbabilityCalibrator:
    """Map L/D/W probabilities; learned maps see only their calibration window."""

    def __init__(self, method: str = "temperature"):
        if method not in CALIBRATORS:
            raise ValueError("未知校准方法")
        self.method = method
        self.is_fitted_ = False

    def fit(self, probabilities, targets):
        self.is_fitted_ = False
        multiclass_log_loss(targets, probabilities)
        import numpy as np

        matrix = np.asarray(probabilities, dtype=float)
        labels = np.asarray(targets, dtype=int)
        self.fit_rows_ = len(labels)
        self.zero_entries_ = int((matrix < 1e-15).sum())
        if self.method == "temperature":
            from scipy.optimize import brentq
            from scipy.special import logsumexp, softmax

            logits = np.log(np.maximum(matrix, 1e-15))

            def loss(inverse_temperature):
                scaled = logits * inverse_temperature
                return float((logsumexp(scaled, axis=1) - scaled[np.arange(len(labels)), labels]).mean())

            def gradient(inverse_temperature):
                expected = (softmax(logits * inverse_temperature, axis=1) * logits).sum(axis=1)
                return float((expected - logits[np.arange(len(labels)), labels]).mean())

            # NLL is convex in inverse temperature. Solve its monotone derivative
            # rather than comparing nearly equal function values near the optimum.
            if abs(gradient(1.0)) <= 1e-12:
                optimum = 1.0
            elif gradient(0.01) >= 0:
                optimum = 0.01
            elif gradient(100.0) <= 0:
                optimum = 100.0
            else:
                try:
                    optimum = brentq(gradient, 0.01, 100.0, xtol=1e-12, rtol=1e-12, maxiter=200)
                except (ValueError, RuntimeError) as exc:
                    raise CalibrationError("温度缩放优化未收敛") from exc
            # Include identity and both endpoints even when the optimum is at a bound.
            inverse = min((1.0, 0.01, 100.0, optimum), key=loss)
            self.temperature_ = 1.0 / inverse
        elif self.method == "dirichlet_l2":
            from sklearn.exceptions import ConvergenceWarning
            from sklearn.linear_model import LogisticRegression

            if set(targets) != {0, 1, 2}:
                raise CalibrationError("Dirichlet 校准窗口必须包含全部三类赛果")
            self.estimator_ = LogisticRegression(C=1.0, l1_ratio=0.0, solver="lbfgs",
                                                 max_iter=2000, tol=1e-8)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    self.estimator_.fit(np.log(np.maximum(matrix, 1e-15)), labels)
            except ConvergenceWarning as exc:
                raise CalibrationError("Dirichlet 校准未收敛") from exc
        elif self.method == "dirichlet_odir":
            from scipy.optimize import minimize
            from scipy.special import logsumexp

            if set(targets) != {0, 1, 2}:
                raise CalibrationError("ODIR 校准窗口必须包含全部三类赛果")
            features = np.log(np.maximum(matrix, 1e-15))
            classes = matrix.shape[1]
            off_diagonal = 1 - np.eye(classes)
            # Normalized ODIR from Kull et al. (2019), section 3. Fixed
            # penalties are local controls, without internal tuning/ensembling.
            self.odir_lambda_ = self.odir_mu_ = 1e-3
            coefficient_scale = self.odir_lambda_ / (classes * (classes - 1))
            intercept_scale = self.odir_mu_ / classes

            def objective(parameters):
                weights = parameters[:-classes].reshape(classes, classes)
                bias = parameters[-classes:]
                logits = features @ weights.T + bias
                normalizers = logsumexp(logits, axis=1)
                loss = (normalizers - logits[np.arange(len(labels)), labels]).mean()
                penalized = weights * off_diagonal
                loss += coefficient_scale * np.sum(penalized ** 2) + intercept_scale * np.sum(bias ** 2)
                residual = np.exp(logits - normalizers[:, None])
                residual[np.arange(len(labels)), labels] -= 1
                gradient = residual.T @ features / len(labels) + 2 * coefficient_scale * penalized
                bias_gradient = residual.mean(axis=0) + 2 * intercept_scale * bias
                return float(loss), np.r_[gradient.ravel(), bias_gradient]

            initial = np.r_[np.eye(classes).ravel(), np.zeros(classes)]
            fitted = minimize(objective, initial, jac=True, method="L-BFGS-B",
                              options={"maxiter": 2000, "maxls": 100, "ftol": 1e-15, "gtol": 1e-9})
            loss, gradient = objective(fitted.x)
            self.max_gradient_ = float(np.max(np.abs(gradient)))
            if (not np.isfinite(fitted.x).all() or not math.isfinite(loss)
                    or not math.isfinite(self.max_gradient_) or self.max_gradient_ > 1e-6):
                raise CalibrationError("ODIR 校准优化未收敛")
            self.odir_coefficients_ = fitted.x[:-classes].reshape(classes, classes).copy()
            self.odir_intercept_ = fitted.x[-classes:].copy()
            self.optimizer_ = {"method": "L-BFGS-B", "reported_success": bool(fitted.success),
                               "status": int(fitted.status), "message": str(fitted.message),
                               "acceptance": "finite parameters and maximum absolute gradient <=1e-6"}
        elif self.method in DRAW_CALIBRATORS:
            from scipy.optimize import minimize
            from scipy.special import expit

            binary = (labels == 1).astype(float)
            if len(np.unique(binary)) != 2:
                raise CalibrationError("平局校准窗口必须同时包含平局与非平局")
            scores = matrix[:, 1]
            self.draw_scores_clipped_ = int(((scores < 1e-15) | (scores > 1 - 1e-15)).sum())
            design = self._draw_design(scores)
            # Fixed coefficient ridge is a local small-window adaptation. The
            # intercept is unpenalized; both binary outcomes are required above.
            self.ridge_ = 1e-3
            if self.method == "draw_constant":
                prevalence = float(binary.mean())
                self.draw_parameters_ = np.array([0.0, math.log(prevalence / (1 - prevalence))])
                self.ridge_ = 0.0
                self.projected_gradient_ = 0.0
                self.optimizer_ = {"method": "closed_form_empirical_prevalence"}
                self.is_fitted_ = True
                return self

            def objective(parameters):
                logits = design @ parameters
                residual = expit(logits) - binary
                penalty = parameters.copy()
                penalty[-1] = 0
                loss = np.logaddexp(0, np.where(binary == 1, -logits, logits)).mean()
                loss += self.ridge_ * np.dot(penalty, penalty) / 2
                gradient = design.T @ residual / len(binary) + self.ridge_ * penalty
                return float(loss), gradient

            initial = np.r_[np.ones(design.shape[1] - 1), 0.0]
            bounds = [(0.0, None)] * (len(initial) - 1) + [(None, None)]
            fitted = minimize(objective, initial, jac=True, method="L-BFGS-B", bounds=bounds,
                              options={"maxiter": 2000, "maxls": 100, "ftol": 1e-15, "gtol": 1e-9})
            loss, gradient = objective(fitted.x)
            projected = gradient.copy()
            for index in range(len(initial) - 1):
                if fitted.x[index] <= 1e-10 and gradient[index] > 0:
                    projected[index] = 0
            self.projected_gradient_ = float(np.max(np.abs(projected)))
            if (not np.isfinite(fitted.x).all() or not math.isfinite(loss)
                    or np.any(fitted.x[:-1] < 0) or not math.isfinite(self.projected_gradient_)
                    or self.projected_gradient_ > 1e-6):
                raise CalibrationError("平局校准优化未收敛")
            self.draw_parameters_ = fitted.x
            self.optimizer_ = {"method": "L-BFGS-B", "reported_success": bool(fitted.success),
                               "status": int(fitted.status), "message": str(fitted.message),
                               "acceptance": "finite feasible parameters and projected gradient <=1e-6"}
        self.is_fitted_ = True
        return self

    def _draw_design(self, scores):
        import numpy as np

        clipped = np.clip(np.asarray(scores, dtype=float), 1e-15, 1 - 1e-15)
        if self.method == "draw_beta":
            return np.column_stack((np.log(clipped), -np.log1p(-clipped), np.ones(len(clipped))))
        return np.column_stack((clipped, np.ones(len(clipped))))

    def predict_proba(self, probabilities):
        if not self.is_fitted_:
            raise CalibrationError("校准器尚未成功拟合")
        if not len(probabilities):
            return []
        _validate_probability_rows(probabilities)
        if self.method == "identity":
            return [[float(value) for value in row] for row in probabilities]
        import numpy as np

        if self.method in DRAW_CALIBRATORS:
            from scipy.special import expit

            matrix = np.asarray(probabilities, dtype=float)
            logits = self._draw_design(matrix[:, 1]) @ self.draw_parameters_
            draw, non_draw = expit(logits), expit(-logits)
            residual_mass = matrix[:, 0] + matrix[:, 2]
            # A certain draw has no defined conditional home/away split.
            # Use a symmetric split only for that zero-mass case.
            away_share = np.divide(matrix[:, 0], residual_mass,
                                   out=np.full(len(matrix), 0.5), where=residual_mass > 0)
            return np.column_stack((non_draw * away_share, draw,
                                    non_draw * (1 - away_share))).tolist()
        logits = np.log(np.maximum(np.asarray(probabilities, dtype=float), 1e-15))
        if self.method == "temperature":
            from scipy.special import softmax

            result = softmax(logits / self.temperature_, axis=1)
        elif self.method == "dirichlet_odir":
            from scipy.special import softmax

            result = softmax(logits @ self.odir_coefficients_.T + self.odir_intercept_, axis=1)
        else:
            result = self.estimator_.predict_proba(logits)
        return result.tolist()

    def summary(self) -> dict:
        if not self.is_fitted_:
            raise CalibrationError("校准器尚未成功拟合")
        summary = {"method": self.method, "fit_rows": self.fit_rows_}
        if self.method != "identity":
            summary.update(probability_floor=1e-15, fit_entries_below_floor=self.zero_entries_)
        if self.method == "temperature":
            summary.update(temperature=self.temperature_, temperature_bounds=[0.01, 100.0])
        elif self.method == "dirichlet_l2":
            summary.update(C=1.0, regularization="L2 on coefficients; unpenalized intercept; no ODIR",
                           classes=self.estimator_.classes_.tolist(),
                           coefficients=self.estimator_.coef_.tolist(),
                           intercept=self.estimator_.intercept_.tolist())
        elif self.method == "dirichlet_odir":
            summary.update(classes=[0, 1, 2], coefficients=self.odir_coefficients_.tolist(),
                           intercept=self.odir_intercept_.tolist(),
                           odir_lambda=self.odir_lambda_, odir_mu=self.odir_mu_,
                           regularization="mean multiclass NLL + lambda/[K*(K-1)] * sum(off-diagonal W squared) + mu/K * sum(b squared); diagonal unpenalized",
                           hyperparameters="fixed local controls; no internal search or ensemble",
                           max_gradient=self.max_gradient_, optimizer=self.optimizer_)
        elif self.method in DRAW_CALIBRATORS:
            summary.update(binary_target="draw_vs_non_draw", class_index=1,
                           draw_score_transform={
                               "draw_constant": "constant",
                               "draw_logistic": "raw_probability",
                               "draw_beta": "log_probability_and_negative_log_complement",
                           }[self.method],
                           draw_mapping={
                               "draw_constant": "sigmoid(c)",
                               "draw_logistic": "sigmoid(a * s + c)",
                               "draw_beta": "sigmoid(a * log(s) - b * log(1 - s) + c)",
                           }[self.method],
                           coefficients=self.draw_parameters_[:-1].tolist(),
                           intercept=float(self.draw_parameters_[-1]),
                           coefficient_bounds="fixed zero" if self.method == "draw_constant" else "nonnegative; no upper bound", ridge=self.ridge_,
                           regularization="mean binary NLL + ridge/2 * squared coefficients; intercept unpenalized",
                           draw_score_clip=[1e-15, 1 - 1e-15],
                           fit_draw_scores_clipped=self.draw_scores_clipped_,
                           max_projected_gradient=self.projected_gradient_,
                           optimizer=self.optimizer_,
                           coupling="preserve conditional away/home shares among non-draws; equal shares if original non-draw mass is zero")
        return summary


def _metrics(targets, probabilities) -> dict:
    from app.ml.calibration_diagnostics import calibration_diagnostics, compact_diagnostics

    reliability = classwise_reliability(targets, probabilities)
    sensitivity = {}
    for bins in (9, 10, 11):
        values = reliability if bins == 10 else classwise_reliability(targets, probabilities, bins=bins)
        sensitivity[str(bins)] = {
            "top_label_ece": top_label_calibration_error(targets, probabilities, bins=bins),
            "classwise_ece": math.fsum(row["calibration_error"] for row in values) / 3,
        }
    predictions = [max(range(3), key=lambda i: row[i]) for row in probabilities]
    return {"rows": len(targets), "accuracy": accuracy_score(targets, predictions),
            "log_loss": multiclass_log_loss(targets, probabilities),
            "brier_score": multiclass_brier_score(targets, probabilities),
            "rps": ranked_probability_score(targets, probabilities),
            "mean_confidence": probability_sharpness(targets, probabilities),
            "reliability": reliability, "ece_by_bins": sensitivity,
            "draw_metrics": {
                "log_loss": math.fsum(-math.log(max(row[1] if target == 1 else 1 - row[1], 1e-15))
                                      for target, row in zip(targets, probabilities)) / len(targets),
                "brier_score": math.fsum((row[1] - (target == 1)) ** 2
                                         for target, row in zip(targets, probabilities)) / len(targets),
                "prevalence": sum(target == 1 for target in targets) / len(targets),
                "mean_probability": math.fsum(row[1] for row in probabilities) / len(targets),
                "ece_10_bins": reliability[1]["calibration_error"],
            },
            "binning_diagnostics": compact_diagnostics(calibration_diagnostics(targets, probabilities))}


def _calibration_start(end: int, kickoffs, fraction: float) -> int:
    start = end - max(1, round(end * fraction))
    while start > 0 and kickoffs[start - 1] == kickoffs[start]:
        start -= 1
    if start < 2 or start >= end:
        raise CalibrationError("独立开球时间不足以分离模型训练与校准窗口")
    return start


def run_calibration_experiment(
    dataset, feature_names: tuple[str, ...], *, model: str = "xgboost", params=None,
    calibrators: Sequence[str] = DEFAULT_CALIBRATORS, calibration_fraction: float = 0.2,
    folds: int = 3, seed: int = 42, include_predictions: bool = False,
) -> dict:
    """Nested train/calibrate/validate windows; final holdout cannot select a map."""
    if (isinstance(calibration_fraction, bool) or not isinstance(calibration_fraction, (int, float))
            or not math.isfinite(calibration_fraction) or not 0 < calibration_fraction < 0.5):
        raise ValueError("calibration_fraction 必须在 0 与 0.5 之间")
    methods = tuple(dict.fromkeys(calibrators))
    if "identity" not in methods or any(method not in CALIBRATORS for method in methods):
        raise ValueError("校准方法必须包含 identity 且全部有效")
    count = len(dataset.rows)
    if (not feature_names or len(dataset.targets) != count or len(dataset.match_ids) != count
            or len(dataset.kickoffs) != count or not count):
        raise ValueError("校准实验需要完整、等长的特征、赛果、比赛 ID 和开球时间")
    if list(dataset.kickoffs) != sorted(dataset.kickoffs) or any(not t for t in dataset.kickoffs):
        raise ValueError("开球时间必须非空且按时间排序")
    split = chronological_split(dataset)
    boundaries = expanding_window_boundaries(split, folds=folds, kickoffs=dataset.kickoffs[:split])
    factory = estimator_factory(model, seed=seed, params=params)
    predictions = {method: [] for method in methods}
    evaluation_targets = []
    failed = set()
    fold_reports = []

    def fit_window(end, evaluation_end):
        start = _calibration_start(end, dataset.kickoffs, calibration_fraction)
        estimator = factory(feature_names)
        fit_estimator(estimator, dataset.rows[:start], dataset.targets[:start],
                      kickoffs=dataset.kickoffs[:start])
        calibration_probabilities = estimator.predict_proba(dataset.rows[start:end])
        raw = estimator.predict_proba(dataset.rows[end:evaluation_end])
        outputs, reports = {}, []
        for method in methods:
            try:
                calibrator = ProbabilityCalibrator(method).fit(calibration_probabilities, dataset.targets[start:end])
                outputs[method] = calibrator.predict_proba(raw)
                reports.append({"method": method, "state": "completed", "calibrator": calibrator.summary()})
            except CalibrationError as exc:
                reports.append({"method": method, "state": "failed", "error": str(exc)})
        ranges = {"training_rows": start, "calibration_rows": end - start,
                  "evaluation_rows": evaluation_end - end,
                  "training_last_kickoff": dataset.kickoffs[start - 1],
                  "calibration_first_kickoff": dataset.kickoffs[start],
                  "calibration_last_kickoff": dataset.kickoffs[end - 1],
                  "evaluation_first_kickoff": dataset.kickoffs[end],
                  "evaluation_last_kickoff": dataset.kickoffs[evaluation_end - 1]}
        return ranges, reports, outputs

    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]), start=1):
        ranges, reports, outputs = fit_window(start, end)
        targets = dataset.targets[start:end]
        evaluation_targets.extend(targets)
        for result in reports:
            method = result["method"]
            if method in outputs:
                predictions[method].extend(outputs[method])
                result["metrics"] = _metrics(targets, outputs[method])
            else:
                failed.add(method)
        fold_reports.append({"fold": index, **ranges, "results": reports})
    candidates = []
    for method in methods:
        row = {"method": method, "selection_log_loss": None, "state": "failed"}
        if method not in failed:
            metrics = _metrics(evaluation_targets, predictions[method])
            row.update(selection_log_loss=metrics["log_loss"], state="completed", development_metrics=metrics)
        candidates.append(row)
    completed = [row for row in candidates if row["state"] == "completed"]
    # Preserve legacy selection unless a draw method is explicitly requested.
    tolerance = 1e-9 if any(method in DRAW_CALIBRATORS for method in methods) else 0.0
    priority = ("identity", "draw_constant", "temperature", "draw_logistic", "draw_beta", "dirichlet_l2")
    if "dirichlet_odir" in methods:
        priority += ("dirichlet_odir",)
    if tolerance:
        best = min(row["selection_log_loss"] for row in completed)
        selected = min((row for row in completed if row["selection_log_loss"] <= best + tolerance),
                       key=lambda row: priority.index(row["method"]))["method"]
    else:
        selected = min(completed, key=lambda row: (row["selection_log_loss"], row["method"] != "identity"))["method"]
    ranges, final_reports, outputs = fit_window(split, count)
    for candidate in candidates:
        final = next(row for row in final_reports if row["method"] == candidate["method"])
        candidate["final_fit"] = final
        if candidate["method"] in outputs:
            candidate["holdout_metrics"] = _metrics(dataset.targets[split:], outputs[candidate["method"]])
    result = {"schema": "SoccerAnalytics.MatchCalibration.v1", "model": model,
              "params": dict(params or {}), "seed": seed, "calibrators": methods,
              "calibration_fraction": calibration_fraction, "folds": fold_reports,
              "development_rows": split, "holdout_rows": count - split, "final_windows": ranges,
              "selected": selected, "selected_on": "nested_expanding_window_log_loss",
              "selection_tolerance": tolerance,
              "selection_priority": priority if tolerance else tuple(["identity"] + [m for m in methods if m != "identity"]),
              "results": candidates,
              "note": "Base models fit only the earlier training prefix; all maps share the same base probabilities. No full-development refit after calibration; no production model saved."}
    if include_predictions:
        result["private_predictions"] = {"targets": dataset.targets[split:],
                                         "match_ids": dataset.match_ids[split:], "probabilities": outputs}
    return result
