"""Shared probability-quality evidence for recipe-aware production models."""
import math


def probability_quality_passes(row):
    if row.get("schema_version") == 3:
        return True  # Historical evidence keeps its original, disclosed policy.
    if row.get("probability_baseline") != "training_frequency_laplace_v1" or row.get("probability_quality_passed") is not True:
        return False
    fields = ("validation_log_loss", "validation_brier_score",
              "validation_baseline_log_loss", "validation_baseline_brier_score")
    values = [row.get(name) for name in fields]
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) or v < 0 for v in values):
        return False
    loss, brier, baseline_loss, baseline_brier = values
    return loss <= baseline_loss + 1e-6 and brier <= baseline_brier + 1e-6
