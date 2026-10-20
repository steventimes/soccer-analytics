"""Compare observed and point-fit expected score counts from a score-model report."""

import argparse
import json
import math
from pathlib import Path
import re

from .style import COLORS, apply_style, decorate, save


def _tables(diagnostics, matches):
    if (not isinstance(diagnostics, dict)
            or diagnostics.get("schema") != "SoccerAnalytics.ScoreCountDiagnostics.v1"
            or type(diagnostics.get("matches")) is not int
            or diagnostics.get("matches") != matches):
        raise ValueError("Missing or inconsistent score-count diagnostics")
    marginals = diagnostics.get("marginals", {})
    if not isinstance(marginals, dict):
        raise ValueError("Missing goal marginals")
    tables = [marginals.get("home"), marginals.get("away"), diagnostics.get("low_scores")]
    labels = [("0", "1", "2", "3", "4+")] * 2 + [("0:0", "0:1", "1:0", "1:1", "other")]
    for rows, bins in zip(tables, labels):
        if (not isinstance(rows, list) or len(rows) != len(bins)
                or any(not isinstance(row, dict) for row in rows)
                or tuple(row.get("bin") for row in rows) != bins):
            raise ValueError("Score-count bins must retain their complete remainder")
        for row in rows:
            count, expected = row.get("observed_count"), row.get("expected_count")
            if (type(count) is not int or count < 0 or type(expected) not in (int, float)
                    or not math.isfinite(expected) or not 0 <= expected <= matches + 1e-7):
                raise ValueError("Invalid observed or expected score count")
        if (sum(row["observed_count"] for row in rows) != matches
                or not math.isclose(math.fsum(row["expected_count"] for row in rows), matches,
                                    rel_tol=1e-9, abs_tol=1e-7)):
            raise ValueError("Every count panel must account for the full cohort")
    return tables


def render(report, output, *, competition, models=None):
    """Render one league's holdout counts; never create missing numerical results."""
    if not isinstance(competition, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", competition):
        raise ValueError("Expected a competition code")
    leagues = [row for row in report.get("results", []) if row.get("competition") == competition]
    if len(leagues) != 1 or leagues[0].get("schema") != "SoccerAnalytics.ScoreModels.v1":
        raise ValueError("Expected one matching score-model league result")
    league = leagues[0]
    matches = league.get("holdout_rows")
    if type(matches) is not int or matches < 1:
        raise ValueError("Expected a nonempty holdout cohort")
    candidates = league["results"]
    if models is not None:
        models = tuple(models)
        if (not models or len(set(models)) != len(models)
                or not set(models) <= {row["model"] for row in candidates}):
            raise ValueError("Expected distinct model names present in the report")
        candidates = [row for row in candidates if row["model"] in models]
    available, unavailable = [], []
    for row in candidates:
        if "holdout_metrics" not in row:
            unavailable.append(row["model"])
            continue
        tables = _tables(row["holdout_metrics"].get("goal_count_diagnostics"), matches)
        observed = [[cell["observed_count"] for cell in panel] for panel in tables]
        if available and observed != available[0][2]:
            raise ValueError("Models have different observed count panels")
        available.append((row["model"], tables, observed))
    if not available or (models is not None and unavailable):
        raise ValueError("Requested models have no complete holdout count diagnostics")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    apply_style()
    styles = {"poisson_constant": ("Common rates", COLORS["muted"], "D"),
              "poisson": ("Poisson", COLORS["blue"], "o"),
              "poisson_decay": ("Decayed Poisson", COLORS["orange"], "s"),
              "dixon_coles": ("Dixon–Coles", COLORS["ink"], "^"),
              "bivariate_poisson": ("Bivariate Poisson", COLORS["blue"], "P")}
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 5.8))
    titles = ("Home goals", "Away goals", "Low scores and remainder")
    handles = [Patch(facecolor=COLORS["light"], edgecolor=COLORS["neutral"], label="Observed count")]
    for name, _, _ in available:
        label, color, marker = styles.get(name, (name, COLORS["muted"], "o"))
        handles.append(Line2D([], [], color=color, marker=marker, linestyle="none", label=label))
    for panel, ax in enumerate(axes):
        observed = available[0][2][panel]
        ax.bar(range(5), observed, width=.72, color=COLORS["light"],
               edgecolor=COLORS["neutral"], linewidth=.8, zorder=2)
        for index, (name, tables, _) in enumerate(available):
            _, color, marker = styles.get(name, (name, COLORS["muted"], "o"))
            offset = (index - (len(available) - 1) / 2) * min(.18, .6 / len(available))
            ax.scatter([x + offset for x in range(5)], [cell["expected_count"] for cell in tables[panel]],
                       color=color, marker=marker, s=44, zorder=3)
        labels = [cell["bin"] for cell in available[0][1][panel]]
        labels = ["Other\nscores" if label == "other" else label for label in labels]
        ax.set_xticks(range(5), labels)
        ax.set_title(titles[panel], loc="left")
        ax.set_ylabel("Matches")
        ax.set_ylim(bottom=0)
        ax.set_xlim(-.6, 4.6)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    note = ("Bars: observed counts. Points: sums of point-fit probabilities; every panel retains its full remainder.\n"
            "No posterior intervals, goodness-of-fit p-values or model selection from these counts.")
    if unavailable:
        note += "\nUnavailable holdout fits: " + ", ".join(unavailable)
    decorate(fig, "Score distribution checks",
             f"{competition} · {matches} holdout matches · {league['holdout_first_kickoff'][:10]} to {league['holdout_last_kickoff'][:10]}",
             note, bottom=.25, top=.70)
    fig.subplots_adjust(left=.065, right=.98, wspace=.34)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.055, .86),
               ncols=min(3, len(handles)), labelspacing=.9, frameon=False, fontsize=10)
    return save(fig, output, f"score-counts-{competition.lower()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--competition", required=True)
    parser.add_argument("--models", help="Optional comma-separated model names from the report")
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    models = tuple(value.strip() for value in args.models.split(",")) if args.models is not None else None
    render(json.loads(args.report.read_text()), args.output, competition=args.competition, models=models)


if __name__ == "__main__":
    main()
