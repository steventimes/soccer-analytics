"""Export reviewed validation results without selecting favorable tests."""

import argparse
import json
from pathlib import Path

from .visualization.style import COLORS, apply_style, decorate, save


def generate(
    report=".tmp/event-game-validation.json", output=".tmp/scouting/validation"
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    data = json.loads(Path(report).read_text())
    tests = data["tests"]
    apply_style()
    labels = ["Short passing", "Long passing", "Vision", "Overall rating"]
    leagues = ["England", "France", "Germany", "Italy", "Spain"]
    matrix = np.full((len(tests), len(leagues)), np.nan)
    for i, test in enumerate(tests):
        for fold in test.get("leave_one_league_out", []):
            if fold["baseline"] > 0:
                matrix[i, leagues.index(fold["held_out_league"])] = (
                    100 * fold["mae_difference"] / fold["baseline"]
                )
    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.2, 1, 1])
    left = fig.add_subplot(grid[:, 0])
    right_axes = [fig.add_subplot(grid[i // 2, 1 + i % 2]) for i in range(len(tests))]
    for i, test in enumerate(tests):
        value = test.get("adjusted_rank_correlation")
        ci = test.get("player_bootstrap_ci95")
        if value is not None and ci:
            left.plot(ci, [i, i], color=COLORS["ink"], linewidth=1.7)
            left.scatter([value], [i], color=COLORS["blue"], s=48, zorder=3)
            left.text(.405, i, f"{value:+.3f}", ha="right", va="center")
    left.axvline(0, color=COLORS["muted"], linewidth=1)
    left.set_yticks(
        range(len(tests)),
        [
            f"Short passing →\nPass completion\nn={tests[0]['players']:,} players",
            f"Long passing →\nLong-pass completion\nn={tests[1]['players']:,} players",
            f"Vision →\nProgressive passes / 90\nn={tests[2]['players']:,} players",
            f"Overall rating →\nSupplied PlayeRank\nn={tests[3]['players']:,} players",
        ],
    )
    left.invert_yaxis()
    left.set_xlim(-.1, .43)
    left.set_title("Adjusted rank correlation", loc="left", pad=16)
    left.set_xlabel("Correlation · lines show 95% intervals")
    left.grid(axis="x")
    left.set_axisbelow(True)
    finite = matrix[np.isfinite(matrix)]
    bounds = (min(-1, min(finite)) - 1.8, max(1, max(finite)) + 2)
    for i, right in enumerate(right_axes):
        right.axvline(0, color=COLORS["muted"], linewidth=1)
        folds = {fold["held_out_league"]: fold for fold in tests[i].get("leave_one_league_out", [])}
        for j, league in enumerate(leagues):
            value = matrix[i, j]
            if np.isfinite(value):
                color = COLORS["blue"] if value <= 0 else COLORS["orange"]
                right.scatter(value, j, s=38, color=color, zorder=3)
                right.annotate(f"{value:+.2f}%", (value, j), xytext=(7, 0), textcoords="offset points", va="center", fontsize=10)
        right.set(yticks=range(len(leagues)), yticklabels=[f"{league} · {folds[league]['test_players']:,}" if league in folds else league for league in leagues], xlim=bounds)
        right.invert_yaxis()
        right.set_title(labels[i], loc="left", pad=12)
        right.grid(axis="x")
        right.set_axisbelow(True)
        if i >= 2:
            right.set_xlabel("MAE change vs controls-only baseline (%)")
        else:
            right.tick_params(labelbottom=False)
    decorate(
        fig, "Historical game ratings and observed performance",
        "FIFA attributes linked to Wyscout 2017/18 · Retrospective selected samples",
        "Left: age, position, team and pass-context adjusted; 500 player bootstraps; 95% intervals unadjusted for multiple tests.\nRight: one attribute added to position/age/team controls; negative MAE change favors it; labels show test players.\nRetrospective single-season samples. League estimates have no intervals; future-season validity is untested.",
        bottom=.23, top=.76,
    )
    fig.subplots_adjust(left=.17, right=.97, wspace=.72, hspace=.48)
    fig.text(.50, .81, "Leave one league out · MAE change (%)", fontsize=12, weight="semibold")
    save(fig, output, "game-real-validation")
    return Path(output) / "game-real-validation.png"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default=".tmp/event-game-validation.json")
    parser.add_argument("--output", default=".tmp/scouting/validation")
    args = parser.parse_args()
    print(generate(args.report, args.output))
