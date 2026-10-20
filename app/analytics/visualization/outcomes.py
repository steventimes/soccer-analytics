"""Render fixed-model paired outcome differences and block-length sensitivity."""

import argparse
import json
from pathlib import Path

from .style import COLORS, apply_style, decorate, save


def render(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if report.get("schema") != "SoccerAnalytics.FrozenOutcomeComparisonStudy.v1":
        raise ValueError("Expected a fixed outcome comparison study")
    apply_style()
    leagues = {"PL": "Premier League", "FL1": "Ligue 1", "BL1": "Bundesliga",
               "SA": "Serie A", "PD": "La Liga"}
    models = {"poisson": "Poisson", "poisson_decay": "Decayed Poisson", "dixon_coles": "Dixon–Coles"}
    colors = [COLORS['blue'], COLORS['orange'], COLORS['muted']]
    markers = ['o', 's', 'D']
    fig, ax = plt.subplots(figsize=(10.4, 7.0))
    labels = []
    for y, league in enumerate(report['results']):
        model = league['development_selected_score_model']
        row = next(c['metrics']['log_loss'] for c in league['comparisons'] if c['model'] == model)
        labels.append(f"{leagues.get(league['competition'], league['competition'])} · {models.get(model, model)}\n"
                      f"{league['matches']} matches · {league['utc_match_days']} observed days")
        for j, length in enumerate([1, 3, 7]):
            interval = next(r for r in row['stationary_bootstrap'] if r['mean_block_days'] == length)
            position = y + (j - 1) * .17
            ax.hlines(position, interval['lower'], interval['upper'], color=colors[j], lw=2)
            ax.scatter([row['difference']], [position], marker=markers[j], color=colors[j], s=32, zorder=3)
    ax.axvline(0, color=COLORS['ink'], lw=1)
    ax.set_yticks(range(len(labels)), labels, fontsize=10)
    ax.set_ylim(len(labels) - .45, -.6)
    ax.set_xlabel("Paired log loss difference: score configuration minus default XGBoost", labelpad=12)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=12)
    ax.grid(axis='x')
    ax.set_axisbelow(True)
    legend = [Line2D([], [], color=color, marker=marker, lw=2, label=f"Mean block {length}")
              for color, marker, length in zip(colors, markers, [1, 3, 7])]
    ax.legend(handles=legend, loc='lower left', bbox_to_anchor=(0, 1.02), ncols=3,
              frameon=False, fontsize=10, borderaxespad=0)
    decorate(fig, "Historical forecast differences",
             "Development-selected score configurations · fixed default XGBoost reference · lower difference favors score model",
             "Lines: 2.5%–97.5% stationary-bootstrap percentiles, 2,000 paired resamples per setting.\n"
             "Blocks count observed UTC match-days. Historical cohort reused; pointwise exploratory ranges, no significance claim.\n"
             f"Run {report['run_id']} · frozen predictions, no refitting · Football-Data 2022/23–2024/25",
             bottom=.22, top=.76)
    fig.subplots_adjust(left=.34, right=.96)
    return save(fig, output, 'outcome-comparison')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=Path('results/outcome-comparison.json'))
    parser.add_argument('--output', type=Path, default=Path('results'))
    args = parser.parse_args()
    render(json.loads(args.report.read_text()), args.output)


if __name__ == '__main__':
    main()
