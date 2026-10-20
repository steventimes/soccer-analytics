"""Joint probability arrows and binning sensitivity for frozen match forecasts."""

import argparse
import json
from pathlib import Path

from .style import COLORS, apply_style, decorate, save


def render(report, output, *, competition='PL', model='xgboost'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter
    import numpy as np

    if report.get('schema') != 'SoccerAnalytics.CalibrationDiagnosticStudy.v1':
        raise ValueError('Expected a frozen calibration diagnostic study')
    group = next((g for g in report['groups'] if (g['competition'], g['model']) == (competition, model)), None)
    if group is None:
        raise ValueError('Requested diagnostic group is absent')
    methods = ['identity', 'temperature', 'dirichlet_l2']
    labels = ['Uncalibrated', 'Temperature', 'Dirichlet L2']
    colors = [COLORS['blue'], COLORS['orange'], COLORS['muted']]
    selected = [next(r for r in group['results'] if r['calibrator'] == method) for method in methods]
    apply_style()
    fig = plt.figure(figsize=(11.8, 9.4))
    decorate(fig, 'Fixed forecast calibration diagnostics',
             f"{competition} · {model} · {selected[0]['rows']} reused holdout matches · model and calibration maps fixed",
             'Arrows show observed frequency minus predicted mean; lengths are unscaled. Circle area is proportional to bin count.\n'
             'Top panels: axis resolution B=10; only cells with at least 10 matches shown. Joint TV uses every match, including hidden cells.\n'
             'Bottom: marginal ECE averages three scalar errors; it is a different statistic from joint total variation (TV).\n'
             f"Descriptive estimates without intervals or calibration certificate · run {report['run_id']}", bottom=.22, top=.79)
    fig.text(.065, .845, 'Open circle: mean prediction   →   orange point: observed class frequencies',
             fontsize=10, color=COLORS['muted'])
    grid = fig.add_gridspec(2, 3, left=.10, right=.96, bottom=.23, top=.79,
                          height_ratios=[1.25, 1], hspace=.65, wspace=.25)
    vertices = np.array([[0., 0.], [.5, np.sqrt(3) / 2], [1., 0.]])
    for column, (label, row) in enumerate(zip(labels, selected)):
        ax = fig.add_subplot(grid[0, column])
        joint = next(r for r in row['results'] if r['resolution'] == 10)['joint_tv']
        ax.plot(*vertices[[0, 1, 2, 0]].T, color=COLORS['neutral'], lw=1)
        for value in [.25, .5, .75]:
            for k in range(3):
                others = [j for j in range(3) if j != k]
                ends = value * vertices[k] + (1 - value) * vertices[others]
                ax.plot(*ends.T, color=COLORS['grid'], lw=.6, zorder=0)
        for cell in joint['cells']:
            predicted = np.array(cell['mean_probability']) @ vertices
            observed = np.array(cell['observed_rate']) @ vertices
            ax.scatter(*predicted, s=3 * cell['count'], facecolors='none', edgecolors=COLORS['blue'], lw=.8, zorder=2)
            ax.annotate('', xy=observed, xytext=predicted,
                        arrowprops={'arrowstyle': '->', 'lw': 1.0, 'color': COLORS['ink'], 'shrinkA': 0, 'shrinkB': 0})
            ax.scatter(*observed, s=16, color=COLORS['orange'], zorder=3)
        ax.text(-.02, -.055, 'Away win', ha='left', fontsize=10)
        ax.text(1.02, -.055, 'Home win', ha='right', fontsize=10)
        ax.text(.5, np.sqrt(3) / 2 + .04, 'Draw', ha='center', fontsize=10)
        ax.text(.5, -.18, f"Joint TV {joint['ece']:.3f} · all {row['rows']} matches", ha='center', fontsize=10)
        ax.set_title(f"{label}\n{joint['displayed_rows']}/{row['rows']} matches shown", fontsize=11, pad=10)
        if not joint['cells']:
            ax.text(.5, .35, 'No cells meet display threshold', ha='center', fontsize=9)
        ax.set(xlim=(-.07, 1.07), ylim=(-.21, 1.01), aspect='equal')
        ax.axis('off')
    ax = fig.add_subplot(grid[1, :])
    handles = []
    for label, color, row in zip(labels, colors, selected):
        for strategy, linestyle, marker in [('equal_width', '-', 'o'), ('equal_mass', '--', 's')]:
            xs = [r['resolution'] for r in row['results']]
            ax.plot(xs, [r[strategy]['classwise_ece'] for r in row['results']], color=color,
                    ls=linestyle, marker=marker, lw=1.6, markersize=5)
            handles.append(Line2D([], [], color=color, ls=linestyle, marker=marker, lw=1.6,
                                  label=f"{label} · {'width' if strategy == 'equal_width' else 'mass'}"))
    ax.set_xticks(xs)
    ax.set_xlabel('Requested scalar bins (mass bins can merge to preserve tied scores)')
    ax.set_ylabel('Classwise ECE\n(percentage points)')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{100 * value:g}'))
    ax.set_ylim(bottom=0)
    ax.grid(axis='y'); ax.set_axisbelow(True)
    ax.legend(handles=handles, ncols=3, loc='lower left', bbox_to_anchor=(0, 1.02),
              borderaxespad=0, frameon=False, fontsize=9)
    return save(fig, output, 'calibration-diagnostics', formats=('png', 'svg', 'pdf'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=Path('results/calibration-diagnostics.json'))
    parser.add_argument('--output', type=Path, default=Path('results'))
    parser.add_argument('--competition', default='PL')
    parser.add_argument('--model', default='xgboost')
    args = parser.parse_args()
    render(json.loads(args.report.read_text()), args.output, competition=args.competition, model=args.model)


if __name__ == '__main__':
    main()
