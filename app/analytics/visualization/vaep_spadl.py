"""Aggregate temporal validation and descriptive incremental SPADL values."""

import json
from pathlib import Path

from .style import COLORS, apply_style, decorate, save


def figures(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for row, metric in enumerate(('log_loss', 'brier')):
        bound = max(abs(v) for league in report['leagues'] for comparison in league['paired_nonterminal_intervals'].values()
                    for head in ('score', 'concede') for v in comparison[head][metric]['interval95']) * 1.25
        for col, head in enumerate(('score', 'concede')):
            ax = axes[row, col]
            labels = []
            for league in report['leagues']:
                for baseline, label in [('prior', 'training frequency'), ('single', 'one action')]:
                    i = len(labels); labels.append(f"{league['league']} vs {label}")
                    values = league['paired_nonterminal_intervals'][baseline][head][metric]
                    low, high = values['interval95']; color = COLORS['blue'] if baseline == 'prior' else COLORS['orange']
                    ax.hlines(i, low, high, color=color, linewidth=2)
                    ax.scatter(values['difference'], i, color=color, s=40, zorder=3)
            ax.axvline(0, color=COLORS['neutral'], linestyle=':', linewidth=1.3)
            ax.set_xlim(-bound, bound); ax.set_yticks(range(4), labels); ax.set_ylim(3.6, -.6)
            ax.set_title(('Scoring' if head == 'score' else 'Conceding') + (' | log loss' if row == 0 else ' | binary Brier'))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
            ax.set_xlabel('Loss difference (lower is better)'); ax.grid(axis='x')
    decorate(fig, 'Do recent SPADL actions improve state probabilities?', 'Wyscout 2017/18 | whole-match calendar holdouts from 2018-03-01 | current goals excluded here',
             'Dots are paired mean loss differences; lines are 95% intervals from 1,000 whole-match bootstrap samples.\nModels are fixed before test evaluation. Intervals do not include training uncertainty or dependence between repeat teams.\nPost-action inputs and a ten-action window (including the current action); not the existing native pre-pass task.', bottom=.19, top=.83)
    fig.subplots_adjust(left=.22, right=.97, wspace=.65, hspace=.63)
    save(fig, output, 'Figure_1', formats=('png', 'svg', 'pdf'))

    types = ['pass', 'cross', 'dribble', 'interception', 'tackle', 'clearance', 'foul', 'shot', 'keeper_save']
    names = ['Pass', 'Cross', 'Dribble (converter)', 'Interception', 'Tackle', 'Clearance', 'Foul', 'Shot', 'Keeper save']
    types = [kind for kind in types if all(kind in league['test_action_values'] for league in report['leagues'])]
    labels = [dict(zip(['pass', 'cross', 'dribble', 'interception', 'tackle', 'clearance', 'foul', 'shot', 'keeper_save'], names))[kind] for kind in types]
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharex=True, sharey=True)
    values = [league['test_action_values'][kind][key] for league in report['leagues'] for kind in types
              for key in ('offensive_value', 'defensive_value', 'vaep_value')]
    low, high = min(0., min(values)), max(0., max(values)); margin = (high - low) * .15
    for ax, league in zip(axes, report['leagues']):
        for i, kind in enumerate(types):
            values = league['test_action_values'][kind]
            for key, color, marker, offset in [('offensive_value', COLORS['blue'], 'o', -.14),
                                             ('defensive_value', COLORS['orange'], 's', .14),
                                             ('vaep_value', COLORS['neutral'], 'D', 0)]:
                ax.scatter(values[key], i + offset, color=color, marker=marker, s=32, zorder=3,
                           label={'offensive_value': 'Offensive', 'defensive_value': 'Defensive', 'vaep_value': 'Total'}[key] if i == 0 else None)
            ax.text(1.02, i, f"n={values['actions']:,}", transform=ax.get_yaxis_transform(), va='center', fontsize=9)
        ax.axvline(0, color=COLORS['muted'], linestyle=':', linewidth=1.2)
        ax.set_xlim(low - margin, high + margin); ax.set_yticks(range(len(types)), labels); ax.set_ylim(len(types) - .5, -.6)
        ax.set_title(league['league']); ax.set_xlabel('Mean change in goal probability per action'); ax.grid(axis='x')
    axes[0].legend(loc='lower left', bbox_to_anchor=(0, 1.06), ncol=3, frameon=False, fontsize=10)
    inferred = ', '.join(f"{r['league']} {sum(g['inferred'] for g in r['test_action_values'].values()):,}" for r in report['leagues'])
    decorate(fig, 'State value becomes an action increment', 'Three-action model | offensive = scoring probability change | defensive = reduction in conceding probability',
             f'Held-out action-type means, not player rankings or causal effects. Action mix and calibration affect these values; no confidence intervals shown.\nThe first action of each period and incomplete ten-action windows are excluded. Inferred dribbles in the value cohort: {inferred}.\nOfficial penalty/corner priors are retained; period boundaries and post-own-goal baselines reset explicitly.', bottom=.23, top=.79)
    fig.subplots_adjust(left=.18, right=.9, wspace=.55)
    save(fig, output, 'Figure_2', formats=('png', 'svg', 'pdf'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', default='results/vaep-spadl.json')
    parser.add_argument('--output', default='results/vaep-spadl')
    args = parser.parse_args()
    figures(json.loads(Path(args.report).read_text()), Path(args.output))
