"""Fixed-output contribution distributions, dependence and explicit pair effects."""

import csv
from pathlib import Path

import numpy as np

from .style import COLORS, apply_style, decorate, save


def figures(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    features = sorted(report['features'], key=lambda x: x['mean_absolute_contribution'], reverse=True)
    subtitle = (f"{report.get('source', 'supplied match records')} | {report.get('competition', 'local cohort')} | fixed output {report['fixed_output']} | {report['rows']} holdout rows\n"
                f"{report['kickoffs'][0][:10]} to {report['kickoffs'][-1][:10]} | model {report['identity']['model_sha256'][:12]} | run {report['run_id']}")
    apply_style()
    shown = features[:12]
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, row in enumerate(shown):
        ax.scatter(row['contributions'], np.full(report['rows'], i), s=12, alpha=.4, color=COLORS['blue'])
    ax.axvline(0, color=COLORS['muted'], linewidth=1)
    ax.set_yticks(range(len(shown)), [r['feature'].replace('_', ' ') for r in shown])
    ax.invert_yaxis(); ax.set_xlabel('Contribution (raw margin)'); ax.grid(axis='x')
    note = 'Dots are matches, with one fixed output across the cohort. Native tree path statistics; no causal or probability-point interpretation.\nTop 12 features shown; the CSV contains every feature, original input value, contribution, main effect and baseline.'
    decorate(fig, 'Fixed-output feature contributions',
             subtitle, note, bottom=.2, top=.79)
    fig.subplots_adjust(left=.28)
    paths = save(fig, output, 'global-contributions')
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, row in zip(axes, features[:2]):
        ax.scatter(row['raw_values'], row['contributions'], color=COLORS['blue'], s=16, alpha=.5)
        ax.axhline(0, color=COLORS['muted'], linewidth=1)
        ax.set_xlabel(f"{row['feature'].replace('_', ' ')} ({row['input_unit']})")
        ax.set_ylabel('Contribution (raw margin)'); ax.grid(axis='y')
    decorate(fig, 'Feature values and contributions', subtitle,
             'The two features with highest mean absolute contribution are displayed. Each point is one observed row.\nThis is an association display, not partial dependence, a causal response or a feature interaction test.', bottom=.25, top=.75)
    paths += save(fig, output, 'feature-dependence')
    names = [r['feature'] for r in features[:8]]
    matrix = np.zeros((len(names), len(names)))
    for row in report['interaction_pairs']:
        if row['feature_a'] in names and row['feature_b'] in names:
            i, j = names.index(row['feature_a']), names.index(row['feature_b'])
            matrix[i, j] = matrix[j, i] = row['mean_absolute_pair_total']
    for i, row in enumerate(features[:8]):
        matrix[i, i] = np.mean(np.abs(row['main_effects']))
    fig, ax = plt.subplots(figsize=(12, 9))
    image = ax.imshow(matrix, cmap='Blues', vmin=0, interpolation='nearest')
    ax.set_xticks(range(len(names)), [n.replace('_', '\n') for n in names], fontsize=9)
    ax.set_yticks(range(len(names)), [n.replace('_', ' ') for n in names], fontsize=10)
    ax.grid(False)
    fig.colorbar(image, ax=ax, label='Mean absolute raw-margin effect', fraction=.04)
    decorate(fig, 'Native TreeSHAP interactions', subtitle,
             'Diagonal: mean absolute main effect. Off diagonal: mean absolute total pair interaction (twice one native half-cell).\nAll feature pairs, including omitted features, remain in interaction-pairs.csv. Statistical significance and causal effects are not inferred.', bottom=.23, top=.79)
    fig.subplots_adjust(left=.28)
    paths += save(fig, output, 'feature-interactions')
    with (output / 'contributions.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(['run_id', 'model_sha256', 'feature_dependence', 'output_unit', 'row', 'kickoff', 'fixed_output', 'feature', 'input_unit', 'raw_value', 'contribution', 'main_effect', 'baseline', 'raw_output'])
        for i, kickoff in enumerate(report['kickoffs']):
            for row in report['features']:
                writer.writerow([report['run_id'], report['identity']['model_sha256'], report['feature_dependence'], report['output_unit'], i, kickoff, report['fixed_output'], row['feature'], row['input_unit'], row['raw_values'][i], row['contributions'][i], row['main_effects'][i], report['baseline'][i], report['raw_outputs'][i]])
    with (output / 'interaction-pairs.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        fields = ['run_id', 'model_sha256', 'fixed_output', 'output_unit', 'rows', 'first_kickoff', 'last_kickoff', 'feature_a', 'feature_b', 'mean_pair_total', 'mean_absolute_pair_total']
        writer = csv.DictWriter(handle, fieldnames=fields)
        metadata = {'run_id': report['run_id'], 'model_sha256': report['identity']['model_sha256'],
                    'fixed_output': report['fixed_output'], 'output_unit': report['output_unit'], 'rows': report['rows'],
                    'first_kickoff': report['kickoffs'][0], 'last_kickoff': report['kickoffs'][-1]}
        writer.writeheader(); writer.writerows({**metadata, **row} for row in report['interaction_pairs'])
    return paths + [output / 'contributions.csv', output / 'interaction-pairs.csv']
