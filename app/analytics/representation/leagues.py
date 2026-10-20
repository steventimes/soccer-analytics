"""Fixed-hyperparameter, same-season league checks for existing model families.

The source warehouse is read-only. Per-row predictions and full receipts remain
in the explicitly selected local output directory; summary.json is aggregate.
"""

from datetime import datetime, UTC
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from app.file_io import write_json_atomic
from .comparison import compare_test_predictions
from .data import LEAGUES, load_dataset
from .experiment import aggregate, baselines, metrics


def evaluate_models(data, *, epochs, seeds):
    """Fit target-only models; validation chooses neural checkpoints."""
    if (type(epochs) is not int or epochs < 1 or not seeds
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('Positive epochs and distinct nonnegative integer seeds required')
    import torch
    from .models import ContextMLP, PassModel, fit_supervised, predict

    predictions, curves = {}, {}
    rows = baselines(data, test_predictions=predictions)
    for seed in seeds:
        curves[str(seed)] = {}
        for name, factory in [('mlp_context', ContextMLP), ('shared_multi_task', PassModel)]:
            torch.manual_seed(seed)
            model = factory()
            curves[str(seed)][name] = fit_supervised(
                model, data['train'], epochs=epochs, seed=seed, validation=data['validation'])
            outputs = {key: predict(model, data[key]) for key in ('validation', 'test')}
            rows.append({'method':name, 'seed':seed,
                         **{key:metrics(data[key][2], value) for key,value in outputs.items()}})
            predictions.setdefault(name, []).append(outputs['test'])
    return {'runs':rows, 'curves':curves, 'predictions':predictions}


def run(database, output_dir, *, train_cap=8000, epochs=15, seeds=(11, 29, 47)):
    import torch
    import sklearn
    import scipy
    from threadpoolctl import threadpool_limits

    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose an empty output directory to preserve existing experiments')
    if (type(train_cap) is not int or train_cap < 1 or type(epochs) is not int or epochs < 1
            or not seeds or any(type(s) is not int or s < 0 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('Invalid training cap, epochs or seeds')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    config = {'train_cap':train_cap, 'epochs':epochs, 'seeds':list(seeds),
              'leagues':list(LEAGUES), 'models':'Existing constant, linear, HistGradientBoosting, ContextMLP and shared multi-task GRU. No source pretraining.',
              'selection':'Existing fixed hyperparameters; target validation chooses neural checkpoints. Parameters were developed using England, before inspecting this extension.',
              'inputs':'Same eight observable event starts; one-hot event types for trees/MLP, learned type embeddings for GRU. Budgets and capacities differ.'}
    code = {name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('leagues.py','data.py','models.py','experiment.py','comparison.py')}
    environment = {'python':platform.python_version(), 'numpy':np.__version__,
                   'torch':torch.__version__, 'sklearn':sklearn.__version__,
                   'scipy':scipy.__version__, 'device':'cpu', 'torch_threads':2}
    report = {'schema':'SoccerAnalytics.RepresentationLeagues.v1',
              'observed_at':datetime.now(UTC).isoformat(), 'configuration':config,
              'code_sha256':code, 'environment':environment, 'leagues':[],
              'limits':[
                  'Single historical season, five target leagues; not cross-season transfer or an independent benchmark-design test.',
                  'England informed the existing architecture and hyperparameters; all league results retained without test-based retuning.',
                  'Fixed-seed mean losses and conditional whole-match percentile ranges; not ensembles, simultaneous coverage or training uncertainty.',
                  'No source pretraining; loader source sampling is retained to preserve the original target sample selection.',
                  'Repeated teams and cross-match dependence are not modelled. No football outcome predictor is replaced.',
              ]}
    with threadpool_limits(limits=2):
        for league in LEAGUES:
            print(f'{league}: loading chronological target data with original sampling.', flush=True)
            data, receipt = load_dataset(database, league, train_cap)
            print(f'{league}: fixed baselines and {len(seeds)} seeds x MLP/GRU, {epochs} epochs.', flush=True)
            result = evaluate_models(data, epochs=epochs, seeds=seeds)
            comparison = compare_test_predictions(data['test'][2], receipt['splits']['test']['sample_match_ids'], result['predictions'])
            comparison['limits'] = [limit for limit in comparison['limits'] if '先前测试对照' not in limit] + [
                '以原英格兰实验的树基线为参照扩展到五个联赛；参数未按本扩展的测试结果调整。多项范围仅作探索比较。']
            local = output/league.lower()
            local.mkdir()
            np.savez_compressed(local/'test-predictions.npz', labels=data['test'][2],
                                match_ids=np.asarray(receipt['splits']['test']['sample_match_ids']),
                                **{f'{name}__{i}':p for name,runs in result['predictions'].items() for i,p in enumerate(runs)})
            full = {'league':league,'data':receipt,'runs':result['runs'], 'curves':result['curves'],
                    'summaries':aggregate(result['runs']), 'test_comparisons':comparison}
            full['run_id'] = hashlib.sha256(json.dumps({'data':receipt,'config':config,'code':code,'environment':environment},sort_keys=True).encode()).hexdigest()[:16]
            write_json_atomic(local/'receipt.json', full)
            # Publish only aggregates; no match IDs, per-pass arrays, curves or weights.
            splits = {name:{key:block[key] for key in ('sequences_available','sequences_used','matches_used','date_start','date_end','completion_rate','label_eligibility')}
                      for name,block in receipt['splits'].items() if name != 'source'}
            report['leagues'].append({key:full[key] for key in ('league','run_id','summaries','test_comparisons')} | {
                'data':{'provider':receipt['provider'],'season':receipt['season'], 'inputs':receipt['inputs'],
                        'validation_start':receipt['validation_start'],'test_start':receipt['test_start'],'splits':splits}})
            print(f'{league}: completed run {full["run_id"]}.', flush=True)
    report['run_id'] = hashlib.sha256(json.dumps({'config':config,'code':code,'environment':environment,
                                                'runs':[r['run_id'] for r in report['leagues']]},sort_keys=True).encode()).hexdigest()[:16]
    write_json_atomic(output/'summary.json', report)
    figure(report, output)
    return report


def figure(report, output):
    import os
    os.environ.setdefault('MPLCONFIGDIR', str(Path('.tmp/matplotlib').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from ..visualization.style import COLORS, apply_style, decorate, save

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 9))
    metrics_and_labels = [('completion_log_loss','Completion log loss difference'),
                          ('forward_displacement_mae_pitch_pct','Displacement MAE difference\nPitch percentage points')]
    for ax, (metric,label) in zip(axes, metrics_and_labels):
        ax.axvline(0,color=COLORS['muted'],linewidth=1,linestyle='--')
        for y, league in enumerate(report['leagues']):
            if y % 2 == 0:
                ax.axhspan(y-.46,y+.46,color=COLORS['light'],zorder=0)
            rows = {r['method']:r for r in league['test_comparisons']['comparisons']}
            for name,offset,color,marker,legend in [('mlp_context',-.13,COLORS['blue'],'o','MLP'),
                                                   ('shared_multi_task',.13,COLORS['orange'],'s','GRU')]:
                value=rows[name][metric]
                ax.hlines(y+offset,value['lower'],value['upper'],color=color,linewidth=2)
                ax.scatter(value['difference'],y+offset,color=color,marker=marker,s=38,label=legend if y==0 else None,zorder=3)
        ax.set_yticks(range(len(report['leagues'])), [f'{r["league"]}  (n={r["data"]["splits"]["test"]["matches_used"]})' for r in report['leagues']])
        ax.set_ylim(len(report['leagues'])-.5,-.5)
        ax.set_xlabel(label)
        ax.grid(axis='x');ax.set_axisbelow(True)
        ax.set_title('Method minus target-trained tree · lower is better', fontsize=11)
        ax.margins(x=.10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper left',bbox_to_anchor=(.05,.82),ncol=2,frameon=False)
    decorate(fig,'Same-history tree, MLP and GRU models across five leagues',
             f'Wyscout 2017/18 · train/validation/test sampled cap {report["configuration"]["train_cap"]:,} each · {len(report["configuration"]["seeds"])} neural seeds · {report["configuration"]["epochs"]} epochs · run {report["run_id"]}',
             'Points: fixed-seed mean loss minus one fitted HistGradientBoosting baseline. Lines: pointwise 2.5–97.5% whole-match resampling ranges.\nThe models and sampled passes stay fixed. Ranges exclude training/selection uncertainty; repeated-team dependence is not modelled.\nEach league trains on its own pre-January data, validates in January–February and tests from March. England informed the original design.\nNo retuning, source pretraining or cross-season claim. n = sampled test matches; displacement is recorded endpoint minus start.',
             bottom=.35,top=.73)
    fig.subplots_adjust(left=.17,right=.97,wspace=.68)
    return save(fig,output,'Figure_1',formats=('png','svg','pdf'))


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',default='data/football.sqlite3')
    parser.add_argument('--output',required=True)
    parser.add_argument('--train-cap',type=int,default=8000)
    parser.add_argument('--epochs',type=int,default=15)
    parser.add_argument('--seeds',type=int,nargs='+',default=[11,29,47])
    args=parser.parse_args(argv)
    run(args.database,args.output,train_cap=args.train_cap,epochs=args.epochs,seeds=tuple(args.seeds))


if __name__ == '__main__':
    main()
