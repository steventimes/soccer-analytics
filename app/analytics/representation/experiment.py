"""Offline representation benchmark; forecasts and model artifacts are untouched."""

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from app.file_io import write_json_atomic
from .data import load_dataset


def metrics(labels, prediction):
    from sklearn.metrics import (
        brier_score_loss,
        log_loss,
        mean_absolute_error,
        roc_auc_score,
    )

    probability = np.clip(prediction[:, 0], 1e-7, 1 - 1e-7)
    return {
        "completion_log_loss": float(
            log_loss(labels[:, 0], probability, labels=[0, 1])
        ),
        "completion_brier": float(brier_score_loss(labels[:, 0], probability)),
        "completion_auc": (
            float(roc_auc_score(labels[:, 0], probability))
            if len(np.unique(labels[:, 0])) == 2
            else None
        ),
        "forward_displacement_mae_pitch_pct": float(
            mean_absolute_error(labels[:, 1], prediction[:, 1]) * 100
        ),
    }


def linear_inputs(data):
    tokens, values = data[:2]
    one_hot = np.eye(11, dtype=np.float32)[tokens]
    return np.concatenate((values, one_hot), axis=-1).reshape(len(tokens), -1)


def baselines(data, *, test_predictions=None):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train, valid, test = (data[key] for key in ("train", "validation", "test"))
    average = train[2].mean(axis=0)
    constant = {
        key: metrics(value[2], np.tile(average, (len(value[0]), 1)))
        for key, value in (("validation", valid), ("test", test))
    }
    completion = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=600, C=0.1)
    )
    progression = make_pipeline(StandardScaler(), Ridge(alpha=10))
    completion.fit(linear_inputs(train), train[2][:, 0])
    progression.fit(linear_inputs(train), train[2][:, 1])
    linear_predictions = {
        key: np.column_stack((completion.predict_proba(linear_inputs(value))[:, 1],
                              progression.predict(linear_inputs(value))))
        for key, value in (("validation", valid), ("test", test))
    }
    linear = {key: metrics(value[2], linear_predictions[key])
              for key, value in (("validation", valid), ("test", test))}
    tree_options = dict(max_iter=100, max_leaf_nodes=15, learning_rate=.05,
                        l2_regularization=1., early_stopping=False, random_state=11)
    tree_completion = HistGradientBoostingClassifier(**tree_options)
    tree_progression = HistGradientBoostingRegressor(**tree_options)
    tree_completion.fit(linear_inputs(train), train[2][:, 0])
    tree_progression.fit(linear_inputs(train), train[2][:, 1])
    tree_predictions = {
        key: np.column_stack((
            tree_completion.predict_proba(linear_inputs(value))[:, 1],
            tree_progression.predict(linear_inputs(value)),
        ))
        for key, value in (("validation", valid), ("test", test))
    }
    tree = {key: metrics(value[2], tree_predictions[key])
            for key, value in (("validation", valid), ("test", test))}
    if test_predictions is not None:
        test_predictions.update({
            "constant_train_prior": [np.tile(average, (len(test[0]), 1))],
            "linear_context": [linear_predictions['test']],
            "hist_gradient_boosting_context": [tree_predictions['test']],
        })
    return [
        {"method": "constant_train_prior", "seed": None, **constant},
        {"method": "linear_context", "seed": None, **linear},
        {"method": "hist_gradient_boosting_context", "seed": None, **tree},
    ]


def train_seed(data, seed, epochs, output_dir=None, *, test_predictions=None):
    import torch
    from .models import ContextMLP, PassModel, fit_supervised, predict, pretrain_masked

    torch.manual_seed(seed)
    train, valid, test = (data[key] for key in ("train", "validation", "test"))
    rows, curves = [], {}

    def new_model(encoder=None):
        torch.manual_seed(seed)
        return PassModel(copy.deepcopy(encoder) if encoder is not None else None)

    def record(name, model, history, *, evaluation=None):
        evaluation = evaluation or {"validation": valid, "test": test}
        predictions = {key: predict(model, value) for key, value in evaluation.items()}
        rows.append(
            {
                "method": name,
                "seed": seed,
                **{
                    key: metrics(value[2], predictions[key])
                    for key, value in evaluation.items()
                },
            }
        )
        if test_predictions is not None:
            test_predictions.setdefault(name, []).append(predictions['test'])
        curves[name] = history
        if output_dir is not None and name in {
            "shared_multi_task",
            "self_supervised_finetune",
        }:
            # Numeric arrays only: downstream reuse needs no pickle deserialization.
            np.savez_compressed(
                Path(output_dir) / f"{name}-seed-{seed}.npz",
                **{
                    key: value.detach().numpy()
                    for key, value in model.state_dict().items()
                },
            )

    # Separate single-task encoders give both task scores, at twice encoder cost.
    singles, single_curves = [], {}
    for task in ("completion", "progression"):
        model = new_model()
        single_curves[task] = fit_supervised(
            model, train, epochs=epochs, seed=seed, validation=valid, task=task
        )
        singles.append(model)
    scores = {}
    for key, value in (("validation", valid), ("test", test)):
        predictions = np.column_stack(
            (predict(singles[0], value)[:, 0], predict(singles[1], value)[:, 1])
        )
        scores[key] = metrics(value[2], predictions)
        if key == 'test' and test_predictions is not None:
            test_predictions.setdefault('separate_single_task', []).append(predictions)
    rows.append({"method": "separate_single_task", "seed": seed, **scores})
    curves["separate_single_task"] = single_curves

    shared = new_model()
    record(
        "shared_multi_task",
        shared,
        fit_supervised(shared, train, epochs=epochs, seed=seed, validation=valid),
    )
    frame = {
        key: (value[0].copy(), value[1].copy(), value[2]) for key, value in data.items()
    }
    for tokens, values, _ in frame.values():
        tokens[:, :-1] = 0
        values[:, :-1] = 0
    model = new_model()
    record(
        "current_event_only",
        model,
        fit_supervised(
            model,
            frame["train"],
            epochs=epochs,
            seed=seed,
            validation=frame["validation"],
        ),
        evaluation={"validation": frame["validation"], "test": frame["test"]},
    )

    torch.manual_seed(seed)
    mlp = ContextMLP()
    record("mlp_context", mlp, fit_supervised(
        mlp, train, epochs=epochs, seed=seed, validation=valid,
    ))

    source = new_model()
    curves["source_pretrain"] = fit_supervised(
        source, data["source"], epochs=epochs, seed=seed
    )
    record("source_zero_shot", source, curves["source_pretrain"])
    ssl_encoder, ssl_history = pretrain_masked(data["source"], epochs=epochs, seed=seed)
    curves["ssl_pretrain"] = ssl_history
    for prefix, encoder in (
        ("supervised_transfer", source.encoder),
        ("self_supervised", ssl_encoder),
    ):
        for frozen in (True, False):
            name = prefix + ("_frozen" if frozen else "_finetune")
            model = new_model(encoder)
            history = fit_supervised(
                model, train, epochs=epochs, seed=seed, validation=valid, freeze=frozen
            )
            record(name, model, history)
    return rows, curves


def aggregate(rows):
    methods = list(dict.fromkeys(row["method"] for row in rows))
    summaries = []
    for method in methods:
        group = [row for row in rows if row["method"] == method]
        summary = {"method": method, "runs": len(group)}
        for split in ("validation", "test"):
            summary[split] = {}
            for metric in group[0][split]:
                values = [
                    row[split][metric]
                    for row in group
                    if row[split][metric] is not None
                ]
                summary[split][metric] = (
                    {
                        "mean": float(np.mean(values)),
                        "seed_sd": (
                            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                        ),
                    }
                    if values
                    else None
                )
        summaries.append(summary)
    return summaries


def figures(report, curves, output):
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(Path(".tmp/matplotlib").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    from ..visualization.style import COLORS, apply_style, decorate, save

    apply_style()
    labels = {
        "constant_train_prior": "Constant training prior",
        "linear_context": "Linear context",
        "hist_gradient_boosting_context": "Gradient-boosted trees",
        "mlp_context": "Feedforward MLP context",
        "separate_single_task": "Separate single tasks",
        "shared_multi_task": "Shared multi-task",
        "current_event_only": "Current event only",
        "source_zero_shot": "Source model, zero-shot",
        "supervised_transfer_frozen": "Supervised transfer · frozen",
        "supervised_transfer_finetune": "Supervised transfer · fine-tuned",
        "self_supervised_frozen": "Self-supervised · frozen",
        "self_supervised_finetune": "Self-supervised · fine-tuned",
    }
    summaries = report["summaries"]
    fig, axes = plt.subplots(1, 2, figsize=(17.5, max(8.3, 4.3 + .4 * len(summaries))), sharey=True)
    for ax, metric, title in zip(
        axes,
        ("completion_log_loss", "forward_displacement_mae_pitch_pct"),
        (
            "Pass completion",
            "Forward displacement",
        ),
    ):
        means = [s["test"][metric]["mean"] for s in summaries]
        deviations = [s["test"][metric]["seed_sd"] for s in summaries]
        y = np.arange(len(summaries))
        for i, (summary, mean, sd) in enumerate(zip(summaries, means, deviations)):
            fixed = summary["runs"] == 1
            ax.errorbar(mean, i, xerr=sd, fmt="D" if fixed else "o", color=COLORS["neutral"] if fixed else COLORS["blue"], markerfacecolor="white" if fixed else COLORS["blue"], markersize=6, capsize=3, elinewidth=1.6, zorder=3)
            precision = 5 if metric == "completion_log_loss" else 3
            value = f"{mean:.{precision}f}" + (" · fixed" if fixed else f" ± {sd:.{precision}f}")
            ax.text(1.035, i, value, transform=ax.get_yaxis_transform(), va="center", fontsize=10)
        ax.text(1.035, 1.04, "Mean ± seed SD", transform=ax.transAxes, fontsize=10, color=COLORS["muted"])
        ax.set_yticks(y, [labels.get(s["method"], s["method"].replace("_", " ").title()) for s in summaries])
        ax.set_title(title, loc="left", pad=26)
        ax.set_xlabel("Log loss · lower is better" if metric == "completion_log_loss" else "MAE (% of pitch length) · lower is better")
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.margins(x=.12)
        ax.xaxis.set_major_locator(MaxNLocator(4))
    axes[0].set_ylim(len(summaries) - .5, -.6)
    test = report["data"]["splits"]["test"]
    seeds = report["configuration"]["seeds"]
    decorate(
        fig, "Representation learning on held-out passes",
        f"Wyscout 2017/18 · {report['data']['target_league']} · {test['sequences_used']:,} sampled pass sequences from {test['matches_used']} test matches",
        f"Neural models: mean ± 1 seed SD across {len(seeds)} seeds ({', '.join(map(str, seeds))}). SD is optimizer variability, not a confidence interval.\nOpen diamonds: {sum(row['runs'] == 1 for row in summaries)} methods fitted once. Both panels keep method order; axes have different units and scales.\nDisplacement measures recorded endpoint minus start position, not intended destination or chance value.",
        bottom=.23, top=.77,
    )
    fig.subplots_adjust(left=.20, right=.85, wspace=.8)
    save(fig, output, "Figure_1")
    fig, ax = plt.subplots(figsize=(12, 6.5))
    seed = seeds[0]
    first = curves[str(seed)]
    traces = []
    for name, color, linestyle in (
        ("shared_multi_task", COLORS["blue"], "-"),
        ("supervised_transfer_finetune", COLORS["orange"], "--"),
        ("self_supervised_finetune", COLORS["muted"], ":"),
    ):
        trace = first[name]
        epochs = [r["epoch"] for r in trace]
        losses = [r["validation_loss"] for r in trace]
        ax.plot(epochs, losses, color=color, linestyle=linestyle, linewidth=2)
        ax.scatter(epochs[-1], losses[-1], s=25, color=color, zorder=3)
        traces.append((name, color, epochs[-1], losses[-1]))
    final_epoch = max(row[2] for row in traces)
    for name, color, epoch, loss in traces:
        ax.annotate(f"{labels[name]}\nFinal epoch {loss:.5f}", (epoch, loss), xytext=(15, 0), textcoords="offset points", va="center", color=color, fontsize=11)
    ax.set(
        xlabel="Target-training epoch",
        ylabel="Validation loss · completion BCE + displacement MSE",
        xlim=(1, final_epoch),
    )
    ax.set_xticks(sorted({1, final_epoch, *range(5, final_epoch, 5)}))
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    decorate(
        fig, "Validation loss during target training",
        f"Wyscout 2017/18 · {report['data']['target_league']} · Three neural training paths · Illustrative seed {seed}",
        "Lower validation loss is better. Checkpoints are selected on validation loss; test outcomes do not select epochs.\nOnly one seed is plotted here. Figure 1 reports all methods and optimizer variability across the complete seed set.",
        bottom=.23, top=.77,
    )
    fig.subplots_adjust(left=.10, right=.68)
    save(fig, output, "Figure_2")


def run(
    database,
    output_dir,
    *,
    target_league="England",
    train_cap=12000,
    epochs=25,
    seeds=(11, 29, 47),
):
    import torch
    import sklearn
    import scipy
    from threadpoolctl import threadpool_limits
    from .models import BATCH_SIZE, HIDDEN, LEARNING_RATE, MASK_RATE

    if epochs <= 0 or not seeds:
        raise ValueError("Positive epochs and at least one seed are required")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    print(
        "Loading same-provider event sequences; source data excludes target validation/test dates.",
        flush=True,
    )
    data, receipt = load_dataset(database, target_league, train_cap)
    config = {
        "target_league": target_league,
        "train_cap": train_cap,
        "epochs": epochs,
        "seeds": list(seeds),
        "encoder": "GRU",
        "hidden_size": HIDDEN,
        "event_embedding_size": 8,
        "sequence_length": 8,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "optimizer": "Adam",
        "mask_rate": MASK_RATE,
        "multi_task_loss": "completion BCE + forward displacement MSE, unit weights",
        "tabular_baselines": {
            "inputs": "Same eight events and observed fields; flattened numeric values and eleven-way event-type one-hot (including padding).",
            "tree": "Separate classification/regression estimators;100 boosting iterations,15 leaves,learning rate0.05,L2=1,seed11;no internal validation or early stopping.",
            "mlp": "One32-unit ReLU layer and shared two-task head;Adam,batch size,epochs,loss and validation checkpoint selection match shared GRU;train-only;no source pretraining.",
        },
    }
    code = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("data.py", "models.py", "experiment.py", "comparison.py")
    }
    environment = {
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "sklearn": sklearn.__version__,
        "scipy": scipy.__version__,
        "device": "cpu", "torch_threads": torch.get_num_threads(),
    }
    run_id = hashlib.sha256(
        json.dumps(
            {"data": receipt, "config": config, "code": code, "environment": environment}, sort_keys=True
        ).encode()
    ).hexdigest()[:16]
    with threadpool_limits(limits=2):
        print("Fixed tabular baselines use target training only; trees do not create a random validation split.", flush=True)
        test_predictions = {}
        rows = baselines(data, test_predictions=test_predictions)
        curves = {}
        for seed in seeds:
            print(
                f"Seed {seed}: feedforward MLP, separate/shared GRU tasks, supervised transfer, masked-event SSL, frozen and fine-tuned encoders.",
                flush=True,
            )
            seed_rows, seed_curves = train_seed(data, seed, epochs, output,
                                               test_predictions=test_predictions)
            rows.extend(seed_rows)
            curves[str(seed)] = seed_curves
            write_json_atomic(
                output / f"seed-{seed}.json", {"rows": seed_rows, "curves": seed_curves}
            )
    from .comparison import compare_test_predictions
    print('Paired whole-match resampling of fixed test losses; no refitting or test selection.', flush=True)
    test_comparisons = compare_test_predictions(
        data['test'][2], receipt['splits']['test']['sample_match_ids'], test_predictions)
    np.savez_compressed(output / 'test-predictions.npz', labels=data['test'][2],
                        match_ids=np.asarray(receipt['splits']['test']['sample_match_ids']),
                        **{f'{name}__{i}': prediction for name, runs in test_predictions.items()
                           for i, prediction in enumerate(runs)})
    report = {
        "run_id": run_id,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "configuration": config,
        "code_sha256": code,
        "environment": environment,
        "test_comparisons": test_comparisons,
        "data": receipt,
        "runs": rows,
        "summaries": aggregate(rows),
        "input_fields": [
            "event type",
            "event-start x/y in current-team frame",
            "time gap to current event",
            "same-team flag",
        ],
        "targets": [
            "recorded pass completion",
            "recorded endpoint x minus start x / pitch length",
        ],
        "limits": [
            "Historical single-season first benchmark; cross-season transfer not yet tested.",
            "Events lack continuous off-ball trajectories and complete pressure observations.",
            "Forward displacement is an observed geometry target, not intended pass destination or chance value.",
            "Validation selects epochs; fixed hyperparameters are not tuned on test scores.",
            "Seed SD describes optimizer variability, not sampling uncertainty or causal effects.",
            "Transfer methods consume extra source-training compute; separate tasks use two encoders.",
            "Tree and MLP baselines use the same observable history with fixed one-hot types; GRU learns type embeddings. Architecture/compute budgets are not identical.",
            "Representations are learned from observed events; they do not identify tactics or player potential by themselves.",
        ],
    }
    write_json_atomic(output / "summary.json", report)
    figures(report, curves, output)
    (output / "notes.txt").write_text(
        "Chronological Wyscout 2017/18 representation benchmark.\n"
        + "\n".join(report["limits"])
        + "\n",
        encoding="utf-8",
    )
    for row in report["summaries"]:
        print(
            f'{row["method"]}: test completion log loss {row["test"]["completion_log_loss"]["mean"]:.4f}, displacement MAE {row["test"]["forward_displacement_mae_pitch_pct"]["mean"]:.3f}%',
            flush=True,
        )
    return report
