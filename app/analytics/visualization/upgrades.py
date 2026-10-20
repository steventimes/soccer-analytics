"""Render reviewed experiment aggregates, with optional native spatial surfaces."""

import argparse
import json
from pathlib import Path

from .style import COLORS, apply_style, decorate, save


METHODS = {
    "constant_train_prior": "Training prior",
    "xt_location_calibrated": "xT + location*",
    "logistic_context": "Logistic context",
    "tree_context": "Tree context",
    "gru_context": "GRU context",
}


def pressing(ax, data):
    total = data["chains"]
    if total <= 0 or sum(data["outcomes"].values()) != total:
        raise ValueError("Pressing outcomes must reconcile to all chains")
    left = 0
    labels = [("regain", "Regain", COLORS["blue"]),
              ("disruption", "Disruption", COLORS["orange"]),
              ("unknown", "Unknown", COLORS["neutral"])]
    for index, (key, label, color) in enumerate(labels):
        count = data["outcomes"][key]
        value = 100 * count / total
        ax.barh(.60, value, left=left, height=.23, color=color, edgecolor="white", lw=2)
        if value >= 12:
            foreground = "white" if key == "regain" else COLORS["ink"]
            ax.text(left + value / 2, .60, f"{value:.1f}%", ha="center", va="center", color=foreground, weight="bold", fontsize=11)
        ax.text(index / 3, -.07, f"{label}\n{count:,} ({value:.1f}%)", transform=ax.transAxes, fontsize=11, color=COLORS["ink"], linespacing=1.6)
        left += value
    ax.set(xlim=(0, 100), ylim=(.30, .90), yticks=[], xticks=[])
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.tick_params(axis="x", length=0)


def value_difference(ax, data):
    import numpy as np

    keys = ["xt_location_calibrated", "logistic_context", "constant_train_prior", "tree_context"]
    rows = [data["paired_match_bootstrap"]["comparisons"]["gru_context-minus-" + key]["score"] for key in keys]
    means = np.array([row["log_loss_difference"] for row in rows])
    bounds = np.array([row["ci95"] for row in rows])
    ax.errorbar(means, np.arange(4), xerr=np.vstack([means - bounds[:, 0], bounds[:, 1] - means]),
                fmt="o", color=COLORS["blue"], markersize=6, elinewidth=1.7, capsize=3)
    ax.set_yticks(range(4), [METHODS[key] for key in keys])
    ax.invert_yaxis()
    ax.axvline(0, color=COLORS["ink"], lw=1)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_xlabel("GRU minus baseline log loss")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def buildup_difference(ax, data):
    import numpy as np
    from matplotlib.ticker import MaxNLocator

    keys = ["augmented_minus_performance", "augmented_minus_carry_forward"]
    rows = [data["paired_test_mae_differences"][key] for key in keys]
    means = np.array([row["mean"] for row in rows])
    bounds = np.array([row["player_cluster_ci95"] for row in rows])
    ax.errorbar(means, np.arange(2), xerr=np.vstack([means-bounds[:, 0], bounds[:, 1]-means]),
                fmt="o", markersize=7, color=COLORS["blue"], elinewidth=1.8, capsize=3)
    ax.set_yticks(range(2), ["Performance only", "Carry-forward buildup"])
    ax.invert_yaxis()
    ax.set_ylim(1.6, -.6)
    ax.axvline(0, color=COLORS["ink"], lw=1)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_xlabel("Augmented minus baseline\nMAE of xGBuildup/90")
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def spatial_differences(ax, data):
    keys = ["nearest_vs_gaussian", "nearest_vs_arrival", "gaussian_vs_arrival"]
    values = [100 * data[key]["mean_absolute_team_share_difference"] for key in keys]
    ax.hlines(range(3), 0, values, color=COLORS["grid"], lw=2)
    ax.scatter(values, range(3), color=COLORS["blue"], s=45, zorder=3)
    for y, value in enumerate(values):
        ax.text(value + .5, y, f"{value:.2f}", va="center", fontsize=11)
    ax.set_yticks(range(3), ["Nearest / Gaussian", "Nearest / Arrival", "Gaussian / Arrival"])
    ax.invert_yaxis()
    ax.set_xlim(0, 21)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_xlabel("Mean absolute difference (percentage points)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def spatial_example(report, output, surfaces, frames):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    from .pitch import draw_pitch

    with np.load(surfaces, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    contexts = [json.loads(line) for line in Path(frames).read_text().splitlines() if line.strip()]
    count = len(contexts)
    if count != report["space"]["sampled_frames"] or count != len(arrays["frame_ids"]):
        raise ValueError("Spatial context and surface sample counts differ")
    grid = report["space"]["config"]["grid"]
    cells = grid["x_cells"] * grid["y_cells"]
    for key in ("nearest", "gaussian", "arrival"):
        if arrays[key].shape != (count, 2, cells) or not np.isfinite(arrays[key]).all():
            raise ValueError("Invalid spatial surface dimensions or values")
    for pair, key in [(('nearest', 'gaussian'), 'nearest_vs_gaussian'),
                      (('nearest', 'arrival'), 'nearest_vs_arrival'),
                      (('gaussian', 'arrival'), 'gaussian_vs_arrival')]:
        observed = float(np.abs(arrays[pair[0]] - arrays[pair[1]]).mean())
        expected = report["space"]["method_differences"][key]["mean_absolute_team_share_difference"]
        if not np.isclose(observed, expected, rtol=0, atol=1e-12):
            raise ValueError("Spatial surfaces do not reproduce the reviewed aggregate")
    for index, frame in enumerate(contexts):
        identity = (frame["match_id"], frame["frame"], frame["period"], frame["time"])
        native = tuple(arrays[key][index] for key in ("match_ids", "frame_ids", "periods", "times"))
        if identity != native:
            raise ValueError("Spatial context and surface identities differ")
    contrasts = np.abs(arrays["gaussian"] - arrays["arrival"]).mean(axis=(1, 2))
    selected = int(np.argsort(contrasts, kind="stable")[count // 2])
    frame = contexts[selected]
    home = report["space"]["method"]["team_order"].index("home")
    rows = report["space"]["config"]["grid"]["y_cells"]
    cols = report["space"]["config"]["grid"]["x_cells"]
    if arrays["targets"].shape != (rows * cols, 2):
        raise ValueError("Unexpected spatial target grid")
    # Verify the native row-major grid before reshaping the surface.
    xs = np.unique(arrays["targets"][:, 0]); ys = np.unique(arrays["targets"][:, 1])
    xx, yy = np.meshgrid(xs, ys)
    expected = np.column_stack([xx.ravel(), yy.ravel()])
    if not np.array_equal(arrays["targets"], expected):
        raise ValueError("Unexpected spatial target order")
    fig, axes = plt.subplots(1, 3, figsize=(13, 6.1))
    fig.subplots_adjust(left=.045, right=.955, bottom=.34, top=.75, wspace=.10)
    fig.text(.045, .95, "Three spatial models, one observed frame", fontsize=18, weight="bold", va="top")
    fig.text(.045, .88, "Metrica open sample · nominal 105 × 68 m grid · home-side control share", fontsize=11, color=COLORS["muted"])
    cmap = LinearSegmentedColormap.from_list("team_control", [COLORS["orange"], "#FAFBFC", COLORS["blue"]])
    for ax, key, label in zip(axes, ["nearest", "gaussian", "arrival"], ["Nearest player", "Gaussian influence", "Arrival competition"]):
        surface = arrays[key][selected, home].reshape(rows, cols)
        image = ax.imshow(surface, extent=(-52.5, 52.5, 34, -34), vmin=0, vmax=1, cmap=cmap, interpolation="nearest")
        draw_pitch(ax, 105, 68, centered=True, invert_y=True)
        for team, color, marker in [("home", COLORS["blue"], "o"), ("away", COLORS["orange"], "^")]:
            players = [player for player in frame["players"] if player["team"] == team]
            ax.scatter([p["x"] for p in players], [p["y"] for p in players], color=color, marker=marker, s=23, edgecolor="white", linewidth=.6)
        ax.scatter(frame["ball"]["x"], frame["ball"]["y"], marker="D", s=25, color=COLORS["ink"], edgecolor="white", linewidth=.7)
        ax.set(xticks=[], yticks=[], title=label)
        for spine in ax.spines.values(): spine.set_visible(False)
    cax = fig.add_axes([.29, .245, .42, .022])
    colorbar = fig.colorbar(image, cax=cax, orientation="horizontal", ticks=[0, .5, 1])
    colorbar.ax.set_xticklabels(["0% home", "50%", "100% home"])
    colorbar.outline.set_visible(False)
    fig.text(.045, .14, "Home: circles · Away: triangles · Ball: diamond. Attacking direction is unknown.", fontsize=10, color=COLORS["muted"])
    fig.text(.045, .075, "Frame nearest the median Gaussian/arrival difference among all 240 samples; no accuracy ground truth.\nFinite-coordinate tracking; supplier detection flags unavailable. Metrica Sports open data.", fontsize=10, color=COLORS["muted"], linespacing=1.6)
    paths = save(fig, output, "space-control")
    return paths, {"selection": "median Gaussian/arrival discrepancy", "anonymous_sample_index": selected,
                   "source_npz": str(surfaces), "source_context": str(frames)}


def render(report="results/analysis-upgrades.json", output="results", *, surfaces=None, frames=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if (surfaces is None) != (frames is None):
        raise ValueError("Spatial surfaces and frame contexts must be supplied together")
    report = json.loads(Path(report).read_text())
    output = Path(output)
    detail = output / "analysis-upgrades"
    apply_style()
    paths = []
    a, b, c, d = [report[key] for key in ("passing", "value", "space", "buildup")]
    prediction = d["understat"]["next_season_prediction"]

    fig, ax = plt.subplots(figsize=(10, 4.6))
    decorate(fig, "How often did pressing end in a regain?", "SkillCorner · 20 A-League matches · 1,567 distinct pressing chains", "Unknown outcomes remain in the total. These supplier labels describe this historical sample.", bottom=.28, top=.72)
    pressing(ax, a["pressing_chains"])
    paths += save(fig, detail, "pressing")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.1))
    decorate(fig, "Does sequence context improve goal prediction?", "Wyscout England 2017/18 · 12,000 held-out states · 103 score-positive and 14 concede-positive states", "Intervals: 95% paired match bootstrap, conditional on fitted models. Negative differences favour GRU.\n*xT + location uses extra training events. Model search and compute budgets differ.", bottom=.23, top=.74)
    keys = ["xt_location_calibrated", "logistic_context", "constant_train_prior", "gru_context", "tree_context"]
    lookup = {row["method"]: row for row in b["summaries"]}
    means = [lookup[key]["test"]["score"]["log_loss"]["mean"] for key in keys]
    axes[0].scatter(means, range(5), color=[COLORS["blue"] if key == "gru_context" else COLORS["neutral"] for key in keys], s=45)
    for y, mean in enumerate(means): axes[0].text(mean + .00016, y, f"{mean:.5f}", va="center", fontsize=10)
    spread = max(max(means) - min(means), .001)
    axes[0].set(yticks=range(5), yticklabels=[METHODS[key] for key in keys], xlim=(max(0, min(means) - .45*spread), max(means)+.65*spread), xlabel="Test score log loss · lower is better")
    axes[0].invert_yaxis(); axes[0].grid(axis="x"); axes[0].set_axisbelow(True)
    axes[0].set_title("Absolute error")
    value_difference(axes[1], b); axes[1].set_title("GRU difference and uncertainty")
    paths += save(fig, detail, "goal-probability")

    spatial_receipt = None
    if surfaces is not None and frames is not None:
        items, spatial_receipt = spatial_example(report, detail, surfaces, frames)
        paths += items
    else:
        fig, ax = plt.subplots(figsize=(10, 5.1))
        decorate(fig, "How much does the spatial model change control?", "Metrica · two full games · 240 sampled eligible frames", "Differences measure sensitivity to modelling assumptions, with no external accuracy target.")
        spatial_differences(ax, c["method_differences"])
        paths += save(fig, detail, "space-control")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.1))
    decorate(fig, "Do buildup features improve next-season prediction?", "Target: next-season xGBuildup /90 · Understat · 1,823 held-out pairs · target seasons 2024/25 and 2025/26", "Intervals: 95% paired player-cluster bootstrap, conditional on this selected sample and fitted models.\nxGBuildup is attributed involvement. Partial 2026/27 profiles are excluded from prediction targets.", bottom=.23, top=.74)
    keys = ["performance", "carry_forward", "augmented"]
    labels = ["Performance only", "Carry-forward buildup", "+ buildup and chain"]
    means = [prediction["models"][key]["test_mae"] for key in keys]
    axes[0].scatter(means, range(3), color=[COLORS["neutral"], COLORS["neutral"], COLORS["blue"]], s=50)
    for y, mean in enumerate(means): axes[0].text(mean+.0006, y, f"{mean:.5f}", va="center", fontsize=10)
    spread = max(max(means) - min(means), .001)
    axes[0].set(yticks=range(3), yticklabels=labels, xlabel="Next-season xGBuildup/90\nMean absolute error", xlim=(max(0, min(means)-.35*spread), max(means)+.50*spread), ylim=(2.7, -.7))
    axes[0].grid(axis="x"); axes[0].set_axisbelow(True); axes[0].set_title("Absolute error")
    buildup_difference(axes[1], prediction); axes[1].set_title("Added-feature difference and uncertainty")
    paths += save(fig, detail, "buildup")

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10))
    decorate(fig, "Football analysis · four independent questions", "Historical samples · each panel retains its own units and denominator", "Details and sources: results/analysis-upgrades.json. Open individual figures for complete context and spatial surfaces.", top=.78, bottom=.16)
    fig.subplots_adjust(left=.19, right=.96, wspace=.8, hspace=1.15)
    pressing(axes[0, 0], a["pressing_chains"])
    axes[0, 0].set_title("Pressing outcomes\n20 matches · 1,567 chains", loc="left")
    value_difference(axes[0, 1], b)
    axes[0, 1].set_title("GRU vs baselines\n12,000 states · 95% paired intervals", loc="left")
    spatial_differences(axes[1, 0], c["method_differences"])
    axes[1, 0].set_title("Spatial model sensitivity\nTwo games · 240 eligible frames", loc="left")
    buildup_difference(axes[1, 1], prediction)
    axes[1, 1].set_title("Next-season buildup\n1,823 pairs · 95% paired intervals", loc="left")
    paths += save(fig, output, "analysis-upgrades", formats=("png", "svg", "pdf"))
    return {"figures": [str(path) for path in paths], "spatial_example": spatial_receipt,
            "bundle_id": report["bundle_id"], "quantitative_inputs_modified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="results/analysis-upgrades.json")
    parser.add_argument("--output", default="results")
    parser.add_argument("--surfaces")
    parser.add_argument("--frames")
    args = parser.parse_args()
    print(json.dumps(render(args.report, args.output, surfaces=args.surfaces, frames=args.frames)))
