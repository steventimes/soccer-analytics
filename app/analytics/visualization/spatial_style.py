"""Style stability comparisons and early-period pass basis maps."""

import numpy as np
from .style import COLORS, apply_style, decorate, save


def figures(report, pass_basis, output, seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    apply_style()
    methods = ["frequency", "heatmaps", "nmf_start_only", "nmf_start_end"]
    labels = ["Action frequencies (4)", "Spatial maps (12,500)", "NMF: starts only (18)", "NMF: pass starts + ends (18)"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    for ax, pool, title in zip(axes, ["same_league", "same_position"], ["Same-league candidates", "Same league + position"]):
        rows = [next(r for r in report["summary"] if r["scope"] == "all" and r["pool"] == pool and r["method"] == method) for method in methods]
        for i, row in enumerate(rows):
            mean, low, high = [row["top1"][k] for k in ("mean", "min", "max")]
            color = COLORS["orange"] if row["method"] == "nmf_start_end" else COLORS["blue"] if row["method"] == "heatmaps" else COLORS["muted"]
            ax.plot([low, high], [i, i], color=color, linewidth=2.5)
            ax.scatter([mean], [i], color=color, s=48, zorder=3)
            ax.text(mean + .025, i, f"{mean:.1%}", va="center", fontsize=10, color=color)
        ax.axvline(rows[0]["random_top1"], color=COLORS["neutral"], linestyle=":", linewidth=1.5)
        ax.set_title(title); ax.set_xlim(0, 1); ax.set_yticks(range(4), labels); ax.set_ylim(3.6, -.6)
        ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.set_xlabel("Top-1 identity retrieval (tie-aware)")
        ax.grid(axis="x")
    decorate(fig, "Can later actions identify the same player?", f"Wyscout 2017/18 | {report['cohort']['players']} outfield players | early basis frozen before 2018-01-01",
             f"Dot = mean; horizontal range = min-max over {len(report['parameters']['seeds'])} NMF seeds, not a confidence interval. Dotted line = random rank-1 chance.\nSame club and at least 900 minutes in both periods. Ground attacking duels are observed take-on proxies, not inferred carries.\nThis closed-set task measures on-ball style stability, not player ability or transfer success.", bottom=.3, top=.75)
    save(fig, output, "Figure_1", formats=("png", "svg", "pdf"))
    bins = report["parameters"]["grid"]
    maps = pass_basis[:2].reshape(2, 2, bins, bins)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    maximum = float(maps.max())
    for component in range(2):
        for channel in range(2):
            ax = axes[component, channel]
            image = ax.imshow(maps[component, channel], origin="upper", extent=(0, 100, 100, 0), cmap="Blues", vmin=0, vmax=maximum, aspect=68 / 105, interpolation="nearest")
            ax.axvline(50, color=COLORS["pitch_line"], linewidth=.7)
            ax.set_title(f"Basis {component + 1}: pass {'starts' if channel == 0 else 'ends'}")
            ax.set_xticks([0, 50, 100]); ax.set_yticks([0, 50, 100]); ax.set_xlabel("Pitch x (%) → attack"); ax.set_ylabel("Pitch y (%)")
    decorate(fig, "Spatial basis learned from early actions only", f"Wyscout 2017/18 | {report['windows']['early']['start']} to {report['windows']['early']['end']} | seed {seed}\nFirst two of five pass components | joint start/end basis has unit L1 mass",
             f"Native Wyscout coordinates; {bins} × {bins} cells, Gaussian input sigma={report['parameters']['sigma_cells']:g} cells; nearest-cell display.\nFixed reference configuration: 4/4/5/5 components; local 70% variance selection was not performed. One shared colour scale.\nAggregate fitted patterns, not occupancy time. Components are not ordered by ability or quality.", bottom=.25, top=.78)
    fig.subplots_adjust(right=.8)
    colorbar = fig.colorbar(image, cax=fig.add_axes([.86, .25, .018, .48]))
    colorbar.set_label("Basis mass per grid cell")
    save(fig, output, "Figure_2", formats=("png", "svg", "pdf"))
