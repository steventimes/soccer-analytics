"""Export compact aggregate research results and plots for the public repository.

Raw provider records, identity links, predictions and fitted model binaries stay
local. The export contains aggregate evaluation metrics and chart-ready summaries.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

from .artifacts import file_digest
from .visualization.style import COLORS, apply_style, decorate, save as save_figure


def read(path):
    return json.loads(Path(path).read_text())


def generate(
    assessment=".tmp/assessments",
    tactics=".tmp/tactical-relations",
    movement=".tmp/movement-lab",
    output="results",
    database=None,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    assessment, tactics, movement, output = map(
        Path, (assessment, tactics, movement, output)
    )
    output.mkdir(parents=True, exist_ok=True)
    player = read(assessment / "report.json")
    tactical = read(tactics / "report.json")
    physical = read(movement / "report.json")
    report = {
        "snapshot_date": "2026-10-09",
        "kind": "Retrospective aggregate research results",
        "inputs_sha256": {
            name: file_digest(path / "report.json")
            for name, path in [
                ("assessment", assessment),
                ("tactics", tactics),
                ("movement", movement),
            ]
        },
        "counts": {
            k: player[k]
            for k in (
                "player_observations",
                "player_profiles",
                "goalkeeper_profiles",
                "team_matches",
                "team_windows",
                "team_ratings",
            )
        },
        "regressions": {},
        "chance_models": tactical["chance_models"],
        "tactical_ablations": tactical["feature_ablations"],
        "poisson": {
            "experiments": player["team_poisson"]["experiments"],
            "failures": player["team_poisson"]["failures"],
        },
        "movement": {
            k: physical[k]
            for k in (
                "matches",
                "player_match_rows",
                "rotation_candidates",
                "classified_transition_windows_ge3s",
                "phase_load_groups",
                "rotation_followup",
                "protocol",
                "professional_sources",
            )
        },
        "limits": player["protocol"]["limits"]
        + [
            "Chance prefixes span a selective public archive, not representative complete leagues.",
            "Tactical associations, rotation follow-ups and observed workloads have no causal identification.",
            "Raw games, private FM conversions and individual model binaries are excluded from this public export.",
        ],
    }
    match_experiments = Path(".tmp/pl-experiments.json")
    history_comparison = Path(".tmp/history-comparison.json")
    if match_experiments.exists():
        report["match_search"] = read(match_experiments)
        report["inputs_sha256"]["match_search"] = file_digest(match_experiments)
    if history_comparison.exists():
        history = read(history_comparison)
        report["multi_season_comparison"] = {
            k: history[k]
            for k in (
                "protocol",
                "summary",
                "paired_comparisons",
                "runtime_versions",
                "errors",
                "collection_errors",
                "manifest_sha256",
            )
        }
        report["inputs_sha256"]["multi_season_comparison"] = file_digest(
            history_comparison
        )
    for name, benchmark in player["benchmarks"].items():
        report["regressions"][name] = {
            "methods": benchmark["methods"],
            "failures": benchmark["failures"],
            "results": benchmark["results"],
            "selected": {
                target: {k: v for k, v in value.items() if k != "model_path"}
                for target, value in benchmark["selected_models"].items()
            },
        }
    from matplotlib.ticker import PercentFormatter

    apply_style()
    charts = []

    def save(fig, name):
        save_figure(fig, output, name)
        charts.append(name)

    intelligence = Path(".tmp/scouting-intelligence")
    if (intelligence / "report.json").exists():
        insights = read(intelligence / "report.json")
        report["scouting_intelligence"] = insights
        report["inputs_sha256"]["scouting_intelligence"] = file_digest(
            intelligence / "report.json"
        )
        watch = read(intelligence / "youth-watchlist.json")
        cards = read(intelligence / "player-cards.json")
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.6))
        ages = list(range(16, 22))
        bottom = np.zeros(len(ages))
        for tier, color, label in [
            ("limited_observation", COLORS["neutral"], "Limited"),
            ("routine_followup", COLORS["orange"], "Routine follow-up"),
            ("priority_for_review", COLORS["blue"], "Review signals"),
        ]:
            values = np.array(
                [
                    sum(
                        row["age_years"] == age and row["observation_tier"] == tier
                        for row in watch
                    )
                    for age in ages
                ]
            )
            axes[0].bar(ages, values, bottom=bottom, color=color, label=label, width=.72)
            bottom += values
        for age, count in zip(ages, bottom):
            axes[0].text(age, count + max(bottom) * .035, f"{count:.0f}", ha="center")
        axes[0].set(xticks=ages, xlabel="Calendar age on 31 May 2018", ylabel="Players", ylim=(0, max(bottom) * 1.2))
        axes[0].set_title(f"Follow-up tiers · {len(watch):,} players aged 16–21", loc="left", pad=42)
        axes[0].legend(frameon=False, ncol=3, fontsize=10, loc="lower left", bbox_to_anchor=(-.04, 1), columnspacing=1)
        axes[0].grid(axis="y")
        axes[0].set_axisbelow(True)
        roles = sorted({row["position"] for row in cards})
        totals = {
            role: {direction: 0 for direction in ["forward", "lateral", "backward"]}
            for role in roles
        }
        for card in cards:
            for direction, count in (
                (card.get("habits") or {}).get("pass_direction_counts", {}).items()
            ):
                totals[card["position"]][direction] += count
        left = np.zeros(len(roles))
        for direction, color in [
            ("forward", COLORS["blue"]),
            ("lateral", COLORS["orange"]),
            ("backward", COLORS["neutral"]),
        ]:
            values = np.array(
                [
                    totals[role][direction] / max(1, sum(totals[role].values()))
                    for role in roles
                ]
            )
            axes[1].barh(range(len(roles)), values, left=left, color=color, label=direction.title(), height=.58)
            for i, value in enumerate(values):
                if value >= .025:
                    axes[1].text(left[i] + value / 2, i, f"{value:.0%}", ha="center", va="center", color="white" if direction != "backward" else COLORS["ink"])
                elif value > 0:
                    axes[1].annotate(f"{direction.title()} {value:.1%}", (left[i] + value / 2, i), xytext=(.99, i + .43), textcoords="data", ha="right", va="center", fontsize=10, color=COLORS["muted"])
            left += values
        axes[1].set(yticks=range(len(roles)), yticklabels=[f"{role}\nn={sum(totals[role].values()):,} passes" for role in roles], xlim=(0, 1), xlabel="Share of located attempted passes")
        axes[1].xaxis.set_major_formatter(PercentFormatter(1))
        axes[1].invert_yaxis()
        axes[1].set_title("Pass direction by recorded position", loc="left", pad=42)
        axes[1].legend(frameon=False, ncol=3, fontsize=10, loc="lower left", bbox_to_anchor=(0, 1), columnspacing=1.2)
        decorate(
            fig, "Youth observations and passing habits",
            "Wyscout 2017/18 · Historical observations at the 31 May 2018 cutoff",
            "Tiers describe current evidence, not future potential. Eligible ages: 16–21; no 16-year-olds were covered.\nPass shares pool located attempts within each position; they do not measure standing positions.",
            bottom=.22, top=.71,
        )
        fig.subplots_adjust(left=.08, right=.97, wspace=.65)
        save(fig, "scouting-observations")

    labels = {
        "goals_per90": "Non-penalty goals / 90",
        "progressive_passes_per90": "Progressive passes / 90",
        "pass_completion": "Pass completion",
        "duel_win_rate": "Recorded duel wins",
        "keeper_success_rate": "Recorded keeper saves",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.4), sharex=True, sharey=True)
    bounds = []
    for ax, mode, title in zip(
        axes,
        ("player_temporal", "player_unseen_players"),
        ("Later chronological windows", "Held-out player identities"),
    ):
        winners = player["benchmarks"][mode]["selected_models"]
        for index, target in enumerate(labels):
            if target not in winners:
                ax.text(.03, index, "Not evaluated", transform=ax.get_yaxis_transform(), va="center", color=COLORS["muted"])
                continue
            row = winners[target]
            base = row["baseline"]["mae"]
            gain = 100 * (base - row["test"]["mae"]) / base
            ci = [-100 * v / base for v in reversed(row["cluster_bootstrap_ci95"])]
            bounds.extend(ci)
            ax.plot(ci, [index, index], color=COLORS["ink"], lw=1.7)
            ax.scatter(
                gain, index, color=COLORS["blue"] if gain >= 0 else COLORS["orange"], s=48, zorder=3
            )
            ax.text(.99, index, f"{gain:+.1f}%\nn={row['test_rows']:,} windows", transform=ax.get_yaxis_transform(), va="center", ha="right", fontsize=10, color=COLORS["ink"])
        ax.axvline(0, color=COLORS["muted"], lw=1)
        ax.set_title(title, loc="left")
        ax.set_xlabel("MAE reduction vs selected baseline (%)")
        ax.grid(axis="x")
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(labels)), list(labels.values()))
    axes[0].set_ylim(len(labels) - .4, -.6)
    axes[0].set_xlim(min(-5, min(bounds) - 2), max(bounds) + 15)
    decorate(
        fig, "Player assessment against historical baselines",
        "Wyscout 2017/18 · Positive values mean lower prediction error; zero means the same MAE as the baseline",
        "Model and baseline selected on validation only. Reduction = 100 × (baseline MAE − model MAE) / baseline MAE.\nLines: 95% entity-bootstrap intervals conditional on fitted models, unadjusted for multiple comparisons.",
        bottom=.23, top=.75,
    )
    fig.subplots_adjust(left=.19, right=.97, wspace=.12)
    save(fig, "player-assessment")

    poisson = player["team_poisson"]["experiments"]
    values = [
        100
        * (r["home_away_mean_deviance"] - r["test_poisson_deviance"])
        / r["home_away_mean_deviance"]
        for r in poisson
    ]
    fig, ax = plt.subplots(figsize=(11.5, 10))
    order = np.argsort(values)[::-1]
    ax.barh(
        range(len(order)),
        [values[i] for i in order],
        color=[COLORS["blue"] if values[i] >= 0 else COLORS["orange"] for i in order], height=.62,
    )
    ax.set_yticks(
        range(len(order)),
        [f"{poisson[i]['league']}   ·   n={poisson[i]['test_matches']:,}" for i in order],
    )
    for y, i in enumerate(order):
        value = values[i]
        ax.text(value + (.22 if value >= 0 else -.22), y, f"{value:+.2f}%", va="center", ha="left" if value >= 0 else "right")
    ax.invert_yaxis()
    ax.axvline(0, color=COLORS["muted"], lw=1)
    ax.set_xlabel("Poisson deviance reduction vs home/away scoring means (%)")
    ax.set_xlim(min(values) - 2, max(values) + 2.8)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    decorate(
        fig, "Team scoring models by league",
        f"{len(poisson)} separate league samples · Held-out 2025 season · n = test matches in each league",
        "Positive values mean lower deviance than the fitted home/away scoring-mean baseline.\nLeague models and scales are fitted separately. These point estimates have no reported uncertainty intervals.",
        bottom=.15, top=.83,
    )
    fig.subplots_adjust(left=.22, right=.95)
    save(fig, "team-assessment")

    ablations = tactical["feature_ablations"]["results"]
    display = {
        "context_baseline": "Context baseline",
        "formation_only": "Context + formation",
        "logistic": "Full logistic",
        "without_formation": "Without formation",
        "without_channels": "Without channels / width",
        "without_opponent": "Without opponent context",
    }
    full_brier = next(row["test"]["brier"] for row in ablations if row["model"] == "logistic")
    ablations = sorted(ablations, key=lambda row: row["test"]["brier"])
    fig, ax = plt.subplots(figsize=(12, 6.3))
    deltas = [row["test"]["brier"] - full_brier for row in ablations]
    ax.axvline(0, color=COLORS["muted"], lw=1.2)
    for i, (row, delta) in enumerate(zip(ablations, deltas)):
        color = COLORS["orange"] if delta > 0 else COLORS["blue"]
        ax.plot([0, delta], [i, i], color=COLORS["grid"], lw=2)
        ax.scatter(delta, i, s=52, color=color, zorder=3, marker="D" if row["model"] == "logistic" else "o")
        ax.text(1.04, i, f"{row['test']['brier']:.6f}", transform=ax.get_yaxis_transform(), va="center")
        ax.text(1.30, i, f"{delta:+.6f}", transform=ax.get_yaxis_transform(), va="center", color=color)
    ax.text(1.04, 1.04, "Brier score", transform=ax.transAxes, color=COLORS["muted"])
    ax.text(1.30, 1.04, "Difference", transform=ax.transAxes, color=COLORS["muted"])
    ax.set_yticks(range(len(ablations)), [display[row["model"]] for row in ablations])
    ax.invert_yaxis()
    span = max(deltas) - min(deltas)
    ax.set_xlim(min(deltas) - span * .08, max(deltas) + span * .08)
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.set_xlabel("Brier score difference from full logistic · lower is better")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    split = tactical["feature_ablations"]["split"]
    decorate(
        fig, "Chance-model feature ablations",
        f"After the third action · {split['test_matches']:,} held-out matches · Full logistic Brier = {full_brier:.6f}",
        "Zero marks full logistic. All rows share the logistic family, chronological split and fitting budget.\nDifferences are descriptive point estimates with no reported intervals; feature associations have no causal interpretation.",
        bottom=.23, top=.75,
    )
    fig.subplots_adjust(left=.25, right=.72)
    save(fig, "chance-ablations")

    groups = defaultdict(lambda: np.zeros(2))
    for row in read(movement / "phase-load.json"):
        groups[(row["position_group"], row["state"])] += [
            row["high_speed_distance_m"],
            row["observed_seconds"],
        ]
    roles = sorted({k[0] for k in groups})
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    x = np.arange(len(roles))
    for offset, state, color in [
        (-0.18, "in_possession", COLORS["blue"]),
        (0.18, "out_of_possession", COLORS["orange"]),
    ]:
        rates = [
            (
                groups[(role, state)][0] * 60 / groups[(role, state)][1]
                if groups[(role, state)][1]
                else 0
            )
            for role in roles
        ]
        axes[0].barh(
            x + offset, rates, height=0.32, label=state.replace("_", " ").title(), color=color
        )
        for y, role, rate in zip(x + offset, roles, rates):
            minutes = groups[(role, state)][1] / 60
            axes[0].text(rate + .12, y, f"{rate:.1f}  ·  {minutes:,.0f} min", va="center", fontsize=10)
    axes[0].set_yticks(x, roles)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Distance >20 km/h (m per observed player minute)")
    axes[0].set_title("Observed high-speed workload", loc="left", pad=38)
    axes[0].legend(
        frameon=False, fontsize=10, ncol=2, loc="lower left", bbox_to_anchor=(-.04, 1), columnspacing=1,
    )
    axes[0].set_xlim(0, max(groups[k][0] * 60 / groups[k][1] for k in groups if groups[k][1]) * 1.65)
    axes[0].grid(axis="x")
    axes[0].set_axisbelow(True)
    rows = read(movement / "players.json")
    coverage = [
        r["detected_segment_coverage_fraction"]
        for r in rows
        if r["detected_segment_coverage_fraction"] is not None
    ]
    axes[1].hist(
        coverage, bins=np.linspace(0, 1, 21), color=COLORS["blue"], edgecolor="white", linewidth=.6
    )
    axes[1].axvline(0.8, color=COLORS["orange"], lw=1.5, linestyle="--")
    axes[1].text(.79, .95, "80% reference\n(window gate)", ha="right", va="top", transform=axes[1].get_xaxis_transform(), color=COLORS["orange"], fontsize=10)
    axes[1].xaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set(xlim=(0, 1), xlabel="Observed segment time / nominal playing time")
    axes[1].set_ylabel("Player-match records")
    axes[1].set_title(f"Broadcast visibility · {len(coverage):,} records", loc="left", pad=38)
    axes[1].grid(axis="y")
    axes[1].set_axisbelow(True)
    decorate(
        fig, "External match load and observation coverage",
        f"{physical['matches']} selected A-League 2024/25 fixtures · Detected positions only; incomplete broadcast exposure",
        f"Workload labels give pooled observed player-minutes. Rates do not fill missing time or measure fitness.\nMedian visibility: {np.median(coverage):.1%}; {sum(v >= .8 for v in coverage):,}/{len(coverage):,} records reach 80%. The 80% reference gates rolling windows, not entire records.",
        bottom=.23, top=.71,
    )
    fig.subplots_adjust(left=.13, right=.97, wspace=.48)
    save(fig, "external-load")
    report["movement"]["player_match_coverage"] = {
        "median": float(np.median(coverage)),
        "records_at_least80pct": sum(v >= 0.8 for v in coverage),
        "records": len(coverage),
    }
    report["charts"] = charts
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    if database:
        from app.data_service.warehouse import Warehouse
        from .artifacts import archive_outputs

        with Warehouse(database) as db:
            archive_outputs(
                db,
                "public_research_results",
                output,
                "public_research_artifact",
                file_digest(output / "summary.json")[:16],
            )
            db.connection.commit()
            db.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    print(
        json.dumps(
            {
                "output": str(output),
                "charts": charts,
                "summary": str(output / "summary.json"),
            },
            indent=2,
        )
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--assessment", default=".tmp/assessments")
    parser.add_argument("--tactics", default=".tmp/tactical-relations")
    parser.add_argument("--movement", default=".tmp/movement-lab")
    parser.add_argument("--output", default="results")
    parser.add_argument(
        "--database",
        help="Optionally archive the aggregate results and figures in the existing warehouse.",
    )
    args = parser.parse_args()
    generate(args.assessment, args.tactics, args.movement, args.output, args.database)
