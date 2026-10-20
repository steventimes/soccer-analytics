"""Primary competition coverage and source-backed opponent-specific case studies.

Report comparisons describe observed choices; they do not identify causal effects
or convert a full-match statistic into a time-resolved pressure measurement.
"""

import argparse
from collections import defaultdict
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from statistics import mean

from app.data_service.warehouse import Warehouse, quote
from app.data_service.recent_matches import _validate_summary
from .artifacts import archive_outputs, file_digest

COMPETITIONS = {
    "premier_league": [
        "Premier League",
        "PL",
        "eng.1",
        "E0",
        "ENG-Premier League",
        "matches_England",
    ],
    "la_liga": ["La Liga", "PD", "esp.1", "SP1", "ESP-La Liga", "matches_Spain"],
    "bundesliga": [
        "1. Bundesliga",
        "Bundesliga",
        "BL1",
        "ger.1",
        "D1",
        "GER-Bundesliga",
        "matches_Germany",
    ],
    "serie_a": ["Serie A", "SA", "ita.1", "I1", "ITA-Serie A", "matches_Italy"],
    "ligue_1": ["Ligue 1", "FL1", "fra.1", "F1", "FRA-Ligue 1", "matches_France"],
    "champions_league": ["Champions League", "uefa.champions"],
    "european_championship": [
        "UEFA Euro",
        "uefa.euro",
        "matches_European_Championship",
    ],
    "world_cup": ["FIFA World Cup", "fifa.world", "matches_World_Cup", "fwc2026"],
    "club_world_cup": ["FIFA Club World Cup", "fifa.club", "fcwc2025"],
}
ALIASES = {alias: key for key, aliases in COMPETITIONS.items() for alias in aliases}
COACH_SOURCE = "https://www.chelseafc.com/en/news/article/maresca-explains-how-club-world-cup-final-was-won"
MATCH_SOURCE = "https://www.chelseafc.com/en/news/article/club-world-cup-final-report-chelsea-3-0-paris-saint-germain"


def core_competition(name, gender="male"):
    return ALIASES.get(name) if gender == "male" else None


def install_scope(c):
    c.execute(
        "CREATE TABLE IF NOT EXISTS research_competition_scope(competition TEXT PRIMARY KEY,priority TEXT,source_aliases TEXT)"
    )
    c.executemany(
        "INSERT OR REPLACE INTO research_competition_scope VALUES(?,?,?)",
        [
            (name, "primary", json.dumps(aliases))
            for name, aliases in COMPETITIONS.items()
        ],
    )
    c.execute(
        "INSERT OR REPLACE INTO research_competition_scope VALUES(?,?,?)",
        (
            "a_league_tracking",
            "method_validation",
            json.dumps(["SkillCorner Open Data 2024/25"]),
        ),
    )
    c.execute(
        "INSERT OR REPLACE INTO research_competition_scope VALUES(?,?,?)",
        ("metrica_anonymous", "method_validation", "[]"),
    )
    names = [s for aliases in COMPETITIONS.values() for s in aliases]
    literals = ",".join("'" + s.replace("'", "''") + "'" for s in names)
    c.executescript(
        f"""CREATE VIEW IF NOT EXISTS core_statsbomb_matches AS SELECT * FROM statsbomb_matches WHERE json_extract(payload,'$.competition.competition_name') IN ({literals}) AND json_extract(payload,'$.home_team.home_team_gender')='male';
    CREATE VIEW IF NOT EXISTS core_wyscout_matches AS SELECT * FROM wyscout_entities WHERE entity_type IN ({literals});
    CREATE VIEW IF NOT EXISTS core_recent_matches AS SELECT * FROM latest_match_observations WHERE competition IN ({literals}) AND gender='male';
    CREATE VIEW IF NOT EXISTS core_fifa_reports AS SELECT * FROM fifa_report_matches WHERE series IN ('fcwc2025','fwc2026');"""
    )


def coverage(c):
    groups = defaultdict(list)
    event_ids = {
        Path(r[0]).stem
        for r in c.execute(
            "SELECT a.path FROM documents d JOIN assets a ON a.id=d.asset_id WHERE d.kind='statsbomb_events'"
        )
    }
    for row in c.execute(
        "SELECT match_id,match_date,payload FROM core_statsbomb_matches"
    ):
        m = json.loads(row[2])
        comp = core_competition(m["competition"]["competition_name"])
        groups[("statsbomb", comp, m["season"]["season_name"], "event")].append(
            (str(row[0]), row[1], str(row[0]) in event_ids)
        )
    for mid, kind, payload in c.execute(
        "SELECT entity_id,entity_type,payload FROM core_wyscout_matches"
    ):
        m = json.loads(payload)
        comp = core_competition(kind)
        season = (
            "2016"
            if comp == "european_championship"
            else "2018" if comp == "world_cup" else "2017/2018"
        )
        groups[("wyscout", comp, season, "event")].append(
            (str(mid), m["dateutc"][:10], None)
        )
    for mid, date, path in c.execute(
        "SELECT m.match_id,json_extract(m.payload,'$.source_date'),a.path FROM football_matches m JOIN assets a ON a.id=m.asset_id JOIN datasets d ON d.id=a.dataset_id WHERE d.name='football-csv'"
    ):
        identity = re.match(r"history/football-csv/([^/]+)/(\d{4})\.json", path)
        if not identity:
            raise ValueError("Football history league/season identity missing")
        comp = core_competition(identity[1])
        if comp:
            year = int(identity[2])
            groups[
                (
                    "football_csv",
                    comp,
                    f"{year}/{year+1}",
                    "result_and_basic_statistics",
                )
            ].append((str(mid), date, None))
    for table, asset in c.execute(
        "SELECT DISTINCT table_name,asset_id FROM soccerdata_imports WHERE provider='Understat' AND method='schedule'"
    ):
        for league, season, mid, date in c.execute(
            f"SELECT index_00_league,index_01_season,field_0002_game_id,field_0003_date FROM {quote(table)} WHERE asset_id=?",
            (asset,),
        ):
            comp = core_competition(league)
            if comp:
                groups[("understat", comp, season, "schedule_with_xg")].append(
                    (str(mid), date[:10], None)
                )
    for mid, comp, date, payload, complete, kind, summary in c.execute(
        "SELECT m.match_id,m.competition,m.match_date,m.payload,a.complete,d.kind,d.payload "
        "FROM core_recent_matches m LEFT JOIN recent_match_summaries s USING(provider,competition,match_id) "
        "LEFT JOIN assets a ON a.id=s.asset_id LEFT JOIN documents d ON d.asset_id=s.asset_id "
        "WHERE m.status='FINISHED' AND m.provider='espn'"
    ):
        summary_status = "missing"
        if complete == 1 and kind == "recent_provider_json" and summary is not None:
            try:
                _validate_summary(json.loads(summary), json.loads(payload))
            except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
                summary_status = "invalid"
            else:
                summary_status = "matched"
        year = int(date[:4])
        if comp in {"uefa.euro", "fifa.world", "fifa.club"}:
            season = str(year)
        else:
            year -= int(date[5:7] < "07")
            season = f"{year}/{year + 1}"
        groups[
            (
                "espn_recent",
                core_competition(comp),
                season,
                "result",
            )
        ].append((mid, date, summary_status))
    if c.execute(
        "SELECT 1 FROM sqlite_master WHERE name='fifa_report_matches'"
    ).fetchone():
        for series, mid, payload in c.execute(
            "SELECT series,match_number,payload FROM fifa_report_matches"
        ):
            m = json.loads(payload)
            groups[
                (
                    "fifa_training_centre",
                    core_competition(series),
                    str(m["date"][:4]),
                    "tracking_derived_report",
                )
            ].append((str(mid), m["date"], None))
    rows = []
    for (provider, comp, season, grain), matches in sorted(groups.items()):
        dates = [m[1] for m in matches if m[1]]
        count = len({m[0] for m in matches})
        row = {
            "provider": provider,
            "competition": comp,
            "season": season,
            "grain": grain,
            "source_match_records": count,
            "date_start": min(dates) if dates else None,
            "date_end": max(dates) if dates else None,
        }
        if provider == "espn_recent":
            summaries = {m[0]: m[2] for m in matches}
            row.update({
                f"{status}_summary_documents": sum(value == status for value in summaries.values())
                for status in ("matched", "missing", "invalid")
            })
        if provider == "statsbomb":
            row["matched_event_documents"] = sum(m[2] for m in matches)
            expected = (
                51
                if comp == "european_championship"
                else (
                    64
                    if comp == "world_cup" and season in ["2018", "2022"]
                    else (
                        {
                            "premier_league": 380,
                            "la_liga": 380,
                            "serie_a": 380,
                            "ligue_1": 380,
                            "bundesliga": 306,
                        }.get(comp)
                        if season == "2015/2016"
                        else None
                    )
                )
            )
            row["expected_full_competition_matches"] = expected
            row["coverage_status"] = (
                "complete"
                if expected == count and row["matched_event_documents"] == count
                else "partial_or_selected_archive"
            )
        elif provider == "fifa_training_centre":
            expected = 63 if comp == "club_world_cup" else 104
            row["expected_full_competition_matches"] = expected
            row["coverage_status"] = "complete" if count == expected else "partial"
        else:
            row["coverage_status"] = (
                "source_records; completeness not inferred from row count"
            )
        rows.append(row)
    return {
        "primary_competitions": list(COMPETITIONS),
        "coverage": rows,
        "scope": "Counts are within source/competition/season and must not be summed across overlapping sources. Schedule rows may be unplayed. Historic selected UCL matches are not complete UCL seasons.",
        "summary_scope": "ESPN summary counts require a complete archived JSON document with matching event, home/away team identities, final status and scores. Missing includes absent links, documents, incomplete assets and unsupported document kinds. Invalid means malformed or inconsistent with the latest result. A matched header does not establish availability of every statistic, lineup or event field.",
        "validation_only": [
            "20 A-League 2024/25 continuous tracking matches",
            "3 anonymous Metrica samples",
        ],
        "current_gaps": [
            "Modern complete UCL event/continuous tracking coverage",
            "Raw named trajectories for this 2025 final",
            "Time-resolved pressure/weak-side runs for 2026 World Cup from these PDFs alone",
        ],
    }


def _report_version(c, catalog):
    return {
        "extractor_sha256": file_digest(__file__),
        "summary_validator_sha256": file_digest(
            Path(__file__).parents[1] / "data_service" / "recent_matches.py"
        ),
        "fifa_parser_sha256": file_digest(
            Path(__file__).parents[1] / "data_service" / "fifa_reports.py"
        ),
        "coverage_sha256": hashlib.sha256(
            json.dumps(catalog, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
        "report_assets": [
            list(r) for r in c.execute(
                "SELECT series,match_number,a.sha256 FROM fifa_report_matches m "
                "JOIN assets a ON a.id=m.asset_id ORDER BY series,match_number"
            )
        ] if c.execute("SELECT 1 FROM sqlite_master WHERE name='fifa_report_matches'").fetchone() else [],
    }


def export_coverage(database, output):
    """Recheck coverage in one read-only snapshot, without rebuilding case studies."""
    uri = Path(database).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as c:
        c.execute("BEGIN")
        catalog = coverage(c)
        version = _report_version(c, catalog)
    run_id = hashlib.sha256(json.dumps(version, sort_keys=True).encode()).hexdigest()[:16]
    catalog.update({"run_id": run_id, "observed_at": datetime.now(timezone.utc).isoformat(), "inputs": version})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "core-coverage.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2))
    return {"run_id": run_id, "coverage_rows": len(catalog["coverage"])}


def team_comparison(matches, final, side):
    team = final["teams"][side]
    prior = sorted(
        [m for m in matches if m["date"] < final["date"] and team in m["teams"]],
        key=lambda m: (m["date"], m["match_number"]),
    )
    items = []
    for m in prior:
        slot = m["teams"].index(team)
        items.append(
            {
                "match_number": m["match_number"],
                "date": m["date"],
                "opponent": m["teams"][1 - slot],
                "goals_for": m["goals"][slot],
                "goals_against": m["goals"][1 - slot],
                "metrics": m["metrics_by_team"][slot],
            }
        )
    comparisons = {}
    for name, value in final["metrics_by_team"][side].items():
        values = [m["metrics"][name] for m in items if name in m["metrics"]]
        comparisons[name] = {
            "final": value,
            "pre_final_available_matches": len(values),
            "pre_final_mean": mean(values) if values else None,
            "pre_final_values": values,
            "final_minus_pre_final": value - mean(values) if values else None,
        }
    return {
        "team": team,
        "pre_final_form": {
            "matches": len(items),
            "wins": sum(m["goals_for"] > m["goals_against"] for m in items),
            "draws": sum(m["goals_for"] == m["goals_against"] for m in items),
            "goals_for": sum(m["goals_for"] for m in items),
            "goals_against": sum(m["goals_against"] for m in items),
            "definition": "reported match scores; includes extra time where played, not per90 or opponent-adjusted paper strength",
        },
        "pre_final_matches": items,
        "metrics": comparisons,
    }


def chelsea_psg_case(c):
    matches = []
    for number, asset, payload in c.execute(
        "SELECT match_number,asset_id,payload FROM fifa_report_matches WHERE series='fcwc2025' ORDER BY match_number"
    ):
        m = json.loads(payload)
        m["asset_id"] = asset
        m["metrics_by_team"] = [{}, {}]
        for slot, name, value, unit, page in c.execute(
            "SELECT team_slot,metric,value,unit,page FROM fifa_report_metrics WHERE asset_id=?",
            (asset,),
        ):
            m["metrics_by_team"][slot][name] = value
        matches.append(m)
    final = next(m for m in matches if m["match_number"] == 63)
    if (
        final["teams"] != ["Chelsea FC", "Paris Saint-Germain"]
        or final["date"] != "2025-07-13"
        or final["goals"] != [3, 0]
    ):
        raise ValueError("CWC final identity mismatch")
    teams = [team_comparison(matches, final, s) for s in [0, 1]]
    if any(t["pre_final_form"]["matches"] != 6 for t in teams):
        raise ValueError("Case requires all six pre-final games for each team")
    return {
        "case_id": "fcwc2025-final-chelsea-psg",
        "match": final,
        "team_comparisons": teams,
        "coaches": {
            "Chelsea FC": {"name": "Enzo Maresca", "source_url": COACH_SOURCE},
            "Paris Saint-Germain": {"name": "Luis Enrique", "source_url": MATCH_SOURCE},
        },
        "mechanisms": [
            {
                "theme": "opening_man_marking",
                "evidence_level": "coach_self_report",
                "source_url": COACH_SOURCE,
                "statement_zh": "Maresca 说明开场十分钟采用强度较高的人盯人压迫；天气使长期维持这一强度困难。",
                "required_test": "Opening ten-minute pressure sequences plus PSG recipient positions; whole-match PDF pressure totals do not measure this interval.",
            },
            {
                "theme": "midfield_matchup",
                "evidence_level": "coach_self_report",
                "source_url": COACH_SOURCE,
                "statement_zh": "教练说明中场逐人对位，并由 Enzo 负责 Vitinha。",
                "required_test": "Match-day marking assignments and role changes rather than nominal formation labels.",
            },
            {
                "theme": "right_side_overload_and_runs",
                "evidence_level": "coach_self_report_and_club_match_narrative",
                "source_url": COACH_SOURCE,
                "sequence_source_url": MATCH_SOURCE,
                "statement_zh": "Palmer 与 Gusto 的同侧配合针对边路空间；官方战报描述第二球前 Gusto、Joao Pedro 的跑动牵制。",
                "required_test": "At release, defender distances/pressing, own-side support, far-side run timing and rest-defence; the report lacks raw synchronized trajectories.",
            },
        ],
        "goal_timeline": [
            {"minute": 22, "player": "Cole Palmer"},
            {"minute": 30, "player": "Cole Palmer"},
            {"minute": 43, "player": "Joao Pedro"},
        ],
        "score_state_context": {
            "half_time": [3, 0],
            "source_url": MATCH_SOURCE,
            "note": "Club report describes a deeper second-half defence. The late PSG red card cannot explain the earlier 3-0 lead.",
        },
        "interpretation_limits": [
            "A single selected final cannot estimate a causal tactical-matchup advantage.",
            "Pre-final form is descriptive, unadjusted for opponent difficulty and extra time; it is not a calibrated paper-strength rating.",
            "Separate coaching plan, player execution/finishing, score state, temperature, substitutions and opponent absences.",
            "Do not turn whole-match pressure counts into intensity without possession exposure and time windows.",
            "Phase shares retain FIFA definitions and are not renormalized to sum to100.",
            "Game ratings may only serve as dated pre-match priors after verified identity joins; future updates/outcome fields must be excluded.",
        ],
    }


def comparison_figure(case, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    from .visualization.style import COLORS, apply_style, decorate, save

    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.9), sharex=True)
    fields = [
        ("controlled_possession_pct", "Controlled possession", "Match possession · in-contest share separate"),
        ("in_possession_long_ball", "Long-ball phase", "FIFA in-possession phase share"),
        ("out_of_possession_mid_block", "Mid-block phase", "FIFA out-of-possession phase share"),
        ("out_of_possession_counter_press", "Counter-press phase", "FIFA out-of-possession phase share"),
    ]
    decorate(fig, "Chelsea 3–0 PSG: the final against each team's earlier games",
             "2025 Club World Cup · 13 July final · six pre-final matches per team · whole-match descriptive comparison")
    fig.subplots_adjust(left=.10, right=.96, top=.75, bottom=.25, wspace=.22, hspace=.90)
    colors = [COLORS["blue"], COLORS["orange"]]
    for ax, (metric, label, denominator) in zip(axes.flat, fields):
        for side, team in enumerate(case["team_comparisons"]):
            row = team["metrics"][metric]
            y = 1 - side
            values = row.get("pre_final_values", [])
            if values:
                ax.scatter(values, y + np.linspace(.06, .16, len(values)), color=colors[side],
                           alpha=.30, s=22, linewidths=0, zorder=2)
            prior, final = row["pre_final_mean"], row["final"]
            if prior is not None:
                ax.plot([prior, final], [y, y], color=colors[side], alpha=.6, lw=2, zorder=2)
                ax.scatter(prior, y, color="white", edgecolor=colors[side], linewidth=1.8, s=75, zorder=4)
                ax.text(prior, y + .28, f"{prior:.1f}", color=colors[side], ha="center", va="center", fontsize=10)
            ax.scatter(final, y, color=colors[side], edgecolor="white", linewidth=.7, s=90, zorder=5)
            ax.text(max(3, final), y - .28, f"{final:.1f}", color=colors[side], ha="center", va="center", fontsize=10, weight="semibold")
        ax.set_title(label, loc="left", pad=30, fontsize=12)
        ax.text(0, 1.09, denominator, transform=ax.transAxes, color=COLORS["muted"], fontsize=10, va="bottom")
        ax.set_yticks([1, 0], ["Chelsea", "PSG"])
        ax.tick_params(axis="y", length=0, pad=10)
        ax.set_xlim(0, 100)
        ax.set_ylim(-.48, 1.48)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        for spine in ["left", "top", "right"]:
            ax.spines[spine].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("Reported share (%)", labelpad=9)
    fig.legend(handles=[Line2D([], [], marker="o", linestyle="none", color=COLORS["neutral"], markersize=5, alpha=.5, label="Earlier match"),
                        Line2D([], [], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor=COLORS["ink"], markersize=8, label="Pre-final mean"),
                        Line2D([], [], marker="o", linestyle="none", color=COLORS["ink"], markersize=8, label="Final")],
               loc="lower left", bbox_to_anchor=(.048, .14), ncol=3, frameon=False, fontsize=11, handletextpad=.7, columnspacing=2.0)
    fig.text(.055, .035,
             "Sources: FIFA official match reports; Chelsea match report for score state. Chelsea led 3–0 at half-time. Selected case; no causal claim.\nSix earlier games per team; means are not opponent-adjusted or per 90. Different phase denominators retain FIFA definitions, without renormalizing.\nWhole-match shares cannot establish opening pressure or synchronized runs; a lead, conditions and substitutions can change later play.",
             fontsize=10, color=COLORS["muted"], va="bottom", linespacing=1.5)
    save(fig, output, "chelsea-psg-final", formats=("png", "pdf", "svg"))


def run(database, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with Warehouse(database) as db:
        install_scope(db.connection)
        catalog = coverage(db.connection)
        case = chelsea_psg_case(db.connection)
        version = _report_version(db.connection, catalog)
        run_id = hashlib.sha256(
            json.dumps(version, sort_keys=True).encode()
        ).hexdigest()[:16]
        for name, value in [("core-coverage", catalog), ("chelsea-psg-final", case)]:
            value.update(
                {
                    "run_id": run_id,
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                    "inputs": version,
                }
            )
            (output / (name + ".json")).write_text(
                json.dumps(value, ensure_ascii=False, indent=2)
            )
        comparison_figure(case, output)
        db.connection.execute(
            "CREATE TABLE IF NOT EXISTS opponent_tactical_cases(case_id TEXT PRIMARY KEY,run_id TEXT,asset_id INTEGER REFERENCES assets(id),payload TEXT)"
        )
        db.connection.execute(
            "INSERT OR REPLACE INTO opponent_tactical_cases VALUES(?,?,?,?)",
            (
                case["case_id"],
                run_id,
                case["match"]["asset_id"],
                json.dumps(case, ensure_ascii=False),
            ),
        )
        db.connection.commit()
        archive_outputs(db, "elite_tactical_artifacts", output, "elite_tactics", run_id)
        db.connection.commit()
        db.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return {
        "run_id": run_id,
        "coverage_rows": len(catalog["coverage"]),
        "pre_final_form": [t["pre_final_form"] for t in case["team_comparisons"]],
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--database", default="data/football.sqlite3")
    p.add_argument("--output", default=".tmp/elite-tactics")
    p.add_argument("--coverage-only", action="store_true", help="Recheck coverage using a read-only database connection")
    a = p.parse_args()
    command = export_coverage if a.coverage_only else run
    print(json.dumps(command(a.database, a.output), indent=2))
