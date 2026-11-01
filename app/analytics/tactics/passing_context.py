"""Read-only, opportunity-conditioned descriptions of local SkillCorner events.

Passing-option model values are retrospective supplier estimates. They are never
used to rank alternatives, calculate regret, or train a release-time model here.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

from app.file_io import write_json_atomic


QUALITY_FIELDS = (
    "fully_extrapolated", "is_player_possession_start_matched",
    "is_player_possession_end_matched", "is_previous_pass_matched",
    "is_pass_reception_matched",
)
EPV_FIELDS = (
    "possession_epv_for_start", "possession_epv_for_end",
    "possession_epv_against_start", "possession_epv_against_end",
    "pass_reception_epv_for", "pass_reception_epv_against",
    "possession_epv_total", "pass_epv_total",
    "possession_epv_delta_for", "possession_epv_delta_against",
    "pass_epv_delta_for", "pass_epv_delta_against",
)
COORDINATES = ("x_start", "y_start", "x_end", "y_end")
KNOWN_CHAIN_OUTCOMES = {"regain", "disruption"}


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _frame(value):
    number = _number(value)
    return int(number) if number is not None and number.is_integer() else None


def _flag(value):
    if value is True or value == "True":
        return "true"
    if value is False or value == "False":
        return "false"
    return "unknown"


def _label(value):
    return value or "unknown"


def _counts(values):
    return dict(sorted(Counter(values).items()))


def _quality(rows):
    return {field: _counts(_flag(row.get(field)) for row in rows) for field in QUALITY_FIELDS}


def _numeric_summary(values):
    values = [number for value in values if (number := _number(value)) is not None]
    return {
        "n": len(values), "sum": sum(values) if values else None,
        "mean": sum(values) / len(values) if values else None,
        "min": min(values) if values else None, "max": max(values) if values else None,
    }


def _epv(rows):
    rows = list(rows)
    return {field: _numeric_summary(row.get(field) for row in rows) for field in EPV_FIELDS}


def _join_reason(option, possession):
    if possession is None:
        return "missing_possession_reference"
    for field in ("period", "team_id"):
        if not option.get(field) or not possession.get(field):
            return "unknown_" + field
        if option[field] != possession[field]:
            return field + "_mismatch"
    start, end, pp_start, pp_end = (
        _frame(row.get(field)) for row, field in (
            (option, "frame_start"), (option, "frame_end"),
            (possession, "frame_start"), (possession, "frame_end"),
        )
    )
    if None in (start, end, pp_start, pp_end):
        return "unknown_frame"
    if start > end or pp_start > pp_end:
        return "invalid_frame_interval"
    if start > pp_end or end < pp_start:
        return "nonoverlapping_possession_interval"
    return None


def build_pass_contexts(rows, match_id):
    """Join options to their possession spell, retaining the release flag itself.

    Supplier release flags sometimes extend beyond the option's own interval.
    Report this discrepancy; do not replace the supplier availability definition
    with interval containment or silently include a targeted unavailable option.
    """
    possessions = {row["event_id"]: row for row in rows if row["event_type"] == "player_possession"}
    options = {row["event_id"]: row for row in rows if row["event_type"] == "passing_option"}
    by_possession = defaultdict(list)
    rejected = Counter()
    peak_timing = Counter()
    for row in options.values():
        pp = possessions.get(row.get("associated_player_possession_event_id"))
        reason = _join_reason(row, pp)
        if reason:
            rejected[reason] += 1
            continue
        by_possession[pp["event_id"]].append(row)
        peak, release = _frame(row.get("peak_passing_option_frame")), _frame(pp.get("frame_end"))
        timing = "unknown" if peak is None else ("before" if peak < release else "after" if peak > release else "same")
        peak_timing[timing] += 1
    cases = []
    for pp in possessions.values():
        if pp.get("end_type") != "pass":
            continue
        joined = by_possession[pp["event_id"]]
        available = [row for row in joined if _flag(row.get("passing_option_at_pass_moment")) == "true"]
        target_id = pp.get("targeted_passing_option_event_id") or None
        target = options.get(target_id)
        if not target_id:
            target_status = "missing_link"
        elif target is None:
            target_status = "unresolved"
        elif target.get("associated_player_possession_event_id") != pp["event_id"] or _join_reason(target, pp):
            target_status = "invalid_join"
        else:
            target_status = _flag(target.get("passing_option_at_pass_moment"))
        release = _frame(pp.get("frame_end"))
        outside = sum(not (_frame(row["frame_start"]) <= release <= _frame(row["frame_end"])) for row in available)
        cases.append({
            "match_id": str(match_id), "event_id": pp["event_id"],
            "period": pp.get("period"), "team_id": pp.get("team_id"),
            "release_frame": release, "pass_outcome": _label(pp.get("pass_outcome")),
            "release_options": len(available),
            "release_option_ids": [row["event_id"] for row in available],
            "joined_options_any_time": len(joined),
            "option_availability": _counts(_flag(row.get("passing_option_at_pass_moment")) for row in joined),
            "selected_option_id": target_id,
            "selected_receiver_availability": target_status,
            "selected_receiver_unavailable": target_status == "false",
            "release_flag_outside_option_interval": outside,
            "supplier_pressure_start": _label(pp.get("overall_pressure_start")),
            "supplier_pressure_release": _label(pp.get("overall_pressure_end")),
            "supplier_attacking_phase": _label(pp.get("team_in_possession_phase_type")),
            "supplier_defending_phase": _label(pp.get("team_out_of_possession_phase_type")),
            "passer_position": _label(pp.get("player_position")),
            "game_state": _label(pp.get("game_state")),
            "quality": {field: _flag(pp.get(field)) for field in QUALITY_FIELDS},
            "release_option_quality": _quality(available),
            "coordinates": {field: _number(pp.get(field)) for field in COORDINATES},
            "supplier_epv": {field: _number(pp.get(field)) for field in EPV_FIELDS},
        })
    return cases, {
        "passing_options": len(options), "passing_options_joined": sum(map(len, by_possession.values())),
        "rejected_option_reasons": dict(sorted(rejected.items())),
        "joined_option_peak_frame_relative_to_possession_end": dict(sorted(peak_timing.items())),
    }


def summarize_passes(cases):
    outcomes = _counts(row["pass_outcome"] for row in cases)
    known = outcomes.get("successful", 0) + outcomes.get("unsuccessful", 0) + outcomes.get("offside", 0)
    return {
        "pass_attempts": len(cases), "outcomes": outcomes, "known_outcomes": known,
        "completion_rate_known": outcomes.get("successful", 0) / known if known else None,
        "passes_with_release_option": sum(row["release_options"] > 0 for row in cases),
        "passes_with_two_or_more_release_options": sum(row["release_options"] >= 2 for row in cases),
        "release_available_options": sum(row["release_options"] for row in cases),
        "release_option_count_distribution": _counts(str(row["release_options"]) for row in cases),
        "selected_receiver_availability": _counts(row["selected_receiver_availability"] for row in cases),
        "release_flag_outside_option_interval": sum(row["release_flag_outside_option_interval"] for row in cases),
        "quality": {field: _counts(row["quality"][field] for row in cases) for field in QUALITY_FIELDS},
        "release_option_quality": {
            field: dict(sorted(sum((Counter(row["release_option_quality"][field]) for row in cases), Counter()).items()))
            for field in QUALITY_FIELDS
        },
        "supplier_epv": _epv(row["supplier_epv"] for row in cases),
    }


def _chain_summary(records):
    outcomes = _counts(row["outcome"] for row in records)
    known = sum(outcomes.get(kind, 0) for kind in KNOWN_CHAIN_OUTCOMES)
    return {
        "chains": len(records), "outcomes": outcomes, "known_outcome_chains": known,
        "conflicting_outcome_chains": outcomes.get("conflict", 0),
        "regain_rate_known": outcomes.get("regain", 0) / known if known else None,
        "disruption_rate_known": outcomes.get("disruption", 0) / known if known else None,
        "observed_regain_fraction_all_chains": outcomes.get("regain", 0) / len(records) if records else None,
        "associated_possession_links": sum(row["associated_possessions"] for row in records),
        "opponent_supplier_epv_chain_means": {
            field: {key: value for key, value in _numeric_summary(
                row["opponent_supplier_epv"][field]["mean"] for row in records
            ).items() if key != "sum"}
            for field in EPV_FIELDS
        },
        "epv_weighting": "Equal weight per contributing chain, after averaging distinct linked opponent possession spells within that chain; n counts chains with that component. Spells can occur in multiple chains. No additive or causal credit.",
    }


def summarize_pressing_chains(rows, match_id):
    """One chain per match/period/pressing-team/index, with explicit unknowns."""
    groups = defaultdict(list)
    engagements = [row for row in rows if row["event_type"] == "on_ball_engagement"]
    possessions = {row["event_id"]: row for row in rows if row["event_type"] == "player_possession"}
    incomplete = 0
    for row in engagements:
        if not row.get("pressing_chain_index"):
            continue
        if not row.get("period") or not row.get("team_id"):
            incomplete += 1
            continue
        groups[(str(match_id), row["period"], row["team_id"], row["pressing_chain_index"])].append(row)
    records = []
    for key, members in sorted(groups.items()):
        observed = sorted({row["pressing_chain_end_type"] for row in members if row.get("pressing_chain_end_type")})
        conflict = len(observed) > 1
        outcome = "conflict" if conflict else observed[0] if observed and observed[0] in KNOWN_CHAIN_OUTCOMES else "unknown"
        associated = {}
        rejected = Counter()
        for row in members:
            pp = possessions.get(row.get("associated_player_possession_event_id"))
            if pp is None:
                rejected["missing_possession_reference"] += 1
            elif row.get("period") != pp.get("period"):
                rejected["period_mismatch"] += 1
            elif not pp.get("team_id") or row.get("team_id") == pp.get("team_id"):
                rejected["unknown_or_same_team"] += 1
            else:
                # Reuse temporal validation with the opponent's team for the join.
                reason = _join_reason(dict(row, team_id=pp["team_id"]), pp)
                if reason:
                    rejected[reason] += 1
                else:
                    associated[pp["event_id"]] = pp
        labels = {}
        for field in ("team_out_of_possession_phase_type", "game_state"):
            values = {_label(row.get(field)) for row in members}
            labels[field] = next(iter(values)) if len(values) == 1 else "mixed"
        records.append({
            "match_id": key[0], "period": key[1], "pressing_team_id": key[2], "chain_index": key[3],
            "members": len(members), "outcome": outcome,
            "observed_outcome_labels": observed, "outcome_conflict": conflict,
            "members_without_outcome": sum(not row.get("pressing_chain_end_type") for row in members),
            "quality": _quality(members), "supplier_defending_phase": labels["team_out_of_possession_phase_type"],
            "game_state": labels["game_state"], "associated_possessions": len(associated),
            "rejected_possession_join_reasons": dict(sorted(rejected.items())),
            "opponent_supplier_epv": _epv(list(associated.values())),
        })
    summary = _chain_summary(records)
    summary.update({
        "engagement_rows": len(engagements),
        "members_with_chain_index": sum(bool(row.get("pressing_chain_index")) for row in engagements),
        "members_missing_chain_identity": incomplete,
        "raw_member_outcomes": _counts(_label(row.get("pressing_chain_end_type")) for row in engagements if row.get("pressing_chain_index")),
        "quality": _quality(engagements),
    })
    return summary, records


def _groups(rows, fields, summarize):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    return [{**dict(zip(fields, key)), **summarize(values)} for key, values in sorted(groups.items())]


def _coordinates(rows, metadata):
    length, width = _number(metadata.get("pitch_length")), _number(metadata.get("pitch_width"))
    outside = []
    unknown_extent = length is None or width is None or length <= 0 or width <= 0
    for row in rows:
        coords = {field: _number(row.get(field)) for field in COORDINATES}
        if not unknown_extent and any(value is not None and abs(value) > (length if field.startswith("x") else width) / 2
                                      for field, value in coords.items()):
            outside.append({"event_id": row["event_id"], "event_type": row["event_type"], **coords})
    return {
        "pitch_length_m": length, "pitch_width_m": width, "unknown_pitch_extent": unknown_extent,
        "rows_outside_nominal_pitch": len(outside), "outside_examples_raw_unclipped": outside[:10],
        "ranges_raw_unclipped": {
            field: {key: value for key, value in _numeric_summary(row.get(field) for row in rows).items() if key in {"n", "min", "max"}}
            for field in COORDINATES
        },
    }


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _split(matches, cases, chains):
    dated, undated = [], []
    for match in matches:
        try:
            stamp = datetime.fromisoformat(match["date_time"].replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                raise ValueError("missing timezone")
            dated.append((stamp.astimezone(timezone.utc), match["match_id"]))
        except (AttributeError, TypeError, ValueError):
            undated.append(match["match_id"])
    dated.sort()
    cut = min(len(dated) - 1, max(1, int(len(dated) * 0.8))) if len(dated) >= 2 else len(dated)
    while 0 < cut < len(dated) and dated[cut - 1][0] == dated[cut][0]:
        cut -= 1
    if cut == 0:
        cut = len(dated)
    result = {
        "method": "Complete matches sorted by UTC kickoff; latest approximately 20 percent, with tied kickoffs kept together.",
        "status": "descriptive_match_holdout" if cut < len(dated) else "insufficient_distinct_match_dates",
        "model_trained": False, "predictive_metrics": None, "excluded_missing_date_match_ids": undated,
    }
    for label, cohort in (("earlier", dated[:cut]), ("later", dated[cut:])):
        ids = [mid for _, mid in cohort]
        result[label] = {
            "match_ids": ids, "kickoffs_utc": [stamp.isoformat() for stamp, _ in cohort],
            "passing": summarize_passes([row for row in cases if row["match_id"] in ids]),
            "pressing_chains": _chain_summary([row for row in chains if row["match_id"] in ids]),
        }
    return result


def run(data_dir=".tmp/open-data/skillcorner/data/matches",
        output=".tmp/analysis-upgrades-20261014/passing"):
    """Consume all local dynamic-event CSV/metadata pairs without database writes."""
    started = time.perf_counter()
    root, output = Path(data_dir), Path(output)
    sources = sorted(root.glob("*/*_dynamic_events.csv"))
    if not sources:
        raise ValueError(f"No SkillCorner dynamic-event CSVs under {root}")
    cases, chain_records, matches, receipts = [], [], [], []
    all_epv_rows, legacy = [], defaultdict(list)
    for source in sources:
        mid = source.parent.name
        metadata_path = source.parent / f"{mid}_match.json"
        if not mid.isdigit() or source.name != f"{mid}_dynamic_events.csv":
            raise ValueError(f"Unexpected SkillCorner source identity: {source}")
        metadata = json.loads(metadata_path.read_text())
        identity = metadata.get("id")
        if isinstance(identity, bool) or not isinstance(identity, int) or identity <= 0 or str(identity) != mid:
            raise ValueError(f"SkillCorner metadata identity differs from source path: {source}")
        with source.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {"event_id", "event_type", "period", "team_id", "frame_start", "frame_end",
                        "end_type", "associated_player_possession_event_id", "passing_option_at_pass_moment"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Missing SkillCorner fields in {source}: {sorted(missing)}")
            rows = list(reader)
        ids = [row["event_id"] for row in rows]
        if not all(ids) or len(set(ids)) != len(ids):
            raise ValueError(f"Missing or duplicated event IDs in {source}")
        if any(row.get("match_id") and row["match_id"] != mid for row in rows):
            raise ValueError(f"Dynamic-event match identity mismatch in {source}")
        match_cases, joins = build_pass_contexts(rows, mid)
        chain_summary, match_chains = summarize_pressing_chains(rows, mid)
        possessions = [row for row in rows if row["event_type"] == "player_possession"]
        by_type = defaultdict(list)
        for row in rows:
            by_type[row["event_type"]].append(row)
            legacy[row["event_type"]].append({field: row.get(field) for field in (
                "xthreat", "possession_danger", "xpass_completion", "passing_option_score",
            )})
        matches.append({
            "match_id": mid, "date_time": metadata.get("date_time"), "rows": len(rows),
            "event_types": _counts(row["event_type"] for row in rows),
            "passing": summarize_passes(match_cases), "joins": joins, "pressing_chains": chain_summary,
            "supplier_epv_all_possessions": _epv(possessions),
            "context_field_audit": {
                field: {"possession_rows": len(possessions),
                        "nonempty": sum(bool(row.get(field)) for row in possessions)}
                for field in ("overall_pressure_start", "overall_pressure_end", "game_state",
                              "team_in_possession_phase_type", "team_out_of_possession_phase_type", "player_position")
            },
            "quality_by_event_type": {kind: _quality(records) for kind, records in sorted(by_type.items())},
            "coordinates": _coordinates(rows, metadata),
        })
        cases.extend(match_cases)
        chain_records.extend(match_chains)
        all_epv_rows.extend({field: row.get(field) for field in EPV_FIELDS} for row in possessions)
        for path in (source, metadata_path):
            receipts.append({"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _digest(path)})
    config = {"version": 1, "provider": "skillcorner", "release_denominator": "passing_option_at_pass_moment=True",
              "chain_key": ["match_id", "period", "pressing_team_id", "pressing_chain_index"],
              "holdout_fraction": 0.2, "model_trained": False}
    extractor_hash = _digest(Path(__file__))
    signature = {"config": config, "extractor_sha256": extractor_hash,
                 "inputs": [{"path": str(Path(row["path"]).relative_to(root.resolve())), "sha256": row["sha256"]} for row in receipts]}
    run_id = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:20]
    join_counts = Counter()
    rejects = Counter()
    peak_timing = Counter()
    for match in matches:
        joins = match["joins"]
        for key in ("passing_options", "passing_options_joined"):
            join_counts[key] += joins[key]
        rejects.update(joins["rejected_option_reasons"])
        peak_timing.update(joins["joined_option_peak_frame_relative_to_possession_end"])
    chain_summary = _chain_summary(chain_records)
    for field in ("engagement_rows", "members_with_chain_index", "members_missing_chain_identity"):
        chain_summary[field] = sum(match["pressing_chains"][field] for match in matches)
    chain_summary["raw_member_outcomes"] = dict(sorted(sum(
        (Counter(match["pressing_chains"]["raw_member_outcomes"]) for match in matches), Counter()
    ).items()))
    report = {
        "run_id": run_id, "provider": "skillcorner", "evidence_type": "executed_local_descriptive_analysis",
        "matches": len(matches), "rows": sum(match["rows"] for match in matches),
        "passing": summarize_passes(cases),
        "joins": {**dict(join_counts), "rejected_option_reasons": dict(sorted(rejects.items())),
                  "joined_option_peak_frame_relative_to_possession_end": dict(sorted(peak_timing.items()))},
        "passing_context_groups": _groups(cases, (
            "release_options", "supplier_pressure_release", "supplier_attacking_phase",
            "supplier_defending_phase", "passer_position", "game_state",
        ), summarize_passes),
        "passing_by_pressure": _groups(cases, ("supplier_pressure_release",), summarize_passes),
        "passing_by_game_state": _groups(cases, ("game_state",), summarize_passes),
        "pressing_chains": chain_summary,
        "pressing_chain_context_groups": _groups(chain_records, ("supplier_defending_phase", "game_state"), _chain_summary),
        "supplier_epv_all_possessions": _epv(all_epv_rows),
        "legacy_supplier_descriptors_by_event_type": {
            kind: {"xthreat": _numeric_summary(row.get("xthreat") for row in values),
                   "possession_danger": _counts(_flag(row.get("possession_danger")) for row in values),
                   "xpass_completion": _numeric_summary(row.get("xpass_completion") for row in values),
                   "passing_option_score": _numeric_summary(row.get("passing_option_score") for row in values)}
            for kind, values in sorted(legacy.items())
        },
        "chronological_split": _split(matches, cases, chain_records), "match_reports": matches,
        "metric_definitions": {
            "supplier_epv": "Supplier estimates: 2026 EPV documentation specifies 30 seconds or interruption. For and against components remain separate; total equals delta_for minus delta_against.",
            "xthreat": "Legacy supplier scoring-danger estimate: 10-second target; option values may be evaluated at an option-specific retrospective peak.",
            "possession_danger": "Legacy supplier boolean label: linked specification describes a 90-second target. It is not the new EPV total.",
            "completion_rate_known": "successful / (successful + unsuccessful + offside); missing or other outcomes remain separate categories.",
            "regain_rate_known": "regain / (regain + disruption); unknown and conflict remain in all-chain counts and are excluded from known-outcome rates.",
        },
        "limits": [
            "Supplier models are not executed, reproduced, trained, calibrated, or causally validated here.",
            "Release options are only linked passing_option rows with supplier availability True. Targeted False or unknown availability never enters that denominator.",
            "Joins require the same match, period, and attacking team, and an overlapping possession interval. A release flag outside the option interval is separately reported.",
            "Option peak values and event phase/structure labels can use future frames. Aggregate legacy descriptors are retrospective; no alternative ranking or decision model is fitted.",
            "Later complete matches are a descriptive holdout cohort. There are no predictive evaluation scores or claims of out-of-sample model accuracy.",
            "False fully_extrapolated does not prove every frame was detected. Missing quality fields remain unknown; quality flags do not silently filter observations.",
            "Coordinates outside nominal pitch extents are retained without clipping; examples and source hashes preserve their audit trail.",
            "Chain-associated EPV describes distinct opponent possession spells within each chain. A spell can occur in multiple chains; this is not additive pressing credit or causal value.",
            "The local public broadcast-tracking sample is not a global player ranking or evidence of coach intent.",
        ],
        "source_fingerprint": hashlib.sha256(json.dumps(signature["inputs"], sort_keys=True).encode()).hexdigest(),
        "extractor_sha256": extractor_hash,
        "artifacts": ["report.json", "provenance.json", "pass-contexts.json", "pressing-chains.json"],
        "runtime_seconds": time.perf_counter() - started,
    }
    provenance = {"run_id": run_id, "signature": signature, "files": receipts,
                  "created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "references": ["https://github.com/SkillCorner/opendata", "https://skillcorner.github.io/opendata/"]}
    for filename, payload in (("provenance.json", provenance), ("pass-contexts.json", {"cases": cases}),
                              ("pressing-chains.json", {"chains": chain_records}), ("report.json", report)):
        write_json_atomic(output / filename, payload)
    return report
