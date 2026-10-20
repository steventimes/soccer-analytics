"""Export validated public research summaries without loading the warehouse.

The projection functions remain importable here for existing callers. Their
implementations live with their event, diagnostic, experiment, or figure schema.
"""

import json
from pathlib import Path

from app.file_io import write_bytes_atomic, write_json_atomic

from app.web.public_diagnostics import (
    public_xt_numerics,
    public_vaep_calibration,
    public_spadl_vaep,
)
from app.web.public_events import (
    public_fifa_offers,
    public_passing_motifs,
    public_shot_sequences,
    public_exploration,
)
from app.web.public_experiments import (
    public_spatial_style,
    public_literature,
    public_market_reference,
    public_representation_comparisons,
    public_representation_summary,
    public_model_lab,
)
from app.web.public_figures import (
    PUBLIC_FIGURES,
    public_figures,
)

CASE_METRICS = (
    "controlled_possession_pct",
    "in_possession_long_ball",
    "out_of_possession_mid_block",
    "out_of_possession_counter_press",
)



def display_season(row):
    raw = row["season"]
    if (
        row["provider"] == "understat"
        and len(raw) == 4
        and raw.isascii()
        and raw.isdecimal()
    ):
        first, second = int(raw[:2]), int(raw[2:])
        if (first + 1) % 100 == second:
            return f"{2000 + first}/{2000 + second}"
    return raw



def export_research_data(results_dir, output_dir, *, experiment_report=None):
    source, output = Path(results_dir), Path(output_dir)
    # Read all required inputs before writing; failures leave an existing page intact.
    coverage = json.loads((source / "core-coverage.json").read_text())
    case = json.loads((source / "chelsea-psg-final.json").read_text())
    methods = json.loads((source / "methods.json").read_text())
    report_path = (
        Path(experiment_report)
        if experiment_report is not None
        else source / "representation-learning.json"
    )
    representation = public_representation_summary(json.loads(report_path.read_text()))
    figure_catalog, figure_assets = public_figures(source)
    if (
        len(case["team_comparisons"]) != 2
        or not coverage["coverage"]
        or not methods["methods"]
    ):
        raise ValueError("Missing research summaries")
    payloads = {
        "coverage.json": {
            "observed_at": coverage["observed_at"],
            "primary_competitions": coverage["primary_competitions"],
            "scope": coverage.get("scope"),
            "summary_scope": coverage.get("summary_scope"),
            "coverage": [
                {**row, "source_season": row["season"], "season": display_season(row)}
                for row in coverage["coverage"]
            ],
        },
        "case.json": {
            "case_id": case["case_id"],
            "observed_at": case["observed_at"],
            "match": {
                key: case["match"][key]
                for key in (
                    "date",
                    "teams",
                    "goals",
                    "source_url",
                    "in_contest_pct",
                    "data_grain",
                )
            },
            "team_comparisons": [
                {
                    "team": team["team"],
                    "pre_final_form": team["pre_final_form"],
                    "metrics": {key: team["metrics"][key] for key in CASE_METRICS},
                }
                for team in case["team_comparisons"]
            ],
            "score_state_context": case["score_state_context"],
            "interpretation_limits": case["interpretation_limits"],
        },
        "methods.json": methods,
        "representation.json": representation,
        "figures.json": figure_catalog,
    }
    exploration = source / "exploration"
    # Explicit derivative artifacts only; never recursively publish the source cache.
    if (exploration / "index.json").is_file():
        payloads.update(public_exploration(exploration))
    offers = source / "fifa-offers.json"
    if offers.is_file():
        payloads["fifa-offers.json"] = public_fifa_offers(json.loads(offers.read_text()))
    market = source / "market-reference.json"
    if market.is_file():
        payloads["market-reference.json"] = public_market_reference(json.loads(market.read_text()))
    literature = source / "literature.json"
    if literature.is_file():
        payloads["literature.json"] = public_literature(json.loads(literature.read_text()))
    spatial = source / "spatial-style.json"
    if spatial.is_file():
        payloads["spatial-style.json"] = public_spatial_style(json.loads(spatial.read_text()))
    vaep = source / "vaep-spadl.json"
    if vaep.is_file():
        payloads["vaep-spadl.json"] = public_spadl_vaep(json.loads(vaep.read_text()))
    calibration = source / "vaep-calibration.json"
    if calibration.is_file():
        payloads["vaep-calibration.json"] = public_vaep_calibration(json.loads(calibration.read_text()), json.loads(vaep.read_text()))
    xt = source / "xt-numerics.json"
    if xt.is_file():
        payloads["xt-numerics.json"] = public_xt_numerics(json.loads(xt.read_text()))
    lab = source / "paper-followup-20261019"
    lab_paths = {name: lab / f"{name}.json" for name in ("scores", "ratings", "calibration", "trees")}
    if any(path.is_file() for path in lab_paths.values()):
        if not all(path.is_file() for path in lab_paths.values()):
            raise ValueError("Incomplete model-lab source set")
        payloads["model-lab.json"] = public_model_lab({name: json.loads(path.read_text()) for name, path in lab_paths.items()})
    # Validate strict JSON for every payload before any mutation.
    for payload in payloads.values():
        json.dumps(payload, allow_nan=False)
    if figure_assets:
        (output / "figures").mkdir(parents=True, exist_ok=True)
        for name, content in figure_assets.items():
            write_bytes_atomic(output / "figures" / name, content)
    for name, payload in payloads.items():
        write_json_atomic(output / name, payload)
    return {
        "files": [str(output / name) for name in payloads],
        "figure_files": [str(output / "figures" / name) for name in figure_assets],
        "data_observed_at": coverage["observed_at"],
        "methods_checked_at": methods["checked_at"],
    }
