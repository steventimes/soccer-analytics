"""Stable research command dispatch with lazy scientific dependencies.

Legacy handler imports are retained; implementations are grouped by task.
"""

import json

from app.cli.collection import (
    collect_open_data,
    scout_bodypose,
    scout_events,
    scout_tracking,
    history,
)
from app.cli.scouting import (
    assess,
    scout_insights,
    tactics,
    movement_lab,
    scout_figures,
    representation_lab,
    analytics_lab,
)
from app.cli.model_research import (
    analyze_or_experiment,
    compare_outcomes,
    calibrate,
    rating_baselines,
    explain_model,
    feature_experiment,
    score_models,
    calibration_diagnostics,
)


def export_research(args):
    from app.web.research import export_research_data

    result = export_research_data(
        args.results_dir, args.output_dir, experiment_report=args.experiment_report
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


HANDLERS = {
    "calibration-diagnostics": calibration_diagnostics,
    "rating-baselines": rating_baselines,
    "explain-model": explain_model,
    "feature-experiment": feature_experiment,
    "compare-outcomes": compare_outcomes,
    "score-models": score_models,
    "calibrate": calibrate,
    "passing-lab": analytics_lab,
    "value-lab": analytics_lab,
    "space-lab": analytics_lab,
    "buildup-lab": analytics_lab,
    "representation-lab": representation_lab,
    "export-research": export_research,
    "assess": assess,
    "scout-insights": scout_insights,
    "tactics": tactics,
    "movement-lab": movement_lab,
    "collect-open-data": collect_open_data,
    "scout-events": scout_events,
    "scout-tracking": scout_tracking,
    "scout-bodypose": scout_bodypose,
    "scout-figures": scout_figures,
    "collect-history": history,
    "compare-history": history,
}


def run_research_command(args, *, trainer_type):
    if args.command in {"analyze", "experiment"}:
        return analyze_or_experiment(args, trainer_type)
    handler = HANDLERS.get(args.command)
    return handler(args) if handler is not None else None
