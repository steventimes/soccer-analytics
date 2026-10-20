"""Chronological model studies and fixed-prediction diagnostics."""

import json

from app.config import load_settings, resolve_competitions


def analyze_or_experiment(args, trainer_type):
    settings = load_settings()
    seasons = args.seasons or settings.training_seasons
    competitions = resolve_competitions(args.competitions, settings)
    from app.data_service.fetch.public import create_data_client

    client = create_data_client()
    report = {
        "source": client.source_name,
        "seasons": seasons,
        "results": [],
        "errors": [],
    }
    for code in competitions:
        try:
            if args.command == "analyze":
                from app.analysis import summarize_matches

                matches = [
                    m
                    for season in seasons
                    for m in client.fetch_matches(
                        code, season=season, status="FINISHED"
                    )
                ]
                result = summarize_matches(matches, backend=args.backend)
            else:
                from app.ml.experiments import run_experiments

                active_trainer = trainer_type(client=client)
                dataset = active_trainer.build_dataset(code, seasons)
                result = run_experiments(
                    dataset,
                    active_trainer.feature_engineer.features,
                    methods=[v.strip() for v in args.models.split(",")],
                    optimizers=[
                        v.strip() for v in args.optimizers.split(",") if v.strip()
                    ],
                    trials=args.trials,
                    seed=args.seed,
                )
            report["results"].append({"competition": code, **result})
        except Exception as exc:
            report["errors"].append(
                {
                    "competition": code,
                    "code": type(exc).__name__,
                    "message": str(exc),
                }
            )
    payload = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0 if report["results"] and not report["errors"] else 1


def compare_outcomes(args):
    from app.file_io import write_json_atomic
    from app.ml.comparison import compare_outcome_file

    report = compare_outcome_file(
        args.predictions, reference=args.reference, replicates=args.replicates, seed=args.seed,
        block_lengths=tuple(int(x.strip()) for x in args.block_lengths.split(",")),
        hac_lags=tuple(int(x.strip()) for x in args.hac_lags.split(",")),
    )
    if args.output:
        write_json_atomic(args.output, report)
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def _match_research_source(args):
    if args.snapshots:
        if args.source in (None, "auto"):
            raise ValueError("本地快照需要明确指定 --source")
        return None, args.source
    from app.data_service.fetch.public import create_data_client

    client = create_data_client()
    return client, client.source_name


def _match_research_matches(args, code, seasons, client, source):
    from app.data_service.history import read_snapshot

    matches, receipts = [], []
    for season in seasons:
        if args.snapshots:
            snapshot = read_snapshot(args.snapshots / code / f"{season}.json")
            if (snapshot.get("source"), snapshot.get("competition"), snapshot.get("season")) != (source, code, season):
                raise ValueError("快照来源、联赛或赛季与请求不符")
            matches.extend(snapshot["matches"])
            receipts.append({k: snapshot[k] for k in ("source", "provider", "competition", "season", "fetched_at", "sha256")})
        else:
            matches.extend(client.fetch_matches(code, season=season, status="FINISHED"))
    return matches, receipts


def _run_match_study(args, evaluate, *, public_error=None):
    """Apply the same source identity and per-league failure policy to each study."""
    import sys
    from app.file_io import write_json_atomic

    settings = load_settings()
    seasons = args.seasons or settings.training_seasons
    competitions = resolve_competitions(args.competitions, settings)
    client, source = _match_research_source(args)
    report = {"source": source, "seasons": seasons, "results": [], "errors": []}
    for code in competitions:
        try:
            matches, receipts = _match_research_matches(args, code, seasons, client, source)
            result = evaluate(matches)
            report["results"].append({"competition": code, "source_snapshots": receipts, **result})
        except Exception as exc:
            if public_error is not None:
                print(f"{code}: {type(exc).__name__}: {exc}", file=sys.stderr)
            report["errors"].append({"competition": code, "code": type(exc).__name__,
                                     "message": public_error if public_error is not None else str(exc)})
    if args.output:
        write_json_atomic(args.output, report)
    else:
        print(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2))
    return 0 if report["results"] and not report["errors"] else 1


def calibrate(args):
    from app.ml.calibration import run_calibration_experiment
    from app.ml.feature_engineering import FeatureEngineer

    def evaluate(matches):
        engineer = FeatureEngineer()
        dataset = engineer.build_training_dataset(matches)
        return run_calibration_experiment(
            dataset, engineer.features, model=args.model, params=args.params,
            calibrators=tuple(value.strip() for value in args.calibrators.split(",") if value.strip()),
            calibration_fraction=args.calibration_fraction, folds=args.folds, seed=args.seed,
        )
    return _run_match_study(args, evaluate)


def rating_baselines(args):
    from app.ml.ordinal import run_rating_experiments

    def evaluate(matches):
        return run_rating_experiments(
            matches, ridge=args.ridge, folds=args.folds, seed=args.seed,
            rating_variants=tuple(value.strip() for value in args.ratings.split(",") if value.strip()),
            warmup_matches=args.warmup_matches,
        )
    return _run_match_study(args, evaluate, public_error="评级基线研究失败；详细原因见本地错误输出。")


def explain_model(args):
    import sys
    from app.file_io import write_json_atomic
    from app.ml.explanations import CLASS_ORDER, run_explanations
    from app.analytics.visualization.explanations import figures

    settings = load_settings()
    seasons = args.seasons or settings.training_seasons
    client, source = _match_research_source(args)
    errors = []
    for code in resolve_competitions(args.competitions, settings):
        try:
            matches, receipts = _match_research_matches(args, code, seasons, client, source)
            report = run_explanations(matches, class_index=CLASS_ORDER.index(args.class_label),
                                      against=CLASS_ORDER.index(args.against) if args.against else None,
                                      max_rows=args.max_rows, seed=args.seed)
            report.update(competition=code, source=source, seasons=seasons, source_snapshots=receipts)
            output = args.output_dir / code
            figures(report, output)
            write_json_atomic(output / 'report.json', report)
        except Exception as exc:
            print(f'{code}: {type(exc).__name__}: {exc}', file=sys.stderr)
            errors.append(code)
    return 1 if errors else 0


def feature_experiment(args):
    from app.ml.feature_experiments import run_feature_experiments

    def evaluate(matches):
        return run_feature_experiments(
            matches, methods=tuple(value.strip() for value in args.models.split(",") if value.strip()),
            optimizer=args.optimizer, trials=args.trials, folds=args.folds, seed=args.seed,
        )
    return _run_match_study(args, evaluate, public_error="特征联合搜索失败；详细原因见本地错误输出。")


def score_models(args):
    from app.ml.score_models import run_score_experiments

    def evaluate(matches):
        return run_score_experiments(
            matches, models=tuple(value.strip() for value in args.models.split(",") if value.strip()),
            half_life_days=args.half_life_days, ridge=args.ridge, folds=args.folds, max_iter=args.max_iter,
        )
    return _run_match_study(args, evaluate)


def calibration_diagnostics(args):
    import numpy as np
    from app.file_io import write_json_atomic
    from app.ml.calibration_diagnostics import calibration_diagnostics as diagnose

    bins = tuple(int(value.strip()) for value in args.bins.split(','))
    with np.load(args.predictions, allow_pickle=False) as data:
        y, p, names = data['targets'], data['probabilities'], data['model_names']
        if (names.ndim != 1 or names.dtype.kind != 'U' or not len(names)
                or len(set(names.tolist())) != len(names) or any(not str(name).strip() for name in names)
                or p.shape != (len(names), len(y), 3)):
            raise ValueError('预测文件的模型名必须非空、唯一，模型/比赛/概率轴须一致')
        results = [{'model': str(name), **diagnose(y, matrix, bins=bins, minimum_bin_count=args.minimum_bin_count)}
                   for name, matrix in zip(names, p)]
    report = {'schema': 'SoccerAnalytics.FixedCalibrationDiagnostics.v1', 'results': results}
    if args.output:
        write_json_atomic(args.output, report)
    else:
        print(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2))
    return 0
