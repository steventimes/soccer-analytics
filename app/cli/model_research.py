"""Chronological model studies and fixed-prediction diagnostics."""

import json

from app.config import load_settings, resolve_competitions
from app.run_receipts import build_run_receipt, content_digest


def _attach_receipt(args, report, *, input_identities=None):
    input_identities = input_identities or {}
    configuration = {name: value for name, value in vars(args).items()
        if name not in {"output", "output_dir", "snapshots", "predictions"}
        and isinstance(value, (str, int, float, bool, list, tuple, dict, type(None)))}
    if getattr(args, 'command', None) in {'score-models', 'rating-baselines', 'feature-experiment', 'calibrate', 'ale', 'search-comparison'}:
        configuration['timing_policy'] = 'observed_result' if getattr(args, 'as_of', None) else 'kickoff_order'
    results, errors = report.get("results", []), report.get("errors", [])
    report["run_receipt"] = build_run_receipt(
        getattr(args, "command", "match-study"), configuration=configuration,
        inputs={"source": report.get("source"), "cohorts": [
            {**{key: row[key] for key in ("competition", "source_snapshots") if key in row},
             **({"input_sha256": input_identities[row["competition"]]}
                if row.get("competition") in input_identities else {})}
            for row in results]},
        stages={row.get("competition", str(i)): (row.get("status", "complete") if getattr(args, "command", None) == "search-comparison" else "complete") for i, row in enumerate(results)}
               | {row.get("competition", str(i)): "failed" for i, row in enumerate(errors)})


def record_forecasts(args):
    import hashlib
    import os
    from pathlib import Path
    from app.ml.study_protocol import register_forecasts

    output = args.output.resolve()
    root = Path(__file__).resolve().parents[2]
    if any(output.is_relative_to(root / name) for name in ('docs', 'results', 'data', 'models')):
        raise ValueError('逐场赛前登记必须写入独立私有目录，例如 .tmp/forecast-registrations')
    raw = args.predictions.read_bytes()
    registration = register_forecasts(json.loads(raw), input_cutoff=args.input_cutoff,
                                      input_sha256=hashlib.sha256(raw).hexdigest())
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'x', encoding='utf-8',
              opener=lambda path, flags: os.open(path, flags, 0o600)) as stream:
        json.dump(registration, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': 'registered', 'path': str(output), 'sha256': registration['sha256'],
                      'prospective_accuracy_verified': False}))
    return 0


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
    input_identities = {}
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
                input_identities[code] = content_digest(matches)
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
                    seed=args.seed, shared_start=getattr(args, "shared_start", False),
                    model_seed=getattr(args, "model_seed", None), beam_width=getattr(args, "beam_width", None),
                )
                input_identities[code] = content_digest({"rows": dataset.rows, "targets": dataset.targets,
                    "match_ids": dataset.match_ids, "kickoffs": dataset.kickoffs})
            report["results"].append({"competition": code, **result})
        except Exception as exc:
            report["errors"].append(
                {
                    "competition": code,
                    "code": type(exc).__name__,
                    "message": str(exc),
                }
            )
    _attach_receipt(args, report, input_identities=input_identities)
    payload = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)
    if args.output:
        from app.file_io import write_json_atomic
        write_json_atomic(args.output, report)
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
    input_identities = {}
    for code in competitions:
        try:
            matches, receipts = _match_research_matches(args, code, seasons, client, source)
            result = evaluate(matches)
            if getattr(args, "command", None) == "search-comparison" and not args.admit_only and result.get("status") != "complete":
                report["errors"].append({"competition": code,
                    "code": "SearchNotReady" if result.get("status") == "not_ready" else "SearchIncomplete",
                    "message": "时间准入通过，但实际训练历史不足以支持本轮特征搜索采用；未执行拟合。"
                    if result.get("status") == "not_ready" else "搜索队列未准入或存在失败比较臂。"})
            report["results"].append({"competition": code, "source_snapshots": receipts, **result})
            input_identities[code] = content_digest(matches)
        except Exception as exc:
            if public_error is not None:
                print(f"{code}: {type(exc).__name__}: {exc}", file=sys.stderr)
            report["errors"].append({"competition": code, "code": type(exc).__name__,
                                     "message": public_error if public_error is not None else str(exc)})
    _attach_receipt(args, report, input_identities=input_identities)
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


def ale(args):
    from app.ml.ale import run_ale_study

    def evaluate(matches):
        return run_ale_study(matches, features=tuple(v.strip() for v in args.features.split(',')),
                             model=args.model, params=args.params, class_index=args.class_index,
                             bins=args.bins, max_rows=args.max_rows, seed=args.seed, as_of=args.as_of)
    return _run_match_study(args, evaluate)


def season_study(args):
    from datetime import UTC, datetime
    from app.data_service.history import load_history
    from app.file_io import write_json_atomic
    from app.ml.availability import aware_time
    from app.ml.seasonal import run_season_matrix
    from app.ml.study_protocol import freeze_season_protocol, validate_season_protocol

    if args.protocol and args.output and args.protocol.resolve() == args.output.resolve():
        raise ValueError('评估输出不能覆盖封存协议')
    collections, manifest = load_history(args.history)
    tests = tuple(v.strip() for v in args.test_seasons.split(','))
    models = tuple(v.strip() for v in args.models.split(','))
    windows = tuple(v.strip() for v in args.windows.split(','))
    as_of = aware_time(args.as_of).isoformat()
    if args.freeze_protocol:
        if len(tests) != 1 or args.timing_policy != 'observed':
            raise ValueError('前瞻注册要求一个测试赛季和严格结果时点')
        if not args.evaluation_start or not args.evaluation_end:
            raise ValueError('注册需要评估区间开始和结束时间')
        # Registration time is the real clock, not a caller-supplied retrospective as-of.
        if aware_time(as_of) > datetime.now(UTC):
            raise ValueError('注册历史截止时点不能位于未来')
        report = freeze_season_protocol(collections, test_season=tests[0],
            evaluation_start=args.evaluation_start, evaluation_end=args.evaluation_end,
            models=models, windows=windows, seed=args.seed, training_as_of=as_of)
        args.freeze_protocol.parent.mkdir(parents=True, exist_ok=True)
        with args.freeze_protocol.open('x') as stream:
            json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write('\n')
        print(json.dumps({'status': 'registered', 'path': str(args.freeze_protocol), 'sha256': report['sha256']}))
        return 0
    registration = None
    if args.protocol:
        protocol = json.loads(args.protocol.read_text())
        if (list(tests) != [protocol['test_season']] or list(models) != protocol['models']
                or list(windows) != protocol['windows'] or args.seed != protocol['seed']
                or args.timing_policy != 'observed'):
            raise ValueError('请求设置与封存协议不一致')
        registration = validate_season_protocol(protocol, collections, as_of=as_of)
    report = run_season_matrix(collections, test_seasons=tests, windows=windows,
                              models=models, seed=args.seed, pooled=False,
                              as_of=as_of if args.timing_policy == 'observed' else None)
    report['registration'] = registration
    report['run_receipt'] = build_run_receipt('season-study',
        configuration={'test_seasons': tests, 'windows': windows, 'models': models,
                       'seed': args.seed, 'timing_policy': args.timing_policy, 'as_of': as_of},
        inputs={'history_manifest': content_digest(manifest)},
        stages={'evaluation': 'complete' if report['results'] and not report['errors'] else 'failed'})
    if args.output:
        write_json_atomic(args.output, report)
    else:
        print(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2))
    return 0 if report['results'] and not report['errors'] else 1


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
            as_of=getattr(args, 'as_of', None), shared_start=getattr(args, "shared_start", False),
            model_seed=getattr(args, "model_seed", None), beam_width=getattr(args, "beam_width", None),
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


def search_comparison(args):
    from pathlib import Path
    from app.file_io import json_bytes, write_bytes_atomic
    from app.ml.feature_experiments import run_search_comparison

    private = []
    if args.output and args.output.resolve().is_relative_to(args.snapshots.resolve()):
        raise ValueError('汇总报告不能覆盖输入快照')
    if args.private_predictions is not None:
        root = Path(__file__).resolve().parents[2]
        target = args.private_predictions.resolve()
        if any(target.is_relative_to(root / name) for name in ('docs', 'results', 'data', 'models', 'app', 'tests', 'scripts', '.git')):
            raise ValueError('逐场预测只允许独立私有目录，例如 .tmp/search-comparison')
        if target.is_relative_to(args.snapshots.resolve()) or (args.output and target == args.output.resolve()):
            raise ValueError('逐场预测不能覆盖输入快照或汇总报告')
        if target.exists():
            raise FileExistsError('逐场预测文件已存在，请使用新的私有输出路径')
    def evaluate(matches):
        return run_search_comparison(matches, as_of=args.as_of, evaluation_start=args.evaluation_start,
            evaluation_end=args.evaluation_end, final_holdout_start=args.final_holdout_start,
            search_seeds=tuple(int(seed.strip()) for seed in args.search_seeds.split(',')),
            model_seed=args.model_seed, trials=args.trials, folds=args.folds,
            arms=tuple(a.strip() for a in args.arms.split(',')), admit_only=args.admit_only,
            prediction_sink=private.append if args.private_predictions else None,
            initial_training_rows=getattr(args, "initial_training_rows", None))
    status = _run_match_study(args, evaluate, public_error='严格搜索对照失败，原因见本地诊断。')
    if args.private_predictions is not None:
        write_bytes_atomic(args.private_predictions,
            json_bytes({'schema': 'SoccerAnalytics.PrivateSearchPredictions.v1', 'rows': private}), file_mode=0o600)
    return status
