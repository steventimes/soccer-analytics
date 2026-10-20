"""Scouting, tactical and representation research commands."""

import json
from pathlib import Path



def assess(args):
    from app.analytics.assessments import run
    from app.analytics.assessment_models import METHODS

    selected = tuple(
        dict.fromkeys(
            value.strip() for value in args.models.split(",") if value.strip()
        )
    )
    if not selected or any(value not in METHODS for value in selected):
        raise ValueError("未知评估模型")
    result = run(str(args.database), str(args.output_dir), selected, args.train_cap)
    return int(
        any(value["failures"] for value in result["benchmarks"].values())
        or bool(result["team_poisson"]["failures"])
    )


def scout_insights(args):
    from app.analytics.scouting_intelligence import run

    run(
        str(args.database),
        str(args.output_dir),
        str(args.assessment_dir),
        args.player_as_of,
        args.backend,
    )
    return 0


def tactics(args):
    from app.analytics.tactical_models import run

    report = run(str(args.database), str(args.output_dir), args.train_cap)
    return int(
        bool(
            report["chance_models"]["failures"]
            or report["feature_ablations"]["failures"]
        )
    )


def movement_lab(args):
    from app.analytics.physical_analysis import run

    run(str(args.database), str(args.output_dir))
    return 0


def scout_figures(args):
    from app.analytics.figures import generate_scouting_figures

    paths = generate_scouting_figures(
        args.data_dir,
        args.scouting_dir,
        args.output_dir,
        match_id=args.match_id,
        network_window=(tuple(args.network_window) if args.network_window else None),
    )
    print(json.dumps({"files": [str(path) for path in paths]}))
    return 0


def representation_lab(args):
    from app.analytics.representation.experiment import run
    from app.analytics.artifacts import archive_outputs
    from app.data_service.warehouse import Warehouse

    seeds = tuple(
        dict.fromkeys(
            int(value.strip()) for value in args.seeds.split(",") if value.strip()
        )
    )
    if not seeds:
        raise ValueError("至少指定一个随机种子")
    report = run(
        args.database,
        args.output_dir,
        target_league=args.target_league,
        train_cap=args.train_cap,
        epochs=args.epochs,
        seeds=seeds,
    )
    with Warehouse(str(args.database)) as db:
        archive_outputs(
            db,
            f"representation:{report['run_id']}",
            args.output_dir,
            "representation_learning",
            report["run_id"],
        )
    return 0


def analytics_lab(args):
    """Import only the selected experiment and archive its completed artifacts."""
    from app.analytics.artifacts import archive_outputs, file_digest
    from app.data_service.warehouse import Warehouse

    if args.command == "passing-lab":
        from app.analytics.passing_context import run

        report = run(args.data_dir, args.output_dir)
        filenames = report["artifacts"]
    elif args.command == "space-lab":
        from app.analytics.receiving_space import run

        report = run(args.data_dir, args.output_dir, samples=args.samples)
        filenames = ["report.json", *report["artifacts"].values()]
    elif args.command == "value-lab":
        from app.analytics.action_value import run

        seeds = tuple(dict.fromkeys(int(v.strip()) for v in args.seeds.split(",") if v.strip()))
        if not seeds:
            raise ValueError("至少指定一个随机种子")
        report = run(
            args.database, args.output_dir, target_league=args.target_league,
            train_cap=args.train_cap, epochs=args.epochs, seeds=seeds,
            horizon=args.horizon,
        )
    else:
        from app.analytics.buildup_roles import run

        report = run(args.database, args.output_dir, minimum_minutes=args.minimum_minutes)
        filenames = [
            "report.json", "provenance.json", "predictions.json", "model-weights.json",
            "understat-panel.json", "current-profiles.json", "wyscout-panels.json", "neighbors.json",
        ]
    if args.command == "value-lab":
        manifest_path = args.output_dir / "artifact-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest["run_id"] != report["run_id"]:
            raise ValueError("实验文件清单与本次运行不符")
        for name, digest in manifest["files"].items():
            relative = Path(name)
            file = args.output_dir / relative
            if (
                relative.is_absolute() or ".." in relative.parts
                or not file.resolve().is_relative_to(args.output_dir.resolve())
            ):
                raise ValueError("实验文件清单含无效路径")
            if file_digest(file) != digest:
                raise ValueError(f"实验文件校验失败: {name}")
        filenames = [*manifest["files"], "artifact-manifest.json"]
    with Warehouse(str(args.database)) as db:
        archive_outputs(
            db, f"{args.command}:{report['run_id']}", args.output_dir,
            args.command, report["run_id"], filenames=filenames,
        )
    print(json.dumps({"command": args.command, "run_id": report["run_id"], "output": str(args.output_dir)}, ensure_ascii=False))
    return 0
