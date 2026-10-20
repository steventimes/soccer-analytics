"""Data collection, history and source import commands."""

import json

from app.config import load_settings


def collect_open_data(args):
    from app.data_service.archive import REPOSITORIES, collect_repository
    from app.data_service.bulk_history import collect_all_csv
    from app.file_io import write_json_atomic as _atomic_write_json

    selected = list(
        dict.fromkeys(
            value.strip() for value in args.datasets.split(",") if value.strip()
        )
    )
    if not selected or any(
        value not in {*REPOSITORIES, "football-csv", "bodypose"} for value in selected
    ):
        raise ValueError("未知公开数据集")
    report = {"datasets": {}, "errors": []}
    for name in selected:
        try:
            if name == "football-csv":
                inventory = collect_all_csv(args.data_dir / name, refresh=args.refresh,
                                            max_age_hours=getattr(args, 'max_age_hours', None),
                                            active_max_age_hours=getattr(args, 'active_max_age_hours', None))
            elif name == "bodypose":
                from app.data_service.bodypose import collect_bodypose

                inventory = collect_bodypose(
                    args.data_dir / "skillcorner", refresh=args.refresh
                )
            else:
                inventory = collect_repository(
                    name, args.data_dir / name, refresh=args.refresh
                )
            report["datasets"][name] = {
                "files": len(inventory["files"]),
                "bytes": inventory["bytes"],
                "complete": inventory["complete"],
                "manifest": (
                    str(
                        args.data_dir
                        / "skillcorner"
                        / "external-bodypose"
                        / "manifest.json"
                    )
                    if name == "bodypose"
                    else str(args.data_dir / name / "manifest.json")
                ),
            }
            report["errors"].extend(inventory.get("errors", []))
        except Exception as exc:
            report["errors"].append({"dataset": name, "message": str(exc)})
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(args.output, report)
    else:
        print(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2))
    return 0 if report["datasets"] and not report["errors"] else 1


def scout_bodypose(args):
    from app.data_service.bodypose import analyze_bodypose

    report = analyze_bodypose(args.data_dir / "skillcorner", args.output_dir)
    print(
        json.dumps(
            {
                "matches": len(report["matches"]),
                "report": str(args.output_dir / "bodypose-quality.json"),
            }
        )
    )
    return 0


def scout_events(args):
    from app.analytics.scouting import analyze_statsbomb

    report = analyze_statsbomb(
        args.data_dir / "statsbomb",
        args.output_dir,
        maximum_matches=args.maximum_matches,
        competition_id=args.competition_id,
        season_id=args.season_id,
    )
    print(
        json.dumps(
            {
                "matches": report["matches_analyzed"],
                "events": report["events"],
                "shots": report["shots"],
                "errors": report["errors"],
                "report": str(args.output_dir / "report.json"),
            },
            ensure_ascii=False,
        )
    )
    return 0 if not report["errors"] else 1


def scout_tracking(args):
    from app.analytics.movement import analyze_movement

    report = analyze_movement(args.data_dir, args.output_dir)
    print(
        json.dumps(
            {
                "matches": len(report["matches"]),
                "errors": report["errors"],
                "report": str(args.output_dir / "report.json"),
            },
            ensure_ascii=False,
        )
    )
    return 0 if not report["errors"] else 1


def history(args):
    from app.data_service.history import (
        PUBLIC_SOURCES,
        collect_history,
        digest,
        load_history,
    )
    from app.data_service.fetch.public import LEAGUES
    from app.file_io import write_json_atomic as _atomic_write_json

    settings = load_settings()
    sources = [value.strip() for value in args.sources.split(",") if value.strip()]
    if args.source and args.source != "auto":
        sources = [args.source]
    if not sources or any(source not in PUBLIC_SOURCES for source in sources):
        raise ValueError("历史快照仅支持 football-csv 和 openfootball")
    competitions = (
        list(
            dict.fromkeys(
                value.strip().upper()
                for value in args.competitions.split(",")
                if value.strip()
            )
        )
        if args.competitions
        else list(LEAGUES)
    )
    if any(code not in LEAGUES for code in competitions):
        raise ValueError("请求的历史联赛不在公开源登记表中")
    if args.command == "collect-history":
        latest = int(max(settings.training_seasons))
        seasons = args.seasons or [str(year) for year in range(latest - 10, latest)]
        report = collect_history(
            args.cache_dir, sources, competitions, seasons, refresh=args.refresh,
            max_age_hours=getattr(args, 'max_age_hours', None),
            active_max_age_hours=getattr(args, 'active_max_age_hours', None),
        )
    else:
        from app.ml.seasonal import run_season_matrix

        collections, manifest = load_history(args.cache_dir)
        reference_collections = {
            key: value for key, value in collections.items() if key[1] in competitions
        }
        collections = {
            key: value
            for key, value in collections.items()
            if key[0] in sources and key[1] in competitions
        }
        checkpoint = args.output.with_suffix(".partial.json") if args.output else None
        if checkpoint:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
        report = run_season_matrix(
            collections,
            test_seasons=args.test_seasons,
            windows=[v.strip() for v in args.windows.split(",")],
            models=[v.strip() for v in args.models.split(",")],
            seed=args.seed,
            pooled=not args.no_pool,
            reference_collections=reference_collections,
            progress=(
                (lambda partial: _atomic_write_json(checkpoint, partial))
                if checkpoint
                else None
            ),
        )
        report["manifest_sha256"] = digest(manifest)
        report["inventory"] = [
            {key: value for key, value in entry.items() if key != "path"}
            for entry in manifest["collections"]
            if entry["source"] in sources and entry["competition"] in competitions
        ]
        report["collection_errors"] = [
            entry
            for entry in manifest.get("errors", [])
            if entry["source"] in sources and entry["competition"] in competitions
        ]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(args.output, report)
    else:
        print(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2))
    successful = bool(report.get("collections") or report.get("results"))
    return (
        0
        if successful
        and not report.get("errors")
        and not report.get("collection_errors")
        else 1
    )
