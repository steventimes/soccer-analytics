from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Sequence

from app.config import load_settings, parse_training_seasons, resolve_competitions
from app.ml.prediction import PredictionEngine, PredictionRun
from app.ml.training import ModelTrainer
from app.web.export_site import export_site_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_training_pipeline(
    *,
    competition_codes: str | None = None,
    seasons: Sequence[str] | None = None,
    trainer: Any | None = None,
) -> dict[str, Any]:
    settings = load_settings()
    competitions = resolve_competitions(competition_codes, settings)
    active_seasons = list(seasons or settings.training_seasons)
    active_trainer = trainer or ModelTrainer(model_dir=settings.model_dir)
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for code, competition_id in competitions.items():
        try:
            bundle = active_trainer.train_competition(
                code,
                competition_id,
                active_seasons,
            )
            result = {
                "competition": code,
                "training_rows": bundle.training_rows,
                "validation_accuracy": bundle.metrics.get("validation_accuracy"),
                "validation_log_loss": bundle.metrics.get("validation_log_loss"),
                "validation_ranked_probability_score": bundle.metrics.get(
                    "validation_ranked_probability_score"
                ),
                "baseline_accuracy": bundle.metrics.get("baseline_accuracy"),
                "majority_baseline_accuracy": bundle.metrics.get(
                    "majority_baseline_accuracy"
                ),
                "elo_baseline_accuracy": bundle.metrics.get("elo_baseline_accuracy"),
                "backtest_folds": bundle.metrics.get("backtest_folds"),
                "backtest_rows": bundle.metrics.get("backtest_rows"),
                "backtest_accuracy": bundle.metrics.get("backtest_accuracy"),
                "backtest_baseline_accuracy": bundle.metrics.get(
                    "backtest_baseline_accuracy"
                ),
                "backtest_majority_baseline_accuracy": bundle.metrics.get(
                    "backtest_majority_baseline_accuracy"
                ),
                "backtest_elo_baseline_accuracy": bundle.metrics.get(
                    "backtest_elo_baseline_accuracy"
                ),
                "backtest_log_loss": bundle.metrics.get("backtest_log_loss"),
                "backtest_brier_score": bundle.metrics.get(
                    "backtest_brier_score"
                ),
                "backtest_calibration_error": bundle.metrics.get(
                    "backtest_calibration_error"
                ),
                "backtest_ranked_probability_score": bundle.metrics.get(
                    "backtest_ranked_probability_score"
                ),
                "backtest_classwise_calibration_error": bundle.metrics.get(
                    "backtest_classwise_calibration_error"
                ),
                "backtest_max_class_calibration_error": bundle.metrics.get(
                    "backtest_max_class_calibration_error"
                ),
                "backtest_sharpness": bundle.metrics.get("backtest_sharpness"),
                "backtest_worst_fold_accuracy_lift": bundle.metrics.get(
                    "backtest_worst_fold_accuracy_lift"
                ),
                "backtest_accuracy_lift_stddev": bundle.metrics.get(
                    "backtest_accuracy_lift_stddev"
                ),
            }
            results.append(result)
            logger.info("%s 训练完成：%s 场", code, bundle.training_rows)
        except Exception as exc:
            # 单联赛失败不抹掉其他已训练资产，但必须进入结构化结果。
            errors.append(
                {
                    "competition": code,
                    "code": type(exc).__name__,
                    "message": str(exc),
                }
            )
            logger.error("%s 训练失败：%s", code, exc)

    return {
        "seasons": active_seasons,
        "trained_count": len(results),
        "failed_count": len(errors),
        "results": results,
        "errors": errors,
    }


def run_predictions_pipeline(
    days: int | None = None,
    *,
    competition_codes: str | None = None,
    engine: PredictionEngine | None = None,
) -> PredictionRun:
    settings = load_settings()
    competitions = resolve_competitions(competition_codes, settings)
    active_engine = engine or PredictionEngine(models_dir=settings.model_dir)
    prediction_days = settings.prediction_days if days is None else days
    run = active_engine.predict_all(competitions, days=prediction_days)
    logger.info(
        "预测完成：%s 场，加载 %s/%s 个模型，错误 %s 个",
        len(run.predictions),
        run.models_loaded,
        run.competitions_requested,
        len(run.errors),
    )
    return run


def run_export_site_pipeline(
    days: int | None = None,
    *,
    competition_codes: str | None = None,
    refresh_errors: Sequence[dict[str, str]] = (),
) -> dict[str, Path]:
    settings = load_settings()
    export_days = settings.site_export_days if days is None else days
    export_kwargs: dict[str, Any] = {
        "days": export_days,
        "models_dir": settings.model_dir,
        "output_dir": settings.site_data_dir,
    }
    if competition_codes is not None:
        export_kwargs["competitions"] = resolve_competitions(competition_codes, settings)
    if refresh_errors:
        export_kwargs["refresh_errors"] = list(refresh_errors)
    outputs = export_site_data(
        **export_kwargs,
    )
    for label, path in outputs.items():
        logger.info("导出 %s -> %s", label, path)
    return outputs


def run_full_pipeline(
    days: int | None = None,
    *,
    competition_codes: str | None = None,
    seasons: Sequence[str] | None = None,
    trainer: Any | None = None,
) -> dict[str, Any]:
    training = run_training_pipeline(
        competition_codes=competition_codes,
        seasons=seasons,
        trainer=trainer,
    )
    outputs = run_export_site_pipeline(
        days=days,
        competition_codes=competition_codes,
        refresh_errors=training.get("errors", []),
    )
    return {"training": training, "outputs": outputs}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须大于 0")
    return parsed


def _add_competition_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--competitions",
        help="逗号分隔的联赛代码，例如 PL,PD,SA。",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Soccer Analytics：采集、训练、预测与静态站点导出。"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="训练并保存所选联赛模型。")
    _add_competition_argument(train_parser)
    train_parser.add_argument(
        "--seasons",
        type=parse_training_seasons,
        help="逗号分隔的赛季起始年，例如 2022,2023,2024。",
    )

    predict_parser = subparsers.add_parser("predict", help="预测未来赛程。")
    _add_competition_argument(predict_parser)
    predict_parser.add_argument("--days", type=_positive_int)

    export_parser = subparsers.add_parser(
        "export-site",
        help="预测并原子导出静态仪表盘数据。",
    )
    _add_competition_argument(export_parser)
    export_parser.add_argument("--days", type=_positive_int, default=None)

    full_parser = subparsers.add_parser(
        "all",
        help="训练所选联赛并导出仪表盘。",
    )
    _add_competition_argument(full_parser)
    full_parser.add_argument("--seasons", type=parse_training_seasons)
    full_parser.add_argument("--days", type=_positive_int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        result = run_training_pipeline(
            competition_codes=args.competitions,
            seasons=args.seasons,
        )
        return 0 if result["trained_count"] > 0 else 1
    if args.command == "predict":
        run = run_predictions_pipeline(
            days=args.days,
            competition_codes=args.competitions,
        )
        print(json.dumps(run.to_dict(), ensure_ascii=False, allow_nan=False))
        return 0 if run.models_loaded > 0 else 1
    if args.command == "export-site":
        run_export_site_pipeline(
            days=args.days,
            competition_codes=args.competitions,
        )
        return 0
    if args.command == "all":
        result = run_full_pipeline(
            days=args.days,
            competition_codes=args.competitions,
            seasons=args.seasons,
        )
        return 0 if result["training"]["trained_count"] > 0 else 1
    raise AssertionError("未知命令")


if __name__ == "__main__":
    raise SystemExit(main())
