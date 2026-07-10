import argparse
import logging

from app.config import load_settings, resolve_competitions
from app.ml.predict_upcoming import UpcomingPredictor
from app.ml.simulate_betting import BettingSimulator
from app.ml.training import ModelTrainer
from app.web.export_site import export_site_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_training_pipeline(*, competition_codes: str | None = None, seasons: list[str] | None = None, tune: bool = True):
    logger.info("Starting Training Pipeline...")
    trainer = ModelTrainer()
    settings = load_settings()
    competitions = resolve_competitions(competition_codes, settings)
    active_seasons = seasons or settings.training_seasons

    success_count = 0
    fail_count = 0

    for code, comp_id in competitions.items():
        logger.info("\n%s", "=" * 40)
        logger.info("Training Model for: %s (ID: %s)", code, comp_id)
        logger.info("%s", "=" * 40)

        try:
            df = trainer.prepare_dataset(comp_id, active_seasons)

            if df.empty:
                logger.warning("SKIPPING %s - No data found.", code)
                fail_count += 1
                continue

            model = trainer.train(df, tune=tune)

            if model:
                safe_name = f"{code.lower()}_model"
                trainer.save_model(safe_name)
                success_count += 1
        except Exception as exc:
            logger.error("Training failed for %s: %s", code, exc)
            fail_count += 1

    logger.info(
        "Pipeline Complete. Success: %s, Failed: %s",
        success_count,
        fail_count,
    )


def run_predictions_pipeline(days: int = 3):
    logger.info("Running Weekend Predictions...")
    predictor = UpcomingPredictor()
    predictor.predict(days=days)
    logger.info("Done.")


def run_betting_simulation_pipeline():
    logger.info("Starting Betting Simulation...")
    simulator = BettingSimulator()
    simulator.run_simulation()
    logger.info("Simulation Complete.")


def run_export_site_pipeline(days: int | None = None):
    settings = load_settings()
    export_days = days or settings.site_export_days
    outputs = export_site_data(days=export_days)
    for label, path in outputs.items():
        logger.info("Exported %s -> %s", label, path)


def run_full_pipeline(days: int = 3, *, competition_codes: str | None = None, seasons: list[str] | None = None, tune: bool = True, export_site: bool = False):
    run_training_pipeline(competition_codes=competition_codes, seasons=seasons, tune=tune)
    run_predictions_pipeline(days=days)
    run_betting_simulation_pipeline()
    if export_site:
        run_export_site_pipeline(days=days)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Soccer analytics execution pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train", help="Train models for configured competitions.")
    train_parser = subparsers.choices["train"]
    train_parser.add_argument(
        "--competitions",
        help="Comma-separated competition codes (for example: PL,PD,SA).",
    )
    train_parser.add_argument(
        "--seasons",
        help="Comma-separated seasons to train on (for example: 2022,2023,2024).",
    )
    train_parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Disable model tuning for faster iteration runs.",
    )

    predict_parser = subparsers.add_parser(
        "predict",
        help="Predict upcoming matches using available models.",
    )
    predict_parser.add_argument(
        "--days",
        type=int,
        default=3,
        help="Number of days ahead to predict.",
    )

    subparsers.add_parser(
        "simulate",
        help="Run the betting simulation with trained models.",
    )

    export_parser = subparsers.add_parser(
        "export-site",
        help="Export static site data artifacts for the docs frontend.",
    )
    export_parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Number of days ahead to include in the exported prediction dataset.",
    )

    full_parser = subparsers.add_parser(
        "all",
        help="Run training, predictions, and betting simulation.",
    )
    full_parser.add_argument(
        "--days",
        type=int,
        default=3,
        help="Number of days ahead to predict.",
    )
    full_parser.add_argument(
        "--competitions",
        help="Comma-separated competition codes (for example: PL,PD,SA).",
    )
    full_parser.add_argument(
        "--seasons",
        help="Comma-separated seasons to train on (for example: 2022,2023,2024).",
    )
    full_parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Disable model tuning for faster iteration runs.",
    )
    full_parser.add_argument(
        "--export-site",
        action="store_true",
        help="Also export docs/data payloads after the pipeline completes.",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        seasons = [part.strip() for part in args.seasons.split(",") if part.strip()] if args.seasons else None
        run_training_pipeline(
            competition_codes=args.competitions,
            seasons=seasons,
            tune=not args.no_tune,
        )
        return
    if args.command == "predict":
        run_predictions_pipeline(days=args.days)
        return
    if args.command == "simulate":
        run_betting_simulation_pipeline()
        return
    if args.command == "export-site":
        run_export_site_pipeline(days=args.days)
        return
    if args.command == "all":
        seasons = [part.strip() for part in args.seasons.split(",") if part.strip()] if args.seasons else None
        run_full_pipeline(
            days=args.days,
            competition_codes=args.competitions,
            seasons=seasons,
            tune=not args.no_tune,
            export_site=args.export_site,
        )
        return

    parser.error("Unknown command")


if __name__ == "__main__":
    main()
