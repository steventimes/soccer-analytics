import argparse
import logging

from app.config import COMPETITIONS_MAP, TRAINING_SEASONS
from app.ml.predict_upcoming import UpcomingPredictor
from app.ml.simulate_betting import BettingSimulator
from app.ml.training import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_training_pipeline():
    logger.info("Starting Training Pipeline...")
    trainer = ModelTrainer()

    success_count = 0
    fail_count = 0

    for code, comp_id in COMPETITIONS_MAP.items():
        logger.info("\n%s", "=" * 40)
        logger.info("Training Model for: %s (ID: %s)", code, comp_id)
        logger.info("%s", "=" * 40)

        try:
            df = trainer.prepare_dataset(comp_id, TRAINING_SEASONS)

            if df.empty:
                logger.warning("SKIPPING %s - No data found.", code)
                fail_count += 1
                continue

            model = trainer.train(df, tune=True)

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


def run_full_pipeline(days: int = 3):
    run_training_pipeline()
    run_predictions_pipeline(days=days)
    run_betting_simulation_pipeline()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Soccer analytics execution pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train", help="Train models for configured competitions.")

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

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        run_training_pipeline()
        return
    if args.command == "predict":
        run_predictions_pipeline(days=args.days)
        return
    if args.command == "simulate":
        run_betting_simulation_pipeline()
        return
    if args.command == "all":
        run_full_pipeline(days=args.days)
        return

    parser.error("Unknown command")


if __name__ == "__main__":
    main()