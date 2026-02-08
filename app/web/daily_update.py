import logging

from app.pipeline import run_training_pipeline
from app.seeds.seed_competitions import seed_competitions
from app.seeds.seed_matches import seed_matches
from app.web.export_site import export_site_data


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_daily_update(days: int = 1) -> None:
    logger.info("Seeding latest competitions...")
    seed_competitions()

    logger.info("Seeding latest matches...")
    seed_matches()

    logger.info("Training models...")
    run_training_pipeline()

    logger.info("Exporting static site data...")
    export_site_data(days=days)

    logger.info("Daily update completed.")


if __name__ == "__main__":
    run_daily_update()