import logging
import time
from app.data_service.fetch.fetcher import fetch_multiple_seasons
from app.data_service.db.database.db_save_operations import save_matches_bulk_db
from app.data_service.data_service_factory import session_local

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

competitions = ["CL","PL", "PPL", "DED", "BL1", "FL1", "SA", "PD", "BSA", "ELC", "EC", "WC"]
competitions_id = [2001, 2021, 2017, 2003, 2002, 2015, 2019, 2014, 2013, 2016, 2018, 2000]
COMPETITIONS = dict(zip(competitions, competitions_id))

SEASONS = [str(x) for x in range(2021, 2025)] 

def seed_database():
    session = session_local()
    total_saved = 0
    
    try:
        logger.info("Starting Database Seed Process...")
        logger.info(f"Targeting {len(COMPETITIONS)} competitions over {len(SEASONS)} seasons.")

        for code, comp_id in COMPETITIONS.items():
            logger.info(f"\n{'='*40}")
            logger.info(f"Processing {code} (ID: {comp_id})")
            logger.info(f"{'='*40}")

            season_data = fetch_multiple_seasons(code, SEASONS)

            all_matches = []
            for season, matches in season_data.items():
                if matches:
                    all_matches.extend(matches)
                    logger.info(f"Queued {len(matches)} matches from season {season}")

            if not all_matches:
                logger.warning(f"No data found for {code}. Moving next.")
                continue

            # Save to Database
            try:
                logger.info(f"Saving {len(all_matches)} matches to DB...")
                save_matches_bulk_db(session, all_matches)
                total_saved += len(all_matches)
                logger.info(f"Success! {code} is up to date.")
            except Exception as e:
                logger.error(f"Failed to save {code}: {e}")
                session.rollback()

    except KeyboardInterrupt:
        logger.warning("Process interrupted by user.")
    finally:
        session.close()
        logger.info(f"\nJob Complete. Total matches processed: {total_saved}")

if __name__ == "__main__":
    seed_database()