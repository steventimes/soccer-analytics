import logging
import time
from app.data_service.data_service_factory import session_local
from app.data_service.fetch.fetcher import fetch_multiple_seasons, fetch_competition_details
from app.data_service.db.database.db_save_operations import save_matches_bulk_db, save_competition_db

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

        for code, comp_id in COMPETITIONS.items():
            logger.info(f"\n{'='*40}")
            logger.info(f"Processing {code} (ID: {comp_id})")
            logger.info(f"{'='*40}")

            comp_details = fetch_competition_details(code)
            if comp_details:
                save_competition_db(session, comp_details)
            else:
                logger.error(f"Could not fetch details for {code}. Skipping...")
                continue

            season_data = fetch_multiple_seasons(code, SEASONS)
            
            all_matches = []
            for season, matches in season_data.items():
                if matches:
                    all_matches.extend(matches)

            if not all_matches:
                continue

            try:
                save_matches_bulk_db(session, all_matches)
                total_saved += len(all_matches)
            except Exception as e:
                logger.error(f"  -> Failed to save matches for {code}: {e}")
                session.rollback()

    except KeyboardInterrupt:
        logger.warning("Process interrupted.")
    finally:
        session.close()
        logger.info(f"Job Complete. Total matches: {total_saved}")

if __name__ == "__main__":
    seed_database()