import logging
import time
from app.config import COMPETITIONS_MAP, SEASONS
from app.data_service.fetch.fetcher import FootballDataClient
from app.data_service.db_session import get_db_service

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def seed_matches():
    client = FootballDataClient()
    total_saved = 0
    
    with get_db_service() as service:
        logger.info("Starting Match Seed Process...")

        for code, comp_id in COMPETITIONS_MAP.items():
            logger.info(f"--- Processing {code} (ID: {comp_id}) ---")

            comp_details = client.fetch_competition_details(code)
            if comp_details:
                comp_details['id'] = comp_id 
                service.competitions.save_competition(comp_details)
            else:
                logger.error(f"Could not fetch details for {code}. Skipping...")
                continue

            season_data = client.fetch_multiple_seasons(code, SEASONS)
            
            all_matches = []
            for season, matches in season_data.items():
                if matches:
                    all_matches.extend(matches)

            if not all_matches:
                continue

            try:
                service.matches.save_bulk(all_matches)
                total_saved += len(all_matches)
                logger.info(f"Saved {len(all_matches)} matches for {code}.")
            except Exception as e:
                logger.error(f"Failed to save matches for {code}: {e}")

    logger.info(f"Seeding Complete. Total Matches Saved: {total_saved}")

if __name__ == "__main__":
    seed_matches()