import logging
import time
from app.config import COMPETITIONS_MAP, SEASONS
from app.data_service.fetch.fetcher import FootballDataClient
from app.data_service.db_session import get_db_service

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def seed_competitions():
    client = FootballDataClient()
    
    with get_db_service() as service:
        try:
            logger.info("Starting Context Seed (Standings & Scorers)...")
            
            for code, comp_id in COMPETITIONS_MAP.items():
                for season in SEASONS:
                    logger.info(f"Processing {code} - {season}...")
                    
                    standings = client.fetch_standings(code, season)
                    if standings and 'standings' in standings:
                        table = standings['standings'][0]['table']
                        service.competitions.save_standings(comp_id, season, table)
                    
                    scorers = client.fetch_top_scorers(code, season)
                    if scorers and 'scorers' in scorers:
                        service.competitions.save_top_scorers(comp_id, season, scorers['scorers'])

                    time.sleep(1) 
                    
        except KeyboardInterrupt:
            logger.warning("Interrupted.")
        except Exception as e:
            logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    seed_competitions()