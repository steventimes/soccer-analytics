import logging
import time
from app.data_service.fetch.fetcher import fetch_standings, fetch_top_scorers
from app.data_service.db.database.db_save_operations import save_standings_db, save_scorers_db
from app.data_service.data_service_factory import session_local
from app.seed_data import COMPETITIONS, SEASONS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def seed_context():
    session = session_local()
    
    try:
        logger.info("Starting Context Seed (Standings & Scorers)...")
        
        for code, comp_id in COMPETITIONS.items():
            for season in SEASONS:
                logger.info(f"Processing {code} - {season}...")
                
                standings = fetch_standings(code, season)
                if standings and 'standings' in standings:
                    save_standings_db(session, comp_id, season, standings['standings'])
                
                scorers = fetch_top_scorers(code, season)
                if scorers and 'scorers' in scorers:
                    save_scorers_db(session, comp_id, season, scorers['scorers'])

                time.sleep(6) 
                
    except KeyboardInterrupt:
        logger.warning("Interrupted.")
    finally:
        session.close()
        logger.info("Context seeding complete.")

if __name__ == "__main__":
    seed_context()