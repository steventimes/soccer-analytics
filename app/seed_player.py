import logging
import time
from app.data_service.fetch.fetcher import fetch_team_squad
from app.data_service.db.database.db_save_operations import save_squad_db
from app.data_service.data_service_factory import session_local
from app.data_service.db.database.db_schema import Team

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def seed_players():
    session = session_local()
    
    try:
        logger.info("Querying database for teams...")
        teams = session.query(Team).all()
        logger.info(f"Found {len(teams)} teams. Starting player fetch...")
        
        for i, team in enumerate(teams):
            logger.info(f"[{i+1}/{len(teams)}] Processing {team.name} (ID: {team.id})...")

            team_data = fetch_team_squad(team.id)
            
            if team_data and 'squad' in team_data:
                save_squad_db(session, team.id, team_data['squad'])
            else:
                logger.warning(f"   -> No squad data found for {team.name}")

            time.sleep(1) 
            
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user.")
    finally:
        session.close()
        logger.info("Player seeding complete.")

if __name__ == "__main__":
    seed_players()