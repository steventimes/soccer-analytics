import logging
import time
from app.data_service.fetch.fetcher import FootballDataClient
from app.data_service.db_session import get_db_service
from app.data_service.db.database.db_schema import Team

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def seed_players():
    client = FootballDataClient()
    
    with get_db_service() as service:
        try:
            logger.info("Querying database for teams...")
            teams = service.session.query(Team).all() 
            logger.info(f"Found {len(teams)} teams. Starting player fetch...")
            
            for i, team in enumerate(teams):
                logger.info(f"[{i+1}/{len(teams)}] Processing {team.name} (ID: {team.id})...")

                team_data = client.fetch_team_squad(team.id)
                
                if team_data and 'squad' in team_data:
                    service.teams.save_squad(team.id, team_data['squad'])
                else:
                    logger.warning(f"   -> No squad data found for {team.name}")

                time.sleep(0.5) 
                
        except KeyboardInterrupt:
            logger.warning("Process interrupted by user.")
        except Exception as e:
            logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    seed_players()