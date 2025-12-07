import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

from app.data_service.db.database.db_schema import Base
from app.data_service.db.data_service import DataService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL is None:
    print("DATABASE_URL environment variable is not set. Using a default SQLite database.")
    DATABASE_URL = "sqlite:///./test.db"
    
engine = create_engine(DATABASE_URL)
session_local = sessionmaker(bind=engine)
Base.metadata.create_all(engine)

def db_test():
    session = session_local()
    
    try:
        service = DataService(session)
        
        print("\n" + "=" * 60)
        print("TEST 1: Fetch Team via Repository")
        print("=" * 60)

        team = service.teams.get_by_id(66)
        if team:
            print(f"Success! Found Team: {team.name} ({team.short_name})")
            print(f"Venue: {team.venue}")
        else:
            print("Team 66 not found in DB (Run seed_player.py first?)")

        print("\n" + "=" * 60)
        print("TEST 2: Fetch Matches via Repository")
        print("=" * 60)
        
        matches = service.matches.get_by_competition(2021, '2023')
        print(f"Found {len(matches)} matches for PL 2023")
        
        if matches:
            m = matches[0]
            print(f"Sample Match: {m.home_team.name} vs {m.away_team.name} ({m.score_home}-{m.score_away})")

    except Exception as e:
        logger.error(f"Test Failed: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    db_test()