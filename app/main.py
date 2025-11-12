import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from app.db.data_type import type_db_data
from app.db.database.db_schema import Base
from app.db.data_service import DataService

#test if docker has correctly set up the env
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL is None:
    print("DATABASE_URL environment variable is not set. Using a default SQLite database.")
    DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(DATABASE_URL)
session_local = sessionmaker(bind=engine)
Base.metadata.create_all(engine)

def main():
    session = session_local()
    
    try:
        service = DataService(session)
        
        print("=" * 60)
        print("Example 1: Get Team Players (Man United - ID: 66)")
        print("=" * 60)
        
        # First call - will hit API
        print("\nFirst call:")
        players_df = service.data_get(type_db_data.TEAM_PLAYER,66)
        print(f"\nRetrieved {len(players_df)} players")
        print(players_df[['name', 'position', 'nationality']].head())
        
        # Second call - will hit cache
        print("\n" + "-" * 60)
        print("Second call (should hit cache):")
        players_df = service.data_get(type_db_data.TEAM_PLAYER,66)
        print(f"Retrieved {len(players_df)} players")
        
        print("\n" + "=" * 60)
        print("Example 2: Get Competition Standings (Premier League)")
        print("=" * 60)
        
        # First call - will hit API
        print("\nFirst call:")
        standings_df = service.data_get(type_db_data.COMPETITION_STANDING,66)
        print(f"\nRetrieved {len(standings_df)} teams")
        print(standings_df[['position', 'team', 'points']].head())
        
        # Second call - will hit cache
        print("\n" + "-" * 60)
        print("Second call (should hit cache):")
        standings_df = service.data_get(type_db_data.COMPETITION_STANDING,66)
        print(f"Retrieved {len(standings_df)} teams")
        
        print("\n" + "=" * 60)
        print("Example 3: Cache Invalidation & DB Retrieval")
        print("=" * 60)
        
        # Invalidate cache
        service.invalidate_cache("team:66:*")
        
        # Third call - will hit DB (cache was cleared)
        print("\nThird call (should hit DB after cache invalidation):")
        players_df = service.data_get(type_db_data.TEAM_PLAYER,66)
        print(f"Retrieved {len(players_df)} players")
        
    finally:
        session.close()


if __name__ == "__main__":
    main()