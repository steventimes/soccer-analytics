from sqlalchemy.orm import Session as SQLSession
from typing import Optional, List, Dict
import pandas as pd
from datetime import datetime
from db_schema import Team, Player, Competition, Match
from cache_management import get_cache, set_cache, clear_all_pattern, TTL
from fetcher import get_team_players as fetch_team_players_api
from fetcher import get_competition_standings as fetch_standings_api
from db_operations import (
    get_team_db, 
    save_team_db,
    get_players_by_team_db,
    save_players_db,
    get_competition_standings_db,
    save_competition_standings_db
)

class DataService:
    "cache -> db -> api"
    def __init__(self, session: SQLSession) -> None:
        self.session = session
        
    def get_team_players(self, team_id: int) -> pd.DataFrame:
        cache_key = f"team:{team_id}:players"
        
        cache_data = get_cache(cache_key)
        
        if cache_data:
            print(f"cache hit for team {team_id} players")
            return pd.DataFrame(cache_data)
        
        players = get_players_by_team_db(self.session, team_id)
        if players:
            print(f"db hit for team {team_id} players")
            df = pd.DataFrame([{
                'id': p.id,
                'name': p.name,
                'position': p.position,
                'dateOfBirth': p.date_of_birth.isoformat() if bool(p.date_of_birth) else None,
                'nationality': p.nationality,
                'shirtNumber': p.shirtNumber,
                'marketValue': p.marketValue
            } for p in players])
            
            set_cache(cache_key, df.to_dict("records"), TTL)
            return df
        
        print(f"db and cache not hit for team {team_id}, fetch from api")
        df = fetch_team_players_api(team_id)
        
        if df is not None and not df.empty:
            save_players_db(self.session, team_id, df)
            set_cache(cache_key, df.to_dict('records'), TTL)
            print(f"save team {team_id} players to cache and db")
            
        return df
    
    def get_competition_standings(self, competition_code: str = "PL") -> pd.DataFrame:
        """
        Get competition standings
        """
        cache_key = f"competition:{competition_code}:standings"
        
        cached_data = get_cache(cache_key)
        if cached_data:
            print(f"Cache hit for {competition_code} standings")
            return pd.DataFrame(cached_data)
        
        standings = get_competition_standings_db(self.session, competition_code)
        if standings:
            print(f"DB hit for {competition_code} standings")
            df = pd.DataFrame(standings)
            set_cache(cache_key, df.to_dict('records'), TTL)
            return df
        
        print(f"✗ Cache & DB miss for {competition_code}, fetching from API...")
        df = fetch_standings_api(competition_code)
        
        if df is not None and not df.empty:
            save_competition_standings_db(self.session, competition_code, df)
            set_cache(cache_key, df.to_dict('records'), TTL)
            print(f"Save {competition_code} standings to DB and cache")
        
        return df
    
    def invalidate_cache(self, pattern: str):
        """Invalidate cache key pattern"""
        from cache_management import redis_client
        
        keys = clear_all_pattern(pattern)
        if keys:
            print(f"Invalidated {keys} cache entries matching '{pattern}'")
        else:
            print(f"No cache entries found matching '{pattern}'")