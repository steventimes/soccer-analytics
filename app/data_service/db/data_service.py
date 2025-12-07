from sqlalchemy.orm import Session as SQLSession
import pandas as pd
from typing import List, Dict, Optional, Any
from app.data_service.data_type import type_db_data
from app.data_service.db.cache.cache_management import get_cache, set_cache, clear_all_pattern, TTL
from app.data_service.db.database.db_get_operations import (
    get_players_by_team_db,
    get_competition_standings_db,
    get_team_recent_form_db,
    get_head_to_head_db,
    get_matches_by_competition_db
)

class DataService:
    def __init__(self, session: SQLSession) -> None:
        self.session = session
        
    def data_get(self, type: type_db_data, data: Any):
        """
        Main entry point for data retrieval. 
        Routes requests to DB operations and handles formatting.
        """
        match type:
            case type_db_data.TEAM_PLAYER:
                return pd.DataFrame()

            case type_db_data.COMPETITION_MATCHES:
                comp_id = data.get('competition_id')
                season = data.get('season')
                return self.get_competition_matches(comp_id, season)

            case type_db_data.TEAM_RECENT:
                return self.get_team_recent_matches(data)
                
            case _:
                return None

    def get_competition_matches(self, competition_id: int, season: str) -> List[Dict]:
        """
        Fetch matches from DB and serialize to Dict for ML service
        """
        matches = get_matches_by_competition_db(self.session, competition_id, str(season))
        
        result = []
        for m in matches:
            match_dict = {
                'id': m.id,
                'utc_date': m.utc_date,
                'status': m.status,
                'matchday': m.matchday,
                'stage': m.stage,
                'season': m.season_year,
                'home_team': {
                    'id': m.home_team_id, 
                    'name': m.home_team.name if m.home_team else f"Team {m.home_team_id}"
                },
                'away_team': {
                    'id': m.away_team_id, 
                    'name': m.away_team.name if m.away_team else f"Team {m.away_team_id}"
                },
                'score': {
                    'winner': m.winner,
                    'fullTime': {
                        'home': m.score_home,
                        'away': m.score_away
                    },
                    'halfTime': {
                        'home': m.halftime_home,
                        'away': m.halftime_away
                    }
                }
            }
            result.append(match_dict)
            
        return result

    def get_team_recent_matches(self, data: Dict) -> pd.DataFrame:
        """
        Used by feature.py to calculate form
        """
        team_id = data.get('team_id')
        match_date = data.get('match_date')
        num = data.get('num_matches', 5)
        
        stats = get_team_recent_form_db(self.session, team_id, match_date, num)
        
        return pd.DataFrame([stats])