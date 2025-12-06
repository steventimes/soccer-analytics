from sqlalchemy.orm import Session as SQLSession
import pandas as pd
from typing import List, Dict, Optional
from app.data_service.data_type import type_db_data
from app.data_service.db.cache.cache_management import get_cache, set_cache, clear_all_pattern, TTL
from app.data_service.db.database.db_get_operations import (
    get_players_by_team_db,
    get_competition_standings_db,
    get_team_recent_form_db,
    get_head_to_head_db
)
from app.data_service.db.database.db_schema import Match, Competition

class DataService:
    def __init__(self, session: SQLSession) -> None:
        self.session = session
        
    def data_get(self, type, data):
        match type:
            case type_db_data.TEAM_PLAYER:
                cache_key = f"team:{data}:players"
                cached_data = get_cache(cache_key)
                if cached_data:
                    return pd.DataFrame(cached_data)
                
                players = get_players_by_team_db(self.session, data)
                if not players:
                    return pd.DataFrame()
                
                result = []
                for p in players:
                    result.append({
                        'id': p.id,
                        'name': p.name,
                        'position': p.position,
                        'nationality': p.nationality,
                        'date_of_birth': str(p.date_of_birth) if p.date_of_birth else None
                    })
                
                if result:
                    set_cache(cache_key, result, TTL)
                    
                return pd.DataFrame(result)
            
            case type_db_data.COMPETITION_STANDING:
                cache_key = f"comp:{data}:standings"
                cached_data = get_cache(cache_key)
                if cached_data:
                    return pd.DataFrame(cached_data)
                    
                standings = get_competition_standings_db(self.session, data)
                
                if standings:
                    set_cache(cache_key, standings, TTL)
                    return pd.DataFrame(standings)
                    
                return pd.DataFrame()
            
            case type_db_data.TEAM_RECENT:
                return get_team_recent_form_db(
                    self.session, 
                    data['team_id'], 
                    data.get('num_matches', 5),
                    data.get('match_date')
                )

            case type_db_data.COMPETITION_MATCHES:
                return self.get_competition_matches(data['competition_id'], data['season'])

            case _:
                return None

    def get_competition_matches(self, competition_id: int, season: str) -> List[Dict]:
        matches = self.session.query(Match).filter(
            Match.competition_id == competition_id,
            Match.season_year == str(season)
        ).all()

        if matches:
            return [self._match_to_dict(m) for m in matches]

        comp = self.session.query(Competition).filter(Competition.id == competition_id).first()
        if not comp:
            return []
        
        from app.data_service.fetch.fetcher import fetch_multiple_seasons
        from app.data_service.db.database.db_save_operations import save_matches_bulk_db
        
        fetched_data = fetch_multiple_seasons(comp.code, [str(season)])
        
        if fetched_data:
            save_matches_bulk_db(self.session, fetched_data.get(str(season), []))
            
            matches = self.session.query(Match).filter(
                Match.competition_id == competition_id,
                Match.season_year == str(season)
            ).all()
            return [self._match_to_dict(m) for m in matches]

        return []

    def _match_to_dict(self, match) -> Dict:
        return {
            'id': match.id,
            'utc_date': match.utc_date,
            'status': match.status,
            'matchday': match.matchday,
            'stage': match.stage,
            'season': match.season_year,
            'home_team': {'id': match.home_team_id, 'name': match.home_team.name if match.home_team else f"Team {match.home_team_id}"},
            'away_team': {'id': match.away_team_id, 'name': match.away_team.name if match.away_team else f"Team {match.away_team_id}"},
            'score': {
                'winner': match.winner,
                'fullTime': {'home': match.score_home, 'away': match.score_away},
                'halfTime': {'home': match.halftime_home, 'away': match.halftime_away}
            }
        }

    def get_head_to_head(self, team1_id: int, team2_id: int, limit: int = 10) -> Dict:
        cache_key = f"h2h:{min(team1_id, team2_id)}:{max(team1_id, team2_id)}:last{limit}"
        
        cached_data = get_cache(cache_key)
        if cached_data:
            return cached_data
        
        h2h = get_head_to_head_db(self.session, team1_id, team2_id, limit)
        
        if h2h:
            set_cache(cache_key, h2h, TTL)
            return h2h
        
        return {
            'total_matches': 0,
            'team1_wins': 0,
            'draws': 0,
            'team2_wins': 0,
            'team1_goals': 0,
            'team2_goals': 0
        }

    def invalidate_cache(self, pattern: str):
        clear_all_pattern(pattern)