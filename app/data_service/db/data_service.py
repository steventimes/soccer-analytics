from sqlalchemy.orm import Session as SQLSession
import pandas as pd
from typing import List, Dict, Optional
from app.data_service.data_type import type_db_data, competitions
from app.data_service.db.cache.cache_management import get_cache, set_cache, clear_all_pattern, TTL
from datetime import datetime
from app.data_service.fetch.fetcher import api_get
from app.data_service.db.database.db_save_operations import (
    save_team_db,
    save_players_db,
    save_matches_bulk_db,
    save_top_scorers_db,
    save_competition_standings_db
)
from app.data_service.db.database.db_get_operations import (
    get_players_by_team_db,
    get_competition_standings_db,
    get_top_scorers_db,
    get_team_recent_form_db,
    get_head_to_head_db,
    get_matches_db,
    get_matches_for_ml_db,
    get_team_db,
    get_team_standing_at_matchday_db
)

class DataService:
    "cache -> db -> api"
    def __init__(self, session: SQLSession) -> None:
        self.session = session
        
    def data_get(self, type, data) -> pd.DataFrame:
        match type:
            case type_db_data.TEAM_PLAYER:
                return self.get_team_players(data)
            case type_db_data.COMPETITION_STANDING:
                return self.get_competition_standings(data)
            case type_db_data.SINGLE_TEAM:
                return self.get_single_team(data)
            case type_db_data.COMPETITION_MATCHES:
                if isinstance(data, dict):
                    return self.get_competition_matches(
                        data.get('competition_code', 'PL'),
                        data.get('season')
                    )
                return pd.DataFrame()
            case type_db_data.TOP_SCORERS:
                if isinstance(data, dict):
                    return self.get_top_scorers(
                        data.get('competition_code', 'PL'),
                        data.get('season')
                    )
                return pd.DataFrame()
            case type_db_data.TEAM_RECENT:
                if isinstance(data, dict):
                    return pd.DataFrame(
                        self.get_team_form(
                            data.get('team_id'), # type: ignore
                            data.get('num_matches', 5),
                            data.get("match_date")
                    ))
                return pd.DataFrame()
            case type_db_data.HEAD_TO_HEAD:
                if isinstance(data, dict):
                    return pd.DataFrame(
                        self.get_head_to_head(
                            data.get('home_id'), # type: ignore
                            data.get('away_id'), # type: ignore
                            data.get('limit', 10)
                    ))
                return pd.DataFrame()
            case type_db_data.STANDING_MATCHDAY:
                if isinstance(data, dict):
                    return pd.DataFrame(
                        get_team_standing_at_matchday_db(
                            self.session,
                            data.get('team_id'), # type: ignore
                            data.get('competition_id'), # type: ignore
                            data.get('season_year') # type: ignore
                        )
                    )
        return pd.DataFrame()
    
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
        df = api_get(type_db_data.TEAM_PLAYER ,team_id)
        
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
        
        print(f"Cache & DB miss for {competition_code}, fetching from API...")
        df = api_get(type_db_data.COMPETITION_STANDING, competition_code)
        
        if df is not None and not df.empty:
            save_competition_standings_db(self.session, competition_code, df)
            set_cache(cache_key, df.to_dict('records'), TTL)
            print(f"Save {competition_code} standings to DB and cache")
        
        return df
    def get_single_team(self, team_id: int) -> pd.DataFrame:
        """
        Get single team information - Cache -> DB -> API
        
        Args:
            team_id: Team ID
            
        Returns:
            DataFrame with single row containing team info
        """
        cache_key = f"team:{team_id}:info"
        
        cached_data = get_cache(cache_key)
        if cached_data:
            print(f"✓ Cache hit for team {team_id} info")
            return pd.DataFrame(cached_data)
        
        team = get_team_db(self.session, team_id)
        if team:
            print(f"✓ DB hit for team {team_id} info")
            team_dict = {
                'id': team.id,
                'name': team.name,
                'shortName': team.short_name,
                'tla': team.tla,
                'founded': team.founded,
                'crest': team.crest,
                'venue': team.venue,
                'address': team.address,
                'website': team.website,
                'clubColors': team.club_colors,
                'area_name': team.area_name,
                'coach_name': team.coach_name,
                'coach_nationality': team.coach_nationality,
                'market_value': team.market_value,

                'playedGames': team.playedGames,
                'won': team.won,
                'draw': team.draw,
                'lost': team.lost,
                'points': team.points,
                'goalsFor': team.goalsFor,
                'goalsAgainst': team.goalsAgainst,
                'goalDifference': team.goalDifference
            }
            df = pd.DataFrame([team_dict])
            set_cache(cache_key, df.to_dict('records'), TTL)
            return df
        
        print(f"Cache & DB miss for team {team_id}, fetching from API...")
        df = api_get(type_db_data.SINGLE_TEAM, team_id)
        
        if df is not None and not df.empty:
            team_data = df.to_dict('records')[0] if not df.empty else {}
            if team_data:
                saved_team = save_team_db(self.session, team_data)
                print(f"Saved team {team_id} to DB")
            
            set_cache(cache_key, df.to_dict('records'), TTL)
            print(f"Cached team {team_id} info")
        
        return df
    
    def get_competition_matches(
        self,
        competition_code: str = "PL",
        season: Optional[str] = None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Get competition matches - Cache -> DB -> API
        
        Args:
            competition_code: Competition code (e.g., 'PL')
            season: Season year (e.g., '2023'), None for all
            force_refresh: Skip cache and fetch from API
            
        Returns:
            DataFrame of matches
        """
        cache_key = f"matches:{competition_code}:{season or 'all'}"
        
        if not force_refresh:
            cached_data = get_cache(cache_key)
            if cached_data:
                print(f"Cache hit for {competition_code} {season} matches")
                return pd.DataFrame(cached_data)
        
        comp_data = api_get(type_db_data.COMPETITION_STANDING, competition_code)
        if not comp_data.empty:
            matches = get_matches_db(
                self.session,
                competition_id=competitions.get(competition_code),
                season_year=season,
                status='FINISHED'
            )
            
            if matches:
                print(f"DB hit for {competition_code} {season} matches: {len(matches)} found")
                df = pd.DataFrame([{
                    'id': m.id,
                    'competition_id': m.competition_id,
                    'season_year': m.season_year,
                    'match_date': m.utc_date.isoformat() if m.utc_date else None, # type: ignore
                    'home_team': m.home_team.name if m.home_team else None,
                    'away_team': m.away_team.name if m.away_team else None,
                    'home_score': m.score_home,
                    'away_score': m.score_away,
                    'winner': m.winner,
                    'status': m.status
                } for m in matches])
                
                set_cache(cache_key, df.to_dict('records'), TTL)
                return df
        
        print(f"Cache & DB miss for {competition_code} {season}, fetching from API...")
        
        if not season:
            from datetime import datetime
            season = str(datetime.now().year)
        
        matches_list = api_get(type_db_data.COMPETITION_MATCHES, (competition_code, season, 'FINISHED'))
        
        if not matches_list.empty:
            saved = save_matches_bulk_db(self.session, matches_list)
            print(f"Saved {saved} matches to DB")
            
            df = pd.DataFrame(matches_list)
            set_cache(cache_key, df.to_dict('records'), TTL)
            
            return df
        
        return pd.DataFrame()
    
    def get_top_scorers(
        self,
        competition_code: str = "PL",
        season: Optional[str] = None,
        limit: int = 10
    ) -> pd.DataFrame:

        if not season:
            from datetime import datetime
            season = str(datetime.now().year)
        
        cache_key = f"scorers:{competition_code}:{season}:top{limit}"
        
        cached_data = get_cache(cache_key)
        if cached_data:
            print(f"Cache hit for {competition_code} {season} top scorers")
            return pd.DataFrame(cached_data)

        scorers = get_top_scorers_db(self.session, competitions.get(competition_code), season, limit)
        
        if scorers:
            print(f"DB hit for {competition_code} {season} top scorers")
            df = pd.DataFrame(scorers)
            set_cache(cache_key, df.to_dict('records'), TTL)
            return df
        
        print(f"Cache & DB miss for top scorers, fetching from API...")
        scorers_list = api_get(type_db_data.TOP_SCORERS, (competition_code, season, limit))
        
        if not scorers_list.empty:
            save_top_scorers_db(self.session, scorers_list, 2021, int(season), season)
            
            df = pd.DataFrame(scorers_list)
            set_cache(cache_key, df.to_dict('records'), TTL)
            print(f"Saved {len(scorers_list)} top scorers to DB and cache")
            
            return df
        
        return pd.DataFrame()
    
    def get_matches_for_ml(
        self,
        competition_code: str = "PL",
        min_matches: int = 100
    ) -> pd.DataFrame:
        cache_key = f"ml:matches:{competition_code}"
        
        cached_data = get_cache(cache_key)
        if cached_data:
            df = pd.DataFrame(cached_data)
            if len(df) >= min_matches:
                print(f"Cache hit for ML dataset: {len(df)} matches")
                return df
        
        df = get_matches_for_ml_db(self.session, competition_id=None, min_matches=min_matches)
        
        if not df.empty and len(df) >= min_matches:
            print(f"DB hit for ML dataset: {len(df)} matches")
            set_cache(cache_key, df.to_dict('records'), TTL * 2) 
            return df
        
        print(f"Not enough matches in DB ({len(df)}/{min_matches})")
        print(f"Run the match fetcher to populate the database")
        
        return df
    
    def get_team_form(self, team_id: int, num_matches: int = 5, match_date: Optional[datetime] = None) -> Dict:
        """
        Get recent form for a team
        
        Args:
            team_id: Team ID
            num_matches: Number of recent matches to analyze
            
        Returns:
            Dictionary with form statistics
        """
        cache_key = f"team:{team_id}:form:last{num_matches}"
        
        cached_data = get_cache(cache_key)
        if cached_data:
            print(f"Cache hit for team {team_id} form")
            return cached_data
        
        form = get_team_recent_form_db(self.session, team_id, num_matches, match_date)
        
        if form:
            set_cache(cache_key, form, TTL)
            return form
        
        return {'wins': 0, 'draws': 0, 'losses': 0, 'goals_scored': 0, 'goals_conceded': 0}
    
    def get_head_to_head(self, team1_id: int, team2_id: int, limit: int = 10) -> Dict:
        """
        Get head-to-head record between two teams
        
        Args:
            team1_id: First team ID
            team2_id: Second team ID
            limit: Number of recent matches
            
        Returns:
            Dictionary with H2H statistics
        """
        cache_key = f"h2h:{min(team1_id, team2_id)}:{max(team1_id, team2_id)}:last{limit}"
        
        cached_data = get_cache(cache_key)
        if cached_data:
            print(f"Cache hit for H2H: {team1_id} vs {team2_id}")
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
        
    
    # right now is for test purpose, might call it when shut down the whole app
    def invalidate_cache(self, pattern: str):
        """Invalidate cache key pattern"""
        
        keys = clear_all_pattern(pattern)
        if keys:
            print(f"Invalidated {keys} cache entries matching '{pattern}'")
        else:
            print(f"No cache entries found matching '{pattern}'")