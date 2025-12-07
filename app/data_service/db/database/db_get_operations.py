from sqlalchemy.orm import Session as SQLSession
from typing import List, Optional, Dict
import pandas as pd
from app.data_service.data_type import type_db_data
from datetime import datetime, timedelta

from app.data_service.db.database.db_schema import (
    Team, Player, Match, Competition, MatchGoal, 
    TeamStanding, TopScorer, get_match_result_label
)

def get_team_db(session: SQLSession, team_id: int) -> Optional[Team]:
    '''get team by id in db'''
    return session.query(Team).filter(Team.id == team_id).first()


def get_players_by_team_db(session: SQLSession, team_id: int) -> List[Player]:
    """Get all players for a specific team"""
    return session.query(Player).filter(Player.team_id == team_id).all()


def get_competition_standings_db(session: SQLSession, competition_code: str) -> Optional[List[Dict]]:
    """
    Get current standings for a competition from database
    """
    competition = session.query(Competition).filter(
        Competition.code == competition_code
    ).first()
    
    if not competition or not competition.teams:
        return None
    
    standings = []
    # Implementation depends on how you store standings relationships, 
    # but strictly for this request we focus on Matches below.
    return standings

def get_matches_by_competition_db(session: SQLSession, competition_id: int, season: str) -> List[Match]:
    """
    Fetch all finished matches for a specific competition and season from the DB.
    """
    return session.query(Match).filter(
        Match.competition_id == competition_id,
        Match.season_year == season, # Using season_year based on your schema conventions
        Match.status == 'FINISHED'
    ).all()

def get_team_recent_form_db(session: SQLSession, team_id: int, match_date: datetime, limit: int = 5) -> Dict:
    """
    Get recent form stats for feature engineering
    """
    matches = session.query(Match).filter(
        ((Match.home_team_id == team_id) | (Match.away_team_id == team_id)),
        Match.utc_date < match_date,
        Match.status == 'FINISHED'
    ).order_by(Match.utc_date.desc()).limit(limit).all()
    
    stats = {'wins': 0, 'draws': 0, 'losses': 0, 'goals_scored': 0, 'goals_conceded': 0, 'total_games': len(matches)}
    
    for m in matches:
        is_home = m.home_team_id == team_id
        goals_for = m.score_home if is_home else m.score_away
        goals_against = m.score_away if is_home else m.score_home
        
        # Handle None values
        goals_for = goals_for if goals_for is not None else 0
        goals_against = goals_against if goals_against is not None else 0
        
        stats['goals_scored'] += goals_for
        stats['goals_conceded'] += goals_against
        
        if m.winner == 'DRAW':
            stats['draws'] += 1
        elif (is_home and m.winner == 'HOME_TEAM') or (not is_home and m.winner == 'AWAY_TEAM'):
            stats['wins'] += 1
        else:
            stats['losses'] += 1
            
    return stats

def get_head_to_head_db(session: SQLSession, team1_id: int, team2_id: int, limit: int = 10) -> Dict:
    """
    Get H2H stats
    """
    matches = session.query(Match).filter(
        ((Match.home_team_id == team1_id) & (Match.away_team_id == team2_id)) |
        ((Match.home_team_id == team2_id) & (Match.away_team_id == team1_id)),
        Match.status == 'FINISHED'
    ).order_by(Match.utc_date.desc()).limit(limit).all()
    
    return {}