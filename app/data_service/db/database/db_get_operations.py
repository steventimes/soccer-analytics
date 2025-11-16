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
    Returns list of dicts with team standings
    """
    competition = session.query(Competition).filter(
        Competition.code == competition_code
    ).first()
    
    if not competition or not competition.teams:
        return None
    
    standings = []
    for team in competition.teams:
        standings.append({
            'team': {'name': team.name, 'id': team.id},
            'playedGames': team.playedGames,
            'won': team.won,
            'draw': team.draw,
            'lost': team.lost,
            'goalsFor': team.goalsFor,
            'goalsAgainst': team.goalsAgainst,
            'goalDifference': team.goalDifference
        })
    
    return standings if standings else None

def get_or_create_competition(session: SQLSession, competition_data: Dict) -> Competition:
    """Get existing competition or create new one"""
    comp_id = competition_data.get('id')
    competition = session.query(Competition).filter(Competition.id == comp_id).first()
    
    if not competition:
        competition = Competition(**competition_data)
        session.add(competition)
        session.commit()
    
    return competition

def get_matches_db(
    session: SQLSession,
    competition_id: Optional[int] = None,
    season_year: Optional[str] = None,
    status: str = 'FINISHED',
    limit: Optional[int] = None
) -> List[Match]:
    """
    Args:
        competition_id: Filter by competition
        season_year: Filter by season
        status: Match status (default: 'FINISHED')
        limit: Maximum number of matches to return
    """
    query = session.query(Match).filter(Match.status == status)
    
    if competition_id:
        query = query.filter(Match.competition_id == competition_id)
    
    if season_year:
        query = query.filter(Match.season_year == season_year)
    
    query = query.order_by(Match.utc_date.desc())
    
    if limit:
        query = query.limit(limit)
    
    return query.all()

def get_matches_for_ml_db(
    session: SQLSession,
    competition_id: Optional[int] = None,
    min_matches: int = 100
) -> pd.DataFrame:
    """
    Get matches formatted for ML training
    
    Returns DataFrame with:
    - match_id
    - home_team_id, away_team_id
    - home_score, away_score
    - result ('H', 'D', 'A')
    - match_date
    - Additional features
    """
    matches = get_matches_db(session, competition_id=competition_id, status='FINISHED')
    
    if len(matches) < min_matches:
        print(f"Warning: Only {len(matches)} matches available (minimum: {min_matches})")
    
    data = []
    for match in matches:
        data.append({
            'match_id': match.id,
            'competition_id': match.competition_id,
            'season_year': match.season_year,
            'match_date': match.utc_date,
            'matchday': match.matchday,
            'home_team_id': match.home_team_id,
            'away_team_id': match.away_team_id,
            'home_score': match.score_home,
            'away_score': match.score_away,
            'result': get_match_result_label(match),
            'winner': match.winner,
            'home_league_rank': match.home_league_rank,
            'away_league_rank': match.away_league_rank,
        })
    
    return pd.DataFrame(data)

def get_team_standing_at_matchday_db(
    session: SQLSession,
    team_id: int,
    competition_id: int,
    season_year: str,
    matchday: Optional[int] = None
) -> Optional[Dict]:
    """
    Get team's standing at a specific matchday (for ML features)
    If matchday not specified, returns most recent
    """
    query = session.query(TeamStanding).filter(
        TeamStanding.team_id == team_id,
        TeamStanding.competition_id == competition_id,
        TeamStanding.season_year == season_year
    )
    
    if matchday:
        query = query.filter(TeamStanding.matchday <= matchday)
    
    standing = query.order_by(TeamStanding.matchday.desc()).first()
    
    if standing:
        return {
            'position': standing.position,
            'points': standing.points,
            'played': standing.played_games,
            'won': standing.won,
            'draw': standing.draw,
            'lost': standing.lost,
            'goal_difference': standing.goal_difference,
            'goals_for': standing.goals_for,
            'goals_against': standing.goals_against,
            'form': standing.form
        }
    
    return None

def get_top_scorers_db(session: SQLSession, competition_id: int, season_year: str, limit: int = 10) -> List[Dict]:
    """Get top scorers for a competition/season"""
    scorers = session.query(TopScorer).filter(
        TopScorer.competition_id == competition_id,
        TopScorer.season_year == season_year
    ).order_by(TopScorer.goals.desc()).limit(limit).all()
    
    return [{
        'player_name': s.player_name,
        'team_name': s.team_name,
        'goals': s.goals,
        'assists': s.assists,
        'penalties': s.penalties
    } for s in scorers]
    
def get_team_recent_form_db(
    session: SQLSession,
    team_id: int,
    num_matches: int = 5,
    before_date: Optional[datetime] = None
) -> Dict:
    
    query = session.query(Match).filter(
        (Match.home_team_id == team_id) | (Match.away_team_id == team_id),
        Match.status == 'FINISHED'
    )

    if before_date:
        query = query.filter(Match.utc_date < before_date)

    recent_matches = query.order_by(Match.utc_date.desc()).limit(num_matches).all()

    if not recent_matches:
        return {'wins': 0, 'draws': 0, 'losses': 0, 'goals_scored': 0, 'goals_conceded': 0}

    wins = draws = losses = 0
    goals_scored = goals_conceded = 0

    for match in recent_matches:
        is_home = match.home_team_id == team_id
    
        if is_home: # type: ignore
            goals_scored += match.score_home or 0
            goals_conceded += match.score_away or 0
        
            if match.winner == 'HOME_TEAM': # type: ignore
                wins += 1
            elif match.winner == 'DRAW': # type: ignore
                draws += 1
            else:
                losses += 1
        else:
            goals_scored += match.score_away or 0
            goals_conceded += match.score_home or 0
            
            if match.winner == 'AWAY_TEAM': # type: ignore
                wins += 1
            elif match.winner == 'DRAW': # type: ignore
                draws += 1
            else:
                losses += 1
    
    return {
        'wins': wins,
        'draws': draws,
        'losses': losses,
        'goals_scored': goals_scored,
        'goals_conceded': goals_conceded,
        'matches_played': len(recent_matches),
        'win_rate': wins / len(recent_matches) if recent_matches else 0
    }
        
def get_head_to_head_db(session: SQLSession, team1_id: int, team2_id: int, limit: int = 10) -> Dict:
    """
    Get head-to-head record between two teams
    Important ML feature
    """
    matches = session.query(Match).filter(
        ((Match.home_team_id == team1_id) & (Match.away_team_id == team2_id)) |
        ((Match.home_team_id == team2_id) & (Match.away_team_id == team1_id)),
        Match.status == 'FINISHED'
    ).order_by(Match.utc_date.desc()).limit(limit).all()
    
    team1_wins = team1_draws = team1_losses = 0
    team1_goals = team2_goals = 0
    
    for match in matches:
        if match.home_team_id == team1_id: # type: ignore
            team1_goals += match.score_home or 0
            team2_goals += match.score_away or 0
            
            if match.winner == 'HOME_TEAM': # type: ignore
                team1_wins += 1
            elif match.winner == 'DRAW': # type: ignore
                team1_draws += 1
            else:
                team1_losses += 1
        else:
            team1_goals += match.score_away or 0
            team2_goals += match.score_home or 0
            
            if match.winner == 'AWAY_TEAM': # type: ignore
                team1_wins += 1
            elif match.winner == 'DRAW': # type: ignore
                team1_draws += 1
            else:
                team1_losses += 1
    
    return {
        'total_matches': len(matches),
        'team1_wins': team1_wins,
        'draws': team1_draws,
        'team2_wins': team1_losses,
        'team1_goals': team1_goals,
        'team2_goals': team2_goals
    }