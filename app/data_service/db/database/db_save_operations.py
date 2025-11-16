from sqlalchemy.orm import Session as SQLSession
from typing import List
import pandas as pd
from datetime import datetime
from app.data_service.db.database.db_schema import (
    Team, Player, Match, Competition, MatchGoal, 
    TeamStanding, TopScorer, get_match_result_label
)
from app.data_service.db.database.db_get_operations import get_team_db

def save_team_db(session: SQLSession, team_Data: dict) -> Team:
    team = get_team_db(session, team_Data['id'])
    
    if team:
        for key, value in team_Data.items():
            if hasattr(team, key):
                setattr(team, key, value)
    else:
        team = Team(**team_Data)
        session.add(team)
        
    session.commit()
    session.refresh(team)
    return team


def save_players_db(session: SQLSession, team_id: int, players_df: pd.DataFrame):
    team = get_team_db(session, team_id)
    if not team:
        team = Team(id=team_id, name=f"Team {team_id}")
        session.add(team)
        session.commit()
    
    for _, row in players_df.iterrows():
        player_id = row.get('id')
        
        player = session.query(Player).filter(Player.id == player_id).first()
        
        dob = pd.to_datetime(row['dateOfBirth'], errors='coerce') if 'dateOfBirth' in row.index else None
        
        if player:
            player.name = row.get('name') # type: ignore
            player.position = row.get('position') # type: ignore
            player.nationality = row.get('nationality') # type: ignore
            if pd.notna(dob):
                player.date_of_birth = dob # type: ignore
            player.team_id = team_id # type: ignore
            player.shirtNumber = row.get('shirtNumber') # type: ignore
            player.marketValue = row.get('marketValue') # type: ignore
        else:
            player = Player(
                id=player_id,
                name=row.get('name'),
                position=row.get('position'),
                nationality=row.get('nationality'),
                date_of_birth=dob,
                team_id=team_id,
                shirtNumber = row.get('shirtNumber'),
                marketValue = row.get('marketValue')
            )
            session.add(player)
    
    session.commit()
    print(f"Saved {len(players_df)} players for team {team_id}")


def save_competition_standings_db(session: SQLSession, competition_code: str, standings_df: pd.DataFrame):
    """
    Save competition standings to database
    Creates/updates competition and team records
    """
    competition = session.query(Competition).filter(
        Competition.code == competition_code
    ).first()
    
    if not competition:
        competition = Competition(
            code=competition_code,
            name=competition_code,
            type="LEAGUE"
        )
        session.add(competition)
        session.commit()
    
    for _, row in standings_df.iterrows():
        team_data = row.get('table', {})
        team = row.get("team", {})
        if isinstance(team, dict):
            team_id = team.get('id')
            team_name = team.get('name')
            tla = team.get('tla')
        else:
            continue
        
        if team_id:
            team = get_team_db(session, team_id)
            
            if team:
                team.name = team_name # type: ignore
                team.competition_id = competition.id
                team.tla = tla # type: ignore
                team.playedGames = team_data.get("playedGame")
                team.won = team_data.get("won")
                team.draw = team_data.get("draw")
                team.lost = team_data.get("lost")
                team.points = team_data.get("points")
                team.goalsFor = team_data.get("goalsFor")
                team.goalsAgainst = team_data.get("goalsAgainst")
                team.goalDifference = team_data.get("goalDifference")
            else:
                team = Team(
                    id=team_id,
                    name=team_name,
                    competition_id=competition.id,
                    playedGames = team_data.get("playedGame"),
                    won = team_data.get("won"),
                    draw = team_data.get("draw"),
                    lost = team_data.get("lost"),
                    points = team_data.get("points"),
                    goalsFor = team_data.get("goalsFor"),
                    goalsAgainst = team_data.get("goalsAgainst"),
                    goalDifference = team_data.get("goalDifference"),
                )
                session.add(team)
    
    session.commit()
    print(f"Saved standings for competition {competition_code}")


def save_match_db(session: SQLSession, match_data: dict) -> Match:
    """
    Save or update a match in the database
    
    Args:
        match_data: Dictionary containing match information from API
        
    Returns:
        Match object
    """
    match_id = match_data['id']
    match = session.query(Match).filter(Match.id == match_id).first()
    
    utc_date = match_data.get('utcDate')
    if isinstance(utc_date, str):
        utc_date = datetime.fromisoformat(utc_date.replace('Z', '+00:00'))
    
    score = match_data.get('score', {})
    full_time = score.get('fullTime', {})
    half_time = score.get('halfTime', {})
    
    odds = match_data.get('odds', {})
    
    match_info = {
        'id': match_id,
        'competition_id': match_data['competition']['id'],
        'season_id': match_data['season']['id'],
        'season_year': match_data['season'].get('startDate', '')[:4],
        'utc_date': utc_date,
        'status': match_data['status'],
        'matchday': match_data.get('matchday'),
        'stage': match_data.get('stage'),
        'venue': match_data.get('venue'),
        'attendance': match_data.get('attendance'),
        'home_team_id': match_data['homeTeam']['id'],
        'away_team_id': match_data['awayTeam']['id'],
        'home_formation': match_data['homeTeam'].get('formation'),
        'away_formation': match_data['awayTeam'].get('formation'),
        'home_league_rank': match_data['homeTeam'].get('leagueRank'),
        'away_league_rank': match_data['awayTeam'].get('leagueRank'),
        'score_home': full_time.get('home'),
        'score_away': full_time.get('away'),
        'halftime_home': half_time.get('home'),
        'halftime_away': half_time.get('away'),
        'winner': score.get('winner'),
        'odds_home_win': odds.get('homeWin'),
        'odds_draw': odds.get('draw'),
        'odds_away_win': odds.get('awayWin'),
        'last_updated': datetime.fromisoformat(match_data['lastUpdated'].replace('Z', '+00:00')) if 'lastUpdated' in match_data else None
    }
    
    if match:
        for key, value in match_info.items():
            if hasattr(match, key):
                setattr(match, key, value)
    else:
        match = Match(**match_info)
        session.add(match)
    
    session.commit()
    session.refresh(match)
    
    if 'goals' in match_data and match_data['goals']:
        save_match_goals_db(session, match_id, match_data['goals'])
    
    return match

def save_matches_bulk_db(session: SQLSession, matches_list) -> int:
    """
    Save multiple matches efficiently
    
    Args:
        matches_list: List of match dictionaries from API
        
    Returns:
        Number of matches saved
    """
    saved_count = 0
    
    for match_data in matches_list:
        try:
            if match_data['status'] == 'FINISHED' and match_data['score']['fullTime']['home'] is not None:
                save_match_db(session, match_data)
                saved_count += 1
        except Exception as e:
            print(f"Error saving match {match_data.get('id')}: {e}")
            continue
    
    return saved_count

def save_match_goals_db(session: SQLSession, match_id: int, goals_list: List[dict]):
    """Save goals for a match"""
    session.query(MatchGoal).filter(MatchGoal.match_id == match_id).delete()
    
    for goal_data in goals_list:
        goal = MatchGoal(
            match_id=match_id,
            minute=goal_data.get('minute'),
            injury_time=goal_data.get('injuryTime'),
            type=goal_data.get('type'),
            team_id=goal_data.get('team', {}).get('id'),
            team_name=goal_data.get('team', {}).get('name'),
            scorer_id=goal_data.get('scorer', {}).get('id'),
            scorer_name=goal_data.get('scorer', {}).get('name'),
            assist_id=goal_data.get('assist', {}).get('id') if goal_data.get('assist') else None,
            assist_name=goal_data.get('assist', {}).get('name') if goal_data.get('assist') else None,
            home_score=goal_data.get('score', {}).get('home'),
            away_score=goal_data.get('score', {}).get('away')
        )
        session.add(goal)
    
    session.commit()


def save_team_standing_db(session: SQLSession, standing_data: dict, competition_id: int, season_id: int, season_year: str):
    """
    Save team standing for a specific matchday
    This allows tracking team performance over time
    """
    team_data = standing_data.get('team', {})
    team_id = team_data.get('id')
    
    existing = session.query(TeamStanding).filter(
        TeamStanding.competition_id == competition_id,
        TeamStanding.season_id == season_id,
        TeamStanding.team_id == team_id,
        TeamStanding.type == 'TOTAL'
    ).first()
    
    standing_info = {
        'competition_id': competition_id,
        'season_id': season_id,
        'season_year': season_year,
        'matchday': standing_data.get('playedGames'),
        'stage': 'REGULAR_SEASON',
        'type': 'TOTAL',
        'team_id': team_id,
        'team_name': team_data.get('name'),
        'position': standing_data.get('position'),
        'played_games': standing_data.get('playedGames'),
        'won': standing_data.get('won'),
        'draw': standing_data.get('draw'),
        'lost': standing_data.get('lost'),
        'points': standing_data.get('points'),
        'goals_for': standing_data.get('goalsFor'),
        'goals_against': standing_data.get('goalsAgainst'),
        'goal_difference': standing_data.get('goalDifference'),
        'form': standing_data.get('form'),
    }
    
    if existing:
        for key, value in standing_info.items():
            if hasattr(existing, key):
                setattr(existing, key, value)
    else:
        standing = TeamStanding(**standing_info)
        session.add(standing)
    
    session.commit()

def save_top_scorers_db(session: SQLSession, scorers_list, competition_id: int, season_id: int, season_year: str):
    """Save top scorers for a competition/season"""
    for scorer_data in scorers_list:
        player_data = scorer_data.get('player', {})
        team_data = scorer_data.get('team', {})
        
        player_id = player_data.get('id')
        
        existing = session.query(TopScorer).filter(
            TopScorer.competition_id == competition_id,
            TopScorer.season_id == season_id,
            TopScorer.player_id == player_id
        ).first()
        
        scorer_info = {
            'competition_id': competition_id,
            'season_id': season_id,
            'season_year': season_year,
            'player_id': player_id,
            'player_name': player_data.get('name'),
            'team_id': team_data.get('id'),
            'team_name': team_data.get('name'),
            'goals': scorer_data.get('goals', 0),
            'assists': scorer_data.get('assists'),
            'penalties': scorer_data.get('penalties')
        }
        
        if existing:
            for key, value in scorer_info.items():
                if hasattr(existing, key):
                    setattr(existing, key, value)
        else:
            scorer = TopScorer(**scorer_info)
            session.add(scorer)
    
    session.commit()
