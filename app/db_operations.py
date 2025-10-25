'''set and retrieve data in db'''
from sqlalchemy import select
from sqlalchemy.orm import Session as SQLSession
from typing import List, Optional, Dict, cast
from datetime import datetime
import pandas as pd

from db_schema import Team, Player, Match, Competition

def get_team_db(session: SQLSession, team_id: int) -> Optional[Team]:
    '''get team by id in db'''
    return session.query(Team).filter(Team.id == team_id).first()

def save_team_db(session: SQLSession, team_Data: dict) -> Team:
    '''save or update team in db'''
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

def get_players_by_team_db(session: SQLSession, team_id: int) -> List[Player]:
    """Get all players for a specific team"""
    return session.query(Player).filter(Player.team_id == team_id).all()


def save_players_db(session: SQLSession, team_id: int, players_df: pd.DataFrame):
    """
    Save players from DataFrame to database
    Links them to the specified team
    """
    # Ensure team exists
    team = get_team_db(session, team_id)
    if not team:
        # Create basic team record if it doesn't exist
        team = Team(id=team_id, name=f"Team {team_id}")
        session.add(team)
        session.commit()
    
    for _, row in players_df.iterrows():
        player_id = row.get('id')
        
        # Check if player already exists
        player = session.query(Player).filter(Player.id == player_id).first()
        
        # Parse date of birth if present
        dob = None
        if 'dateOfBirth' in row and pd.notna(row['dateOfBirth']):
            try:
                dob = datetime.fromisoformat(row['dateOfBirth'].replace('Z', '+00:00')).date()
            except:
                dob = None
        
        if player:
            player.name = row.get('name') # type: ignore casting issue
            player.position = row.get('position') # type: ignore
            player.nationality = row.get('nationality') # type: ignore
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
    
    # Build standings from teams
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


def save_competition_standings_db(session: SQLSession, competition_code: str, standings_df: pd.DataFrame):
    """
    Save competition standings to database
    Creates/updates competition and team records
    """
    # Get or create competition
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
    
    # Save teams from standings
    for _, row in standings_df.iterrows():
        team_data = row.get('table', {})
        team = row.get("team", {})
        if isinstance(team, dict):
            team_id = team.get('id')
            team_name = team.get('name')
            tla = team.get('tla')
        else:
            # If team is already extracted
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


def get_or_create_competition(session: SQLSession, competition_data: Dict) -> Competition:
    """Get existing competition or create new one"""
    comp_id = competition_data.get('id')
    competition = session.query(Competition).filter(Competition.id == comp_id).first()
    
    if not competition:
        competition = Competition(**competition_data)
        session.add(competition)
        session.commit()
    
    return competition