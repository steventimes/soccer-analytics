from sqlalchemy import Column, Index, Integer, String, Date, ForeignKey, Float, DateTime, Text, DECIMAL
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime

Base = declarative_base()

class Competition(Base):
    __tablename__ = "competitions"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    area_name = Column(String(100))
    area_id = Column(Integer)
    area_code = Column(String(10))
    code = Column(String(10), nullable=False)
    type = Column(String(20))
    emblem = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    teams = relationship("Team", back_populates="competition")
    matches = relationship("Match", back_populates="competition")
    standings = relationship("TeamStanding", back_populates="competition")
    top_scorers = relationship("TopScorer", back_populates="competition")


class Team(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True)
    name = Column(String(150), nullable=False)
    tla = Column(String(10))
    short_name = Column(String(100))
    founded = Column(Integer)
    crest = Column(Text)
    address = Column(Text)
    website = Column(Text)
    club_colors = Column(String(100))
    venue = Column(String(150))
    area_id = Column(Integer)
    area_name = Column(String(100))
    market_value = Column(Integer)
    
    playedGames = Column(Integer)
    won = Column(Integer)
    draw = Column(Integer)
    lost = Column(Integer)
    points = Column(Integer)
    goalsFor = Column(Integer)
    goalsAgainst = Column(Integer)
    goalDifference = Column(Integer)
    
    # Coach info
    coach_id = Column(Integer)
    coach_name = Column(String(150))
    coach_nationality = Column(String(100))
    
    competition_id = Column(Integer, ForeignKey("competitions.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    competition = relationship("Competition", back_populates="teams")
    players = relationship("Player", back_populates="team")
    home_matches = relationship("Match", foreign_keys="Match.home_team_id", back_populates="home_team")
    away_matches = relationship("Match", foreign_keys="Match.away_team_id", back_populates="away_team")
    standings = relationship("TeamStanding", back_populates="team")
    top_scorers = relationship("TopScorer", back_populates="team")



class Player(Base):
    __tablename__ = "players"

    id = Column(Integer, primary_key=True)
    name = Column(String(150), nullable=False)
    first_name = Column(String(100))
    last_name = Column(String(100))
    position = Column(String(50))
    date_of_birth = Column(Date)
    nationality = Column(String(100))
    shirtNumber = Column(Integer)
    marketValue = Column(Integer)
    
    # Contract info
    contract_start = Column(Date)
    contract_until = Column(Date)
    
    team_id = Column(Integer, ForeignKey("teams.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    team = relationship("Team", back_populates="players")
    stats = relationship("PlayerStat", back_populates="player")

    goals = relationship("MatchGoal", foreign_keys="[MatchGoal.scorer_id]", back_populates="scorer")
    assists = relationship("MatchGoal", foreign_keys="[MatchGoal.assist_id]", back_populates="assister")
    top_scorer_records = relationship("TopScorer", back_populates="player")

class Match(Base):
    __tablename__ = "matches"

    id = Column(Integer, primary_key=True)
    
    competition_id = Column(Integer, ForeignKey("competitions.id"), nullable=False)
    season_id = Column(Integer, nullable=False)
    season_year = Column(String(10))
    
    # Match Details
    utc_date = Column(DateTime, nullable=False)
    status = Column(String(20), nullable=False)  # FINISHED, SCHEDULED, LIVE...
    matchday = Column(Integer)
    stage = Column(String(50))
    venue = Column(String(150))
    attendance = Column(Integer)
    
    # Teams
    home_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    away_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    home_formation = Column(String(20))
    away_formation = Column(String(20))
    home_league_rank = Column(Integer)
    away_league_rank = Column(Integer)
    
    # Scores
    score_home = Column(Integer) 
    score_away = Column(Integer)
    halftime_home = Column(Integer)
    halftime_away = Column(Integer)
    winner = Column(String(20))  # 'HOME_TEAM', 'AWAY_TEAM', 'DRAW'
    
    # Odds
    odds_home_win = Column(DECIMAL(10, 2))
    odds_draw = Column(DECIMAL(10, 2))
    odds_away_win = Column(DECIMAL(10, 2))
    
    # Metadata
    last_updated = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    competition = relationship("Competition", back_populates="matches")
    home_team = relationship("Team", foreign_keys=[home_team_id], back_populates="home_matches")
    away_team = relationship("Team", foreign_keys=[away_team_id], back_populates="away_matches")
    player_stats = relationship("PlayerStat", back_populates="match")
    goals = relationship("MatchGoal", back_populates="match")

    __table_args__ = (
        Index('ix_matches_competition_status', 'competition_id', 'status'),
        Index('ix_matches_utc_date', 'utc_date'),
        Index('ix_matches_team_dates', 'home_team_id', 'away_team_id', 'utc_date'),
        Index('ix_matches_season_competition', 'season_year', 'competition_id'),
    )
    
class MatchGoal(Base):
    """Goals scored in matches"""
    __tablename__ = "match_goals"
    
    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    
    minute = Column(Integer)
    injury_time = Column(Integer)
    type = Column(String(20))  # 'REGULAR', 'PENALTY', 'OWN_GOAL'
    
    team_id = Column(Integer)
    team_name = Column(String(150))
    
    scorer_id = Column(Integer, ForeignKey("players.id"))
    scorer_name = Column(String(150))
    
    assist_id = Column(Integer, ForeignKey("players.id"))
    assist_name = Column(String(150))
    
    home_score = Column(Integer)
    away_score = Column(Integer)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    match = relationship("Match", back_populates="goals")
    scorer = relationship("Player", foreign_keys=[scorer_id], back_populates="goals")
    assister = relationship("Player", foreign_keys=[assist_id], back_populates="assists")
    
    
class TeamStanding(Base):
    """
    Team standings by matchday - IMPORTANT FOR ML FEATURES
    Stores historical standings at different points in season
    """
    __tablename__ = "team_standings"
    
    id = Column(Integer, primary_key=True)
    
    competition_id = Column(Integer, ForeignKey("competitions.id"), nullable=False)
    season_id = Column(Integer, nullable=False)
    season_year = Column(String(10))
    matchday = Column(Integer)
    stage = Column(String(50))
    type = Column(String(20))  # 'TOTAL', 'HOME', 'AWAY'
    
    # Team
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    team_name = Column(String(150))
    position = Column(Integer, nullable=False)
    
    # Stats
    played_games = Column(Integer)
    won = Column(Integer)
    draw = Column(Integer)
    lost = Column(Integer)
    points = Column(Integer)
    goals_for = Column(Integer)
    goals_against = Column(Integer)
    goal_difference = Column(Integer)
    form = Column(String(50))  # 'W,W,L,D,W'
    
    fetched_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    competition = relationship("Competition", back_populates="standings")
    team = relationship("Team", back_populates="standings")
    
    __table_args__ = (
        Index('ix_standings_team_competition', 'team_id', 'competition_id'),
        Index('ix_standings_matchday', 'matchday'),
        Index('ix_standings_season_team', 'season_year', 'team_id'),
    )

class TopScorer(Base):
    """Top goal scorers per competition/season"""
    __tablename__ = "top_scorers"
    
    id = Column(Integer, primary_key=True)
    
    competition_id = Column(Integer, ForeignKey("competitions.id"), nullable=False)
    season_id = Column(Integer, nullable=False)
    season_year = Column(String(10))
    
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    player_name = Column(String(150))
    
    team_id = Column(Integer, ForeignKey("teams.id"))
    team_name = Column(String(150))
    
    goals = Column(Integer, nullable=False)
    assists = Column(Integer)
    penalties = Column(Integer)
    
    fetched_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    competition = relationship("Competition", back_populates="top_scorers")
    player = relationship("Player", back_populates="top_scorer_records")
    team = relationship("Team", back_populates="top_scorers")


class PlayerStat(Base):
    __tablename__ = "player_stats"

    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey("matches.id"))
    player_id = Column(Integer, ForeignKey("players.id"))
    team_id = Column(Integer, ForeignKey("teams.id")) 

    minutes_played = Column(Integer)
    goals = Column(Integer)
    assists = Column(Integer)
    yellow_cards = Column(Integer)
    red_cards = Column(Integer)
    
    # Additional stats
    shots = Column(Integer)
    shots_on_target = Column(Integer)
    passes = Column(Integer)
    pass_accuracy = Column(Float)
    
    # Relationships
    player = relationship("Player", back_populates="stats")
    match = relationship("Match", back_populates="player_stats")
    
    team = relationship("Team")

# Helper function to get ML-ready match result label
def get_match_result_label(match: Match) -> str:
    """
    Convert match winner to simple label for ML
    Returns: 'H' (home win), 'D' (draw), 'A' (away win)
    """
    winner = match.winner
    
    if winner == 'HOME_TEAM': # type: ignore
        return 'H'
    elif winner == 'AWAY_TEAM': # type: ignore
        return 'A'
    elif winner == 'DRAW': # type: ignore
        return 'D'
    return 'U'  # Unknown/not finished