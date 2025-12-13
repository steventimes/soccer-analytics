from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Float
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime

class SerializationMixin:
    """Helper to convert SQLAlchemy models to dictionaries"""
    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

Base = declarative_base(cls=SerializationMixin)

class Competition(Base):
    __tablename__ = "competitions"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    code = Column(String(10), nullable=False)
    area_name = Column(String(100))
    area_code = Column(String(10))
    type = Column(String(20))
    emblem = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    teams = relationship("Team", back_populates="competition")
    matches = relationship("Match", back_populates="competition")
    standings = relationship("TeamStanding", back_populates="competition")
    top_scorers = relationship("TopScorer", back_populates="competition")

class Team(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True)
    name = Column(String(150), nullable=False)
    short_name = Column(String(100))
    tla = Column(String(10))
    crest = Column(Text)
    venue = Column(String(150))
    competition_id = Column(Integer, ForeignKey("competitions.id"))
    
    competition = relationship("Competition", back_populates="teams")
    players = relationship("Player", back_populates="team")
    top_scorers = relationship("TopScorer", back_populates="team")

class Player(Base):
    __tablename__ = "players"

    id = Column(Integer, primary_key=True)
    name = Column(String(150), nullable=False)
    position = Column(String(50))
    date_of_birth = Column(String(20))
    nationality = Column(String(100))
    team_id = Column(Integer, ForeignKey("teams.id"))
    
    team = relationship("Team", back_populates="players")
    top_scorer_records = relationship("TopScorer", back_populates="player")
    stats = relationship("PlayerStat", back_populates="player")

class Match(Base):
    __tablename__ = "matches"

    id = Column(Integer, primary_key=True)
    utc_date = Column(DateTime)
    status = Column(String(20))
    matchday = Column(Integer)
    stage = Column(String(30))
    group = Column(String(30))
    season_year = Column(String(10))
    
    competition_id = Column(Integer, ForeignKey("competitions.id"))
    home_team_id = Column(Integer, ForeignKey("teams.id"), nullable=True)
    away_team_id = Column(Integer, ForeignKey("teams.id"), nullable=True)
    
    score_home = Column(Integer)
    score_away = Column(Integer)
    halftime_home = Column(Integer)
    halftime_away = Column(Integer)
    winner = Column(String(20))

    home_xg = Column(Float, nullable=True)
    away_xg = Column(Float, nullable=True)
    
    competition = relationship("Competition", back_populates="matches")
    home_team = relationship("Team", foreign_keys=[home_team_id])
    away_team = relationship("Team", foreign_keys=[away_team_id])
    player_stats = relationship("PlayerStat", back_populates="match")

class TeamStanding(Base):
    __tablename__ = "team_standings"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    competition_id = Column(Integer, ForeignKey("competitions.id"))
    season_year = Column(String(10))
    position = Column(Integer)
    team_id = Column(Integer, ForeignKey("teams.id"))
    points = Column(Integer)
    won = Column(Integer)
    draw = Column(Integer)
    lost = Column(Integer)
    goals_for = Column(Integer)
    goals_against = Column(Integer)
    goal_difference = Column(Integer)
    
    competition = relationship("Competition", back_populates="standings")
    team = relationship("Team")

class TopScorer(Base):
    __tablename__ = "top_scorers"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    competition_id = Column(Integer, ForeignKey("competitions.id"))
    season_year = Column(String(10))
    player_id = Column(Integer, ForeignKey("players.id"))
    team_id = Column(Integer, ForeignKey("teams.id"))
    goals = Column(Integer)
    assists = Column(Integer)
    penalties = Column(Integer)
    
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
    
    player = relationship("Player", back_populates="stats")
    match = relationship("Match", back_populates="player_stats")
    team = relationship("Team")

class PlayerForm(Base):
    __tablename__ = "player_form"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=True)
    team_id = Column(Integer, ForeignKey("teams.id"))
    period_label = Column(String(50))  # e.g., "2023_season"
    
    goals = Column(Integer, default=0)
    xg = Column(Float, default=0.0)
    assists = Column(Integer, default=0)
    xa = Column(Float, default=0.0)
    shots = Column(Integer, default=0)
    key_passes = Column(Integer, default=0)
    yellow_cards = Column(Integer, default=0)
    red_cards = Column(Integer, default=0)
    npg = Column(Float, default=0.0)
    npxg = Column(Float, default=0.0)
    xg_chain = Column(Float, default=0.0)
    xg_buildup = Column(Float, default=0.0)
    minutes_played = Column(Integer, default=0)
    games_played = Column(Integer, default=0)