from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Float, JSON
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime

class SerializationMixin:
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
    name = Column(String(100), nullable=False)
    short_name = Column(String(50))
    tla = Column(String(10))
    crest = Column(Text)
    address = Column(String(200))
    website = Column(String(100))
    founded = Column(Integer)
    club_colors = Column(String(100))
    venue = Column(String(100))
    
    competition_id = Column(Integer, ForeignKey("competitions.id"))
    competition = relationship("Competition", back_populates="teams")

class Match(Base):
    __tablename__ = "matches"

    id = Column(Integer, primary_key=True)
    competition_id = Column(Integer, ForeignKey("competitions.id"))
    season_year = Column(String(10))
    utc_date = Column(DateTime)
    status = Column(String(20))
    matchday = Column(Integer)
    stage = Column(String(30))
    
    home_team_id = Column(Integer, ForeignKey("teams.id"))
    away_team_id = Column(Integer, ForeignKey("teams.id"))
    
    score_home = Column(Integer, nullable=True)
    score_away = Column(Integer, nullable=True)
    halftime_home = Column(Integer, nullable=True)
    halftime_away = Column(Integer, nullable=True)
    winner = Column(String(20), nullable=True)
    
    odds_home = Column(Float, nullable=True)
    odds_draw = Column(Float, nullable=True)
    odds_away = Column(Float, nullable=True)

    home_xg = Column(Float, nullable=True)
    away_xg = Column(Float, nullable=True)

    referees = Column(JSON, nullable=True)

    competition = relationship("Competition", back_populates="matches")
    player_stats = relationship("PlayerStat", back_populates="match")

class Player(Base):
    __tablename__ = "players"
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    position = Column(String(50))
    date_of_birth = Column(String(20))
    nationality = Column(String(50))
    shirt_number = Column(Integer)
    team_id = Column(Integer, ForeignKey("teams.id"))
    stats = relationship("PlayerStat", back_populates="player")
    top_scorers = relationship("TopScorer", back_populates="player")

class TeamStanding(Base):
    __tablename__ = "standings"
    id = Column(Integer, primary_key=True, autoincrement=True)
    competition_id = Column(Integer, ForeignKey("competitions.id"))
    season_year = Column(String(10))
    position = Column(Integer)
    team_id = Column(Integer, ForeignKey("teams.id"))
    played_games = Column(Integer)
    won = Column(Integer)
    draw = Column(Integer)
    lost = Column(Integer)
    points = Column(Integer)
    goals_for = Column(Integer)
    goals_against = Column(Integer)
    goal_difference = Column(Integer)
    competition = relationship("Competition", back_populates="standings")

class TopScorer(Base):
    __tablename__ = "top_scorers"
    id = Column(Integer, primary_key=True, autoincrement=True)
    competition_id = Column(Integer, ForeignKey("competitions.id"))
    season_year = Column(String(10))
    player_id = Column(Integer, ForeignKey("players.id"))
    team_id = Column(Integer, ForeignKey("teams.id"))
    goals = Column(Integer)
    assists = Column(Integer, nullable=True)
    penalties = Column(Integer, nullable=True)
    competition = relationship("Competition", back_populates="top_scorers")
    player = relationship("Player", back_populates="top_scorers")

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
    period_label = Column(String(50))
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