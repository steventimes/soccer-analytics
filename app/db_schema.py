from sqlalchemy import Column, Integer, String, Date, ForeignKey, Float
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Competition(Base):
    __tablename__ = "competitions"

    id = Column(Integer, primary_key=True)
    name = Column(String)
    area_name = Column(String)
    code = Column(String)
    type = Column(String)

    teams = relationship("Team", back_populates="competition")
    matches = relationship("Match", back_populates="competition")


class Team(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True)
    name = Column(String)
    tla = Column(String)
    short_name = Column(String)
    founded = Column(Integer)
    playedGames = Column(Integer)
    won = Column(Integer)
    draw = Column(Integer)
    lost = Column(Integer)
    points = Column(Integer)
    goalsFor = Column(Integer)
    goalsAgainst = Column(Integer)
    goalDifference = Column(Integer)
    competition_id = Column(Integer, ForeignKey("competitions.id"))

    competition = relationship("Competition", back_populates="teams")
    players = relationship("Player", back_populates="team")
    home_matches = relationship("Match", foreign_keys="Match.home_team_id", back_populates="home_team")
    away_matches = relationship("Match", foreign_keys="Match.away_team_id", back_populates="away_team")


class Player(Base):
    __tablename__ = "players"

    id = Column(Integer, primary_key=True)
    name = Column(String)
    position = Column(String)
    date_of_birth = Column(Date)
    nationality = Column(String)
    shirtNumber = Column(Integer)
    marketValue = Column(Integer)
    team_id = Column(Integer, ForeignKey("teams.id"))

    team = relationship("Team", back_populates="players")
    stats = relationship("PlayerStat", back_populates="player")


class Match(Base):
    __tablename__ = "matches"

    id = Column(Integer, primary_key=True)
    utc_date = Column(String)
    status = Column(String)
    matchday = Column(Integer)
    home_team_id = Column(Integer, ForeignKey("teams.id"))
    away_team_id = Column(Integer, ForeignKey("teams.id"))
    competition_id = Column(Integer, ForeignKey("competitions.id"))
    score_home = Column(Integer)
    score_away = Column(Integer)

    competition = relationship("Competition", back_populates="matches")
    home_team = relationship("Team", foreign_keys=[home_team_id], back_populates="home_matches")
    away_team = relationship("Team", foreign_keys=[away_team_id], back_populates="away_matches")
    player_stats = relationship("PlayerStat", back_populates="match")


class PlayerStat(Base):
    __tablename__ = "player_stats"

    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey("matches.id"))
    player_id = Column(Integer, ForeignKey("players.id"))
    minutes_played = Column(Integer)
    goals = Column(Integer)
    assists = Column(Integer)
    yellow_cards = Column(Integer)
    red_cards = Column(Integer)

    player = relationship("Player", back_populates="stats")
    match = relationship("Match", back_populates="player_stats")
