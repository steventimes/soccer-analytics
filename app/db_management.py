from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class Player(Base):
    __tablename__ = "players"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    position = Column(String)
    nationality = Column(String)
    team_id = Column(Integer)

def init_db(db_path="sqlite:///football.db"):
    engine = create_engine(db_path)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()

def save_players(session, players_df, team_id):
    for _, row in players_df.iterrows():
        player = Player(
            id=row["id"],
            name=row["name"],
            position=row.get("position", ""),
            nationality=row.get("nationality", ""),
            team_id=team_id,
        )
        session.merge(player)  # update if exists
    session.commit()
