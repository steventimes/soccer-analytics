from sqlalchemy.orm import Session
from app.data_service.db.repositories.match_repository import MatchRepository
from app.data_service.db.repositories.team_repository import TeamRepository
from app.data_service.db.repositories.competition_repository import CompetitionRepository

class DataService:
    def __init__(self, session: Session):
        self.session = session
        self.matches = MatchRepository(session)
        self.teams = TeamRepository(session)
        self.competitions = CompetitionRepository(session)