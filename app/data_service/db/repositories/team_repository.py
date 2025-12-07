from sqlalchemy.orm import Session
from typing import List, Optional, Dict
from app.data_service.db.database.db_schema import Team, Player
import logging

logger = logging.getLogger(__name__)

class TeamRepository:
    def __init__(self, session: Session):
        self.session = session

    def get_by_id(self, team_id: int) -> Optional[Team]:
        return self.session.query(Team).filter(Team.id == team_id).first()

    def get_players(self, team_id: int) -> List[Player]:
        return self.session.query(Player).filter(Player.team_id == team_id).all()

    def save_team(self, team_data: Dict) -> Team:
        team = self.get_by_id(team_data['id'])
        if team:
            for k, v in team_data.items():
                if hasattr(team, k):
                    setattr(team, k, v)
        else:
            team = Team(**team_data)
            self.session.add(team)
        
        self.session.commit()
        return team

    def save_squad(self, team_id: int, squad_list: List[Dict]):
        """Save a list of players for a specific team."""
        if not self.get_by_id(team_id):
            self.save_team({'id': team_id, 'name': f"Team {team_id}"})

        try:
            count = 0
            for p_data in squad_list:
                player_id = p_data['id']
                player_info = {
                    'id': player_id,
                    'name': p_data['name'],
                    'position': p_data.get('position'),
                    'date_of_birth': p_data.get('dateOfBirth'),
                    'nationality': p_data.get('nationality'),
                    'team_id': team_id
                }
                
                existing = self.session.query(Player).filter_by(id=player_id).first()
                if existing:
                    for k, v in player_info.items():
                        setattr(existing, k, v)
                else:
                    self.session.add(Player(**player_info))
                count += 1
            
            self.session.commit()
            logger.info(f"Saved {count} players for Team {team_id}")
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error saving squad for team {team_id}: {e}")