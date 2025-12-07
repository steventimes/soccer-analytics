from sqlalchemy.orm import Session
from typing import List, Dict, Optional
from app.data_service.db.database.db_schema import Competition, TeamStanding, TopScorer
import logging

logger = logging.getLogger(__name__)

class CompetitionRepository:
    def __init__(self, session: Session):
        self.session = session

    def get_by_code(self, code: str) -> Optional[Competition]:
        return self.session.query(Competition).filter(Competition.code == code).first()

    def save_competition(self, comp_data: Dict):
        """Save competition metadata."""
        try:
            c_info = {
                'id': comp_data['id'],
                'name': comp_data['name'],
                'code': comp_data['code'],
                'area_name': comp_data['area']['name'],
                'area_code': comp_data['area']['code'],
                'type': comp_data['type'],
                'emblem': comp_data.get('emblem')
            }
            existing = self.session.query(Competition).filter_by(id=c_info['id']).first()
            if existing:
                for k, v in c_info.items():
                    setattr(existing, k, v)
            else:
                self.session.add(Competition(**c_info))
            self.session.commit()
            logger.info(f"Competition saved: {c_info['name']}")
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error saving competition: {e}")

    def save_standings(self, competition_id: int, season: str, table_data: List[Dict]):
        """Save league table."""
        try:
            self.session.query(TeamStanding).filter_by(
                competition_id=competition_id, season_year=season
            ).delete()
            
            for row in table_data:
                standing = TeamStanding(
                    competition_id=competition_id,
                    season_year=season,
                    position=row['position'],
                    team_id=row['team']['id'],
                    points=row['points'],
                    won=row['won'],
                    draw=row['draw'],
                    lost=row['lost'],
                    goals_for=row['goalsFor'],
                    goals_against=row['goalsAgainst'],
                    goal_difference=row['goalDifference']
                )
                self.session.add(standing)
            self.session.commit()
            logger.info(f"Standings saved for Comp {competition_id} Season {season}")
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error saving standings: {e}")

    def save_top_scorers(self, competition_id: int, season: str, scorers_data: List[Dict]):
        """Save top scorers list."""
        from app.data_service.db.database.db_schema import Player, TopScorer

        try:
            self.session.query(TopScorer).filter_by(
                competition_id=competition_id, season_year=season
            ).delete()

            for s in scorers_data:
                player_id = s['player']['id']

                if not self.session.query(Player).filter_by(id=player_id).first():
                    placeholder = Player(
                        id=player_id, 
                        name=s['player']['name'], 
                        team_id=s['team']['id']
                    )
                    self.session.add(placeholder)
                    self.session.flush()
                    
                scorer = TopScorer(
                    competition_id=competition_id,
                    season_year=season,
                    player_id=s['player']['id'],
                    team_id=s['team']['id'],
                    goals=s['goals'],
                    assists=s.get('assists'),
                    penalties=s.get('penalties')
                )
                self.session.add(scorer)
            
            self.session.commit()
            logger.info(f"Top Scorers saved for Comp {competition_id} Season {season}")
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error saving scorers: {e}")