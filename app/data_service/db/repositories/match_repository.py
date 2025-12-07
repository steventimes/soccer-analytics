from sqlalchemy.orm import Session
from sqlalchemy import or_
from typing import List, Dict, Optional
from datetime import datetime
import logging
from app.data_service.db.database.db_schema import Match, Team

logger = logging.getLogger(__name__)

class MatchRepository:
    def __init__(self, session: Session):
        self.session = session

    def get_by_competition(self, competition_id: int, season: str) -> List[Match]:
        """Fetch all finished matches for a competition/season."""
        return self.session.query(Match).filter(
            Match.competition_id == competition_id,
            Match.season_year == str(season),
            Match.status == 'FINISHED'
        ).all()

    def get_recent_form(self, team_id: int, match_date: datetime, limit: int = 5) -> Dict:
        """Calculate recent form (wins/losses/goals) for a team before a specific date."""
        matches = self.session.query(Match).filter(
            or_(Match.home_team_id == team_id, Match.away_team_id == team_id),
            Match.utc_date < match_date,
            Match.status == 'FINISHED'
        ).order_by(Match.utc_date.desc()).limit(limit).all()

        stats = {'wins': 0, 'draws': 0, 'losses': 0, 'goals_scored': 0, 'goals_conceded': 0, 'total_games': len(matches)}

        for m in matches:
            is_home = m.home_team_id == team_id
            goals_for = m.score_home if is_home else m.score_away
            goals_against = m.score_away if is_home else m.score_home
            
            stats['goals_scored'] += goals_for or 0
            stats['goals_conceded'] += goals_against or 0

            if m.winner == 'DRAW':
                stats['draws'] += 1
            elif (is_home and m.winner == 'HOME_TEAM') or (not is_home and m.winner == 'AWAY_TEAM'):
                stats['wins'] += 1
            else:
                stats['losses'] += 1
        
        return stats

    def get_head_to_head(self, team1_id: int, team2_id: int, limit: int = 10) -> List[Match]:
        """Get last N matches between two teams."""
        return self.session.query(Match).filter(
            or_(
                (Match.home_team_id == team1_id) & (Match.away_team_id == team2_id),
                (Match.home_team_id == team2_id) & (Match.away_team_id == team1_id)
            ),
            Match.status == 'FINISHED'
        ).order_by(Match.utc_date.desc()).limit(limit).all()

    def save_bulk(self, matches_data: List[Dict]):
        """Save a list of matches, creating placeholder teams if needed."""
        try:
            for m in matches_data:
                self._ensure_team(m.get('homeTeam'))
                self._ensure_team(m.get('awayTeam'))

                match_info = {
                    'id': m['id'],
                    'utc_date': datetime.strptime(m['utcDate'], "%Y-%m-%dT%H:%M:%SZ"),
                    'status': m['status'],
                    'matchday': m.get('matchday'),
                    'stage': m.get('stage'),
                    'season_year': str(m['season']['startDate'])[:4],
                    'competition_id': m['competition']['id'],
                    'home_team_id': m['homeTeam']['id'],
                    'away_team_id': m['awayTeam']['id'],
                    'score_home': m['score']['fullTime']['home'],
                    'score_away': m['score']['fullTime']['away'],
                    'halftime_home': m['score']['halfTime']['home'],
                    'halftime_away': m['score']['halfTime']['away'],
                    'winner': m['score']['winner']
                }

                existing = self.session.query(Match).filter_by(id=match_info['id']).first()
                if existing:
                    for k, v in match_info.items():
                        setattr(existing, k, v)
                else:
                    self.session.add(Match(**match_info))
            
            self.session.commit()
            logger.info(f"Saved {len(matches_data)} matches.")
        except Exception as e:
            self.session.rollback()
            logger.error(f"Failed to save matches: {e}")
            raise

    def _ensure_team(self, team_data: Dict):
        """Helper to create a placeholder team if it doesn't exist during match save."""
        if not team_data or 'id' not in team_data: 
            return
        
        team_id = team_data['id']
        if not self.session.query(Team).filter_by(id=team_id).first():
            placeholder = Team(id=team_id, name=team_data.get('name', f"Team {team_id}"))
            self.session.add(placeholder)
            self.session.flush()