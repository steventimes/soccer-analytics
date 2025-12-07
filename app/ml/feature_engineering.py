import pandas as pd
from datetime import datetime
from typing import Dict, Any

from app.data_service.db.database.db_schema import Match
from app.data_service.db.data_service import DataService

class FeatureEngineer:
    def __init__(self): 
        self.feature_names = [
            'home_wins_last5', 'away_wins_last5', 
            'goal_diff_diff', 'rest_days_diff',
            'home_win_rate', 'away_win_rate'
        ]
        
    def _safe_div(self, n, d, default=0.0):
        return n / d if d else default

    def calculate_rest_days(self, team_id: int, current_match_date: datetime, service: DataService) -> int:
        """
        Calculate days since the last match for a team.
        """
        
        last_match = service.matches.get_recent_form(team_id, current_match_date, limit=1)
        
        from app.data_service.db.database.db_schema import Match as MatchModel
        from sqlalchemy import or_
        
        last_match_obj = service.session.query(MatchModel).filter(
            or_(MatchModel.home_team_id == team_id, MatchModel.away_team_id == team_id),
            MatchModel.utc_date < current_match_date,
            MatchModel.status == 'FINISHED'
        ).order_by(MatchModel.utc_date.desc()).first()
        
        if not last_match_obj:
            return 7 
            
        delta = (current_match_date - last_match_obj.utc_date).days
        return max(0, delta)

    def calculate_features(self, match: Match, service: DataService) -> Dict[str, Any]:
        """
        Calculate features for a given Match object.
        """
        features = {}
        
        h_stats = service.matches.get_recent_form(match.home_team_id, match.utc_date, limit=5)
        a_stats = service.matches.get_recent_form(match.away_team_id, match.utc_date, limit=5)
        
        features['home_wins_last5'] = h_stats.get('wins', 0)
        features['away_wins_last5'] = a_stats.get('wins', 0)
        
        features['home_draws_last5'] = h_stats.get('draws', 0)
        features['away_draws_last5'] = a_stats.get('draws', 0)
        
        features['home_losses_last5'] = h_stats.get('losses', 0)
        features['away_losses_last5'] = a_stats.get('losses', 0)
        
        h_gd = h_stats.get('goals_scored', 0) - h_stats.get('goals_conceded', 0)
        a_gd = a_stats.get('goals_scored', 0) - a_stats.get('goals_conceded', 0)
        features['goal_diff_diff'] = h_gd - a_gd

        h_games = h_stats.get('total_games', 5)
        a_games = a_stats.get('total_games', 5)
        
        features['home_win_rate'] = self._safe_div(h_stats.get('wins', 0), h_games)
        features['away_win_rate'] = self._safe_div(a_stats.get('wins', 0), a_games)
        features['win_rate_diff'] = features['home_win_rate'] - features['away_win_rate']
        
        features['rest_days_diff'] = self.calculate_rest_days(match.home_team_id, match.utc_date, service) - \
                                     self.calculate_rest_days(match.away_team_id, match.utc_date, service)
                                     
        return features