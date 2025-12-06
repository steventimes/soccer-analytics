import numpy as np
import pandas as pd
from typing import Dict, Optional
from datetime import datetime
from app.data_service.data_service_factory import get
from app.data_service.data_type import type_db_data as type

class feature:
    def __init__(self): 
        self.feature_name = [
            'home_wins_last5', 'away_wins_last5', 
            'goal_diff_diff', 'rest_days_diff',
            'home_win_rate', 'away_win_rate'
        ]
        
    def _safe_div(self, n, d, default=0.0):
        return n / d if d else default

    def calculate_rest_days(self, team_id: int, current_match_date: datetime) -> int:
        recent_df = get(type.TEAM_RECENT, {
            'team_id': team_id, 
            'num_matches': 1, 
            'match_date': current_match_date
        })
        
        if recent_df is None or recent_df.empty:
            return 7 
            
        recent = recent_df.iloc[0]
        
        if 'last_match_date' not in recent or pd.isna(recent['last_match_date']):
            return 7

        last_date_str = recent['last_match_date'] 
        
        try:
            if isinstance(last_date_str, str):
                last_date_str = last_date_str.replace('Z', '')
                try:
                    last_date = datetime.fromisoformat(last_date_str)
                except ValueError:
                    last_date = datetime.strptime(last_date_str, "%Y-%m-%dT%H:%M:%S")
            else:
                last_date = last_date_str

            if last_date.tzinfo is None and current_match_date.tzinfo is not None:
                current_match_date = current_match_date.replace(tzinfo=None)
            
            delta = (current_match_date - last_date).days
            return max(0, delta)
            
        except Exception:
            return 7

    def create_match_feature(
        self,
        home_id: int,
        away_id: int,
        competition_id: int,
        season_yr: str,
        match_date: Optional[datetime] = None
    ) -> Dict:
        features = {}
        
        if isinstance(match_date, str):
            try:
                match_date = datetime.strptime(match_date, "%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                match_date = datetime.strptime(match_date, "%Y-%m-%d")

        home_df = get(type.TEAM_RECENT, {'team_id': home_id, 'num_matches': 5, 'match_date': match_date})
        away_df = get(type.TEAM_RECENT, {'team_id': away_id, 'num_matches': 5, 'match_date': match_date})
        
        if home_df is not None and not home_df.empty:
            h_stats = home_df.iloc[0]
        else:
            h_stats = {}
            
        if away_df is not None and not away_df.empty:
            a_stats = away_df.iloc[0]
        else:
            a_stats = {}
        
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

        if match_date:
            h_rest = self.calculate_rest_days(home_id, match_date)
            a_rest = self.calculate_rest_days(away_id, match_date)
            features['home_rest_days'] = h_rest
            features['away_rest_days'] = a_rest
            features['rest_days_diff'] = h_rest - a_rest
        else:
            features['home_rest_days'] = 7
            features['away_rest_days'] = 7
            features['rest_days_diff'] = 0

        return features