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
                last_date = datetime.strptime(last_date_str, "%Y-%m-%dT%H:%M:%SZ")
            else:
                last_date = last_date_str
                
            if isinstance(current_match_date, str):
                current_date = datetime.strptime(current_match_date, "%Y-%m-%dT%H:%M:%SZ")
            else:
                current_date = current_match_date
                
            delta = (current_date - last_date).days
            return max(0, delta)
        except Exception:
            return 7

    def calculate_features(self, match: Dict) -> Dict:
        features = {}
        
        home_id = match['home_team']['id']
        away_id = match['away_team']['id']
        match_date = match['utc_date']
        
        h_stats = get(type.TEAM_RECENT, {'team_id': home_id, 'match_date': match_date, 'num_matches': 5})
        a_stats = get(type.TEAM_RECENT, {'team_id': away_id, 'match_date': match_date, 'num_matches': 5})
        
        if h_stats is None: h_stats = {}
        if a_stats is None: a_stats = {}
        
        # Helper to extract stats from the aggregated dictionary returned by TEAM_RECENT
        # Assuming TEAM_RECENT returns a DataFrame or Dict with aggregated stats
        # If it returns a DataFrame of matches, we need to aggregate. 
        # Based on snippet, it likely returns a summary dict or single row DataFrame.
        
        if isinstance(h_stats, pd.DataFrame) and not h_stats.empty:
            h_stats = h_stats.iloc[0].to_dict()
        elif isinstance(h_stats, pd.DataFrame):
            h_stats = {}
            
        if isinstance(a_stats, pd.DataFrame) and not a_stats.empty:
            a_stats = a_stats.iloc[0].to_dict()
        elif isinstance(a_stats, pd.DataFrame):
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
        
        features['rest_days_diff'] = self.calculate_rest_days(home_id, match_date) - \
                                     self.calculate_rest_days(away_id, match_date)
                                     
        return features