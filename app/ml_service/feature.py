import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from sqlalchemy.orm import session as SQLSession
from datetime import datetime
from app.data_service.data_service_factory import get
from app.data_service.data_type import type_db_data as type

class feature:
    def __init__(self, session: SQLSession): # type: ignore
        self.session = session
        self.feature_name = []
        
    def create_match_feature(
        self,
        home_id: int,
        away_id:int,
        competition_id: int,
        season_yr: str,
        match_date: Optional[datetime] = None
    ) -> Dict:
        features = {}
        home_form = get(self.session, {'team_id':home_id, 'num_matches':5, 'match_date':match_date})
        away_form = get(self.session, {'team_id':away_id, 'num_matches':5, 'match_date':match_date})
        
        features['home_wins_last5'] = home_form.get('wins', 0)
        features['home_draws_last5'] = home_form.get('draws', 0)
        features['home_losses_last5'] = home_form.get('losses', 0)
        features['home_goals_scored_last5'] = home_form.get('goals_scored', 0)
        features['home_goals_conceded_last5'] = home_form.get('goals_conceded', 0)
        features['home_win_rate'] = home_form.get('win_rate', 0)
        
        features['away_wins_last5'] = away_form.get('wins', 0)
        features['away_draws_last5'] = away_form.get('draws', 0)
        features['away_losses_last5'] = away_form.get('losses', 0)
        features['away_goals_scored_last5'] = away_form.get('goals_scored', 0)
        features['away_goals_conceded_last5'] = away_form.get('goals_conceded', 0)
        features['away_win_rate'] = away_form.get('win_rate', 0) 

        features['form_diff_wins'] = features['home_wins_last5'] - features['away_wins_last5']
        features['form_diff_goals'] = features['home_goals_scored_last5'] - features['away_goals_scored_last5']
        features['win_rate_diff'] = features['home_win_rate'] - features['away_win_rate']
        
        h2h = get_head_to_head_db(self.session, home_team_id, away_team_id, 10)
        
        features['h2h_total_matches'] = h2h.get('total_matches', 0)
        features['h2h_home_wins'] = h2h.get('team1_wins', 0)
        features['h2h_draws'] = h2h.get('draws', 0)
        features['h2h_away_wins'] = h2h.get('team2_wins', 0)
        features['h2h_home_goals'] = h2h.get('team1_goals', 0)
        features['h2h_away_goals'] = h2h.get('team2_goals', 0)
        
        # H2H win rate
        if features['h2h_total_matches'] > 0:
            features['h2h_home_win_rate'] = features['h2h_home_wins'] / features['h2h_total_matches']
        else:
            features['h2h_home_win_rate'] = 0.33  # Default equal probability
        
        # 4. League Standing Features (if available)
        home_standing = get_team_standing_at_matchday_db(
            self.session, home_team_id, competition_id, season_year
        )
        away_standing = get_team_standing_at_matchday_db(
            self.session, away_team_id, competition_id, season_year
        )
        
        if home_standing:
            features['home_position'] = home_standing.get('position', 10)
            features['home_points'] = home_standing.get('points', 0)
            features['home_goal_diff'] = home_standing.get('goal_difference', 0)
        else:
            features['home_position'] = 10
            features['home_points'] = 0
            features['home_goal_diff'] = 0
        
        if away_standing:
            features['away_position'] = away_standing.get('position', 10)
            features['away_points'] = away_standing.get('points', 0)
            features['away_goal_diff'] = away_standing.get('goal_difference', 0)
        else:
            features['away_position'] = 10
            features['away_points'] = 0
            features['away_goal_diff'] = 0
        
        features['position_diff'] = features['away_position'] - features['home_position']
        features['points_diff'] = features['home_points'] - features['away_points']
        features['goal_diff_diff'] = features['home_goal_diff'] - features['away_goal_diff']
        
        features['is_home'] = 1
        
        return features