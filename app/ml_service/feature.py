import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime
from app.data_service.data_service_factory import get
from app.data_service.data_type import type_db_data as type

class feature:
    def __init__(self): # type: ignore
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
        home_form = get(type.TEAM_RECENT, {'team_id':home_id, 'num_matches':5, 'match_date':match_date})
        away_form = get(type.TEAM_RECENT, {'team_id':away_id, 'num_matches':5, 'match_date':match_date})
        
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
        
        h2h = get(type.HEAD_TO_HEAD, {'home_id': home_id, 'away_id': away_id,'limit': 10})
        
        features['h2h_total_matches'] = h2h.get('total_matches', 0)
        features['h2h_home_wins'] = h2h.get('team1_wins', 0)
        features['h2h_draws'] = h2h.get('draws', 0)
        features['h2h_away_wins'] = h2h.get('team2_wins', 0)
        features['h2h_home_goals'] = h2h.get('team1_goals', 0)
        features['h2h_away_goals'] = h2h.get('team2_goals', 0)
        
        if features['h2h_total_matches'] > 0: # type: ignore
            features['h2h_home_win_rate'] = features['h2h_home_wins'] / features['h2h_total_matches']
        else:
            features['h2h_home_win_rate'] = 0.33
        
        home_standing = get(
            type.STANDING_MATCHDAY, {'team_id': home_id, 'competition_id': competition_id, 'season_year': season_yr}
        )
        away_standing = get(
            type.STANDING_MATCHDAY, {'team_id': away_id, 'competition_id': competition_id, 'season_year': season_yr}
        )
        
        if home_standing: # type: ignore
            features['home_position'] = home_standing.get('position', 10)
            features['home_points'] = home_standing.get('points', 0)
            features['home_goal_diff'] = home_standing.get('goal_difference', 0)
        else:
            features['home_position'] = 10
            features['home_points'] = 0
            features['home_goal_diff'] = 0
        
        if away_standing: # type: ignore
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
    
    
    def prepare_training_data(self, matches_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Prepare historical match data for training
        
        Args:
            matches_df: DataFrame with columns:
                - match_id
                - home_team_id, away_team_id
                - competition_id, season_year
                - match_date
                - result ('H', 'D', 'A')
        
        Returns:
            X: Features DataFrame
            y: Target Series (match results)
        """
        print(f"Preparing training data from {len(matches_df)} matches")
        
        feature_list = []
        labels = []
        valid_matches = 0
        
        for idx, row in matches_df.iterrows():
            try:
                features = self.create_match_feature(
                    home_id=row['home_team_id'],
                    away_id=row['away_team_id'],
                    competition_id=row['competition_id'],
                    season_yr=row['season_year'],
                    match_date=row.get('match_date')
                )
                
                feature_list.append(features)
                labels.append(row['result'])
                valid_matches += 1
                
            except Exception as e:
                print(f"Skipping match {row.get('match_id')}: {e}")
                continue
        
        print(f"Successfully prepared {valid_matches}/{len(matches_df)} matches")
        
        X = pd.DataFrame(feature_list)
        y = pd.Series(labels)
        
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return self.feature_names
    
    def validate_features(self, X: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
        """
        Validate features and handle missing values
        
        Returns:
            Cleaned DataFrame and list of warnings
        """
        warnings = []
        
        missing = X.isnull().sum()
        if missing.any():
            warnings.append(f"Missing values found: {missing[missing > 0].to_dict()}")
            X = X.fillna(0)
        
        if np.isinf(X.select_dtypes(include=[np.number])).any().any():
            warnings.append("Infinite values found, replacing with 0")
            X = X.replace([np.inf, -np.inf], 0)

        low_variance = X.var() < 0.01
        if low_variance.any():
            low_var_features = X.columns[low_variance].tolist()
            warnings.append(f"Low variance features: {low_var_features}")
        
        return X, warnings


def create_simple_features(home_team_stats: Dict, away_team_stats: Dict) -> Dict:
    features = {
        'home_position': home_team_stats.get('position', 10),
        'away_position': away_team_stats.get('position', 10),
        'home_points': home_team_stats.get('points', 0),
        'away_points': away_team_stats.get('points', 0),
        'home_goal_diff': home_team_stats.get('goal_difference', 0),
        'away_goal_diff': away_team_stats.get('goal_difference', 0),
        
        'position_diff': away_team_stats.get('position', 10) - home_team_stats.get('position', 10),
        'points_diff': home_team_stats.get('points', 0) - away_team_stats.get('points', 0),
        'goal_diff_diff': home_team_stats.get('goal_difference', 0) - away_team_stats.get('goal_difference', 0),

        'home_wins_last5': home_team_stats.get('wins_last5', 0),
        'away_wins_last5': away_team_stats.get('wins_last5', 0),
        'home_goals_scored_last5': home_team_stats.get('goals_scored_last5', 0),
        'away_goals_scored_last5': away_team_stats.get('goals_scored_last5', 0),
        'home_goals_conceded_last5': home_team_stats.get('goals_conceded_last5', 0),
        'away_goals_conceded_last5': away_team_stats.get('goals_conceded_last5', 0),

        'home_win_rate': home_team_stats.get('win_rate', 0.33),
        'away_win_rate': away_team_stats.get('win_rate', 0.33),
        'win_rate_diff': home_team_stats.get('win_rate', 0.33) - away_team_stats.get('win_rate', 0.33),

        'form_diff_wins': home_team_stats.get('wins_last5', 0) - away_team_stats.get('wins_last5', 0),
        'form_diff_goals': home_team_stats.get('goals_scored_last5', 0) - away_team_stats.get('goals_scored_last5', 0),

        'home_draws_last5': 0,
        'home_losses_last5': 0,
        'away_draws_last5': 0,
        'away_losses_last5': 0,
        'h2h_total_matches': 0,
        'h2h_home_wins': 0,
        'h2h_draws': 0,
        'h2h_away_wins': 0,
        'h2h_home_goals': 0,
        'h2h_away_goals': 0,
        'h2h_home_win_rate': 0.33,
        'is_home': 1
    }
    
    return features

def get_feature_descriptions(self) -> Dict[str, str]:
    """
    Get descriptions for all features (useful for model interpretation)
    """
    return {
        'home_wins_last5': 'Home team wins in last 5 matches',
        'away_wins_last5': 'Away team wins in last 5 matches',
        'home_win_rate': 'Home team win rate in recent matches',
        'away_win_rate': 'Away team win rate in recent matches',
        'form_diff_wins': 'Difference in recent wins between teams',
        'form_diff_goals': 'Difference in recent goals scored',
        'win_rate_diff': 'Difference in win rates',
        'h2h_home_win_rate': 'Historical home win rate in head-to-head',
        'home_position': 'Home team league position',
        'away_position': 'Away team league position',
        'position_diff': 'Difference in league positions',
        'points_diff': 'Difference in league points',
        'goal_diff_diff': 'Difference in goal differences',
        'is_home': 'Home advantage indicator (always 1 for home team)'
    }