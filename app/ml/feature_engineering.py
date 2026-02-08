import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.features = [
            'rolling_xG', 
            'rolling_xGA', 
            'rolling_deep', 
            'rolling_ppda',
            'rolling_goals', 
            'rolling_wins', 
            'is_home',
            'xG_diff', 
            'ppda_diff', 
            'deep_diff', 
            'points_diff',
            'team_elo',
            'opp_elo',
            'elo_diff',
            'rest_days'
        ]

    def _calculate_elo(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Iterates through matches chronologically to calculate ELO ratings.
        """
        current_elo = {} 
        k_factor = 20
        base_rating = 1500

        df = df.sort_values('date').reset_index(drop=True)
        
        team_elo_list = []
        opp_elo_list = []
        processed_matches = set()
        rows = df.to_dict('records')

        for row in rows:
            t_id = row['teamID']
            o_id = row['opponentID']
            m_id = row['id']

            if t_id not in current_elo: current_elo[t_id] = base_rating
            if o_id not in current_elo: current_elo[o_id] = base_rating

            team_elo_list.append(current_elo[t_id])
            opp_elo_list.append(current_elo[o_id])

            if m_id in processed_matches:
                continue
            
            if row['result'] == 'W': actual_score = 1.0
            elif row['result'] == 'D': actual_score = 0.5
            else: actual_score = 0.0

            ra = current_elo[t_id]
            rb = current_elo[o_id]
            expected_score = 1 / (1 + 10 ** ((rb - ra) / 400))

            rating_change = k_factor * (actual_score - expected_score)
            current_elo[t_id] += rating_change
            current_elo[o_id] -= rating_change
            processed_matches.add(m_id)

        df['team_elo'] = team_elo_list
        df['opp_elo'] = opp_elo_list
        return df

    def calculate_rolling_features(self, df: pd.DataFrame, window=5) -> pd.DataFrame:
        df = df.sort_values(['teamID', 'date'])
        
        if 'result' in df.columns:
            result_map = {'L': 0, 'D': 1, 'W': 2}
            df['target'] = df['result'].map(result_map)
        
        df['is_home'] = np.where(df['location'] == 'h', 1, 0)
        
        for col in ['xGoals', 'deep', 'ppda', 'goals']:
            if col not in df.columns: df[col] = 0

        df['date'] = pd.to_datetime(df['date'])
        grouped = df.groupby('teamID')
        df['last_date'] = grouped['date'].shift(1)
        diff_series = df['date'] - df['last_date']
        df['rest_days'] = diff_series.apply(lambda x: x.days if pd.notnull(x) else 7)
        df['rest_days'] = df['rest_days'].clip(upper=14)
  
        def roll_mean(col):
            return grouped[col].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

        df['rolling_xG'] = roll_mean('xGoals')
        df['rolling_deep'] = roll_mean('deep')
        df['rolling_ppda'] = roll_mean('ppda')
        df['rolling_goals'] = roll_mean('goals')
        
        df['win_numeric'] = np.where(df['result'] == 'W', 3, np.where(df['result'] == 'D', 1, 0))
        df['rolling_points'] = grouped['win_numeric'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        df['rolling_wins'] = np.where(df['result'] == 'W', 1, 0)
        df['rolling_wins'] = grouped['rolling_wins'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).sum())

        match_id_col = 'id' if 'id' in df.columns else 'gameID'

        if match_id_col in df.columns:
            opp_stats = df[[match_id_col, 'teamID', 'rolling_xG', 'rolling_ppda', 'rolling_deep', 'rolling_points']].copy()
            opp_stats.rename(columns={
                'teamID': 'oppID', 
                'rolling_xG': 'opp_rolling_xG', 
                'rolling_ppda': 'opp_rolling_ppda',
                'rolling_deep': 'opp_rolling_deep',
                'rolling_points': 'opp_rolling_points'
            }, inplace=True)
            
            df = df.merge(opp_stats, left_on=[match_id_col, 'opponentID'], right_on=[match_id_col, 'oppID'], how='left')
            
            if 'opp_rolling_xG' in df.columns:
                df['rolling_xGA'] = df['opp_rolling_xG']
                df['xG_diff'] = (df['rolling_xG'] - df['opp_rolling_xG'])
                df['ppda_diff'] = (df['rolling_ppda'] - df['opp_rolling_ppda'])
                df['deep_diff'] = (df['rolling_deep'] - df['opp_rolling_deep'])
                df['points_diff'] = (df['rolling_points'] - df['opp_rolling_points'])
            else:
                logger.warning("Opponent stats missing after merge. Filling with 0.")
                for c in ['rolling_xGA', 'xG_diff', 'ppda_diff', 'deep_diff', 'points_diff']:
                    df[c] = 0
            
            df = df.fillna(0)

        df = self._calculate_elo(df)
        df['elo_diff'] = df['team_elo'] - df['opp_elo']

        for col in ['odds_home', 'odds_draw', 'odds_away']:
            if col not in df.columns: df[col] = 1.0
            else: df[col] = df[col].fillna(1.0)

        df = df.dropna(subset=['target', 'rolling_xG'])
        
        return df