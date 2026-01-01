import pandas as pd
import numpy as np

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
            'points_diff'
        ]

    def calculate_rolling_features(self, df: pd.DataFrame, window=5) -> pd.DataFrame:
        df = df.sort_values(['teamID', 'date'])
        
        if 'result' in df.columns:
            result_map = {'L': 0, 'D': 1, 'W': 2}
            df['target'] = df['result'].map(result_map)
        
        df['is_home'] = np.where(df['location'] == 'h', 1, 0)
        
        for col in ['xGoals', 'deep', 'ppda', 'goals']:
            if col not in df.columns: 
                df[col] = 0

        grouped = df.groupby('teamID')
        
        def roll_mean(col):
            return grouped[col].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

        df['rolling_xG'] = roll_mean('xGoals')
        df['rolling_xGA'] = roll_mean('xGoals')
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
            
            df['xG_diff'] = (df['rolling_xG'] - df['opp_rolling_xG'])
            df['ppda_diff'] = (df['rolling_ppda'] - df['opp_rolling_ppda'])
            df['deep_diff'] = (df['rolling_deep'] - df['opp_rolling_deep'])
            df['points_diff'] = (df['rolling_points'] - df['opp_rolling_points'])

            df = df.fillna(0)

        return df