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
        Assign pre-match ELO ratings and update ratings once per match.

        If both team/opponent viewpoints of the same match are present, both
        rows get the same pre-match ratings and only one update is applied.
        """
        current_elo = {}
        k_factor = 20
        base_rating = 1500

        working_df = df.copy()
        working_df['_elo_row_order'] = np.arange(len(working_df))

        match_id_col = 'id' if 'id' in working_df.columns else ('gameID' if 'gameID' in working_df.columns else None)

        sort_cols = ['date'] if 'date' in working_df.columns else []
        if match_id_col:
            sort_cols.append(match_id_col)
        sort_cols.append('_elo_row_order')
        working_df = working_df.sort_values(sort_cols, kind='mergesort').reset_index(drop=False)
        working_df.rename(columns={'index': '_orig_order'}, inplace=True)

        rows = working_df.to_dict('records')
        team_elo = np.zeros(len(rows), dtype=float)
        opp_elo = np.zeros(len(rows), dtype=float)

        if match_id_col:
            grouped_rows = []
            previous_match = object()
            for idx, row in enumerate(rows):
                current_match = row[match_id_col]
                if idx == 0 or current_match != previous_match:
                    grouped_rows.append([idx])
                    previous_match = current_match
                else:
                    grouped_rows[-1].append(idx)
        else:
            grouped_rows = [[idx] for idx in range(len(rows))]

        for match_rows in grouped_rows:
            for idx in match_rows:
                row = rows[idx]
                team_id = row['teamID']
                opp_id = row['opponentID']
                if team_id not in current_elo:
                    current_elo[team_id] = base_rating
                if opp_id not in current_elo:
                    current_elo[opp_id] = base_rating
                team_elo[idx] = current_elo[team_id]
                opp_elo[idx] = current_elo[opp_id]

            row0 = rows[match_rows[0]]
            team_id = row0['teamID']
            opp_id = row0['opponentID']
            if row0['result'] == 'W':
                actual_score = 1.0
            elif row0['result'] == 'D':
                actual_score = 0.5
            else:
                actual_score = 0.0

            ra = current_elo[team_id]
            rb = current_elo[opp_id]
            expected_score = 1 / (1 + 10 ** ((rb - ra) / 400))
            rating_change = k_factor * (actual_score - expected_score)
            current_elo[team_id] += rating_change
            current_elo[opp_id] -= rating_change

        working_df['team_elo'] = team_elo
        working_df['opp_elo'] = opp_elo

        working_df = working_df.sort_values('_orig_order', kind='mergesort').drop(
            columns=['_elo_row_order', '_orig_order']
        )
        return working_df.reset_index(drop=True)

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
