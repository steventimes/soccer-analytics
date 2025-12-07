import pandas as pd
import logging

logger = logging.getLogger(__name__)

class CSVLoader:
    def __init__(self, data_path="./archive/"):
        self.path = data_path

    def load_and_merge(self):
        """Loads CSVs and merges them into a single analytical dataframe."""
        try:
            games = pd.read_csv(f"{self.path}games.csv")
            stats = pd.read_csv(f"{self.path}teamstats.csv")
            
            games['date'] = pd.to_datetime(games['date'])
            stats['date'] = pd.to_datetime(stats['date'])
            
            opp_stats = stats[['gameID', 'teamID', 'xGoals']].copy()
            opp_stats.rename(columns={'teamID': 'opponentID', 'xGoals': 'opp_xGoals'}, inplace=True)
            
            merged_stats = stats.merge(opp_stats, on='gameID')
            
            merged_stats = merged_stats[merged_stats['teamID'] != merged_stats['opponentID']]
            
            odds_cols = ['gameID', 'B365H', 'B365D', 'B365A']
            final_df = merged_stats.merge(games[odds_cols], on='gameID', how='left')
            
            final_df = final_df.sort_values(['teamID', 'date'])
            
            logger.info(f"Loaded {len(final_df)} rows with opponentID.")
            return final_df
            
        except FileNotFoundError as e:
            logger.error(f"Missing CSV file: {e}")
            return pd.DataFrame()