import pandas as pd
import numpy as np
import joblib
import logging
from app.utils.csv_loader import CSVLoader
from app.ml.feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class BettingSimulator:
    def __init__(self, model_path="models/hybrid_model.joblib"):
        self.model_path = model_path
        self.loader = CSVLoader()
        self.fe = FeatureEngineer()
        self.bankroll = 1000
        self.unit_size = 50
        self.threshold = 0.05

    def run_simulation(self):
        try:
            model = joblib.load(self.model_path)
        except FileNotFoundError:
            logger.error(f"Model not found at {self.model_path}. Run training first.")
            return

        logger.info("Loading Data for Simulation...")
        raw_df = self.loader.load_and_merge()
        if raw_df.empty: return

        df = self.fe.calculate_rolling_features(raw_df).dropna()
        
        split_idx = int(len(df) * 0.8)
        test_df = df.iloc[split_idx:].copy()
        
        X_test = test_df[self.fe.features]
        probs = model.predict_proba(X_test)
        
        current_bankroll = self.bankroll
        bets_placed = 0
        wins = 0

        logger.info(f"--- STARTING SIMULATION ---")
        logger.info(f"Bankroll: ${self.bankroll} | Unit: ${self.unit_size} | Edge > {self.threshold}")

        test_df.reset_index(drop=True, inplace=True)
        
        for i, row in test_df.iterrows():
            p_win = probs[i][2]

            if row['location'] == 'h':
                odds_win = row['B365H']
            else:
                odds_win = row['B365A']

            edge = (p_win * odds_win) - 1
            
            if edge > self.threshold:
                bets_placed += 1
                current_bankroll -= self.unit_size

                if row['target'] == 2:
                    current_bankroll += (self.unit_size * odds_win)
                    wins += 1

        roi = ((current_bankroll - self.bankroll) / self.bankroll) * 100
        win_rate = (wins / bets_placed) if bets_placed > 0 else 0
        
        logger.info(f"\n--- RESULTS ---")
        logger.info(f"Total Bets: {bets_placed} / {len(test_df)}")
        logger.info(f"Win Rate:   {win_rate:.2%}")
        logger.info(f"Final Bank: ${current_bankroll:.2f}")
        logger.info(f"Total ROI:  {roi:.2f}%")
        
        if roi > 0:
            logger.info("STRATEGY IS PROFITABLE")
        else:
            logger.info("STRATEGY IS LOSING MONEY")

if __name__ == "__main__":
    sim = BettingSimulator()
    sim.run_simulation()